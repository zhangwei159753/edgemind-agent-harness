import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .evaluation import EvaluationRunner
from .experiments import ExperimentService, QualityGates
from .feedback import FeedbackService
from .ingestion import build_corpus_manifest, load_chunks_json
from .models import RetrievalConfig
from .storage import SqliteStore

DEFAULT_STATE = Path(".data/edgemind.sqlite3")
DEFAULT_CHUNKS = Path("all_pdf_page_chunks.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgemind", description="EdgeMind retrieval evaluation and controlled improvement tools")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE, help="SQLite state database (default: .data/edgemind.sqlite3)")
    commands = parser.add_subparsers(dest="command", required=True)
    query = commands.add_parser("query", help="ask a question and persist its trace")
    query.add_argument("question")
    query.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    query.add_argument("--top-k", type=int, default=5)
    feedback = commands.add_parser("feedback", help="submit or exchange reviewed feedback")
    feedback_commands = feedback.add_subparsers(dest="feedback_action", required=True)
    submit = feedback_commands.add_parser("submit")
    submit.add_argument("run_id")
    submit.add_argument("label")
    submit.add_argument("--note", default="")
    export = feedback_commands.add_parser("export")
    export.add_argument("path", type=Path)
    importing = feedback_commands.add_parser("import")
    importing.add_argument("path", type=Path)
    memory = commands.add_parser("memory", help="manage reviewed long-term memories")
    memory_commands = memory.add_subparsers(dest="memory_action", required=True)
    memory_export = memory_commands.add_parser("export", help="export memories as JSONL")
    memory_export.add_argument("--output", type=Path, required=True)
    memory_export.add_argument("--approved-only", action="store_true")
    evaluation = commands.add_parser("eval", help="run an offline JSONL benchmark")
    evaluation.add_argument("--dataset", type=Path, required=True)
    evaluation.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    evaluation.add_argument("--top-k", type=int, default=5)
    evaluation.add_argument("--output", type=Path, default=Path(".data/evaluations/latest.json"))
    experiment = commands.add_parser("experiment", help="evaluate a retrieval candidate against a baseline")
    experiment.add_argument("--dataset", type=Path, required=True)
    experiment.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    experiment.add_argument("--top-k", type=int, default=5)
    experiment.add_argument("--candidate-rerank-top-k", type=int, default=3)
    experiment.add_argument("--min-recall-at-k", type=float, default=0.0)
    experiment.add_argument("--max-recall-regression", type=float, default=0.05)
    experiment.add_argument("--output", type=Path, default=Path(".data/evaluations/latest-experiment.json"))
    promote = commands.add_parser("promote", help="promote a candidate that passed its quality gates")
    promote.add_argument("experiment_id")
    rollback = commands.add_parser("rollback", help="activate a previously recorded config version")
    rollback.add_argument("version_id")
    manifest = commands.add_parser("manifest", help="build and persist a deterministic corpus manifest")
    manifest.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    manifest.add_argument("--parser", default="legacy-json")
    manifest.add_argument("--parser-version", default="unknown")
    manifest.add_argument("--chunk-size", type=int, default=800)
    manifest.add_argument("--chunk-overlap", type=int, default=100)
    return parser


def _services(store: SqliteStore, chunks_path: Path, config=None, config_version="baseline"):
    chunks = load_chunks_json(chunks_path)
    manifest = build_corpus_manifest(chunks, "legacy-json", "unknown", {})
    store.record_corpus_manifest(manifest["corpus_version"], manifest)
    from myrag import SimpleRAG
    rag = SimpleRAG(str(chunks_path))
    rag.setup()

    class ConfiguredRAG:
        def generate_answer(self, question, top_k=5):
            configured_top_k = config.rerank_top_k if config is not None else top_k
            return rag.generate_answer(question, top_k=configured_top_k)

    from .query import QueryService
    return QueryService(lambda _: ConfiguredRAG(), store, manifest["corpus_version"], config_version)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        store = SqliteStore(args.state)
        if args.command == "feedback":
            service = FeedbackService(store)
            if args.feedback_action == "submit":
                print(json.dumps(asdict(service.submit(args.run_id, args.label, args.note)), ensure_ascii=False))
            elif args.feedback_action == "export":
                print(f"exported={service.export_jsonl(args.path)}")
            else:
                print(f"imported={service.import_jsonl(args.path)}")
            return 0
        if args.command == "memory":
            from .harness.store import HarnessStore

            count = HarnessStore(args.state).export_memories(args.output, approved_only=args.approved_only)
            print(json.dumps({"exported": count, "output": str(args.output)}, ensure_ascii=False))
            return 0
        if args.command == "manifest":
            chunks = load_chunks_json(args.chunks)
            value = build_corpus_manifest(chunks, args.parser, args.parser_version,
                                          {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap})
            store.record_corpus_manifest(value["corpus_version"], value)
            print(json.dumps(value, ensure_ascii=False, indent=2))
            return 0
        if args.command == "promote":
            print(ExperimentService(store, None, QualityGates({}, {})).promote(args.experiment_id))
            return 0
        if args.command == "rollback":
            print(ExperimentService(store, None, QualityGates({}, {})).rollback(args.version_id))
            return 0
        if args.command == "query":
            run = _services(store, args.chunks).ask(args.question, args.top_k)
            print(json.dumps(asdict(run), ensure_ascii=False, indent=2))
            return 0 if run.status == "succeeded" else 1
        if args.command == "eval":
            if not args.dataset.is_file():
                raise ValueError(f"dataset file does not exist: {args.dataset}")
            runner = EvaluationRunner(lambda config, version: _services(store, args.chunks, config, version))
            result = runner.run(args.dataset, top_k=args.top_k)
            _write_json(args.output, result)
            print(json.dumps({"metrics": result["metrics"], "sample_count": result["sample_count"], "output": str(args.output)}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "experiment":
            if not args.dataset.is_file():
                raise ValueError(f"dataset file does not exist: {args.dataset}")
            runner = EvaluationRunner(lambda config, version: _services(store, args.chunks, config, version))
            if store.get_active_version() is None:
                baseline_result = runner.run(args.dataset, config_version="baseline", top_k=args.top_k)
                baseline_id = "baseline-" + baseline_result["config_version"]
                store.record_version(baseline_id, {"config": asdict(RetrievalConfig()), "metrics": baseline_result["metrics"], "experiment_id": None})
                store.set_active_version(baseline_id)
            config = RetrievalConfig(rerank_top_k=args.candidate_rerank_top_k)
            service = ExperimentService(store, runner, QualityGates(
                {"recall_at_k": args.min_recall_at_k}, {"recall_at_k": args.max_recall_regression}))
            result = service.evaluate_candidate(config, args.dataset)
            _write_json(args.output, result)
            print(json.dumps({"experiment_id": result["experiment_id"], "status": result["status"],
                              "gate_reasons": result["gate_reasons"], "output": str(args.output)}, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "passed" else 1
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 2
