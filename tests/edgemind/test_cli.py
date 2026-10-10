import json

import pytest

from edgemind.cli import main


def test_help_lists_workflows_without_loading_model_or_api_key(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for command in ("query", "feedback", "eval", "experiment", "promote", "rollback", "manifest"):
        assert command in output


def test_feedback_export_and_eval_parser_accept_state_override(tmp_path, capsys):
    state = tmp_path / "custom.sqlite3"
    exported = tmp_path / "feedback.jsonl"
    assert main(["--state", str(state), "feedback", "export", str(exported)]) == 0
    assert exported.read_text(encoding="utf-8") == ""
    assert main(["--state", str(state), "eval", "--dataset", str(tmp_path / "missing.jsonl")]) == 2
    assert "dataset" in capsys.readouterr().err.lower()


def test_promote_command_requires_passing_experiment(tmp_path, capsys):
    code = main(["--state", str(tmp_path / "state.sqlite3"), "promote", "missing-experiment"])
    assert code == 2
    assert "not promotable" in capsys.readouterr().err
import json

from edgemind.cli import main


def test_manifest_command_is_idempotent_for_same_corpus(tmp_path):
    chunks = tmp_path / "chunks.json"
    chunks.write_text(json.dumps([{"content": "x", "metadata": {"file_name": "a.pdf", "page": 1}}]), encoding="utf-8")
    state = tmp_path / "state.sqlite3"
    assert main(["--state", str(state), "manifest", "--chunks", str(chunks)]) == 0
    assert main(["--state", str(state), "manifest", "--chunks", str(chunks)]) == 0


def test_experiment_cli_separates_metric_k_from_candidate_rerank_k(tmp_path):
    from edgemind.cli import _parser

    args = _parser().parse_args(["experiment", "--dataset", str(tmp_path / "set.jsonl"),
                                 "--top-k", "5", "--candidate-rerank-top-k", "2"])
    assert args.top_k == 5
    assert args.candidate_rerank_top_k == 2
