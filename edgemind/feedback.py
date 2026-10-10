import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .models import Feedback
from .storage import FEEDBACK_LABELS


class FeedbackService:
    def __init__(self, store):
        self.store = store

    def submit(self, run_id: str, label: str, note: str = "", feedback_id: str | None = None) -> Feedback:
        if label not in FEEDBACK_LABELS:
            raise ValueError(f"unknown feedback label: {label}")
        feedback = Feedback(
            feedback_id=feedback_id or str(uuid.uuid4()),
            run_id=run_id,
            label=label,
            note=note,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.store.add_feedback(feedback)
        return feedback

    def export_jsonl(self, path: str | Path) -> int:
        records = self.store.list_feedback()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n" for item in records), encoding="utf-8")
        return len(records)

    def import_jsonl(self, path: str | Path) -> int:
        records: list[Feedback] = []
        for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
                item = Feedback(**value)
                if item.label not in FEEDBACK_LABELS:
                    raise ValueError(f"unknown feedback label: {item.label}")
                if self.store.get_run(item.run_id) is None:
                    raise ValueError(f"unknown query run: {item.run_id}")
                records.append(item)
            except Exception as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
        for item in records:
            self.store.add_feedback(item)
        return len(records)
