from .models import TASK_STATUSES, TaskRecord
from .store import HarnessStore


class TaskService:
    def __init__(self, store: HarnessStore):
        self.store = store

    def create(self, title: str, objective: str) -> TaskRecord:
        task = self.store.create_task(title, objective)
        self.store.append_event(task.task_id, "task_created", {"title": task.title, "objective": task.objective})
        return task

    def checkpoint(self, task_id: str, updates: dict) -> TaskRecord:
        task = self._require(task_id)
        state = {**task.state, **updates}
        result = self.store.update_task_state(task_id, state=state)
        self.store.append_event(task_id, "checkpoint_saved", {"keys": sorted(updates)})
        return result

    def pause(self, task_id: str) -> TaskRecord:
        return self._transition(task_id, "paused", {"paused": True})

    def resume(self, task_id: str) -> TaskRecord:
        return self._transition(task_id, "active", {"paused": False})

    def complete(self, task_id: str) -> TaskRecord:
        return self._transition(task_id, "completed", {"completed": True})

    def fail(self, task_id: str, error_type: str) -> TaskRecord:
        return self._transition(task_id, "failed", {"last_error_type": error_type})

    def context(self, task_id: str) -> dict:
        task = self._require(task_id)
        return {"task_id": task.task_id, "title": task.title, "objective": task.objective,
                "status": task.status, **task.state}

    def _transition(self, task_id: str, status: str, updates: dict) -> TaskRecord:
        task = self._require(task_id)
        allowed = {"active": {"paused", "completed", "failed"}, "paused": {"active", "completed", "failed"},
                   "failed": {"active", "completed"}, "completed": set()}
        if status not in TASK_STATUSES or status not in allowed[task.status]:
            raise ValueError(f"invalid task transition: {task.status} -> {status}")
        result = self.store.update_task_state(task_id, status, {**task.state, **updates})
        self.store.append_event(task_id, f"task_{status}", {})
        return result

    def _require(self, task_id: str) -> TaskRecord:
        task = self.store.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task: {task_id}")
        return task
