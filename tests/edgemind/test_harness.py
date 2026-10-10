from edgemind.harness.memory import MemoryService
from edgemind.harness.store import HarnessStore
from edgemind.harness.tasks import TaskService
from edgemind.harness.media import MediaProcessor


def test_task_checkpoint_survives_store_reopen_and_legacy_tables_remain(tmp_path):
    database = tmp_path / "state.sqlite3"
    import sqlite3
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE legacy_rag_marker (value TEXT)")
        connection.execute("INSERT INTO legacy_rag_marker VALUES ('preserved')")

    store = HarnessStore(database)
    task = TaskService(store).create("Long task", "Finish the implementation")
    TaskService(store).checkpoint(task.task_id, {"last_progress": "phase one", "next_step": "phase two"})

    reopened = HarnessStore(database)
    restored = reopened.get_task(task.task_id)
    with sqlite3.connect(database) as connection:
        legacy_value = connection.execute("SELECT value FROM legacy_rag_marker").fetchone()[0]

    assert restored.state["last_progress"] == "phase one"
    assert restored.state["next_step"] == "phase two"
    assert legacy_value == "preserved"
    assert reopened.schema_version == 4


def test_pending_memory_is_hidden_until_review_approval(tmp_path):
    store = HarnessStore(tmp_path / "state.sqlite3")
    task = store.create_task("Agent project", "Build a personal agent")
    memory = MemoryService(store)
    candidate = memory.propose("semantic", "Use SQLite for durable memory", source_task_id=task.task_id)

    assert memory.retrieve("SQLite durable memory", task_id=task.task_id) == []
    approved = memory.review(candidate.memory_id, "approved")
    hits = memory.retrieve("SQLite durable memory", task_id=task.task_id)

    assert approved.status == "approved"
    assert any(hit.item.memory_id == candidate.memory_id for hit in hits)


def test_image_is_normalized_to_openai_compatible_image_url(tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\npayload")

    block, metadata = MediaProcessor().process_image(image_path)

    assert block["type"] == "image_url"
    assert block["image_url"]["url"].startswith("data:image/png;base64,")
    assert metadata["name"] == "sample.png"
    assert metadata["size"] == image_path.stat().st_size


def test_image_rejects_non_image_bytes(tmp_path):
    image_path = tmp_path / "not-an-image.png"
    image_path.write_bytes(b"not an image")

    try:
        MediaProcessor().process_image(image_path)
    except ValueError as error:
        assert "JPEG" in str(error)
    else:
        raise AssertionError("non-image bytes must not be accepted as an image")



def test_agent_workspace_collapses_secondary_controls_by_default():
    from edgemind.web import build_demo

    demo = build_demo()
    accordions = {
        component.get("props", {}).get("label"): component.get("props", {}).get("open")
        for component in demo.config["components"]
        if component.get("type") == "accordion"
    }

    assert accordions.get("任务与进度") is False
    assert accordions.get("添加图片与语音") is False
    assert accordions.get("纠错与运行详情") is False

def test_harness_package_keeps_public_service_exports():
    from edgemind.harness import AgentService, HarnessStore, MemoryService, TaskService

    assert AgentService.__name__ == "AgentService"
    assert HarnessStore.__name__ == "HarnessStore"
    assert MemoryService.__name__ == "MemoryService"
    assert TaskService.__name__ == "TaskService"




def test_memory_retrieval_applies_policy_layer_weights_and_tag_filters(tmp_path):
    from edgemind.harness.models import MemoryItem

    store = HarnessStore(tmp_path / "state.sqlite3")
    task = store.create_task("Memory policy", "Validate retrieval filters")
    store.put_memory(MemoryItem("m-work", "semantic", "SQLite policy memory", "approved",
                                task.task_id, None, 0.8, {"tags": ["work"]}))
    store.put_memory(MemoryItem("m-personal", "episodic", "SQLite policy memory", "approved",
                                task.task_id, None, 0.8, {"tags": ["personal"]}))
    memory = MemoryService(store)

    hits = memory.retrieve(
        "SQLite policy memory", task_id=task.task_id, limit=8,
        layer_weights={"working": 0.5, "episodic": 0.5, "semantic": 1.8, "procedural": 0.5},
        time_decay_days=90, allowed_tags=("work",), excluded_tags=("personal",),
    )

    assert [hit.item.memory_id for hit in hits] == ["m-work"]
    assert hits[0].explanation["layer_weight"] == 1.8
