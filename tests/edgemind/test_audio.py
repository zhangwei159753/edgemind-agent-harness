from types import SimpleNamespace
import sys

import pytest

from edgemind.harness.media import MediaProcessor


class FakeWhisperModel:
    calls = 0

    def __init__(self, name, device, compute_type):
        type(self).calls += 1

    def transcribe(self, path, vad_filter=True):
        return [SimpleNamespace(text=" hello")], SimpleNamespace(language="en", duration=2.0)


def test_audio_transcription_reuses_whisper_model_and_reports_latency(monkeypatch, tmp_path):
    import edgemind.harness.media as media_module

    media_module.clear_asr_model_cache()
    FakeWhisperModel.calls = 0
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"test audio payload")

    first_text, first = MediaProcessor("test-cache-model").process_audio(audio)
    second_text, second = MediaProcessor("test-cache-model").process_audio(audio)

    assert first_text == second_text == "hello"
    assert FakeWhisperModel.calls == 1
    assert first["model_cache_hit"] is False
    assert first["model_load_ms"] >= 0
    assert first["transcription_ms"] >= 0
    assert first["audio_duration_sec"] == 2.0
    assert first["realtime_factor"] >= 0
    assert second["model_cache_hit"] is True
    assert second["model_load_ms"] == 0
    media_module.clear_asr_model_cache()


def test_missing_asr_dependency_has_install_instructions(monkeypatch, tmp_path):
    audio = tmp_path / "missing.wav"
    audio.write_bytes(b"test audio payload")
    monkeypatch.setitem(sys.modules, "faster_whisper", None)

    with pytest.raises(RuntimeError, match=r"uv sync --extra audio"):
        MediaProcessor("test-missing-model").process_audio(audio)


def test_agent_exposes_actionable_audio_error_and_asr_stage_time(tmp_path):
    from edgemind.harness.agent import AgentService
    from edgemind.harness.store import HarnessStore
    from edgemind.harness.tasks import TaskService

    store = HarnessStore(tmp_path / "audio-agent.sqlite3")
    task = TaskService(store).create("Audio test", "Verify actionable errors")

    class MissingASR:
        def process_audio(self, path):
            raise RuntimeError("faster-whisper 语音组件未安装；请运行 uv sync --extra audio 后重启。")

    turn = AgentService(store, media=MissingASR()).run_turn(task.task_id, audio_path="upload.wav")

    assert turn.status == "failed"
    assert "uv sync --extra audio" in turn.answer
    assert turn.trace["stage_ms"]["asr"] >= 0


def test_agent_trace_contains_memory_and_deepseek_stage_times(tmp_path):
    from edgemind.harness.agent import AgentService
    from edgemind.harness.store import HarnessStore
    from edgemind.harness.tasks import TaskService

    store = HarnessStore(tmp_path / "audio-agent-success.sqlite3")
    task = TaskService(store).create("Latency test", "Inspect stage timings")
    message = SimpleNamespace(content="ok", tool_calls=[])
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class Completions:
        def create(self, **kwargs):
            return response

    class Tools:
        def model_tools(self):
            return []

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    turn = AgentService(store, client=client, tools=Tools()).run_turn(task.task_id, "hello")

    assert turn.status == "succeeded"
    assert turn.trace["stage_ms"]["memory_retrieval"] >= 0
    assert turn.trace["stage_ms"]["deepseek"] >= 0
    assert turn.trace["elapsed_ms"] >= turn.trace["stage_ms"]["deepseek"]


def test_failed_turn_keeps_audio_attachment_for_retry(monkeypatch):
    import edgemind.web as web_module

    monkeypatch.setattr(web_module, "run_agent_turn", lambda *args: (
        "语音组件未安装。", {"status": "failed", "error_type": "RuntimeError"}, "run-1"
    ))
    images, audio = ["image.png"], "voice.wav"

    result = web_module.agent_chat_turn("task-1", "transcribe", images, audio, [])

    assert result[3] == "transcribe"
    assert result[4] == images
    assert result[5] == audio


def test_audio_backend_status_mentions_local_processing_and_setup(monkeypatch):
    import importlib.util
    import edgemind.web as web_module

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    status = web_module.audio_backend_status()

    assert "本地" in status
    assert "uv sync --extra audio" in status


def test_frontend_css_does_not_contain_literal_newline_escape():
    import edgemind.web as web_module

    assert r"\n#ed-compose-actions" not in web_module.CSS

def test_audio_extras_pin_pyav_before_incompatible_major():
    import tomllib
    from pathlib import Path

    project_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    config = tomllib.loads(project_file.read_text(encoding="utf-8"))
    extras = config["project"]["optional-dependencies"]

    assert any(requirement.startswith("av>=11,<19") for requirement in extras["audio"])
    assert any(requirement.startswith("av>=11,<19") for requirement in extras["agent"])
