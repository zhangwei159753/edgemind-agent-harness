import json
from pathlib import Path

import pytest

from edgemind.harness.skills import SkillRegistry


def test_registry_loads_weather_and_web_research_workflows():
    registry = SkillRegistry()

    skills = {skill.id: skill for skill in registry.list_skills()}

    assert set(skills) == {"weather_briefing", "web_research"}
    assert skills["weather_briefing"].version == "1.0"
    assert skills["weather_briefing"].allowed_tools == ("edgemind__get_weather",)
    assert set(skills["web_research"].allowed_tools) == {
        "edgemind__web_search", "edgemind__fetch_search_result"
    }
    assert skills["web_research"].input_schema["type"] == "object"


def test_registry_rejects_invalid_json_and_schema(tmp_path):
    (tmp_path / "broken.json").write_text('{"id": "broken",', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid skill JSON"):
        SkillRegistry(tmp_path)

    (tmp_path / "broken.json").write_text(json.dumps({
        "id": "bad", "version": "1", "title": "Bad", "description": "Bad",
        "allowed_tools": ["os__run"], "max_tool_rounds": 3,
        "instructions": "Run code", "input_schema": {"type": "object"},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown tool"):
        SkillRegistry(tmp_path)


def test_skill_definition_is_immutable():
    skill = SkillRegistry().get_skill("weather_briefing")

    with pytest.raises((AttributeError, TypeError)):
        skill.id = "changed"
    with pytest.raises((AttributeError, TypeError)):
        skill.allowed_tools += ("os__run",)
    with pytest.raises((AttributeError, TypeError)):
        skill.input_schema["properties"]["location"]["type"] = "integer"


def test_unknown_skill_returns_none():
    assert SkillRegistry().get_skill("not-a-skill") is None
