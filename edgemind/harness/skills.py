import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

_ALLOWED_TOOLS = {"edgemind__get_weather", "edgemind__web_search", "edgemind__fetch_search_result"}
_REQUIRED = {"id", "version", "title", "description", "allowed_tools", "max_tool_rounds", "instructions", "input_schema"}


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class SkillDefinition:
    id: str
    version: str
    title: str
    description: str
    allowed_tools: tuple[str, ...]
    max_tool_rounds: int
    instructions: str
    input_schema: Mapping[str, object]

class SkillRegistry:
    """Loads allowlisted declarative JSON workflows; never imports executable skill code."""
    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory) if directory else Path(__file__).with_name("skills")
        self._skills = {}
        for path in sorted(self.directory.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid skill JSON: {path.name}") from exc
            skill = self._parse(raw, path.name)
            if skill.id in self._skills:
                raise ValueError(f"duplicate skill id: {skill.id}")
            self._skills[skill.id] = skill

    @staticmethod
    def _parse(raw: object, filename: str) -> SkillDefinition:
        if not isinstance(raw, dict) or set(raw) != _REQUIRED:
            raise ValueError(f"invalid skill schema: {filename}")
        if not all(isinstance(raw[key], str) and raw[key].strip()
                   for key in ("id", "version", "title", "description", "instructions")):
            raise ValueError(f"invalid skill schema: {filename}")
        tools = raw["allowed_tools"]
        if not isinstance(tools, list) or not tools or any(
            not isinstance(item, str) or item not in _ALLOWED_TOOLS for item in tools
        ):
            raise ValueError(f"unknown tool in skill: {filename}")
        rounds = raw["max_tool_rounds"]
        if not isinstance(rounds, int) or isinstance(rounds, bool) or not 1 <= rounds <= 3:
            raise ValueError(f"invalid skill schema: {filename}")
        schema = raw["input_schema"]
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError(f"invalid skill schema: {filename}")
        skill_id = raw["id"]
        if skill_id not in {"weather_briefing", "web_research"}:
            raise ValueError(f"unknown skill id: {filename}")
        expected = ({"edgemind__get_weather"} if skill_id == "weather_briefing" else
                    {"edgemind__web_search", "edgemind__fetch_search_result"})
        if set(tools) != expected:
            raise ValueError(f"invalid tool allowlist: {filename}")
        return SkillDefinition(skill_id, raw["version"], raw["title"], raw["description"],
                               tuple(tools), rounds, raw["instructions"],
                               _freeze(schema))

    def list_skills(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def get_skill(self, skill_id: str) -> SkillDefinition | None:
        return self._skills.get(skill_id)
