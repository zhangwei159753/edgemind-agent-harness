import os
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class DeepSeekSettings:
    api_key: str = field(repr=False)
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DeepSeekSettings":
        values = os.environ if env is None else env
        api_key = values.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for text generation")
        return cls(
            api_key=api_key,
            base_url=values.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/"),
            model=values.get("DEEPSEEK_MODEL", "deepseek-flash").strip(),
        )


def create_deepseek_client(settings: DeepSeekSettings, client_class=None):
    if client_class is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install project dependencies to use DeepSeek generation (openai SDK)") from exc
        client_class = OpenAI
    return client_class(api_key=settings.api_key, base_url=settings.base_url)
