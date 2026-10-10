import pytest

from edgemind.providers import DeepSeekSettings, create_deepseek_client


def test_deepseek_settings_require_key_and_apply_endpoint_model_defaults():
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        DeepSeekSettings.from_env({})
    settings = DeepSeekSettings.from_env({"DEEPSEEK_API_KEY": "secret"})
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.model == "deepseek-flash"


def test_deepseek_settings_accept_environment_overrides():
    settings = DeepSeekSettings.from_env({
        "DEEPSEEK_API_KEY": "secret",
        "DEEPSEEK_BASE_URL": "https://proxy.example/v1",
        "DEEPSEEK_MODEL": "custom-chat",
    })
    assert settings.base_url == "https://proxy.example/v1"
    assert settings.model == "custom-chat"


def test_deepseek_client_factory_uses_injected_client_without_network():
    calls = {}

    class FakeClient:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    settings = DeepSeekSettings("secret", "https://api.deepseek.com", "deepseek-flash")
    result = create_deepseek_client(settings, client_class=FakeClient)
    assert isinstance(result, FakeClient)
    assert calls == {"api_key": "secret", "base_url": "https://api.deepseek.com"}
