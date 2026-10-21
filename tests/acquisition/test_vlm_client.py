"""OpenAI backend key handling: `api_key_env` names an environment variable, None -> local server."""
import pytest

from vlnav.acquisition.vlm_client import OpenAIVLMClient, VLMError, load_vlm_config


def client(**over):
    return OpenAIVLMClient(load_vlm_config(overrides={"backend": "openai", "cache_dir": None, **over}))


def test_local_server_needs_no_key():
    assert client().client.api_key == "EMPTY"


def test_key_read_from_named_env(monkeypatch):
    monkeypatch.setenv("VLNAV_TEST_KEY", "sk-test")
    assert client(api_key_env="VLNAV_TEST_KEY").client.api_key == "sk-test"


def test_missing_env_fails_at_construction(monkeypatch):
    monkeypatch.delenv("VLNAV_TEST_KEY", raising=False)
    with pytest.raises(VLMError, match="VLNAV_TEST_KEY"):
        client(api_key_env="VLNAV_TEST_KEY")
