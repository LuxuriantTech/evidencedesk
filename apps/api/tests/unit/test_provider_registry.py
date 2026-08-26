import pytest
from evidencedesk_api.provider_registry import UnsupportedProviderMode, build_provider_bundle


def test_extractive_local_bundle_is_explicit_and_free() -> None:
    providers = build_provider_bundle("extractive-local")

    assert providers.embedding.mode == "deterministic-hash-v1"
    assert providers.answer.mode == "extractive-local"
    assert providers.estimated_cost_usd == 0.0


def test_unknown_provider_mode_fails_closed() -> None:
    with pytest.raises(UnsupportedProviderMode, match="unsupported answer mode"):
        build_provider_bundle("openai-compatible")
