"""Price-key resolution, including through a gateway's namespaced model names."""

from decimal import Decimal

from seer_annotator.pricing import (
    PriceBasis, cost_from_usage, price_key_candidates, resolve,
)


class TestCandidates:
    def test_plain_model_has_no_namespace_step(self):
        assert price_key_candidates("anthropic", "claude-sonnet-5") == [
            ("anthropic/claude-sonnet-5", False),
            ("claude-sonnet-5", False),
        ]

    def test_gateway_name_adds_inferred_steps(self):
        assert price_key_candidates("openai", "google.claude-haiku-4.5") == [
            ("openai/google.claude-haiku-4.5", False),
            ("google.claude-haiku-4.5", False),
            ("claude-haiku-4.5", True),
            ("claude-haiku-4-5", True),
        ]

    def test_version_number_is_not_a_namespace(self):
        # "gpt-4.1" must not be read as namespace "gpt-4" holding model "1".
        assert price_key_candidates("openai", "gpt-4.1") == [
            ("openai/gpt-4.1", False),
            ("gpt-4.1", False),
        ]

    def test_dot_dash_step_only_when_the_tail_has_a_dot(self):
        keys = [k for k, _ in price_key_candidates("openai", "google.gemini-3-flash")]
        assert keys == ["openai/google.gemini-3-flash", "google.gemini-3-flash",
                        "gemini-3-flash"]


class TestResolve:
    def test_direct_name_is_not_inferred(self):
        basis = resolve("anthropic", "claude-sonnet-5")
        assert basis == PriceBasis(key="anthropic/claude-sonnet-5", inferred=False)

    def test_gateway_name_resolves_to_the_underlying_model(self):
        basis = resolve("openai", "google.gemini-3.5-flash")
        assert basis is not None
        assert basis.inferred is True
        assert basis.key == "gemini-3.5-flash"

    def test_gateway_anthropic_name_crosses_the_dot_dash_spelling(self):
        # litellm spells Gemini with dots and Anthropic with dashes; step 4 is
        # what stops that inconsistency losing every Claude model on a gateway.
        basis = resolve("openai", "google.claude-haiku-4.5")
        assert basis is not None
        assert basis.key == "claude-haiku-4-5"
        assert basis.inferred is True

    def test_unknown_model_stays_unpriced(self):
        # A local model no table knows records no cost at all, rather than $0.
        assert resolve("hosted_vllm", "Qwen3.8-27B-UD-Q8_K_L") is None

    def test_self_hosted_model_with_an_unknown_name_stays_unpriced(self):
        assert resolve("openai", "kit.glm-5.3") is None


class TestCostFromUsage:
    """The fallback that prices a gateway model litellm cannot price itself."""

    def test_prices_a_gateway_model_and_reports_the_guess(self):
        cost, basis = cost_from_usage(
            "openai", "google.gemini-3.5-flash-lite",
            input_tokens=1000, output_tokens=100,
        )
        assert basis is not None and basis.inferred is True
        rates = _rates("gemini-3.5-flash-lite")
        assert cost == Decimal(str(rates[0])) * 1000 + Decimal(str(rates[1])) * 100

    def test_cached_tokens_are_billed_at_the_cache_rate_and_not_twice(self):
        # The OpenAI dialect reports cached tokens INSIDE the prompt count, and a
        # gateway speaks it -- so 1000 in / 400 cached is 600 at the full rate.
        info = _info("gemini-3.5-flash-lite")
        cached_rate = info.get("cache_read_input_token_cost") or info["input_cost_per_token"]
        cost, _ = cost_from_usage(
            "openai", "google.gemini-3.5-flash-lite",
            input_tokens=1000, output_tokens=0, cached_tokens=400,
        )
        expected = (Decimal(str(info["input_cost_per_token"])) * 600
                    + Decimal(str(cached_rate)) * 400)
        assert cost == expected

    def test_unpriced_model_yields_no_cost_and_no_basis(self):
        assert cost_from_usage(
            "hosted_vllm", "Qwen3.8-27B-UD-Q8_K_L", input_tokens=10, output_tokens=10,
        ) == (None, None)


def _info(key):
    import litellm
    return litellm.get_model_info(model=key)


def _rates(key):
    info = _info(key)
    return info["input_cost_per_token"], info["output_cost_per_token"]
