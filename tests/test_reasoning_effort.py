import unittest

from src.reasoning_effort import apply_reasoning_effort, requested_level


class ReasoningEffortTests(unittest.TestCase):
    MODEL = {
        "name": "GLM-5.3",
        "reasoning_effort_config": {
            "support_thinking": True,
            "options": ["light", "high"],
            "default_level": "high",
        },
    }

    def test_openai_levels_map_and_clamp(self):
        cases = {"low": "light", "medium": "high", "high": "high", "xhigh": "high"}
        for effort, expected in cases.items():
            updated, level = apply_reasoning_effort(self.MODEL, {"reasoning_effort": effort})
            self.assertEqual(level, expected, effort)
            self.assertEqual(updated["reasoning_effort"], expected)
            self.assertNotIn("reasoning_effort_level", updated)

    def test_modern_key_is_preserved(self):
        model = dict(self.MODEL, reasoning_effort_level="high")
        updated, level = apply_reasoning_effort(model, {"reasoning_effort": "low"})
        self.assertEqual(level, "light")
        self.assertEqual(updated["reasoning_effort_level"], "light")
        self.assertNotIn("reasoning_effort", updated)

    def test_responses_and_anthropic_shapes(self):
        self.assertEqual(requested_level({"reasoning": {"effort": "minimal"}}), "light")
        self.assertEqual(
            requested_level({"thinking": {"type": "enabled", "budget_tokens": 32000}}),
            "extra_high",
        )
        self.assertEqual(requested_level({"thinking": {"budget_tokens": 2048}}), "light")

    def test_unsupported_model_or_no_request_is_untouched(self):
        plain = {"name": "Kimi"}
        updated, level = apply_reasoning_effort(plain, {"reasoning_effort": "high"})
        self.assertEqual(level, "")
        self.assertEqual(updated, plain)
        updated, level = apply_reasoning_effort(self.MODEL, {})
        self.assertEqual(level, "")
        self.assertNotIn("reasoning_effort", updated)


if __name__ == "__main__":
    unittest.main()