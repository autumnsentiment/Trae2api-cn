import unittest

from src.reasoning_effort import apply_reasoning_effort, normalize_options, requested_level


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
        self.assertEqual(requested_level({"reasoning_effort": "ultra"}), "extra_high")
        self.assertEqual(
            requested_level({"thinking": {"type": "enabled", "budget_tokens": 32000}}),
            "extra_high",
        )
        self.assertEqual(requested_level({"thinking": {"budget_tokens": 2048}}), "light")

    def test_empty_canonical_value_does_not_hide_alias(self):
        for empty in (None, "", " "):
            with self.subTest(canonical=empty):
                self.assertEqual(
                    normalize_options({"reasoning_effort": empty, "reasoningEffort": "low"})[
                        "reasoning_effort"
                    ],
                    "light",
                )

    def test_aliases_are_normalized_without_overwriting_explicit_canonical_value(self):
        self.assertEqual(
            normalize_options({"reasoningEffort": "low"})["reasoning_effort"],
            "light",
        )
        self.assertEqual(
            normalize_options({"reasoning": {"effort": "high"}})["reasoning_effort"],
            "high",
        )
        self.assertEqual(
            normalize_options({"reasoning_effort": "max", "reasoningEffort": "low"})[
                "reasoning_effort"
            ],
            "max",
        )

    def test_unsupported_model_or_no_request_is_untouched(self):
        plain = {"name": "Kimi"}
        updated, level = apply_reasoning_effort(plain, {"reasoning_effort": "high"})
        self.assertEqual(level, "")
        self.assertEqual(updated, plain)
        updated, level = apply_reasoning_effort(self.MODEL, {})
        self.assertEqual(level, "")
        self.assertNotIn("reasoning_effort", updated)

    def test_json_string_descriptor_is_supported_without_name_inference(self):
        model = {
            "name": "arbitrary-model",
            "reasoning_effort_config": (
                '{"support_thinking":true,"options":["light","extra_high"]}'
            ),
        }
        updated, level = apply_reasoning_effort(model, {"reasoning_effort": "max"})
        self.assertEqual(level, "extra_high")
        self.assertEqual(updated["reasoning_effort"], "extra_high")

        unsupported, level = apply_reasoning_effort(
            {
                "name": "official-reasoning-model",
                "reasoning_effort_config": (
                    '{"support_thinking":false,"options":["extra_high"]}'
                ),
            },
            {"reasoning_effort": "max"},
        )
        self.assertEqual(level, "")
        self.assertNotIn("reasoning_effort", unsupported)

    def test_camel_case_reasoning_descriptor_fields_are_normalized(self):
        model = {
            "reasoningEffortConfig": {
                "supportThinking": "true",
                "reasoningEffortOptions": '["light","high"]',
            },
            "reasoning_effort_level": "high",
        }
        updated, level = apply_reasoning_effort(model, {"reasoning_effort": "low"})
        self.assertEqual(level, "light")
        self.assertEqual(updated["reasoning_effort_level"], "light")


if __name__ == "__main__":
    unittest.main()
