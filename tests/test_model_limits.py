import unittest
from unittest.mock import patch

from src.model_limits import (
    clamp_max_completion_tokens,
    descriptor_denies_pay_identity,
    descriptor_max_profile,
    descriptor_supports_max_mode,
)


class ModelDescriptorLimitTests(unittest.TestCase):
    def test_completion_limit_uses_explicit_descriptor_max_tokens(self):
        with patch.dict(
            "os.environ", {"TRAE_MAX_COMPLETION_TOKENS": "48000"}, clear=False
        ):
            self.assertEqual(
                clamp_max_completion_tokens(100000, "glm-5.3", {"max_tokens": 32000}),
                32000,
            )

    def test_completion_limit_does_not_infer_from_model_name(self):
        with patch.dict(
            "os.environ", {"TRAE_MAX_COMPLETION_TOKENS": "48000"}, clear=False
        ):
            self.assertEqual(clamp_max_completion_tokens(100000, "glm-5.3"), 48000)
            self.assertEqual(
                clamp_max_completion_tokens(100000, "glm-5.3", {"max_tokens": "unknown"}),
                48000,
            )

    def test_max_mode_requires_explicit_one_million_context_profile(self):
        common = {
            "max_mode": True,
            "prompt_max_tokens": 168000,
            "max_tokens": 32000,
        }
        self.assertFalse(
            descriptor_supports_max_mode(
                {**common, "context_window_size": {"max": 200000}}
            )
        )
        self.assertTrue(
            descriptor_supports_max_mode(
                {
                    **common,
                    "prompt_max_tokens": 936000,
                    "max_tokens": 64000,
                    "context_window_size": {"max": [200000]},
                    "context_window_tokens": {"max": [200000, 1000000]},
                }
            )
        )

    def test_max_mode_requires_all_explicit_wire_limits(self):
        descriptor = {
            "max_mode": True,
            "context_window_tokens": {"max": 1000000},
        }
        self.assertFalse(descriptor_supports_max_mode(descriptor))
        self.assertFalse(
            descriptor_supports_max_mode(
                {**descriptor, "prompt_max_tokens": 936000}
            )
        )

    def test_max_mode_can_derive_prompt_budget_from_live_detail(self):
        descriptor = {
            "context_window_tokens": {"dev": 200000, "max": 1048576},
            "model_detail_list": [
                {"model_name": "glm-5.2__dev", "max_tokens": 32000},
                {"model_name": "glm-5.2__max", "max_tokens": 64000},
            ],
        }
        profile = descriptor_max_profile(descriptor)
        self.assertTrue(descriptor_supports_max_mode(descriptor))
        self.assertEqual(profile["model_name"], "glm-5.2__max")
        self.assertEqual(profile["context"], 1048576)
        self.assertEqual(profile["output"], 64000)
        self.assertEqual(profile["prompt"], 984576)
        self.assertEqual(profile["base_context"], 200000)

    def test_access_identity_requires_explicit_integer_list(self):
        descriptor = {
            "features": {
                "access": {
                    "data": {
                        "identity_list": [5, 1, 2, 3, 100],
                    }
                }
            }
        }
        self.assertTrue(descriptor_denies_pay_identity(descriptor, 0))
        self.assertFalse(descriptor_denies_pay_identity(descriptor, 1))
        self.assertFalse(descriptor_denies_pay_identity(descriptor, None))
        self.assertFalse(descriptor_denies_pay_identity({}, 0))
        self.assertFalse(
            descriptor_denies_pay_identity(
                {"features": {"access": {"data": {"identity_list": []}}}},
                0,
            )
        )

    def test_access_identity_accepts_serialized_catalog_list(self):
        descriptor = {
            "features": {
                "access": {
                    "data": {
                        "identity_list": "[5,1,2,3,100]",
                    }
                }
            }
        }
        self.assertTrue(descriptor_denies_pay_identity(descriptor, 0))
        self.assertTrue(descriptor_denies_pay_identity(descriptor, "Free"))

    def test_invalid_access_identity_values_do_not_deny_an_account(self):
        for value in (True, None, 1.5, float("inf"), float("nan"), {}, "1.5"):
            with self.subTest(value=value):
                descriptor = {"identity_list": [5, value]}
                self.assertFalse(descriptor_denies_pay_identity(descriptor, 0))

    def test_access_identity_reads_flat_and_feature_nested_catalogs(self):
        cases = (
            {"identity_list": [5, 1, 2, 3, 100]},
            {"features": {"identity_list": [5, 1, 2, 3, 100]}},
            {
                "feature": {
                    "access": {
                        "data": {
                            "identityList": [5, 1, 2, 3, 100],
                        }
                    }
                }
            },
        )
        for descriptor in cases:
            with self.subTest(descriptor=descriptor):
                self.assertTrue(descriptor_denies_pay_identity(descriptor, 0))
                self.assertFalse(descriptor_denies_pay_identity(descriptor, 5))


if __name__ == "__main__":
    unittest.main()
