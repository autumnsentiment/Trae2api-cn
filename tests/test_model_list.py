"""/v1/models comes from the live account list and lists lowercase ids only."""

from __future__ import annotations

import time
import unittest
from unittest.mock import AsyncMock, patch

from src import trae_client


UPSTREAM = {
    "Doubao-Seed-Code": {"name": "Doubao-Seed-Code", "display_name": "Seed-Code"},
    "glm-5.3": {"name": "glm-5.3", "display_name": "GLM-5.3"},
    "DeepSeek-V4-Pro-Official": {
        "name": "DeepSeek-V4-Pro-Official",
        "display_name": "DeepSeek-V4-Pro \u6b63\u5f0f\u7248",
    },
    "kimi-k2.7-code": {"name": "kimi-k2.7-code", "display_name": "Kimi-K2.7-Code"},
    "Brand-New-Model": {"name": "Brand-New-Model", "display_name": "Brand New"},
}


class ModelListTests(unittest.IsolatedAsyncioTestCase):
    def _patches(self, fetch, solo=None):
        return (
            patch.object(trae_client, "_MODEL_LIST_CACHE", {}),
            patch.object(trae_client, "_UPSTREAM_MODEL_CASE", {}),
            patch.object(trae_client, "_fetch_web_model_configs", side_effect=fetch),
            patch.object(
                trae_client,
                "_fetch_solo_model_configs",
                new=AsyncMock(return_value=solo or {}),
            ),
            patch.object(
                trae_client.auth,
                "get_active_account_snapshot",
                return_value=("acc-1", {"token": "tok-1"}),
            ),
        )

    async def test_live_list_is_lowercase_without_display_duplicates(self):
        calls = []

        async def fetch(**kwargs):
            calls.append(kwargs)
            return dict(UPSTREAM)

        p1, p2, p3, p4, p5 = self._patches(fetch)
        with p1, p2, p3, p4, p5:
            items = await trae_client.get_models()
            ids = [item["id"] for item in items]
            self.assertEqual(
                ids,
                sorted(
                    [
                        "auto",
                        "brand-new-model",
                        "deepseek-v4-pro-official",
                        "doubao-seed-code",
                        "glm-5.3",
                        "kimi-k2.7-code",
                        "work",
                    ]
                ),
            )
            self.assertTrue(all(i == i.lower() for i in ids))
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(calls[0]["token_override"], "tok-1")
            # Lowercase ids map back to the exact upstream config name.
            self.assertEqual(
                trae_client.convert_model_name("brand-new-model"), "Brand-New-Model"
            )
            self.assertEqual(
                trae_client.convert_model_name("doubao-seed-code"), "Doubao-Seed-Code"
            )
            self.assertEqual(
                trae_client.convert_model_name("deepseek-v4-pro-official"),
                "DeepSeek-V4-Pro-Official",
            )

            # Cached until forced.
            await trae_client.get_models()
            self.assertEqual(len(calls), 1)
            await trae_client.get_models(force=True)
            self.assertEqual(len(calls), 2)

    async def test_empty_upstream_falls_back_to_lowercase_builtin_without_caching(self):
        async def fetch(**kwargs):
            return {}

        p1, p2, p3, p4, p5 = self._patches(fetch)
        with p1, p2, p3, p4, p5:
            items = await trae_client.get_models()
            ids = [item["id"] for item in items]
            self.assertIn("auto", ids)
            self.assertIn("glm-5.3", ids)
            self.assertTrue(all(i == i.lower() for i in ids))
            self.assertNotIn("DeepSeek-V4-Pro", ids)
            self.assertEqual(trae_client._MODEL_LIST_CACHE, {})

    async def test_native_solo_only_model_is_merged_into_public_list(self):
        async def fetch(**kwargs):
            return {"glm-5.3": {"name": "glm-5.3"}}

        solo = {
            "glm-5.3-flash": {
                "name": "glm-5.3-flash",
                "config_name": "glm-5.3-flash",
                "display_name": "GLM-5.3-Flash",
            }
        }
        p1, p2, p3, p4, p5 = self._patches(fetch, solo)
        with p1, p2, p3, p4, p5:
            items = await trae_client.get_models(force=True)
        ids = [item["id"] for item in items]
        self.assertIn("glm-5.3-flash", ids)
        self.assertIn("glm-5.3", ids)

    def test_builtin_aliases_still_resolve_exact_config_names(self):
        self.assertEqual(
            trae_client.convert_model_name("deepseek-v4-flash-official"),
            "DeepSeek-V4-Flash-Official",
        )
        self.assertEqual(
            trae_client.convert_model_name("doubao-seed-evolving"), "Doubao-Seed-Evolving"
        )
        self.assertEqual(trae_client.convert_model_name("gpt-4o"), "DeepSeek-V4-Pro")
        self.assertEqual(trae_client.convert_model_name("unknown-x"), "unknown-x")


if __name__ == "__main__":
    unittest.main()
