"""/v1/models keeps exact config names, lower-case aliases, and endpoint metadata."""

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
    def setUp(self):
        self.cache_patch = patch.object(trae_client, "_WEB_MODEL_CACHE", {})
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)

    def _patches(self, fetch, solo=None):
        async def bound_fetch(token_override, **kwargs):
            return await fetch(token_override=token_override, **kwargs)

        return (
            patch.object(trae_client, "_MODEL_LIST_CACHE", {}),
            patch.object(trae_client, "_UPSTREAM_MODEL_CASE", {}),
            patch.object(trae_client, "_fetch_web_model_configs", side_effect=bound_fetch),
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

    async def test_live_list_keeps_exact_and_lowercase_aliases(self):
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
                [
                    "Brand-New-Model",
                    "brand-new-model",
                    "DeepSeek-V4-Pro-Official",
                    "deepseek-v4-pro-official",
                    "Doubao-Seed-Code",
                    "doubao-seed-code",
                    "glm-5.3",
                    "kimi-k2.7-code",
                    "auto",
                ],
            )
            self.assertEqual(len(ids), len(set(ids)))
            self.assertNotIn("work", ids)
            exact = next(item for item in items if item["id"] == "Brand-New-Model")
            alias = next(item for item in items if item["id"] == "brand-new-model")
            self.assertEqual(exact["endpoint"], "remote")
            self.assertEqual(alias["alias_for"], "Brand-New-Model")
            self.assertEqual(alias["endpoints"], ["remote"])
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

    async def test_empty_upstream_falls_back_to_builtin_without_caching(self):
        async def fetch(**kwargs):
            return {}

        p1, p2, p3, p4, p5 = self._patches(fetch)
        with p1, p2, p3, p4, p5:
            items = await trae_client.get_models()
            ids = [item["id"] for item in items]
            self.assertIn("auto", ids)
            self.assertIn("glm-5.3", ids)
            self.assertIn("DeepSeek-V4-Pro", ids)
            self.assertNotIn("work", ids)
            self.assertTrue(
                all("capabilities_by_scope" not in item for item in items)
            )
            self.assertEqual(trae_client._MODEL_LIST_CACHE, {})

    async def test_capabilities_remain_separate_by_scope_and_copy_to_alias(self):
        async def fetch(**kwargs):
            return {
                "GLM-5.3": {
                    "name": "GLM-5.3",
                    "_trae_endpoints": ["remote", "work"],
                    "_trae_capabilities_by_scope": {
                        "solo_agent_remote": {
                            "multimodal": False,
                            "reasoning_effort": ["light", "high", "extra_high"],
                            "max_mode": True,
                            "max_context": 1_000_000,
                        },
                        "solo_work_remote": {
                            "multimodal": False,
                            "max_mode": False,
                        },
                    },
                }
            }

        p1, p2, p3, p4, p5 = self._patches(fetch)
        with p1, p2, p3, p4, p5:
            items = await trae_client.get_models(force=True)

        exact = next(item for item in items if item["id"] == "GLM-5.3")
        alias = next(item for item in items if item["id"] == "glm-5.3")
        scopes = exact["capabilities_by_scope"]
        self.assertTrue(scopes["solo_agent_remote"]["max_mode"])
        self.assertEqual(scopes["solo_agent_remote"]["max_context"], 1_000_000)
        self.assertFalse(scopes["solo_work_remote"]["max_mode"])
        self.assertNotIn("max_context", scopes["solo_work_remote"])
        self.assertNotIn("reasoning_effort", scopes["solo_work_remote"])
        self.assertEqual(alias["capabilities_by_scope"], scopes)

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

    async def test_shared_model_merges_remote_work_and_traecode_groups(self):
        async def fetch(**kwargs):
            return {
                "GLM-5.3": {
                    "name": "GLM-5.3",
                    "_trae_endpoints": ["remote", "work"],
                }
            }

        solo = {
            "GLM-5.3": {
                "name": "GLM-5.3",
                "config_name": "GLM-5.3",
                "_trae_endpoints": ["solo", "ide"],
            }
        }
        p1, p2, p3, p4, p5 = self._patches(fetch, solo)
        with p1, p2, p3, p4, p5:
            items = await trae_client.get_models(force=True)

        exact = next(item for item in items if item["id"] == "GLM-5.3")
        alias = next(item for item in items if item["id"] == "glm-5.3")
        self.assertEqual(exact["endpoints"], ["remote", "work", "solo", "ide"])
        self.assertEqual(exact["endpoint_groups"], ["remote", "traecode", "traework"])
        self.assertEqual(exact["endpoint_group"], "remote")
        self.assertEqual(alias["endpoints"], exact["endpoints"])
        self.assertEqual(alias["endpoint_groups"], exact["endpoint_groups"])
        self.assertEqual(alias["alias_for"], "GLM-5.3")

    async def test_endpoint_order_and_internal_control_models_are_filtered(self):
        async def fetch(**kwargs):
            return {
                "work-model": {
                    "name": "work-model",
                    "_trae_endpoints": ["work"],
                },
                "Remote-Model": {
                    "name": "Remote-Model",
                    "_trae_endpoints": ["remote"],
                },
                "Solo-Model": {
                    "name": "Solo-Model",
                    "_trae_endpoints": ["solo", "ide"],
                },
                "computer_use_subagent": {
                    "name": "computer_use_subagent",
                    "_trae_endpoints": ["remote"],
                },
                "AQUILA": {"name": "AQUILA", "_trae_endpoints": ["work"]},
            }

        p1, p2, p3, p4, p5 = self._patches(fetch)
        with p1, p2, p3, p4, p5:
            items = await trae_client.get_models(force=True)

        ids = [item["id"] for item in items]
        self.assertEqual(
            ids,
            [
                "Remote-Model",
                "remote-model",
                "work-model",
                "Solo-Model",
                "solo-model",
                "auto",
            ],
        )
        self.assertNotIn("computer_use_subagent", ids)
        self.assertNotIn("AQUILA", ids)
        solo = next(item for item in items if item["id"] == "Solo-Model")
        self.assertEqual(solo["endpoint"], "solo")
        self.assertEqual(solo["endpoints"], ["solo", "ide"])
        self.assertEqual(solo["endpoint_group"], "traecode")
        self.assertEqual(solo["endpoint_groups"], ["traecode"])

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

    async def test_unknown_lowercase_request_is_resolved_to_exact_config(self):
        with (
            patch.object(trae_client, "_UPSTREAM_MODEL_CASE", {}),
            patch.object(
                trae_client,
                "resolve_model_config",
                new=AsyncMock(
                    return_value={
                        "config_name": "Brand-New-Model",
                        "name": "Brand-New-Model",
                    }
                ),
            ) as resolve,
        ):
            resolved = await trae_client._resolve_request_model_name(
                "brand-new-model",
                token="tok-1",
                user_id="acc-1",
            )
            learned = trae_client._UPSTREAM_MODEL_CASE["brand-new-model"]

        self.assertEqual(resolved, "Brand-New-Model")
        self.assertEqual(learned, "Brand-New-Model")
        resolve.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
