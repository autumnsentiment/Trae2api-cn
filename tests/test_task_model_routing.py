import os
import unittest
from unittest.mock import AsyncMock, patch

from src import main


class TaskModelRoutingTests(unittest.TestCase):
    def _route(self, messages, model="auto", options=None, **env):
        values = {
            "TRAE_TASK_MODEL_ROUTING": "1",
            "TRAE_SIMPLE_MODEL": "deepseek-v4.1-flash",
            "TRAE_COMPLEX_MODEL": "6.1sol",
        }
        values.update(env)
        request_options = dict(options or {})
        if (
            str(model or "").strip().lower() in {"", "auto"}
            and str(values.get("TRAE_TASK_MODEL_ROUTING", "1")).strip().lower()
            not in {
            "0",
            "false",
            "no",
            "off",
            "disabled",
            }
        ):
            request_options = {
                "_upstream_mode": "remote",
                "_account_id": "account-1",
                **request_options,
            }
        with patch.dict(os.environ, values, clear=False):
            return main._apply_task_model_routing(
                messages, model, request_options
            )

    def test_simple_auto_task_without_scoped_catalog_keeps_provider_auto(self):
        model, options = self._route(
            [{"role": "user", "content": "What is the current time?"}]
        )

        self.assertEqual(model, "auto")
        self.assertEqual(options["reasoning_effort"], "ultra")
        self.assertEqual(options["_task_model_class"], "simple")
        self.assertEqual(options["_upstream_trace"]["task_model_effective"], "auto")
        self.assertIn("endpoint=remote", options["_upstream_trace"]["task_model_fallback_reason"])

    def test_complex_auto_task_uses_catalogued_six_one_sol(self):
        model, options = self._route(
            [{"role": "user", "content": "Implement the file download flow and run tests"}],
            options={
                "_available_models": ["6.1sol"],
                "_available_models_endpoint": "remote",
                "_available_models_account_key": "account-1",
            },
        )

        self.assertEqual(model, "6.1sol")
        self.assertEqual(options["reasoning_effort"], "ultra")
        self.assertEqual(options["_task_model_class"], "complex")
        self.assertNotIn(
            "task_model_fallback", options["_upstream_trace"]
        )

    def test_missing_complex_model_keeps_auto_with_explicit_trace(self):
        model, options = self._route(
            [{"role": "user", "content": "Refactor the request pipeline and add tests"}]
        )

        self.assertEqual(model, "auto")
        self.assertNotEqual(model, "kimi-k2.7-code")
        self.assertEqual(options["reasoning_effort"], "ultra")
        trace = options["_upstream_trace"]
        self.assertEqual(trace["task_model_fallback"], "auto")
        self.assertIn("6.1sol", trace["task_model_fallback_reason"])

    def test_tools_are_complex(self):
        model, options = self._route(
            [{"role": "user", "content": "Call the download tool"}],
            options={
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "download",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "_available_models": ["6.1sol"],
                "_available_models_endpoint": "remote",
                "_available_models_account_key": "account-1",
            },
        )

        self.assertEqual(model, "6.1sol")
        self.assertEqual(options["_task_model_reason"], "tools")

    def test_catalog_from_other_endpoint_or_account_is_not_accepted(self):
        for catalog_endpoint, catalog_account in (
            ("ide", "account-1"),
            ("remote", "account-2"),
        ):
            with self.subTest(
                endpoint=catalog_endpoint, account=catalog_account
            ):
                model, options = self._route(
                    [{"role": "user", "content": "What is the current time?"}],
                    options={
                        "_available_models": ["deepseek-v4.1-flash"],
                        "_available_models_endpoint": catalog_endpoint,
                        "_available_models_account_key": catalog_account,
                    },
                )
                self.assertEqual(model, "auto")
                self.assertIn(
                    "task_model_fallback_reason",
                    options["_upstream_trace"],
                )

    def test_global_alias_and_merged_catalog_do_not_prove_endpoint_support(self):
        with (
            patch.object(
                main.trae_client,
                "_MODEL_LIST_CACHE",
                {"account-1": (main.time.time(), [{"id": "deepseek-v4.1-flash"}])},
            ),
            patch.object(
                main.trae_client,
                "_WEB_MODEL_CACHE",
                {
                    "account-2:solo_agent_remote": (
                        main.time.time(),
                        {"deepseek-v4.1-flash": {"config_name": "deepseek-v4.1-flash"}},
                    )
                },
            ),
        ):
            self.assertFalse(
                main._task_model_available(
                    "deepseek-v4.1-flash",
                    {"_upstream_mode": "ide", "_account_id": "account-1"},
                )
            )
            self.assertFalse(
                main._task_model_available(
                    "deepseek-v4.1-flash",
                    {"_upstream_mode": "remote", "_account_id": "account-1"},
                )
            )

    def test_remote_tier_catalog_is_account_scoped(self):
        cache = {
            main.trae_client._model_catalog_cache_key(
                user_id_override="account-1", agent_type="solo_agent_remote",
                raw_chat_function="solo_agent_remote",
            ): (
                main.time.time(),
                {
                    "deepseek-v4.1-flash": {
                        "config_name": "deepseek-v4.1-flash",
                        "model_name": "deepseek-v4.1-flash",
                    }
                },
            )
        }
        with patch.object(main.trae_client, "_WEB_MODEL_CACHE", cache):
            self.assertTrue(
                main._task_model_available(
                    "deepseek-v4.1-flash",
                    {"_upstream_mode": "remote", "_account_id": "account-1"},
                )
            )
            self.assertFalse(
                main._task_model_available(
                    "deepseek-v4.1-flash",
                    {"_upstream_mode": "remote", "_account_id": "account-2"},
                )
            )

    def test_work_catalog_uses_work_executor_and_not_agent_cache(self):
        model = {
            "deepseek-v4.1-flash": {
                "config_name": "deepseek-v4.1-flash",
                "model_name": "deepseek-v4.1-flash",
            }
        }
        work_cache = {
            main.trae_client._model_catalog_cache_key(
                user_id_override="account-1", agent_type="solo_work_remote",
                raw_chat_function="solo_work_remote",
            ): (main.time.time(), model),
        }
        with patch.object(main.trae_client, "_WEB_MODEL_CACHE", work_cache):
            self.assertTrue(
                main._task_model_available(
                    "deepseek-v4.1-flash",
                    {"_upstream_mode": "work", "_account_id": "account-1"},
                )
            )

        agent_cache = {
            main.trae_client._model_catalog_cache_key(
                user_id_override="account-1", agent_type="solo_agent_remote",
                raw_chat_function="solo_agent_remote",
            ): (main.time.time(), model),
        }
        with patch.object(main.trae_client, "_WEB_MODEL_CACHE", agent_cache):
            self.assertFalse(
                main._task_model_available(
                    "deepseek-v4.1-flash",
                    {"_upstream_mode": "work", "_account_id": "account-1"},
                )
            )

    def test_explicit_model_and_effort_are_unchanged(self):
        model, options = self._route(
            [{"role": "user", "content": "Implement this change"}],
            model="glm-5.3",
            options={"reasoning_effort": "low"},
        )

        self.assertEqual(model, "glm-5.3")
        self.assertEqual(options, {"reasoning_effort": "low"})

    def test_disabled_routing_does_not_touch_auto(self):
        model, options = self._route(
            [{"role": "user", "content": "hello"}],
            TRAE_TASK_MODEL_ROUTING="0",
        )

        self.assertEqual(model, "auto")
        self.assertEqual(options, {})


class TaskModelDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_auto_endpoint_is_resolved_before_task_model_selection(self):
        remote = AsyncMock(return_value={"route": "remote"})
        ide = AsyncMock(return_value={"route": "ide"})
        cache = {
            main.trae_client._model_catalog_cache_key(
                user_id_override="account-1", agent_type="solo_agent_remote",
                raw_chat_function="solo_agent_remote",
            ): (
                main.time.time(),
                {
                    "deepseek-v4.1-flash": {
                        "config_name": "deepseek-v4.1-flash",
                        "model_name": "deepseek-v4.1-flash",
                    }
                },
            )
        }
        with (
            patch.object(main, "UPSTREAM_MODE", "raw"),
            patch.object(main, "_auto_route_enabled", return_value=True),
            patch.object(main, "_remote_only_models", return_value=set()),
            patch.object(main.trae_client, "_WEB_MODEL_CACHE", cache),
            patch.object(main, "_run_remote_with_retry", remote),
            patch.object(main, "run_ide_chat", ide),
        ):
            chat_result = await main._dispatch_chat(
                [{"role": "user", "content": "hello"}],
                "auto",
                False,
                {"_account_id": "account-1"},
            )
            tool_result = await main._dispatch_chat(
                [{"role": "user", "content": "read file"}],
                "auto",
                False,
                {
                    "_account_id": "account-1",
                    "tools": [{"type": "function", "function": {"name": "read_file"}}],
                },
            )

        self.assertEqual(chat_result, {"route": "remote"})
        self.assertEqual(remote.await_args_list[0].args[1], "deepseek-v4.1-flash")
        self.assertEqual(tool_result, {"route": "ide"})
        self.assertEqual(ide.await_args.args[1], "auto")

    async def test_endpoint_fallback_rechecks_model_for_target_catalog(self):
        ide = AsyncMock(side_effect=RuntimeError("IDE unavailable"))
        remote = AsyncMock(return_value={"route": "remote"})
        cache = {
            main.trae_client._model_catalog_cache_key(
                user_id_override="account-1", agent_type="solo_agent_remote",
                raw_chat_function="solo_agent_remote",
            ): (
                main.time.time(),
                {
                    "deepseek-v4.1-flash": {
                        "config_name": "deepseek-v4.1-flash",
                        "model_name": "deepseek-v4.1-flash",
                    }
                },
            )
        }
        with (
            patch.object(main, "UPSTREAM_MODE", "ide"),
            patch.object(main, "_auto_route_enabled", return_value=False),
            patch.object(main, "_remote_only_models", return_value=set()),
            patch.object(main.trae_client, "_WEB_MODEL_CACHE", cache),
            patch.object(main, "run_ide_chat", ide),
            patch.object(main, "_run_remote_with_retry", remote),
        ):
            result = await main._dispatch_chat(
                [{"role": "user", "content": "hello"}],
                "auto",
                False,
                {"_account_id": "account-1"},
            )

        self.assertEqual(result, {"route": "remote"})
        self.assertEqual(ide.await_args.args[1], "auto")
        self.assertEqual(remote.await_args.args[1], "deepseek-v4.1-flash")
        self.assertEqual(
            remote.await_args.args[3]["_task_model_routed_endpoint"],
            "remote",
        )


if __name__ == "__main__":
    unittest.main()
