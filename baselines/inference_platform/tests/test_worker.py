from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from inference_platform import worker


class WorkerPoolTests(unittest.TestCase):
    @staticmethod
    def config_payload(root: Path) -> dict:
        return {
            "manifest_path": str(root / "manifest.jsonl"),
            "video_root": str(root / "data"),
            "output_path": str(root / "output.jsonl"),
            "model": "GLM-4.6V",
            "api_base_url": "https://example.invalid/v1",
            "workers": 1,
        }

    def test_ephemeral_pool_reaches_shared_scheduler_without_persisting_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            payload = {
                "run_config": {
                    "manifest_path": str(root / "manifest.jsonl"),
                    "video_root": str(root / "data"),
                    "output_path": str(root / "output.jsonl"),
                    "model": "GLM-4.6V",
                    "api_base_url": "https://example.invalid/v1",
                    "workers": 3,
                },
                "api_pool_source": "environment",
            }
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            encoded = json.dumps([
                {"name": "cloud-a", "base_url": "https://example.invalid/v1",
                 "api_key": "secret-a", "workers": 2},
                {"name": "cloud-b", "base_url": "https://example.invalid/v1",
                 "api_key": "secret-b", "workers": 1},
            ])
            captured = {}

            def fake_run(config, *, api_profiles, **kwargs):
                captured["model"] = config.model
                captured["profiles"] = [(p.name, p.workers, p.api_key) for p in api_profiles]
                captured["environment_removed"] = "ROLE_STVG_API_POOL_JSON" not in os.environ
                return {}

            with patch.object(sys, "argv", ["worker", "--config", str(config_path)]), \
                 patch.dict(os.environ, {"ROLE_STVG_API_POOL_JSON": encoded}, clear=False), \
                 patch.object(worker, "run_inference_pool", side_effect=fake_run):
                self.assertEqual(worker.main(), 0)

            persisted = config_path.read_text(encoding="utf-8")
            self.assertNotIn("secret-a", persisted)
            self.assertNotIn("secret-b", persisted)
            self.assertEqual(captured["model"], "GLM-4.6V")
            self.assertEqual(captured["profiles"], [
                ("cloud-a", 2, "secret-a"), ("cloud-b", 1, "secret-b"),
            ])
            self.assertTrue(captured["environment_removed"])

    def test_cloud_pool_never_falls_back_to_school_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "run_config": self.config_payload(root),
                "api_pool_source": "environment",
            }))
            with patch.object(sys, "argv", ["worker", "--config", str(config_path)]), \
                 patch.dict(os.environ, {
                     "SCHOOL_API_KEY": "must-not-be-used",
                     "SCHOOL_BASE_URL": "https://must-not-be-used.invalid",
                 }, clear=True), \
                 patch.object(worker, "run_inference_pool") as pooled, \
                 patch.object(worker, "run_inference") as single:
                self.assertEqual(worker.main(), 1)
                pooled.assert_not_called()
                single.assert_not_called()
            metadata = (root / "run.json").read_text(encoding="utf-8")
            self.assertNotIn("must-not-be-used", metadata)

    def test_cloud_single_api_uses_only_role_stvg_environment_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(self.config_payload(root)))
            captured = {}

            def fake_single(config, *, api_key, **kwargs):
                captured["api_key"] = api_key
                return {}

            with patch.object(sys, "argv", ["worker", "--config", str(config_path)]), \
                 patch.dict(os.environ, {
                     "SCHOOL_API_KEY": "must-not-be-used",
                     "SCHOOL_BASE_URL": "https://must-not-be-used.invalid",
                 }, clear=True), \
                 patch.object(worker, "run_inference", side_effect=fake_single):
                self.assertEqual(worker.main(), 0)
            self.assertEqual(captured["api_key"], "")


if __name__ == "__main__":
    unittest.main()
