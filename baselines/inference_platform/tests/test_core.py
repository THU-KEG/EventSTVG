from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from inference_platform.core import (
    RunConfig,
    RuntimeApiProfile,
    discover_manifests,
    load_manifest,
    resolve_video_path,
    run_inference,
    run_inference_pool,
    summarize_results,
)


class FakeAdapter:
    def __init__(self, responses: dict[str, str]):
        self.responses = responses

    def infer(self, model: str, video_path: Path, prompt: str):
        return self.responses[prompt], {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }


class RetryOnceAdapter:
    def __init__(self):
        self.calls = 0

    def infer(self, model: str, video_path: Path, prompt: str):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("temporary timeout")
        return "{}", {}


class PermanentFailureAdapter:
    def __init__(self):
        self.calls = 0

    def infer(self, model: str, video_path: Path, prompt: str):
        self.calls += 1
        raise ValueError("permanent invalid request")


class AlwaysTransientAdapter:
    def infer(self, model: str, video_path: Path, prompt: str):
        raise TimeoutError("temporary profile outage")


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifests = self.root / "prompt"
        self.videos = self.root / "data"
        self.manifests.mkdir()
        (self.videos / "oracle").mkdir(parents=True)
        self.records = [
            {
                "sample_id": "game/a",
                "setting": "oracle",
                "prompt": "prompt-a",
                "video_path": "oracle/game_a.mp4",
                "answer": {"must": "never enter inference"},
            },
            {
                "sample_id": "game/b",
                "setting": "oracle",
                "prompt": "prompt-b",
                "video_path": "oracle/game_b.mp4",
            },
        ]
        self.manifest = self.manifests / "oracle.jsonl"
        self.manifest.write_text(
            "".join(json.dumps(record) + "\n" for record in self.records), encoding="utf-8"
        )
        for name in ("game_a.mp4", "game_b.mp4"):
            (self.videos / "oracle" / name).write_bytes(b"video")

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, output: Path) -> RunConfig:
        return RunConfig(
            manifest_path=str(self.manifest), video_root=str(self.videos),
            output_path=str(output), model="test-model", api_base_url="https://example.invalid/v1",
            workers=2, max_retries=0,
        )

    def test_manifest_discovery_and_ground_truth_exclusion(self):
        self.assertEqual(discover_manifests(self.manifests), {"oracle": self.manifest.resolve()})
        normalized = load_manifest(self.manifest)
        self.assertNotIn("answer", normalized[0])
        self.assertEqual(normalized[0]["prompt"], "prompt-a")

    def test_video_paths_must_be_relative_and_inside_root(self):
        self.assertEqual(
            resolve_video_path(self.videos, "oracle/game_a.mp4"),
            (self.videos / "oracle/game_a.mp4").resolve(),
        )
        with self.assertRaises(ValueError):
            resolve_video_path(self.videos, "/tmp/video.mp4")
        with self.assertRaises(ValueError):
            resolve_video_path(self.videos, "../video.mp4")

    def test_incremental_results_resume_and_sample_based_progress(self):
        output = self.root / "results.jsonl"
        run_dir = self.root / "run"
        responses = {"prompt-a": '[{"time_sec": 0}]', "prompt-b": "not json"}
        final = run_inference(
            self.config(output), api_key="secret-value",
            adapter_factory=lambda: FakeAdapter(responses),
            stop_path=run_dir / "stop.requested", metadata_path=run_dir / "run.json",
            summary_path=run_dir / "summary.json",
        )
        self.assertEqual(final["completed"], 2)
        self.assertEqual(final["parse_failures"], 1)
        first_rows = output.read_text().splitlines()
        self.assertEqual(len(first_rows), 2)
        self.assertNotIn("prompt-a", first_rows[0])
        self.assertNotIn("secret-value", output.read_text())

        run_inference(
            self.config(output), api_key="secret-value",
            adapter_factory=lambda: FakeAdapter(responses),
            stop_path=run_dir / "stop.requested", metadata_path=run_dir / "run.json",
            summary_path=run_dir / "summary.json",
        )
        self.assertEqual(len(output.read_text().splitlines()), 2)

    def test_failed_attempt_does_not_complete_sample(self):
        output = self.root / "failed.jsonl"
        output.write_text(json.dumps({
            "sample_id": "game/a", "status": "error", "retry_count": 2,
            "latency_seconds": 1.0, "usage": {}, "error_type": "TimeoutError",
            "error_message": "timeout",
        }) + "\n")
        summary = summarize_results(output, total_samples=2)
        self.assertEqual(summary["attempted_samples"], 1)
        self.assertEqual(summary["completed"], 0)
        self.assertEqual(summary["remaining"], 2)
        self.assertEqual(summary["request_failures"], 1)

    def test_resume_refuses_a_different_protocol(self):
        output = self.root / "mixed.jsonl"
        output.write_text(json.dumps({
            "sample_id": "game/a", "setting": "oracle", "model": "test-model",
            "status": "ok", "inference_protocol": "old_tuned_protocol",
        }) + "\n")
        run_dir = self.root / "mixed-run"
        with self.assertRaisesRegex(ValueError, "not from 'provider_defaults'"):
            run_inference(
                self.config(output), api_key="secret-value",
                adapter_factory=lambda: FakeAdapter({}),
                stop_path=run_dir / "stop.requested", metadata_path=run_dir / "run.json",
                summary_path=run_dir / "summary.json",
            )

    def test_transient_failure_retries_then_saves_success(self):
        output = self.root / "retry.jsonl"
        config = self.config(output)
        config = RunConfig(**{**config.__dict__, "workers": 1, "sample_limit": 1, "max_retries": 2})
        run_dir = self.root / "retry-run"
        adapter = RetryOnceAdapter()
        final = run_inference(
            config, api_key="secret-value", adapter_factory=lambda: adapter,
            stop_path=run_dir / "stop.requested", metadata_path=run_dir / "run.json",
            summary_path=run_dir / "summary.json",
        )
        saved = json.loads(output.read_text())
        self.assertEqual(saved["status"], "ok")
        self.assertEqual(saved["retry_count"], 1)
        self.assertEqual(final["retried"], 1)

    def test_explicit_rerun_appends_without_erasing_success(self):
        output = self.root / "rerun.jsonl"
        run_dir = self.root / "rerun-run"
        responses = {
            "prompt-a": "refusal text",
            "prompt-b": '[{"time_sec": 0, "goalkeeper": null}]',
        }
        base = self.config(output)
        run_inference(
            base, api_key="secret-value", adapter_factory=lambda: FakeAdapter(responses),
            stop_path=run_dir / "stop.requested", metadata_path=run_dir / "run.json",
            summary_path=run_dir / "summary.json",
        )
        first_rows = output.read_text().splitlines()
        self.assertEqual(len(first_rows), 2)
        first_by_id = {row["sample_id"]: row for row in map(json.loads, first_rows)}
        self.assertFalse(first_by_id["game/a"]["parse_ok"])

        rerun = RunConfig(**{
            **base.__dict__, "rerun_sample_ids": ["game/a"],
        })
        final = run_inference(
            rerun, api_key="secret-value", adapter_factory=lambda: FakeAdapter(responses),
            stop_path=run_dir / "stop.requested", metadata_path=run_dir / "run.json",
            summary_path=run_dir / "summary.json",
        )
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["sample_id"] for row in rows].count("game/a"), 2)
        self.assertEqual(final["completed"], 2)
        self.assertEqual(final["parse_failures"], 1)

    def test_explicit_rerun_appends_to_unmarked_legacy_without_auto_resume(self):
        output = self.root / "legacy.jsonl"
        run_dir = self.root / "legacy-run"
        legacy_rows = [
            {"sample_id": "game/a", "setting": "oracle", "model": "test-model",
             "status": "error", "error_type": "TimeoutError", "inference_parameters": {}},
            {"sample_id": "game/b", "setting": "oracle", "model": "test-model",
             "status": "ok", "raw_response": "{}", "inference_parameters": {}},
        ]
        output.write_text("".join(json.dumps(row) + "\n" for row in legacy_rows))
        base = self.config(output)

        rerun = RunConfig(**{
            **base.__dict__, "rerun_sample_ids": ["game/a"],
        })
        run_inference(
            rerun, api_key="secret-value",
            adapter_factory=lambda: FakeAdapter({"prompt-a": "{}", "prompt-b": "{}"}),
            stop_path=run_dir / "stop.requested", metadata_path=run_dir / "run.json",
            summary_path=run_dir / "summary.json",
        )

        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[:2], legacy_rows)
        self.assertEqual(rows[2]["sample_id"], "game/a")
        self.assertEqual(rows[2]["inference_protocol"], "provider_defaults")

    def test_permanent_failure_requires_explicit_rerun(self):
        output = self.root / "permanent.jsonl"
        run_dir = self.root / "permanent-run"
        base = RunConfig(**{
            **self.config(output).__dict__, "workers": 1, "sample_limit": 1,
        })
        first_adapter = PermanentFailureAdapter()
        run_inference(
            base, api_key="secret-value", adapter_factory=lambda: first_adapter,
            stop_path=run_dir / "stop.requested", metadata_path=run_dir / "run.json",
            summary_path=run_dir / "summary.json",
        )
        first = json.loads(output.read_text())
        self.assertFalse(first["error_retryable"])
        self.assertEqual(first_adapter.calls, 1)

        second_adapter = PermanentFailureAdapter()
        run_inference(
            base, api_key="secret-value", adapter_factory=lambda: second_adapter,
            stop_path=run_dir / "stop.requested", metadata_path=run_dir / "run.json",
            summary_path=run_dir / "summary.json",
        )
        self.assertEqual(second_adapter.calls, 0)
        self.assertEqual(len(output.read_text().splitlines()), 1)

        forced = RunConfig(**{
            **base.__dict__, "rerun_sample_ids": ["game/a"],
        })
        third_adapter = PermanentFailureAdapter()
        run_inference(
            forced, api_key="secret-value", adapter_factory=lambda: third_adapter,
            stop_path=run_dir / "stop.requested", metadata_path=run_dir / "run.json",
            summary_path=run_dir / "summary.json",
        )
        self.assertEqual(third_adapter.calls, 1)
        self.assertEqual(len(output.read_text().splitlines()), 2)

    def test_api_pool_records_profile_and_resumes_transient_failure(self):
        output = self.root / "pool.jsonl"
        run_dir = self.root / "pool-run"
        config = RunConfig(**{**self.config(output).__dict__, "workers": 2})
        profiles = [
            RuntimeApiProfile(
                name="down", api_key="secret-down", base_url="https://example.invalid/v1",
                workers=1, adapter_factory=AlwaysTransientAdapter,
            ),
            RuntimeApiProfile(
                name="healthy", api_key="secret-healthy", base_url="https://example.invalid/v1",
                workers=1, adapter_factory=lambda: FakeAdapter({"prompt-a": "{}", "prompt-b": "{}"}),
            ),
        ]
        first = run_inference_pool(
            config, api_profiles=profiles,
            stop_path=run_dir / "stop.requested", metadata_path=run_dir / "run.json",
            summary_path=run_dir / "summary.json",
        )
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual({row["api_profile"] for row in rows}, {"down", "healthy"})
        self.assertEqual(first["completed"], 2)
        self.assertNotIn("secret-down", output.read_text())
        self.assertNotIn("secret-healthy", output.read_text())

        recovered = [
            RuntimeApiProfile(
                name="recovered", api_key="secret-new", base_url="https://example.invalid/v1",
                workers=1, adapter_factory=lambda: FakeAdapter({"prompt-a": "{}", "prompt-b": "{}"}),
            )
        ]
        second = run_inference_pool(
            config, api_profiles=recovered,
            stop_path=run_dir / "stop.requested", metadata_path=run_dir / "run.json",
            summary_path=run_dir / "summary.json",
        )
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(rows), len(output.read_text().splitlines()))
        self.assertEqual(second["completed"], 2)
        self.assertNotIn("recovered", {row["api_profile"] for row in rows})

    def test_api_pool_single_writer_persists_valid_unique_rows(self):
        output = self.root / "pool-many.jsonl"
        run_dir = self.root / "pool-many-run"
        records = []
        responses = {}
        for index in range(40):
            prompt = f"prompt-{index}"
            records.append({
                "sample_id": f"game/{index}", "setting": "oracle", "prompt": prompt,
                "video_path": f"oracle/video_{index}.mp4",
            })
            (self.videos / "oracle" / f"video_{index}.mp4").write_bytes(b"video")
            responses[prompt] = "{}"
        self.manifest.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        config = RunConfig(**{**self.config(output).__dict__, "workers": 4})
        profiles = [
            RuntimeApiProfile(
                name="pool-a", api_key="key-a", base_url="https://example.invalid/v1",
                workers=2, adapter_factory=lambda: FakeAdapter(responses),
            ),
            RuntimeApiProfile(
                name="pool-b", api_key="key-b", base_url="https://example.invalid/v1",
                workers=2, adapter_factory=lambda: FakeAdapter(responses),
            ),
        ]
        run_inference_pool(
            config, api_profiles=profiles,
            stop_path=run_dir / "stop.requested", metadata_path=run_dir / "run.json",
            summary_path=run_dir / "summary.json",
        )
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({row["sample_id"] for row in rows}), 40)
        self.assertEqual({row["api_profile"] for row in rows}, {"pool-a", "pool-b"})


if __name__ == "__main__":
    unittest.main()
