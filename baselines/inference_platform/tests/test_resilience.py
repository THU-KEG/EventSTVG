from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest

from inference_platform.core import RunConfig, RuntimeApiProfile, infer_one, run_inference_pool
from inference_platform.resilience import CircuitBreaker, CircuitPolicy


class FakeClock:
    def __init__(self):
        self.value = 0.0
        self.sleeps: list[float] = []
        self.lock = threading.Lock()

    def monotonic(self):
        with self.lock:
            return self.value

    def sleep(self, seconds):
        with self.lock:
            self.sleeps.append(seconds)
            self.value += seconds


class SequenceAdapter:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def infer(self, model, video_path, prompt):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else "{}"
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, {}


class StatusError(Exception):
    def __init__(self, status_code, message=None):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


class ResilienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        (self.data / "oracle").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def record(self, index=0):
        path = self.data / "oracle" / f"v{index}.mp4"
        path.write_bytes(b"video")
        return {"sample_id": f"game/{index}", "setting": "oracle", "prompt": f"p{index}",
                "video_path": f"oracle/v{index}.mp4", "prompt_metadata": {}}

    def config(self, count, **changes):
        manifest = self.root / "oracle.jsonl"
        records = []
        for index in range(count):
            record = self.record(index)
            records.append({k: v for k, v in record.items() if k != "prompt_metadata"})
        manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
        values = dict(manifest_path=str(manifest), video_root=str(self.data),
                      output_path=str(self.root / "out.jsonl"), model="m",
                      api_base_url="https://gateway.invalid", workers=4,
                      max_retries=1, retry_delays_seconds=(0,), retry_jitter_fraction=0,
                      circuit_profile_failure_threshold=4,
                      circuit_gateway_failure_threshold=8)
        values.update(changes)
        return RunConfig(**values)

    def test_isolated_connection_error_backoff_then_success(self):
        record = self.record()
        adapter = SequenceAdapter([ConnectionError("reset by peer"), "{}"])
        sleeps = []
        config = self.config(1, max_retries=6,
                             retry_delays_seconds=(5, 15, 30, 60, 120, 300))
        result = infer_one(record, config, adapter, "secret", sleep=sleeps.append,
                           random_uniform=lambda low, high: 1.0)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(sleeps, [5.0])
        self.assertEqual(result["transport_diagnostics"][0]["error_category"], "connection_reset")

    def test_permanent_400_has_no_retry(self):
        record = self.record()
        adapter = SequenceAdapter([StatusError(400)])
        sleeps = []
        result = infer_one(record, self.config(1, max_retries=6), adapter, "secret",
                           sleep=sleeps.append)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["error_retryable"])
        self.assertEqual(adapter.calls, 1)
        self.assertEqual(sleeps, [])

    def test_balance_error_is_permanent_and_never_retried_by_infer_one(self):
        record = self.record()
        adapter = SequenceAdapter([StatusError(402, "payment required: insufficient balance")])
        sleeps = []
        result = infer_one(record, self.config(1, max_retries=6), adapter, "secret",
                           sleep=sleeps.append)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["error_retryable"])
        self.assertEqual(adapter.calls, 1)
        self.assertEqual(sleeps, [])

    def test_profile_isolation_and_shared_gateway_threshold(self):
        clock = FakeClock()
        policy = CircuitPolicy()
        breaker = CircuitBreaker({"dead": "g1", "healthy": "g2"}, policy=policy,
                                 monotonic=clock.monotonic)
        for _ in range(4):
            breaker.record("dead", transient_failure=True, success=False)
        self.assertFalse(breaker.eligible("dead"))
        self.assertTrue(breaker.eligible("healthy"))

        shared = CircuitBreaker({"a": "g", "b": "g"}, policy=policy,
                                monotonic=clock.monotonic)
        for _ in range(4):
            shared.record("a", transient_failure=True, success=False)
            shared.record("b", transient_failure=True, success=False)
        self.assertEqual(shared.snapshot()["gateways"]["g"]["state"], "open")

    def _profiles(self, outcomes):
        adapters = {name: SequenceAdapter(list(values)) for name, values in outcomes.items()}
        profiles = [RuntimeApiProfile(name=name, api_key=f"key-{name}",
                    base_url="https://gateway.invalid", workers=2,
                    adapter_factory=lambda adapter=adapter: adapter)
                    for name, adapter in adapters.items()]
        return adapters, profiles

    def test_gateway_outage_stops_queue_without_manifest_sweep(self):
        clock = FakeClock()
        stop = self.root / "run" / "stop.requested"
        stop.parent.mkdir()
        def controlled_sleep(seconds):
            clock.sleep(seconds)
            if seconds >= 600:
                stop.touch()
        _, profiles = self._profiles({"a": [ConnectionError("dns failure")] * 20,
                                      "b": [ConnectionError("dns failure")] * 20})
        config = self.config(20)
        final = run_inference_pool(config, api_profiles=profiles, stop_path=stop,
                    metadata_path=stop.parent / "run.json", summary_path=stop.parent / "summary.json",
                    monotonic=clock.monotonic, sleep=controlled_sleep,
                    random_uniform=lambda low, high: 1.0)
        rows = list((self.root / "out.jsonl").read_text().splitlines())
        self.assertEqual(final["status"], "stopped")
        self.assertLessEqual(len(rows), 4)
        self.assertEqual(final["completed"], 0)
        self.assertIn(600.0, clock.sleeps)

    def test_cooldown_probe_recovers_and_completes_without_duplicates(self):
        clock = FakeClock()
        # Each of four slots fails twice, opening the shared gateway. Later calls succeed.
        failures = [ConnectionError("gateway down")] * 2
        _, profiles = self._profiles({"a": failures * 2, "b": failures * 2})
        config = self.config(12)
        run = self.root / "run"
        final = run_inference_pool(config, api_profiles=profiles, stop_path=run / "stop.requested",
                    metadata_path=run / "run.json", summary_path=run / "summary.json",
                    monotonic=clock.monotonic, sleep=clock.sleep,
                    random_uniform=lambda low, high: 1.0)
        rows = [json.loads(x) for x in (self.root / "out.jsonl").read_text().splitlines()]
        ok_ids = [r["sample_id"] for r in rows if r["status"] == "ok"]
        self.assertEqual(final["completed"], 12)
        self.assertEqual(len(ok_ids), len(set(ok_ids)))
        self.assertIn(600.0, clock.sleeps)

    def test_failed_half_open_uses_fifteen_minute_cooldown(self):
        clock = FakeClock()
        stop = self.root / "run" / "stop.requested"
        stop.parent.mkdir()
        cooldowns = []
        def controlled_sleep(seconds):
            clock.sleep(seconds)
            if seconds >= 600:
                cooldowns.append(seconds)
                if len(cooldowns) == 2:
                    stop.touch()
        _, profiles = self._profiles({"a": [ConnectionError("down")] * 30,
                                      "b": [ConnectionError("down")] * 30})
        run_inference_pool(self.config(20), api_profiles=profiles, stop_path=stop,
                    metadata_path=stop.parent / "run.json", summary_path=stop.parent / "summary.json",
                    monotonic=clock.monotonic, sleep=controlled_sleep,
                    random_uniform=lambda low, high: 1.0)
        self.assertEqual(cooldowns[:2], [600.0, 900.0])
        self.assertLess(len((self.root / "out.jsonl").read_text().splitlines()), 20)

    def test_balance_exhausted_profile_is_disabled_while_healthy_profile_finishes(self):
        exhausted = SequenceAdapter(
            [StatusError(402, "insufficient balance")] * 20
        )
        healthy = SequenceAdapter(["{}"] * 20)
        profiles = [
            RuntimeApiProfile(
                name="empty", api_key="key-empty", base_url="https://gateway.invalid",
                workers=2, adapter_factory=lambda: exhausted,
            ),
            RuntimeApiProfile(
                name="funded", api_key="key-funded", base_url="https://gateway.invalid",
                workers=2, adapter_factory=lambda: healthy,
            ),
        ]
        run = self.root / "balance-isolation"
        final = run_inference_pool(
            self.config(12), api_profiles=profiles,
            stop_path=run / "stop.requested", metadata_path=run / "run.json",
            summary_path=run / "summary.json",
        )
        rows = [json.loads(x) for x in (self.root / "out.jsonl").read_text().splitlines()]
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["completed"], 12)
        self.assertEqual(final["disabled_profiles"], ["empty"])
        self.assertLessEqual(exhausted.calls, 2)  # only already-dispatched slots
        self.assertEqual(len({r["sample_id"] for r in rows if r["status"] == "ok"}), 12)

    def test_model_access_denied_profile_is_disabled_and_sample_is_requeued(self):
        denied = SequenceAdapter(
            [StatusError(403, "team_model_access_denied: team not allowed to access model")] * 20
        )
        healthy = SequenceAdapter(["{}"] * 20)
        profiles = [
            RuntimeApiProfile(
                name="denied", api_key="key-denied", base_url="https://gateway.invalid",
                workers=2, adapter_factory=lambda: denied,
            ),
            RuntimeApiProfile(
                name="allowed", api_key="key-allowed", base_url="https://gateway.invalid",
                workers=2, adapter_factory=lambda: healthy,
            ),
        ]
        run = self.root / "model-access-isolation"
        final = run_inference_pool(
            self.config(12), api_profiles=profiles,
            stop_path=run / "stop.requested", metadata_path=run / "run.json",
            summary_path=run / "summary.json",
        )
        rows = [json.loads(x) for x in (self.root / "out.jsonl").read_text().splitlines()]
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["completed"], 12)
        self.assertEqual(final["disabled_profiles"], ["denied"])
        self.assertEqual(final["model_access_denied_profiles"], ["denied"])
        self.assertLessEqual(denied.calls, 2)
        self.assertEqual(len({r["sample_id"] for r in rows if r["status"] == "ok"}), 12)

    def test_all_profiles_balance_exhausted_auto_stop_and_later_resume(self):
        empty_a = SequenceAdapter([StatusError(402, "payment required")] * 20)
        empty_b = SequenceAdapter([StatusError(429, "account quota exhausted")] * 20)
        empty_profiles = [
            RuntimeApiProfile(
                name="a", api_key="key-a", base_url="https://gateway.invalid",
                workers=2, adapter_factory=lambda: empty_a,
            ),
            RuntimeApiProfile(
                name="b", api_key="key-b", base_url="https://gateway.invalid",
                workers=2, adapter_factory=lambda: empty_b,
            ),
        ]
        run = self.root / "balance-stop"
        config = self.config(20)
        first = run_inference_pool(
            config, api_profiles=empty_profiles,
            stop_path=run / "stop.requested", metadata_path=run / "run.json",
            summary_path=run / "summary.json",
        )
        first_rows = [json.loads(x) for x in (self.root / "out.jsonl").read_text().splitlines()]
        self.assertEqual(first["status"], "stopped_balance_exhausted")
        self.assertTrue(first["balance_auto_stopped"])
        self.assertEqual(first["completed"], 0)
        self.assertEqual(first["remaining"], 20)
        self.assertEqual(first["disabled_profiles"], ["a", "b"])
        self.assertTrue(all(not row["error_retryable"] for row in first_rows))
        self.assertLessEqual(empty_a.calls, 2)
        self.assertLessEqual(empty_b.calls, 2)

        funded = SequenceAdapter(["{}"] * 30)
        funded_profiles = [RuntimeApiProfile(
            name="funded", api_key="key-funded", base_url="https://gateway.invalid",
            workers=2, adapter_factory=lambda: funded,
        )]
        resumed = run_inference_pool(
            config, api_profiles=funded_profiles,
            stop_path=run / "stop.requested", metadata_path=run / "run.json",
            summary_path=run / "summary.json",
        )
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(resumed["completed"], 20)
        all_rows = [json.loads(x) for x in (self.root / "out.jsonl").read_text().splitlines()]
        self.assertEqual(len({r["sample_id"] for r in all_rows if r["status"] == "ok"}), 20)


if __name__ == "__main__":
    unittest.main()
