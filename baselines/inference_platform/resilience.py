"""Transport retry diagnostics and profile/gateway circuit breaking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
from typing import Callable


@dataclass(frozen=True)
class CircuitPolicy:
    window_seconds: float = 60.0
    profile_failure_threshold: int = 4
    gateway_failure_threshold: int = 8
    transient_failure_rate: float = 0.8
    initial_cooldown_seconds: float = 600.0
    repeated_cooldown_seconds: float = 900.0


@dataclass
class _Circuit:
    state: str = "closed"
    open_until: float = 0.0
    probe_profiles: set[str] = field(default_factory=set)
    successful_probe_profiles: set[str] = field(default_factory=set)
    failed_probe_profiles: set[str] = field(default_factory=set)
    outage_cycles: int = 0


class CircuitBreaker:
    """Track failures separately by credential profile and shared base URL."""

    def __init__(self, profile_gateways: dict[str, str], *, policy: CircuitPolicy,
                 monotonic: Callable[[], float]):
        self.profile_gateways = profile_gateways
        self.policy = policy
        self.monotonic = monotonic
        self.profile_events = {name: deque() for name in profile_gateways}
        self.gateway_events = {url: deque() for url in set(profile_gateways.values())}
        self.profiles = {name: _Circuit() for name in profile_gateways}
        self.gateways = {url: _Circuit() for url in set(profile_gateways.values())}
        self.lock = threading.RLock()

    def _advance(self, circuit: _Circuit, now: float) -> None:
        if circuit.state == "open" and now >= circuit.open_until:
            circuit.state = "half_open"
            circuit.probe_profiles.clear()
            circuit.successful_probe_profiles.clear()
            circuit.failed_probe_profiles.clear()

    def eligible(self, profile: str) -> bool:
        with self.lock:
            now = self.monotonic()
            gateway = self.profile_gateways[profile]
            pc, gc = self.profiles[profile], self.gateways[gateway]
            self._advance(pc, now)
            self._advance(gc, now)
            if pc.state == "open" or gc.state == "open":
                return False
            if pc.state == "half_open" and profile in pc.probe_profiles:
                return False
            if gc.state == "half_open" and profile in gc.probe_profiles:
                return False
            return True

    def mark_started(self, profile: str) -> None:
        with self.lock:
            gateway = self.profile_gateways[profile]
            for circuit in (self.profiles[profile], self.gateways[gateway]):
                if circuit.state == "half_open":
                    circuit.probe_profiles.add(profile)

    def _trim(self, events: deque[tuple[float, bool, str]], now: float) -> None:
        cutoff = now - self.policy.window_seconds
        while events and events[0][0] < cutoff:
            events.popleft()

    def _open(self, circuit: _Circuit, now: float, *, repeated: bool = False) -> None:
        # Other in-flight workers may report the same outage after the circuit
        # has opened. They must not extend the cooldown or count as new cycles.
        if circuit.state == "open" and now < circuit.open_until:
            return
        circuit.outage_cycles += 1
        cooldown = (self.policy.repeated_cooldown_seconds if repeated or circuit.outage_cycles > 1
                    else self.policy.initial_cooldown_seconds)
        circuit.state = "open"
        circuit.open_until = now + cooldown
        circuit.probe_profiles.clear()
        circuit.successful_probe_profiles.clear()
        circuit.failed_probe_profiles.clear()

    def _resolve_gateway_probe(self, gateway: str, circuit: _Circuit, now: float) -> None:
        expected = {p for p, url in self.profile_gateways.items() if url == gateway}
        finished = circuit.successful_probe_profiles | circuit.failed_probe_profiles
        if not expected <= finished:
            return
        if circuit.successful_probe_profiles:
            # The transport path is alive. Profiles whose individual probes
            # failed remain isolated by their own profile circuits.
            circuit.state, circuit.outage_cycles = "closed", 0
            self.gateway_events[gateway].clear()
        else:
            self._open(circuit, now, repeated=True)

    def record(self, profile: str, *, transient_failure: bool, success: bool) -> None:
        with self.lock:
            self._record(profile, transient_failure=transient_failure, success=success)

    def _record(self, profile: str, *, transient_failure: bool, success: bool) -> None:
        now = self.monotonic()
        gateway = self.profile_gateways[profile]
        pc, gc = self.profiles[profile], self.gateways[gateway]
        self._advance(pc, now)
        self._advance(gc, now)
        if success:
            if pc.state == "half_open":
                pc.state, pc.outage_cycles = "closed", 0
                self.profile_events[profile].clear()
            if gc.state in {"open", "half_open"}:
                # Resolve the shared gateway after each profile's probe reports.
                gc.successful_probe_profiles.add(profile)
                self._resolve_gateway_probe(gateway, gc, now)
            self.profile_events[profile].append((now, False, profile))
            self.gateway_events[gateway].append((now, False, profile))
            return
        if not transient_failure:
            return
        if pc.state == "half_open":
            self._open(pc, now, repeated=True)
        if gc.state == "half_open":
            gc.failed_probe_profiles.add(profile)
            self._resolve_gateway_probe(gateway, gc, now)
        pe, ge = self.profile_events[profile], self.gateway_events[gateway]
        pe.append((now, True, profile))
        ge.append((now, True, profile))
        self._trim(pe, now)
        self._trim(ge, now)
        p_failures = sum(failed for _, failed, _ in pe)
        if (len(pe) >= self.policy.profile_failure_threshold and
                p_failures / len(pe) >= self.policy.transient_failure_rate):
            self._open(pc, now)
        g_failures = sum(failed for _, failed, _ in ge)
        failing_profiles = {p for _, failed, p in ge if failed}
        if (len(failing_profiles) >= 2 and
                len(ge) >= self.policy.gateway_failure_threshold and
                g_failures / len(ge) >= self.policy.transient_failure_rate):
            self._open(gc, now)

    def seconds_until_any_eligible(self, profiles: list[str]) -> float:
        with self.lock:
            now = self.monotonic()
            waits = []
            for profile in profiles:
                gateway = self.profile_gateways[profile]
                circuits = (self.profiles[profile], self.gateways[gateway])
                deadlines = [c.open_until for c in circuits if c.state == "open"]
                waits.append(max(deadlines, default=now) - now)
            return max(0.0, min(waits, default=0.0))

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "profiles": {
                    name: {
                        "state": circuit.state,
                        "open_until_monotonic": circuit.open_until,
                        "outage_cycles": circuit.outage_cycles,
                    }
                    for name, circuit in self.profiles.items()
                },
                "gateways": {
                    url: {
                        "state": circuit.state,
                        "open_until_monotonic": circuit.open_until,
                        "outage_cycles": circuit.outage_cycles,
                    }
                    for url, circuit in self.gateways.items()
                },
            }
