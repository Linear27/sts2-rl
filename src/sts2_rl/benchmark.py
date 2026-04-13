from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from time import perf_counter

from .env import Sts2Client, build_candidate_actions
from .env.models import AvailableActionsPayload, GameStatePayload, HealthPayload


@dataclass(frozen=True)
class MeasurementSummary:
    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float


@dataclass(frozen=True)
class HealthBenchmarkResult:
    health: HealthPayload
    latency: MeasurementSummary


@dataclass(frozen=True)
class ObserveBenchmarkResult:
    state_latency: MeasurementSummary
    actions_latency: MeasurementSummary
    candidate_build_latency: MeasurementSummary
    candidate_count: MeasurementSummary
    last_screen: str
    last_run_id: str
    build_warnings: list[str]


def benchmark_health(base_url: str, *, samples: int = 5) -> HealthBenchmarkResult:
    with Sts2Client(base_url) as client:
        latencies: list[float] = []
        health: HealthPayload | None = None
        for _ in range(samples):
            started = perf_counter()
            health = client.get_health()
            latencies.append((perf_counter() - started) * 1000)

    assert health is not None
    return HealthBenchmarkResult(
        health=health,
        latency=summarize_measurements(latencies),
    )


def benchmark_observe(base_url: str, *, samples: int = 5) -> ObserveBenchmarkResult:
    with Sts2Client(base_url) as client:
        return benchmark_observe_with_client(client, samples=samples)


def benchmark_observe_with_client(client: Sts2Client, *, samples: int = 5) -> ObserveBenchmarkResult:
    state_latencies: list[float] = []
    action_latencies: list[float] = []
    build_latencies: list[float] = []
    candidate_counts: list[float] = []

    state: GameStatePayload | None = None
    actions: AvailableActionsPayload | None = None
    warnings: list[str] = []

    for _ in range(samples):
        started = perf_counter()
        state = client.get_state()
        state_latencies.append((perf_counter() - started) * 1000)

        started = perf_counter()
        actions = client.get_available_actions()
        action_latencies.append((perf_counter() - started) * 1000)

        started = perf_counter()
        result = build_candidate_actions(state, actions)
        build_latencies.append((perf_counter() - started) * 1000)
        candidate_counts.append(float(len(result.candidates)))
        warnings = result.unsupported_actions

    assert state is not None
    assert actions is not None
    return ObserveBenchmarkResult(
        state_latency=summarize_measurements(state_latencies),
        actions_latency=summarize_measurements(action_latencies),
        candidate_build_latency=summarize_measurements(build_latencies),
        candidate_count=summarize_measurements(candidate_counts),
        last_screen=state.screen,
        last_run_id=state.run_id,
        build_warnings=warnings,
    )


def summarize_measurements(values: list[float]) -> MeasurementSummary:
    if not values:
        raise ValueError("values must not be empty")

    ordered = sorted(values)
    return MeasurementSummary(
        count=len(ordered),
        min_ms=ordered[0],
        max_ms=ordered[-1],
        mean_ms=mean(ordered),
        p50_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
    )


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] + (values[upper] - values[lower]) * weight
