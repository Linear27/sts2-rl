from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Callable, Protocol

from sts2_rl.data.trajectory import TrajectorySessionMetadata, TrajectorySessionRecorder
from sts2_rl.env.wrapper import Sts2Env
from sts2_rl.game_run_contract import GameRunContract, merge_game_run_contract_config
from sts2_rl.lifecycle import ObservationHeartbeat, normalize_optional_limit
from sts2_rl.predict import PredictorRuntimeConfig
from sts2_rl.runtime import InstanceSpec

from .community_prior import CommunityPriorRuntimeConfig
from .strategic_runtime import StrategicRuntimeConfig
from .policy import SimplePolicy, build_policy_pack


class SupportsEnv(Protocol):
    def observe(self): ...

    def step(self, action): ...

    def close(self) -> None: ...


class _NoopEnv:
    def observe(self):
        raise RuntimeError("Environment is unavailable because startup preparation failed.")

    def step(self, action):
        raise RuntimeError("Environment is unavailable because startup preparation failed.")

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class CollectionReport:
    instance_id: str
    base_url: str
    output_path: Path
    summary_path: Path
    combat_outcomes_path: Path
    step_count: int
    last_screen: str
    last_run_id: str
    stop_reason: str
    completed_run_count: int = 0
    completed_combat_count: int = 0
    observed_run_seeds: list[str] = field(default_factory=list)
    observed_run_seed_histogram: dict[str, int] = field(default_factory=dict)
    runs_without_observed_seed: int = 0
    last_observed_seed: str | None = None
    error: str | None = None


@dataclass
class _CollectionContext:
    spec: InstanceSpec
    env: SupportsEnv
    recorder: TrajectorySessionRecorder
    output_path: Path
    summary_path: Path
    combat_outcomes_path: Path
    step_count: int = 0
    last_screen: str = "UNKNOWN"
    last_run_id: str = "run_unknown"
    stop_reason: str | None = None
    error: str | None = None
    final_observation: object | None = None
    heartbeat: ObservationHeartbeat = field(default_factory=ObservationHeartbeat)


def collect_round_robin(
    instance_specs: list[InstanceSpec],
    *,
    output_root: str | Path,
    policy: SimplePolicy | None = None,
    policy_profile: str = "baseline",
    predictor_config: PredictorRuntimeConfig | None = None,
    community_prior_config: CommunityPriorRuntimeConfig | None = None,
    strategic_model_config: StrategicRuntimeConfig | None = None,
    max_steps_per_instance: int | None = 200,
    max_runs_per_instance: int | None = 1,
    max_combats_per_instance: int | None = None,
    poll_interval_seconds: float = 0.25,
    idle_timeout_seconds: float = 15.0,
    game_run_contract: GameRunContract | None = None,
    env_factory: Callable[[str], SupportsEnv] = Sts2Env.from_base_url,
) -> list[CollectionReport]:
    from sts2_rl.runtime.custom_run import contract_requires_custom_run_prepare, prepare_custom_run_from_contract

    collector_policy = policy or build_policy_pack(
        policy_profile,
        predictor_config=predictor_config,
        community_prior_config=community_prior_config,
        strategic_model_config=strategic_model_config,
    )
    max_steps_per_instance = normalize_optional_limit(max_steps_per_instance)
    max_runs_per_instance = normalize_optional_limit(max_runs_per_instance)
    max_combats_per_instance = normalize_optional_limit(max_combats_per_instance)
    session_root = Path(output_root)
    session_root.mkdir(parents=True, exist_ok=True)
    session_name = session_root.name
    should_prepare_custom_run = contract_requires_custom_run_prepare(game_run_contract) and env_factory is Sts2Env.from_base_url

    contexts: list[_CollectionContext] = []
    for spec in instance_specs:
        output_path = session_root / f"{spec.instance_id}.jsonl"
        summary_path = session_root / f"{spec.instance_id}-summary.json"
        combat_outcomes_path = session_root / f"{spec.instance_id}-combat-outcomes.jsonl"
        recorder = TrajectorySessionRecorder(
            log_path=output_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            metadata=TrajectorySessionMetadata(
                session_name=session_name,
                session_kind="collect",
                base_url=spec.base_url,
                policy_name=collector_policy.name,
                config=merge_game_run_contract_config(
                    {
                        "instance_id": spec.instance_id,
                        "max_steps_per_instance": max_steps_per_instance,
                        "max_runs_per_instance": max_runs_per_instance,
                        "max_combats_per_instance": max_combats_per_instance,
                        "poll_interval_seconds": poll_interval_seconds,
                        "idle_timeout_seconds": idle_timeout_seconds,
                        "predictor": predictor_config.as_dict() if predictor_config is not None else None,
                        "community_prior": (
                            community_prior_config.as_dict() if community_prior_config is not None else None
                        ),
                        "strategic_model": (
                            strategic_model_config.as_dict() if strategic_model_config is not None else None
                        ),
                    },
                    game_run_contract,
                ),
                game_run_contract=game_run_contract,
            ),
        )
        env: SupportsEnv = _NoopEnv()
        preparation_error: str | None = None
        if contract_requires_custom_run_prepare(game_run_contract) and not should_prepare_custom_run:
            recorder.metadata.config["custom_run_prepare_skipped"] = "custom_env_factory"
        if should_prepare_custom_run:
            try:
                preparation = prepare_custom_run_from_contract(
                    base_url=spec.base_url,
                    contract=game_run_contract,
                    request_timeout_seconds=max(idle_timeout_seconds, 30.0),
                    poll_interval_seconds=poll_interval_seconds,
                    max_idle_polls=max(1, int(idle_timeout_seconds / max(poll_interval_seconds, 0.01))),
                    max_prepare_steps=None,
                    env_factory=lambda base_url, _timeout: env_factory(base_url),
                )
                recorder.metadata.config["custom_run_prepare"] = preparation.as_dict()
                recorder.logger.log_event(
                    "custom_run_prepared",
                    {
                        **recorder.metadata.as_dict(),
                        "instance_id": spec.instance_id,
                        "custom_run_prepare": preparation.as_dict(),
                    },
                )
            except Exception as exc:
                preparation_error = f"custom_run_prepare_failed: {exc}"
                recorder.metadata.config["custom_run_prepare_error"] = preparation_error
                recorder.logger.log_event(
                    "custom_run_prepare_failed",
                    {
                        **recorder.metadata.as_dict(),
                        "instance_id": spec.instance_id,
                        "error": preparation_error,
                    },
                )
        if preparation_error is None:
            env = env_factory(spec.base_url)
        contexts.append(
            _CollectionContext(
                spec=spec,
                env=env,
                recorder=recorder,
                output_path=output_path,
                summary_path=summary_path,
                combat_outcomes_path=combat_outcomes_path,
                error=preparation_error,
            )
        )
        if preparation_error is not None:
            _stop_context(contexts[-1], "error")

    try:
        while any(context.stop_reason is None for context in contexts):
            made_progress = False

            for context in contexts:
                if context.stop_reason is not None:
                    continue

                try:
                    observation = context.env.observe()
                except Exception as exc:
                    _fail_context(context, f"observe_failed: {exc}")
                    continue

                context.final_observation = observation
                context.last_screen = observation.screen_type
                context.last_run_id = observation.run_id
                context.heartbeat.observe(observation)
                context.recorder.sync_observation(
                    observation,
                    instance_id=context.spec.instance_id,
                    step_index=context.step_count,
                )
                contract_stop_reason = context.recorder.enforce_game_run_contract(
                    instance_id=context.spec.instance_id,
                    step_index=context.step_count,
                )
                if contract_stop_reason is not None:
                    _stop_context(context, contract_stop_reason)
                    continue

                stop_reason = _collection_stop_reason(
                    context,
                    max_steps_per_instance=max_steps_per_instance,
                    max_runs_per_instance=max_runs_per_instance,
                    max_combats_per_instance=max_combats_per_instance,
                )
                if stop_reason is not None:
                    _stop_context(context, stop_reason)
                    continue

                if not observation.legal_actions:
                    context.heartbeat.note_wait()
                    if context.heartbeat.reached_idle_timeout(idle_timeout_seconds):
                        _fail_context(
                            context,
                            f"idle_timeout_after_{idle_timeout_seconds:.1f}s_on_{observation.screen_type.lower()}",
                        )
                    continue

                decision = collector_policy.choose(observation)
                if decision.action is None:
                    context.heartbeat.note_wait()
                    if context.heartbeat.reached_idle_timeout(idle_timeout_seconds):
                        stop_reason = f"policy_no_action_timeout:{observation.screen_type.lower()}:{decision.reason}"
                        context.recorder.record_capability_diagnostics(
                            instance_id=context.spec.instance_id,
                            step_index=context.step_count,
                            observation=observation,
                            decision_reason=decision.reason,
                            stop_reason=stop_reason,
                            decision_metadata=decision.trace_metadata,
                        )
                        _fail_context(
                            context,
                            stop_reason,
                        )
                    continue

                try:
                    result = context.env.step(decision.action)
                except Exception as exc:
                    _fail_context(context, f"step_failed: {exc}")
                    continue

                context.step_count += 1
                context.heartbeat.mark_progress(result.observation)
                context.last_screen = result.observation.screen_type
                context.last_run_id = result.observation.run_id
                context.final_observation = result.observation
                context.recorder.log_step(
                    instance_id=context.spec.instance_id,
                    step_index=context.step_count,
                    previous_observation=observation,
                    result=result,
                    chosen_action=decision.action,
                    policy_name=collector_policy.name,
                    policy_pack=decision.policy_pack,
                    policy_handler=decision.policy_handler,
                    decision_source="heuristic",
                    decision_stage=decision.stage,
                    decision_reason=decision.reason,
                    decision_score=decision.score,
                    planner_name=decision.planner_name,
                    planner_strategy=decision.planner_strategy,
                    ranked_actions=[_ranked_action_payload(item) for item in decision.ranked_actions],
                    decision_metadata=decision.trace_metadata,
                    reward_source="collection",
                )
                context.recorder.sync_observation(
                    result.observation,
                    instance_id=context.spec.instance_id,
                    step_index=context.step_count,
                )
                contract_stop_reason = context.recorder.enforce_game_run_contract(
                    instance_id=context.spec.instance_id,
                    step_index=context.step_count,
                )
                if contract_stop_reason is not None:
                    _stop_context(context, contract_stop_reason)
                    continue
                made_progress = True

                stop_reason = _collection_stop_reason(
                    context,
                    max_steps_per_instance=max_steps_per_instance,
                    max_runs_per_instance=max_runs_per_instance,
                    max_combats_per_instance=max_combats_per_instance,
                )
                if stop_reason is not None:
                    _stop_context(context, stop_reason)
                    continue

            if not made_progress and any(context.stop_reason is None for context in contexts):
                sleep(poll_interval_seconds)
    finally:
        for context in contexts:
            try:
                context.env.close()
            except Exception:
                pass
            if context.stop_reason is None:
                _stop_context(context, "session_closed")

    reports: list[CollectionReport] = []
    for context in contexts:
        summary_payload = json.loads(context.summary_path.read_text(encoding="utf-8"))
        progress = context.recorder.progress()
        reports.append(
            CollectionReport(
                instance_id=context.spec.instance_id,
                base_url=context.spec.base_url,
                output_path=context.output_path,
                summary_path=context.summary_path,
                combat_outcomes_path=context.combat_outcomes_path,
                step_count=context.step_count,
                last_screen=context.last_screen,
                last_run_id=context.last_run_id,
                stop_reason=context.stop_reason or "unknown",
                completed_run_count=progress.completed_run_count,
                completed_combat_count=progress.completed_combat_count,
                observed_run_seeds=list(summary_payload.get("observed_run_seeds", [])),
                observed_run_seed_histogram=dict(summary_payload.get("observed_run_seed_histogram", {})),
                runs_without_observed_seed=int(summary_payload.get("runs_without_observed_seed", 0) or 0),
                last_observed_seed=summary_payload.get("last_observed_seed"),
                error=context.error,
            )
        )
    return reports


def default_collection_session_name() -> str:
    return datetime.now(UTC).strftime("rollouts-%Y%m%d-%H%M%S")


def _ranked_action_payload(item) -> dict[str, object]:
    return {
        "action_id": item.action_id,
        "action": item.action,
        "score": item.score,
        "reason": item.reason,
        "metadata": dict(item.metadata),
    }


def _stop_context(context: _CollectionContext, reason: str) -> None:
    if context.stop_reason is not None:
        return
    context.stop_reason = reason
    context.recorder.finalize(
        instance_id=context.spec.instance_id,
        stop_reason=reason,
        step_index=context.step_count,
        error=context.error,
        final_observation=context.final_observation,
    )


def _fail_context(context: _CollectionContext, message: str) -> None:
    context.error = message
    _stop_context(context, "error")


def _collection_stop_reason(
    context: _CollectionContext,
    *,
    max_steps_per_instance: int | None,
    max_runs_per_instance: int | None,
    max_combats_per_instance: int | None,
) -> str | None:
    progress = context.recorder.progress()
    if max_steps_per_instance is not None and context.step_count >= max_steps_per_instance:
        return "max_steps_per_instance_reached"
    if max_runs_per_instance is not None and progress.completed_run_count >= max_runs_per_instance:
        return "max_runs_reached"
    if max_combats_per_instance is not None and progress.completed_combat_count >= max_combats_per_instance:
        return "max_combats_reached"
    return None
