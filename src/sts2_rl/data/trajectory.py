from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sts2_rl.capability import (
    diagnostics_from_no_action,
    diagnostics_from_observation,
    empty_capability_summary,
    summarize_capability_diagnostics,
)
from sts2_rl.env.models import GameStatePayload
from sts2_rl.env.types import CandidateAction, StepObservation, StepResult
from sts2_rl.game_run_contract import (
    GameRunContract,
    GameRunContractObservation,
    build_game_run_contract_validation_payload,
    inspect_game_run_contract,
)

TRAJECTORY_SCHEMA_VERSION = 4


class DataModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrajectoryStepRecord(DataModel):
    schema_version: int = TRAJECTORY_SCHEMA_VERSION
    record_type: str = "step"
    timestamp_utc: str
    session_name: str
    session_kind: str
    instance_id: str
    step_index: int
    run_id: str
    observed_seed: str | None = None
    screen_type: str
    floor: int | None = None
    legal_action_count: int
    legal_action_ids: list[str] = Field(default_factory=list)
    build_warnings: list[str] = Field(default_factory=list)
    next_run_id: str | None = None
    next_observed_seed: str | None = None
    next_screen_type: str | None = None
    next_floor: int | None = None
    next_legal_action_count: int | None = None
    next_legal_action_ids: list[str] = Field(default_factory=list)
    next_build_warnings: list[str] = Field(default_factory=list)
    chosen_action_id: str | None = None
    chosen_action_label: str | None = None
    chosen_action_source: str | None = None
    chosen_action: dict[str, Any] | None = None
    policy_name: str | None = None
    policy_pack: str | None = None
    policy_handler: str | None = None
    algorithm: str | None = None
    decision_source: str | None = None
    decision_stage: str | None = None
    decision_reason: str | None = None
    decision_score: float | None = None
    planner_name: str | None = None
    planner_strategy: str | None = None
    ranked_action_count: int = 0
    ranked_actions: list[dict[str, Any]] = Field(default_factory=list)
    decision_metadata: dict[str, Any] = Field(default_factory=dict)
    capability_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    reward: float = 0.0
    reward_source: str | None = None
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)
    model_metrics: dict[str, Any] = Field(default_factory=dict)
    state_summary: dict[str, Any] = Field(default_factory=dict)
    action_descriptors: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    next_state_summary: dict[str, Any] = Field(default_factory=dict)
    next_action_descriptors: dict[str, Any] = Field(default_factory=dict)
    next_state: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] | None = None


class CombatOutcomeRecord(DataModel):
    schema_version: int = TRAJECTORY_SCHEMA_VERSION
    record_type: str = "combat_finished"
    timestamp_utc: str
    session_name: str
    session_kind: str
    instance_id: str
    run_id: str
    observed_seed: str | None = None
    floor: int | None = None
    combat_index: int
    started_step_index: int
    finished_step_index: int
    outcome: str
    cumulative_reward: float
    step_count: int
    enemy_ids: list[str] = Field(default_factory=list)
    damage_dealt: int = 0
    damage_taken: int = 0
    start_summary: dict[str, Any] = Field(default_factory=dict)
    end_summary: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class TrajectorySessionMetadata:
    session_name: str
    session_kind: str
    base_url: str
    policy_name: str
    algorithm: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    game_run_contract: GameRunContract | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "session_name": self.session_name,
            "session_kind": self.session_kind,
            "base_url": self.base_url,
            "policy_name": self.policy_name,
            "algorithm": self.algorithm,
            "config": self.config,
            "game_run_contract": self.game_run_contract.as_dict() if self.game_run_contract is not None else None,
        }


@dataclass(frozen=True)
class TrajectorySessionSummary:
    schema_version: int
    session_name: str
    session_kind: str
    base_url: str
    policy_name: str
    algorithm: str | None
    config: dict[str, Any]
    game_run_contract: dict[str, Any] | None
    game_run_contract_validation: dict[str, Any]
    started_at_utc: str
    ended_at_utc: str
    stop_reason: str
    error: str | None
    env_steps: int
    total_reward: float
    run_count: int
    completed_run_count: int
    won_runs: int
    lost_runs: int
    interrupted_runs: int
    observed_run_seeds: list[str]
    observed_run_seed_histogram: dict[str, int]
    runs_without_observed_seed: int
    floor_count: int
    completed_floor_count: int
    combat_count: int
    completed_combat_count: int
    won_combats: int
    lost_combats: int
    interrupted_combats: int
    run_outcome_histogram: dict[str, int]
    run_finish_reason_histogram: dict[str, int]
    screen_histogram: dict[str, int]
    action_histogram: dict[str, int]
    policy_pack_histogram: dict[str, int]
    policy_handler_histogram: dict[str, int]
    decision_stage_histogram: dict[str, int]
    decision_reason_histogram: dict[str, int]
    decision_source_histogram: dict[str, int]
    action_source_histogram: dict[str, int]
    planner_histogram: dict[str, int]
    planner_candidate_count_stats: dict[str, float | int | None]
    planner_decision_score_stats: dict[str, float | int | None]
    route_planner_step_count: int
    route_planner_boss_histogram: dict[str, int]
    route_planner_reason_tag_histogram: dict[str, int]
    route_planner_path_length_stats: dict[str, float | int | None]
    route_planner_selected_score_stats: dict[str, float | int | None]
    predictor_mode_histogram: dict[str, int]
    predictor_domain_histogram: dict[str, int]
    predictor_model_histogram: dict[str, int]
    predictor_value_estimate_stats: dict[str, float | int | None]
    predictor_outcome_win_probability_stats: dict[str, float | int | None]
    predictor_expected_reward_stats: dict[str, float | int | None]
    predictor_expected_damage_delta_stats: dict[str, float | int | None]
    non_combat_capability: dict[str, Any]
    last_screen: str
    last_run_id: str
    last_observed_seed: str | None
    last_floor: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_name": self.session_name,
            "session_kind": self.session_kind,
            "base_url": self.base_url,
            "policy_name": self.policy_name,
            "algorithm": self.algorithm,
            "config": self.config,
            "game_run_contract": self.game_run_contract,
            "game_run_contract_validation": self.game_run_contract_validation,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "env_steps": self.env_steps,
            "total_reward": self.total_reward,
            "run_count": self.run_count,
            "completed_run_count": self.completed_run_count,
            "won_runs": self.won_runs,
            "lost_runs": self.lost_runs,
            "interrupted_runs": self.interrupted_runs,
            "observed_run_seeds": self.observed_run_seeds,
            "observed_run_seed_histogram": self.observed_run_seed_histogram,
            "runs_without_observed_seed": self.runs_without_observed_seed,
            "floor_count": self.floor_count,
            "completed_floor_count": self.completed_floor_count,
            "combat_count": self.combat_count,
            "completed_combat_count": self.completed_combat_count,
            "won_combats": self.won_combats,
            "lost_combats": self.lost_combats,
            "interrupted_combats": self.interrupted_combats,
            "run_outcome_histogram": self.run_outcome_histogram,
            "run_finish_reason_histogram": self.run_finish_reason_histogram,
            "screen_histogram": self.screen_histogram,
            "action_histogram": self.action_histogram,
            "policy_pack_histogram": self.policy_pack_histogram,
            "policy_handler_histogram": self.policy_handler_histogram,
            "decision_stage_histogram": self.decision_stage_histogram,
            "decision_reason_histogram": self.decision_reason_histogram,
            "decision_source_histogram": self.decision_source_histogram,
            "action_source_histogram": self.action_source_histogram,
            "planner_histogram": self.planner_histogram,
            "planner_candidate_count_stats": self.planner_candidate_count_stats,
            "planner_decision_score_stats": self.planner_decision_score_stats,
            "route_planner_step_count": self.route_planner_step_count,
            "route_planner_boss_histogram": self.route_planner_boss_histogram,
            "route_planner_reason_tag_histogram": self.route_planner_reason_tag_histogram,
            "route_planner_path_length_stats": self.route_planner_path_length_stats,
            "route_planner_selected_score_stats": self.route_planner_selected_score_stats,
            "predictor_mode_histogram": self.predictor_mode_histogram,
            "predictor_domain_histogram": self.predictor_domain_histogram,
            "predictor_model_histogram": self.predictor_model_histogram,
            "predictor_value_estimate_stats": self.predictor_value_estimate_stats,
            "predictor_outcome_win_probability_stats": self.predictor_outcome_win_probability_stats,
            "predictor_expected_reward_stats": self.predictor_expected_reward_stats,
            "predictor_expected_damage_delta_stats": self.predictor_expected_damage_delta_stats,
            "non_combat_capability": self.non_combat_capability,
            "last_screen": self.last_screen,
            "last_run_id": self.last_run_id,
            "last_observed_seed": self.last_observed_seed,
            "last_floor": self.last_floor,
        }


@dataclass
class _SessionStats:
    started_at_utc: str
    env_steps: int = 0
    total_reward: float = 0.0
    run_count: int = 0
    completed_run_count: int = 0
    won_runs: int = 0
    lost_runs: int = 0
    interrupted_runs: int = 0
    runs_without_observed_seed: int = 0
    floor_count: int = 0
    completed_floor_count: int = 0
    combat_count: int = 0
    completed_combat_count: int = 0
    won_combats: int = 0
    lost_combats: int = 0
    interrupted_combats: int = 0
    observation_contract_check_count: int = 0
    observation_contract_match_count: int = 0
    observation_contract_mismatch_count: int = 0
    last_screen: str = "UNKNOWN"
    last_run_id: str = "run_unknown"
    last_observed_seed: str | None = None
    last_floor: int | None = None
    run_outcome_histogram: Counter[str] = field(default_factory=Counter)
    run_finish_reason_histogram: Counter[str] = field(default_factory=Counter)
    observed_run_seed_histogram: Counter[str] = field(default_factory=Counter)
    observed_character_histogram: Counter[str] = field(default_factory=Counter)
    observed_ascension_histogram: Counter[int] = field(default_factory=Counter)
    contract_mismatch_histogram: Counter[str] = field(default_factory=Counter)
    screen_histogram: Counter[str] = field(default_factory=Counter)
    action_histogram: Counter[str] = field(default_factory=Counter)
    policy_pack_histogram: Counter[str] = field(default_factory=Counter)
    policy_handler_histogram: Counter[str] = field(default_factory=Counter)
    decision_stage_histogram: Counter[str] = field(default_factory=Counter)
    decision_reason_histogram: Counter[str] = field(default_factory=Counter)
    decision_source_histogram: Counter[str] = field(default_factory=Counter)
    action_source_histogram: Counter[str] = field(default_factory=Counter)
    planner_histogram: Counter[str] = field(default_factory=Counter)
    planner_candidate_counts: list[int] = field(default_factory=list)
    planner_decision_scores: list[float] = field(default_factory=list)
    route_planner_steps: int = 0
    route_planner_boss_histogram: Counter[str] = field(default_factory=Counter)
    route_planner_reason_tag_histogram: Counter[str] = field(default_factory=Counter)
    route_planner_path_lengths: list[int] = field(default_factory=list)
    route_planner_selected_scores: list[float] = field(default_factory=list)
    predictor_mode_histogram: Counter[str] = field(default_factory=Counter)
    predictor_domain_histogram: Counter[str] = field(default_factory=Counter)
    predictor_model_histogram: Counter[str] = field(default_factory=Counter)
    predictor_value_estimates: list[float] = field(default_factory=list)
    predictor_outcome_win_probabilities: list[float] = field(default_factory=list)
    predictor_expected_rewards: list[float] = field(default_factory=list)
    predictor_expected_damage_deltas: list[float] = field(default_factory=list)
    last_contract_mismatches: list[str] = field(default_factory=list)
    capability_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    capability_diagnostic_keys: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class TrajectorySessionProgress:
    env_steps: int
    total_reward: float
    run_count: int
    completed_run_count: int
    won_runs: int
    lost_runs: int
    interrupted_runs: int
    runs_without_observed_seed: int
    floor_count: int
    completed_floor_count: int
    combat_count: int
    completed_combat_count: int
    won_combats: int
    lost_combats: int
    interrupted_combats: int
    last_screen: str
    last_run_id: str
    last_observed_seed: str | None
    last_floor: int | None
    current_run_id: str | None
    current_floor: int | None
    combat_in_progress: bool


@dataclass
class _RunSpan:
    run_id: str
    started_step_index: int
    start_summary: dict[str, Any]
    observed_seed: str | None
    observed_character_id: str | None
    observed_ascension: int | None


@dataclass
class _FloorSpan:
    run_id: str
    floor: int
    started_step_index: int
    start_summary: dict[str, Any]


@dataclass
class _CombatSpan:
    run_id: str
    floor: int | None
    combat_index: int
    started_step_index: int
    start_summary: dict[str, Any]
    enemy_ids: list[str]
    start_player_hp: int
    start_enemy_hp: int
    cumulative_reward: float = 0.0


class JsonlTrajectoryLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_record(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    def log_event(self, record_type: str, payload: dict[str, Any]) -> None:
        record = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "record_type": record_type,
            "timestamp_utc": utc_now_iso(),
            **payload,
        }
        self.append_record(record)


class TrajectorySessionRecorder:
    def __init__(
        self,
        *,
        log_path: str | Path,
        summary_path: str | Path,
        metadata: TrajectorySessionMetadata,
        combat_outcomes_path: str | Path | None = None,
    ) -> None:
        self.metadata = metadata
        self.logger = JsonlTrajectoryLogger(log_path)
        self.summary_path = Path(summary_path)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._combat_outcomes_logger = (
            JsonlTrajectoryLogger(combat_outcomes_path) if combat_outcomes_path is not None else None
        )
        self._stats = _SessionStats(started_at_utc=utc_now_iso())
        self._current_run: _RunSpan | None = None
        self._current_floor: _FloorSpan | None = None
        self._current_combat: _CombatSpan | None = None
        self._combat_index = 0
        self._last_contract_observation = GameRunContractObservation(checked=False, matches=True)
        self.logger.log_event(
            "session_started",
            {
                **metadata.as_dict(),
                "started_at_utc": self._stats.started_at_utc,
            },
        )

    def progress(self) -> TrajectorySessionProgress:
        return TrajectorySessionProgress(
            env_steps=self._stats.env_steps,
            total_reward=self._stats.total_reward,
            run_count=self._stats.run_count,
            completed_run_count=self._stats.completed_run_count,
            won_runs=self._stats.won_runs,
            lost_runs=self._stats.lost_runs,
            interrupted_runs=self._stats.interrupted_runs,
            runs_without_observed_seed=self._stats.runs_without_observed_seed,
            floor_count=self._stats.floor_count,
            completed_floor_count=self._stats.completed_floor_count,
            combat_count=self._stats.combat_count,
            completed_combat_count=self._stats.completed_combat_count,
            won_combats=self._stats.won_combats,
            lost_combats=self._stats.lost_combats,
            interrupted_combats=self._stats.interrupted_combats,
            last_screen=self._stats.last_screen,
            last_run_id=self._stats.last_run_id,
            last_observed_seed=self._stats.last_observed_seed,
            last_floor=self._stats.last_floor,
            current_run_id=self._current_run.run_id if self._current_run is not None else None,
            current_floor=self._current_floor.floor if self._current_floor is not None else None,
            combat_in_progress=self._current_combat is not None,
        )

    def sync_observation(
        self,
        observation: StepObservation,
        *,
        instance_id: str,
        step_index: int,
    ) -> None:
        observed_seed = observed_seed_from_state(observation.state, run_id=observation.run_id)
        observed_character_id = observed_character_id_from_state(observation.state)
        observed_ascension = observed_ascension_from_state(observation.state)

        self._stats.last_screen = observation.screen_type
        self._stats.last_run_id = observation.run_id
        self._stats.last_observed_seed = observed_seed
        self._stats.last_floor = floor_from_state(observation.state)
        self._last_contract_observation = inspect_game_run_contract(
            contract=self.metadata.game_run_contract,
            screen_type=observation.screen_type,
            run_id=observation.run_id,
            observed_seed=observed_seed,
            observed_character_id=observed_character_id,
            observed_ascension=observed_ascension,
        )
        if self._last_contract_observation.checked:
            self._stats.observation_contract_check_count += 1
            if self._last_contract_observation.matches:
                self._stats.observation_contract_match_count += 1
                self._stats.last_contract_mismatches = []
            else:
                self._stats.observation_contract_mismatch_count += 1
                self._stats.last_contract_mismatches = list(self._last_contract_observation.mismatches)
                for mismatch in self._last_contract_observation.mismatches:
                    self._stats.contract_mismatch_histogram[mismatch] += 1

        next_run_id = observation.run_id if is_real_run_id(observation.run_id) else None
        if self._current_run is not None and next_run_id != self._current_run.run_id:
            self._close_open_spans(
                instance_id=instance_id,
                step_index=step_index,
                observation=observation,
                reason="run_changed",
            )

        if next_run_id is not None and self._current_run is None and observation.screen_type != "GAME_OVER":
            self._start_run(observation=observation, instance_id=instance_id, step_index=step_index)
        elif self._current_run is not None:
            if self._current_run.observed_seed is None and observed_seed is not None:
                self._current_run.observed_seed = observed_seed
            if self._current_run.observed_character_id is None and observed_character_id is not None:
                self._current_run.observed_character_id = observed_character_id
            if self._current_run.observed_ascension is None and observed_ascension is not None:
                self._current_run.observed_ascension = observed_ascension

        floor = floor_from_state(observation.state)
        if self._current_run is not None:
            if self._current_floor is not None and floor != self._current_floor.floor:
                self._finish_floor(
                    observation=observation,
                    instance_id=instance_id,
                    step_index=step_index,
                    reason="floor_changed",
                )
            if floor is not None and floor > 0 and self._current_floor is None and observation.screen_type != "GAME_OVER":
                self._start_floor(observation=observation, instance_id=instance_id, step_index=step_index, floor=floor)

        if self._current_combat is not None and observation.screen_type != "COMBAT":
            self._finish_combat(
                observation=observation,
                instance_id=instance_id,
                step_index=step_index,
                reason="combat_exited",
            )

        if self._current_run is not None and observation.screen_type == "COMBAT" and self._current_combat is None:
            self._start_combat(observation=observation, instance_id=instance_id, step_index=step_index)

        if self._current_run is not None and observation.screen_type == "GAME_OVER":
            self._close_open_spans(
                instance_id=instance_id,
                step_index=step_index,
                observation=observation,
                reason="game_over",
            )

    def enforce_game_run_contract(self, *, instance_id: str, step_index: int) -> str | None:
        contract = self.metadata.game_run_contract
        if contract is None or not contract.strict:
            return None
        if self._last_contract_observation.matches:
            return None
        if not self._last_contract_observation.checked:
            return None
        self.logger.log_event(
            "game_run_contract_mismatch",
            {
                **self.metadata.as_dict(),
                "instance_id": instance_id,
                "step_index": step_index,
                "validation": self._last_contract_observation.as_dict(),
            },
        )
        return "game_run_contract_mismatch"

    def log_step(
        self,
        *,
        instance_id: str,
        step_index: int,
        previous_observation: StepObservation,
        result: StepResult,
        chosen_action: CandidateAction | None = None,
        policy_name: str | None = None,
        policy_pack: str | None = None,
        policy_handler: str | None = None,
        algorithm: str | None = None,
        decision_source: str | None = None,
        decision_stage: str | None = None,
        decision_reason: str | None = None,
        decision_score: float | None = None,
        planner_name: str | None = None,
        planner_strategy: str | None = None,
        ranked_actions: list[dict[str, Any]] | None = None,
        decision_metadata: dict[str, Any] | None = None,
        reward_source: str | None = None,
        model_metrics: dict[str, Any] | None = None,
    ) -> TrajectoryStepRecord:
        observation = previous_observation
        next_observation = result.observation
        ranked_action_payloads = list(ranked_actions or [])
        capability_diagnostics = self._store_capability_diagnostics(
            diagnostics_from_observation(
                observation=observation,
                step_index=step_index,
            ),
            instance_id=instance_id,
            emit_events=False,
        )
        record = TrajectoryStepRecord(
            timestamp_utc=utc_now_iso(),
            session_name=self.metadata.session_name,
            session_kind=self.metadata.session_kind,
            instance_id=instance_id,
            step_index=step_index,
            run_id=observation.run_id,
            observed_seed=observed_seed_from_state(observation.state, run_id=observation.run_id),
            screen_type=observation.screen_type,
            floor=floor_from_state(observation.state),
            legal_action_count=len(observation.legal_actions),
            legal_action_ids=[candidate.action_id for candidate in observation.legal_actions],
            build_warnings=observation.build_warnings,
            next_run_id=next_observation.run_id,
            next_observed_seed=observed_seed_from_state(next_observation.state, run_id=next_observation.run_id),
            next_screen_type=next_observation.screen_type,
            next_floor=floor_from_state(next_observation.state),
            next_legal_action_count=len(next_observation.legal_actions),
            next_legal_action_ids=[candidate.action_id for candidate in next_observation.legal_actions],
            next_build_warnings=next_observation.build_warnings,
            chosen_action_id=chosen_action.action_id if chosen_action is not None else None,
            chosen_action_label=chosen_action.label if chosen_action is not None else None,
            chosen_action_source=chosen_action.source if chosen_action is not None else None,
            chosen_action=chosen_action.model_dump(mode="json") if chosen_action is not None else None,
            policy_name=policy_name,
            policy_pack=policy_pack,
            policy_handler=policy_handler,
            algorithm=algorithm,
            decision_source=decision_source,
            decision_stage=decision_stage,
            decision_reason=decision_reason,
            decision_score=decision_score,
            planner_name=planner_name,
            planner_strategy=planner_strategy,
            ranked_action_count=len(ranked_action_payloads),
            ranked_actions=ranked_action_payloads,
            decision_metadata=decision_metadata or {},
            capability_diagnostics=capability_diagnostics,
            reward=result.reward,
            reward_source=reward_source,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
            model_metrics=model_metrics or {},
            state_summary=build_state_summary(observation),
            action_descriptors=observation.action_descriptors.model_dump(mode="json"),
            state=observation.state.model_dump(mode="json"),
            next_state_summary=build_state_summary(next_observation),
            next_action_descriptors=next_observation.action_descriptors.model_dump(mode="json"),
            next_state=next_observation.state.model_dump(mode="json"),
            response=result.response.model_dump(mode="json") if result.response is not None else None,
        )
        self.logger.append_record(record.model_dump(mode="json"))

        self._stats.env_steps += 1
        self._stats.total_reward += result.reward
        self._stats.last_screen = next_observation.screen_type
        self._stats.last_run_id = next_observation.run_id
        self._stats.last_observed_seed = observed_seed_from_state(next_observation.state, run_id=next_observation.run_id)
        self._stats.last_floor = floor_from_state(next_observation.state)
        self._stats.screen_histogram[observation.screen_type] += 1
        if chosen_action is not None:
            self._stats.action_histogram[chosen_action.action] += 1
            self._stats.action_source_histogram[chosen_action.source] += 1
        if policy_pack is not None:
            self._stats.policy_pack_histogram[policy_pack] += 1
        if policy_handler is not None:
            self._stats.policy_handler_histogram[policy_handler] += 1
        if decision_stage is not None:
            self._stats.decision_stage_histogram[decision_stage] += 1
        if decision_reason is not None:
            self._stats.decision_reason_histogram[decision_reason] += 1
        if decision_source is not None:
            self._stats.decision_source_histogram[decision_source] += 1
        if planner_name is not None:
            self._stats.planner_histogram[planner_name] += 1
            self._stats.planner_candidate_counts.append(len(ranked_action_payloads))
            if decision_score is not None:
                self._stats.planner_decision_scores.append(float(decision_score))
        predictor_payload = (decision_metadata or {}).get("predictor")
        if isinstance(predictor_payload, dict):
            predictor_mode = predictor_payload.get("mode")
            predictor_domain = predictor_payload.get("domain")
            predictor_model = predictor_payload.get("model_label") or predictor_payload.get("model_path")
            selected_predictor = predictor_payload.get("selected")
            if predictor_mode is not None:
                self._stats.predictor_mode_histogram[str(predictor_mode)] += 1
            if predictor_domain is not None:
                self._stats.predictor_domain_histogram[str(predictor_domain)] += 1
            if predictor_model is not None:
                self._stats.predictor_model_histogram[str(predictor_model)] += 1
            if isinstance(selected_predictor, dict):
                if selected_predictor.get("value_estimate") is not None:
                    self._stats.predictor_value_estimates.append(float(selected_predictor["value_estimate"]))
                if selected_predictor.get("outcome_win_probability") is not None:
                    self._stats.predictor_outcome_win_probabilities.append(
                        float(selected_predictor["outcome_win_probability"])
                    )
                if selected_predictor.get("expected_reward") is not None:
                    self._stats.predictor_expected_rewards.append(float(selected_predictor["expected_reward"]))
                if selected_predictor.get("expected_damage_delta") is not None:
                    self._stats.predictor_expected_damage_deltas.append(
                        float(selected_predictor["expected_damage_delta"])
                    )
        route_planner_payload = (decision_metadata or {}).get("route_planner")
        if isinstance(route_planner_payload, dict):
            self._stats.route_planner_steps += 1
            boss_id = route_planner_payload.get("boss_encounter_id")
            if boss_id is not None:
                self._stats.route_planner_boss_histogram[str(boss_id)] += 1
            selected_route = route_planner_payload.get("selected")
            if isinstance(selected_route, dict):
                path = selected_route.get("path")
                if isinstance(path, list):
                    self._stats.route_planner_path_lengths.append(len(path))
                if selected_route.get("score") is not None:
                    self._stats.route_planner_selected_scores.append(float(selected_route["score"]))
                reason_tags = selected_route.get("reason_tags")
                if isinstance(reason_tags, list):
                    for tag in reason_tags:
                        self._stats.route_planner_reason_tag_histogram[str(tag)] += 1
        if self._current_combat is not None:
            self._current_combat.cumulative_reward += result.reward

        return record

    def record_capability_diagnostics(
        self,
        *,
        instance_id: str,
        step_index: int,
        observation: StepObservation | None,
        decision_reason: str | None = None,
        stop_reason: str | None = None,
        decision_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        diagnostics = [
            *diagnostics_from_observation(observation=observation, step_index=step_index),
            *diagnostics_from_no_action(
                observation=observation,
                step_index=step_index,
                decision_reason=decision_reason,
                stop_reason=stop_reason,
                decision_metadata=decision_metadata,
            ),
        ]
        return self._store_capability_diagnostics(
            diagnostics,
            instance_id=instance_id,
            emit_events=True,
        )

    def finalize(
        self,
        *,
        instance_id: str,
        stop_reason: str,
        step_index: int,
        error: str | None = None,
        final_observation: StepObservation | None = None,
    ) -> TrajectorySessionSummary:
        if final_observation is not None:
            self.sync_observation(final_observation, instance_id=instance_id, step_index=step_index)
        self.record_capability_diagnostics(
            instance_id=instance_id,
            step_index=step_index,
            observation=final_observation,
            stop_reason=stop_reason,
        )
        if self._current_combat is not None:
            self._finish_combat(
                observation=final_observation,
                instance_id=instance_id,
                step_index=step_index,
                reason="session_stopped",
            )
        if self._current_floor is not None:
            self._finish_floor(
                observation=final_observation,
                instance_id=instance_id,
                step_index=step_index,
                reason="session_stopped",
            )
        if self._current_run is not None:
            self._finish_run(
                observation=final_observation,
                instance_id=instance_id,
                step_index=step_index,
                reason="session_stopped",
            )

        ended_at = utc_now_iso()
        summary = TrajectorySessionSummary(
            schema_version=TRAJECTORY_SCHEMA_VERSION,
            session_name=self.metadata.session_name,
            session_kind=self.metadata.session_kind,
            base_url=self.metadata.base_url,
            policy_name=self.metadata.policy_name,
            algorithm=self.metadata.algorithm,
            config=self.metadata.config,
            game_run_contract=(
                self.metadata.game_run_contract.as_dict() if self.metadata.game_run_contract is not None else None
            ),
            game_run_contract_validation=build_game_run_contract_validation_payload(
                contract=self.metadata.game_run_contract,
                observation_check_count=self._stats.observation_contract_check_count,
                observation_match_count=self._stats.observation_contract_match_count,
                observation_mismatch_count=self._stats.observation_contract_mismatch_count,
                mismatch_histogram=dict(self._stats.contract_mismatch_histogram),
                last_mismatches=list(self._stats.last_contract_mismatches),
                observed_seed_histogram=dict(self._stats.observed_run_seed_histogram),
                observed_character_histogram=dict(self._stats.observed_character_histogram),
                observed_ascension_histogram=dict(self._stats.observed_ascension_histogram),
            ),
            started_at_utc=self._stats.started_at_utc,
            ended_at_utc=ended_at,
            stop_reason=stop_reason,
            error=error,
            env_steps=self._stats.env_steps,
            total_reward=self._stats.total_reward,
            run_count=self._stats.run_count,
            completed_run_count=self._stats.completed_run_count,
            won_runs=self._stats.won_runs,
            lost_runs=self._stats.lost_runs,
            interrupted_runs=self._stats.interrupted_runs,
            observed_run_seeds=sorted(self._stats.observed_run_seed_histogram),
            observed_run_seed_histogram=dict(self._stats.observed_run_seed_histogram),
            runs_without_observed_seed=self._stats.runs_without_observed_seed,
            floor_count=self._stats.floor_count,
            completed_floor_count=self._stats.completed_floor_count,
            combat_count=self._stats.combat_count,
            completed_combat_count=self._stats.completed_combat_count,
            won_combats=self._stats.won_combats,
            lost_combats=self._stats.lost_combats,
            interrupted_combats=self._stats.interrupted_combats,
            run_outcome_histogram=dict(self._stats.run_outcome_histogram),
            run_finish_reason_histogram=dict(self._stats.run_finish_reason_histogram),
            screen_histogram=dict(self._stats.screen_histogram),
            action_histogram=dict(self._stats.action_histogram),
            policy_pack_histogram=dict(self._stats.policy_pack_histogram),
            policy_handler_histogram=dict(self._stats.policy_handler_histogram),
            decision_stage_histogram=dict(self._stats.decision_stage_histogram),
            decision_reason_histogram=dict(self._stats.decision_reason_histogram),
            decision_source_histogram=dict(self._stats.decision_source_histogram),
            action_source_histogram=dict(self._stats.action_source_histogram),
            planner_histogram=dict(self._stats.planner_histogram),
            planner_candidate_count_stats=_numeric_stats(self._stats.planner_candidate_counts),
            planner_decision_score_stats=_numeric_stats(self._stats.planner_decision_scores),
            route_planner_step_count=self._stats.route_planner_steps,
            route_planner_boss_histogram=dict(self._stats.route_planner_boss_histogram),
            route_planner_reason_tag_histogram=dict(self._stats.route_planner_reason_tag_histogram),
            route_planner_path_length_stats=_numeric_stats(self._stats.route_planner_path_lengths),
            route_planner_selected_score_stats=_numeric_stats(self._stats.route_planner_selected_scores),
            predictor_mode_histogram=dict(self._stats.predictor_mode_histogram),
            predictor_domain_histogram=dict(self._stats.predictor_domain_histogram),
            predictor_model_histogram=dict(self._stats.predictor_model_histogram),
            predictor_value_estimate_stats=_numeric_stats(self._stats.predictor_value_estimates),
            predictor_outcome_win_probability_stats=_numeric_stats(self._stats.predictor_outcome_win_probabilities),
            predictor_expected_reward_stats=_numeric_stats(self._stats.predictor_expected_rewards),
            predictor_expected_damage_delta_stats=_numeric_stats(self._stats.predictor_expected_damage_deltas),
            non_combat_capability=(
                summarize_capability_diagnostics(self._stats.capability_diagnostics)
                if self._stats.capability_diagnostics
                else empty_capability_summary()
            ),
            last_screen=self._stats.last_screen,
            last_run_id=self._stats.last_run_id,
            last_observed_seed=self._stats.last_observed_seed,
            last_floor=self._stats.last_floor,
        )
        payload = summary.as_dict()
        self.summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.log_event(
            "session_finished",
            {
                **self.metadata.as_dict(),
                "ended_at_utc": ended_at,
                "stop_reason": stop_reason,
                "error": error,
                "summary_path": str(self.summary_path),
                "summary": payload,
            },
        )
        return summary

    def _store_capability_diagnostics(
        self,
        diagnostics,
        *,
        instance_id: str,
        emit_events: bool,
    ) -> list[dict[str, Any]]:
        stored: list[dict[str, Any]] = []
        for item in diagnostics:
            payload = item.as_dict()
            key = self._capability_diagnostic_key(payload)
            if key in self._stats.capability_diagnostic_keys:
                continue
            self._stats.capability_diagnostic_keys.add(key)
            self._stats.capability_diagnostics.append(payload)
            stored.append(payload)
            if emit_events:
                self.logger.log_event(
                    "capability_diagnostic",
                    {
                        **self.metadata.as_dict(),
                        "instance_id": instance_id,
                        "step_index": payload.get("step_index"),
                        "diagnostic": payload,
                    },
                )
        return stored

    @staticmethod
    def _capability_diagnostic_key(payload: dict[str, Any]) -> str:
        return "|".join(
            [
                str(payload.get("step_index")),
                str(payload.get("bucket")),
                str(payload.get("category")),
                str(payload.get("screen_type")),
                str(payload.get("descriptor")),
                str(payload.get("decision_reason")),
                str(payload.get("stop_reason")),
            ]
        )

    def _start_run(self, *, observation: StepObservation, instance_id: str, step_index: int) -> None:
        summary = build_state_summary(observation)
        observed_seed = observed_seed_from_state(observation.state, run_id=observation.run_id)
        observed_character_id = observed_character_id_from_state(observation.state)
        observed_ascension = observed_ascension_from_state(observation.state)
        self._current_run = _RunSpan(
            run_id=observation.run_id,
            started_step_index=step_index,
            start_summary=summary,
            observed_seed=observed_seed,
            observed_character_id=observed_character_id,
            observed_ascension=observed_ascension,
        )
        self._stats.run_count += 1
        self.logger.log_event(
            "run_started",
            {
                **self.metadata.as_dict(),
                "instance_id": instance_id,
                "step_index": step_index,
                "run_id": observation.run_id,
                "observed_seed": observed_seed,
                "observed_character_id": observed_character_id,
                "observed_ascension": observed_ascension,
                "state_summary": summary,
            },
        )

    def _finish_run(
        self,
        *,
        observation: StepObservation | None,
        instance_id: str,
        step_index: int,
        reason: str,
    ) -> None:
        if self._current_run is None:
            return
        run_id = self._current_run.run_id
        summary = build_state_summary(observation)
        observed_seed = self._current_run.observed_seed or observed_seed_from_state(
            observation.state if observation is not None else None,
            run_id=observation.run_id if observation is not None else None,
        )
        observed_character_id = self._current_run.observed_character_id or observed_character_id_from_state(
            observation.state if observation is not None else None
        )
        observed_ascension = self._current_run.observed_ascension
        if observed_ascension is None:
            observed_ascension = observed_ascension_from_state(observation.state if observation is not None else None)
        victory = bool(observation and observation.state.game_over and observation.state.game_over.is_victory)
        outcome = run_outcome(observation, reason)
        self._stats.completed_run_count += 1
        self._stats.run_outcome_histogram[outcome] += 1
        self._stats.run_finish_reason_histogram[reason] += 1
        if observed_seed is None:
            self._stats.runs_without_observed_seed += 1
        else:
            self._stats.observed_run_seed_histogram[observed_seed] += 1
        if observed_character_id is not None:
            self._stats.observed_character_histogram[observed_character_id] += 1
        if observed_ascension is not None:
            self._stats.observed_ascension_histogram[observed_ascension] += 1
        if outcome == "won":
            self._stats.won_runs += 1
        elif outcome == "lost":
            self._stats.lost_runs += 1
        else:
            self._stats.interrupted_runs += 1
        self.logger.log_event(
            "run_finished",
            {
                **self.metadata.as_dict(),
                "instance_id": instance_id,
                "run_id": run_id,
                "observed_seed": observed_seed,
                "observed_character_id": observed_character_id,
                "observed_ascension": observed_ascension,
                "started_step_index": self._current_run.started_step_index,
                "finished_step_index": step_index,
                "reason": reason,
                "victory": victory,
                "outcome": outcome,
                "state_summary": summary,
            },
        )
        self._current_run = None

    def _start_floor(
        self,
        *,
        observation: StepObservation,
        instance_id: str,
        step_index: int,
        floor: int,
    ) -> None:
        summary = build_state_summary(observation)
        self._current_floor = _FloorSpan(
            run_id=observation.run_id,
            floor=floor,
            started_step_index=step_index,
            start_summary=summary,
        )
        self._stats.floor_count += 1
        self.logger.log_event(
            "floor_started",
            {
                **self.metadata.as_dict(),
                "instance_id": instance_id,
                "step_index": step_index,
                "run_id": observation.run_id,
                "floor": floor,
                "state_summary": summary,
            },
        )

    def _finish_floor(
        self,
        *,
        observation: StepObservation | None,
        instance_id: str,
        step_index: int,
        reason: str,
    ) -> None:
        if self._current_floor is None:
            return
        floor = self._current_floor.floor
        run_id = self._current_floor.run_id
        summary = build_state_summary(observation)
        self._stats.completed_floor_count += 1
        self.logger.log_event(
            "floor_finished",
            {
                **self.metadata.as_dict(),
                "instance_id": instance_id,
                "run_id": run_id,
                "floor": floor,
                "started_step_index": self._current_floor.started_step_index,
                "finished_step_index": step_index,
                "reason": reason,
                "state_summary": summary,
            },
        )
        self._current_floor = None

    def _start_combat(self, *, observation: StepObservation, instance_id: str, step_index: int) -> None:
        summary = build_state_summary(observation)
        self._combat_index += 1
        self._current_combat = _CombatSpan(
            run_id=observation.run_id,
            floor=floor_from_state(observation.state),
            combat_index=self._combat_index,
            started_step_index=step_index,
            start_summary=summary,
            enemy_ids=enemy_ids_from_state(observation.state),
            start_player_hp=current_player_hp(observation.state),
            start_enemy_hp=current_enemy_hp(observation.state),
        )
        self._stats.combat_count += 1
        self.logger.log_event(
            "combat_started",
            {
                **self.metadata.as_dict(),
                "instance_id": instance_id,
                "step_index": step_index,
                "run_id": observation.run_id,
                "floor": self._current_combat.floor,
                "combat_index": self._combat_index,
                "enemy_ids": self._current_combat.enemy_ids,
                "state_summary": summary,
            },
        )

    def _finish_combat(
        self,
        *,
        observation: StepObservation | None,
        instance_id: str,
        step_index: int,
        reason: str,
    ) -> None:
        if self._current_combat is None:
            return

        combat = self._current_combat
        end_summary = build_state_summary(observation)
        end_player_hp = current_player_hp(observation.state) if observation is not None else combat.start_player_hp
        end_enemy_hp = current_enemy_hp(observation.state) if observation is not None else combat.start_enemy_hp
        outcome = combat_outcome(observation, reason)
        record = CombatOutcomeRecord(
            timestamp_utc=utc_now_iso(),
            session_name=self.metadata.session_name,
            session_kind=self.metadata.session_kind,
            instance_id=instance_id,
            run_id=combat.run_id,
            observed_seed=self._current_run.observed_seed
            if self._current_run is not None
            else observed_seed_from_state(
                observation.state if observation is not None else None,
                run_id=observation.run_id if observation is not None else None,
            ),
            floor=combat.floor,
            combat_index=combat.combat_index,
            started_step_index=combat.started_step_index,
            finished_step_index=step_index,
            outcome=outcome,
            cumulative_reward=combat.cumulative_reward,
            step_count=max(0, step_index - combat.started_step_index),
            enemy_ids=combat.enemy_ids,
            damage_dealt=max(0, combat.start_enemy_hp - end_enemy_hp),
            damage_taken=max(0, combat.start_player_hp - end_player_hp),
            start_summary=combat.start_summary,
            end_summary=end_summary,
        )
        payload = record.model_dump(mode="json")
        payload["reason"] = reason
        self.logger.append_record(payload)
        if self._combat_outcomes_logger is not None:
            self._combat_outcomes_logger.append_record(payload)

        self._stats.completed_combat_count += 1
        if outcome == "won":
            self._stats.won_combats += 1
        elif outcome == "lost":
            self._stats.lost_combats += 1
        else:
            self._stats.interrupted_combats += 1
        self._current_combat = None

    def _close_open_spans(
        self,
        *,
        instance_id: str,
        step_index: int,
        observation: StepObservation | None,
        reason: str,
    ) -> None:
        if self._current_combat is not None:
            self._finish_combat(
                observation=observation,
                instance_id=instance_id,
                step_index=step_index,
                reason=reason,
            )
        if self._current_floor is not None:
            self._finish_floor(
                observation=observation,
                instance_id=instance_id,
                step_index=step_index,
                reason=reason,
            )
        if self._current_run is not None:
            self._finish_run(
                observation=observation,
                instance_id=instance_id,
                step_index=step_index,
                reason=reason,
            )


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def build_state_summary(observation: StepObservation | None) -> dict[str, Any]:
    if observation is None:
        return {}

    state = observation.state
    summary: dict[str, Any] = {
        "screen_type": observation.screen_type,
        "run_id": observation.run_id,
        "state_version": state.state_version,
        "turn": state.turn,
        "in_combat": state.in_combat,
        "available_action_count": len(observation.legal_actions),
        "build_warning_count": len(observation.build_warnings),
        "session_phase": state.session.phase,
        "control_scope": state.session.control_scope,
        "observed_seed": observed_seed_from_state(state, run_id=observation.run_id),
    }

    if state.run is not None:
        deck_summary = _deck_summary_from_agent_view(state.agent_view)
        boss_encounter = _encounter_summary_from_payload(state.run.boss_encounter)
        second_boss_encounter = _encounter_summary_from_payload(state.run.second_boss_encounter)
        summary["run"] = {
            "character_id": state.run.character_id,
            "character_name": state.run.character_name,
            "seed": _coerce_seed(state.run.seed) or _seed_from_agent_view(state.agent_view) or (observation.run_id if is_real_run_id(observation.run_id) else None),
            "ascension": state.run.ascension,
            "floor": state.run.floor,
            "act_index": state.run.act_index,
            "act_number": state.run.act_number,
            "act_id": state.run.act_id,
            "act_name": state.run.act_name,
            "has_second_boss": state.run.has_second_boss,
            "boss_encounter": boss_encounter,
            "boss_encounter_id": boss_encounter["encounter_id"] if boss_encounter is not None else None,
            "boss_encounter_name": boss_encounter["name"] if boss_encounter is not None else None,
            "boss_encounter_room_type": boss_encounter["room_type"] if boss_encounter is not None else None,
            "second_boss_encounter": second_boss_encounter,
            "second_boss_encounter_id": (
                second_boss_encounter["encounter_id"] if second_boss_encounter is not None else None
            ),
            "second_boss_encounter_name": second_boss_encounter["name"] if second_boss_encounter is not None else None,
            "second_boss_encounter_room_type": (
                second_boss_encounter["room_type"] if second_boss_encounter is not None else None
            ),
            "current_hp": state.run.current_hp,
            "max_hp": state.run.max_hp,
            "gold": state.run.gold,
            "max_energy": state.run.max_energy,
            "occupied_potions": len([potion for potion in state.run.potions if potion.occupied]),
            **deck_summary,
        }

    if state.character_select is not None:
        summary["character_select"] = {
            "selected_character_id": state.character_select.selected_character_id,
            "ascension": state.character_select.ascension,
            "seed": _coerce_seed(state.character_select.seed),
            "can_embark": state.character_select.can_embark,
            "local_ready": state.character_select.local_ready,
            "character_ids": [character.character_id for character in state.character_select.characters],
        }

    if state.custom_run is not None:
        summary["custom_run"] = {
            "selected_character_id": state.custom_run.selected_character_id,
            "ascension": state.custom_run.ascension,
            "seed": _coerce_seed(state.custom_run.seed),
            "can_embark": state.custom_run.can_embark,
            "local_ready": state.custom_run.local_ready,
            "character_ids": [character.character_id for character in state.custom_run.characters],
            "modifier_ids": list(state.custom_run.modifier_ids),
            "modifier_names": [modifier.name for modifier in state.custom_run.modifiers if modifier.is_selected],
        }

    if state.combat is not None:
        summary["combat"] = {
            "player_hp": state.combat.player.current_hp,
            "player_block": state.combat.player.block,
            "energy": state.combat.player.energy,
            "stars": state.combat.player.stars,
            "focus": state.combat.player.focus,
            "enemy_ids": enemy_ids_from_state(state),
            "enemy_hp": [enemy.current_hp for enemy in state.combat.enemies if enemy.is_alive],
            "hand_card_ids": [card.card_id for card in state.combat.hand],
            "playable_hand_count": len([card for card in state.combat.hand if card.playable]),
        }

    if state.map is not None:
        summary["map"] = {
            "available_node_types": [node.node_type for node in state.map.available_nodes],
            "available_node_count": len(state.map.available_nodes),
            "travel_enabled": state.map.is_travel_enabled,
            "traveling": state.map.is_traveling,
            **_build_map_graph_summary(state),
        }

    if state.reward is not None:
        summary["reward"] = {
            "pending_card_choice": state.reward.pending_card_choice,
            "source_type": state.reward.source_type,
            "source_room_type": state.reward.source_room_type,
            "source_action": state.reward.source_action,
            "source_event_id": state.reward.source_event_id,
            "source_event_option_index": state.reward.source_event_option_index,
            "source_event_option_text_key": state.reward.source_event_option_text_key,
            "source_event_option_title": state.reward.source_event_option_title,
            "source_rest_option_id": state.reward.source_rest_option_id,
            "source_rest_option_index": state.reward.source_rest_option_index,
            "source_rest_option_title": state.reward.source_rest_option_title,
            "reward_types": [reward.reward_type for reward in state.reward.rewards],
            "card_option_ids": [card.card_id for card in state.reward.card_options],
            "reward_count": len(state.reward.rewards),
            "card_option_count": len(state.reward.card_options),
        }

    if state.selection is not None:
        summary["selection"] = {
            "kind": state.selection.kind,
            "selection_family": state.selection.selection_family,
            "semantic_mode": state.selection.semantic_mode,
            "source_type": state.selection.source_type,
            "source_room_type": state.selection.source_room_type,
            "source_action": state.selection.source_action,
            "source_event_id": state.selection.source_event_id,
            "source_event_option_index": state.selection.source_event_option_index,
            "source_event_option_text_key": state.selection.source_event_option_text_key,
            "source_event_option_title": state.selection.source_event_option_title,
            "source_rest_option_id": state.selection.source_rest_option_id,
            "source_rest_option_index": state.selection.source_rest_option_index,
            "source_rest_option_title": state.selection.source_rest_option_title,
            "prompt": state.selection.prompt,
            "prompt_loc_table": state.selection.prompt_loc_table,
            "prompt_loc_key": state.selection.prompt_loc_key,
            "min_select": state.selection.min_select,
            "max_select": state.selection.max_select,
            "required_count": state.selection.required_count,
            "remaining_count": state.selection.remaining_count,
            "selected_count": state.selection.selected_count,
            "requires_confirmation": state.selection.requires_confirmation,
            "can_confirm": state.selection.can_confirm,
            "supports_multi_select": state.selection.supports_multi_select,
            "card_ids": [card.card_id for card in state.selection.cards],
            "card_count": len(state.selection.cards),
        }

    if state.build is not None:
        summary["build"] = {
            "build_id": state.build.build_id,
            "game_version": state.build.game_version,
            "branch": state.build.branch,
            "content_channel": state.build.content_channel,
            "commit": state.build.commit,
            "build_date": state.build.build_date,
            "main_assembly_hash": state.build.main_assembly_hash,
        }

    if state.shop is not None:
        summary["shop"] = {
            "is_open": state.shop.is_open,
            "card_count": len([card for card in state.shop.cards if card.is_stocked]),
            "relic_count": len([relic for relic in state.shop.relics if relic.is_stocked]),
            "potion_count": len([potion for potion in state.shop.potions if potion.is_stocked]),
            "card_removal_available": bool(state.shop.card_removal and state.shop.card_removal.available),
            "card_removal_price": (
                state.shop.card_removal.price if state.shop.card_removal is not None else None
            ),
        }

    if state.event is not None:
        summary["event"] = {
            "event_id": state.event.event_id,
            "option_titles": [option.title for option in state.event.options],
            "option_count": len(state.event.options),
        }

    if state.rest is not None:
        summary["rest"] = {
            "option_ids": [option.option_id for option in state.rest.options],
            "option_count": len(state.rest.options),
        }

    if state.game_over is not None:
        summary["game_over"] = {
            "is_victory": state.game_over.is_victory,
            "floor": state.game_over.floor,
            "character_id": state.game_over.character_id,
        }

    return summary


def floor_from_state(state: GameStatePayload) -> int | None:
    if state.run is None:
        return None
    return state.run.floor


def observed_seed_from_state(state: GameStatePayload | None, *, run_id: str | None = None) -> str | None:
    if state is None:
        return None

    for candidate in (
        state.run.seed if state.run is not None else None,
        state.character_select.seed if state.character_select is not None else None,
        state.custom_run.seed if state.custom_run is not None else None,
        _seed_from_agent_view(state.agent_view),
        run_id if is_real_run_id(run_id) else None,
    ):
        normalized = _coerce_seed(candidate)
        if normalized is not None:
            return normalized
    return None


def observed_character_id_from_state(state: GameStatePayload | None) -> str | None:
    if state is None:
        return None
    for candidate in (
        state.run.character_id if state.run is not None else None,
        state.character_select.selected_character_id if state.character_select is not None else None,
        state.custom_run.selected_character_id if state.custom_run is not None else None,
    ):
        normalized = _coerce_seed(candidate)
        if normalized is not None:
            return normalized
    return None


def observed_ascension_from_state(state: GameStatePayload | None) -> int | None:
    if state is None:
        return None
    for candidate in (
        state.run.ascension if state.run is not None else None,
        state.character_select.ascension if state.character_select is not None else None,
        state.custom_run.ascension if state.custom_run is not None else None,
    ):
        if candidate is not None:
            return int(candidate)
    return None


def is_real_run_id(run_id: str | None) -> bool:
    return bool(run_id and run_id != "run_unknown")


def _encounter_summary_from_payload(encounter: Any) -> dict[str, Any] | None:
    if encounter is None:
        return None
    return {
        "encounter_id": getattr(encounter, "encounter_id", None),
        "name": getattr(encounter, "name", None),
        "room_type": getattr(encounter, "room_type", None),
    }


def _build_map_graph_summary(state: GameStatePayload) -> dict[str, Any]:
    map_payload = state.map
    if map_payload is None:
        return {}

    node_type_counts: Counter[str] = Counter()
    graph_edge_count = 0
    visited_node_count = 0
    available_graph_node_count = 0
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for node in map_payload.nodes:
        node_type_counts[str(node.node_type)] += 1
        graph_edge_count += len(node.children)
        if node.visited:
            visited_node_count += 1
        if node.is_available:
            available_graph_node_count += 1
        adjacency[(node.row, node.col)] = [(child.row, child.col) for child in node.children]

    current_key = _map_coord_key(map_payload.current_node)
    boss_key = _map_coord_key(map_payload.boss_node)
    second_boss_key = _map_coord_key(map_payload.second_boss_node)
    frontier_keys = [(node.row, node.col) for node in map_payload.available_nodes]

    return {
        "current_node": _map_coord_dict(map_payload.current_node),
        "starting_node": _map_coord_dict(map_payload.starting_node),
        "boss_node": _map_coord_dict(map_payload.boss_node),
        "second_boss_node": _map_coord_dict(map_payload.second_boss_node),
        "map_generation_count": map_payload.map_generation_count,
        "rows": map_payload.rows,
        "cols": map_payload.cols,
        "graph_node_count": len(map_payload.nodes),
        "graph_edge_count": graph_edge_count,
        "visited_node_count": visited_node_count,
        "available_graph_node_count": available_graph_node_count,
        "node_type_counts": dict(node_type_counts),
        "current_to_boss_distance": _shortest_graph_distance(adjacency, current_key, boss_key),
        "current_to_second_boss_distance": _shortest_graph_distance(adjacency, current_key, second_boss_key),
        "frontier_to_boss_min_distance": _min_shortest_graph_distance(adjacency, frontier_keys, boss_key),
        "frontier_to_second_boss_min_distance": _min_shortest_graph_distance(adjacency, frontier_keys, second_boss_key),
    }


def _map_coord_key(coord: Any) -> tuple[int, int] | None:
    if coord is None:
        return None
    return (int(coord.row), int(coord.col))


def _map_coord_dict(coord: Any) -> dict[str, int] | None:
    key = _map_coord_key(coord)
    if key is None:
        return None
    return {"row": key[0], "col": key[1]}


def _shortest_graph_distance(
    adjacency: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int] | None,
    goal: tuple[int, int] | None,
) -> int | None:
    if start is None or goal is None:
        return None
    if start == goal:
        return 0

    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    visited = {start}

    while queue:
        node, distance = queue.popleft()
        for child in adjacency.get(node, []):
            if child in visited:
                continue
            if child == goal:
                return distance + 1
            visited.add(child)
            queue.append((child, distance + 1))
    return None


def _min_shortest_graph_distance(
    adjacency: dict[tuple[int, int], list[tuple[int, int]]],
    starts: list[tuple[int, int]],
    goal: tuple[int, int] | None,
) -> int | None:
    distances = [
        distance
        for distance in (_shortest_graph_distance(adjacency, start, goal) for start in starts)
        if distance is not None
    ]
    return min(distances) if distances else None


def enemy_ids_from_state(state: GameStatePayload) -> list[str]:
    if state.combat is None:
        return []
    return [enemy.enemy_id for enemy in state.combat.enemies if enemy.is_alive]


def current_player_hp(state: GameStatePayload) -> int:
    if state.combat is not None:
        return state.combat.player.current_hp
    if state.run is not None:
        return state.run.current_hp
    return 0


def current_enemy_hp(state: GameStatePayload) -> int:
    if state.combat is None:
        return 0
    return sum(enemy.current_hp for enemy in state.combat.enemies if enemy.is_alive)


def combat_outcome(observation: StepObservation | None, reason: str) -> str:
    if observation is None:
        return "interrupted"
    if observation.screen_type == "GAME_OVER" and observation.state.game_over is not None:
        return "won" if observation.state.game_over.is_victory else "lost"
    if observation.screen_type != "COMBAT":
        return "won"
    if reason == "session_stopped":
        return "interrupted"
    return "interrupted"


def run_outcome(observation: StepObservation | None, reason: str) -> str:
    if observation is None:
        return "interrupted"
    if observation.screen_type == "GAME_OVER" and observation.state.game_over is not None:
        return "won" if observation.state.game_over.is_victory else "lost"
    if reason == "session_stopped":
        return "interrupted"
    return "interrupted"


def _numeric_stats(values: list[int] | list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    numeric_values = [float(value) for value in values]
    return {
        "count": len(numeric_values),
        "min": min(numeric_values),
        "mean": sum(numeric_values) / len(numeric_values),
        "max": max(numeric_values),
    }


def _deck_summary_from_agent_view(agent_view: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(agent_view, dict):
        return {}
    run_view = agent_view.get("run")
    if not isinstance(run_view, dict):
        return {}

    raw_deck = run_view.get("deck")
    deck_lines: list[str] = []
    if isinstance(raw_deck, list):
        for item in raw_deck:
            if isinstance(item, dict):
                line = item.get("line")
                if line:
                    deck_lines.append(str(line))
            elif item:
                deck_lines.append(str(item))

    titles = [_card_title_from_line(line) for line in deck_lines]
    relics = run_view.get("relics")
    relic_count = len(relics) if isinstance(relics, list) else 0
    return {
        "deck_size": len(deck_lines),
        "strike_count": sum(1 for title in titles if title.startswith("strike") or "打击" in title),
        "defend_count": sum(1 for title in titles if title.startswith("defend") or "防御" in title),
        "curse_count": sum(1 for title in titles if "curse" in title or "诅咒" in title),
        "status_count": sum(
            1
            for title in titles
            if any(fragment in title for fragment in ("wound", "burn", "dazed", "slime", "伤口", "灼伤", "眩晕", "黏液"))
        ),
        "relic_count": relic_count,
    }


def _seed_from_agent_view(agent_view: dict[str, Any] | None) -> str | None:
    if not isinstance(agent_view, dict):
        return None

    run_view = agent_view.get("run")
    if isinstance(run_view, dict):
        for key in ("seed", "string_seed"):
            normalized = _coerce_seed(run_view.get(key))
            if normalized is not None:
                return normalized

    character_select_view = agent_view.get("character_select")
    if isinstance(character_select_view, dict):
        normalized = _coerce_seed(character_select_view.get("seed"))
        if normalized is not None:
            return normalized

    custom_run_view = agent_view.get("custom_run")
    if isinstance(custom_run_view, dict):
        normalized = _coerce_seed(custom_run_view.get("seed"))
        if normalized is not None:
            return normalized

    normalized = _coerce_seed(agent_view.get("seed"))
    if normalized is not None:
        return normalized
    return None


def _coerce_seed(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _card_title_from_line(line: str) -> str:
    normalized = line.strip().lower()
    for separator in (" [", "[", "：", ":"):
        if separator in normalized:
            return normalized.split(separator, 1)[0].strip()
    return normalized
