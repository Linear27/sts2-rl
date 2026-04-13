from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sts2_rl.data import build_state_summary
from sts2_rl.env.types import StepObservation


def fingerprint_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_action_space_snapshot(observation: StepObservation) -> dict[str, Any]:
    return {
        "legal_action_count": len(observation.legal_actions),
        "legal_action_ids": [candidate.action_id for candidate in observation.legal_actions],
        "build_warnings": list(observation.build_warnings),
    }


def build_start_state_snapshot(observation: StepObservation) -> dict[str, Any]:
    state = observation.state
    return {
        "screen_type": observation.screen_type,
        "run_id": observation.run_id,
        "session_phase": state.session.phase,
        "control_scope": state.session.control_scope,
        "state_summary": build_state_summary(observation),
        "character_select": (
            {
                "selected_character_id": state.character_select.selected_character_id,
                "ascension": state.character_select.ascension,
                "can_embark": state.character_select.can_embark,
            }
            if state.character_select is not None
            else None
        ),
        "custom_run": (
            {
                "selected_character_id": state.custom_run.selected_character_id,
                "ascension": state.custom_run.ascension,
                "seed": state.custom_run.seed,
                "can_embark": state.custom_run.can_embark,
                "modifier_ids": list(state.custom_run.modifier_ids),
            }
            if state.custom_run is not None
            else None
        ),
        "game_over": (
            {
                "can_continue": state.game_over.can_continue,
                "can_return_to_main_menu": state.game_over.can_return_to_main_menu,
                "showing_summary": state.game_over.showing_summary,
                "is_victory": state.game_over.is_victory,
            }
            if state.game_over is not None
            else None
        ),
    }


def build_start_payload(observation: StepObservation) -> dict[str, Any]:
    state_snapshot = build_start_state_snapshot(observation)
    action_space_snapshot = build_action_space_snapshot(observation)
    comparison_payload = {
        "state_snapshot": state_snapshot,
        "action_space_snapshot": action_space_snapshot,
    }
    return {
        "start_signature": fingerprint_payload(comparison_payload),
        "state_fingerprint": fingerprint_payload(state_snapshot),
        "action_space_fingerprint": fingerprint_payload(action_space_snapshot),
        "comparison_payload": comparison_payload,
        "state_snapshot": state_snapshot,
        "action_space_snapshot": action_space_snapshot,
        "state_summary": state_snapshot["state_summary"],
    }


def build_runtime_metadata_snapshot(
    *,
    base_url: str,
    prepare_target: str,
    summary_payload: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    checkpoint_label: str | None = None,
    policy_profile: str | None = None,
) -> dict[str, Any]:
    config_payload = summary_payload.get("config")
    if not isinstance(config_payload, dict):
        config_payload = {}
    return {
        "base_url": base_url,
        "prepare_target": prepare_target,
        "session_kind": summary_payload.get("session_kind"),
        "policy_name": summary_payload.get("policy_name"),
        "algorithm": summary_payload.get("algorithm"),
        "policy_profile": policy_profile if policy_profile is not None else summary_payload.get("policy_profile"),
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else summary_payload.get("checkpoint_path"),
        "checkpoint_metadata": summary_payload.get("checkpoint_metadata"),
        "predictor": config_payload.get("predictor"),
        "request_timeout_seconds": config_payload.get("request_timeout_seconds"),
        "max_env_steps": config_payload.get("max_env_steps"),
        "max_runs": config_payload.get("max_runs"),
        "max_combats": config_payload.get("max_combats"),
        "poll_interval_seconds": config_payload.get("poll_interval_seconds"),
        "max_idle_polls": config_payload.get("max_idle_polls"),
    }
