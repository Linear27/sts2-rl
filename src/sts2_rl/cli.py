from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .capability import load_capability_summary
from .collect import CommunityPriorRuntimeConfig, StrategicRuntimeConfig
from .benchmark import benchmark_health, benchmark_observe
from .collect import collect_round_robin
from .collect.runner import default_collection_session_name
from .collect.strategic_runtime import normalize_strategic_hooks, normalize_strategic_mode
from .data import (
    build_dataset_from_manifest,
    build_game_run_contract,
    import_community_card_stats,
    import_spiremeta_community_card_stats,
    load_dataset_summary,
    load_community_card_stats_summary,
    load_public_run_normalized_summary,
    load_public_run_archive_summary,
    normalize_public_run_archive,
    sync_sts2runs_public_run_archive,
    validate_dataset_manifest,
)
from .predict import (
    PredictorBenchmarkComparisonThresholds,
    PredictorCalibrationThresholds,
    PredictorRankingThresholds,
    build_predictor_benchmark_comparison_report,
    build_predictor_calibration_report,
    build_predictor_ranking_report,
    CombatOutcomePredictorTrainConfig,
    PredictorRuntimeConfig,
    default_predictor_training_session_name,
    default_predictor_report_session_name,
    extract_predictor_dataset,
    normalize_predictor_hooks,
    normalize_predictor_mode,
    train_combat_outcome_predictor,
)
from .registry import (
    build_registry_leaderboard,
    compare_registry_experiments,
    get_registry_experiment,
    initialize_registry,
    list_registry_experiments,
    load_registry_aliases,
    register_experiment,
    set_registry_alias,
)
from .runtime.job_manifest import load_runtime_job_manifest, load_runtime_job_summary
from .runtime.job_runner import run_runtime_job
from .runtime import (
    build_instance_specs,
    build_windows_launch_plans,
    bootstrap_instance_user_data,
    collect_instance_statuses,
    load_experiment_dag_manifest,
    load_experiment_dag_state,
    load_experiment_dag_summary,
    initialize_instances,
    load_instance_config,
    normalize_runtime_state,
    plan_instances,
    provision_instances,
    resume_experiment_dag,
    run_experiment_dag,
    run_preflight,
    seed_instance_user_data,
    validate_experiment_dag_manifest,
    write_runtime_normalization_report,
)
from .shadow import (
    load_shadow_combat_report,
    run_shadow_combat_comparison,
    run_shadow_combat_evaluation,
)
from .train import (
    BehaviorCloningFloorBandWeight,
    BehaviorCloningTrainConfig,
    StrategicFinetuneTrainConfig,
    StrategicPretrainTrainConfig,
    OfflineCqlTrainConfig,
    load_benchmark_suite_manifest,
    load_benchmark_suite_summary,
    load_divergence_summary,
    DqnConfig,
    default_behavior_cloning_training_session_name,
    default_strategic_finetune_session_name,
    default_strategic_pretrain_session_name,
    default_offline_cql_training_session_name,
    run_benchmark_suite,
    run_behavior_cloning_evaluation,
    run_combat_dqn_checkpoint_comparison,
    run_combat_dqn_evaluation,
    run_combat_dqn_replay_suite,
    run_combat_dqn_schedule,
    run_combat_dqn_training,
    run_offline_cql_evaluation,
    run_policy_checkpoint_comparison,
    run_policy_pack_evaluation,
    train_behavior_cloning_policy,
    train_strategic_finetune_policy,
    train_strategic_pretrain_policy,
    train_offline_cql_policy,
)

app = typer.Typer(no_args_is_help=True, help="STS2 RL workspace utilities.")
instances_app = typer.Typer(no_args_is_help=True, help="Instance planning and manifest commands.")
job_app = typer.Typer(no_args_is_help=True, help="Multi-instance runtime job commands.")
dag_app = typer.Typer(no_args_is_help=True, help="Resumable experiment DAG orchestration commands.")
benchmark_app = typer.Typer(no_args_is_help=True, help="Read-only benchmark commands for STS2-Agent APIs.")
benchmark_suite_app = typer.Typer(no_args_is_help=True, help="Benchmark suite manifest and execution commands.")
collect_app = typer.Typer(no_args_is_help=True, help="Trajectory collection commands.")
train_app = typer.Typer(no_args_is_help=True, help="Training commands.")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation commands.")
dataset_app = typer.Typer(no_args_is_help=True, help="Manifest-driven dataset commands.")
community_app = typer.Typer(no_args_is_help=True, help="Community card stats snapshot commands.")
public_runs_app = typer.Typer(no_args_is_help=True, help="Public run archive commands.")
registry_app = typer.Typer(no_args_is_help=True, help="Local experiment registry commands.")
registry_alias_app = typer.Typer(no_args_is_help=True, help="Registry alias commands.")
predict_app = typer.Typer(no_args_is_help=True, help="Predictor dataset and training commands.")
predict_dataset_app = typer.Typer(no_args_is_help=True, help="Predictor dataset commands.")
predict_report_app = typer.Typer(no_args_is_help=True, help="Predictor reporting commands.")
predict_train_app = typer.Typer(no_args_is_help=True, help="Predictor training commands.")
shadow_app = typer.Typer(no_args_is_help=True, help="Shadow combat dataset evaluation commands.")
app.add_typer(instances_app, name="instances")
instances_app.add_typer(job_app, name="job")
instances_app.add_typer(dag_app, name="dag")
app.add_typer(benchmark_app, name="benchmark")
benchmark_app.add_typer(benchmark_suite_app, name="suite")
app.add_typer(collect_app, name="collect")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(dataset_app, name="dataset")
app.add_typer(community_app, name="community")
app.add_typer(public_runs_app, name="public-runs")
app.add_typer(registry_app, name="registry")
registry_app.add_typer(registry_alias_app, name="alias")
app.add_typer(predict_app, name="predict")
predict_app.add_typer(predict_dataset_app, name="dataset")
predict_app.add_typer(predict_report_app, name="report")
predict_app.add_typer(predict_train_app, name="train")
app.add_typer(shadow_app, name="shadow")

console = Console(width=160)


def _format_metric(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _json_table(title: str, rows: list[tuple[str, object]]) -> Table:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value")
    for label, value in rows:
        table.add_row(label, _format_metric(value))
    return table


def _dag_stage_table(title: str, stages: list[dict[str, object]]) -> Table:
    table = Table(title=title)
    table.add_column("Stage")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("Attempts")
    table.add_column("Depends On")
    table.add_column("Summary")
    for stage in stages:
        table.add_row(
            str(stage["stage_id"]),
            str(stage["kind"]),
            str(stage["status"]),
            str(stage.get("attempt_count", 0)),
            ", ".join(stage.get("depends_on", [])) or "-",
            str(stage.get("summary_path") or "-"),
        )
    return table


def _capability_overview_table(title: str, summary: dict[str, object]) -> Table:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Diagnostics", str(summary.get("diagnostic_count", 0)))
    table.add_row("Buckets", json.dumps(summary.get("bucket_histogram", {}), ensure_ascii=False))
    table.add_row("Owners", json.dumps(summary.get("owner_histogram", {}), ensure_ascii=False))
    table.add_row("Categories", json.dumps(summary.get("category_histogram", {}), ensure_ascii=False))
    table.add_row("Screens", json.dumps(summary.get("screen_histogram", {}), ensure_ascii=False))
    table.add_row("Unsupported Descriptors", str(summary.get("unsupported_descriptor_count", 0)))
    table.add_row("No-Action Timeouts", str(summary.get("no_action_timeout_count", 0)))
    table.add_row("Ambiguous Semantic Blocks", str(summary.get("ambiguous_semantic_block_count", 0)))
    table.add_row("Unexpected Runtime Divergence", str(summary.get("unexpected_runtime_divergence_count", 0)))
    return table


def _capability_diagnostics_table(title: str, diagnostics_payload: list[dict[str, object]]) -> Table:
    table = Table(title=title)
    table.add_column("Bucket")
    table.add_column("Owner")
    table.add_column("Category")
    table.add_column("Screen")
    table.add_column("Step")
    table.add_column("Detail", overflow="fold")
    for item in diagnostics_payload:
        detail = item.get("descriptor") or item.get("decision_reason") or item.get("stop_reason") or "-"
        table.add_row(
            str(item.get("bucket")),
            str(item.get("owner")),
            str(item.get("category")),
            str(item.get("screen_type")),
            str(item.get("step_index") or "-"),
            str(detail),
        )
    return table


def _resolve_prepare_target_option(*, prepare_main_menu: bool, prepare_target: str | None) -> str:
    if prepare_target is None:
        return "main_menu" if prepare_main_menu else "none"
    normalized = prepare_target.strip().lower()
    if normalized not in {"none", "main_menu", "character_select"}:
        raise typer.BadParameter("prepare-target must be one of: none, main_menu, character_select.")
    return normalized


def _resolve_runtime_target_option(target: str) -> str:
    normalized = target.strip().lower()
    if normalized not in {"main_menu", "character_select"}:
        raise typer.BadParameter("target must be either 'main_menu' or 'character_select'.")
    return normalized


def _resolve_community_source_type_option(source_type: str) -> str:
    normalized = source_type.strip().lower()
    if normalized not in {"reward", "shop", "event", "colorless", "starter", "unknown"}:
        raise typer.BadParameter(
            "source-type must be one of: reward, shop, event, colorless, starter, unknown."
        )
    return normalized


def _build_predictor_runtime_config(
    *,
    model_path: Path | None,
    mode: str,
    hooks: list[str] | None,
) -> PredictorRuntimeConfig | None:
    normalized_mode = normalize_predictor_mode(mode)
    normalized_hooks = normalize_predictor_hooks(hooks)
    resolved_model_path = None if model_path is None else model_path.expanduser().resolve()
    if normalized_mode == "heuristic_only" and resolved_model_path is None:
        return None
    if normalized_mode != "heuristic_only" and resolved_model_path is None:
        raise typer.BadParameter("predictor-guided modes require --predictor-model-path.")
    return PredictorRuntimeConfig(
        model_path=resolved_model_path,
        mode=normalized_mode,
        hooks=normalized_hooks,
    )


def _build_community_prior_runtime_config(
    *,
    source_path: Path | None,
    route_source_path: Path | None = None,
    reward_pick_weight: float = 1.15,
    selection_pick_weight: float = 1.05,
    selection_upgrade_weight: float = 0.55,
    selection_remove_weight: float = 0.95,
    shop_buy_weight: float = 1.00,
    route_weight: float = 0.90,
    reward_pick_neutral_rate: float = 0.33,
    shop_buy_neutral_rate: float = 0.10,
    route_neutral_win_rate: float = 0.50,
    pick_rate_scale: float = 3.0,
    buy_rate_scale: float = 5.0,
    win_delta_scale: float = 12.0,
    route_win_rate_scale: float = 8.0,
    min_sample_size: int = 40,
    route_min_sample_size: int = 30,
    max_confidence_sample_size: int = 1200,
    max_source_age_days: int | None = None,
) -> CommunityPriorRuntimeConfig | None:
    if source_path is None:
        return None
    return CommunityPriorRuntimeConfig(
        source_path=source_path.expanduser().resolve(),
        route_source_path=None if route_source_path is None else route_source_path.expanduser().resolve(),
        reward_pick_weight=reward_pick_weight,
        selection_pick_weight=selection_pick_weight,
        selection_upgrade_weight=selection_upgrade_weight,
        selection_remove_weight=selection_remove_weight,
        shop_buy_weight=shop_buy_weight,
        route_weight=route_weight,
        reward_pick_neutral_rate=reward_pick_neutral_rate,
        shop_buy_neutral_rate=shop_buy_neutral_rate,
        route_neutral_win_rate=route_neutral_win_rate,
        pick_rate_scale=pick_rate_scale,
        buy_rate_scale=buy_rate_scale,
        win_delta_scale=win_delta_scale,
        route_win_rate_scale=route_win_rate_scale,
        min_sample_size=min_sample_size,
        route_min_sample_size=route_min_sample_size,
        max_confidence_sample_size=max_confidence_sample_size,
        max_source_age_days=max_source_age_days,
    )


def _build_strategic_runtime_config(
    *,
    checkpoint_path: Path | None,
    mode: str,
    hooks: list[str] | None,
) -> StrategicRuntimeConfig | None:
    normalized_mode = normalize_strategic_mode(mode)
    normalized_hooks = normalize_strategic_hooks(hooks)
    resolved_checkpoint_path = None if checkpoint_path is None else checkpoint_path.expanduser().resolve()
    if normalized_mode == "heuristic_only" and resolved_checkpoint_path is None:
        return None
    if normalized_mode != "heuristic_only" and resolved_checkpoint_path is None:
        raise typer.BadParameter("strategic-guided modes require --strategic-checkpoint-path.")
    return StrategicRuntimeConfig(
        checkpoint_path=resolved_checkpoint_path,
        mode=normalized_mode,
        hooks=normalized_hooks,
    )


def _build_live_game_run_contract(
    *,
    run_mode: str | None,
    game_seed: str | None,
    seed_source: str | None,
    game_character_id: str | None,
    game_ascension: int | None,
    custom_modifier: list[str] | None,
    progress_profile: str | None,
    benchmark_contract_id: str | None,
    strict_game_run_contract: bool,
):
    return build_game_run_contract(
        run_mode=run_mode,
        game_seed=game_seed,
        seed_source=seed_source,
        character_id=game_character_id,
        ascension=game_ascension,
        custom_modifiers=custom_modifier,
        progress_profile=progress_profile,
        benchmark_contract_id=benchmark_contract_id,
        strict=strict_game_run_contract,
    )


def _build_dqn_config(
    *,
    learning_rate: float | None,
    gamma: float | None,
    epsilon_start: float | None,
    epsilon_end: float | None,
    epsilon_decay_steps: int | None,
    replay_capacity: int,
    batch_size: int,
    min_replay_size: int,
    target_sync_interval: int,
    updates_per_env_step: int,
    huber_delta: float,
    hidden_sizes: list[int],
    seed: int | None,
    double_dqn: bool,
    n_step: int,
    prioritized_replay: bool,
    priority_alpha: float,
    priority_beta_start: float,
    priority_beta_end: float,
    priority_beta_decay_steps: int,
    priority_epsilon: float,
) -> DqnConfig:
    return DqnConfig(
        learning_rate=0.001 if learning_rate is None else learning_rate,
        gamma=0.95 if gamma is None else gamma,
        epsilon_start=0.20 if epsilon_start is None else epsilon_start,
        epsilon_end=0.02 if epsilon_end is None else epsilon_end,
        epsilon_decay_steps=2000 if epsilon_decay_steps is None else epsilon_decay_steps,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        min_replay_size=min_replay_size,
        target_sync_interval=target_sync_interval,
        updates_per_env_step=updates_per_env_step,
        huber_delta=huber_delta,
        hidden_sizes=tuple(hidden_sizes),
        seed=0 if seed is None else seed,
        double_dqn=double_dqn,
        n_step=n_step,
        prioritized_replay=prioritized_replay,
        priority_alpha=priority_alpha,
        priority_beta_start=priority_beta_start,
        priority_beta_end=priority_beta_end,
        priority_beta_decay_steps=priority_beta_decay_steps,
        priority_epsilon=priority_epsilon,
    )


def _parse_weight_map(values: list[str] | None, *, option_name: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for item in values or []:
        if "=" not in item:
            raise typer.BadParameter(f"{option_name} entries must use key=value format.")
        key, raw_value = item.split("=", maxsplit=1)
        key = key.strip().lower()
        if not key:
            raise typer.BadParameter(f"{option_name} keys must not be empty.")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise typer.BadParameter(f"{option_name} values must be numeric.") from exc
        if value <= 0.0:
            raise typer.BadParameter(f"{option_name} values must be positive.")
        parsed[key] = value
    return parsed


def _parse_floor_band_weights(values: list[str] | None) -> tuple[BehaviorCloningFloorBandWeight, ...]:
    parsed: list[BehaviorCloningFloorBandWeight] = []
    for item in values or []:
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise typer.BadParameter("--floor-band-weight entries must use min:max:weight format.")
        raw_min, raw_max, raw_weight = parts
        min_floor = None if raw_min in {"", "*"} else int(raw_min)
        max_floor = None if raw_max in {"", "*"} else int(raw_max)
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise typer.BadParameter("--floor-band-weight weights must be numeric.") from exc
        parsed.append(
            BehaviorCloningFloorBandWeight(
                min_floor=min_floor,
                max_floor=max_floor,
                weight=weight,
            )
        )
    return tuple(parsed)


@instances_app.command("plan")
def plan_instances_command(
    config: Path = typer.Option(
        Path("configs/instances/local.example.toml"),
        "--config",
        help="Path to instance TOML config.",
    ),
) -> None:
    resolved = load_instance_config(config)
    specs = plan_instances(resolved)

    table = Table(title=f"Planned Instances ({config})")
    table.add_column("Instance")
    table.add_column("Port")
    table.add_column("Base URL")
    table.add_column("Runtime Root")
    table.add_column("Logs Root")

    for spec in specs:
        table.add_row(
            spec.instance_id,
            str(spec.api_port),
            spec.base_url,
            str(spec.instance_root),
            str(spec.logs_root),
        )

    console.print(table)


@instances_app.command("init")
def init_instances_command(
    config: Path = typer.Option(
        Path("configs/instances/local.example.toml"),
        "--config",
        help="Path to instance TOML config.",
    ),
) -> None:
    resolved = load_instance_config(config)
    initialized = initialize_instances(resolved)

    table = Table(title=f"Initialized Instances ({config})")
    table.add_column("Instance")
    table.add_column("Port")
    table.add_column("Manifest")

    for item in initialized:
        table.add_row(
            item.spec.instance_id,
            str(item.spec.api_port),
            str(item.manifest_path),
        )

    console.print(table)


@instances_app.command("provision")
def provision_instances_command(
    config: Path = typer.Option(
        Path("configs/instances/local.example.toml"),
        "--config",
        help="Path to instance TOML config.",
    ),
    replace_existing: bool = typer.Option(
        True,
        "--replace-existing/--no-replace-existing",
        help="Replace existing runtime instance roots with a clean copy of the baseline.",
    ),
) -> None:
    resolved = load_instance_config(config)
    provisioned = provision_instances(resolved, replace_existing=replace_existing)

    table = Table(title=f"Provisioned Instances ({config})")
    table.add_column("Instance")
    table.add_column("Port")
    table.add_column("Exe Path")
    table.add_column("Manifest")

    for item in provisioned:
        table.add_row(
            item.spec.instance_id,
            str(item.spec.api_port),
            str(item.spec.instance_root / "SlayTheSpire2.exe"),
            str(item.manifest_path),
        )

    console.print(table)


@instances_app.command("status")
def status_instances_command(
    config: Path = typer.Option(
        Path("configs/instances/local.example.toml"),
        "--config",
        help="Path to instance TOML config.",
    ),
    timeout_seconds: float = typer.Option(0.5, "--timeout-seconds", min=0.1),
) -> None:
    resolved = load_instance_config(config)
    statuses = collect_instance_statuses(resolved, timeout_seconds=timeout_seconds)

    table = Table(title=f"Instance Status ({config})")
    table.add_column("Instance")
    table.add_column("Port")
    table.add_column("Manifest")
    table.add_column("API")
    table.add_column("Game")
    table.add_column("Mod")
    table.add_column("Error")

    for status in statuses:
        table.add_row(
            status.instance_id,
            str(status.api_port),
            status.manifest_status or ("missing" if not status.manifest_exists else "unknown"),
            status.api_status or ("down" if not status.api_reachable else "unknown"),
            status.game_version or "-",
            status.mod_version or "-",
            status.error or "-",
        )

    console.print(table)


@instances_app.command("normalize")
def normalize_instances_command(
    config: Path = typer.Option(
        Path("configs/instances/local.example.toml"),
        "--config",
        help="Path to instance TOML config.",
    ),
    target: str = typer.Option(
        "main_menu",
        "--target",
        help="Normalization target: main_menu or character_select.",
    ),
    instance_id: list[str] | None = typer.Option(
        None,
        "--instance-id",
        help="Optional repeatable instance id filter. Defaults to every instance in the config.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/runtime-normalize"),
        "--output-root",
        help="Root directory for runtime normalization reports.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional normalization session directory name. Defaults to a UTC timestamp.",
    ),
    poll_interval_seconds: float = typer.Option(0.25, "--poll-interval-seconds", min=0.01),
    max_idle_polls: int = typer.Option(40, "--max-idle-polls", min=1),
    max_steps: int = typer.Option(64, "--max-steps", min=1),
    request_timeout_seconds: float = typer.Option(30.0, "--request-timeout-seconds", min=1.0),
) -> None:
    resolved_target = _resolve_runtime_target_option(target)
    resolved = load_instance_config(config)
    specs = build_instance_specs(resolved)
    requested_ids = set(instance_id or [])
    known_ids = {spec.instance_id for spec in specs}
    unknown_ids = sorted(requested_ids - known_ids)
    if unknown_ids:
        raise typer.BadParameter(f"Unknown instance ids: {', '.join(unknown_ids)}")

    selected_specs = [spec for spec in specs if not requested_ids or spec.instance_id in requested_ids]
    if not selected_specs:
        raise typer.BadParameter("No instances selected for normalization.")

    session_root = output_root / (session_name or datetime.now(UTC).strftime("runtime-normalize-%Y%m%d-%H%M%S"))
    session_root.mkdir(parents=True, exist_ok=True)
    summary_path = session_root / "normalize-summary.json"

    rows: list[dict[str, object]] = []
    for spec in selected_specs:
        report = normalize_runtime_state(
            base_url=spec.base_url,
            target=resolved_target,
            poll_interval_seconds=poll_interval_seconds,
            max_idle_polls=max_idle_polls,
            max_steps=max_steps,
            request_timeout_seconds=request_timeout_seconds,
        )
        report_path = write_runtime_normalization_report(report, session_root / f"{spec.instance_id}.json")
        rows.append(
            {
                "instance_id": spec.instance_id,
                "base_url": spec.base_url,
                "target": resolved_target,
                "reached_target": report.reached_target,
                "stop_reason": report.stop_reason,
                "initial_screen": report.initial_screen,
                "final_screen": report.final_screen,
                "step_count": report.step_count,
                "wait_count": report.wait_count,
                "action_sequence": report.action_sequence,
                "strategy_histogram": report.strategy_histogram,
                "report_path": str(report_path),
            }
        )

    summary_payload = {
        "config_path": str(config),
        "session_root": str(session_root),
        "target": resolved_target,
        "instance_count": len(rows),
        "reached_target_count": sum(1 for item in rows if item["reached_target"]),
        "summary_path": str(summary_path),
        "reports": rows,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    table = Table(title=f"Runtime Normalize ({config})")
    table.add_column("Instance")
    table.add_column("Port")
    table.add_column("Target")
    table.add_column("Reached")
    table.add_column("Stop Reason")
    table.add_column("Initial")
    table.add_column("Final")
    table.add_column("Steps")
    table.add_column("Report")

    for spec, row in zip(selected_specs, rows, strict=True):
        table.add_row(
            spec.instance_id,
            str(spec.api_port),
            str(row["target"]),
            "yes" if bool(row["reached_target"]) else "no",
            str(row["stop_reason"]),
            str(row["initial_screen"]),
            str(row["final_screen"]),
            str(row["step_count"]),
            str(row["report_path"]),
        )

    console.print(table)
    console.print(f"Summary: {summary_path}")


@job_app.command("validate")
def validate_runtime_job_command(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        help="Path to a runtime job manifest in JSON or TOML format.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Optional instance config used to validate referenced instance ids.",
    ),
) -> None:
    known_instance_ids = None
    if config is not None:
        resolved_config = load_instance_config(config)
        known_instance_ids = {spec.instance_id for spec in build_instance_specs(resolved_config)}
    job_manifest = load_runtime_job_manifest(manifest, known_instance_ids=known_instance_ids)

    summary = Table(title=f"Runtime Job ({manifest})")
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("Job Name", job_manifest.job_name)
    summary.add_row("Description", job_manifest.description or "-")
    summary.add_row("Output Root", str(job_manifest.output_root))
    summary.add_row("Task Count", str(len(job_manifest.tasks)))
    summary.add_row("Concurrency Limit", str(job_manifest.concurrency_limit or "auto"))
    summary.add_row("Watchdog", job_manifest.watchdog.model_dump_json())
    console.print(summary)

    tasks = Table(title="Runtime Job Tasks")
    tasks.add_column("Task")
    tasks.add_column("Kind")
    tasks.add_column("Instances")
    tasks.add_column("Attempts")
    tasks.add_column("Normalize")
    for task in job_manifest.tasks:
        tasks.add_row(
            task.task_id,
            task.kind,
            ", ".join(task.instance_ids) if task.instance_ids else "<all>",
            str(task.max_attempts),
            task.normalize_target,
        )
    console.print(tasks)


@job_app.command("run")
def run_runtime_job_command(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        help="Path to a runtime job manifest in JSON or TOML format.",
    ),
    config: Path = typer.Option(
        Path("configs/instances/local.example.toml"),
        "--config",
        help="Path to instance TOML config.",
    ),
    job_name: str | None = typer.Option(
        None,
        "--job-name",
        help="Optional job output directory name. Defaults to manifest.job_name.",
    ),
    output_root: Path | None = typer.Option(
        None,
        "--output-root",
        help="Optional override for the job output root.",
    ),
    replace_existing: bool = typer.Option(
        False,
        "--replace-existing/--no-replace-existing",
        help="Replace an existing job directory when job_name already exists.",
    ),
) -> None:
    report = run_runtime_job(
        manifest,
        config=config,
        job_name=job_name,
        output_root=output_root,
        replace_existing=replace_existing,
    )
    summary_payload = load_runtime_job_summary(report.summary_path)

    summary = Table(title=f"Runtime Job Run ({summary_payload['job_name']})")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Job Root", str(summary_payload["job_root"]))
    summary.add_row("Executions", str(summary_payload["execution_count"]))
    summary.add_row("Statuses", json.dumps(summary_payload["execution_status_histogram"], ensure_ascii=False))
    summary.add_row("Task Kinds", json.dumps(summary_payload["task_kind_histogram"], ensure_ascii=False))
    summary.add_row("Summary Path", str(report.summary_path))
    summary.add_row("Log Path", str(report.log_path))
    console.print(summary)

    executions = Table(title="Execution Results")
    executions.add_column("Task")
    executions.add_column("Kind")
    executions.add_column("Instance")
    executions.add_column("Status")
    executions.add_column("Attempts")
    executions.add_column("Watchdog")
    executions.add_column("Summary")
    for execution in summary_payload["executions"]:
        executions.add_row(
            str(execution["task_id"]),
            str(execution["task_kind"]),
            str(execution["instance_id"]),
            str(execution["status"]),
            str(execution["attempt_count"]),
            str(execution["watchdog_state"]),
            str(execution["summary_path"] or "-"),
        )
    console.print(executions)


@job_app.command("summary")
def runtime_job_summary_command(
    source: Path = typer.Option(
        ...,
        "--source",
        help="Runtime job directory or job-summary.json path.",
    ),
) -> None:
    summary_payload = load_runtime_job_summary(source)

    overview = Table(title=f"Runtime Job Summary ({summary_payload['job_name']})")
    overview.add_column("Metric")
    overview.add_column("Value")
    overview.add_row("Job Root", str(summary_payload["job_root"]))
    overview.add_row("Executions", str(summary_payload["execution_count"]))
    overview.add_row("Statuses", json.dumps(summary_payload["execution_status_histogram"], ensure_ascii=False))
    overview.add_row("Task Kinds", json.dumps(summary_payload["task_kind_histogram"], ensure_ascii=False))
    overview.add_row("Summary Path", str(summary_payload["summary_path"]))
    overview.add_row("Log Path", str(summary_payload["log_path"]))
    console.print(overview)

    watchdogs = Table(title="Instance Watchdogs")
    watchdogs.add_column("Instance")
    watchdogs.add_column("State")
    watchdogs.add_column("Failures")
    watchdogs.add_column("Successes")
    watchdogs.add_column("Cooldown")
    watchdogs.add_column("Last Failure")
    for instance_id, payload in summary_payload["watchdogs"].items():
        watchdogs.add_row(
            str(instance_id),
            str(payload["state"]),
            str(payload["total_failures"]),
            str(payload["total_successes"]),
            str(payload["cooldown_until_utc"] or "-"),
            str(payload["last_failure_kind"] or "-"),
        )
    console.print(watchdogs)


@dag_app.command("validate")
def validate_experiment_dag_command(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        help="Path to an experiment DAG manifest in JSON or TOML format.",
    ),
    deep: bool = typer.Option(
        True,
        "--deep/--no-deep",
        help="Run stage-specific validation for static inputs such as runtime job manifests and dataset manifests.",
    ),
) -> None:
    dag_manifest = load_experiment_dag_manifest(manifest)
    report = validate_experiment_dag_manifest(dag_manifest, deep=deep)

    summary = Table(title=f"Experiment DAG ({manifest})")
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("DAG Name", report.dag_name)
    summary.add_row("Description", dag_manifest.description or "-")
    summary.add_row("Output Root", str(dag_manifest.output_root))
    summary.add_row("Lock Root", str(dag_manifest.lock_root))
    summary.add_row("Stage Count", str(report.stage_count))
    summary.add_row("Stage Kinds", json.dumps(report.stage_kind_histogram, ensure_ascii=False))
    summary.add_row("Stage Order", " -> ".join(report.stage_order))
    console.print(summary)

    stages_table = Table(title="DAG Stages")
    stages_table.add_column("Stage")
    stages_table.add_column("Kind")
    stages_table.add_column("Depends On")
    stages_table.add_column("Attempts")
    stages_table.add_column("Resources")
    for stage in dag_manifest.stages:
        stages_table.add_row(
            stage.stage_id,
            stage.kind,
            ", ".join(stage.depends_on) or "-",
            str(stage.max_attempts),
            ", ".join(report.stage_resource_hints.get(stage.stage_id, [])) or "-",
        )
    console.print(stages_table)


@dag_app.command("run")
def run_experiment_dag_command(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        help="Path to an experiment DAG manifest in JSON or TOML format.",
    ),
    run_name: str | None = typer.Option(
        None,
        "--run-name",
        help="Optional run directory name. Defaults to manifest.dag_name.",
    ),
    output_root: Path | None = typer.Option(
        None,
        "--output-root",
        help="Optional override for the DAG output root.",
    ),
    replace_existing: bool = typer.Option(
        False,
        "--replace-existing/--no-replace-existing",
        help="Replace an existing DAG run directory when run_name already exists.",
    ),
) -> None:
    report = run_experiment_dag(
        manifest,
        run_name=run_name,
        output_root=output_root,
        replace_existing=replace_existing,
    )
    summary_payload = load_experiment_dag_summary(report.summary_path)
    console.print(
        _json_table(
            f"Experiment DAG Run ({summary_payload['dag_name']})",
            [
                ("Run Root", summary_payload["run_root"]),
                ("Status", summary_payload["status"]),
                ("Invocation Count", summary_payload["invocation_count"]),
                ("Stage Count", summary_payload["stage_count"]),
                ("Reused Stages", summary_payload["reused_stage_count"]),
                ("Critical Path Seconds", summary_payload["critical_path_seconds"]),
                ("Summary Path", report.summary_path),
                ("State Path", report.state_path),
                ("Log Path", report.log_path),
            ],
        )
    )
    console.print(_dag_stage_table("DAG Stages", list(summary_payload["stages"])))


@dag_app.command("resume")
def resume_experiment_dag_command(
    source: Path = typer.Option(
        ...,
        "--source",
        help="Existing DAG run directory or dag-state.json path.",
    ),
    retry_stage: list[str] | None = typer.Option(
        None,
        "--retry-stage",
        help="Optional repeatable stage id filter. When omitted, every non-succeeded stage is retried.",
    ),
) -> None:
    report = resume_experiment_dag(source, retry_stage_ids=retry_stage or [])
    summary_payload = load_experiment_dag_summary(report.summary_path)
    console.print(
        _json_table(
            f"Experiment DAG Resume ({summary_payload['dag_name']})",
            [
                ("Run Root", summary_payload["run_root"]),
                ("Status", summary_payload["status"]),
                ("Invocation Count", summary_payload["invocation_count"]),
                ("Summary Path", report.summary_path),
                ("State Path", report.state_path),
                ("Log Path", report.log_path),
            ],
        )
    )
    console.print(_dag_stage_table("DAG Stages", list(summary_payload["stages"])))


@dag_app.command("inspect")
def inspect_experiment_dag_command(
    source: Path = typer.Option(
        ...,
        "--source",
        help="DAG run directory or dag-state.json path.",
    ),
) -> None:
    state_payload = load_experiment_dag_state(source)
    console.print(
        _json_table(
            f"Experiment DAG State ({state_payload['dag_name']})",
            [
                ("Run Root", state_payload["run_root"]),
                ("Status", state_payload["status"]),
                ("Invocation Count", state_payload["invocation_count"]),
                ("Manifest", state_payload["resolved_manifest_path"]),
                ("Started", state_payload["started_at_utc"]),
                ("Finished", state_payload["finished_at_utc"]),
            ],
        )
    )
    console.print(_dag_stage_table("Stage State", [dict(state_payload["stages"][stage_id]) for stage_id in state_payload["stage_order"]]))


@dag_app.command("summary")
def experiment_dag_summary_command(
    source: Path = typer.Option(
        ...,
        "--source",
        help="DAG run directory or dag-summary.json path.",
    ),
) -> None:
    summary_payload = load_experiment_dag_summary(source)
    console.print(
        _json_table(
            f"Experiment DAG Summary ({summary_payload['dag_name']})",
            [
                ("Run Root", summary_payload["run_root"]),
                ("Status", summary_payload["status"]),
                ("Invocation Count", summary_payload["invocation_count"]),
                ("Stage Count", summary_payload["stage_count"]),
                ("Statuses", json.dumps(summary_payload["stage_status_histogram"], ensure_ascii=False)),
                ("Reused Stages", summary_payload["reused_stage_count"]),
                ("Critical Path Seconds", summary_payload["critical_path_seconds"]),
                ("Manifest", summary_payload["resolved_manifest_path"]),
            ],
        )
    )
    console.print(_dag_stage_table("Stage Summary", list(summary_payload["stages"])))


@instances_app.command("bootstrap-user-data")
def bootstrap_user_data_command(
    config: Path = typer.Option(
        Path("configs/instances/local.example.toml"),
        "--config",
        help="Path to instance TOML config.",
    ),
    source_user_dir: Path = typer.Option(
        ...,
        "--source-user-dir",
        help="User-data directory from a manually prepared instance with mods approved and unlocks applied.",
    ),
    template_dir: Path = typer.Option(
        Path("data/user-data-templates/golden"),
        "--template-dir",
        help="Repo-local golden template directory to refresh before seeding instances.",
    ),
    replace_existing: bool = typer.Option(
        True,
        "--replace-existing/--no-replace-existing",
        help="Replace any existing golden template and per-instance user-data roots.",
    ),
    seed_instances: bool = typer.Option(
        True,
        "--seed-instances/--no-seed-instances",
        help="Clone the refreshed golden template into every per-instance user-data root.",
    ),
    window_width: int = typer.Option(960, "--window-width", min=320),
    window_height: int = typer.Option(540, "--window-height", min=240),
    window_pos_x: int = typer.Option(-1, "--window-pos-x"),
    window_pos_y: int = typer.Option(-1, "--window-pos-y"),
    skip_intro_logo: bool = typer.Option(True, "--skip-intro-logo/--keep-intro-logo"),
    mods_enabled: bool = typer.Option(True, "--mods-enabled/--mods-disabled"),
    enable_all_listed_mods: bool = typer.Option(
        True,
        "--enable-all-listed-mods/--keep-mod-list-state",
        help="Force every mod already listed in settings.save to enabled=true.",
    ),
) -> None:
    resolved = load_instance_config(config)
    report = bootstrap_instance_user_data(
        resolved,
        source_user_dir=source_user_dir,
        template_dir=template_dir,
        replace_existing=replace_existing,
        seed_instances=seed_instances,
        window_width=window_width,
        window_height=window_height,
        window_pos_x=window_pos_x,
        window_pos_y=window_pos_y,
        skip_intro_logo=skip_intro_logo,
        mods_enabled=mods_enabled,
        enable_all_listed_mods=enable_all_listed_mods,
    )

    summary = Table(title=f"Bootstrapped User Data ({config})")
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("Golden Template", str(report.template_root))
    summary.add_row("Patched settings.save files", str(len(report.settings_paths)))
    summary.add_row("Pruned runtime artifacts", str(len(report.pruned_paths)))
    summary.add_row("Seeded instance dirs", str(len(report.seeded_paths)))
    console.print(summary)

    if report.settings_paths:
        settings_table = Table(title="Patched Settings Files")
        settings_table.add_column("Path")
        for path in report.settings_paths:
            settings_table.add_row(str(path))
        console.print(settings_table)

    if report.seeded_paths:
        seeded_table = Table(title="Seeded Instance User Dirs")
        seeded_table.add_column("Path")
        for path in report.seeded_paths:
            seeded_table.add_row(str(path))
        console.print(seeded_table)


@instances_app.command("seed-user-data")
def seed_user_data_command(
    config: Path = typer.Option(
        Path("configs/instances/local.example.toml"),
        "--config",
        help="Path to instance TOML config.",
    ),
    source_user_dir: Path = typer.Option(
        ...,
        "--source-user-dir",
        help="An already-approved STS2 user-data directory to clone into each instance-specific user-data root.",
    ),
    replace_existing: bool = typer.Option(
        True,
        "--replace-existing/--no-replace-existing",
        help="Replace any existing per-instance user-data roots before cloning the seed.",
    ),
) -> None:
    resolved = load_instance_config(config)
    seeded_paths = seed_instance_user_data(
        resolved,
        source_user_dir=source_user_dir,
        replace_existing=replace_existing,
    )

    table = Table(title=f"Seeded User Data ({config})")
    table.add_column("Instance")
    table.add_column("User Data Dir")

    for index, path in enumerate(seeded_paths, start=1):
        table.add_row(f"inst-{index:02d}", str(path))

    console.print(table)


@instances_app.command("launch-plan")
def launch_plan_command(
    config: Path = typer.Option(
        Path("configs/instances/local.example.toml"),
        "--config",
        help="Path to instance TOML config.",
    ),
    rendering_driver: str = typer.Option(
        "vulkan",
        "--rendering-driver",
        help="Rendering driver to pass through at startup. Use 'default' to omit the flag.",
    ),
    steam_app_id: int = typer.Option(2868840, "--steam-app-id"),
    launch_retries: int = typer.Option(1, "--launch-retries", min=0),
    enable_debug_actions: bool = typer.Option(False, "--enable-debug-actions"),
    attempts: int = typer.Option(40, "--attempts", min=1),
    delay_seconds: int = typer.Option(2, "--delay-seconds", min=1),
) -> None:
    resolved = load_instance_config(config)
    plans = build_windows_launch_plans(
        resolved,
        rendering_driver=rendering_driver,
        steam_app_id=steam_app_id,
        launch_retries=launch_retries,
        enable_debug_actions=enable_debug_actions,
        attempts=attempts,
        delay_seconds=delay_seconds,
    )

    table = Table(title=f"Launch Plan ({config})")
    table.add_column("Instance")
    table.add_column("Port")
    table.add_column("Command")

    for plan in plans:
        table.add_row(plan.instance_id, str(plan.api_port), plan.command)

    console.print(table)


@instances_app.command("preflight")
def preflight_instances_command(
    config: Path = typer.Option(
        Path("configs/instances/local.example.toml"),
        "--config",
        help="Path to instance TOML config.",
    ),
    sts2_agent_root: Path | None = typer.Option(None, "--sts2-agent-root", help="Optional local STS2-Agent repo root."),
    exe_path: Path | None = typer.Option(None, "--exe-path", help="Optional SlayTheSpire2.exe path."),
) -> None:
    resolved = load_instance_config(config)
    report = run_preflight(resolved, sts2_agent_root=sts2_agent_root, exe_path=exe_path)

    table = Table(title=f"Preflight ({config})")
    table.add_column("Check")
    table.add_column("Level")
    table.add_column("Message")

    for check in report.checks:
        table.add_row(check.name, check.level, check.message)

    console.print(table)
    if report.game_version:
        console.print(f"Detected reference game version: [bold]{report.game_version}[/bold]")


@benchmark_app.command("health")
def benchmark_health_command(
    base_url: str = typer.Option("http://127.0.0.1:8080", "--base-url"),
    samples: int = typer.Option(5, "--samples", min=1),
) -> None:
    result = benchmark_health(base_url, samples=samples)
    console.print_json(
        json.dumps(
            {
                "service": result.health.service,
                "mod_version": result.health.mod_version,
                "protocol_version": result.health.protocol_version,
                "game_version": result.health.game_version,
                "status": result.health.status,
                "latency": result.latency.__dict__,
            }
        )
    )


@benchmark_app.command("observe")
def benchmark_observe_command(
    base_url: str = typer.Option("http://127.0.0.1:8080", "--base-url"),
    samples: int = typer.Option(5, "--samples", min=1),
) -> None:
    result = benchmark_observe(base_url, samples=samples)
    console.print_json(
        json.dumps(
            {
                "last_screen": result.last_screen,
                "last_run_id": result.last_run_id,
                "state_latency": result.state_latency.__dict__,
                "actions_latency": result.actions_latency.__dict__,
                "candidate_build_latency": result.candidate_build_latency.__dict__,
                "candidate_count": result.candidate_count.__dict__,
                "build_warnings": result.build_warnings,
            }
        )
    )


@benchmark_suite_app.command("validate")
def benchmark_suite_validate_command(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        help="Path to a benchmark suite manifest in JSON or TOML format.",
    ),
) -> None:
    suite = load_benchmark_suite_manifest(manifest)
    table = Table(title=f"Benchmark Suite ({suite.suite_name})")
    table.add_column("Case")
    table.add_column("Mode")
    table.add_column("Repeats")
    table.add_column("Prepare")
    table.add_column("Scenario")

    for case in suite.cases:
        table.add_row(
            case.case_id,
            case.mode,
            str(case.repeats),
            case.prepare_target,
            json.dumps(case.scenario.model_dump(mode="json"), ensure_ascii=False),
        )

    console.print(table)
    console.print(
        _json_table(
            "Suite Config",
            [
                ("Base URL", suite.base_url),
                ("Stats", suite.stats.model_dump(mode="json")),
                ("Cases", len(suite.cases)),
            ],
        )
    )


@benchmark_suite_app.command("run")
def benchmark_suite_run_command(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        help="Path to a benchmark suite manifest in JSON or TOML format.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/benchmarks"),
        "--output-root",
        help="Root directory for benchmark suite artifacts.",
    ),
    suite_name: str | None = typer.Option(
        None,
        "--suite-name",
        help="Optional suite output directory name. Defaults to manifest suite_name.",
    ),
    replace_existing: bool = typer.Option(
        False,
        "--replace-existing/--no-replace-existing",
        help="Replace an existing suite output directory.",
    ),
) -> None:
    report = run_benchmark_suite(
        manifest,
        output_root=output_root,
        suite_name=suite_name,
        replace_existing=replace_existing,
    )
    summary = load_benchmark_suite_summary(report.summary_path)

    table = Table(title=f"Benchmark Suite ({report.suite_name})")
    table.add_column("Case")
    table.add_column("Mode")
    table.add_column("Primary Metric")
    table.add_column("Estimate")
    table.add_column("CI")

    for case in summary.get("cases", []):
        primary_metric = case.get("primary_metric", {})
        table.add_row(
            str(case.get("case_id")),
            str(case.get("mode")),
            str(primary_metric.get("name", "-")),
            _format_metric(primary_metric.get("estimate")),
            f"{_format_metric(primary_metric.get('ci_low'))} .. {_format_metric(primary_metric.get('ci_high'))}",
        )

    console.print(table)
    console.print(
        _json_table(
            "Suite Output",
            [
                ("Suite Dir", report.suite_dir),
                ("Summary Path", report.summary_path),
                ("Log Path", report.log_path),
                ("Case Count", len(report.case_reports)),
                ("Shadow Cases", summary.get("shadow", {}).get("configured_case_count")),
                ("Shadow Comparable Encounters", summary.get("shadow", {}).get("comparable_encounter_count")),
            ],
        )
    )


@benchmark_suite_app.command("summary")
def benchmark_suite_summary_command(
    source: Path = typer.Option(
        ...,
        "--source",
        help="Suite directory or benchmark-suite-summary.json path.",
    ),
) -> None:
    summary = load_benchmark_suite_summary(source)
    table = Table(title=f"Benchmark Suite ({summary.get('suite_name', 'unknown')})")
    table.add_column("Case")
    table.add_column("Mode")
    table.add_column("Primary Metric")
    table.add_column("Estimate")
    table.add_column("CI")

    for case in summary.get("cases", []):
        primary_metric = case.get("primary_metric", {})
        table.add_row(
            str(case.get("case_id")),
            str(case.get("mode")),
            str(primary_metric.get("name", "-")),
            _format_metric(primary_metric.get("estimate")),
            f"{_format_metric(primary_metric.get('ci_low'))} .. {_format_metric(primary_metric.get('ci_high'))}",
        )

    console.print(table)
    console.print(
        _json_table(
            "Suite Summary",
            [
                ("Base URL", summary.get("base_url")),
                ("Case Count", summary.get("case_count")),
                ("Case Modes", summary.get("case_mode_histogram")),
                ("Shadow", summary.get("shadow")),
                ("Stats", summary.get("stats")),
                ("Summary Path", summary.get("summary_path")),
                ("Log Path", summary.get("log_path")),
            ],
        )
    )


@collect_app.command("rollouts")
def collect_rollouts_command(
    config: Path = typer.Option(
        Path("configs/instances/local.example.toml"),
        "--config",
        help="Path to instance TOML config.",
    ),
    output_root: Path = typer.Option(
        Path("data/trajectories"),
        "--output-root",
        help="Root directory where per-session trajectory files will be written.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional subdirectory name under output-root. Defaults to a UTC timestamp.",
    ),
    max_steps_per_instance: int = typer.Option(
        0,
        "--max-steps-per-instance",
        min=0,
        help="Hard per-instance step budget. Use 0 to disable the step budget.",
    ),
    max_runs_per_instance: int = typer.Option(
        1,
        "--max-runs-per-instance",
        "--max-episodes-per-instance",
        min=0,
        help="Completed-run budget per instance. Use 0 to disable the run budget.",
    ),
    max_combats_per_instance: int = typer.Option(
        0,
        "--max-combats-per-instance",
        min=0,
        help="Completed-combat budget per instance. Use 0 to disable the combat budget.",
    ),
    poll_interval_seconds: float = typer.Option(0.25, "--poll-interval-seconds", min=0.01),
    idle_timeout_seconds: float = typer.Option(15.0, "--idle-timeout-seconds", min=0.1),
    policy_pack: str = typer.Option(
        "baseline",
        "--policy-pack",
        help="Policy pack profile: baseline, legacy, planner, planner_assist, or conservative.",
    ),
    predictor_model_path: Path | None = typer.Option(
        None,
        "--predictor-model-path",
        help="Optional predictor model path for runtime-guided scoring.",
    ),
    predictor_mode: str = typer.Option(
        "heuristic_only",
        "--predictor-mode",
        help="Predictor guidance mode: heuristic_only, assist, or dominant.",
    ),
    predictor_hook: list[str] | None = typer.Option(
        None,
        "--predictor-hook",
        help="Repeatable predictor hook filter. Defaults to all supported hooks when predictor is enabled.",
    ),
    strategic_checkpoint_path: Path | None = typer.Option(
        None,
        "--strategic-checkpoint-path",
        help="Optional strategic_pretrain or strategic_finetune checkpoint for runtime strategic guidance.",
    ),
    strategic_mode: str = typer.Option(
        "heuristic_only",
        "--strategic-mode",
        help="Strategic guidance mode: heuristic_only, assist, or dominant.",
    ),
    strategic_hook: list[str] | None = typer.Option(
        None,
        "--strategic-hook",
        help="Repeatable strategic hook filter. Defaults to all supported hooks when strategic guidance is enabled.",
    ),
    community_prior_source_path: Path | None = typer.Option(
        None,
        "--community-prior-source-path",
        help="Optional imported community-card-stats artifact directory or jsonl file.",
    ),
    community_route_prior_source_path: Path | None = typer.Option(
        None,
        "--community-route-prior-source-path",
        help="Optional public-run strategic-route-stats artifact directory or jsonl file.",
    ),
    community_reward_pick_weight: float = typer.Option(1.15, "--community-reward-pick-weight", min=0.0),
    community_selection_pick_weight: float = typer.Option(1.05, "--community-selection-pick-weight", min=0.0),
    community_selection_upgrade_weight: float = typer.Option(0.55, "--community-selection-upgrade-weight", min=0.0),
    community_selection_remove_weight: float = typer.Option(0.95, "--community-selection-remove-weight", min=0.0),
    community_shop_buy_weight: float = typer.Option(1.0, "--community-shop-buy-weight", min=0.0),
    community_route_weight: float = typer.Option(0.90, "--community-route-weight", min=0.0),
    community_reward_pick_neutral_rate: float = typer.Option(0.33, "--community-reward-pick-neutral-rate", min=0.0, max=1.0),
    community_shop_buy_neutral_rate: float = typer.Option(0.10, "--community-shop-buy-neutral-rate", min=0.0, max=1.0),
    community_route_neutral_win_rate: float = typer.Option(0.50, "--community-route-neutral-win-rate", min=0.0, max=1.0),
    community_pick_rate_scale: float = typer.Option(3.0, "--community-pick-rate-scale", min=0.0),
    community_buy_rate_scale: float = typer.Option(5.0, "--community-buy-rate-scale", min=0.0),
    community_win_delta_scale: float = typer.Option(12.0, "--community-win-delta-scale", min=0.0),
    community_route_win_rate_scale: float = typer.Option(8.0, "--community-route-win-rate-scale", min=0.0),
    community_min_sample_size: int = typer.Option(40, "--community-min-sample-size", min=1),
    community_route_min_sample_size: int = typer.Option(30, "--community-route-min-sample-size", min=1),
    community_max_confidence_sample_size: int = typer.Option(1200, "--community-max-confidence-sample-size", min=1),
    community_max_source_age_days: int | None = typer.Option(
        None,
        "--community-max-source-age-days",
        min=0,
    ),
    run_mode: str | None = typer.Option(
        None,
        "--run-mode",
        help="Intended in-game run mode contract, for example custom.",
    ),
    game_seed: str | None = typer.Option(
        None,
        "--game-seed",
        help="Expected in-game run seed recorded and validated from live observations.",
    ),
    seed_source: str | None = typer.Option(
        None,
        "--seed-source",
        help="Seed origin label, for example custom_mode_manual.",
    ),
    game_character_id: str | None = typer.Option(
        None,
        "--game-character-id",
        help="Expected character id in the live run contract.",
    ),
    game_ascension: int | None = typer.Option(
        None,
        "--game-ascension",
        min=0,
        help="Expected ascension in the live run contract.",
    ),
    custom_modifier: list[str] | None = typer.Option(
        None,
        "--custom-modifier",
        help="Repeatable Custom Mode modifier recorded in the live run contract.",
    ),
    progress_profile: str | None = typer.Option(
        None,
        "--progress-profile",
        help="Progress/unlock profile label recorded in the live run contract.",
    ),
    benchmark_contract_id: str | None = typer.Option(
        None,
        "--benchmark-contract-id",
        help="Optional benchmark contract id recorded in artifacts.",
    ),
    strict_game_run_contract: bool = typer.Option(
        True,
        "--strict-game-run-contract/--no-strict-game-run-contract",
        help="Stop immediately when observed seed/character/ascension diverge from the configured contract.",
    ),
) -> None:
    resolved = load_instance_config(config)
    specs = build_instance_specs(resolved)
    session_root = output_root / (session_name or default_collection_session_name())
    predictor_config = _build_predictor_runtime_config(
        model_path=predictor_model_path,
        mode=predictor_mode,
        hooks=predictor_hook,
    )
    strategic_model_config = _build_strategic_runtime_config(
        checkpoint_path=strategic_checkpoint_path,
        mode=strategic_mode,
        hooks=strategic_hook,
    )
    community_prior_config = _build_community_prior_runtime_config(
        source_path=community_prior_source_path,
        route_source_path=community_route_prior_source_path,
        reward_pick_weight=community_reward_pick_weight,
        selection_pick_weight=community_selection_pick_weight,
        selection_upgrade_weight=community_selection_upgrade_weight,
        selection_remove_weight=community_selection_remove_weight,
        shop_buy_weight=community_shop_buy_weight,
        route_weight=community_route_weight,
        reward_pick_neutral_rate=community_reward_pick_neutral_rate,
        shop_buy_neutral_rate=community_shop_buy_neutral_rate,
        route_neutral_win_rate=community_route_neutral_win_rate,
        pick_rate_scale=community_pick_rate_scale,
        buy_rate_scale=community_buy_rate_scale,
        win_delta_scale=community_win_delta_scale,
        route_win_rate_scale=community_route_win_rate_scale,
        min_sample_size=community_min_sample_size,
        route_min_sample_size=community_route_min_sample_size,
        max_confidence_sample_size=community_max_confidence_sample_size,
        max_source_age_days=community_max_source_age_days,
    )
    game_run_contract = _build_live_game_run_contract(
        run_mode=run_mode,
        game_seed=game_seed,
        seed_source=seed_source,
        game_character_id=game_character_id,
        game_ascension=game_ascension,
        custom_modifier=custom_modifier,
        progress_profile=progress_profile,
        benchmark_contract_id=benchmark_contract_id,
        strict_game_run_contract=strict_game_run_contract,
    )
    reports = collect_round_robin(
        specs,
        output_root=session_root,
        max_steps_per_instance=max_steps_per_instance,
        max_runs_per_instance=max_runs_per_instance,
        max_combats_per_instance=max_combats_per_instance,
        poll_interval_seconds=poll_interval_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
        policy_profile=policy_pack,
        predictor_config=predictor_config,
        strategic_model_config=strategic_model_config,
        community_prior_config=community_prior_config,
        game_run_contract=game_run_contract,
    )

    table = Table(title=f"Collection Reports ({session_root})")
    table.add_column("Instance")
    table.add_column("Port")
    table.add_column("Steps")
    table.add_column("Runs")
    table.add_column("Combats")
    table.add_column("Stop Reason")
    table.add_column("Last Screen")
    table.add_column("Run ID")
    table.add_column("Output")
    table.add_column("Error")

    for spec, report in zip(specs, reports, strict=True):
        table.add_row(
            report.instance_id,
            str(spec.api_port),
            str(report.step_count),
            str(report.completed_run_count),
            str(report.completed_combat_count),
            report.stop_reason,
            report.last_screen,
            report.last_run_id,
            str(report.output_path),
            report.error or "-",
        )

    console.print(table)


@train_app.command("behavior-cloning")
def train_behavior_cloning_command(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Trajectory dataset directory or steps.jsonl file.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/behavior-cloning"),
        "--output-root",
        help="Root directory for behavior-cloning training outputs.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional behavior-cloning session name. Defaults to a UTC timestamp.",
    ),
    epochs: int = typer.Option(40, "--epochs", min=1),
    learning_rate: float = typer.Option(0.035, "--learning-rate", min=0.000001),
    l2: float = typer.Option(0.0001, "--l2", min=0.0),
    validation_fraction: float = typer.Option(0.15, "--validation-fraction", min=0.0, max=0.9),
    test_fraction: float = typer.Option(0.15, "--test-fraction", min=0.0, max=0.9),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Training RNG seed for dataset shuffling and optimizer initialization. Does not set the in-game run seed.",
    ),
    include_stage: list[str] | None = typer.Option(
        None,
        "--include-stage",
        help="Optional repeatable decision-stage allowlist.",
    ),
    include_decision_source: list[str] | None = typer.Option(
        None,
        "--include-decision-source",
        help="Optional repeatable decision-source allowlist.",
    ),
    include_policy_pack: list[str] | None = typer.Option(
        None,
        "--include-policy-pack",
        help="Optional repeatable policy-pack allowlist.",
    ),
    include_policy_name: list[str] | None = typer.Option(
        None,
        "--include-policy-name",
        help="Optional repeatable policy-name allowlist.",
    ),
    min_floor: int | None = typer.Option(None, "--min-floor"),
    max_floor: int | None = typer.Option(None, "--max-floor"),
    min_legal_actions: int = typer.Option(2, "--min-legal-actions", min=1),
    stage_weight: list[str] | None = typer.Option(
        None,
        "--stage-weight",
        help="Repeatable stage weight in stage=value form.",
    ),
    decision_source_weight: list[str] | None = typer.Option(
        None,
        "--decision-source-weight",
        help="Repeatable decision-source weight in source=value form.",
    ),
    policy_pack_weight: list[str] | None = typer.Option(
        None,
        "--policy-pack-weight",
        help="Repeatable policy-pack weight in pack=value form.",
    ),
    policy_name_weight: list[str] | None = typer.Option(
        None,
        "--policy-name-weight",
        help="Repeatable policy-name weight in name=value form.",
    ),
    run_outcome_weight: list[str] | None = typer.Option(
        None,
        "--run-outcome-weight",
        help="Repeatable run-outcome weight in outcome=value form.",
    ),
    floor_band_weight: list[str] | None = typer.Option(
        None,
        "--floor-band-weight",
        help="Repeatable floor-band weight in min:max:weight form. Use * for open bounds.",
    ),
    top_k: list[int] | None = typer.Option(
        None,
        "--top-k",
        help="Repeatable top-k accuracy metric to report. Defaults to 1 and 3.",
    ),
    live_base_url: str | None = typer.Option(
        None,
        "--live-base-url",
        help="Optional live runtime base URL for post-train rollout evaluation.",
    ),
    live_eval_max_env_steps: int = typer.Option(0, "--live-eval-max-env-steps", min=0),
    live_eval_max_runs: int = typer.Option(1, "--live-eval-max-runs", min=0),
    live_eval_max_combats: int = typer.Option(0, "--live-eval-max-combats", min=0),
    benchmark_manifest: Path | None = typer.Option(
        None,
        "--benchmark-manifest",
        help="Optional benchmark-suite manifest to run against the best BC checkpoint after training.",
    ),
    benchmark_suite_name: str | None = typer.Option(
        None,
        "--benchmark-suite-name",
        help="Optional benchmark-suite output directory name.",
    ),
) -> None:
    session_name = session_name or default_behavior_cloning_training_session_name()
    config = BehaviorCloningTrainConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        include_stages=tuple(include_stage or ()),
        include_decision_sources=tuple(include_decision_source or ()),
        include_policy_packs=tuple(include_policy_pack or ()),
        include_policy_names=tuple(include_policy_name or ()),
        min_floor=min_floor,
        max_floor=max_floor,
        min_legal_actions=min_legal_actions,
        top_k=tuple(top_k or (1, 3)),
        stage_weights=_parse_weight_map(stage_weight, option_name="--stage-weight"),
        decision_source_weights=_parse_weight_map(
            decision_source_weight,
            option_name="--decision-source-weight",
        ),
        policy_pack_weights=_parse_weight_map(policy_pack_weight, option_name="--policy-pack-weight"),
        policy_name_weights=_parse_weight_map(policy_name_weight, option_name="--policy-name-weight"),
        run_outcome_weights=_parse_weight_map(run_outcome_weight, option_name="--run-outcome-weight"),
        floor_band_weights=_parse_floor_band_weights(floor_band_weight),
        benchmark_manifest_path=None if benchmark_manifest is None else benchmark_manifest.expanduser().resolve(),
        live_base_url=live_base_url,
        live_eval_max_env_steps=live_eval_max_env_steps,
        live_eval_max_runs=live_eval_max_runs,
        live_eval_max_combats=live_eval_max_combats,
    )
    report = train_behavior_cloning_policy(
        dataset_source=dataset,
        output_root=output_root,
        session_name=session_name,
        config=config,
        benchmark_suite_name=benchmark_suite_name,
    )

    table = Table(title=f"Behavior Cloning Training ({report.output_dir})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Examples", str(report.example_count))
    table.add_row("Train Examples", str(report.train_example_count))
    table.add_row("Validation Examples", str(report.validation_example_count))
    table.add_row("Test Examples", str(report.test_example_count))
    table.add_row("Split Strategy", report.split_strategy)
    table.add_row("Feature Count", str(report.feature_count))
    table.add_row("Stage Count", str(report.stage_count))
    table.add_row("Best Epoch", str(report.best_epoch))
    table.add_row("Checkpoint Path", str(report.checkpoint_path))
    table.add_row("Best Checkpoint Path", str(report.best_checkpoint_path))
    table.add_row("Metrics Path", str(report.metrics_path))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row(
        "Live Eval Summary",
        str(report.live_eval_summary_path) if report.live_eval_summary_path is not None else "-",
    )
    table.add_row(
        "Benchmark Summary",
        str(report.benchmark_summary_path) if report.benchmark_summary_path is not None else "-",
    )
    console.print(table)


@train_app.command("strategic-pretrain")
def train_strategic_pretrain_command(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Public strategic decision dataset directory or strategic-decisions.jsonl file.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/strategic-pretrain"),
        "--output-root",
        help="Root directory for strategic pretraining outputs.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional strategic-pretraining session name. Defaults to a UTC timestamp.",
    ),
    epochs: int = typer.Option(60, "--epochs", min=1),
    learning_rate: float = typer.Option(0.05, "--learning-rate", min=0.000001),
    l2: float = typer.Option(0.0001, "--l2", min=0.0),
    validation_fraction: float = typer.Option(0.15, "--validation-fraction", min=0.0, max=0.9),
    test_fraction: float = typer.Option(0.15, "--test-fraction", min=0.0, max=0.9),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Training RNG seed for dataset shuffling and optimization.",
    ),
    include_decision_type: list[str] | None = typer.Option(
        None,
        "--include-decision-type",
        help="Optional repeatable decision-type allowlist.",
    ),
    include_support_quality: list[str] | None = typer.Option(
        None,
        "--include-support-quality",
        help="Optional repeatable support-quality allowlist.",
    ),
    include_source_name: list[str] | None = typer.Option(
        None,
        "--include-source-name",
        help="Optional repeatable source-name allowlist.",
    ),
    include_build_id: list[str] | None = typer.Option(
        None,
        "--include-build-id",
        help="Optional repeatable build-id allowlist.",
    ),
    min_floor: int | None = typer.Option(None, "--min-floor"),
    max_floor: int | None = typer.Option(None, "--max-floor"),
    min_confidence: float = typer.Option(0.0, "--min-confidence", min=0.0, max=1.0),
    decision_type_weight: list[str] | None = typer.Option(
        None,
        "--decision-type-weight",
        help="Repeatable decision-type weight in type=value form.",
    ),
    support_quality_weight: list[str] | None = typer.Option(
        None,
        "--support-quality-weight",
        help="Repeatable support-quality weight in quality=value form.",
    ),
    source_name_weight: list[str] | None = typer.Option(
        None,
        "--source-name-weight",
        help="Repeatable source-name weight in source=value form.",
    ),
    build_id_weight: list[str] | None = typer.Option(
        None,
        "--build-id-weight",
        help="Repeatable build-id weight in build=value form.",
    ),
    run_outcome_weight: list[str] | None = typer.Option(
        None,
        "--run-outcome-weight",
        help="Repeatable run-outcome weight in outcome=value form.",
    ),
    confidence_power: float = typer.Option(1.0, "--confidence-power", min=0.0),
    chosen_only_positive_weight: float = typer.Option(0.35, "--chosen-only-positive-weight", min=0.000001),
    auxiliary_value_weight: float = typer.Option(0.75, "--auxiliary-value-weight", min=0.0),
    top_k: list[int] | None = typer.Option(
        None,
        "--top-k",
        help="Repeatable top-k ranking metric to report. Defaults to 1 and 3.",
    ),
) -> None:
    session_name = session_name or default_strategic_pretrain_session_name()
    config = StrategicPretrainTrainConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        include_decision_types=tuple(include_decision_type or ()),
        include_support_qualities=tuple(include_support_quality or ()),
        include_source_names=tuple(include_source_name or ()),
        include_build_ids=tuple(include_build_id or ()),
        min_floor=min_floor,
        max_floor=max_floor,
        min_confidence=min_confidence,
        top_k=tuple(top_k or (1, 3)),
        decision_type_weights=_parse_weight_map(decision_type_weight, option_name="--decision-type-weight"),
        support_quality_weights=_parse_weight_map(support_quality_weight, option_name="--support-quality-weight"),
        source_name_weights=_parse_weight_map(source_name_weight, option_name="--source-name-weight"),
        build_id_weights=_parse_weight_map(build_id_weight, option_name="--build-id-weight"),
        run_outcome_weights=_parse_weight_map(run_outcome_weight, option_name="--run-outcome-weight"),
        confidence_power=confidence_power,
        chosen_only_positive_weight=chosen_only_positive_weight,
        auxiliary_value_weight=auxiliary_value_weight,
    )
    report = train_strategic_pretrain_policy(
        dataset_source=dataset,
        output_root=output_root,
        session_name=session_name,
        config=config,
    )

    table = Table(title=f"Strategic Pretraining ({report.output_dir})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Examples", str(report.example_count))
    table.add_row("Train Examples", str(report.train_example_count))
    table.add_row("Validation Examples", str(report.validation_example_count))
    table.add_row("Test Examples", str(report.test_example_count))
    table.add_row("Split Strategy", report.split_strategy)
    table.add_row("Feature Count", str(report.feature_count))
    table.add_row("Decision Type Count", str(report.decision_type_count))
    table.add_row("Best Epoch", str(report.best_epoch))
    table.add_row("Checkpoint Path", str(report.checkpoint_path))
    table.add_row("Best Checkpoint Path", str(report.best_checkpoint_path))
    table.add_row("Metrics Path", str(report.metrics_path))
    table.add_row("Summary Path", str(report.summary_path))
    console.print(table)


@train_app.command("strategic-finetune")
def train_strategic_finetune_command(
    runtime_dataset: Path = typer.Option(
        ...,
        "--runtime-dataset",
        help="Runtime trajectory-step dataset directory or steps.jsonl file.",
    ),
    public_dataset: Path | None = typer.Option(
        None,
        "--public-dataset",
        help="Optional public strategic decision dataset directory or strategic-decisions.jsonl file.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/strategic-finetune"),
        "--output-root",
        help="Root directory for strategic fine-tuning outputs.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional strategic-finetuning session name. Defaults to a UTC timestamp.",
    ),
    warmstart_checkpoint_path: Path | None = typer.Option(
        None,
        "--warmstart-checkpoint-path",
        help="Optional strategic_pretrain or strategic_finetune checkpoint used for warm-start.",
    ),
    epochs: int = typer.Option(60, "--epochs", min=1),
    learning_rate: float = typer.Option(0.04, "--learning-rate", min=0.000001),
    l2: float = typer.Option(0.0001, "--l2", min=0.0),
    validation_fraction: float = typer.Option(0.15, "--validation-fraction", min=0.0, max=0.9),
    test_fraction: float = typer.Option(0.15, "--test-fraction", min=0.0, max=0.9),
    seed: int = typer.Option(0, "--seed"),
    include_decision_type: list[str] | None = typer.Option(None, "--include-decision-type"),
    include_runtime_decision_source: list[str] | None = typer.Option(None, "--include-runtime-decision-source"),
    include_runtime_policy_pack: list[str] | None = typer.Option(None, "--include-runtime-policy-pack"),
    include_runtime_policy_name: list[str] | None = typer.Option(None, "--include-runtime-policy-name"),
    include_public_support_quality: list[str] | None = typer.Option(None, "--include-public-support-quality"),
    include_public_source_name: list[str] | None = typer.Option(None, "--include-public-source-name"),
    include_public_build_id: list[str] | None = typer.Option(None, "--include-public-build-id"),
    runtime_min_floor: int | None = typer.Option(None, "--runtime-min-floor"),
    runtime_max_floor: int | None = typer.Option(None, "--runtime-max-floor"),
    public_min_floor: int | None = typer.Option(None, "--public-min-floor"),
    public_max_floor: int | None = typer.Option(None, "--public-max-floor"),
    public_min_confidence: float = typer.Option(0.0, "--public-min-confidence", min=0.0, max=1.0),
    decision_type_weight: list[str] | None = typer.Option(None, "--decision-type-weight"),
    source_name_weight: list[str] | None = typer.Option(None, "--source-name-weight"),
    build_id_weight: list[str] | None = typer.Option(None, "--build-id-weight"),
    run_outcome_weight: list[str] | None = typer.Option(None, "--run-outcome-weight"),
    runtime_example_weight: float = typer.Option(1.0, "--runtime-example-weight", min=0.000001),
    public_example_weight: float = typer.Option(1.0, "--public-example-weight", min=0.000001),
    confidence_power: float = typer.Option(1.0, "--confidence-power", min=0.0),
    chosen_only_positive_weight: float = typer.Option(0.35, "--chosen-only-positive-weight", min=0.000001),
    auxiliary_value_weight: float = typer.Option(0.75, "--auxiliary-value-weight", min=0.0),
    schedule: str = typer.Option("weighted_shuffle", "--schedule"),
    runtime_replay_passes: int = typer.Option(1, "--runtime-replay-passes", min=1),
    public_replay_passes: int = typer.Option(1, "--public-replay-passes", min=1),
    freeze_transferred_ranking_epochs: int = typer.Option(0, "--freeze-transferred-ranking-epochs", min=0),
    freeze_transferred_value_epochs: int = typer.Option(0, "--freeze-transferred-value-epochs", min=0),
    enforce_public_build_match: bool = typer.Option(
        True,
        "--enforce-public-build-match/--no-enforce-public-build-match",
    ),
    runtime_source_name: str = typer.Option("local_runtime", "--runtime-source-name"),
    runtime_game_mode: str = typer.Option("standard", "--runtime-game-mode"),
    runtime_platform_type: str = typer.Option("local", "--runtime-platform-type"),
    runtime_build_id: str | None = typer.Option(None, "--runtime-build-id"),
    top_k: list[int] | None = typer.Option(None, "--top-k"),
) -> None:
    session_name = session_name or default_strategic_finetune_session_name()
    config = StrategicFinetuneTrainConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        include_decision_types=tuple(include_decision_type or ()),
        include_runtime_decision_sources=tuple(include_runtime_decision_source or ()),
        include_runtime_policy_packs=tuple(include_runtime_policy_pack or ()),
        include_runtime_policy_names=tuple(include_runtime_policy_name or ()),
        include_public_support_qualities=tuple(include_public_support_quality or ()),
        include_public_source_names=tuple(include_public_source_name or ()),
        include_public_build_ids=tuple(include_public_build_id or ()),
        runtime_min_floor=runtime_min_floor,
        runtime_max_floor=runtime_max_floor,
        public_min_floor=public_min_floor,
        public_max_floor=public_max_floor,
        public_min_confidence=public_min_confidence,
        top_k=tuple(top_k or (1, 3)),
        decision_type_weights=_parse_weight_map(decision_type_weight, option_name="--decision-type-weight"),
        source_name_weights=_parse_weight_map(source_name_weight, option_name="--source-name-weight"),
        build_id_weights=_parse_weight_map(build_id_weight, option_name="--build-id-weight"),
        run_outcome_weights=_parse_weight_map(run_outcome_weight, option_name="--run-outcome-weight"),
        runtime_example_weight=runtime_example_weight,
        public_example_weight=public_example_weight,
        confidence_power=confidence_power,
        chosen_only_positive_weight=chosen_only_positive_weight,
        auxiliary_value_weight=auxiliary_value_weight,
        schedule=schedule.strip().lower(),
        runtime_replay_passes=runtime_replay_passes,
        public_replay_passes=public_replay_passes,
        warmstart_checkpoint_path=None if warmstart_checkpoint_path is None else warmstart_checkpoint_path.expanduser().resolve(),
        freeze_transferred_ranking_epochs=freeze_transferred_ranking_epochs,
        freeze_transferred_value_epochs=freeze_transferred_value_epochs,
        enforce_public_build_match=enforce_public_build_match,
        runtime_source_name=runtime_source_name,
        runtime_game_mode=runtime_game_mode,
        runtime_platform_type=runtime_platform_type,
        runtime_build_id=runtime_build_id,
    )
    report = train_strategic_finetune_policy(
        runtime_dataset_source=runtime_dataset,
        public_dataset_source=public_dataset,
        output_root=output_root,
        session_name=session_name,
        config=config,
    )

    table = Table(title=f"Strategic Finetuning ({report.output_dir})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Examples", str(report.example_count))
    table.add_row("Runtime Examples", str(report.runtime_example_count))
    table.add_row("Public Examples", str(report.public_example_count))
    table.add_row("Train Examples", str(report.train_example_count))
    table.add_row("Validation Examples", str(report.validation_example_count))
    table.add_row("Test Examples", str(report.test_example_count))
    table.add_row("Feature Count", str(report.feature_count))
    table.add_row("Decision Type Count", str(report.decision_type_count))
    table.add_row("Best Epoch", str(report.best_epoch))
    table.add_row("Checkpoint Path", str(report.checkpoint_path))
    table.add_row("Best Checkpoint Path", str(report.best_checkpoint_path))
    table.add_row("Warmstart Checkpoint", str(report.warmstart_checkpoint_path) if report.warmstart_checkpoint_path is not None else "-")
    table.add_row("Metrics Path", str(report.metrics_path))
    table.add_row("Summary Path", str(report.summary_path))
    console.print(table)


@train_app.command("offline-cql")
def train_offline_cql_command(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Offline RL dataset directory or transitions.jsonl file.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/offline-cql"),
        "--output-root",
        help="Root directory for offline CQL training outputs.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional offline CQL session name. Defaults to a UTC timestamp.",
    ),
    epochs: int = typer.Option(40, "--epochs", min=1),
    batch_size: int = typer.Option(32, "--batch-size", min=1),
    learning_rate: float = typer.Option(0.001, "--learning-rate", min=0.000001),
    gamma: float = typer.Option(0.97, "--gamma", min=0.0, max=1.0),
    huber_delta: float = typer.Option(1.0, "--huber-delta", min=0.000001),
    hidden_size: list[int] | None = typer.Option(None, "--hidden-size", min=1),
    l2: float = typer.Option(0.0001, "--l2", min=0.0),
    conservative_alpha: float = typer.Option(1.0, "--conservative-alpha", min=0.0),
    conservative_temperature: float = typer.Option(1.0, "--conservative-temperature", min=0.000001),
    target_sync_interval: int = typer.Option(50, "--target-sync-interval", min=1),
    validation_fraction: float = typer.Option(0.15, "--validation-fraction", min=0.0, max=0.9),
    test_fraction: float = typer.Option(0.15, "--test-fraction", min=0.0, max=0.9),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Training RNG seed for data ordering and optimization. Does not set the in-game run seed.",
    ),
    early_stopping_patience: int = typer.Option(8, "--early-stopping-patience", min=1),
    include_action_space_name: list[str] | None = typer.Option(
        None,
        "--include-action-space-name",
        help="Repeatable action-space allowlist. Defaults to combat_v1.",
    ),
    min_floor: int | None = typer.Option(None, "--min-floor"),
    max_floor: int | None = typer.Option(None, "--max-floor"),
    min_reward: float | None = typer.Option(None, "--min-reward"),
    max_reward: float | None = typer.Option(None, "--max-reward"),
    live_base_url: str | None = typer.Option(
        None,
        "--live-base-url",
        help="Optional live runtime base URL for post-train rollout evaluation.",
    ),
    live_eval_max_env_steps: int = typer.Option(0, "--live-eval-max-env-steps", min=0),
    live_eval_max_runs: int = typer.Option(1, "--live-eval-max-runs", min=0),
    live_eval_max_combats: int = typer.Option(0, "--live-eval-max-combats", min=0),
    benchmark_manifest: Path | None = typer.Option(
        None,
        "--benchmark-manifest",
        help="Optional benchmark-suite manifest to run against the best offline CQL checkpoint after training.",
    ),
    benchmark_suite_name: str | None = typer.Option(
        None,
        "--benchmark-suite-name",
        help="Optional benchmark-suite output directory name.",
    ),
) -> None:
    session_name = session_name or default_offline_cql_training_session_name()
    config = OfflineCqlTrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        huber_delta=huber_delta,
        hidden_sizes=tuple(hidden_size or (64, 64)),
        l2=l2,
        conservative_alpha=conservative_alpha,
        conservative_temperature=conservative_temperature,
        target_sync_interval=target_sync_interval,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        early_stopping_patience=early_stopping_patience,
        include_action_space_names=tuple(include_action_space_name or ("combat_v1",)),
        min_floor=min_floor,
        max_floor=max_floor,
        min_reward=min_reward,
        max_reward=max_reward,
        live_base_url=live_base_url,
        live_eval_max_env_steps=live_eval_max_env_steps,
        live_eval_max_runs=live_eval_max_runs,
        live_eval_max_combats=live_eval_max_combats,
        benchmark_manifest_path=None if benchmark_manifest is None else benchmark_manifest.expanduser().resolve(),
    )
    report = train_offline_cql_policy(
        dataset_source=dataset,
        output_root=output_root,
        session_name=session_name,
        config=config,
        benchmark_suite_name=benchmark_suite_name,
    )

    table = Table(title=f"Offline CQL Training ({report.output_dir})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Examples", str(report.example_count))
    table.add_row("Train Examples", str(report.train_example_count))
    table.add_row("Validation Examples", str(report.validation_example_count))
    table.add_row("Test Examples", str(report.test_example_count))
    table.add_row("Split Strategy", report.split_strategy)
    table.add_row("Feature Count", str(report.feature_count))
    table.add_row("Action Count", str(report.action_count))
    table.add_row("Best Epoch", str(report.best_epoch))
    table.add_row("Checkpoint Path", str(report.checkpoint_path))
    table.add_row("Best Checkpoint Path", str(report.best_checkpoint_path))
    table.add_row("Warm Start Path", str(report.warmstart_checkpoint_path))
    table.add_row("Metrics Path", str(report.metrics_path))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Live Eval Summary", str(report.live_eval_summary_path) if report.live_eval_summary_path is not None else "-")
    table.add_row("Benchmark Summary", str(report.benchmark_summary_path) if report.benchmark_summary_path is not None else "-")
    console.print(table)


@train_app.command("combat-dqn")
def train_combat_dqn_command(
    base_url: str = typer.Option("http://127.0.0.1:8080", "--base-url"),
    output_root: Path = typer.Option(
        Path("artifacts/training"),
        "--output-root",
        help="Root directory for training logs and checkpoints.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional session directory name. Defaults to a UTC timestamp.",
    ),
    max_env_steps: int = typer.Option(
        0,
        "--max-env-steps",
        min=0,
        help="Hard environment-step budget. Use 0 to disable the step budget.",
    ),
    max_runs: int = typer.Option(
        1,
        "--max-runs",
        "--max-episodes",
        min=0,
        help="Completed-run budget. Use 0 to disable the run budget.",
    ),
    max_combats: int = typer.Option(
        0,
        "--max-combats",
        min=0,
        help="Completed-combat budget. Use 0 to disable the combat budget.",
    ),
    poll_interval_seconds: float = typer.Option(0.25, "--poll-interval-seconds", min=0.01),
    max_idle_polls: int = typer.Option(40, "--max-idle-polls", min=1),
    request_timeout_seconds: float = typer.Option(30.0, "--request-timeout-seconds", min=1.0),
    resume_from: Path | None = typer.Option(
        None,
        "--resume-from",
        help="Optional checkpoint path to continue training from.",
    ),
    checkpoint_every_rl_steps: int = typer.Option(
        25,
        "--checkpoint-every-rl-steps",
        min=0,
        help="Save a latest checkpoint and a numbered snapshot every N RL steps. Use 0 to disable periodic snapshots.",
    ),
    learning_rate: float | None = typer.Option(None, "--learning-rate", min=0.000001),
    gamma: float | None = typer.Option(None, "--gamma", min=0.0, max=1.0),
    epsilon_start: float | None = typer.Option(None, "--epsilon-start", min=0.0, max=1.0),
    epsilon_end: float | None = typer.Option(None, "--epsilon-end", min=0.0, max=1.0),
    epsilon_decay_steps: int | None = typer.Option(None, "--epsilon-decay-steps", min=1),
    replay_capacity: int = typer.Option(4096, "--replay-capacity", min=64),
    batch_size: int = typer.Option(32, "--batch-size", min=1),
    min_replay_size: int = typer.Option(64, "--min-replay-size", min=1),
    target_sync_interval: int = typer.Option(50, "--target-sync-interval", min=1),
    updates_per_env_step: int = typer.Option(1, "--updates-per-env-step", min=1),
    huber_delta: float = typer.Option(1.0, "--huber-delta", min=0.0001),
    double_dqn: bool = typer.Option(True, "--double-dqn/--no-double-dqn"),
    n_step: int = typer.Option(3, "--n-step", min=1),
    prioritized_replay: bool = typer.Option(True, "--prioritized-replay/--no-prioritized-replay"),
    priority_alpha: float = typer.Option(0.6, "--priority-alpha", min=0.0),
    priority_beta_start: float = typer.Option(0.4, "--priority-beta-start", min=0.0),
    priority_beta_end: float = typer.Option(1.0, "--priority-beta-end", min=0.0),
    priority_beta_decay_steps: int = typer.Option(10000, "--priority-beta-decay-steps", min=1),
    priority_epsilon: float = typer.Option(0.0001, "--priority-epsilon", min=0.000001),
    hidden_sizes: list[int] = typer.Option([64, 64], "--hidden-size", min=8),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Agent RNG seed override for DQN exploration and replay sampling. Does not set the in-game run seed.",
    ),
    policy_pack: str = typer.Option(
        "baseline",
        "--policy-pack",
        help="Heuristic policy pack for non-combat control: baseline, legacy, planner, planner_assist, or conservative.",
    ),
    predictor_model_path: Path | None = typer.Option(
        None,
        "--predictor-model-path",
        help="Optional predictor model path for runtime-guided non-combat scoring.",
    ),
    predictor_mode: str = typer.Option(
        "heuristic_only",
        "--predictor-mode",
        help="Predictor guidance mode: heuristic_only, assist, or dominant.",
    ),
    predictor_hook: list[str] | None = typer.Option(
        None,
        "--predictor-hook",
        help="Repeatable predictor hook filter. Defaults to all supported hooks when predictor is enabled.",
    ),
    community_prior_source_path: Path | None = typer.Option(
        None,
        "--community-prior-source-path",
        help="Optional imported community-card-stats artifact directory or jsonl file.",
    ),
    community_route_prior_source_path: Path | None = typer.Option(
        None,
        "--community-route-prior-source-path",
        help="Optional public-run strategic-route-stats artifact directory or jsonl file.",
    ),
    community_reward_pick_weight: float = typer.Option(1.15, "--community-reward-pick-weight", min=0.0),
    community_selection_pick_weight: float = typer.Option(1.05, "--community-selection-pick-weight", min=0.0),
    community_selection_upgrade_weight: float = typer.Option(0.55, "--community-selection-upgrade-weight", min=0.0),
    community_selection_remove_weight: float = typer.Option(0.95, "--community-selection-remove-weight", min=0.0),
    community_shop_buy_weight: float = typer.Option(1.0, "--community-shop-buy-weight", min=0.0),
    community_route_weight: float = typer.Option(0.90, "--community-route-weight", min=0.0),
    community_reward_pick_neutral_rate: float = typer.Option(0.33, "--community-reward-pick-neutral-rate", min=0.0, max=1.0),
    community_shop_buy_neutral_rate: float = typer.Option(0.10, "--community-shop-buy-neutral-rate", min=0.0, max=1.0),
    community_route_neutral_win_rate: float = typer.Option(0.50, "--community-route-neutral-win-rate", min=0.0, max=1.0),
    community_pick_rate_scale: float = typer.Option(3.0, "--community-pick-rate-scale", min=0.0),
    community_buy_rate_scale: float = typer.Option(5.0, "--community-buy-rate-scale", min=0.0),
    community_win_delta_scale: float = typer.Option(12.0, "--community-win-delta-scale", min=0.0),
    community_route_win_rate_scale: float = typer.Option(8.0, "--community-route-win-rate-scale", min=0.0),
    community_min_sample_size: int = typer.Option(40, "--community-min-sample-size", min=1),
    community_route_min_sample_size: int = typer.Option(30, "--community-route-min-sample-size", min=1),
    community_max_confidence_sample_size: int = typer.Option(1200, "--community-max-confidence-sample-size", min=1),
    community_max_source_age_days: int | None = typer.Option(None, "--community-max-source-age-days", min=0),
    run_mode: str | None = typer.Option(None, "--run-mode", help="Intended in-game run mode contract, for example custom."),
    game_seed: str | None = typer.Option(None, "--game-seed", help="Expected in-game run seed recorded and validated from live observations."),
    seed_source: str | None = typer.Option(None, "--seed-source", help="Seed origin label, for example custom_mode_manual."),
    game_character_id: str | None = typer.Option(None, "--game-character-id", help="Expected character id in the live run contract."),
    game_ascension: int | None = typer.Option(None, "--game-ascension", min=0, help="Expected ascension in the live run contract."),
    custom_modifier: list[str] | None = typer.Option(None, "--custom-modifier", help="Repeatable Custom Mode modifier recorded in the live run contract."),
    progress_profile: str | None = typer.Option(None, "--progress-profile", help="Progress/unlock profile label recorded in the live run contract."),
    benchmark_contract_id: str | None = typer.Option(None, "--benchmark-contract-id", help="Optional benchmark contract id recorded in artifacts."),
    strict_game_run_contract: bool = typer.Option(True, "--strict-game-run-contract/--no-strict-game-run-contract", help="Stop immediately when observed seed/character/ascension diverge from the configured contract."),
) -> None:
    dqn_config: DqnConfig | None = None
    if resume_from is None:
        dqn_config = _build_dqn_config(
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            target_sync_interval=target_sync_interval,
            updates_per_env_step=updates_per_env_step,
            huber_delta=huber_delta,
            hidden_sizes=hidden_sizes,
            seed=seed,
            double_dqn=double_dqn,
            n_step=n_step,
            prioritized_replay=prioritized_replay,
            priority_alpha=priority_alpha,
            priority_beta_start=priority_beta_start,
            priority_beta_end=priority_beta_end,
            priority_beta_decay_steps=priority_beta_decay_steps,
            priority_epsilon=priority_epsilon,
        )

    predictor_config = _build_predictor_runtime_config(
        model_path=predictor_model_path,
        mode=predictor_mode,
        hooks=predictor_hook,
    )
    community_prior_config = _build_community_prior_runtime_config(
        source_path=community_prior_source_path,
        route_source_path=community_route_prior_source_path,
        reward_pick_weight=community_reward_pick_weight,
        selection_pick_weight=community_selection_pick_weight,
        selection_upgrade_weight=community_selection_upgrade_weight,
        selection_remove_weight=community_selection_remove_weight,
        shop_buy_weight=community_shop_buy_weight,
        route_weight=community_route_weight,
        reward_pick_neutral_rate=community_reward_pick_neutral_rate,
        shop_buy_neutral_rate=community_shop_buy_neutral_rate,
        route_neutral_win_rate=community_route_neutral_win_rate,
        pick_rate_scale=community_pick_rate_scale,
        buy_rate_scale=community_buy_rate_scale,
        win_delta_scale=community_win_delta_scale,
        route_win_rate_scale=community_route_win_rate_scale,
        min_sample_size=community_min_sample_size,
        route_min_sample_size=community_route_min_sample_size,
        max_confidence_sample_size=community_max_confidence_sample_size,
        max_source_age_days=community_max_source_age_days,
    )
    game_run_contract = _build_live_game_run_contract(
        run_mode=run_mode,
        game_seed=game_seed,
        seed_source=seed_source,
        game_character_id=game_character_id,
        game_ascension=game_ascension,
        custom_modifier=custom_modifier,
        progress_profile=progress_profile,
        benchmark_contract_id=benchmark_contract_id,
        strict_game_run_contract=strict_game_run_contract,
    )
    report = run_combat_dqn_training(
        base_url=base_url,
        output_root=output_root,
        session_name=session_name,
        max_env_steps=max_env_steps,
        max_runs=max_runs,
        max_combats=max_combats,
        poll_interval_seconds=poll_interval_seconds,
        max_idle_polls=max_idle_polls,
        dqn_config=dqn_config,
        resume_from=resume_from,
        learning_rate_override=learning_rate,
        gamma_override=gamma,
        epsilon_start_override=epsilon_start,
        epsilon_end_override=epsilon_end,
        epsilon_decay_steps_override=epsilon_decay_steps,
        replay_capacity_override=replay_capacity,
        batch_size_override=batch_size,
        min_replay_size_override=min_replay_size,
        target_sync_interval_override=target_sync_interval,
        updates_per_env_step_override=updates_per_env_step,
        huber_delta_override=huber_delta,
        seed_override=seed,
        double_dqn_override=double_dqn,
        n_step_override=n_step,
        prioritized_replay_override=prioritized_replay,
        priority_alpha_override=priority_alpha,
        priority_beta_start_override=priority_beta_start,
        priority_beta_end_override=priority_beta_end,
        priority_beta_decay_steps_override=priority_beta_decay_steps,
        priority_epsilon_override=priority_epsilon,
        checkpoint_every_rl_steps=checkpoint_every_rl_steps,
        request_timeout_seconds=request_timeout_seconds,
        policy_profile=policy_pack,
        predictor_config=predictor_config,
        community_prior_config=community_prior_config,
        game_run_contract=game_run_contract,
    )

    table = Table(title=f"Combat DQN Training ({report.base_url})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Env Steps", str(report.env_steps))
    table.add_row("RL Steps", str(report.rl_steps))
    table.add_row("Heuristic Steps", str(report.heuristic_steps))
    table.add_row("Update Steps", str(report.update_steps))
    table.add_row("Completed Runs", str(report.completed_run_count))
    table.add_row("Completed Combats", str(report.completed_combat_count))
    table.add_row("Stop Reason", report.stop_reason)
    table.add_row("Total Reward", f"{report.total_reward:.3f}")
    table.add_row("Final Screen", report.final_screen)
    table.add_row("Final Run ID", report.final_run_id)
    table.add_row("Log Path", str(report.log_path))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Combat Outcomes", str(report.combat_outcomes_path))
    table.add_row("Latest Checkpoint", str(report.checkpoint_path))
    table.add_row("Best Checkpoint", str(report.best_checkpoint_path) if report.best_checkpoint_path is not None else "-")
    table.add_row("Periodic Checkpoints", str(report.periodic_checkpoint_count))
    table.add_row("Mean Loss", _format_metric(report.learning_metrics.get("mean_loss")))
    table.add_row("Mean TD Error", _format_metric(report.learning_metrics.get("mean_abs_td_error")))
    table.add_row("Final Epsilon", _format_metric(report.learning_metrics.get("final_epsilon")))
    table.add_row("Priority Beta", _format_metric(report.learning_metrics.get("final_priority_beta")))
    table.add_row("Replay Utilization", _format_metric(report.replay_metrics.get("utilization")))
    console.print(table)


@train_app.command("combat-dqn-schedule")
def train_combat_dqn_schedule_command(
    base_url: str = typer.Option("http://127.0.0.1:8080", "--base-url"),
    output_root: Path = typer.Option(
        Path("artifacts/training-schedules"),
        "--output-root",
        help="Root directory for schedule runs.",
    ),
    schedule_name: str | None = typer.Option(
        None,
        "--schedule-name",
        help="Optional schedule directory name. Defaults to a UTC timestamp.",
    ),
    max_sessions: int = typer.Option(3, "--max-sessions", min=1),
    session_max_env_steps: int = typer.Option(
        0,
        "--session-max-env-steps",
        min=0,
        help="Hard per-session environment-step budget. Use 0 to disable the step budget.",
    ),
    session_max_runs: int = typer.Option(
        1,
        "--session-max-runs",
        "--session-max-episodes",
        min=0,
        help="Completed-run budget per session. Use 0 to disable the run budget.",
    ),
    session_max_combats: int = typer.Option(
        0,
        "--session-max-combats",
        min=0,
        help="Completed-combat budget per session. Use 0 to disable the combat budget.",
    ),
    checkpoint_source: str = typer.Option(
        "latest",
        "--checkpoint-source",
        help="Which checkpoint each session should hand to the next one: latest, best, or best_eval.",
    ),
    initial_resume_from: Path | None = typer.Option(
        None,
        "--initial-resume-from",
        help="Optional checkpoint path for the first session.",
    ),
    poll_interval_seconds: float = typer.Option(0.25, "--poll-interval-seconds", min=0.01),
    max_idle_polls: int = typer.Option(40, "--max-idle-polls", min=1),
    checkpoint_every_rl_steps: int = typer.Option(25, "--checkpoint-every-rl-steps", min=0),
    request_timeout_seconds: float = typer.Option(30.0, "--request-timeout-seconds", min=1.0),
    best_eval_repeats: int = typer.Option(3, "--best-eval-repeats", min=1),
    best_eval_max_env_steps: int = typer.Option(
        0,
        "--best-eval-max-env-steps",
        min=0,
        help="Hard environment-step budget for latest-vs-best evaluation when checkpoint-source=best_eval.",
    ),
    best_eval_max_runs: int = typer.Option(
        1,
        "--best-eval-max-runs",
        "--best-eval-max-episodes",
        min=0,
        help="Completed-run budget for latest-vs-best evaluation when checkpoint-source=best_eval.",
    ),
    best_eval_max_combats: int = typer.Option(
        0,
        "--best-eval-max-combats",
        min=0,
        help="Completed-combat budget for latest-vs-best evaluation when checkpoint-source=best_eval.",
    ),
    best_eval_prepare_target: str = typer.Option(
        "main_menu",
        "--best-eval-prepare-target",
        help="Normalization target for latest-vs-best evaluation: main_menu or character_select.",
    ),
    best_eval_prepare_max_steps: int = typer.Option(8, "--best-eval-prepare-max-steps", min=1),
    best_eval_prepare_max_idle_polls: int = typer.Option(40, "--best-eval-prepare-max-idle-polls", min=1),
    best_eval_fallback: str = typer.Option(
        "latest",
        "--best-eval-fallback",
        help="Checkpoint to prefer when best-eval ties or comparison fails: latest or best.",
    ),
    learning_rate: float | None = typer.Option(None, "--learning-rate", min=0.000001),
    gamma: float | None = typer.Option(None, "--gamma", min=0.0, max=1.0),
    epsilon_start: float | None = typer.Option(None, "--epsilon-start", min=0.0, max=1.0),
    epsilon_end: float | None = typer.Option(None, "--epsilon-end", min=0.0, max=1.0),
    epsilon_decay_steps: int | None = typer.Option(None, "--epsilon-decay-steps", min=1),
    replay_capacity: int = typer.Option(4096, "--replay-capacity", min=64),
    batch_size: int = typer.Option(32, "--batch-size", min=1),
    min_replay_size: int = typer.Option(64, "--min-replay-size", min=1),
    target_sync_interval: int = typer.Option(50, "--target-sync-interval", min=1),
    updates_per_env_step: int = typer.Option(1, "--updates-per-env-step", min=1),
    huber_delta: float = typer.Option(1.0, "--huber-delta", min=0.0001),
    double_dqn: bool = typer.Option(True, "--double-dqn/--no-double-dqn"),
    n_step: int = typer.Option(3, "--n-step", min=1),
    prioritized_replay: bool = typer.Option(True, "--prioritized-replay/--no-prioritized-replay"),
    priority_alpha: float = typer.Option(0.6, "--priority-alpha", min=0.0),
    priority_beta_start: float = typer.Option(0.4, "--priority-beta-start", min=0.0),
    priority_beta_end: float = typer.Option(1.0, "--priority-beta-end", min=0.0),
    priority_beta_decay_steps: int = typer.Option(10000, "--priority-beta-decay-steps", min=1),
    priority_epsilon: float = typer.Option(0.0001, "--priority-epsilon", min=0.000001),
    hidden_sizes: list[int] = typer.Option([64, 64], "--hidden-size", min=8),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Agent RNG seed override for DQN schedule sessions. Does not set the in-game run seed.",
    ),
) -> None:
    checkpoint_source = checkpoint_source.lower()
    if checkpoint_source not in {"latest", "best", "best_eval"}:
        raise typer.BadParameter("checkpoint-source must be one of: latest, best, best_eval.")

    best_eval_prepare_target = _resolve_runtime_target_option(best_eval_prepare_target)
    best_eval_fallback = best_eval_fallback.lower()
    if best_eval_fallback not in {"latest", "best"}:
        raise typer.BadParameter("best-eval-fallback must be either 'latest' or 'best'.")

    dqn_config: DqnConfig | None = None
    if initial_resume_from is None:
        dqn_config = _build_dqn_config(
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            target_sync_interval=target_sync_interval,
            updates_per_env_step=updates_per_env_step,
            huber_delta=huber_delta,
            hidden_sizes=hidden_sizes,
            seed=seed,
            double_dqn=double_dqn,
            n_step=n_step,
            prioritized_replay=prioritized_replay,
            priority_alpha=priority_alpha,
            priority_beta_start=priority_beta_start,
            priority_beta_end=priority_beta_end,
            priority_beta_decay_steps=priority_beta_decay_steps,
            priority_epsilon=priority_epsilon,
        )

    report = run_combat_dqn_schedule(
        base_url=base_url,
        output_root=output_root,
        schedule_name=schedule_name,
        max_sessions=max_sessions,
        session_max_env_steps=session_max_env_steps,
        session_max_runs=session_max_runs,
        session_max_combats=session_max_combats,
        checkpoint_source=checkpoint_source,
        initial_resume_from=initial_resume_from,
        poll_interval_seconds=poll_interval_seconds,
        max_idle_polls=max_idle_polls,
        checkpoint_every_rl_steps=checkpoint_every_rl_steps,
        request_timeout_seconds=request_timeout_seconds,
        best_eval_repeats=best_eval_repeats,
        best_eval_max_env_steps=best_eval_max_env_steps,
        best_eval_max_runs=best_eval_max_runs,
        best_eval_max_combats=best_eval_max_combats,
        best_eval_prepare_target=best_eval_prepare_target,
        best_eval_prepare_max_steps=best_eval_prepare_max_steps,
        best_eval_prepare_max_idle_polls=best_eval_prepare_max_idle_polls,
        best_eval_fallback=best_eval_fallback,
        dqn_config=dqn_config,
        learning_rate_override=learning_rate,
        gamma_override=gamma,
        epsilon_start_override=epsilon_start,
        epsilon_end_override=epsilon_end,
        epsilon_decay_steps_override=epsilon_decay_steps,
        replay_capacity_override=replay_capacity,
        batch_size_override=batch_size,
        min_replay_size_override=min_replay_size,
        target_sync_interval_override=target_sync_interval,
        updates_per_env_step_override=updates_per_env_step,
        huber_delta_override=huber_delta,
        seed_override=seed,
        double_dqn_override=double_dqn,
        n_step_override=n_step,
        prioritized_replay_override=prioritized_replay,
        priority_alpha_override=priority_alpha,
        priority_beta_start_override=priority_beta_start,
        priority_beta_end_override=priority_beta_end,
        priority_beta_decay_steps_override=priority_beta_decay_steps,
        priority_epsilon_override=priority_epsilon,
    )

    table = Table(title=f"Combat DQN Schedule ({report.base_url})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Sessions", str(report.session_count))
    table.add_row("Total Env Steps", str(report.total_env_steps))
    table.add_row("Total RL Steps", str(report.total_rl_steps))
    table.add_row("Total Reward", f"{report.total_reward:.3f}")
    table.add_row("Checkpoint Source", report.checkpoint_source)
    table.add_row(
        "Promotion Artifacts",
        str(report.promotion_artifacts_root) if report.promotion_artifacts_root is not None else "-",
    )
    table.add_row("Final Checkpoint", str(report.final_checkpoint_path) if report.final_checkpoint_path is not None else "-")
    table.add_row("Schedule Dir", str(report.schedule_dir))
    table.add_row("Summary", str(report.summary_path))
    table.add_row("Log", str(report.log_path))
    console.print(table)


@eval_app.command("behavior-cloning")
def eval_behavior_cloning_command(
    base_url: str = typer.Option("http://127.0.0.1:8080", "--base-url"),
    checkpoint_path: Path = typer.Option(
        ...,
        "--checkpoint-path",
        help="Checkpoint generated by train behavior-cloning.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/eval"),
        "--output-root",
        help="Root directory for evaluation logs.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional session directory name. Defaults to a UTC timestamp.",
    ),
    max_env_steps: int = typer.Option(0, "--max-env-steps", min=0),
    max_runs: int = typer.Option(1, "--max-runs", "--max-episodes", min=0),
    max_combats: int = typer.Option(0, "--max-combats", min=0),
    poll_interval_seconds: float = typer.Option(0.25, "--poll-interval-seconds", min=0.01),
    max_idle_polls: int = typer.Option(40, "--max-idle-polls", min=1),
    request_timeout_seconds: float = typer.Option(30.0, "--request-timeout-seconds", min=1.0),
    run_mode: str | None = typer.Option(None, "--run-mode", help="Intended in-game run mode contract, for example custom."),
    game_seed: str | None = typer.Option(None, "--game-seed", help="Expected in-game run seed recorded and validated from live observations."),
    seed_source: str | None = typer.Option(None, "--seed-source", help="Seed origin label, for example custom_mode_manual."),
    game_character_id: str | None = typer.Option(None, "--game-character-id", help="Expected character id in the live run contract."),
    game_ascension: int | None = typer.Option(None, "--game-ascension", min=0, help="Expected ascension in the live run contract."),
    custom_modifier: list[str] | None = typer.Option(None, "--custom-modifier", help="Repeatable Custom Mode modifier recorded in the live run contract."),
    progress_profile: str | None = typer.Option(None, "--progress-profile", help="Progress/unlock profile label recorded in the live run contract."),
    benchmark_contract_id: str | None = typer.Option(None, "--benchmark-contract-id", help="Optional benchmark contract id recorded in artifacts."),
    strict_game_run_contract: bool = typer.Option(True, "--strict-game-run-contract/--no-strict-game-run-contract", help="Stop immediately when observed seed/character/ascension diverge from the configured contract."),
) -> None:
    game_run_contract = _build_live_game_run_contract(
        run_mode=run_mode,
        game_seed=game_seed,
        seed_source=seed_source,
        game_character_id=game_character_id,
        game_ascension=game_ascension,
        custom_modifier=custom_modifier,
        progress_profile=progress_profile,
        benchmark_contract_id=benchmark_contract_id,
        strict_game_run_contract=strict_game_run_contract,
    )
    report = run_behavior_cloning_evaluation(
        base_url=base_url,
        checkpoint_path=checkpoint_path,
        output_root=output_root,
        session_name=session_name,
        max_env_steps=max_env_steps,
        max_runs=max_runs,
        max_combats=max_combats,
        poll_interval_seconds=poll_interval_seconds,
        max_idle_polls=max_idle_polls,
        request_timeout_seconds=request_timeout_seconds,
        game_run_contract=game_run_contract,
    )

    table = Table(title=f"Behavior Cloning Evaluation ({report.base_url})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Env Steps", str(report.env_steps))
    table.add_row("Combat Steps", str(report.combat_steps))
    table.add_row("Policy Steps", str(report.heuristic_steps))
    table.add_row("Completed Runs", str(report.completed_run_count))
    table.add_row("Completed Combats", str(report.completed_combat_count))
    table.add_row("Stop Reason", report.stop_reason)
    table.add_row("Total Reward", f"{report.total_reward:.3f}")
    table.add_row("Final Screen", report.final_screen)
    table.add_row("Final Run ID", report.final_run_id)
    table.add_row("Log Path", str(report.log_path))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Combat Outcomes", str(report.combat_outcomes_path))
    table.add_row("Combat Win Rate", _format_metric(report.combat_performance.get("combat_win_rate")))
    table.add_row("Reward / Combat", _format_metric(report.combat_performance.get("reward_per_combat")))
    console.print(table)


@eval_app.command("offline-cql")
def eval_offline_cql_command(
    base_url: str = typer.Option("http://127.0.0.1:8080", "--base-url"),
    checkpoint_path: Path = typer.Option(
        ...,
        "--checkpoint-path",
        help="Checkpoint generated by train offline-cql.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/eval"),
        "--output-root",
        help="Root directory for evaluation logs.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional session directory name. Defaults to a UTC timestamp.",
    ),
    max_env_steps: int = typer.Option(0, "--max-env-steps", min=0),
    max_runs: int = typer.Option(1, "--max-runs", "--max-episodes", min=0),
    max_combats: int = typer.Option(0, "--max-combats", min=0),
    poll_interval_seconds: float = typer.Option(0.25, "--poll-interval-seconds", min=0.01),
    max_idle_polls: int = typer.Option(40, "--max-idle-polls", min=1),
    request_timeout_seconds: float = typer.Option(30.0, "--request-timeout-seconds", min=1.0),
    community_prior_source_path: Path | None = typer.Option(
        None,
        "--community-prior-source-path",
        help="Optional imported community-card-stats artifact directory or jsonl file.",
    ),
    community_route_prior_source_path: Path | None = typer.Option(
        None,
        "--community-route-prior-source-path",
        help="Optional public-run strategic-route-stats artifact directory or jsonl file.",
    ),
    community_reward_pick_weight: float = typer.Option(1.15, "--community-reward-pick-weight", min=0.0),
    community_selection_pick_weight: float = typer.Option(1.05, "--community-selection-pick-weight", min=0.0),
    community_selection_upgrade_weight: float = typer.Option(0.55, "--community-selection-upgrade-weight", min=0.0),
    community_selection_remove_weight: float = typer.Option(0.95, "--community-selection-remove-weight", min=0.0),
    community_shop_buy_weight: float = typer.Option(1.0, "--community-shop-buy-weight", min=0.0),
    community_route_weight: float = typer.Option(0.90, "--community-route-weight", min=0.0),
    community_reward_pick_neutral_rate: float = typer.Option(0.33, "--community-reward-pick-neutral-rate", min=0.0, max=1.0),
    community_shop_buy_neutral_rate: float = typer.Option(0.10, "--community-shop-buy-neutral-rate", min=0.0, max=1.0),
    community_route_neutral_win_rate: float = typer.Option(0.50, "--community-route-neutral-win-rate", min=0.0, max=1.0),
    community_pick_rate_scale: float = typer.Option(3.0, "--community-pick-rate-scale", min=0.0),
    community_buy_rate_scale: float = typer.Option(5.0, "--community-buy-rate-scale", min=0.0),
    community_win_delta_scale: float = typer.Option(12.0, "--community-win-delta-scale", min=0.0),
    community_route_win_rate_scale: float = typer.Option(8.0, "--community-route-win-rate-scale", min=0.0),
    community_min_sample_size: int = typer.Option(40, "--community-min-sample-size", min=1),
    community_route_min_sample_size: int = typer.Option(30, "--community-route-min-sample-size", min=1),
    community_max_confidence_sample_size: int = typer.Option(1200, "--community-max-confidence-sample-size", min=1),
    community_max_source_age_days: int | None = typer.Option(None, "--community-max-source-age-days", min=0),
    run_mode: str | None = typer.Option(None, "--run-mode", help="Intended in-game run mode contract, for example custom."),
    game_seed: str | None = typer.Option(None, "--game-seed", help="Expected in-game run seed recorded and validated from live observations."),
    seed_source: str | None = typer.Option(None, "--seed-source", help="Seed origin label, for example custom_mode_manual."),
    game_character_id: str | None = typer.Option(None, "--game-character-id", help="Expected character id in the live run contract."),
    game_ascension: int | None = typer.Option(None, "--game-ascension", min=0, help="Expected ascension in the live run contract."),
    custom_modifier: list[str] | None = typer.Option(None, "--custom-modifier", help="Repeatable Custom Mode modifier recorded in the live run contract."),
    progress_profile: str | None = typer.Option(None, "--progress-profile", help="Progress/unlock profile label recorded in the live run contract."),
    benchmark_contract_id: str | None = typer.Option(None, "--benchmark-contract-id", help="Optional benchmark contract id recorded in artifacts."),
    strict_game_run_contract: bool = typer.Option(True, "--strict-game-run-contract/--no-strict-game-run-contract", help="Stop immediately when observed seed/character/ascension diverge from the configured contract."),
) -> None:
    game_run_contract = _build_live_game_run_contract(
        run_mode=run_mode,
        game_seed=game_seed,
        seed_source=seed_source,
        game_character_id=game_character_id,
        game_ascension=game_ascension,
        custom_modifier=custom_modifier,
        progress_profile=progress_profile,
        benchmark_contract_id=benchmark_contract_id,
        strict_game_run_contract=strict_game_run_contract,
    )
    community_prior_config = _build_community_prior_runtime_config(
        source_path=community_prior_source_path,
        route_source_path=community_route_prior_source_path,
        reward_pick_weight=community_reward_pick_weight,
        selection_pick_weight=community_selection_pick_weight,
        selection_upgrade_weight=community_selection_upgrade_weight,
        selection_remove_weight=community_selection_remove_weight,
        shop_buy_weight=community_shop_buy_weight,
        route_weight=community_route_weight,
        reward_pick_neutral_rate=community_reward_pick_neutral_rate,
        shop_buy_neutral_rate=community_shop_buy_neutral_rate,
        route_neutral_win_rate=community_route_neutral_win_rate,
        pick_rate_scale=community_pick_rate_scale,
        buy_rate_scale=community_buy_rate_scale,
        win_delta_scale=community_win_delta_scale,
        route_win_rate_scale=community_route_win_rate_scale,
        min_sample_size=community_min_sample_size,
        route_min_sample_size=community_route_min_sample_size,
        max_confidence_sample_size=community_max_confidence_sample_size,
        max_source_age_days=community_max_source_age_days,
    )
    report = run_offline_cql_evaluation(
        base_url=base_url,
        checkpoint_path=checkpoint_path,
        output_root=output_root,
        session_name=session_name,
        max_env_steps=max_env_steps,
        max_runs=max_runs,
        max_combats=max_combats,
        poll_interval_seconds=poll_interval_seconds,
        max_idle_polls=max_idle_polls,
        request_timeout_seconds=request_timeout_seconds,
        community_prior_config=community_prior_config,
        game_run_contract=game_run_contract,
    )

    table = Table(title=f"Offline CQL Evaluation ({report.base_url})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Env Steps", str(report.env_steps))
    table.add_row("Combat Steps", str(report.combat_steps))
    table.add_row("Policy Steps", str(report.heuristic_steps))
    table.add_row("Completed Runs", str(report.completed_run_count))
    table.add_row("Completed Combats", str(report.completed_combat_count))
    table.add_row("Stop Reason", report.stop_reason)
    table.add_row("Total Reward", f"{report.total_reward:.3f}")
    table.add_row("Final Screen", report.final_screen)
    table.add_row("Final Run ID", report.final_run_id)
    table.add_row("Log Path", str(report.log_path))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Combat Outcomes", str(report.combat_outcomes_path))
    table.add_row("Combat Win Rate", _format_metric(report.combat_performance.get("combat_win_rate")))
    table.add_row("Reward / Combat", _format_metric(report.combat_performance.get("reward_per_combat")))
    console.print(table)


@eval_app.command("checkpoint-compare")
def eval_checkpoint_compare_command(
    base_url: str = typer.Option("http://127.0.0.1:8080", "--base-url"),
    baseline_checkpoint_path: Path = typer.Option(..., "--baseline-checkpoint-path"),
    candidate_checkpoint_path: Path = typer.Option(..., "--candidate-checkpoint-path"),
    output_root: Path = typer.Option(Path("artifacts/eval-compare"), "--output-root"),
    comparison_name: str | None = typer.Option(None, "--comparison-name"),
    repeats: int = typer.Option(3, "--repeats", min=1),
    max_env_steps: int = typer.Option(0, "--max-env-steps", min=0),
    max_runs: int = typer.Option(1, "--max-runs", "--max-episodes", min=0),
    max_combats: int = typer.Option(0, "--max-combats", min=0),
    poll_interval_seconds: float = typer.Option(0.25, "--poll-interval-seconds", min=0.01),
    max_idle_polls: int = typer.Option(40, "--max-idle-polls", min=1),
    request_timeout_seconds: float = typer.Option(30.0, "--request-timeout-seconds", min=1.0),
    prepare_target: str = typer.Option("main_menu", "--prepare-target"),
    prepare_max_steps: int = typer.Option(
        0,
        "--prepare-max-steps",
        min=0,
        help="Preparation step budget. Use 0 to disable the budget and keep normalizing until the target is reached.",
    ),
    prepare_max_idle_polls: int = typer.Option(40, "--prepare-max-idle-polls", min=1),
) -> None:
    report = run_policy_checkpoint_comparison(
        base_url=base_url,
        baseline_checkpoint_path=baseline_checkpoint_path,
        candidate_checkpoint_path=candidate_checkpoint_path,
        output_root=output_root,
        comparison_name=comparison_name,
        repeats=repeats,
        max_env_steps=max_env_steps,
        max_runs=max_runs,
        max_combats=max_combats,
        poll_interval_seconds=poll_interval_seconds,
        max_idle_polls=max_idle_polls,
        request_timeout_seconds=request_timeout_seconds,
        prepare_target=prepare_target,
        prepare_max_steps=prepare_max_steps,
        prepare_max_idle_polls=prepare_max_idle_polls,
    )

    table = Table(title=f"Checkpoint Compare ({report.base_url})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Repeats", str(report.repeat_count))
    table.add_row("Prepare Target", report.prepare_target)
    table.add_row("Better Checkpoint", report.better_checkpoint_label or "tie")
    table.add_row("Baseline Reward", _format_metric(report.baseline.get("mean_total_reward")))
    table.add_row("Candidate Reward", _format_metric(report.candidate.get("mean_total_reward")))
    table.add_row("Delta Reward", _format_metric(report.delta_metrics.get("mean_total_reward")))
    table.add_row("Baseline Win Rate", _format_metric(report.baseline.get("combat_win_rate")))
    table.add_row("Candidate Win Rate", _format_metric(report.candidate.get("combat_win_rate")))
    table.add_row("Delta Win Rate", _format_metric(report.delta_metrics.get("combat_win_rate")))
    table.add_row("Comparison Dir", str(report.comparison_dir))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Iterations Path", str(report.iterations_path))
    console.print(table)


@eval_app.command("combat-dqn")
def eval_combat_dqn_command(
    base_url: str = typer.Option("http://127.0.0.1:8080", "--base-url"),
    checkpoint_path: Path = typer.Option(
        ...,
        "--checkpoint-path",
        help="Checkpoint generated by train combat-dqn.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/eval"),
        "--output-root",
        help="Root directory for evaluation logs.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional session directory name. Defaults to a UTC timestamp.",
    ),
    max_env_steps: int = typer.Option(
        0,
        "--max-env-steps",
        min=0,
        help="Hard environment-step budget. Use 0 to disable the step budget.",
    ),
    max_runs: int = typer.Option(
        1,
        "--max-runs",
        "--max-episodes",
        min=0,
        help="Completed-run budget. Use 0 to disable the run budget.",
    ),
    max_combats: int = typer.Option(
        0,
        "--max-combats",
        min=0,
        help="Completed-combat budget. Use 0 to disable the combat budget.",
    ),
    poll_interval_seconds: float = typer.Option(0.25, "--poll-interval-seconds", min=0.01),
    max_idle_polls: int = typer.Option(40, "--max-idle-polls", min=1),
    request_timeout_seconds: float = typer.Option(30.0, "--request-timeout-seconds", min=1.0),
    policy_pack: str = typer.Option(
        "baseline",
        "--policy-pack",
        help="Heuristic policy pack for non-combat control: baseline, legacy, planner, planner_assist, or conservative.",
    ),
    predictor_model_path: Path | None = typer.Option(
        None,
        "--predictor-model-path",
        help="Optional predictor model path for runtime-guided non-combat scoring.",
    ),
    predictor_mode: str = typer.Option(
        "heuristic_only",
        "--predictor-mode",
        help="Predictor guidance mode: heuristic_only, assist, or dominant.",
    ),
    predictor_hook: list[str] | None = typer.Option(
        None,
        "--predictor-hook",
        help="Repeatable predictor hook filter. Defaults to all supported hooks when predictor is enabled.",
    ),
    community_prior_source_path: Path | None = typer.Option(
        None,
        "--community-prior-source-path",
        help="Optional imported community-card-stats artifact directory or jsonl file.",
    ),
    community_route_prior_source_path: Path | None = typer.Option(
        None,
        "--community-route-prior-source-path",
        help="Optional public-run strategic-route-stats artifact directory or jsonl file.",
    ),
    community_reward_pick_weight: float = typer.Option(1.15, "--community-reward-pick-weight", min=0.0),
    community_selection_pick_weight: float = typer.Option(1.05, "--community-selection-pick-weight", min=0.0),
    community_selection_upgrade_weight: float = typer.Option(0.55, "--community-selection-upgrade-weight", min=0.0),
    community_selection_remove_weight: float = typer.Option(0.95, "--community-selection-remove-weight", min=0.0),
    community_shop_buy_weight: float = typer.Option(1.0, "--community-shop-buy-weight", min=0.0),
    community_route_weight: float = typer.Option(0.90, "--community-route-weight", min=0.0),
    community_reward_pick_neutral_rate: float = typer.Option(0.33, "--community-reward-pick-neutral-rate", min=0.0, max=1.0),
    community_shop_buy_neutral_rate: float = typer.Option(0.10, "--community-shop-buy-neutral-rate", min=0.0, max=1.0),
    community_route_neutral_win_rate: float = typer.Option(0.50, "--community-route-neutral-win-rate", min=0.0, max=1.0),
    community_pick_rate_scale: float = typer.Option(3.0, "--community-pick-rate-scale", min=0.0),
    community_buy_rate_scale: float = typer.Option(5.0, "--community-buy-rate-scale", min=0.0),
    community_win_delta_scale: float = typer.Option(12.0, "--community-win-delta-scale", min=0.0),
    community_route_win_rate_scale: float = typer.Option(8.0, "--community-route-win-rate-scale", min=0.0),
    community_min_sample_size: int = typer.Option(40, "--community-min-sample-size", min=1),
    community_route_min_sample_size: int = typer.Option(30, "--community-route-min-sample-size", min=1),
    community_max_confidence_sample_size: int = typer.Option(1200, "--community-max-confidence-sample-size", min=1),
    community_max_source_age_days: int | None = typer.Option(None, "--community-max-source-age-days", min=0),
    run_mode: str | None = typer.Option(None, "--run-mode", help="Intended in-game run mode contract, for example custom."),
    game_seed: str | None = typer.Option(None, "--game-seed", help="Expected in-game run seed recorded and validated from live observations."),
    seed_source: str | None = typer.Option(None, "--seed-source", help="Seed origin label, for example custom_mode_manual."),
    game_character_id: str | None = typer.Option(None, "--game-character-id", help="Expected character id in the live run contract."),
    game_ascension: int | None = typer.Option(None, "--game-ascension", min=0, help="Expected ascension in the live run contract."),
    custom_modifier: list[str] | None = typer.Option(None, "--custom-modifier", help="Repeatable Custom Mode modifier recorded in the live run contract."),
    progress_profile: str | None = typer.Option(None, "--progress-profile", help="Progress/unlock profile label recorded in the live run contract."),
    benchmark_contract_id: str | None = typer.Option(None, "--benchmark-contract-id", help="Optional benchmark contract id recorded in artifacts."),
    strict_game_run_contract: bool = typer.Option(True, "--strict-game-run-contract/--no-strict-game-run-contract", help="Stop immediately when observed seed/character/ascension diverge from the configured contract."),
) -> None:
    predictor_config = _build_predictor_runtime_config(
        model_path=predictor_model_path,
        mode=predictor_mode,
        hooks=predictor_hook,
    )
    community_prior_config = _build_community_prior_runtime_config(
        source_path=community_prior_source_path,
        route_source_path=community_route_prior_source_path,
        reward_pick_weight=community_reward_pick_weight,
        selection_pick_weight=community_selection_pick_weight,
        selection_upgrade_weight=community_selection_upgrade_weight,
        selection_remove_weight=community_selection_remove_weight,
        shop_buy_weight=community_shop_buy_weight,
        route_weight=community_route_weight,
        reward_pick_neutral_rate=community_reward_pick_neutral_rate,
        shop_buy_neutral_rate=community_shop_buy_neutral_rate,
        route_neutral_win_rate=community_route_neutral_win_rate,
        pick_rate_scale=community_pick_rate_scale,
        buy_rate_scale=community_buy_rate_scale,
        win_delta_scale=community_win_delta_scale,
        route_win_rate_scale=community_route_win_rate_scale,
        min_sample_size=community_min_sample_size,
        route_min_sample_size=community_route_min_sample_size,
        max_confidence_sample_size=community_max_confidence_sample_size,
        max_source_age_days=community_max_source_age_days,
    )
    game_run_contract = _build_live_game_run_contract(
        run_mode=run_mode,
        game_seed=game_seed,
        seed_source=seed_source,
        game_character_id=game_character_id,
        game_ascension=game_ascension,
        custom_modifier=custom_modifier,
        progress_profile=progress_profile,
        benchmark_contract_id=benchmark_contract_id,
        strict_game_run_contract=strict_game_run_contract,
    )
    report = run_combat_dqn_evaluation(
        base_url=base_url,
        checkpoint_path=checkpoint_path,
        output_root=output_root,
        session_name=session_name,
        max_env_steps=max_env_steps,
        max_runs=max_runs,
        max_combats=max_combats,
        poll_interval_seconds=poll_interval_seconds,
        max_idle_polls=max_idle_polls,
        request_timeout_seconds=request_timeout_seconds,
        policy_profile=policy_pack,
        predictor_config=predictor_config,
        community_prior_config=community_prior_config,
        game_run_contract=game_run_contract,
    )

    table = Table(title=f"Combat DQN Evaluation ({report.base_url})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Env Steps", str(report.env_steps))
    table.add_row("Combat Steps", str(report.combat_steps))
    table.add_row("Heuristic Steps", str(report.heuristic_steps))
    table.add_row("Completed Runs", str(report.completed_run_count))
    table.add_row("Completed Combats", str(report.completed_combat_count))
    table.add_row("Stop Reason", report.stop_reason)
    table.add_row("Total Reward", f"{report.total_reward:.3f}")
    table.add_row("Final Screen", report.final_screen)
    table.add_row("Final Run ID", report.final_run_id)
    table.add_row("Log Path", str(report.log_path))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Combat Outcomes", str(report.combat_outcomes_path))
    table.add_row("Checkpoint", str(report.checkpoint_path))
    table.add_row("Combat Win Rate", _format_metric(report.combat_performance.get("combat_win_rate")))
    table.add_row("Reward / Combat", _format_metric(report.combat_performance.get("reward_per_combat")))
    table.add_row("Reward / Combat Step", _format_metric(report.combat_performance.get("reward_per_combat_step")))
    console.print(table)


@eval_app.command("policy-pack")
def eval_policy_pack_command(
    base_url: str = typer.Option("http://127.0.0.1:8080", "--base-url"),
    policy_pack: str = typer.Option(
        "baseline",
        "--policy-pack",
        help="Policy pack profile: baseline, legacy, planner, planner_assist, or conservative.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/eval-policy"),
        "--output-root",
        help="Root directory for policy-pack evaluation logs.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional session directory name. Defaults to a UTC timestamp.",
    ),
    max_env_steps: int = typer.Option(0, "--max-env-steps", min=0),
    max_runs: int = typer.Option(1, "--max-runs", "--max-episodes", min=0),
    max_combats: int = typer.Option(0, "--max-combats", min=0),
    poll_interval_seconds: float = typer.Option(0.25, "--poll-interval-seconds", min=0.01),
    max_idle_polls: int = typer.Option(40, "--max-idle-polls", min=1),
    request_timeout_seconds: float = typer.Option(30.0, "--request-timeout-seconds", min=1.0),
    prepare_target: str | None = typer.Option(
        None,
        "--prepare-target",
        help="Preparation target before evaluation: none, main_menu, or character_select.",
    ),
    prepare_main_menu: bool = typer.Option(
        True,
        "--prepare-main-menu/--no-prepare-main-menu",
        help="Legacy toggle for MAIN_MENU preparation when --prepare-target is not provided.",
    ),
    prepare_max_steps: int = typer.Option(
        0,
        "--prepare-max-steps",
        min=0,
        help="Preparation step budget. Use 0 to disable the budget and keep normalizing until the target is reached.",
    ),
    prepare_max_idle_polls: int = typer.Option(40, "--prepare-max-idle-polls", min=1),
    predictor_model_path: Path | None = typer.Option(
        None,
        "--predictor-model-path",
        help="Optional predictor model path for runtime-guided scoring.",
    ),
    predictor_mode: str = typer.Option(
        "heuristic_only",
        "--predictor-mode",
        help="Predictor guidance mode: heuristic_only, assist, or dominant.",
    ),
    predictor_hook: list[str] | None = typer.Option(
        None,
        "--predictor-hook",
        help="Repeatable predictor hook filter. Defaults to all supported hooks when predictor is enabled.",
    ),
    strategic_checkpoint_path: Path | None = typer.Option(
        None,
        "--strategic-checkpoint-path",
        help="Optional strategic_pretrain or strategic_finetune checkpoint for runtime strategic guidance.",
    ),
    strategic_mode: str = typer.Option(
        "heuristic_only",
        "--strategic-mode",
        help="Strategic guidance mode: heuristic_only, assist, or dominant.",
    ),
    strategic_hook: list[str] | None = typer.Option(
        None,
        "--strategic-hook",
        help="Repeatable strategic hook filter. Defaults to all supported hooks when strategic guidance is enabled.",
    ),
    community_prior_source_path: Path | None = typer.Option(
        None,
        "--community-prior-source-path",
        help="Optional imported community-card-stats artifact directory or jsonl file.",
    ),
    community_route_prior_source_path: Path | None = typer.Option(
        None,
        "--community-route-prior-source-path",
        help="Optional public-run strategic-route-stats artifact directory or jsonl file.",
    ),
    community_reward_pick_weight: float = typer.Option(1.15, "--community-reward-pick-weight", min=0.0),
    community_selection_pick_weight: float = typer.Option(1.05, "--community-selection-pick-weight", min=0.0),
    community_selection_upgrade_weight: float = typer.Option(0.55, "--community-selection-upgrade-weight", min=0.0),
    community_selection_remove_weight: float = typer.Option(0.95, "--community-selection-remove-weight", min=0.0),
    community_shop_buy_weight: float = typer.Option(1.0, "--community-shop-buy-weight", min=0.0),
    community_route_weight: float = typer.Option(0.90, "--community-route-weight", min=0.0),
    community_reward_pick_neutral_rate: float = typer.Option(0.33, "--community-reward-pick-neutral-rate", min=0.0, max=1.0),
    community_shop_buy_neutral_rate: float = typer.Option(0.10, "--community-shop-buy-neutral-rate", min=0.0, max=1.0),
    community_route_neutral_win_rate: float = typer.Option(0.50, "--community-route-neutral-win-rate", min=0.0, max=1.0),
    community_pick_rate_scale: float = typer.Option(3.0, "--community-pick-rate-scale", min=0.0),
    community_buy_rate_scale: float = typer.Option(5.0, "--community-buy-rate-scale", min=0.0),
    community_win_delta_scale: float = typer.Option(12.0, "--community-win-delta-scale", min=0.0),
    community_route_win_rate_scale: float = typer.Option(8.0, "--community-route-win-rate-scale", min=0.0),
    community_min_sample_size: int = typer.Option(40, "--community-min-sample-size", min=1),
    community_route_min_sample_size: int = typer.Option(30, "--community-route-min-sample-size", min=1),
    community_max_confidence_sample_size: int = typer.Option(1200, "--community-max-confidence-sample-size", min=1),
    community_max_source_age_days: int | None = typer.Option(None, "--community-max-source-age-days", min=0),
    run_mode: str | None = typer.Option(None, "--run-mode", help="Intended in-game run mode contract, for example custom."),
    game_seed: str | None = typer.Option(None, "--game-seed", help="Expected in-game run seed recorded and validated from live observations."),
    seed_source: str | None = typer.Option(None, "--seed-source", help="Seed origin label, for example custom_mode_manual."),
    game_character_id: str | None = typer.Option(None, "--game-character-id", help="Expected character id in the live run contract."),
    game_ascension: int | None = typer.Option(None, "--game-ascension", min=0, help="Expected ascension in the live run contract."),
    custom_modifier: list[str] | None = typer.Option(None, "--custom-modifier", help="Repeatable Custom Mode modifier recorded in the live run contract."),
    progress_profile: str | None = typer.Option(None, "--progress-profile", help="Progress/unlock profile label recorded in the live run contract."),
    benchmark_contract_id: str | None = typer.Option(None, "--benchmark-contract-id", help="Optional benchmark contract id recorded in artifacts."),
    strict_game_run_contract: bool = typer.Option(True, "--strict-game-run-contract/--no-strict-game-run-contract", help="Stop immediately when observed seed/character/ascension diverge from the configured contract."),
) -> None:
    resolved_prepare_target = _resolve_prepare_target_option(
        prepare_main_menu=prepare_main_menu,
        prepare_target=prepare_target,
    )
    predictor_config = _build_predictor_runtime_config(
        model_path=predictor_model_path,
        mode=predictor_mode,
        hooks=predictor_hook,
    )
    strategic_model_config = _build_strategic_runtime_config(
        checkpoint_path=strategic_checkpoint_path,
        mode=strategic_mode,
        hooks=strategic_hook,
    )
    community_prior_config = _build_community_prior_runtime_config(
        source_path=community_prior_source_path,
        route_source_path=community_route_prior_source_path,
        reward_pick_weight=community_reward_pick_weight,
        selection_pick_weight=community_selection_pick_weight,
        selection_upgrade_weight=community_selection_upgrade_weight,
        selection_remove_weight=community_selection_remove_weight,
        shop_buy_weight=community_shop_buy_weight,
        route_weight=community_route_weight,
        reward_pick_neutral_rate=community_reward_pick_neutral_rate,
        shop_buy_neutral_rate=community_shop_buy_neutral_rate,
        route_neutral_win_rate=community_route_neutral_win_rate,
        pick_rate_scale=community_pick_rate_scale,
        buy_rate_scale=community_buy_rate_scale,
        win_delta_scale=community_win_delta_scale,
        route_win_rate_scale=community_route_win_rate_scale,
        min_sample_size=community_min_sample_size,
        route_min_sample_size=community_route_min_sample_size,
        max_confidence_sample_size=community_max_confidence_sample_size,
        max_source_age_days=community_max_source_age_days,
    )
    game_run_contract = _build_live_game_run_contract(
        run_mode=run_mode,
        game_seed=game_seed,
        seed_source=seed_source,
        game_character_id=game_character_id,
        game_ascension=game_ascension,
        custom_modifier=custom_modifier,
        progress_profile=progress_profile,
        benchmark_contract_id=benchmark_contract_id,
        strict_game_run_contract=strict_game_run_contract,
    )
    report = run_policy_pack_evaluation(
        base_url=base_url,
        output_root=output_root,
        session_name=session_name,
        policy_profile=policy_pack,
        max_env_steps=max_env_steps,
        max_runs=max_runs,
        max_combats=max_combats,
        poll_interval_seconds=poll_interval_seconds,
        max_idle_polls=max_idle_polls,
        request_timeout_seconds=request_timeout_seconds,
        prepare_target=resolved_prepare_target,
        prepare_max_steps=prepare_max_steps,
        prepare_max_idle_polls=prepare_max_idle_polls,
        predictor_config=predictor_config,
        strategic_model_config=strategic_model_config,
        community_prior_config=community_prior_config,
        game_run_contract=game_run_contract,
    )
    display_prepare_target = "custom_run" if game_run_contract is not None and game_run_contract.run_mode == "custom" else resolved_prepare_target

    table = Table(title=f"Policy Pack Evaluation ({policy_pack})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Prepare Target", display_prepare_target)
    table.add_row("Env Steps", str(report.env_steps))
    table.add_row("Combat Steps", str(report.combat_steps))
    table.add_row("Policy Steps", str(report.heuristic_steps))
    table.add_row("Completed Runs", str(report.completed_run_count))
    table.add_row("Completed Combats", str(report.completed_combat_count))
    table.add_row("Stop Reason", report.stop_reason)
    table.add_row("Total Reward", f"{report.total_reward:.3f}")
    table.add_row("Final Screen", report.final_screen)
    table.add_row("Final Run ID", report.final_run_id)
    table.add_row("Log Path", str(report.log_path))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Combat Outcomes", str(report.combat_outcomes_path))
    table.add_row("Combat Win Rate", _format_metric(report.combat_performance.get("combat_win_rate")))
    table.add_row("Reward / Combat", _format_metric(report.combat_performance.get("reward_per_combat")))
    console.print(table)


@eval_app.command("combat-dqn-compare")
def eval_combat_dqn_compare_command(
    base_url: str = typer.Option("http://127.0.0.1:8080", "--base-url"),
    baseline_checkpoint_path: Path = typer.Option(
        ...,
        "--baseline-checkpoint-path",
        help="Baseline checkpoint path.",
    ),
    candidate_checkpoint_path: Path = typer.Option(
        ...,
        "--candidate-checkpoint-path",
        help="Candidate checkpoint path.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/eval-compare"),
        "--output-root",
        help="Root directory for checkpoint comparison artifacts.",
    ),
    comparison_name: str | None = typer.Option(
        None,
        "--comparison-name",
        help="Optional comparison directory name. Defaults to a UTC timestamp.",
    ),
    repeats: int = typer.Option(3, "--repeats", min=1),
    max_env_steps: int = typer.Option(0, "--max-env-steps", min=0),
    max_runs: int = typer.Option(1, "--max-runs", "--max-episodes", min=0),
    max_combats: int = typer.Option(0, "--max-combats", min=0),
    poll_interval_seconds: float = typer.Option(0.25, "--poll-interval-seconds", min=0.01),
    max_idle_polls: int = typer.Option(40, "--max-idle-polls", min=1),
    request_timeout_seconds: float = typer.Option(30.0, "--request-timeout-seconds", min=1.0),
    prepare_target: str | None = typer.Option(
        None,
        "--prepare-target",
        help="Preparation target before each evaluation: none, main_menu, or character_select.",
    ),
    prepare_main_menu: bool = typer.Option(
        True,
        "--prepare-main-menu/--no-prepare-main-menu",
        help="Legacy toggle for MAIN_MENU preparation when --prepare-target is not provided.",
    ),
    prepare_max_steps: int = typer.Option(8, "--prepare-max-steps", min=1),
    prepare_max_idle_polls: int = typer.Option(40, "--prepare-max-idle-polls", min=1),
) -> None:
    resolved_prepare_target = _resolve_prepare_target_option(
        prepare_main_menu=prepare_main_menu,
        prepare_target=prepare_target,
    )
    report = run_combat_dqn_checkpoint_comparison(
        base_url=base_url,
        baseline_checkpoint_path=baseline_checkpoint_path,
        candidate_checkpoint_path=candidate_checkpoint_path,
        output_root=output_root,
        comparison_name=comparison_name,
        repeats=repeats,
        max_env_steps=max_env_steps,
        max_runs=max_runs,
        max_combats=max_combats,
        poll_interval_seconds=poll_interval_seconds,
        max_idle_polls=max_idle_polls,
        request_timeout_seconds=request_timeout_seconds,
        prepare_main_menu=prepare_main_menu,
        prepare_target=resolved_prepare_target,
        prepare_max_steps=prepare_max_steps,
        prepare_max_idle_polls=prepare_max_idle_polls,
    )

    table = Table(title=f"Combat DQN Compare ({report.base_url})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Repeats", str(report.repeat_count))
    table.add_row("Prepare Target", report.prepare_target)
    table.add_row("Better Checkpoint", report.better_checkpoint_label or "tie")
    table.add_row("Baseline Reward", _format_metric(report.baseline.get("mean_total_reward")))
    table.add_row("Candidate Reward", _format_metric(report.candidate.get("mean_total_reward")))
    table.add_row("Delta Reward", _format_metric(report.delta_metrics.get("mean_total_reward")))
    table.add_row("Baseline Win Rate", _format_metric(report.baseline.get("combat_win_rate")))
    table.add_row("Candidate Win Rate", _format_metric(report.candidate.get("combat_win_rate")))
    table.add_row("Delta Win Rate", _format_metric(report.delta_metrics.get("combat_win_rate")))
    table.add_row("Comparison Dir", str(report.comparison_dir))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Iterations Path", str(report.iterations_path))
    table.add_row("Log Path", str(report.log_path))
    console.print(table)


@eval_app.command("combat-dqn-replay")
def eval_combat_dqn_replay_command(
    base_url: str = typer.Option("http://127.0.0.1:8080", "--base-url"),
    checkpoint_path: Path = typer.Option(
        ...,
        "--checkpoint-path",
        help="Checkpoint generated by train combat-dqn.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/replay"),
        "--output-root",
        help="Root directory for replay-suite artifacts.",
    ),
    suite_name: str | None = typer.Option(
        None,
        "--suite-name",
        help="Optional replay-suite directory name. Defaults to a UTC timestamp.",
    ),
    repeats: int = typer.Option(3, "--repeats", min=2),
    max_env_steps: int = typer.Option(
        0,
        "--max-env-steps",
        min=0,
        help="Hard environment-step budget per replay iteration. Use 0 to disable the step budget.",
    ),
    max_runs: int = typer.Option(
        1,
        "--max-runs",
        "--max-episodes",
        min=0,
        help="Completed-run budget per replay iteration. Use 0 to disable the run budget.",
    ),
    max_combats: int = typer.Option(
        0,
        "--max-combats",
        min=0,
        help="Completed-combat budget per replay iteration. Use 0 to disable the combat budget.",
    ),
    poll_interval_seconds: float = typer.Option(0.25, "--poll-interval-seconds", min=0.01),
    max_idle_polls: int = typer.Option(40, "--max-idle-polls", min=1),
    request_timeout_seconds: float = typer.Option(30.0, "--request-timeout-seconds", min=1.0),
    prepare_target: str | None = typer.Option(
        None,
        "--prepare-target",
        help="Preparation target before each replay iteration: none, main_menu, or character_select.",
    ),
    prepare_main_menu: bool = typer.Option(
        True,
        "--prepare-main-menu/--no-prepare-main-menu",
        help="Legacy toggle for MAIN_MENU preparation when --prepare-target is not provided.",
    ),
    prepare_max_steps: int = typer.Option(8, "--prepare-max-steps", min=1),
    prepare_max_idle_polls: int = typer.Option(40, "--prepare-max-idle-polls", min=1),
) -> None:
    resolved_prepare_target = _resolve_prepare_target_option(
        prepare_main_menu=prepare_main_menu,
        prepare_target=prepare_target,
    )
    report = run_combat_dqn_replay_suite(
        base_url=base_url,
        checkpoint_path=checkpoint_path,
        output_root=output_root,
        suite_name=suite_name,
        repeats=repeats,
        max_env_steps=max_env_steps,
        max_runs=max_runs,
        max_combats=max_combats,
        poll_interval_seconds=poll_interval_seconds,
        max_idle_polls=max_idle_polls,
        request_timeout_seconds=request_timeout_seconds,
        prepare_main_menu=prepare_main_menu,
        prepare_target=resolved_prepare_target,
        prepare_max_steps=prepare_max_steps,
        prepare_max_idle_polls=prepare_max_idle_polls,
    )

    table = Table(title=f"Combat DQN Replay ({report.base_url})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Repeats", str(report.repeat_count))
    table.add_row("Comparisons", str(report.comparison_count))
    table.add_row("Exact Matches", str(report.exact_match_count))
    table.add_row("Divergent Iterations", str(report.divergent_iteration_count))
    table.add_row("Prepare Target", report.prepare_target)
    table.add_row("Status Histogram", json.dumps(report.status_histogram, ensure_ascii=False))
    table.add_row("Suite Dir", str(report.suite_dir))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Comparisons Path", str(report.comparisons_path))
    table.add_row("Log Path", str(report.log_path))
    console.print(table)


@eval_app.command("divergence-summary")
def eval_divergence_summary_command(
    source: Path = typer.Option(
        ...,
        "--source",
        help="Replay or comparison summary path, or a directory containing replay-summary.json or comparison-summary.json.",
    ),
) -> None:
    summary = load_divergence_summary(source)

    overview = Table(title=f"Divergence Summary ({summary['artifact_kind']})")
    overview.add_column("Metric")
    overview.add_column("Value")
    overview.add_row("Source", str(summary["source_path"]))
    overview.add_row("Diagnostics", str(summary["diagnostic_count"]))
    overview.add_row("Families", json.dumps(summary["family_histogram"], ensure_ascii=False))
    overview.add_row("Categories", json.dumps(summary["category_histogram"], ensure_ascii=False))
    overview.add_row("Statuses", json.dumps(summary["status_histogram"], ensure_ascii=False))
    console.print(overview)

    diagnostics = Table(title="Diagnostics")
    diagnostics.add_column("Status")
    diagnostics.add_column("Family")
    diagnostics.add_column("Category")
    diagnostics.add_column("Step")
    diagnostics.add_column("Explanation")
    for item in summary["diagnostics"]:
        diagnostics.add_row(
            str(item.get("status")),
            str(item.get("family")),
            str(item.get("category")),
            str(item.get("step_index") or "-"),
            str(item.get("explanation")),
        )
    console.print(diagnostics)


@eval_app.command("capability-summary")
def eval_capability_summary_command(
    source: Path = typer.Option(
        ...,
        "--source",
        help="Session summary path, case summary path, suite summary path, or a directory containing one of those files.",
    ),
) -> None:
    summary = load_capability_summary(source)
    overview = Table(title=f"Capability Summary ({summary['artifact_kind']})")
    overview.add_column("Metric")
    overview.add_column("Value")
    overview.add_row("Source", str(summary["source_path"]))
    if summary["artifact_kind"] == "benchmark_compare_case":
        comparison = dict(summary.get("comparison", {}))
        overview.add_row("Case", str(summary.get("case_id") or "-"))
        overview.add_row("Baseline Issues", str(comparison.get("baseline_issue_count", 0)))
        overview.add_row("Candidate Issues", str(comparison.get("candidate_issue_count", 0)))
        overview.add_row("Delta Issues", str(comparison.get("delta_issue_count", 0)))
        overview.add_row("New Regressions", str(comparison.get("new_regression_count", 0)))
        overview.add_row("Regression Keys", json.dumps(comparison.get("new_regression_keys", []), ensure_ascii=False))
        console.print(overview)
        console.print(_capability_overview_table("Baseline Capability", dict(summary.get("baseline", {}))))
        console.print(_capability_overview_table("Candidate Capability", dict(summary.get("candidate", {}))))
        console.print(
            _capability_diagnostics_table(
                "Candidate Diagnostics",
                list(dict(summary.get("candidate", {})).get("diagnostics", [])),
            )
        )
        return
    if summary["artifact_kind"] == "benchmark_suite":
        compare_payload = dict(dict(summary.get("compare", {})).get("comparison", {}))
        overview.add_row("Suite", str(summary.get("suite_name") or "-"))
        overview.add_row("Eval Issues", str(dict(summary.get("eval", {})).get("diagnostic_count", 0)))
        overview.add_row(
            "Compare Candidate Issues",
            str(dict(dict(summary.get("compare", {})).get("candidate", {})).get("diagnostic_count", 0)),
        )
        overview.add_row("New Regressions", str(compare_payload.get("new_regression_count", 0)))
        overview.add_row("Regression Keys", json.dumps(compare_payload.get("new_regression_keys", []), ensure_ascii=False))
        console.print(overview)
        console.print(_capability_overview_table("Eval Capability", dict(summary.get("eval", {}))))
        console.print(
            _capability_overview_table(
                "Compare Candidate Capability",
                dict(dict(summary.get("compare", {})).get("candidate", {})),
            )
        )
        return

    scoped_summary = dict(summary.get("summary", {}))
    overview.add_row("Name", str(summary.get("session_name") or summary.get("case_id") or "-"))
    overview.add_row("Diagnostics", str(scoped_summary.get("diagnostic_count", 0)))
    overview.add_row("Buckets", json.dumps(scoped_summary.get("bucket_histogram", {}), ensure_ascii=False))
    overview.add_row("Owners", json.dumps(scoped_summary.get("owner_histogram", {}), ensure_ascii=False))
    console.print(overview)
    console.print(_capability_overview_table("Capability Overview", scoped_summary))
    console.print(_capability_diagnostics_table("Diagnostics", list(scoped_summary.get("diagnostics", []))))


@dataset_app.command("validate")
def dataset_validate_command(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        help="Path to a dataset manifest in JSON or TOML format.",
    ),
) -> None:
    report = validate_dataset_manifest(manifest)

    console.print(
        _json_table(
            f"Dataset Manifest ({report.dataset_name})",
            [
                ("Manifest Path", report.manifest_path),
                ("Dataset Kind", report.dataset_kind),
                ("Source Files", len(report.source_files)),
                ("Filters", report.filters),
                ("Split", report.split),
                ("Output", report.output),
            ],
        )
    )


@dataset_app.command("build")
def dataset_build_command(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        help="Path to a dataset manifest in JSON or TOML format.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        help="Dataset output directory.",
    ),
    replace_existing: bool = typer.Option(
        False,
        "--replace-existing/--no-replace-existing",
        help="Replace an existing output directory.",
    ),
) -> None:
    report = build_dataset_from_manifest(
        manifest,
        output_dir=output_dir,
        replace_existing=replace_existing,
    )
    summary_payload = load_dataset_summary(report.output_dir)

    console.print(
        _json_table(
            f"Dataset Build ({report.output_dir})",
            [
                ("Dataset Kind", report.dataset_kind),
                ("Records", report.record_count),
                ("Features", report.feature_count),
                ("Source Files", report.source_file_count),
                ("Source Records", report.source_record_count),
                ("Filtered Out", report.filtered_out_count),
                ("Split Counts", report.split_counts),
                ("Manifest Path", report.manifest_path),
                ("Summary Path", report.summary_path),
                ("Records Path", report.records_path),
                ("Lineage", summary_payload.get("lineage")),
            ],
        )
    )


@dataset_app.command("summary")
def dataset_summary_command(
    source: Path = typer.Option(
        ...,
        "--source",
        help="Dataset directory or dataset-summary.json path.",
    ),
) -> None:
    summary_payload = load_dataset_summary(source)

    console.print(
        _json_table(
            f"Dataset Summary ({summary_payload.get('dataset_name', 'unknown')})",
            [
                ("Dataset Kind", summary_payload.get("dataset_kind")),
                ("Records", summary_payload.get("record_count")),
                ("Source Files", summary_payload.get("source_file_count")),
                ("Filtered Out", summary_payload.get("filtered_out_count")),
                ("Split", summary_payload.get("split")),
                ("Session Kinds", summary_payload.get("session_kind_histogram")),
                ("Characters", summary_payload.get("character_histogram")),
                ("Outcomes", summary_payload.get("outcome_histogram")),
                ("Screens", summary_payload.get("screen_histogram")),
                ("Decision Sources", summary_payload.get("decision_source_histogram")),
                ("Episodes", summary_payload.get("episode_count")),
                ("Supported Transitions", summary_payload.get("supported_transition_count")),
                ("Reward Stats", summary_payload.get("reward_stats")),
                ("Return Stats", summary_payload.get("return_stats")),
                ("Action Support", summary_payload.get("action_support_histogram")),
                ("Normalization", summary_payload.get("normalization")),
                ("Exports", summary_payload.get("exports")),
                ("Lineage", summary_payload.get("lineage")),
            ],
        )
    )


@community_app.command("import")
def community_import_command(
    source: Path = typer.Option(
        ...,
        "--source",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="CSV, JSONL, or JSON file containing community card stats.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/community-card-stats"),
        "--output-root",
        help="Root directory for canonicalized community card stats snapshots.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional output directory name. Defaults to a UTC timestamp.",
    ),
    source_name: str | None = typer.Option(
        None,
        "--source-name",
        help="Override source label, for example ststracker or slaythestats.",
    ),
    source_url: str | None = typer.Option(
        None,
        "--source-url",
        help="Optional source URL captured into the snapshot lineage.",
    ),
    snapshot_date: str | None = typer.Option(
        None,
        "--snapshot-date",
        help="Snapshot date in YYYY-MM-DD format. Defaults to today when omitted.",
    ),
    snapshot_label: str | None = typer.Option(
        None,
        "--snapshot-label",
        help="Optional snapshot label, for example community-april-2026.",
    ),
    game_version: str | None = typer.Option(
        None,
        "--game-version",
        help="Optional game version label applied to imported rows.",
    ),
    replace_existing: bool = typer.Option(
        False,
        "--replace-existing/--no-replace-existing",
        help="Replace an existing output session directory instead of failing.",
    ),
) -> None:
    report = import_community_card_stats(
        source=source,
        output_root=output_root,
        session_name=session_name,
        source_name=source_name,
        source_url=source_url,
        snapshot_date=snapshot_date,
        snapshot_label=snapshot_label,
        game_version=game_version,
        replace_existing=replace_existing,
    )
    summary = load_community_card_stats_summary(report.output_dir)
    console.print(
        _json_table(
            f"Community Card Stats Import ({summary.get('record_count', 0)} records)",
            [
                ("Session", report.output_dir.name),
                ("Records", report.record_count),
                ("Cards", report.card_count),
                ("Source Kind", summary.get("source_kind")),
                ("Requests", summary.get("source_request_count")),
                ("Sources", summary.get("source_histogram")),
                ("Characters", summary.get("character_histogram")),
                ("Pick Rate", summary.get("pick_rate_stats")),
                ("Buy Rate", summary.get("buy_rate_stats")),
                ("Win Delta", summary.get("win_delta_stats")),
                ("Manifest Path", report.source_manifest_path),
                ("Raw Payload Root", report.raw_payload_root),
                ("Summary Path", report.summary_path),
            ],
        )
    )


@community_app.command("import-spiremeta")
def community_import_spiremeta_command(
    character: list[str] = typer.Option(
        ...,
        "--character",
        help="Repeatable SpireMeta character slug, for example ironclad or regent.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/community-card-stats"),
        "--output-root",
        help="Root directory for canonicalized community card stats snapshots.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="Optional SpireMeta API key. Defaults to SPIREMETA_API_KEY from the environment.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional output directory name. Defaults to a UTC timestamp.",
    ),
    snapshot_date: str | None = typer.Option(
        None,
        "--snapshot-date",
        help="Snapshot date in YYYY-MM-DD format. Defaults to today when omitted.",
    ),
    snapshot_label: str | None = typer.Option(
        None,
        "--snapshot-label",
        help="Optional snapshot label, for example spiremeta-april-2026.",
    ),
    game_version: str | None = typer.Option(
        None,
        "--game-version",
        help="Optional game version label applied to imported rows.",
    ),
    per_page: int = typer.Option(
        100,
        "--per-page",
        min=1,
        help="SpireMeta page size for card-stat requests.",
    ),
    max_pages: int | None = typer.Option(
        None,
        "--max-pages",
        min=1,
        help="Optional per-character page cap for partial snapshot imports.",
    ),
    source_type: str = typer.Option(
        "reward",
        "--source-type",
        help="Community source type stored on imported rows. Defaults to reward.",
    ),
    api_base_url: str = typer.Option(
        "https://api.spiremeta.gg",
        "--api-base-url",
        help="SpireMeta API base URL.",
    ),
    request_timeout_seconds: float = typer.Option(
        20.0,
        "--request-timeout-seconds",
        min=1.0,
        help="HTTP timeout for each SpireMeta API request.",
    ),
    replace_existing: bool = typer.Option(
        False,
        "--replace-existing/--no-replace-existing",
        help="Replace an existing output session directory instead of failing.",
    ),
) -> None:
    resolved_source_type = _resolve_community_source_type_option(source_type)
    report = import_spiremeta_community_card_stats(
        output_root=output_root,
        characters=character,
        api_key=api_key,
        session_name=session_name,
        snapshot_date=snapshot_date,
        snapshot_label=snapshot_label,
        game_version=game_version,
        per_page=per_page,
        max_pages=max_pages,
        source_type=resolved_source_type,  # type: ignore[arg-type]
        replace_existing=replace_existing,
        api_base_url=api_base_url,
        request_timeout_seconds=request_timeout_seconds,
    )
    summary = load_community_card_stats_summary(report.output_dir)
    console.print(
        _json_table(
            f"SpireMeta Community Import ({summary.get('record_count', 0)} records)",
            [
                ("Session", report.output_dir.name),
                ("Records", report.record_count),
                ("Cards", report.card_count),
                ("Source Kind", summary.get("source_kind")),
                ("Requests", summary.get("source_request_count")),
                ("Characters", summary.get("character_histogram")),
                ("Snapshot Labels", summary.get("snapshot_label_histogram")),
                ("Pick Rate", summary.get("pick_rate_stats")),
                ("Win Rate With Card", summary.get("win_rate_with_card_stats")),
                ("Manifest Path", report.source_manifest_path),
                ("Raw Payload Root", report.raw_payload_root),
                ("Summary Path", report.summary_path),
            ],
        )
    )


@community_app.command("summary")
def community_summary_command(
    source: Path = typer.Option(
        ...,
        "--source",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Community card stats directory or summary.json path.",
    ),
) -> None:
    summary = load_community_card_stats_summary(source)
    console.print(
        _json_table(
            "Community Card Stats Summary",
            [
                ("Records", summary.get("record_count")),
                ("Cards", summary.get("card_count")),
                ("Source Kind", summary.get("source_kind")),
                ("Requests", summary.get("source_request_count")),
                ("Sources", summary.get("source_histogram")),
                ("Snapshot Dates", summary.get("snapshot_date_histogram")),
                ("Snapshot Labels", summary.get("snapshot_label_histogram")),
                ("Characters", summary.get("character_histogram")),
                ("Source Types", summary.get("source_type_histogram")),
                ("Game Versions", summary.get("game_version_histogram")),
                ("Pick Rate", summary.get("pick_rate_stats")),
                ("Buy Rate", summary.get("buy_rate_stats")),
                ("Win Rate With Card", summary.get("win_rate_with_card_stats")),
                ("Win Delta", summary.get("win_delta_stats")),
                ("Manifest Path", summary.get("source_manifest_path")),
                ("Raw Payload Root", summary.get("raw_payload_root")),
                ("Top Pick Rate Cards", summary.get("top_pick_rate_cards")),
                ("Top Win Delta Cards", summary.get("top_win_delta_cards")),
                ("Summary Path", summary.get("summary_path")),
            ],
        )
    )


@public_runs_app.command("sync")
def public_runs_sync_command(
    archive_root: Path = typer.Option(
        Path("artifacts/public-run-archive/sts2runs-main"),
        "--archive-root",
        help="Persistent archive root updated in place across sync sessions.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional sync session name. Defaults to a UTC timestamp.",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        min=1,
        help="List endpoint page size.",
    ),
    max_list_pages: int | None = typer.Option(
        None,
        "--max-list-pages",
        min=1,
        help="Optional cap on fetched list pages.",
    ),
    max_detail_fetches: int | None = typer.Option(
        None,
        "--max-detail-fetches",
        min=1,
        help="Optional cap on fetched run-detail payloads.",
    ),
    stop_after_consecutive_known_pages: int = typer.Option(
        2,
        "--stop-after-consecutive-known-pages",
        min=1,
        help="Stop after this many consecutive pages add no new run ids.",
    ),
    initial_page: int = typer.Option(
        0,
        "--initial-page",
        min=0,
        help="Starting list page. STS2Runs uses zero-based pages.",
    ),
    source_base_url: str = typer.Option(
        "https://sts2runs.com",
        "--source-base-url",
        help="STS2Runs base URL.",
    ),
    request_timeout_seconds: float = typer.Option(
        30.0,
        "--request-timeout-seconds",
        min=1.0,
        help="HTTP timeout for each request.",
    ),
    max_retries: int = typer.Option(
        3,
        "--max-retries",
        min=0,
        help="Retry count for list/detail HTTP failures.",
    ),
    retry_backoff_seconds: float = typer.Option(
        0.5,
        "--retry-backoff-seconds",
        min=0.0,
        help="Linear retry backoff multiplier in seconds.",
    ),
    replace_existing_archive: bool = typer.Option(
        False,
        "--replace-existing-archive/--no-replace-existing-archive",
        help="Replace the current archive root before syncing.",
    ),
) -> None:
    report = sync_sts2runs_public_run_archive(
        archive_root=archive_root,
        session_name=session_name,
        limit=limit,
        max_list_pages=max_list_pages,
        max_detail_fetches=max_detail_fetches,
        stop_after_consecutive_known_pages=stop_after_consecutive_known_pages,
        initial_page=initial_page,
        source_base_url=source_base_url,
        request_timeout_seconds=request_timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        replace_existing_archive=replace_existing_archive,
    )
    summary = load_public_run_archive_summary(report.archive_root)
    console.print(
        _json_table(
            f"Public Run Archive Sync ({summary.get('known_run_count', 0)} runs)",
            [
                ("Archive Root", report.archive_root),
                ("New Runs", report.new_run_count),
                ("Duplicate Runs", report.duplicate_run_count),
                ("Duplicate Sha256", report.duplicate_sha256_count),
                ("Detail Fetched", report.detail_fetched_count),
                ("Pending Detail", report.pending_detail_run_count),
                ("Failed Detail", report.failed_detail_run_count),
                ("List Pages", report.list_page_count),
                ("Detail Requests", report.detail_request_count),
                ("Session Summary", report.session_summary_path),
                ("Summary Path", report.summary_path),
            ],
        )
    )


@public_runs_app.command("summary")
def public_runs_summary_command(
    source: Path = typer.Option(
        ...,
        "--source",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Public run archive directory or summary.json path.",
    ),
) -> None:
    summary = load_public_run_archive_summary(source)
    console.print(
        _json_table(
            "Public Run Archive Summary",
            [
                ("Known Runs", summary.get("known_run_count")),
                ("Detailed Runs", summary.get("detailed_run_count")),
                ("Pending Detail", summary.get("pending_detail_run_count")),
                ("Failed Detail", summary.get("failed_detail_run_count")),
                ("Detail Coverage", summary.get("detail_coverage")),
                ("List Requests", summary.get("total_list_requests")),
                ("Detail Requests", summary.get("total_detail_requests")),
                ("Last Sync Session", summary.get("last_sync_session")),
                ("Highest Run Id", summary.get("highest_source_run_id")),
                ("Characters", summary.get("character_histogram")),
                ("Builds", summary.get("build_id_histogram")),
                ("Ascensions", summary.get("ascension_histogram")),
                ("Wins", summary.get("win_histogram")),
                ("Summary Path", summary.get("summary_path") or source),
            ],
        )
    )


@public_runs_app.command("normalize")
def public_runs_normalize_command(
    source: Path = typer.Option(
        ...,
        "--source",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Public run archive directory or archive file path.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/public-run-normalized"),
        "--output-root",
        help="Root directory for normalized public run artifacts.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional normalized artifact directory name. Defaults to a UTC timestamp.",
    ),
    replace_existing: bool = typer.Option(
        False,
        "--replace-existing/--no-replace-existing",
        help="Replace the target output directory if it already exists.",
    ),
) -> None:
    report = normalize_public_run_archive(
        source=source,
        output_root=output_root,
        session_name=session_name,
        replace_existing=replace_existing,
    )
    summary = load_public_run_normalized_summary(report.output_dir)
    console.print(
        _json_table(
            f"Public Run Normalize ({summary.get('record_count', 0)} runs)",
            [
                ("Output Dir", report.output_dir),
                ("Normalized Runs", report.normalized_runs_path),
                ("Normalized Table", report.normalized_runs_table_path),
                ("Strategic Cards", report.strategic_card_stats_path),
                ("Strategic Shops", report.strategic_shop_stats_path),
                ("Strategic Events", report.strategic_event_stats_path),
                ("Strategic Relics", report.strategic_relic_stats_path),
                ("Strategic Encounters", report.strategic_encounter_stats_path),
                ("Strategic Routes", report.strategic_route_stats_path),
                ("Summary Path", report.summary_path),
            ],
        )
    )


@public_runs_app.command("normalized-summary")
def public_runs_normalized_summary_command(
    source: Path = typer.Option(
        ...,
        "--source",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Normalized public run artifact directory or summary.json path.",
    ),
) -> None:
    summary = load_public_run_normalized_summary(source)
    console.print(
        _json_table(
            "Public Run Normalized Summary",
            [
                ("Record Count", summary.get("record_count")),
                ("Detail Coverage", summary.get("detail_coverage_count")),
                ("Characters", summary.get("character_histogram")),
                ("Builds", summary.get("build_id_histogram")),
                ("Ascensions", summary.get("ascension_histogram")),
                ("Outcomes", summary.get("outcome_histogram")),
                ("Room Types", summary.get("room_type_histogram")),
                ("Acts Reached", summary.get("acts_reached_histogram")),
                ("Strategic Cards", summary.get("strategic_card_stats_path")),
                ("Strategic Shops", summary.get("strategic_shop_stats_path")),
                ("Strategic Events", summary.get("strategic_event_stats_path")),
                ("Strategic Relics", summary.get("strategic_relic_stats_path")),
                ("Strategic Encounters", summary.get("strategic_encounter_stats_path")),
                ("Strategic Routes", summary.get("strategic_route_stats_path")),
                ("Summary Path", summary.get("summary_path") or source),
            ],
        )
    )


@shadow_app.command("combat-eval")
def shadow_combat_eval_command(
    source: Path = typer.Option(
        ...,
        "--source",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Shadow encounter dataset directory or encounters.jsonl file.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/shadow"),
        "--output-root",
        help="Root directory for shadow evaluation reports.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional output directory name. Defaults to a UTC timestamp.",
    ),
    policy_profile: str = typer.Option(
        "planner",
        "--policy-profile",
        help="Policy profile to replay over stored combat snapshots.",
    ),
    predictor_model_path: Path | None = typer.Option(
        None,
        "--predictor-model-path",
        help="Optional predictor model path for runtime-guided shadow scoring.",
    ),
    predictor_mode: str = typer.Option(
        "heuristic_only",
        "--predictor-mode",
        help="Predictor guidance mode: heuristic_only, assist, or dominant.",
    ),
    predictor_hook: list[str] | None = typer.Option(
        None,
        "--predictor-hook",
        help="Repeatable predictor hook filter. Defaults to all supported hooks when predictor is enabled.",
    ),
    replace_existing: bool = typer.Option(
        False,
        "--replace-existing/--no-replace-existing",
        help="Replace an existing session directory instead of failing.",
    ),
) -> None:
    predictor_config = _build_predictor_runtime_config(
        model_path=predictor_model_path,
        mode=predictor_mode,
        hooks=predictor_hook,
    )
    report = run_shadow_combat_evaluation(
        source=source,
        output_root=output_root,
        session_name=session_name,
        policy_profile=policy_profile,
        predictor_config=predictor_config,
        replace_existing=replace_existing,
    )
    summary_payload = load_shadow_combat_report(report.summary_path)
    metrics = summary_payload["metrics"]
    table = Table(title=f"Shadow Combat Eval ({policy_profile})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Encounters", str(summary_payload["encounter_count"]))
    table.add_row("Usable", str(summary_payload["usable_encounter_count"]))
    table.add_row("Skipped", str(summary_payload["skipped_encounter_count"]))
    table.add_row("First Action Match", _format_metric(metrics.get("first_action_match_rate")))
    table.add_row("Trace Hit", _format_metric(metrics.get("trace_hit_rate")))
    table.add_row("Decision Score", _format_metric(metrics.get("decision_score_stats")))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Results Path", str(report.results_path))
    console.print(table)


@shadow_app.command("combat-compare")
def shadow_combat_compare_command(
    source: Path = typer.Option(
        ...,
        "--source",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Shadow encounter dataset directory or encounters.jsonl file.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/shadow"),
        "--output-root",
        help="Root directory for shadow comparison reports.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional output directory name. Defaults to a UTC timestamp.",
    ),
    baseline_policy_profile: str = typer.Option(
        "baseline",
        "--baseline-policy-profile",
        help="Baseline policy profile.",
    ),
    candidate_policy_profile: str = typer.Option(
        "planner",
        "--candidate-policy-profile",
        help="Candidate policy profile.",
    ),
    baseline_predictor_model_path: Path | None = typer.Option(
        None,
        "--baseline-predictor-model-path",
        help="Optional baseline predictor model path.",
    ),
    baseline_predictor_mode: str = typer.Option(
        "heuristic_only",
        "--baseline-predictor-mode",
        help="Baseline predictor guidance mode: heuristic_only, assist, or dominant.",
    ),
    baseline_predictor_hook: list[str] | None = typer.Option(
        None,
        "--baseline-predictor-hook",
        help="Repeatable baseline predictor hook filter.",
    ),
    candidate_predictor_model_path: Path | None = typer.Option(
        None,
        "--candidate-predictor-model-path",
        help="Optional candidate predictor model path.",
    ),
    candidate_predictor_mode: str = typer.Option(
        "heuristic_only",
        "--candidate-predictor-mode",
        help="Candidate predictor guidance mode: heuristic_only, assist, or dominant.",
    ),
    candidate_predictor_hook: list[str] | None = typer.Option(
        None,
        "--candidate-predictor-hook",
        help="Repeatable candidate predictor hook filter.",
    ),
    replace_existing: bool = typer.Option(
        False,
        "--replace-existing/--no-replace-existing",
        help="Replace an existing session directory instead of failing.",
    ),
) -> None:
    baseline_predictor_config = _build_predictor_runtime_config(
        model_path=baseline_predictor_model_path,
        mode=baseline_predictor_mode,
        hooks=baseline_predictor_hook,
    )
    candidate_predictor_config = _build_predictor_runtime_config(
        model_path=candidate_predictor_model_path,
        mode=candidate_predictor_mode,
        hooks=candidate_predictor_hook,
    )
    report = run_shadow_combat_comparison(
        source=source,
        output_root=output_root,
        session_name=session_name,
        baseline_policy_profile=baseline_policy_profile,
        candidate_policy_profile=candidate_policy_profile,
        baseline_predictor_config=baseline_predictor_config,
        candidate_predictor_config=candidate_predictor_config,
        replace_existing=replace_existing,
    )
    summary_payload = load_shadow_combat_report(report.summary_path)
    delta_metrics = summary_payload["delta_metrics"]
    table = Table(title=f"Shadow Combat Compare ({baseline_policy_profile} vs {candidate_policy_profile})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Encounters", str(summary_payload["encounter_count"]))
    table.add_row("Comparable", str(summary_payload["comparable_encounter_count"]))
    table.add_row("Agreement", _format_metric(summary_payload.get("agreement_rate")))
    table.add_row("Candidate Advantage", _format_metric(summary_payload.get("candidate_advantage_rate")))
    table.add_row("Delta First Action Match", _format_metric(delta_metrics.get("delta_first_action_match_rate")))
    table.add_row("Delta Trace Hit", _format_metric(delta_metrics.get("delta_trace_hit_rate")))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Comparisons Path", str(report.comparisons_path))
    console.print(table)


@shadow_app.command("summary")
def shadow_summary_command(
    source: Path = typer.Option(
        ...,
        "--source",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Shadow report directory or summary.json path.",
    ),
) -> None:
    summary_payload = load_shadow_combat_report(source)
    report_kind = str(summary_payload.get("report_kind", "shadow"))
    if report_kind == "shadow_combat_compare":
        console.print(
            _json_table(
                "Shadow Combat Compare Summary",
                [
                    ("Source", summary_payload.get("source_path")),
                    ("Baseline Policy", summary_payload.get("baseline_policy_profile")),
                    ("Candidate Policy", summary_payload.get("candidate_policy_profile")),
                    ("Encounters", summary_payload.get("encounter_count")),
                    ("Comparable", summary_payload.get("comparable_encounter_count")),
                    ("Agreement", summary_payload.get("agreement_rate")),
                    ("Candidate Advantage", summary_payload.get("candidate_advantage_rate")),
                    ("Delta Metrics", summary_payload.get("delta_metrics")),
                    ("Families", summary_payload.get("encounter_family_histogram")),
                    ("Summary Path", summary_payload.get("summary_path")),
                ],
            )
        )
        return

    console.print(
        _json_table(
            "Shadow Combat Eval Summary",
            [
                ("Source", summary_payload.get("source_path")),
                ("Policy", summary_payload.get("policy_profile")),
                ("Encounters", summary_payload.get("encounter_count")),
                ("Usable", summary_payload.get("usable_encounter_count")),
                ("Skipped", summary_payload.get("skipped_encounter_count")),
                ("Metrics", summary_payload.get("metrics")),
                ("Families", summary_payload.get("encounter_family_histogram")),
                ("Bosses", summary_payload.get("boss_histogram")),
                ("Summary Path", summary_payload.get("summary_path")),
            ],
        )
    )


@registry_app.command("init")
def registry_init_command(
    root: Path = typer.Option(
        Path("artifacts/registry"),
        "--root",
        help="Registry root directory.",
    ),
    registry_name: str = typer.Option("sts2-rl-local", "--registry-name"),
    replace_existing: bool = typer.Option(
        False,
        "--replace-existing/--no-replace-existing",
        help="Replace an existing registry root.",
    ),
) -> None:
    report = initialize_registry(root, registry_name=registry_name, replace_existing=replace_existing)
    console.print(
        _json_table(
            f"Registry Init ({report.root_dir})",
            [
                ("Manifest Path", report.manifest_path),
                ("Experiments Dir", report.experiments_dir),
                ("Reports Dir", report.reports_dir),
                ("Aliases Path", report.aliases_path),
                ("Alias History Path", report.alias_history_path),
            ],
        )
    )


@registry_app.command("register")
def registry_register_command(
    source: Path = typer.Option(
        ...,
        "--source",
        help="Artifact summary file or artifact directory.",
    ),
    root: Path = typer.Option(
        Path("artifacts/registry"),
        "--root",
        help="Registry root directory.",
    ),
    experiment_id: str | None = typer.Option(
        None,
        "--experiment-id",
        help="Optional explicit experiment id.",
    ),
    tag: list[str] | None = typer.Option(
        None,
        "--tag",
        help="Repeatable tag applied to the registry entry.",
    ),
    alias: list[str] | None = typer.Option(
        None,
        "--alias",
        help="Repeatable alias to assign after registration.",
    ),
    notes: str | None = typer.Option(None, "--notes"),
    replace_existing: bool = typer.Option(
        False,
        "--replace-existing/--no-replace-existing",
        help="Replace an existing registry entry when no aliases point to it.",
    ),
) -> None:
    report = register_experiment(
        root,
        source=source,
        experiment_id=experiment_id,
        tags=tag,
        aliases=alias,
        notes=notes,
        replace_existing=replace_existing,
    )
    console.print(
        _json_table(
            f"Registry Register ({report.experiment_id})",
            [
                ("Family", report.family),
                ("Artifact Kind", report.artifact_kind),
                ("Display Name", report.display_name),
                ("Entry Path", report.entry_path),
                ("Source Summary Path", report.source_summary_path),
                ("Primary Metric", report.primary_metric_name),
                ("Primary Value", report.primary_metric_value),
            ],
        )
    )


@registry_app.command("list")
def registry_list_command(
    root: Path = typer.Option(
        Path("artifacts/registry"),
        "--root",
        help="Registry root directory.",
    ),
    family: str | None = typer.Option(None, "--family"),
    tag: str | None = typer.Option(None, "--tag"),
    alias: str | None = typer.Option(None, "--alias"),
) -> None:
    entries = list_registry_experiments(root, family=family, tag=tag, alias=alias)
    table = Table(title=f"Registry Entries ({root.expanduser().resolve()})")
    table.add_column("Experiment")
    table.add_column("Family")
    table.add_column("Kind")
    table.add_column("Primary")
    table.add_column("Aliases")
    for entry in entries:
        primary = dict(entry.get("metrics", {}).get("primary") or {})
        primary_text = "-"
        if primary:
            primary_text = f"{primary.get('name')}={_format_metric(primary.get('value'))}"
        table.add_row(
            str(entry["experiment_id"]),
            str(entry["family"]),
            str(entry["artifact_kind"]),
            primary_text,
            json.dumps(entry.get("aliases", []), ensure_ascii=False),
        )
    console.print(table)


@registry_app.command("show")
def registry_show_command(
    experiment: str = typer.Option(
        ...,
        "--experiment",
        help="Experiment id or alias.",
    ),
    root: Path = typer.Option(
        Path("artifacts/registry"),
        "--root",
        help="Registry root directory.",
    ),
) -> None:
    entry = get_registry_experiment(root, experiment)
    console.print(
        _json_table(
            f"Registry Experiment ({entry['experiment_id']})",
            [
                ("Family", entry.get("family")),
                ("Artifact Kind", entry.get("artifact_kind")),
                ("Display Name", entry.get("display_name")),
                ("Aliases", entry.get("aliases")),
                ("Tags", entry.get("tags")),
                ("Source Summary Path", entry.get("source_summary_path")),
                ("Output Dir", entry.get("output_dir")),
                ("Primary Metric", dict(entry.get("metrics", {}).get("primary") or {})),
                ("Lineage", entry.get("lineage")),
                ("References", entry.get("references")),
                ("Artifact Paths", entry.get("artifact_paths")),
            ],
        )
    )


@registry_app.command("leaderboard")
def registry_leaderboard_command(
    root: Path = typer.Option(
        Path("artifacts/registry"),
        "--root",
        help="Registry root directory.",
    ),
    output_root: Path | None = typer.Option(
        None,
        "--output-root",
        help="Optional custom output root. Defaults to <registry>/reports.",
    ),
    session_name: str | None = typer.Option(None, "--session-name"),
    family: str | None = typer.Option(None, "--family"),
    tag: str | None = typer.Option(None, "--tag"),
    benchmark_suite_name: str | None = typer.Option(None, "--benchmark-suite-name"),
) -> None:
    report = build_registry_leaderboard(
        root,
        output_root=output_root,
        session_name=session_name,
        family=family,
        tag=tag,
        benchmark_suite_name=benchmark_suite_name,
    )
    summary_payload = json.loads(report.summary_path.read_text(encoding="utf-8"))
    table = Table(title=f"Registry Leaderboard ({report.output_dir})")
    table.add_column("Rank")
    table.add_column("Experiment")
    table.add_column("Family")
    table.add_column("Primary")
    for row in summary_payload["rows"][:20]:
        primary = dict(row.get("primary_metric") or {})
        primary_text = "-"
        if primary:
            primary_text = f"{primary.get('name')}={_format_metric(primary.get('value'))}"
        table.add_row(str(row["rank"]), str(row["experiment_id"]), str(row["family"]), primary_text)
    console.print(table)
    console.print(_json_table("Leaderboard Artifacts", [("Summary Path", report.summary_path), ("Markdown Path", report.markdown_path)]))


@registry_app.command("compare")
def registry_compare_command(
    experiment: list[str] = typer.Option(
        ...,
        "--experiment",
        help="Repeatable experiment id or alias.",
    ),
    root: Path = typer.Option(
        Path("artifacts/registry"),
        "--root",
        help="Registry root directory.",
    ),
    output_root: Path | None = typer.Option(
        None,
        "--output-root",
        help="Optional custom output root. Defaults to <registry>/reports.",
    ),
    session_name: str | None = typer.Option(None, "--session-name"),
) -> None:
    report = compare_registry_experiments(
        root,
        experiment_ids=experiment,
        output_root=output_root,
        session_name=session_name,
    )
    summary_payload = json.loads(report.summary_path.read_text(encoding="utf-8"))
    table = Table(title=f"Registry Compare ({report.output_dir})")
    table.add_column("Experiment")
    table.add_column("Family")
    table.add_column("Primary")
    for item in summary_payload["experiments"]:
        primary = dict(item.get("primary_metric") or {})
        primary_text = "-"
        if primary:
            primary_text = f"{primary.get('name')}={_format_metric(primary.get('value'))}"
        table.add_row(str(item["experiment_id"]), str(item["family"]), primary_text)
    console.print(table)
    console.print(_json_table("Compare Artifacts", [("Summary Path", report.summary_path), ("Markdown Path", report.markdown_path)]))


@registry_alias_app.command("list")
def registry_alias_list_command(
    root: Path = typer.Option(
        Path("artifacts/registry"),
        "--root",
        help="Registry root directory.",
    ),
) -> None:
    aliases = load_registry_aliases(root)
    table = Table(title=f"Registry Aliases ({root.expanduser().resolve()})")
    table.add_column("Alias")
    table.add_column("Experiment")
    table.add_column("Artifact")
    for alias_name, payload in sorted(aliases.items()):
        table.add_row(alias_name, str(payload.get("experiment_id")), str(payload.get("artifact_path") or "-"))
    console.print(table)


@registry_alias_app.command("set")
def registry_alias_set_command(
    alias_name: str = typer.Option(
        ...,
        "--alias-name",
        help="Alias to update.",
    ),
    experiment: str = typer.Option(
        ...,
        "--experiment",
        help="Experiment id or alias.",
    ),
    root: Path = typer.Option(
        Path("artifacts/registry"),
        "--root",
        help="Registry root directory.",
    ),
    artifact_path_key: str | None = typer.Option(
        None,
        "--artifact-path-key",
        help="Optional artifact path key to bind the alias to.",
    ),
    reason: str | None = typer.Option(None, "--reason"),
) -> None:
    resolved = get_registry_experiment(root, experiment)
    report = set_registry_alias(
        root,
        alias_name=alias_name,
        experiment_id=str(resolved["experiment_id"]),
        artifact_path_key=artifact_path_key,
        reason=reason,
    )
    console.print(
        _json_table(
            f"Registry Alias ({report.alias_name})",
            [
                ("Experiment", report.experiment_id),
                ("Artifact Path Key", report.artifact_path_key),
                ("Artifact Path", report.artifact_path),
                ("Aliases Path", report.aliases_path),
                ("Alias History Path", report.alias_history_path),
            ],
        )
    )


@registry_app.command("promote")
def registry_promote_command(
    alias_name: str = typer.Option(
        ...,
        "--alias-name",
        help="Promotion alias to update, for example best_bc or recommended_default.",
    ),
    experiment: str | None = typer.Option(
        None,
        "--experiment",
        help="Experiment id or alias to promote directly.",
    ),
    root: Path = typer.Option(
        Path("artifacts/registry"),
        "--root",
        help="Registry root directory.",
    ),
    family: str | None = typer.Option(
        None,
        "--family",
        help="If --experiment is omitted, pick the top leaderboard row for this family.",
    ),
    tag: str | None = typer.Option(None, "--tag"),
    benchmark_suite_name: str | None = typer.Option(None, "--benchmark-suite-name"),
    artifact_path_key: str | None = typer.Option(None, "--artifact-path-key"),
    reason: str | None = typer.Option(None, "--reason"),
) -> None:
    selected_entry = None
    if experiment is not None:
        selected_entry = get_registry_experiment(root, experiment)
    else:
        leaderboard = build_registry_leaderboard(
            root,
            family=family,
            tag=tag,
            benchmark_suite_name=benchmark_suite_name,
        )
        leaderboard_payload = json.loads(leaderboard.summary_path.read_text(encoding="utf-8"))
        if not leaderboard_payload["rows"]:
            raise typer.BadParameter("No leaderboard rows matched the provided filters.")
        selected_entry = get_registry_experiment(root, str(leaderboard_payload["rows"][0]["experiment_id"]))
    report = set_registry_alias(
        root,
        alias_name=alias_name,
        experiment_id=str(selected_entry["experiment_id"]),
        artifact_path_key=artifact_path_key,
        reason=reason or "promote",
        updated_by="promote",
    )
    console.print(
        _json_table(
            f"Registry Promote ({report.alias_name})",
            [
                ("Experiment", report.experiment_id),
                ("Artifact Path Key", report.artifact_path_key),
                ("Artifact Path", report.artifact_path),
                ("Aliases Path", report.aliases_path),
                ("Alias History Path", report.alias_history_path),
            ],
        )
    )


@predict_dataset_app.command("extract")
def predict_dataset_extract_command(
    source: list[Path] = typer.Option(
        ...,
        "--source",
        help="Repeatable combat-outcomes source. Each source may be a file or a directory searched recursively.",
    ),
    output_dir: Path = typer.Option(
        Path("data/predict/latest"),
        "--output-dir",
        help="Output directory for predictor examples and dataset summary.",
    ),
    replace_existing: bool = typer.Option(
        False,
        "--replace-existing/--no-replace-existing",
        help="Replace an existing output directory.",
    ),
    split_seed: int = typer.Option(0, "--split-seed"),
    train_fraction: float = typer.Option(0.8, "--train-fraction", min=0.0, max=1.0),
    validation_fraction: float = typer.Option(0.1, "--validation-fraction", min=0.0, max=1.0),
    test_fraction: float = typer.Option(0.1, "--test-fraction", min=0.0, max=1.0),
    split_group_by: str = typer.Option(
        "session_run",
        "--split-group-by",
        help="Split grouping mode: record, run_id, or session_run.",
    ),
) -> None:
    report = extract_predictor_dataset(
        source,
        output_dir=output_dir,
        replace_existing=replace_existing,
        split_seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        split_group_by=split_group_by,
    )

    table = Table(title=f"Predictor Dataset ({report.output_dir})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Source Paths", str(len(report.source_paths)))
    table.add_row("Combat Outcome Files", str(len(report.combat_outcome_paths)))
    table.add_row("Examples", str(report.example_count))
    table.add_row("Features", str(report.feature_count))
    table.add_row("Manifest Path", str(report.manifest_path))
    table.add_row("Examples Path", str(report.examples_path))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Split Counts", json.dumps(report.split_counts, ensure_ascii=False))
    table.add_row("Outcomes", json.dumps(report.outcome_histogram, ensure_ascii=False))
    table.add_row("Characters", json.dumps(report.character_histogram, ensure_ascii=False))
    console.print(table)


@predict_train_app.command("combat-outcome")
def predict_train_combat_outcome_command(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Predictor dataset directory or examples.jsonl file.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/predict"),
        "--output-root",
        help="Root directory for predictor training outputs.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional predictor training session name. Defaults to a UTC timestamp.",
    ),
    epochs: int = typer.Option(250, "--epochs", min=1),
    learning_rate: float = typer.Option(0.05, "--learning-rate", min=0.000001),
    l2: float = typer.Option(0.0005, "--l2", min=0.0),
    validation_fraction: float = typer.Option(0.2, "--validation-fraction", min=0.0, max=0.9),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Training RNG seed for predictor sample ordering and initialization. Does not set the in-game run seed.",
    ),
) -> None:
    session_name = session_name or default_predictor_training_session_name()
    report = train_combat_outcome_predictor(
        dataset_source=dataset,
        output_root=output_root,
        session_name=session_name,
        config=CombatOutcomePredictorTrainConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            validation_fraction=validation_fraction,
            seed=seed,
        ),
    )

    table = Table(title=f"Predictor Training ({report.output_dir})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Examples", str(report.example_count))
    table.add_row("Train Examples", str(report.train_example_count))
    table.add_row("Validation Examples", str(report.validation_example_count))
    table.add_row("Split Strategy", report.split_strategy)
    table.add_row("Feature Count", str(report.feature_count))
    table.add_row("Best Epoch", str(report.best_epoch))
    table.add_row("Examples Path", str(report.examples_path) if report.examples_path is not None else "-")
    table.add_row(
        "Train Split Path",
        str(report.train_examples_path) if report.train_examples_path is not None else "-",
    )
    table.add_row(
        "Validation Split Path",
        str(report.validation_examples_path) if report.validation_examples_path is not None else "-",
    )
    table.add_row("Model Path", str(report.model_path))
    table.add_row("Metrics Path", str(report.metrics_path))
    table.add_row("Summary Path", str(report.summary_path))
    console.print(table)


@predict_report_app.command("calibration")
def predict_report_calibration_command(
    model_path: Path = typer.Option(
        ...,
        "--model-path",
        help="Predictor model JSON path.",
    ),
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Predictor dataset directory or examples.jsonl file.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/predict-reports"),
        "--output-root",
        help="Root directory for predictor report artifacts.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional report session name. Defaults to a UTC timestamp.",
    ),
    public_aggregate_source: Path | None = typer.Option(
        None,
        "--public-aggregate-source",
        help="Optional community-card-stats artifact directory or jsonl file for public-source diagnostics.",
    ),
    public_run_source: Path | None = typer.Option(
        None,
        "--public-run-source",
        help="Optional normalized public-run artifact directory or strategic stats file for diagnostics.",
    ),
    split: str = typer.Option("validation", "--split"),
    bin_count: int = typer.Option(10, "--bin-count", min=2),
    min_slice_examples: int = typer.Option(5, "--min-slice-examples", min=1),
    outcome_ece_max: float = typer.Option(0.12, "--outcome-ece-max", min=0.0),
    outcome_brier_max: float = typer.Option(0.25, "--outcome-brier-max", min=0.0),
    reward_rmse_max: float = typer.Option(3.0, "--reward-rmse-max", min=0.0),
    damage_rmse_max: float = typer.Option(24.0, "--damage-rmse-max", min=0.0),
) -> None:
    session_name = session_name or default_predictor_report_session_name("predict-calibration")
    report = build_predictor_calibration_report(
        model_path=model_path,
        dataset_source=dataset,
        output_root=output_root,
        session_name=session_name,
        split=split,
        bin_count=bin_count,
        min_slice_examples=min_slice_examples,
        thresholds=PredictorCalibrationThresholds(
            outcome_ece_max=outcome_ece_max,
            outcome_brier_max=outcome_brier_max,
            reward_rmse_max=reward_rmse_max,
            damage_rmse_max=damage_rmse_max,
        ),
        public_aggregate_source=public_aggregate_source,
        public_run_source=public_run_source,
    )
    summary_payload = json.loads(report.summary_path.read_text(encoding="utf-8"))
    outcome_payload = summary_payload["overall"]["outcome_win_probability"]
    reward_payload = summary_payload["overall"]["expected_reward"]
    damage_payload = summary_payload["overall"]["expected_damage_delta"]

    table = Table(title=f"Predictor Calibration ({report.output_dir})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Examples", str(summary_payload["example_count"]))
    table.add_row("Split", str(summary_payload["split"]))
    table.add_row("Promotion Passed", str(summary_payload["promotion"]["passed"]))
    table.add_row("Outcome ECE", _format_metric(outcome_payload.get("ece")))
    table.add_row("Outcome Brier", _format_metric(outcome_payload.get("brier_score")))
    table.add_row("Reward RMSE", _format_metric(reward_payload.get("rmse")))
    table.add_row("Damage RMSE", _format_metric(damage_payload.get("rmse")))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Markdown Path", str(report.markdown_path))
    console.print(table)


@predict_report_app.command("ranking")
def predict_report_ranking_command(
    model_path: Path = typer.Option(
        ...,
        "--model-path",
        help="Predictor model JSON path.",
    ),
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Predictor dataset directory or examples.jsonl file.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/predict-reports"),
        "--output-root",
        help="Root directory for predictor report artifacts.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional report session name. Defaults to a UTC timestamp.",
    ),
    public_aggregate_source: Path | None = typer.Option(
        None,
        "--public-aggregate-source",
        help="Optional community-card-stats artifact directory or jsonl file for public-source diagnostics.",
    ),
    public_run_source: Path | None = typer.Option(
        None,
        "--public-run-source",
        help="Optional normalized public-run artifact directory or strategic stats file for diagnostics.",
    ),
    split: str = typer.Option("validation", "--split"),
    group_by: list[str] | None = typer.Option(
        None,
        "--group-by",
        help="Repeatable ranking group dimension. Defaults to character,floor_band,encounter_family.",
    ),
    top_k: int = typer.Option(3, "--top-k", min=1),
    min_group_size: int = typer.Option(2, "--min-group-size", min=2),
    outcome_pairwise_accuracy_min: float = typer.Option(0.58, "--outcome-pairwise-accuracy-min", min=0.0, max=1.0),
    reward_pairwise_accuracy_min: float = typer.Option(0.58, "--reward-pairwise-accuracy-min", min=0.0, max=1.0),
    damage_pairwise_accuracy_min: float = typer.Option(0.58, "--damage-pairwise-accuracy-min", min=0.0, max=1.0),
    reward_ndcg_at_3_min: float = typer.Option(0.62, "--reward-ndcg-min", min=0.0, max=1.0),
    damage_ndcg_at_3_min: float = typer.Option(0.62, "--damage-ndcg-min", min=0.0, max=1.0),
) -> None:
    session_name = session_name or default_predictor_report_session_name("predict-ranking")
    report = build_predictor_ranking_report(
        model_path=model_path,
        dataset_source=dataset,
        output_root=output_root,
        session_name=session_name,
        split=split,
        group_by=group_by,
        top_k=top_k,
        min_group_size=min_group_size,
        thresholds=PredictorRankingThresholds(
            outcome_pairwise_accuracy_min=outcome_pairwise_accuracy_min,
            reward_pairwise_accuracy_min=reward_pairwise_accuracy_min,
            damage_pairwise_accuracy_min=damage_pairwise_accuracy_min,
            reward_ndcg_at_3_min=reward_ndcg_at_3_min,
            damage_ndcg_at_3_min=damage_ndcg_at_3_min,
        ),
        public_aggregate_source=public_aggregate_source,
        public_run_source=public_run_source,
    )
    summary_payload = json.loads(report.summary_path.read_text(encoding="utf-8"))
    outcome_payload = summary_payload["overall"]["outcome_win_probability"]
    reward_payload = summary_payload["overall"]["expected_reward"]
    damage_payload = summary_payload["overall"]["expected_damage_delta"]

    table = Table(title=f"Predictor Ranking ({report.output_dir})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Examples", str(summary_payload["example_count"]))
    table.add_row("Groups", str(summary_payload["group_count"]))
    table.add_row("Group By", json.dumps(summary_payload["group_by"], ensure_ascii=False))
    table.add_row("Promotion Passed", str(summary_payload["promotion"]["passed"]))
    table.add_row("Outcome Pairwise", _format_metric(outcome_payload.get("pairwise_accuracy")))
    table.add_row("Reward Pairwise", _format_metric(reward_payload.get("pairwise_accuracy")))
    table.add_row(f"Reward NDCG@{summary_payload['top_k']}", _format_metric(reward_payload.get("ndcg_at_k")))
    table.add_row("Damage Pairwise", _format_metric(damage_payload.get("pairwise_accuracy")))
    table.add_row(f"Damage NDCG@{summary_payload['top_k']}", _format_metric(damage_payload.get("ndcg_at_k")))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Markdown Path", str(report.markdown_path))
    console.print(table)


@predict_report_app.command("compare")
def predict_report_compare_command(
    source: list[Path] = typer.Option(
        ...,
        "--source",
        help="Repeatable benchmark suite summary, benchmark suite dir, case summary, or case dir.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts/predict-reports"),
        "--output-root",
        help="Root directory for predictor report artifacts.",
    ),
    session_name: str | None = typer.Option(
        None,
        "--session-name",
        help="Optional report session name. Defaults to a UTC timestamp.",
    ),
    public_aggregate_source: Path | None = typer.Option(
        None,
        "--public-aggregate-source",
        help="Optional community-card-stats artifact directory or jsonl file for public-source diagnostics.",
    ),
    public_run_source: Path | None = typer.Option(
        None,
        "--public-run-source",
        help="Optional normalized public-run artifact directory or strategic stats file for diagnostics.",
    ),
    delta_total_reward_min: float = typer.Option(0.0, "--delta-total-reward-min"),
    delta_combat_win_rate_min: float = typer.Option(0.0, "--delta-combat-win-rate-min"),
) -> None:
    session_name = session_name or default_predictor_report_session_name("predict-benchmark-compare")
    report = build_predictor_benchmark_comparison_report(
        sources=source,
        output_root=output_root,
        session_name=session_name,
        thresholds=PredictorBenchmarkComparisonThresholds(
            delta_total_reward_min=delta_total_reward_min,
            delta_combat_win_rate_min=delta_combat_win_rate_min,
        ),
        public_aggregate_source=public_aggregate_source,
        public_run_source=public_run_source,
    )
    summary_payload = json.loads(report.summary_path.read_text(encoding="utf-8"))

    table = Table(title=f"Predictor Benchmark Compare ({report.output_dir})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Sources", str(len(summary_payload["source_paths"])))
    table.add_row("Cases", str(summary_payload["case_count"]))
    table.add_row("Compare Cases", str(summary_payload["compare_case_count"]))
    table.add_row("Promotion Passed", str(summary_payload["promotion"]["passed"]))
    table.add_row("Promotion Candidates", str(summary_payload["promotion"]["promotion_candidate_count"]))
    table.add_row("Rollback Signals", str(summary_payload["promotion"]["rollback_signal_count"]))
    table.add_row("Summary Path", str(report.summary_path))
    table.add_row("Markdown Path", str(report.markdown_path))
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
