from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "COMBAT_OUTCOMES_FILENAME": (".dataset", "COMBAT_OUTCOMES_FILENAME"),
    "DATASET_SUMMARY_FILENAME": (".dataset", "DATASET_SUMMARY_FILENAME"),
    "PREDICTOR_EXAMPLES_FILENAME": (".dataset", "PREDICTOR_EXAMPLES_FILENAME"),
    "DatasetExtractionReport": (".dataset", "DatasetExtractionReport"),
    "discover_combat_outcome_paths": (".dataset", "discover_combat_outcome_paths"),
    "extract_predictor_dataset": (".dataset", "extract_predictor_dataset"),
    "load_predictor_examples": (".dataset", "load_predictor_examples"),
    "resolve_predictor_examples_path": (".dataset", "resolve_predictor_examples_path"),
    "extract_feature_map_from_summary": (".features", "extract_feature_map_from_summary"),
    "CombatOutcomePredictor": (".model", "CombatOutcomePredictor"),
    "PredictorHead": (".model", "PredictorHead"),
    "PredictorBenchmarkComparisonThresholds": (".reports", "PredictorBenchmarkComparisonThresholds"),
    "PredictorCalibrationThresholds": (".reports", "PredictorCalibrationThresholds"),
    "PredictorRankingThresholds": (".reports", "PredictorRankingThresholds"),
    "PredictorReportArtifacts": (".reports", "PredictorReportArtifacts"),
    "build_predictor_benchmark_comparison_report": (".reports", "build_predictor_benchmark_comparison_report"),
    "build_predictor_calibration_report": (".reports", "build_predictor_calibration_report"),
    "build_predictor_ranking_report": (".reports", "build_predictor_ranking_report"),
    "default_predictor_report_session_name": (".reports", "default_predictor_report_session_name"),
    "PredictorScores": (".model", "PredictorScores"),
    "PREDICTOR_GUIDANCE_HOOKS": (".runtime", "PREDICTOR_GUIDANCE_HOOKS"),
    "PREDICTOR_GUIDANCE_MODES": (".runtime", "PREDICTOR_GUIDANCE_MODES"),
    "PredictorGuidanceHook": (".runtime", "PredictorGuidanceHook"),
    "PredictorGuidanceMode": (".runtime", "PredictorGuidanceMode"),
    "PredictorRuntimeAdapter": (".runtime", "PredictorRuntimeAdapter"),
    "PredictorRuntimeConfig": (".runtime", "PredictorRuntimeConfig"),
    "PredictorRuntimeTrace": (".runtime", "PredictorRuntimeTrace"),
    "normalize_predictor_hooks": (".runtime", "normalize_predictor_hooks"),
    "normalize_predictor_mode": (".runtime", "normalize_predictor_mode"),
    "CombatOutcomePredictorTrainConfig": (".trainer", "CombatOutcomePredictorTrainConfig"),
    "CombatOutcomePredictorTrainingReport": (".trainer", "CombatOutcomePredictorTrainingReport"),
    "default_predictor_training_session_name": (".trainer", "default_predictor_training_session_name"),
    "train_combat_outcome_predictor": (".trainer", "train_combat_outcome_predictor"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - Python import protocol
        raise AttributeError(name) from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
