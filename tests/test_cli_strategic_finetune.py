from pathlib import Path

from typer.testing import CliRunner

from sts2_rl.cli import app
from tests.test_strategic_finetune import _write_public_dataset, _write_runtime_dataset

runner = CliRunner()


def test_strategic_finetune_cli_train_command(tmp_path: Path) -> None:
    runtime_dataset = _write_runtime_dataset(tmp_path)
    public_dataset = _write_public_dataset(tmp_path)

    result = runner.invoke(
        app,
        [
            "train",
            "strategic-finetune",
            "--runtime-dataset",
            str(runtime_dataset),
            "--public-dataset",
            str(public_dataset),
            "--output-root",
            str(tmp_path / "artifacts" / "strategic-finetune"),
            "--session-name",
            "cli-strategic-ft",
            "--epochs",
            "20",
            "--learning-rate",
            "0.08",
            "--runtime-build-id",
            "v0.103.0",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "artifacts" / "strategic-finetune" / "cli-strategic-ft" / "strategic-finetune-checkpoint.json").exists()
    assert (tmp_path / "artifacts" / "strategic-finetune" / "cli-strategic-ft" / "strategic-finetune-best.json").exists()
    assert (tmp_path / "artifacts" / "strategic-finetune" / "cli-strategic-ft" / "summary.json").exists()
