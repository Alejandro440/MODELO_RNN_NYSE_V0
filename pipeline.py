"""Pipeline runner to execute the data preparation scripts sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

from pipeline_config import ensure_data_directories

REPO_DIR = Path(__file__).resolve().parent

STEPS = {
    "download-stocks": REPO_DIR / "01_download_stocks.py",
    "preprocess-index": REPO_DIR / "02_preprocess_index.py",
    "build-features": REPO_DIR / "03_build_features.py",
    "add-target": REPO_DIR / "04_create_target.py",
    "merge-index": REPO_DIR / "05_merge_index.py",
    "train-model": REPO_DIR / "train_model.py",
}

DEFAULT_STEPS: List[str] = [
    "download-stocks",
    "preprocess-index",
    "build-features",
    "add-target",
    "merge-index",
]


def run_step(step: str) -> None:
    script_path = STEPS[step]
    print(f"\n▶️ Ejecutando paso: {step} -> {script_path}\n")
    subprocess.run([sys.executable, str(script_path)], check=True)


def parse_steps(selected: Iterable[str], include_train: bool) -> List[str]:
    if not selected or selected == ["all"]:
        steps_to_run = list(DEFAULT_STEPS)
    else:
        steps_to_run = list(selected)

    if include_train and "train-model" not in steps_to_run:
        steps_to_run.append("train-model")

    return steps_to_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta la canalización de datos y entrenamiento.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(STEPS.keys()) + ["all"],
        default=["all"],
        help="Pasos a ejecutar. Usa 'all' para los pasos de datos y añade --include-train para entrenar.",
    )
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Incluye el entrenamiento del modelo al final de los pasos seleccionados.",
    )
    args = parser.parse_args()

    steps_to_run = parse_steps(args.steps, args.include_train)
    ensure_data_directories()

    for step in steps_to_run:
        run_step(step)


if __name__ == "__main__":
    main()