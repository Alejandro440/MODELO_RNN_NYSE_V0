"""Orchestrates training, evaluation, and reporting for the RNN model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from model_data import DateRange, prepare_training_data
from model_definition import ModelConfig, build_rnn_model
from pipeline_config import MODEL_OUTPUT_DIR, ensure_data_directories

# Use non-interactive backend for saving plots
plt.switch_backend("Agg")

CLASS_LABELS: Sequence[str] = [
    "Muy Alcista",
    "Alcista",
    "Muy Bajista",
    "Bajista",
    "Lateral",
]


def plot_training_history(history, output_dir: Path) -> None:
    """Save training/validation loss and accuracy plots."""
    history_dict = history.history
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history_dict["loss"], label="Pérdida de Entrenamiento")
    plt.plot(history_dict["val_loss"], label="Pérdida de Validación")
    plt.title("Pérdida durante el Entrenamiento y la Validación")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history_dict["accuracy"], label="Precisión de Entrenamiento")
    plt.plot(history_dict["val_accuracy"], label="Precisión de Validación")
    plt.title("Precisión durante el Entrenamiento y la Validación")
    plt.xlabel("Épocas")
    plt.ylabel("Precisión")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_plots.png")


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> np.ndarray:
    """Compute and save confusion matrix plot."""
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")

    return conf_matrix


def write_report(
    output_dir: Path,
    history,
    test_loss: float,
    test_accuracy: float,
    conf_matrix: np.ndarray,
    class_report: str,
) -> None:
    """Write a text report summarising training and test metrics."""
    history_dict = history.history
    report_path = output_dir / "model_training_report.txt"

    with report_path.open("w", encoding="utf-8") as report_file:
        report_file.write("Informe de Entrenamiento del Modelo\n")
        report_file.write("===================================\n\n")

        report_file.write("Resultados del Entrenamiento:\n")
        report_file.write(
            f"- Pérdida de Entrenamiento Final: {history_dict['loss'][-1]:.4f}\n"
        )
        report_file.write(
            f"- Pérdida de Validación Final: {history_dict['val_loss'][-1]:.4f}\n"
        )
        report_file.write(
            f"- Precisión de Entrenamiento Final: {history_dict['accuracy'][-1]:.4f}\n"
        )
        report_file.write(
            f"- Precisión de Validación Final: {history_dict['val_accuracy'][-1]:.4f}\n\n"
        )

        report_file.write("Evaluación en el Conjunto de Prueba:\n")
        report_file.write(f"- Pérdida en Test: {test_loss:.4f}\n")
        report_file.write(f"- Precisión en Test: {test_accuracy:.4f}\n\n")

        report_file.write("Matriz de Confusión:\n")
        report_file.write(f"{conf_matrix}\n\n")

        report_file.write("Reporte de Clasificación:\n")
        report_file.write(f"{class_report}\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Entrena el modelo RNN para predecir tendencias bursátiles."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número de épocas para el entrenamiento",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Tamaño de batch",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=120,
        help="Número de pasos para las secuencias",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2002-01-01",
        help="Fecha inicial de los datos (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-01-01",
        help="Fecha final de los datos (YYYY-MM-DD)",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()

    ensure_data_directories()
    output_dir = Path(MODEL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    date_range = DateRange(start=args.start_date, end=args.end_date)
    data = prepare_training_data(n_steps=args.n_steps, date_range=date_range)

    # Build model
    model = build_rnn_model(
        n_steps=data.n_steps,
        feature_dim=data.X_train.shape[2],
        sp500_dim=data.X_sp500_train.shape[2],
        num_companies=len(data.company_mapping),
        config=ModelConfig(),
    )

    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
    )

    # Train
    history = model.fit(
        [data.indices_train, data.X_train, data.X_sp500_train],
        data.y_train,
        validation_data=(
            [data.indices_test, data.X_test, data.X_sp500_test],
            data.y_test,
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stopping, reduce_lr],
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(
        [data.indices_test, data.X_test, data.X_sp500_test],
        data.y_test,
    )
    print(f"Pérdida en Test: {test_loss:.4f}")
    print(f"Precisión en Test: {test_accuracy:.4f}")

    # Predictions
    y_pred = model.predict(
        [data.indices_test, data.X_test, data.X_sp500_test],
    )
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Plots & report
    plot_training_history(history, output_dir)
    conf_matrix = plot_confusion(data.y_test, y_pred_classes, output_dir)
    class_report = classification_report(
        data.y_test,
        y_pred_classes,
        target_names=CLASS_LABELS,
    )

    write_report(
        output_dir,
        history,
        test_loss,
        test_accuracy,
        conf_matrix,
        class_report,
    )

    # Save model
    model_path = output_dir / "my_model.h5"
    model.save(model_path)

    print(f"Modelo guardado: {model_path}")
    print(f"Informe generado: {output_dir / 'model_training_report.txt'}")
    print(f"Gráficas de entrenamiento: {output_dir / 'training_plots.png'}")
    print(f"Matriz de confusión: {output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()

