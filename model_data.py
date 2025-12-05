"""Utilities to load merged company data and build training sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline_config import MERGED_WITH_INDEX_DIR

FEATURE_COLS: List[str] = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_30",
    "SMA_50",
    "EMA_30",
    "EMA_50",
    "MACD",
    "Signal_Line",
    "RSI",
    "Upper_Band",
    "Lower_Band",
]

TARGET_COL = "Trend"

SP500_COLS: List[str] = [
    "SP500_Open",
    "SP500_High",
    "SP500_Low",
    "SP500_Close",
    "SP500_Adj Close",
    "SP500_Volume",
    "SP500_SMA_30",
    "SP500_SMA_50",
    "SP500_EMA_30",
    "SP500_EMA_50",
    "SP500_MACD",
    "SP500_Signal_Line",
    "SP500_RSI",
]


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza que el DataFrame tenga todas las columnas necesarias.

    Cualquier columna faltante se crea con valores 0 para evitar fallos en el
    escalado o la generación de secuencias cuando haya archivos incompletos.
    """

    df = df.copy()
    required_cols = FEATURE_COLS + SP500_COLS + [TARGET_COL]
    missing_cols: List[str] = []

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0
            missing_cols.append(col)

    if missing_cols:
        print(f"Columnas faltantes añadidas con 0: {', '.join(missing_cols)}")

    return df


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_sp500_train: np.ndarray
    y_train: np.ndarray
    indices_train: np.ndarray
    X_test: np.ndarray
    X_sp500_test: np.ndarray
    y_test: np.ndarray
    indices_test: np.ndarray
    company_mapping: Dict[str, int]
    n_steps: int


@dataclass
class DateRange:
    start: str = "2002-01-01"
    end: str = "2024-01-01"


def split_data(df: pd.DataFrame, train_size: float = 2 / 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_rows = len(df)
    train_rows = int(total_rows * train_size)
    return df.iloc[:train_rows], df.iloc[train_rows:]


def load_company_frames(
    directory_path: Path = Path(MERGED_WITH_INDEX_DIR),
    date_range: DateRange | None = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
    date_range = date_range or DateRange()
    company_data: Dict[str, pd.DataFrame] = {}
    company_mapping: Dict[str, int] = {}

    for file_path in sorted(directory_path.glob("*.csv")):
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[(df["Date"] >= date_range.start) & (df["Date"] <= date_range.end)]
        df.sort_values("Date", inplace=True)

        if df.empty:
            print(f"No hay datos dentro del rango de fechas para {file_path.name}, se omite este archivo.")
            continue

        for company in df["Company"].unique():
            if company not in company_mapping:
                company_mapping[company] = len(company_mapping)

            company_df = df[df["Company"] == company].copy()
            if company_df.empty:
                continue

            company_df = ensure_required_columns(company_df)

            if company in company_data:
                company_data[company] = pd.concat([company_data[company], company_df], ignore_index=True)
            else:
                company_data[company] = company_df

    return company_data, company_mapping


def scale_company_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    """Scale numeric features while keeping labels untouched."""
    scaler = StandardScaler()
    feature_set = FEATURE_COLS + SP500_COLS

    # Preserve labels to avoid leaking scaled targets
    train_labels = train_df[TARGET_COL].copy()
    test_labels = test_df[TARGET_COL].copy()

    scaler.fit(train_df[feature_set])
    train_scaled = scaler.transform(train_df[feature_set])
    test_scaled = scaler.transform(test_df[feature_set])

    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()

    train_df_scaled.loc[:, feature_set] = train_scaled
    test_df_scaled.loc[:, feature_set] = test_scaled

    # Restore original labels so they remain in the 0-4 range
    train_df_scaled[TARGET_COL] = train_labels.values
    test_df_scaled[TARGET_COL] = test_labels.values

    return train_df_scaled, test_df_scaled


def create_sequences(
    df: pd.DataFrame,
    company_index: int,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_array = df[FEATURE_COLS].values.astype("float32")
    sp500_array = df[SP500_COLS].values.astype("float32")
    labels = df[TARGET_COL].values.astype("float32")

    X: List[np.ndarray] = []
    X_sp500: List[np.ndarray] = []
    y: List[np.ndarray] = []
    indices: List[np.ndarray] = []

    if len(df) <= n_steps:
        return np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,))
    
    invalid_labels = 0

    for i in range(len(feature_array) - n_steps):
        seq_features = feature_array[i : i + n_steps - 1]
        seq_sp500_features = sp500_array[i : i + n_steps - 1]
        seq_label = labels[i + n_steps - 1]

        if np.isnan(seq_label) or not 0 <= seq_label <= 4:
            invalid_labels += 1
            continue

        X.append(seq_features)
        X_sp500.append(seq_sp500_features)
        y.append(int(seq_label))
        indices.append(np.full((n_steps - 1,), company_index, dtype="int32"))

    if invalid_labels: 
        print(
            f"Secuencias descartadas por etiquetas inválidas fuera de 0-4: {invalid_labels}"
        )    

    return (
        np.array(X, dtype="float32"),
        np.array(X_sp500, dtype="float32"),
        np.array(y, dtype="int32"),
        np.array(indices, dtype="int32"),
    )


def prepare_training_data(
    directory_path: Path = Path(MERGED_WITH_INDEX_DIR),
    n_steps: int = 120,
    date_range: DateRange | None = None,
    train_size: float = 2 / 3,
) -> PreparedData:
    company_data, company_mapping = load_company_frames(directory_path=directory_path, date_range=date_range)

    all_X_train: List[np.ndarray] = []
    all_X_test: List[np.ndarray] = []
    all_y_train: List[np.ndarray] = []
    all_y_test: List[np.ndarray] = []
    all_X_sp500_train: List[np.ndarray] = []
    all_X_sp500_test: List[np.ndarray] = []
    all_indices_train: List[np.ndarray] = []
    all_indices_test: List[np.ndarray] = []

    for company, df in company_data.items():
        company_index = company_mapping[company]
        print(f"Procesando compañía: {company}, Índice: {company_index}")

        train_df, test_df = split_data(df, train_size=train_size)

        train_df = ensure_required_columns(train_df)
        test_df = ensure_required_columns(test_df)

        if train_df.empty or test_df.empty:
            print(f"Datos insuficientes para {company}; se omite.")
            continue

        train_df_scaled, test_df_scaled = scale_company_features(train_df, test_df)

        X_train, X_sp500_train, y_train, indices_train = create_sequences(
            train_df_scaled, company_index, n_steps
        )
        X_test, X_sp500_test, y_test, indices_test = create_sequences(test_df_scaled, company_index, n_steps)

        if X_train.size == 0 or X_test.size == 0:
            print(f"No se pudieron crear secuencias para {company}; se omite.")
            continue

        all_X_train.extend(X_train)
        all_X_test.extend(X_test)
        all_y_train.extend(y_train)
        all_y_test.extend(y_test)
        all_X_sp500_train.extend(X_sp500_train)
        all_X_sp500_test.extend(X_sp500_test)
        all_indices_train.extend(indices_train)
        all_indices_test.extend(indices_test)

    X_train_array = np.array(all_X_train, dtype="float32")
    X_sp500_train_array = np.array(all_X_sp500_train, dtype="float32")
    y_train_array = np.array(all_y_train, dtype="int32")
    indices_train_array = np.array(all_indices_train, dtype="int32")

    X_test_array = np.array(all_X_test, dtype="float32")
    X_sp500_test_array = np.array(all_X_sp500_test, dtype="float32")
    y_test_array = np.array(all_y_test, dtype="int32")
    indices_test_array = np.array(all_indices_test, dtype="int32")

    if X_train_array.size == 0 or X_test_array.size == 0:
        raise ValueError("No se generaron secuencias para entrenamiento o prueba. Revisa los datos de entrada.")

    print(f"Forma de X_train: {X_train_array.shape}")
    print(f"Forma de X_sp500_train: {X_sp500_train_array.shape}")
    print(f"Forma de indices_train: {indices_train_array.shape}")
    print(f"Forma de y_train: {y_train_array.shape}")

    print(f"Forma de X_test: {X_test_array.shape}")
    print(f"Forma de X_sp500_test: {X_sp500_test_array.shape}")
    print(f"Forma de indices_test: {indices_test_array.shape}")
    print(f"Forma de y_test: {y_test_array.shape}")

    return PreparedData(
        X_train=X_train_array,
        X_sp500_train=X_sp500_train_array,
        y_train=y_train_array,
        indices_train=indices_train_array,
        X_test=X_test_array,
        X_sp500_test=X_sp500_test_array,
        y_test=y_test_array,
        indices_test=indices_test_array,
        company_mapping=company_mapping,
        n_steps=n_steps,
    )