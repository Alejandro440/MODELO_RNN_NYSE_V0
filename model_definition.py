"""Model factory for the stock trend predictor."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Concatenate, Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


@dataclass
class ModelConfig:
    embedding_size: int = 16
    lstm_units: int = 254
    dense_units: int = 128
    dropout_rate: float = 0.3
    learning_rate: float = 0.0005
    l2_reg: float = 0.0005


def build_rnn_model(
    n_steps: int,
    feature_dim: int,
    sp500_dim: int,
    num_companies: int,
    config: ModelConfig | None = None,
) -> Model:
    cfg = config or ModelConfig()
    time_steps = n_steps - 1

    input_company = Input(shape=(time_steps,), name="company")
    input_features = Input(shape=(time_steps, feature_dim), name="features")
    input_sp500 = Input(shape=(time_steps, sp500_dim), name="sp500")

    embedding = Embedding(input_dim=num_companies, output_dim=cfg.embedding_size)(input_company)
    concatenated_inputs = Concatenate(axis=-1)([embedding, input_features, input_sp500])

    lstm_out = LSTM(
        units=cfg.lstm_units,
        return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(cfg.l2_reg),
        recurrent_dropout=0.2,
    )(concatenated_inputs)
    lstm_out = Dropout(rate=cfg.dropout_rate)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)

    lstm_out = LSTM(
        units=cfg.lstm_units,
        return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(cfg.l2_reg),
        recurrent_dropout=0.2,
    )(lstm_out)
    lstm_out = Dropout(rate=cfg.dropout_rate)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)

    lstm_out = LSTM(
        units=cfg.lstm_units,
        return_sequences=False,
        kernel_regularizer=tf.keras.regularizers.l2(cfg.l2_reg),
        recurrent_dropout=0.2,
    )(lstm_out)
    lstm_out = Dropout(rate=cfg.dropout_rate)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)

    dense_out = Dense(
        units=cfg.dense_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(cfg.l2_reg),
    )(lstm_out)
    dense_out = Dropout(rate=cfg.dropout_rate)(dense_out)
    dense_out = BatchNormalization()(dense_out)

    output = Dense(units=5, activation="softmax")(dense_out)

    model = Model(inputs=[input_company, input_features, input_sp500], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model