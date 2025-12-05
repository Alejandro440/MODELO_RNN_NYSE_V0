# MODELO_RNN_NYSE
End-to-end RNN pipeline for NYSE stock trend prediction using historical price data and the S&P 500 index from Yahoo Finance.

## Tech stack

- **Language:** Python  
- **Core libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras  
- **Data source:** Yahoo Finance (yfinance)  
- **Visualization:** Matplotlib, Seaborn  
- **Orchestration:** CLI scripts + `pipeline.py`  
- **Outputs:** Trained model (`.h5`), training plots, confusion matrix, classification report


## Visión general del flujo
1. **Descarga de datos de acciones** (`01_download_stocks.py`): baja históricos de cientos de tickers NYSE y crea `data/raw/stocks` (reintentos y creación de carpetas incluidas).
2. **Descarga y procesado del índice S&P 500** (`02_preprocess_index.py`): obtiene el índice desde Yahoo Finance, limpia duplicados/nulos, calcula indicadores (SMA, EMA, MACD, RSI) y guarda crudos/procesados en `data/raw/indices` y `data/processed/indices`.
3. **Cálculo de indicadores por acción** (`03_build_features.py`): limpia datos de cada compañía, genera indicadores técnicos y escribe `*_Processed.csv` en `data/processed/stocks`.
4. **Creación de variable objetivo** (`04_create_target.py`): añade la columna `Trend` según retornos futuros (ventanas de 30-40 días) y guarda `*_Var.csv` en `data/processed/with_target`.
5. **Fusión con el índice** (`05_merge_index.py`): une cada acción con el S&P 500 procesado (prefijo `SP500_`), rellena valores faltantes y produce archivos en `data/processed/with_index` listos para entrenar.
6. **Entrenamiento y evaluación** (`train_model.py`): prepara secuencias temporales, entrena el modelo RNN, genera métricas, gráficas y artefactos en `data/models`.

Todos los pasos crean sus carpetas si no existen. El pipeline completo está orquestado por `pipeline.py` y centraliza rutas en `pipeline_config.py`.

## Estructura de datos
Al ejecutar los scripts se crean carpetas locales bajo `data/`:
- `data/raw/stocks`: descargas originales de Yahoo Finance.
- `data/raw/indices`: índice S&P 500 crudo descargado automáticamente.
- `data/processed/stocks`: datos de acciones con indicadores técnicos.
- `data/processed/with_target`: datos con la variable objetivo `Trend`.
- `data/processed/indices`: índice procesado con indicadores técnicos.
- `data/processed/with_index`: unión de acciones con el índice.
- `data/models`: artefactos del entrenamiento (modelo y reportes).

## Orquestación con pipeline
`pipeline.py` ejecuta los pasos en orden usando las rutas definidas en `pipeline_config.py`. Puedes:
- **Correr todo el pipeline de datos:**
  ```bash
  python pipeline.py
  ```
- **Añadir el entrenamiento al final:**
  ```bash
  python pipeline.py --include-train
  ```
- **Elegir pasos concretos:**
  ```bash
  python pipeline.py --steps build-features add-target
  ```

### Configuración del pipeline (`pipeline_config.py`)
- Centraliza rutas como `RAW_STOCKS_DIR`, `PROCESSED_INDEX_DIR`, `WITH_INDEX_DIR`, `MODELS_DIR` y crea las carpetas si no existen.
- Define la lista por defecto de pasos que consume `pipeline.py` para asegurar un flujo secuencial coherente.
- Puedes modificar rutas o añadir nuevas carpetas (por ejemplo, para pruebas) en un único sitio sin tocar el resto de scripts.

### Detalle de `pipeline.py`
- Implementa un registro de pasos (`steps_registry`) que mapea nombres legibles a funciones concretas (descarga, indicadores, variable objetivo, fusión, etc.).
- Valida los nombres de pasos que recibe por CLI y los ejecuta en orden, mostrando logs amigables.
- Opción `--include-train` agrega el paso de entrenamiento estándar al final, reutilizando los datos generados en el pipeline de preparación.

## Preparación de datos para el modelo
### `model_data.py`
- **Carga** los CSV fusionados en `data/processed/with_index`, ignora archivos vacíos o fuera del rango de fechas y ordena cronológicamente por fecha.
- **Comprobación de columnas:** garantiza que las columnas esperadas (tanto de acciones como del S&P 500) existan, creando vacíos si faltan para evitar fallos al escalar.
- **Escalado por compañía sin fuga:** divide primero cada compañía en train/test respetando el orden temporal y **ajusta el `StandardScaler` solo con el train** antes de transformar ambos subconjuntos, evitando leak.
- **Generación de secuencias:** construye ventanas temporales de longitud `n_steps` con características de la acción e índice, y genera etiquetas de tendencia alineadas.
- **Split train/test temporal:** reparte cada compañía en proporción (2/3 train, 1/3 test) usando las primeras fechas para train y las más recientes para test; entrega tensores listos para el modelo junto con el mapeo de compañías.

### `model_definition.py`
- Define la arquitectura RNN estándar con:
  - **Embedding** de compañía para capturar diferencias entre empresas.
  - **Capas LSTM** apiladas con dropout para modelar dependencias temporales.
  - **Capa densa** final para clasificar la tendencia.
- Expone una función `build_model` que recibe hiperparámetros clave (tamaño de embedding, unidades LSTM, dropout, número de clases) y devuelve un modelo ya compilado con `optimizer='adam'` y pérdida de clasificación.

## Entrenamiento y evaluación
### `train_model.py`

- CLI configurable para ajustar hiperparámetros y rango temporal sin tocar código:
  - `--epochs`, `--batch-size`, `--n-steps` para la configuración de las secuencias y del entrenamiento.
  - `--start-date` y `--end-date` para acotar el historial usado.

- Flujo interno:
  1. Llama a `prepare_training_data` de `model_data.py` con el rango y `n_steps` solicitados.
  2. Construye el modelo vía `build_rnn_model` de `model_definition.py`, respetando el número de compañías y clases detectadas.
  3. Ajusta el modelo con callbacks (early stopping y reducción de LR), evalúa en test y guarda:
     - Modelo `.h5` y pesos.
     - Gráficas de pérdida/precisión.
     - Matriz de confusión y reporte de clasificación.

- Salida: todos los artefactos se escriben en `data/models`.

## Entrenamiento manual
Si prefieres entrenar de forma aislada tras generar los CSV fusionados:
```bash
python train_model.py --epochs 30 --batch-size 128 --n-steps 120 --start-date 2002-01-01 --end-date 2024-01-01
```
El script prepara los datos, entrena, evalúa en test y guarda el modelo, matriz de confusión, gráficas y reporte en `data/models`.

## Notas profesionales
- El flujo recomendado y único es el estándar (`pipeline.py` + `train_model.py`).
- No se requieren scripts adicionales para entrenar: se eliminó el flujo MODELO15 para evitar duplicidad y simplificar la operación.
- Ajusta parámetros (fechas, `n_steps`, batch, épocas, callbacks) mediante CLI sin tocar código.
