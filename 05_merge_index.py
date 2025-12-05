import glob
import logging
import os

import pandas as pd

from pipeline_config import INDEX_PROCESSED_DIR, MERGED_WITH_INDEX_DIR, TARGET_STOCKS_DIR, ensure_data_directories

def merge_stock_with_index(stocks_folder, index_file, output_folder):
    # Configurar el logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Leer el archivo del índice
    try:
        index_data = pd.read_csv(index_file, parse_dates=['Date'])
        index_data.rename(columns=lambda x: f"SP500_{x}" if x != "Date" else x, inplace=True)
        index_data['Date'] = pd.to_datetime(index_data['Date']).dt.date  # Convertir a fecha sin hora
    except Exception as e:
        logging.error(f"Error al leer el archivo del índice: {e}")
        return

    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Leer los archivos de las acciones
    stock_files = glob.glob(os.path.join(stocks_folder, "*.csv"))

    for stock_file in stock_files:
        try:
            # Leer el archivo de la acción
            stock_data = pd.read_csv(stock_file, parse_dates=['Date'])

            # Verificar si la columna 'Date' existe y convertir a UTC si tiene zona horaria
            if 'Date' in stock_data.columns:
                stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True).dt.date

            # Fusionar los datos basándonos en la columna 'Date'
            merged_data = pd.merge(stock_data, index_data, on='Date', how='left')

            # Rellenar los valores NaN con los valores anteriores o con 0 si no hay anterior
            for column in index_data.columns:
                if column != 'Date':
                    merged_data[column] = merged_data[column].ffill().fillna(0)

            # Crear el nombre del archivo de salida
            stock_filename = os.path.basename(stock_file)
            output_filename = f"{os.path.splitext(stock_filename)[0]}_Index.csv"
            output_filepath = os.path.join(output_folder, output_filename)

            # Guardar el archivo de salida
            merged_data.to_csv(output_filepath, index=False)
            logging.info(f"Archivo guardado: {output_filepath}")

        except Exception as e:
            logging.error(f"Error al procesar el archivo {stock_file}: {e}")

# Rutas de los archivos y carpetas
def main():
    ensure_data_directories()
    merge_stock_with_index(
        TARGET_STOCKS_DIR,
        INDEX_PROCESSED_DIR / "sp500_processed.csv",
        MERGED_WITH_INDEX_DIR,
    )


if __name__ == "__main__":
    main()