
import os

import numpy as np
import pandas as pd

from pipeline_config import PROCESSED_STOCKS_DIR, TARGET_STOCKS_DIR, ensure_data_directories

# Función para calcular el precio de cierre promedio futuro
def calcular_precio_cierre_promedio(precios, dia_actual, rango_dias):
    inicio, fin = rango_dias
    if dia_actual + fin <= len(precios):
        return precios[dia_actual + inicio: dia_actual + fin].mean()
    else:
        return np.nan

# Función para calcular el cambio porcentual
def calcular_cambio_porcentual(precio_actual, precio_futuro):
    if precio_actual == 0:
        return np.nan
    else:
        return ((precio_futuro - precio_actual) / precio_actual) * 100

# Función para asignar la categoría de tendencia
def asignar_tendencia(cambio_porcentual):
    if cambio_porcentual >= 15:
        return 0  # MUY ALZISTA
    elif 5 <= cambio_porcentual < 15:
        return 1  # ALZISTA
    elif cambio_porcentual <= -15:
        return 2  # MUY BAJISTA
    elif -15 < cambio_porcentual <= -5:
        return 3  # BAJISTA
    else:
        return 4  # LATERAL

# Función para procesar un archivo CSV, calcular la tendencia, y manejar el nombre del archivo de salida
def procesar_archivo_csv_eliminar_ultimas(ruta_archivo, output_dir, rango_dias_futuros=(30, 40), num_entradas_eliminar=50):
    df = pd.read_csv(ruta_archivo)
    
    precios_cierre_promedio_futuro = [
        calcular_precio_cierre_promedio(df['Close'], i, rango_dias_futuros)
        for i in range(len(df))
    ]
    
    cambios_porcentuales = [
        calcular_cambio_porcentual(df.loc[i, 'Close'], precios_cierre_promedio_futuro[i])
        for i in range(len(df))
    ]
    
    df['Trend'] = [asignar_tendencia(cambio) for cambio in cambios_porcentuales]
    df_final = df[:-num_entradas_eliminar] if num_entradas_eliminar <= len(df) else df
    
    nombre_archivo = os.path.basename(ruta_archivo)
    nombre_salida = f"{nombre_archivo.split('.')[0]}_Var.csv"
    ruta_salida = os.path.join(output_dir, nombre_salida)
    
    df_final.to_csv(ruta_salida, index=False)
    print(f'Processed and saved as {nombre_salida}')
    
    return df_final

# Función para procesar todos los archivos CSV en un directorio
def procesar_todos_los_csv(directorio, output_dir):
    archivos = [os.path.join(directorio, f) for f in os.listdir(directorio) if f.endswith('.csv')]
    for archivo in archivos:
        procesar_archivo_csv_eliminar_ultimas(archivo, output_dir)
def main():
    ensure_data_directories()
    procesar_todos_los_csv(PROCESSED_STOCKS_DIR, TARGET_STOCKS_DIR)


if __name__ == "__main__":
    main()