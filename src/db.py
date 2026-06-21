"""
Manejo de la conexión a SQLite para el proyecto credit-default-risk-model.

Carga el CSV crudo (data/raw/hmeq.csv) a una base SQLite (data/raw/hmeq.db)
sin ninguna transformación de los datos -- es el mismo dato crudo, solo en
otro formato de acceso, para poder practicar consultas SQL sobre él.
"""

import sqlite3
from pathlib import Path
import pandas as pd

from src.exceptions import DataFileNotFoundError, DatabaseConnectionError

RAW_CSV_PATH = Path("data/raw/hmeq.csv")
DB_PATH = Path("data/raw/hmeq.db")
TABLE_NAME = "hmeq_raw"


def _get_dtype_mapping() -> dict:
    """
    Define explícitamente el tipo SQL de cada columna, en lugar de dejar
    que pandas infiera. Las columnas con valores faltantes se declaran
    REAL aunque sean conteos enteros, porque pandas convierte a float64
    cualquier columna entera que tenga NaN al leerla de vuelta -- así
    evitamos una discrepancia entre el tipo declarado y el tipo real recibido.
    """
    return {
        "BAD": "INTEGER",
        "LOAN": "INTEGER",
        "MORTDUE": "REAL",
        "VALUE": "REAL",
        "REASON": "TEXT",
        "JOB": "TEXT",
        "YOJ": "REAL",
        "DEROG": "REAL",
        "DELINQ": "REAL",
        "CLAGE": "REAL",
        "NINQ": "REAL",
        "CLNO": "REAL",
        "DEBTINC": "REAL",
    }


def crear_db_desde_csv(csv_path: Path = RAW_CSV_PATH, db_path: Path = DB_PATH) -> None:
    """
    Carga el CSV crudo a una tabla SQLite, sin transformar nada,
    pero declarando explícitamente el tipo SQL de cada columna
    (ver _get_dtype_mapping) para evitar inferencias erróneas de tipo.
    Si la base ya existe, la reemplaza para que siempre refleje el CSV actual.
    """
    if not csv_path.exists():
        raise DataFileNotFoundError(
            f"No se encontró el archivo {csv_path}. "
            f"Verificá que hayas descargado el dataset a data/raw/."
        )

    try:
        df = pd.read_csv(csv_path)
        dtype_mapping = _get_dtype_mapping()

        conn = sqlite3.connect(db_path)
        df.to_sql(
            TABLE_NAME,
            conn,
            if_exists="replace",
            index=False,
            dtype=dtype_mapping,
        )
        conn.close()
        print(f"Base de datos creada en {db_path} con tabla '{TABLE_NAME}' ({len(df)} filas).")
    except sqlite3.Error as e:
        raise DatabaseConnectionError(f"Error al crear la base de datos: {e}")


def ejecutar_query(query: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Ejecuta una consulta SQL contra la base y devuelve el resultado
    como DataFrame de pandas, listo para seguir analizando o graficando.
    """
    if not db_path.exists():
        raise DataFileNotFoundError(
            f"No se encontró la base de datos en {db_path}. "
            f"Corré crear_db_desde_csv() primero."
        )

    try:
        conn = sqlite3.connect(db_path)
        resultado = pd.read_sql_query(query, conn)
        conn.close()
        return resultado
    except sqlite3.Error as e:
        raise DatabaseConnectionError(f"Error al ejecutar la query: {e}")
