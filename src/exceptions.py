"""
Excepciones propias del proyecto credit-default-risk-model.

Definir excepciones específicas (en lugar de usar Exception genérica)
permite que el código que llama a estas funciones pueda capturar y manejar
cada tipo de error de forma distinta, y que los mensajes sean claros sobre
qué se esperaba vs. qué se encontró.
"""


class CreditRiskError(Exception):
    """Excepción base del proyecto. Todas las demás heredan de esta,
    así que un 'except CreditRiskError' atrapa cualquier error propio
    del proyecto sin tener que listar cada subtipo."""
    pass


class DataFileNotFoundError(CreditRiskError):
    """Se lanza cuando el archivo de datos esperado (CSV o DB) no existe
    en la ruta indicada."""
    pass


class DatasetSchemaError(CreditRiskError):
    """Se lanza cuando el dataset no tiene las columnas esperadas
    (por ejemplo, si falta la columna objetivo 'BAD')."""
    pass


class DatabaseConnectionError(CreditRiskError):
    """Se lanza cuando falla la conexión o una operación sobre la
    base de datos SQLite."""
    pass
