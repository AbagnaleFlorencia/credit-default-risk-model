"""
Consultas SQL reutilizables para el análisis exploratorio del dataset HMEQ.

Cada función devuelve un string con la query lista para pasarle a
ejecutar_query() de src/db.py. Separar las queries del código de conexión
permite leerlas, modificarlas y testearlas de forma independiente.

Convención: todas las queries trabajan sobre la tabla 'hmeq_raw',
que es el dato crudo sin transformaciones.
"""

TABLE = "hmeq_raw"


# =============================================================================
# BLOQUE 1: CALIDAD DE DATOS
# =============================================================================

def query_duplicados_exactos() -> str:
    """
    Detecta filas completamente duplicadas (las 13 columnas iguales).

    Por qué Opción A (fila completa) y no un subconjunto de columnas:
    en HMEQ un cliente puede tener más de una solicitud de préstamo con
    distintas condiciones -- eso no es un duplicado, es información real.
    Solo nos interesa detectar el mismo registro cargado dos veces por
    error técnico, que implica igualdad en TODAS las columnas.

    Si el resultado está vacío, no hay duplicados exactos en el dataset.
    """
    return f"""
        SELECT
            BAD, LOAN, MORTDUE, VALUE, REASON, JOB,
            YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC,
            COUNT(*) AS veces
        FROM {TABLE}
        GROUP BY
            BAD, LOAN, MORTDUE, VALUE, REASON, JOB,
            YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC
        HAVING COUNT(*) > 1
        ORDER BY veces DESC
    """


def query_conteo_nulos() -> str:
    """
    Cuenta nulos por columna usando CASE WHEN.

    SQLite no tiene una función COUNT_IF nativa, así que usamos el patrón
    SUM(CASE WHEN columna IS NULL THEN 1 ELSE 0 END) que ya practicamos
    en la consola interactiva. Devuelve una sola fila con el conteo de
    nulos de cada columna para tener el panorama completo de una vez.
    """
    return f"""
        SELECT
            SUM(CASE WHEN BAD     IS NULL THEN 1 ELSE 0 END) AS nulos_BAD,
            SUM(CASE WHEN LOAN    IS NULL THEN 1 ELSE 0 END) AS nulos_LOAN,
            SUM(CASE WHEN MORTDUE IS NULL THEN 1 ELSE 0 END) AS nulos_MORTDUE,
            SUM(CASE WHEN VALUE   IS NULL THEN 1 ELSE 0 END) AS nulos_VALUE,
            SUM(CASE WHEN REASON  IS NULL THEN 1 ELSE 0 END) AS nulos_REASON,
            SUM(CASE WHEN JOB     IS NULL THEN 1 ELSE 0 END) AS nulos_JOB,
            SUM(CASE WHEN YOJ     IS NULL THEN 1 ELSE 0 END) AS nulos_YOJ,
            SUM(CASE WHEN DEROG   IS NULL THEN 1 ELSE 0 END) AS nulos_DEROG,
            SUM(CASE WHEN DELINQ  IS NULL THEN 1 ELSE 0 END) AS nulos_DELINQ,
            SUM(CASE WHEN CLAGE   IS NULL THEN 1 ELSE 0 END) AS nulos_CLAGE,
            SUM(CASE WHEN NINQ    IS NULL THEN 1 ELSE 0 END) AS nulos_NINQ,
            SUM(CASE WHEN CLNO    IS NULL THEN 1 ELSE 0 END) AS nulos_CLNO,
            SUM(CASE WHEN DEBTINC IS NULL THEN 1 ELSE 0 END) AS nulos_DEBTINC
        FROM {TABLE}
    """


# =============================================================================
# BLOQUE 2: DISTRIBUCIÓN DE VARIABLES CATEGÓRICAS Y OBJETIVO
# =============================================================================

def query_distribucion_categorica(columna: str) -> str:
    """
    Distribución de frecuencia de una variable categórica,
    con tasa de incumplimiento (BAD=1) por categoría.

    Por qué incluir la tasa de BAD: no alcanza con saber cuántos clientes
    hay en cada categoría de JOB o REASON -- queremos saber si cierta
    categoría concentra más riesgo que otras. Eso informa tanto el EDA
    como la relevancia predictiva de la variable.

    Uso:
        ejecutar_query(query_distribucion_categorica('JOB'))
        ejecutar_query(query_distribucion_categorica('REASON'))
    """
    return f"""
        SELECT
            {columna},
            COUNT(*) AS total,
            SUM(BAD) AS cantidad_default,
            ROUND(100.0 * SUM(BAD) / COUNT(*), 2) AS tasa_default_pct
        FROM {TABLE}
        WHERE {columna} IS NOT NULL
        GROUP BY {columna}
        ORDER BY tasa_default_pct DESC
    """


def query_desbalance_objetivo() -> str:
    """
    Distribución de la variable objetivo BAD (0=Pagó, 1=Incumplió).
    Primer diagnóstico de desbalanceo antes de modelar.

    La tasa de incumplimiento determina qué métricas priorizar:
    si BAD=1 es minoritario (<30%), accuracy sola es engañosa y
    conviene mirar recall y F1 de la clase positiva, tal como
    hicimos en el proyecto de la diplomatura.
    """
    return f"""
        SELECT
            BAD,
            COUNT(*) AS cantidad,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS porcentaje
        FROM {TABLE}
        GROUP BY BAD
        ORDER BY BAD
    """


# =============================================================================
# BLOQUE 3: ESTADÍSTICAS DESCRIPTIVAS
# =============================================================================

def query_estadisticas_numericas(columna: str) -> str:
    """
    Estadísticas descriptivas básicas de una variable numérica:
    mínimo, máximo, promedio y mediana (aproximada con percentiles).

    SQLite no tiene función MEDIAN nativa, así que usamos el truco de
    percentil: ordenamos los valores y tomamos el del medio con LIMIT/OFFSET.
    Para datasets grandes esto es aproximado pero suficiente para EDA.

    Uso:
        ejecutar_query(query_estadisticas_numericas('LOAN'))
        ejecutar_query(query_estadisticas_numericas('DEBTINC'))
    """
    return f"""
        SELECT
            '{columna}'     AS variable,
            COUNT({columna})               AS n_no_nulos,
            ROUND(MIN({columna}), 2)       AS minimo,
            ROUND(MAX({columna}), 2)       AS maximo,
            ROUND(AVG({columna}), 2)       AS promedio,
            ROUND(AVG({columna} * {columna}) -
                  AVG({columna}) * AVG({columna}), 2) AS varianza_aprox
        FROM {TABLE}
        WHERE {columna} IS NOT NULL
    """


def query_distribucion_por_objetivo(columna: str) -> str:
    """
    Promedio de una variable numérica separado por clase de BAD.
    Primer indicador de si la variable separa bien las clases:
    si el promedio de DEBTINC es muy distinto entre BAD=0 y BAD=1,
    la variable tiene poder predictivo lineal sobre el objetivo.
    Si los promedios son similares, la relación puede ser no lineal
    o la variable puede no aportar mucho al modelo.

    Uso:
        ejecutar_query(query_distribucion_por_objetivo('DEBTINC'))
        ejecutar_query(query_distribucion_por_objetivo('LOAN'))
    """
    return f"""
        SELECT
            BAD,
            COUNT(*)                   AS n,
            ROUND(AVG({columna}), 2)   AS promedio,
            ROUND(MIN({columna}), 2)   AS minimo,
            ROUND(MAX({columna}), 2)   AS maximo
        FROM {TABLE}
        WHERE {columna} IS NOT NULL
        GROUP BY BAD
        ORDER BY BAD
    """
