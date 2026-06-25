"""
Funciones reutilizables de análisis exploratorio de datos.
Diseñadas para ser usadas en notebooks de EDA de forma directa:
cada función muestra su output (gráfico o tabla) sin retornar nada,
para facilitar la exploración interactiva.

Reutilizables en cualquier proyecto de clasificación tabular

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import chi2_contingency


# =============================================================================
# BLOQUE 1: ANÁLISIS DE DATOS FALTANTES
# =============================================================================

def resumen_nulos(df: pd.DataFrame) -> None:
    """
    Muestra una tabla con el conteo y porcentaje de nulos por columna,
    ordenada de mayor a menor porcentaje. Solo muestra columnas con al
    menos un nulo.

    Por qué: antes de cualquier decisión de imputación o eliminación,
    necesitamos saber cuánto falta y dónde. El porcentaje importa más
    que el conteo absoluto porque depende del tamaño del dataset.
    """
    total = df.isnull().sum()
    porcentaje = (df.isnull().sum() / len(df)) * 100

    resumen = pd.DataFrame({
        'nulos': total,
        'porcentaje': porcentaje.round(2)
    }).sort_values('porcentaje', ascending=False)

    resumen = resumen[resumen['nulos'] > 0]

    if resumen.empty:
        print("No hay valores nulos en el dataset.")
        return

    print("=== VALORES FALTANTES ===")
    print(resumen.to_string())
    print(f"\nTotal de columnas con nulos: {len(resumen)} de {df.shape[1]}")


def resumen_ceros(df: pd.DataFrame) -> None:
    """
    Muestra conteo y porcentaje de ceros por columna numérica.
    Solo muestra columnas con al menos un cero.

    Por qué: un cero puede significar cosas distintas según la variable.
    Un cero es un valor válido y frecuente o un error de carga.
    Esta función es el primer paso para distinguirlos
    -- la decisión de qué hacer con cada uno requiere criterio de negocio,
    no se puede automatizar.
    """
    numericas = df.select_dtypes(include=[np.number])
    conteo = (numericas == 0).sum()
    porcentaje = ((numericas == 0).sum() / len(df) * 100).round(2)

    resumen = pd.DataFrame({
        'ceros': conteo,
        'porcentaje': porcentaje
    }).sort_values('porcentaje', ascending=False)

    resumen = resumen[resumen['ceros'] > 0]

    if resumen.empty:
        print("No hay ceros en columnas numéricas.")
        return

    print("=== VALORES CERO EN COLUMNAS NUMÉRICAS ===")
    print(resumen.to_string())
    print("\nNota: verificar si cada cero es un valor válido o ausencia enmascarada.")


def plot_mapa_nulos(df: pd.DataFrame) -> None:
    """
    Visualiza el patrón de valores faltantes con missingno.matrix().
    Permite identificar visualmente si los nulos son:
    - MCAR (Missing Completely At Random): patrón aleatorio sin estructura
    - MAR (Missing At Random): la ausencia se relaciona con otras variables
    - MNAR (Missing Not At Random): la ausencia depende del valor faltante mismo

    Por qué importa la distinción: determina la estrategia de imputación.
    MCAR permite imputar con estadísticos simples (media/mediana).
    MAR requiere imputación multivariada (KNN, regresión).
    MNAR es el más complejo -- a veces conviene crear una variable indicadora
    de "este dato faltaba" antes de imputar.
    """
    print("=== MATRIZ DE NULOS (missingno.matrix) ===")
    print("Líneas verticales negras = dato presente. Espacio blanco = nulo.")
    print("Buscar patrones: si varias columnas tienen blancos en las mismas filas,")
    print("probablemente sean MAR (la ausencia está relacionada entre variables).\n")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    msno.matrix(df, ax=axes[0], sparkline=False)
    axes[0].set_title("Matriz de nulos", fontsize=13)

    msno.bar(df, ax=axes[1], color='steelblue')
    axes[1].set_title("Completitud por columna", fontsize=13)

    plt.tight_layout()
    plt.show()


def plot_heatmap_nulos(df: pd.DataFrame) -> None:
    """
    Heatmap de correlación entre columnas con nulos usando missingno.heatmap().
    Un valor cercano a 1 entre dos columnas significa que cuando una tiene nulo,
    la otra también tiende a tenerlo -- señal fuerte de MAR.
    Un valor cercano a 0 sugiere independencia entre las ausencias (más cercano a MCAR).
    """
    columnas_con_nulos = df.columns[df.isnull().any()].tolist()

    if len(columnas_con_nulos) < 2:
        print("Se necesitan al menos 2 columnas con nulos para el heatmap.")
        return

    print("=== CORRELACIÓN ENTRE PATRONES DE NULOS ===")
    print("Valores cercanos a 1: cuando una columna tiene nulo, la otra también.")
    print("Valores cercanos a 0: las ausencias son independientes entre sí.\n")

    plt.figure(figsize=(10, 6))
    msno.heatmap(df, figsize=(10, 6))
    plt.title("Correlación entre columnas con valores faltantes", fontsize=13)
    plt.tight_layout()
    plt.show()

# =============================================================================
# BLOQUE 2: LIMPIEZA — CEROS A NULOS Y ELIMINACIÓN DE REGISTROS
# =============================================================================

def ceros_a_nulos(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """
    Reemplaza ceros por NaN en las columnas indicadas.
    Devuelve un nuevo DataFrame sin modificar el original.

    Cuándo usarla: cuando un cero no es un valor válido sino ausencia
    enmascarada.
    La decisión de qué columnas incluir es de negocio, no técnica.

    Uso:
        df_limpio = ceros_a_nulos(df, columnas=['MORTDUE', 'VALUE'])
    """
    df_resultado = df.copy()
    for col in columnas:
        if col not in df_resultado.columns:
            print(f"Advertencia: columna '{col}' no encontrada, se omite.")
            continue
        n_reemplazados = (df_resultado[col] == 0).sum()
        df_resultado[col] = df_resultado[col].replace(0, np.nan)
        print(f"{col}: {n_reemplazados} ceros reemplazados por NaN.")
    return df_resultado


def eliminar_registros_con_nulos(df: pd.DataFrame,
                                  columnas: list = None,
                                  umbral_pct: float = None) -> pd.DataFrame:
    """
    Elimina filas con valores nulos. Dos modos de uso:

    Modo 1 — columnas específicas (columnas=['A', 'B']):
        Elimina filas que tengan nulo en CUALQUIERA de esas columnas.
        Útil cuando ciertas variables son imprescindibles para el modelo.

    Modo 2 — umbral de porcentaje (umbral_pct=0.5):
        Elimina filas donde más del X% de sus valores son nulos.
        Útil para limpiar registros casi vacíos sin importar qué columna.

    Si no se pasa ninguno de los dos, elimina cualquier fila con algún nulo.

    Uso:
        df_limpio = eliminar_registros_con_nulos(df, columnas=['DEBTINC'])
        df_limpio = eliminar_registros_con_nulos(df, umbral_pct=0.5)
    """
    n_original = len(df)

    if columnas is not None:
        df_resultado = df.dropna(subset=columnas)
    elif umbral_pct is not None:
        minimo_no_nulos = int((1 - umbral_pct) * df.shape[1])
        df_resultado = df.dropna(thresh=minimo_no_nulos)
    else:
        df_resultado = df.dropna()

    n_eliminados = n_original - len(df_resultado)
    print(f"Registros eliminados: {n_eliminados} ({n_eliminados/n_original*100:.1f}%)")
    print(f"Registros restantes: {len(df_resultado)}")
    return df_resultado


def eliminar_registros_con_ceros(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """
    Elimina filas donde alguna de las columnas indicadas tenga valor cero.

    Cuándo usarla: cuando un cero en cierta variable hace al
    registro inválido o poco confiable, y preferís eliminarlo antes que
    imputarlo. Alternativa más agresiva que ceros_a_nulos() + imputación.

    Uso:
        df_limpio = eliminar_registros_con_ceros(df, columnas=['LOAN', 'VALUE'])
    """
    n_original = len(df)
    mascara = (df[columnas] == 0).any(axis=1)
    df_resultado = df[~mascara]
    n_eliminados = n_original - len(df_resultado)
    print(f"Registros eliminados por ceros en {columnas}: {n_eliminados} ({n_eliminados/n_original*100:.1f}%)")
    print(f"Registros restantes: {len(df_resultado)}")
    return df_resultado


# =============================================================================
# BLOQUE 3: DISTRIBUCIONES
# =============================================================================

def plot_distribuciones_numericas(df: pd.DataFrame,
                                   objetivo: str = None) -> None:
    """
    Histograma + KDE para cada variable numérica.
    Si se pasa 'objetivo', superpone la distribución de cada variable
    separada por clase (0 vs 1), lo que permite ver visualmente si la
    variable separa bien las clases -- señal de poder predictivo.

    Sin objetivo: sirve para detectar asimetrías, bimodalidad, outliers.
    Con objetivo: sirve para decidir si la relación con la clase es
    lineal (diferencia de medias clara) o no lineal (distribuciones
    solapadas pero con colas distintas).

    Uso:
        plot_distribuciones_numericas(df)
        plot_distribuciones_numericas(df, objetivo='BAD')
    """
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if objetivo in numericas:
        numericas.remove(objetivo)

    n_cols = 3
    n_rows = (len(numericas) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numericas):
        ax = axes[i]
        if objetivo is not None and objetivo in df.columns:
            for clase in sorted(df[objetivo].dropna().unique()):
                subset = df[df[objetivo] == clase][col].dropna()
                label = f"{objetivo}={int(clase)}"
                sns.kdeplot(subset, ax=ax, label=label, fill=True, alpha=0.4)
            ax.legend(fontsize=8)
        else:
            sns.histplot(df[col].dropna(), ax=ax, kde=True, color='steelblue')
        ax.set_title(col, fontsize=11)
        ax.set_xlabel('')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    titulo = "Distribución por clase" if objetivo else "Distribución de variables numéricas"
    fig.suptitle(titulo, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_distribuciones_categoricas(df: pd.DataFrame,
                                     objetivo: str = None) -> None:
    """
    Gráfico de barras para cada variable categórica.
    Si se pasa 'objetivo', muestra la tasa de la clase positiva por categoría
    (porcentaje de BAD=1 dentro de cada categoría), lo que permite ver
    si cierta categoría tiene más riesgo que otras.

    Uso:
        plot_distribuciones_categoricas(df)
        plot_distribuciones_categoricas(df, objetivo='BAD')
    """
    categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categoricas:
        print("No hay variables categóricas en el dataset.")
        return

    n_cols = 2
    n_rows = (len(categoricas) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(categoricas):
        ax = axes[i]
        if objetivo is not None and objetivo in df.columns:
            tasa = df.groupby(col)[objetivo].mean().sort_values(ascending=False)
            tasa.plot(kind='bar', ax=ax, color='salmon', edgecolor='black')
            ax.set_ylabel(f'Tasa {objetivo}=1')
            ax.set_title(f'{col} — tasa de incumplimiento por categoría', fontsize=10)
        else:
            orden = df[col].value_counts().index
            sns.countplot(data=df, x=col, order=orden, ax=ax, color='steelblue')
            ax.set_title(f'Distribución de {col}', fontsize=10)
        ax.tick_params(axis='x', rotation=30)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


# =============================================================================
# BLOQUE 4: CORRELACIONES
# =============================================================================

def plot_correlacion_numerica(df: pd.DataFrame,
                               metodo: str = 'spearman') -> None:
    """
    Heatmap de correlación entre variables numéricas.

    Pearson: mide relaciones lineales. Sensible a outliers.
    Spearman: mide relaciones monótonas (lineales y no lineales suaves).
              Más robusto a outliers. Recomendado como punto de partida
              cuando no sabés si las relaciones son lineales.

    Por qué importa para el modelo: correlaciones altas entre predictores
    indican multicolinealidad -- no es un problema para XGBoost (que lo
    maneja bien) pero sí para modelos lineales como regresión logística.
    También ayuda a detectar variables redundantes que podés descartar
    sin perder información.

    Uso:
        plot_correlacion_numerica(df)
        plot_correlacion_numerica(df, metodo='pearson')
    """
    numericas = df.select_dtypes(include=[np.number])
    corr = numericas.corr(method=metodo)

    mascara = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr,
        mask=mascara,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        annot_kws={'size': 9}
    )
    plt.title(f'Correlación {metodo.capitalize()} — variables numéricas', fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_correlacion_categorica(df: pd.DataFrame,
                                 objetivo: str = None) -> None:
    """
    Heatmap de asociación entre variables categóricas usando Cramér's V.

    Cramér's V es el equivalente de Pearson para variables categóricas:
    valores cercanos a 1 indican asociación fuerte, cercanos a 0 indican
    independencia. Se basa en el estadístico chi-cuadrado pero normalizado
    para que sea comparable entre variables con distinto número de categorías.

    Si se pasa 'objetivo' y es numérica binaria (0/1), se incluye en el
    análisis convirtiéndola temporalmente a categórica, para ver su
    asociación con las variables categóricas del dataset.

    Uso:
        plot_correlacion_categorica(df)
        plot_correlacion_categorica(df, objetivo='BAD')
    """

    def cramers_v(x, y):
        tabla = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(tabla)
        n = tabla.sum().sum()
        k = min(tabla.shape) - 1
        if k == 0 or n == 0:
            return 0.0
        return np.sqrt(chi2 / (n * k))

    categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if objetivo is not None and objetivo in df.columns:
        df_temp = df.copy()
        df_temp[objetivo] = df_temp[objetivo].astype(str)
        if objetivo not in categoricas:
            categoricas.append(objetivo)
    else:
        df_temp = df

    if len(categoricas) < 2:
        print("Se necesitan al menos 2 variables categóricas.")
        return

    matriz = pd.DataFrame(index=categoricas, columns=categoricas, dtype=float)
    for col1 in categoricas:
        for col2 in categoricas:
            datos = df_temp[[col1, col2]].dropna()
            matriz.loc[col1, col2] = cramers_v(datos[col1], datos[col2])

    mascara = np.triu(np.ones_like(matriz, dtype=bool))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matriz.astype(float),
        mask=mascara,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        vmin=0, vmax=1,
        linewidths=0.5,
        annot_kws={'size': 10}
    )
    plt.title("Asociación Cramér's V — variables categóricas", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_desbalance_objetivo(df: pd.DataFrame, objetivo: str) -> None:
    """
    Muestra la distribución de clases de la variable objetivo para
    diagnosticar desbalanceo.

    Por qué importa antes de modelar: si una clase representa el 80%
    y la otra el 20%, un modelo que prediga siempre la clase mayoritaria
    tendría 80% de accuracy sin aprender nada útil. El desbalanceo
    determina qué métrica priorizar (F1, recall) y si es necesario
    aplicar técnicas de balanceo (scale_pos_weight, SMOTE, undersampling).

    Muestra:
    - Conteo absoluto y porcentaje de cada clase
    - Gráfico de barras con los valores anotados
    - Ratio de desbalanceo (clase mayoritaria / clase minoritaria)

    Uso:
        plot_desbalance_objetivo(df, objetivo='BAD')
    """
    if objetivo not in df.columns:
        print(f"Columna '{objetivo}' no encontrada en el dataset.")
        return

    conteo = df[objetivo].value_counts().sort_index()
    porcentaje = (df[objetivo].value_counts(normalize=True) * 100).sort_index().round(2)
    ratio = conteo.max() / conteo.min()

    print("=== DISTRIBUCIÓN DE LA VARIABLE OBJETIVO ===")
    resumen = pd.DataFrame({
        'clase': conteo.index,
        'conteo': conteo.values,
        'porcentaje': porcentaje.values
    })
    print(resumen.to_string(index=False))
    print(f"\nRatio de desbalanceo: {ratio:.1f}:1 (mayoritaria/minoritaria)")

    if ratio > 10:
        print("Desbalanceo severo (>10:1). Considerar SMOTE o undersampling.")
    elif ratio > 3:
        print(" esbalanceo moderado (>3:1). Considerar scale_pos_weight o class_weight.")
    else:
        print(" Desbalanceo leve (<3:1). Puede no requerir tratamiento especial.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Barras con conteo absoluto
    bars = axes[0].bar(
        conteo.index.astype(str),
        conteo.values,
        color=['steelblue', 'salmon'],
        edgecolor='black'
    )
    for bar, pct in zip(bars, porcentaje.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f'{pct}%',
            ha='center', fontsize=11
        )
    axes[0].set_title(f'Distribución de {objetivo}', fontsize=12)
    axes[0].set_xlabel('Clase')
    axes[0].set_ylabel('Cantidad de registros')

    # Pie chart para visualizar la proporción
    axes[1].pie(
        conteo.values,
        labels=[f'Clase {c}' for c in conteo.index],
        autopct='%1.1f%%',
        colors=['steelblue', 'salmon'],
        startangle=90
    )
    axes[1].set_title('Proporción de clases', fontsize=12)

    plt.tight_layout()
    plt.show()

# =============================================================================
# BLOQUE 5: DETECCIÓN DE OUTLIERS
# =============================================================================

def plot_outliers(df: pd.DataFrame, objetivo: str = None) -> None:
    """
    Boxplots para detectar outliers en variables numéricas.
    Si se pasa 'objetivo', separa los boxplots por clase para ver si
    los outliers se concentran más en una clase que en otra -- lo cual
    puede ser información predictiva valiosa (ej: montos extremos de
    préstamo podrían concentrarse en clientes que incumplieron).

    Uso:
        plot_outliers(df)
        plot_outliers(df, objetivo='BAD')
    """
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if objetivo in numericas:
        numericas.remove(objetivo)

    n_cols = 3
    n_rows = (len(numericas) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numericas):
        ax = axes[i]
        if objetivo is not None and objetivo in df.columns:
            sns.boxplot(
                data=df,
                x=objetivo,
                y=col,
                ax=ax,
                palette={0: 'steelblue', 1: 'salmon'}
            )
            ax.set_xlabel(objetivo)
        else:
            sns.boxplot(data=df, y=col, ax=ax, color='steelblue')
        ax.set_title(col, fontsize=11)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    titulo = "Outliers por clase" if objetivo else "Detección de outliers"
    fig.suptitle(titulo, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def resumen_outliers_iqr(df: pd.DataFrame) -> None:
    """
    Tabla con conteo de outliers por variable numérica usando el método IQR.
    Un valor es outlier si cae fuera del rango [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Por qué IQR y no z-score: IQR es robusto a la distribución de los datos
    (no asume normalidad) y no es sensible a los propios outliers para
    calcular el umbral -- el z-score sí lo es, lo que puede subestimar
    outliers extremos en distribuciones sesgadas como las financieras.

    Uso:
        resumen_outliers_iqr(df)
    """
    numericas = df.select_dtypes(include=[np.number])
    resultados = []

    for col in numericas.columns:
        q1 = numericas[col].quantile(0.25)
        q3 = numericas[col].quantile(0.75)
        iqr = q3 - q1
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr
        n_outliers = ((numericas[col] < limite_inf) | (numericas[col] > limite_sup)).sum()
        pct = round(n_outliers / len(df) * 100, 2)
        resultados.append({
            'variable': col,
            'Q1': round(q1, 2),
            'Q3': round(q3, 2),
            'IQR': round(iqr, 2),
            'limite_inf': round(limite_inf, 2),
            'limite_sup': round(limite_sup, 2),
            'outliers': n_outliers,
            'porcentaje': pct
        })

    resumen = pd.DataFrame(resultados).sort_values('porcentaje', ascending=False)
    print("=== OUTLIERS POR VARIABLE (método IQR) ===")
    print(resumen.to_string(index=False))
