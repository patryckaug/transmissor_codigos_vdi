from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from typing import Dict, Sequence, Optional
import numpy as np
import pandas as pd
import uuid


def _to_pandas_safe(sdf: DataFrame, col_name: str) -> pd.DataFrame:
    """
    toPandas() seguro pro Spark 3.3.2 — desliga Arrow temporariamente pra evitar
    o bug do datetime64 sem unidade.
    """
    spark = sdf.sparkSession
    arrow_was_on = spark.conf.get("spark.sql.execution.arrow.pyspark.enabled", "false")
    try:
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
        pdf = sdf.select(col_name).toPandas()
    finally:
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", arrow_was_on)
    return pdf


def learn_column_profile(sdf: DataFrame, col_name: str, max_categories: int = 100) -> dict:
    """Aprende um perfil simples da coluna pra usar como base de amostragem."""
    field = next(f for f in sdf.schema.fields if f.name == col_name)
    dt = field.dataType

    # Coleta a coluna inteira (com nulls) pra pandas — sem Arrow pra evitar bug datetime
    pdf = _to_pandas_safe(sdf, col_name)
    series = pdf[col_name]
    total = len(series)
    non_null = series.dropna()
    null_pct = 1 - (len(non_null) / total) if total > 0 else 0.0

    profile = {"dtype": dt, "null_pct": null_pct, "name": col_name}

    # === STRING ===
    if isinstance(dt, T.StringType):
        vc = non_null.value_counts(normalize=True)
        if len(vc) <= max_categories:
            profile["kind"] = "categorical"
            profile["values"] = vc.index.tolist()
            profile["probs"] = vc.values.tolist()
        else:
            profile["kind"] = "token"

    # === NUMÉRICO INTEIRO ===
    elif isinstance(dt, (T.ByteType, T.ShortType, T.IntegerType, T.LongType)):
        profile["kind"] = "numeric"
        profile["pool"] = non_null.astype(np.int64).values
        profile["is_integer"] = True
        profile["spark_type"] = dt

    # === NUMÉRICO FLOAT ===
    elif isinstance(dt, (T.FloatType, T.DoubleType)):
        profile["kind"] = "numeric"
        profile["pool"] = non_null.astype(np.float64).values
        profile["is_integer"] = False
        profile["spark_type"] = dt

    # === DECIMAL (vem como object/Decimal no pandas) ===
    elif isinstance(dt, T.DecimalType):
        profile["kind"] = "numeric"
        profile["pool"] = non_null.astype(np.float64).values
        profile["is_integer"] = False
        profile["spark_type"] = dt
        profile["is_decimal"] = True

    # === DATA / TIMESTAMP ===
    elif isinstance(dt, (T.DateType, T.TimestampType)):
        profile["kind"] = "datetime"
        # Força datetime64[ns] já aqui pra não ter problema na amostragem
        profile["pool"] = pd.to_datetime(non_null, errors="coerce").dropna().astype("datetime64[ns]").values
        profile["spark_type"] = dt

    # === BOOLEAN ===
    elif isinstance(dt, T.BooleanType):
        vc = non_null.value_counts(normalize=True)
        profile["kind"] = "categorical"
        profile["values"] = vc.index.tolist()
        profile["probs"] = vc.values.tolist()

    else:
        profile["kind"] = "unsupported"

    return profile


def sample_column(profile: dict, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Gera n_rows valores sintéticos a partir do perfil."""
    kind = profile["kind"]
    null_pct = profile.get("null_pct", 0.0)

    if kind == "categorical":
        if len(profile["values"]) == 0:
            values = np.array([None] * n_rows, dtype=object)
        else:
            idx = rng.choice(len(profile["values"]), size=n_rows, p=profile["probs"])
            values = np.array([profile["values"][i] for i in idx], dtype=object)

    elif kind == "numeric":
        pool = profile["pool"]
        if len(pool) == 0:
            values = np.array([None] * n_rows, dtype=object)
        else:
            idx = rng.integers(0, len(pool), size=n_rows)
            values = pool[idx]
            if profile.get("is_integer"):
                values = values.astype(np.int64)

    elif kind == "datetime":
        pool = profile["pool"]
        if len(pool) == 0:
            values = np.array([None] * n_rows, dtype=object)
        else:
            idx = rng.integers(0, len(pool), size=n_rows)
            values = pool[idx].astype("datetime64[ns]")

    elif kind == "token":
        values = np.array([f"tok_{uuid.uuid4().hex[:12]}" for _ in range(n_rows)], dtype=object)

    else:
        values = np.array([None] * n_rows, dtype=object)

    # Reinjeta nulls na proporção original
    if null_pct > 0 and kind != "unsupported":
        mask = rng.random(n_rows) < null_pct
        values = pd.Series(values)
        values[mask] = None
        values = values.values

    return values


def synthesize_table_v1(
    sdf: DataFrame,
    n_rows: int,
    *,
    key_cols: Sequence[str] = (),
    seed: int = 42,
    spark: Optional[SparkSession] = None,
) -> DataFrame:
    """
    Sintetizador v1: amostragem independente por coluna.
    - Aprende distribuição marginal de cada coluna.
    - Gera n_rows novas linhas amostrando de cada perfil.
    - Chaves (key_cols) ganham UUIDs novos, não amostrados.
    - NÃO preserva correlações entre colunas nem relacionamentos cross-table.
    """
    rng = np.random.default_rng(seed)
    spark = spark or sdf.sparkSession

    key_cols_set = set(key_cols or [])
    columns = sdf.columns

    # 1. Aprende perfis das não-chaves
    print(f"Aprendendo perfis de {len(columns) - len(key_cols_set)} colunas...")
    profiles = {}
    for c in columns:
        if c not in key_cols_set:
            profiles[c] = learn_column_profile(sdf, c)
            print(f"  {c}: {profiles[c]['kind']}")

    # 2. Amostra cada coluna
    print(f"Amostrando {n_rows} linhas...")
    data = {}
    for c in columns:
        if c in key_cols_set:
            # Detecta o tipo da chave pra gerar valores coerentes
            field = next(f for f in sdf.schema.fields if f.name == c)
            if isinstance(field.dataType, (T.ByteType, T.ShortType, T.IntegerType, T.LongType)):
                # Chave inteira: sequência nova
                data[c] = np.arange(1, n_rows + 1, dtype=np.int64)
            else:
                # Chave string: UUID
                data[c] = np.array([str(uuid.uuid4()) for _ in range(n_rows)], dtype=object)
        else:
            data[c] = sample_column(profiles[c], n_rows, rng)

    # 3. Monta pandas DF
    pdf = pd.DataFrame(data)[columns]

    # 4. Força datetime64[ns] nas colunas de data/timestamp (fix do bug Spark 3.3.2)
    for f in sdf.schema.fields:
        if isinstance(f.dataType, (T.DateType, T.TimestampType)):
            pdf[f.name] = pd.to_datetime(pdf[f.name], errors="coerce").astype("datetime64[ns]")

    # 5. Converte de volta pra Spark, sem Arrow pra evitar conflito
    arrow_was_on = spark.conf.get("spark.sql.execution.arrow.pyspark.enabled", "false")
    try:
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
        result = spark.createDataFrame(pdf, schema=sdf.schema)
    finally:
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", arrow_was_on)

    return result