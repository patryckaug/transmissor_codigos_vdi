from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from typing import Dict, Sequence, Optional
import numpy as np
import pandas as pd
import uuid


def learn_column_profile(sdf: DataFrame, col_name: str, max_categories: int = 100) -> dict:
    """Aprende um perfil simples da coluna pra usar como base de amostragem."""
    field = next(f for f in sdf.schema.fields if f.name == col_name)
    dt = field.dataType
    
    # Coleta valores não-nulos pra pandas (amostra se for muito grande)
    pdf = sdf.select(col_name).where(F.col(col_name).isNotNull()).toPandas()
    null_pct = 1 - len(pdf) / max(sdf.count(), 1)
    
    profile = {"dtype": dt, "null_pct": null_pct, "name": col_name}
    
    if isinstance(dt, (T.StringType,)):
        # Categórica: top-K com frequências
        vc = pdf[col_name].value_counts(normalize=True)
        if len(vc) <= max_categories:
            profile["kind"] = "categorical"
            profile["values"] = vc.index.tolist()
            profile["probs"] = vc.values.tolist()
        else:
            # alta cardinalidade → tratar como token
            profile["kind"] = "token"
    
    elif isinstance(dt, (T.ByteType, T.ShortType, T.IntegerType, T.LongType, T.FloatType, T.DoubleType, T.DecimalType)):
        profile["kind"] = "numeric"
        profile["pool"] = pdf[col_name].values  # bootstrap pool
        profile["is_integer"] = isinstance(dt, (T.ByteType, T.ShortType, T.IntegerType, T.LongType))
    
    elif isinstance(dt, (T.DateType, T.TimestampType)):
        profile["kind"] = "datetime"
        profile["pool"] = pdf[col_name].values
    
    elif isinstance(dt, T.BooleanType):
        profile["kind"] = "categorical"
        vc = pdf[col_name].value_counts(normalize=True)
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
        values = rng.choice(profile["values"], size=n_rows, p=profile["probs"])
    elif kind == "numeric":
        values = rng.choice(profile["pool"], size=n_rows, replace=True)
        if profile.get("is_integer"):
            values = values.astype(np.int64)
    elif kind == "datetime":
        values = rng.choice(profile["pool"], size=n_rows, replace=True)
    elif kind == "token":
        values = np.array([f"tok_{uuid.uuid4().hex[:12]}" for _ in range(n_rows)])
    else:
        values = np.array([None] * n_rows)
    
    # Reinjeta nulls na proporção original
    if null_pct > 0:
        mask = rng.random(n_rows) < null_pct
        values = np.where(mask, None, values)
    
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
    spark = spark or SparkSession.builder.getOrCreate()
    
    key_cols = set(key_cols or [])
    columns = sdf.columns
    
    # Aprende perfis
    profiles = {c: learn_column_profile(sdf, c) for c in columns if c not in key_cols}
    
    # Amostra cada coluna
    data = {}
    for c in columns:
        if c in key_cols:
            data[c] = [str(uuid.uuid4()) for _ in range(n_rows)]
        else:
            data[c] = sample_column(profiles[c], n_rows, rng)
    
    # Monta pandas DF e devolve como Spark DF preservando schema original
    pdf = pd.DataFrame(data)[columns]  # mantém ordem
    return spark.createDataFrame(pdf, schema=sdf.schema)