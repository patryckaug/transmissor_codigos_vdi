from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from typing import Dict, Sequence, Optional, List
import uuid


# ===========================
# Helpers de tipo
# ===========================
def _is_integer_type(dt) -> bool:
    return isinstance(dt, (T.ByteType, T.ShortType, T.IntegerType, T.LongType))

def _is_float_type(dt) -> bool:
    return isinstance(dt, (T.FloatType, T.DoubleType, T.DecimalType))

def _is_numeric_type(dt) -> bool:
    return _is_integer_type(dt) or _is_float_type(dt)

def _is_datetime_type(dt) -> bool:
    return isinstance(dt, (T.DateType, T.TimestampType))

def _is_string_type(dt) -> bool:
    return isinstance(dt, T.StringType)


# ===========================
# Perfil em Spark (sem coletar pra driver)
# ===========================
def learn_column_profile_spark(
    sdf: DataFrame,
    col_name: str,
    max_categories: int = 100,
    total_rows: Optional[int] = None,
) -> dict:
    """
    Aprende perfil da coluna usando agregações Spark.
    Não coleta dados pra driver — só metadados (null_pct, cardinalidade, top-K se categórica).
    """
    field = next(f for f in sdf.schema.fields if f.name == col_name)
    dt = field.dataType

    if total_rows is None:
        total_rows = sdf.count()

    # null count via agg
    nulls = sdf.agg(F.sum(F.col(col_name).isNull().cast("int")).alias("n")).collect()[0]["n"] or 0
    null_pct = nulls / total_rows if total_rows > 0 else 0.0

    profile = {
        "name": col_name,
        "spark_type": dt,
        "null_pct": float(null_pct),
        "total_rows": total_rows,
    }

    # === STRING / BOOLEAN: categórica ===
    if _is_string_type(dt) or isinstance(dt, T.BooleanType):
        # Cardinalidade aproximada — barato
        approx_card = sdf.select(F.approx_count_distinct(col_name).alias("c")).collect()[0]["c"] or 0

        if approx_card <= max_categories:
            # Coleta top-K com frequências (só metadados, K pequeno)
            freq_df = (
                sdf.where(F.col(col_name).isNotNull())
                .groupBy(col_name).count()
                .orderBy(F.desc("count"))
            )
            rows = freq_df.collect()  # K <= max_categories, OK coletar
            total_non_null = sum(r["count"] for r in rows)
            profile["kind"] = "categorical"
            profile["values"] = [r[col_name] for r in rows]
            profile["probs"] = [r["count"] / total_non_null for r in rows] if total_non_null else []
        else:
            profile["kind"] = "token"

    # === NUMÉRICO / DATETIME: bootstrap por join ===
    elif _is_numeric_type(dt) or _is_datetime_type(dt):
        non_null_count = total_rows - nulls
        profile["kind"] = "numeric" if _is_numeric_type(dt) else "datetime"
        profile["is_integer"] = _is_integer_type(dt)
        profile["non_null_count"] = non_null_count

    else:
        profile["kind"] = "unsupported"

    return profile


# ===========================
# Pool indexado pra bootstrap distribuído
# ===========================
def _build_indexed_pool(sdf: DataFrame, col_name: str) -> DataFrame:
    """
    Retorna DataFrame com schema (pool_idx LONG, valor <tipo_original>),
    contendo apenas valores não-nulos da coluna, indexados de 0 a (N-1).
    """
    w = Window.orderBy(F.monotonically_increasing_id())
    return (
        sdf.where(F.col(col_name).isNotNull())
        .select(F.col(col_name).alias("valor"))
        .withColumn("pool_idx", F.row_number().over(w) - F.lit(1))
        .select("pool_idx", "valor")
    )


# ===========================
# Amostragem distribuída de uma coluna
# ===========================
def _sample_categorical(
    base_df: DataFrame,
    col_name: str,
    profile: dict,
    seed: int,
) -> DataFrame:
    """
    Amostra valor categórico segundo probs usando lookup de CDF.
    base_df já tem coluna 'row_idx' (0..n_rows-1).
    """
    values = profile["values"]
    probs = profile["probs"]

    if not values:
        return base_df.withColumn(col_name, F.lit(None).cast(profile["spark_type"]))

    # Constrói CDF: [(valor, cum_prob), ...]
    cum = 0.0
    cdf = []
    for v, p in zip(values, probs):
        cum += p
        cdf.append((v, cum))

    # Cria coluna com rand() e mapeia pro valor via CASE WHEN
    rnd_col = F.rand(seed=seed + hash(col_name) % 10000)

    expr = None
    for v, threshold in cdf:
        cond = rnd_col <= F.lit(threshold)
        # Cast pra preservar tipo original
        val_lit = F.lit(v).cast(profile["spark_type"])
        expr = F.when(cond, val_lit) if expr is None else expr.when(cond, val_lit)

    # Fallback (caso rand == 1.0 exato): último valor
    expr = expr.otherwise(F.lit(cdf[-1][0]).cast(profile["spark_type"]))

    return base_df.withColumn(col_name, expr)


def _sample_bootstrap(
    base_df: DataFrame,
    col_name: str,
    profile: dict,
    pool_df: DataFrame,
    seed: int,
) -> DataFrame:
    """
    Amostra via bootstrap: gera índice aleatório no pool e faz join.
    """
    pool_size = profile["non_null_count"]

    if pool_size == 0:
        return base_df.withColumn(col_name, F.lit(None).cast(profile["spark_type"]))

    # Índice aleatório de 0 a pool_size-1
    rand_idx_col = F.floor(F.rand(seed=seed + hash(col_name) % 10000) * F.lit(pool_size)).cast("long")
    enriched = base_df.withColumn("_lookup_idx", rand_idx_col)

    # Renomeia pool pra evitar colisão
    pool_renamed = pool_df.select(
        F.col("pool_idx").alias("_lookup_idx"),
        F.col("valor").alias(col_name),
    )

    joined = enriched.join(F.broadcast(pool_renamed), on="_lookup_idx", how="left") \
        if pool_size < 1_000_000 else enriched.join(pool_renamed, on="_lookup_idx", how="left")

    return joined.drop("_lookup_idx")


def _sample_token(base_df: DataFrame, col_name: str, seed: int) -> DataFrame:
    """Token único por linha usando uuid via expr."""
    return base_df.withColumn(col_name, F.expr("concat('tok_', substring(uuid(), 1, 12))"))


def _apply_nulls(
    sdf: DataFrame,
    col_name: str,
    null_pct: float,
    spark_type,
    seed: int,
) -> DataFrame:
    """Reinjeta nulos na proporção original."""
    if null_pct <= 0:
        return sdf
    rnd = F.rand(seed=seed + hash(col_name + "_null") % 10000)
    return sdf.withColumn(
        col_name,
        F.when(rnd < F.lit(null_pct), F.lit(None).cast(spark_type)).otherwise(F.col(col_name)),
    )


# ===========================
# Função principal
# ===========================
def synthesize_table_v1(
    sdf: DataFrame,
    n_rows: int,
    *,
    key_cols: Sequence[str] = (),
    seed: int = 42,
    spark: Optional[SparkSession] = None,
    max_categories: int = 100,
) -> DataFrame:
    """
    Sintetizador v1 100% Spark:
    - Aprende distribuição marginal de cada coluna via agregações Spark.
    - Gera n_rows novas linhas amostrando de cada perfil de forma distribuída.
    - Chaves (key_cols) ganham IDs novos (sequenciais ou UUID).
    - NÃO preserva correlações entre colunas nem relacionamentos cross-table.
    """
    spark = spark or sdf.sparkSession
    key_cols_set = set(key_cols or [])
    columns = sdf.columns

    total_rows = sdf.count()
    print(f"Tabela origem: {total_rows} linhas, {len(columns)} colunas")
    print(f"Gerando {n_rows} linhas sintéticas...")

    # 1. Aprende perfis (só metadados — não coleta dados)
    print("Aprendendo perfis...")
    profiles = {}
    for c in columns:
        if c not in key_cols_set:
            profiles[c] = learn_column_profile_spark(sdf, c, max_categories, total_rows)
            print(f"  {c}: {profiles[c]['kind']}")

    # 2. Cria DataFrame base com n_rows linhas e um row_idx
    print("Construindo base de n_rows...")
    base = spark.range(0, n_rows).withColumnRenamed("id", "row_idx")

    # 3. Pra cada coluna, aplica a estratégia adequada
    for c in columns:
        field = next(f for f in sdf.schema.fields if f.name == c)
        dt = field.dataType

        # --- CHAVE ---
        if c in key_cols_set:
            if _is_integer_type(dt):
                base = base.withColumn(c, (F.col("row_idx") + F.lit(1)).cast(dt))
            elif _is_string_type(dt):
                base = base.withColumn(c, F.expr("uuid()"))
            else:
                # Outros tipos de chave: usa row_idx castado
                base = base.withColumn(c, F.col("row_idx").cast(dt))
            continue

        # --- NÃO-CHAVE ---
        prof = profiles[c]
        kind = prof["kind"]

        if kind == "categorical":
            base = _sample_categorical(base, c, prof, seed)

        elif kind in ("numeric", "datetime"):
            pool_df = _build_indexed_pool(sdf, c).cache()
            pool_df.count()  # materializa cache
            base = _sample_bootstrap(base, c, prof, pool_df, seed)

        elif kind == "token":
            base = _sample_token(base, c, seed)

        else:
            base = base.withColumn(c, F.lit(None).cast(dt))

        # Reinjeta nulos
        base = _apply_nulls(base, c, prof["null_pct"], dt, seed)

    # 4. Remove row_idx e ordena colunas como o original
    result = base.select(*columns)

    return result