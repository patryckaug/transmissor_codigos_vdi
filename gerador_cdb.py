"""
Gerador de tabelas sintéticas para POC EngordAI - multi-tabela com FKs.
Gera TIPO_IF → INSTRUMENTO_FINANCEIRO → OPERACAO com relacionamentos coerentes.

Uso típico:
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    tipo_if, instrumento, operacao = generate_cdb_tables(
        spark,
        n_instrumentos=1000,
        n_operacoes=50000,
        seed=42,
    )
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from typing import Tuple
from datetime import datetime


# ============================================================
# Constantes de domínio (realista pra CDB Simplificado)
# ============================================================
TIPOS_IF = [
    # (NUM_TIPO_IF, COD_TIPO_IF, NOM_TIPO_IF, peso_relativo)
    (1, "CDB",   "Certificado de Depósito Bancário",        0.55),
    (2, "LCI",   "Letra de Crédito Imobiliário",            0.18),
    (3, "LCA",   "Letra de Crédito do Agronegócio",         0.15),
    (4, "LF",    "Letra Financeira",                        0.08),
    (5, "DPGE",  "Depósito a Prazo com Garantia Especial",  0.04),
]

SITUACOES_OPERACAO = [
    ("ATIVA",      0.60),
    ("LIQUIDADA",  0.35),
    ("CANCELADA",  0.05),
]

# Janela temporal
DAT_MIN = datetime(2020, 1, 1)
DAT_MAX = datetime(2025, 12, 31)


# ============================================================
# 1) TIPO_IF
# ============================================================
def generate_tipo_if(spark: SparkSession) -> DataFrame:
    """Tabela de referência: 5 tipos fixos."""
    schema = T.StructType([
        T.StructField("NUM_TIPO_IF",   T.IntegerType(),   False),
        T.StructField("COD_TIPO_IF",   T.StringType(),    False),
        T.StructField("NOM_TIPO_IF",   T.StringType(),    False),
        T.StructField("DAT_INCLUSAO",  T.TimestampType(), False),
        T.StructField("DAT_ALTERACAO", T.TimestampType(), False),
    ])
    base_date = datetime(2019, 12, 1)
    rows = [
        (num, cod, nom, base_date, base_date)
        for (num, cod, nom, _) in TIPOS_IF
    ]
    return spark.createDataFrame(rows, schema=schema)


# ============================================================
# 2) INSTRUMENTO_FINANCEIRO
# ============================================================
def generate_instrumento_financeiro(
    spark: SparkSession,
    n_rows: int,
    seed: int = 42,
) -> DataFrame:
    """
    Gera N instrumentos financeiros.
    - NUM_TIPO_IF segue distribuição enviesada (mais CDBs)
    - VAL_NOMINAL_EMISSAO log-normal (cauda longa típica de renda fixa)
    - VAL_NOMINAL_ATUAL = EMISSAO * (1 + drift positivo aleatório)
    - DAT_ALTERACAO >= DAT_INCLUSAO
    """
    # CDF dos tipos pra amostragem ponderada
    cum = 0.0
    cdf = []
    for (num, _, _, peso) in TIPOS_IF:
        cum += peso
        cdf.append((num, cum))

    base = spark.range(0, n_rows).withColumnRenamed("id", "NUM_IF")
    base = base.withColumn("NUM_IF", (F.col("NUM_IF") + F.lit(1)).cast("long"))

    # Random columns (uma por dimensão pra evitar correlação espúria)
    base = base.withColumn("_r_tipo",   F.rand(seed=seed))
    base = base.withColumn("_r_val",    F.rand(seed=seed + 1))
    base = base.withColumn("_r_drift",  F.rand(seed=seed + 2))
    base = base.withColumn("_r_inc",    F.rand(seed=seed + 3))
    base = base.withColumn("_r_alt",    F.rand(seed=seed + 4))

    # NUM_TIPO_IF via CDF lookup
    tipo_expr = None
    for (num, threshold) in cdf:
        cond = F.col("_r_tipo") <= F.lit(threshold)
        val = F.lit(num).cast("int")
        tipo_expr = F.when(cond, val) if tipo_expr is None else tipo_expr.when(cond, val)
    tipo_expr = tipo_expr.otherwise(F.lit(cdf[-1][0]).cast("int"))
    base = base.withColumn("NUM_TIPO_IF", tipo_expr)

    # COD_IF: prefixo do tipo + numero sequencial zerado
    # Ex: CDB000001, LCI000234
    tipo_to_cod = F.create_map([
        x for (num, cod, _, _) in TIPOS_IF
        for x in (F.lit(num), F.lit(cod))
    ])
    base = base.withColumn(
        "COD_IF",
        F.concat(
            tipo_to_cod[F.col("NUM_TIPO_IF")],
            F.lpad(F.col("NUM_IF").cast("string"), 6, "0")
        )
    )

    # COD_ISIN: BR + 10 alfanuméricos. Aqui simulamos com hash determinístico do NUM_IF
    # (ISIN real tem dígito verificador, mas pro POC isso basta)
    base = base.withColumn(
        "COD_ISIN",
        F.concat(
            F.lit("BR"),
            F.upper(F.substring(F.sha2(F.col("NUM_IF").cast("string"), 256), 1, 10))
        )
    )

    # VAL_NOMINAL_EMISSAO: log-normal aproximado
    # exp(N(mu, sigma)) onde mu=12 (e^12 ≈ 163k) e sigma=1.5
    # Usamos Box-Muller aproximado: -2*ln(U1) * cos(2*pi*U2)
    base = base.withColumn("_z",
        F.sqrt(F.lit(-2.0) * F.log(F.col("_r_val") + F.lit(1e-10))) *
        F.cos(F.lit(2.0 * 3.14159265) * F.col("_r_drift"))
    )
    base = base.withColumn(
        "VAL_NOMINAL_EMISSAO",
        F.round(F.exp(F.lit(12.0) + F.lit(1.5) * F.col("_z")), 2)
    )
    # Clipa pra range razoável: 1k a 10M
    base = base.withColumn(
        "VAL_NOMINAL_EMISSAO",
        F.least(F.greatest(F.col("VAL_NOMINAL_EMISSAO"), F.lit(1000.0)), F.lit(10_000_000.0))
    )

    # VAL_NOMINAL_ATUAL = EMISSAO * (1 + drift 0% a 30%)
    base = base.withColumn(
        "VAL_NOMINAL_ATUAL",
        F.round(F.col("VAL_NOMINAL_EMISSAO") * (F.lit(1.0) + F.col("_r_drift") * F.lit(0.30)), 2)
    )

    # DAT_INCLUSAO: distribui entre DAT_MIN e DAT_MAX
    total_seconds = int((DAT_MAX - DAT_MIN).total_seconds())
    base = base.withColumn(
        "DAT_INCLUSAO",
        F.from_unixtime(
            F.lit(int(DAT_MIN.timestamp())) +
            (F.col("_r_inc") * F.lit(total_seconds)).cast("long")
        ).cast("timestamp")
    )

    # DAT_ALTERACAO = DAT_INCLUSAO + 0 a 365 dias
    base = base.withColumn(
        "DAT_ALTERACAO",
        F.from_unixtime(
            F.unix_timestamp(F.col("DAT_INCLUSAO")) +
            (F.col("_r_alt") * F.lit(365 * 86400)).cast("long")
        ).cast("timestamp")
    )

    return base.select(
        "NUM_IF", "COD_IF", "COD_ISIN", "NUM_TIPO_IF",
        "VAL_NOMINAL_EMISSAO", "VAL_NOMINAL_ATUAL",
        "DAT_INCLUSAO", "DAT_ALTERACAO"
    )


# ============================================================
# 3) OPERACAO
# ============================================================
def generate_operacao(
    spark: SparkSession,
    instrumento_df: DataFrame,
    n_rows: int,
    seed: int = 42,
) -> DataFrame:
    """
    Gera N operações com FK válida para INSTRUMENTO_FINANCEIRO.
    - NUM_IF + COD_IF mantidos consistentes via JOIN
    - DAT_OPERACAO >= DAT_INCLUSAO do IF
    - VAL_FINANCEIRO = QTD * VAL_PRECO_UNITARIO (constraint de negócio)
    - Cardinalidade variável: alguns IFs concentram mais operações
    """
    n_instrumentos = instrumento_df.count()

    # Pool de IFs com índice 0..N-1 pra lookup
    if_pool = (
        instrumento_df
        .select("NUM_IF", "COD_IF", "DAT_INCLUSAO")
        .withColumn("_if_idx",
            F.row_number().over(
                __import__("pyspark.sql.window", fromlist=["Window"]).Window.orderBy("NUM_IF")
            ) - F.lit(1)
        )
    )

    # Base de operações
    base = spark.range(0, n_rows).withColumnRenamed("id", "NUM_ID_OPERACAO")
    base = base.withColumn("NUM_ID_OPERACAO", (F.col("NUM_ID_OPERACAO") + F.lit(1)).cast("long"))

    # Distribuição enviesada: usa rand^2 pra concentrar operações em poucos IFs (cauda longa)
    # rand^2 puxa pra perto de 0 → primeiros IFs do pool ficam com mais operações
    base = base.withColumn("_r_if",     F.pow(F.rand(seed=seed),      F.lit(2.0)))
    base = base.withColumn("_r_qtd",    F.rand(seed=seed + 10))
    base = base.withColumn("_r_preco",  F.rand(seed=seed + 11))
    base = base.withColumn("_r_dat",    F.rand(seed=seed + 12))
    base = base.withColumn("_r_sit",    F.rand(seed=seed + 13))
    base = base.withColumn("_r_inc",    F.rand(seed=seed + 14))

    # Índice no pool (0..N-1)
    base = base.withColumn(
        "_if_idx",
        F.floor(F.col("_r_if") * F.lit(n_instrumentos)).cast("long")
    )

    # JOIN com pool → traz NUM_IF, COD_IF, DAT_INCLUSAO do IF
    joined = base.join(F.broadcast(if_pool), on="_if_idx", how="left")

    # QTD_OPERACAO: 1 a 100.000, log-normal aproximado
    joined = joined.withColumn(
        "QTD_OPERACAO",
        F.greatest(
            F.round(F.exp(F.lit(4.0) + F.col("_r_qtd") * F.lit(3.0))),
            F.lit(1.0)
        ).cast("long")
    )

    # VAL_PRECO_UNITARIO: 100 a 10.000 (preço de unidade típico de CDB)
    joined = joined.withColumn(
        "VAL_PRECO_UNITARIO",
        F.round(F.lit(100.0) + F.col("_r_preco") * F.lit(9900.0), 4)
    )

    # VAL_FINANCEIRO = QTD * PRECO (constraint de negócio)
    joined = joined.withColumn(
        "VAL_FINANCEIRO",
        F.round(F.col("QTD_OPERACAO") * F.col("VAL_PRECO_UNITARIO"), 2)
    )

    # DAT_OPERACAO: entre DAT_INCLUSAO do IF e DAT_MAX
    joined = joined.withColumn(
        "_dat_if_unix", F.unix_timestamp(F.col("DAT_INCLUSAO"))
    )
    dat_max_unix = int(DAT_MAX.timestamp())
    joined = joined.withColumn(
        "DAT_OPERACAO",
        F.from_unixtime(
            F.col("_dat_if_unix") +
            (F.col("_r_dat") * (F.lit(dat_max_unix) - F.col("_dat_if_unix"))).cast("long")
        ).cast("timestamp")
    )

    # COD_SITUACAO_OPERACAO via CDF
    cum = 0.0
    sit_cdf = []
    for (s, p) in SITUACOES_OPERACAO:
        cum += p
        sit_cdf.append((s, cum))

    sit_expr = None
    for (s, threshold) in sit_cdf:
        cond = F.col("_r_sit") <= F.lit(threshold)
        val = F.lit(s)
        sit_expr = F.when(cond, val) if sit_expr is None else sit_expr.when(cond, val)
    sit_expr = sit_expr.otherwise(F.lit(sit_cdf[-1][0]))
    joined = joined.withColumn("COD_SITUACAO_OPERACAO", sit_expr)

    # DAT_INCLUSAO da operação: 0 a 90 dias após DAT_OPERACAO
    joined = joined.withColumn(
        "DAT_INCLUSAO",
        F.from_unixtime(
            F.unix_timestamp(F.col("DAT_OPERACAO")) +
            (F.col("_r_inc") * F.lit(90 * 86400)).cast("long")
        ).cast("timestamp")
    )

    return joined.select(
        "NUM_ID_OPERACAO",
        "NUM_IF", "COD_IF",
        "DAT_OPERACAO",
        "QTD_OPERACAO", "VAL_PRECO_UNITARIO", "VAL_FINANCEIRO",
        "COD_SITUACAO_OPERACAO",
        "DAT_INCLUSAO"
    )


# ============================================================
# Orquestração
# ============================================================
def generate_cdb_tables(
    spark: SparkSession,
    n_instrumentos: int = 1000,
    n_operacoes: int = 50000,
    seed: int = 42,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Gera as 3 tabelas em ordem topológica com relacionamentos válidos.
    Retorna (tipo_if, instrumento_financeiro, operacao).
    """
    print(f"Gerando TIPO_IF ({len(TIPOS_IF)} linhas)...")
    tipo_if = generate_tipo_if(spark)

    print(f"Gerando INSTRUMENTO_FINANCEIRO ({n_instrumentos} linhas)...")
    instrumento = generate_instrumento_financeiro(spark, n_instrumentos, seed=seed)
    instrumento.cache()
    instrumento.count()  # materializa

    print(f"Gerando OPERACAO ({n_operacoes} linhas)...")
    operacao = generate_operacao(spark, instrumento, n_operacoes, seed=seed)

    return tipo_if, instrumento, operacao


# ============================================================
# Validação rápida (smoke tests)
# ============================================================
def validate_relationships(tipo_if: DataFrame, instrumento: DataFrame, operacao: DataFrame):
    """Roda checks básicos de integridade referencial e constraints."""
    print("\n=== VALIDAÇÃO ===")

    # FK INSTRUMENTO -> TIPO_IF
    orfaos_if = instrumento.join(tipo_if, on="NUM_TIPO_IF", how="left_anti").count()
    print(f"INSTRUMENTOs sem TIPO_IF correspondente: {orfaos_if}")

    # FK OPERACAO -> INSTRUMENTO (NUM_IF)
    orfaos_op = operacao.join(instrumento.select("NUM_IF"), on="NUM_IF", how="left_anti").count()
    print(f"OPERACAOs sem INSTRUMENTO correspondente (por NUM_IF): {orfaos_op}")

    # Consistência NUM_IF + COD_IF em OPERACAO
    inconsist = (
        operacao.alias("o")
        .join(instrumento.alias("i"), on="NUM_IF")
        .where(F.col("o.COD_IF") != F.col("i.COD_IF"))
        .count()
    )
    print(f"OPERACAOs com NUM_IF e COD_IF inconsistentes: {inconsist}")

    # Constraint VAL_FINANCEIRO = QTD * PRECO
    quebras = (
        operacao
        .where(
            F.abs(F.col("VAL_FINANCEIRO") - F.col("QTD_OPERACAO") * F.col("VAL_PRECO_UNITARIO"))
            > F.lit(0.01)
        )
        .count()
    )
    print(f"OPERACAOs onde VAL_FINANCEIRO != QTD * PRECO: {quebras}")

    # DAT_OPERACAO >= DAT_INCLUSAO do IF
    temporal = (
        operacao.alias("o")
        .join(instrumento.alias("i"), on="NUM_IF")
        .where(F.col("o.DAT_OPERACAO") < F.col("i.DAT_INCLUSAO"))
        .count()
    )
    print(f"OPERACAOs com DAT_OPERACAO < DAT_INCLUSAO do IF: {temporal}")

    # Distribuição de TIPO_IF
    print("\nDistribuição NUM_TIPO_IF em INSTRUMENTO_FINANCEIRO:")
    (instrumento.groupBy("NUM_TIPO_IF").count()
        .join(tipo_if.select("NUM_TIPO_IF", "COD_TIPO_IF"), on="NUM_TIPO_IF")
        .orderBy("NUM_TIPO_IF").show())

    # Cardinalidade pai-filho
    print("Operações por INSTRUMENTO (top 10):")
    (operacao.groupBy("NUM_IF").count()
        .orderBy(F.desc("count")).show(10))

    print("Operações por INSTRUMENTO (stats):")
    (operacao.groupBy("NUM_IF").count()
        .agg(
            F.min("count").alias("min"),
            F.max("count").alias("max"),
            F.avg("count").alias("avg"),
            F.expr("percentile(count, 0.5)").alias("median"),
        ).show())