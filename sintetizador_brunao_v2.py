"""
synthetic_multitable_spark_generic.py
=====================================

Gerador de dados sintéticos MULTI-TABELA em PySpark, sem tabelas hardcoded.

Objetivo:
    Gerar dados sintéticos preservando:
      - estrutura relacional;
      - primary keys únicas;
      - foreign keys válidas;
      - distribuições por bootstrap de linhas inteiras;
      - relacionamentos entre tabelas via remapeamento old_key -> synthetic_key.

Características:
    1. PySpark.
    2. Suporta múltiplas tabelas.
    3. Suporta FK simples e composta.
    4. Suporta DAG de dependências entre tabelas.
    5. Suporta tabela estática/dimensão.
    6. Permite postprocess por tabela, injetado por configuração.
    7. Valida PK/FK com modos configuráveis.
    8. Evita broadcast obrigatório de mappings grandes.
    9. Usa cache com MEMORY_AND_DISK.

Limitações:
    - Self-reference não é suportado.
    - Ciclos relacionais não são suportados.
    - Bootstrap de linhas inteiras pode repetir linhas.
    - Não é anonimização forte por si só.
    - Para 100+ tabelas grandes, recomenda-se reduzir validate_mode para "none"
      após validar a corretude em ambiente controlado.
"""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple
import warnings
import zlib

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ============================================================
# 1. Tipagens e specs
# ============================================================

NullableFkPolicy = Literal["allow_any_null", "allow_all_null", "invalid_null"]
ValidateMode = Literal["none", "full"]


@dataclass(frozen=True)
class ForeignKeySpec:
    """
    Declara uma FK do filho apontando para um pai.

    Exemplo de FK simples:
        ForeignKeySpec(
            columns=("ID_CLIENTE",),
            parent_table="cliente",
            parent_columns=("ID_CLIENTE",),
        )

    Exemplo de FK composta:
        ForeignKeySpec(
            columns=("ID_PRODUTO", "COD_PRODUTO"),
            parent_table="produto",
            parent_columns=("ID_PRODUTO", "COD_PRODUTO"),
        )
    """
    columns: Tuple[str, ...]
    parent_table: str
    parent_columns: Tuple[str, ...]


PostProcessor = Callable[[DataFrame, Mapping[str, DataFrame]], DataFrame]


@dataclass(frozen=True)
class TableSpec:
    """
    Especificação declarativa de uma tabela.

    Args:
        name:
            Nome lógico da tabela.
        pk_cols:
            Colunas que formam a primary key.
        foreign_keys:
            FKs da tabela.
        static:
            Se True, a tabela é copiada como está.
        postprocess:
            Função opcional aplicada depois de PK e FK serem ajustadas.
            Assinatura:
                postprocess(work_df, generated_tables) -> work_df
    """
    name: str
    pk_cols: Tuple[str, ...]
    foreign_keys: Tuple[ForeignKeySpec, ...] = field(default_factory=tuple)
    static: bool = False
    postprocess: Optional[PostProcessor] = None


SpecConfig = Mapping[str, Mapping[str, Any]]
TablePaths = Mapping[str, str]


# ============================================================
# 2. Utilitários de tipo, seed e persistência
# ============================================================

def _stable_seed(base_seed: int, *parts: object) -> int:
    """
    Gera seed determinístico sem depender de hash() do Python.
    """
    txt = "|".join(str(p) for p in (base_seed,) + parts)
    return int(zlib.crc32(txt.encode("utf-8")) % 2_000_000_000)


def _is_integer_type(dt: T.DataType) -> bool:
    return isinstance(dt, (T.ByteType, T.ShortType, T.IntegerType, T.LongType))


def _is_string_type(dt: T.DataType) -> bool:
    return isinstance(dt, T.StringType)


def _is_safe_pk_type(dt: T.DataType) -> bool:
    return _is_integer_type(dt) or _is_string_type(dt)


def _get_field_type(df: DataFrame, col_name: str) -> T.DataType:
    for f in df.schema.fields:
        if f.name == col_name:
            return f.dataType
    raise ValueError(f"Coluna `{col_name}` não existe no DataFrame.")


def _persist(df: DataFrame, storage_level: StorageLevel) -> DataFrame:
    return df.persist(storage_level)


def _safe_unpersist(df: Optional[DataFrame]) -> None:
    if df is None:
        return
    try:
        df.unpersist()
    except Exception:
        pass


# ============================================================
# 3. Validação das specs e ordenação topológica
# ============================================================

def _validate_specs(
    tables: Mapping[str, DataFrame],
    specs: Mapping[str, TableSpec],
) -> None:
    """
    Valida estrutura das specs antes de executar Spark jobs pesados.
    """
    if not tables:
        raise ValueError("`tables` está vazio.")

    if not specs:
        raise ValueError("`specs` está vazio.")

    for name, spec in specs.items():
        if name not in tables:
            raise ValueError(f"Tabela `{name}` está em specs, mas não está em tables.")

        if spec.name != name:
            raise ValueError(
                f"Inconsistência: chave specs=`{name}`, mas TableSpec.name=`{spec.name}`."
            )

        if not spec.pk_cols:
            raise ValueError(f"Tabela `{name}` precisa ter pelo menos uma coluna de PK.")

        df_cols = set(tables[name].columns)

        for pk in spec.pk_cols:
            if pk not in df_cols:
                raise ValueError(
                    f"PK col `{pk}` não existe na tabela `{name}`. "
                    f"Colunas disponíveis: {sorted(df_cols)}"
                )

        seen_fk_cols: set[str] = set()

        for fk in spec.foreign_keys:
            if not fk.columns:
                raise ValueError(f"FK vazia declarada na tabela `{name}`.")

            if len(fk.columns) != len(fk.parent_columns):
                raise ValueError(
                    f"FK inválida em `{name}`: columns={fk.columns}, "
                    f"parent_columns={fk.parent_columns}. Tamanhos diferentes."
                )

            if fk.parent_table == name:
                raise ValueError(
                    f"Self-reference não suportado: tabela `{name}` referencia ela mesma."
                )

            if fk.parent_table not in specs:
                raise ValueError(
                    f"FK em `{name}` referencia `{fk.parent_table}`, "
                    f"mas essa tabela não existe em specs."
                )

            if fk.parent_table not in tables:
                raise ValueError(
                    f"FK em `{name}` referencia `{fk.parent_table}`, "
                    f"mas essa tabela não existe em tables."
                )

            for c in fk.columns:
                if c not in df_cols:
                    raise ValueError(
                        f"FK col `{c}` não existe na tabela filha `{name}`."
                    )

                if c in seen_fk_cols:
                    raise ValueError(
                        f"Coluna `{c}` em `{name}` participa de mais de uma FK. "
                        f"Este framework evita remapeamento ambíguo. "
                        f"Se isso for necessário, crie postprocess customizado."
                    )

                seen_fk_cols.add(c)

            parent_cols = set(tables[fk.parent_table].columns)
            for pc in fk.parent_columns:
                if pc not in parent_cols:
                    raise ValueError(
                        f"FK em `{name}` referencia coluna `{pc}`, "
                        f"mas ela não existe no pai `{fk.parent_table}`."
                    )


def _topological_order(specs: Mapping[str, TableSpec]) -> List[str]:
    """
    Ordena as tabelas garantindo que pais sejam gerados antes dos filhos.
    """
    remaining = set(specs.keys())
    done: set[str] = set()
    order: List[str] = []

    while remaining:
        ready = [
            name
            for name in remaining
            if {fk.parent_table for fk in specs[name].foreign_keys}.issubset(done)
        ]

        if not ready:
            unresolved = {
                table: [fk.parent_table for fk in specs[table].foreign_keys]
                for table in remaining
            }
            raise ValueError(
                "Ciclo relacional, self-reference ou pai ausente detectado. "
                f"Pendências: {unresolved}"
            )

        for name in sorted(ready):
            order.append(name)
            done.add(name)
            remaining.remove(name)

    return order


def _referenced_parent_columns(
    specs: Mapping[str, TableSpec],
) -> Dict[str, set[Tuple[str, ...]]]:
    """
    Para cada tabela pai, lista quais conjuntos de colunas são referenciados.
    """
    refs: Dict[str, set[Tuple[str, ...]]] = {}

    for child_spec in specs.values():
        for fk in child_spec.foreign_keys:
            refs.setdefault(fk.parent_table, set()).add(tuple(fk.parent_columns))

    return refs


# ============================================================
# 4. Indexação e bootstrap
# ============================================================

def _with_contiguous_row_id(df: DataFrame, id_col: str) -> DataFrame:
    """
    Adiciona ID contíguo 0..N-1.

    Observação:
        Usa RDD zipWithIndex porque é mais adequado para índice contíguo
        do que Window global com row_number, que tende a concentrar em uma partição.
    """
    spark = df.sparkSession

    schema_with_id = T.StructType(
        df.schema.fields + [T.StructField(id_col, T.LongType(), False)]
    )

    indexed_rdd = df.rdd.zipWithIndex().map(
        lambda row_with_idx: tuple(row_with_idx[0]) + (row_with_idx[1],)
    )

    return spark.createDataFrame(indexed_rdd, schema=schema_with_id)


def _bootstrap_rows_exact(
    src_indexed: DataFrame,
    n_rows: int,
    *,
    src_count: int,
    seed: int,
    spark: SparkSession,
    keep_all_source_rows: bool,
) -> DataFrame:
    """
    Gera exatamente n_rows por bootstrap de linhas inteiras.

    Se keep_all_source_rows=True:
        Garante que toda linha original apareça pelo menos uma vez.
        Isso é necessário para tabela pai, pois assegura cobertura total
        do mapping old_key -> synthetic_key.

    Se keep_all_source_rows=False:
        Faz bootstrap uniforme puro.
    """
    if n_rows < 0:
        raise ValueError("n_rows deve ser >= 0.")

    src_cols = [c for c in src_indexed.columns if c != "__src_row_id"]

    if n_rows == 0:
        empty_schema = T.StructType(
            [
                T.StructField("__synthetic_pos", T.LongType(), False),
                T.StructField("__orig_src_row_id", T.LongType(), True),
            ]
            + [f for f in src_indexed.schema.fields if f.name in src_cols]
        )
        return spark.createDataFrame([], schema=empty_schema)

    if src_count == 0:
        raise ValueError(
            "A tabela fonte está vazia, mas n_rows > 0. "
            "Não há linhas para amostrar."
        )

    if keep_all_source_rows:
        if n_rows < src_count:
            raise ValueError(
                f"Tabela pai precisa de n_rows >= cardinalidade original. "
                f"Recebido n_rows={n_rows}, src_count={src_count}."
            )

        base_keep = (
            src_indexed
            .withColumn("__synthetic_pos", F.col("__src_row_id"))
            .withColumn("__orig_src_row_id", F.col("__src_row_id"))
            .select("__synthetic_pos", "__orig_src_row_id", *src_cols)
        )

        extra_n = n_rows - src_count

        if extra_n == 0:
            return base_keep

        extra_positions = (
            spark.range(src_count, n_rows)
            .withColumnRenamed("id", "__synthetic_pos")
            .withColumn(
                "__lookup_src_row_id",
                F.floor(F.rand(seed) * F.lit(src_count)).cast("long"),
            )
        )

        extra = (
            extra_positions
            .join(
                src_indexed,
                extra_positions["__lookup_src_row_id"] == src_indexed["__src_row_id"],
                "left",
            )
            .withColumn("__orig_src_row_id", F.col("__src_row_id"))
            .select("__synthetic_pos", "__orig_src_row_id", *src_cols)
        )

        return base_keep.unionByName(extra)

    positions = (
        spark.range(0, n_rows)
        .withColumnRenamed("id", "__synthetic_pos")
        .withColumn(
            "__lookup_src_row_id",
            F.floor(F.rand(seed) * F.lit(src_count)).cast("long"),
        )
    )

    return (
        positions
        .join(
            src_indexed,
            positions["__lookup_src_row_id"] == src_indexed["__src_row_id"],
            "left",
        )
        .withColumn("__orig_src_row_id", F.col("__src_row_id"))
        .select("__synthetic_pos", "__orig_src_row_id", *src_cols)
    )


# ============================================================
# 5. Geração de PK
# ============================================================

_INT_TYPE_LIMITS = (
    (T.ByteType, 127),
    (T.ShortType, 32_767),
    (T.IntegerType, 2_147_483_647),
)


def _max_pk_value(df_cached: DataFrame, pk: str) -> Optional[int]:
    row = df_cached.agg(F.max(F.col(pk)).alias("max_pk")).collect()[0]
    return int(row["max_pk"]) if row["max_pk"] is not None else None


def _set_unique_pk_column(
    work: DataFrame,
    source_cached: DataFrame,
    pk: str,
    *,
    append_after_max: bool,
    target_n: int,
    offset: int = 0,
) -> DataFrame:
    """
    Sobrescreve uma coluna PK com valores únicos.
    """
    dt = _get_field_type(source_cached, pk)

    if _is_integer_type(dt):
        start = (_max_pk_value(source_cached, pk) or 0) + 1 if append_after_max else 1
        highest = start + target_n - 1 + offset

        for type_cls, limit in _INT_TYPE_LIMITS:
            if isinstance(dt, type_cls) and highest > limit:
                raise OverflowError(
                    f"PK `{pk}` é {type_cls.__name__}, mas o maior valor sintético "
                    f"seria {highest:,}, excedendo o limite {limit:,}. "
                    f"Converta a coluna para LongType ou reduza n_rows."
                )

        return work.withColumn(
            pk,
            (F.col("__synthetic_pos") + F.lit(start + offset)).cast(dt),
        )

    if _is_string_type(dt):
        return work.withColumn(
            pk,
            F.concat(
                F.lit(f"SYN_{pk}_"),
                F.lpad(
                    (F.col("__synthetic_pos") + F.lit(offset)).cast("string"),
                    14,
                    "0",
                ),
            ).cast(dt),
        )

    raise TypeError(
        f"PK `{pk}` possui tipo {dt!r}, sem estratégia automática segura. "
        f"Use PK IntegerType, LongType ou StringType, ou implemente postprocess customizado."
    )


def _generate_pk_columns(
    work: DataFrame,
    source_cached: DataFrame,
    spec: TableSpec,
    *,
    append_after_max: bool,
    target_n: int,
) -> DataFrame:
    """
    Gera PK sintética.

    Estratégia:
        - PK simples: sobrescreve a única coluna.
        - PK composta:
            - se última coluna é int/string, usa última coluna como driver único;
            - caso contrário, lança erro.
    """
    if len(spec.pk_cols) == 1:
        return _set_unique_pk_column(
            work,
            source_cached,
            spec.pk_cols[0],
            append_after_max=append_after_max,
            target_n=target_n,
            offset=0,
        )

    last_pk = spec.pk_cols[-1]
    last_type = _get_field_type(source_cached, last_pk)

    if not _is_safe_pk_type(last_type):
        raise TypeError(
            f"PK composta da tabela `{spec.name}` usa última coluna `{last_pk}` "
            f"com tipo {last_type!r}. Para PK composta automática, a última coluna "
            f"precisa ser int ou string."
        )

    return _set_unique_pk_column(
        work,
        source_cached,
        last_pk,
        append_after_max=append_after_max,
        target_n=target_n,
        offset=0,
    )


# ============================================================
# 6. Mapping old -> new e remapeamento de FKs
# ============================================================

def _build_mapping_for_parent_cols(
    work_cached: DataFrame,
    parent_cols: Tuple[str, ...],
    storage_level: StorageLevel,
) -> DataFrame:
    """
    Cria mapping:
        chave original do pai -> chave sintética do pai

    Se uma chave original gerar vários candidatos sintéticos, cada candidato
    recebe rank. Depois o filho sorteia um rank válido.
    """
    old_cols = [f"__old__{c}" for c in parent_cols]

    missing_old = [c for c in old_cols if c not in work_cached.columns]
    if missing_old:
        raise ValueError(
            f"Não foi possível construir mapping. Colunas antigas ausentes: {missing_old}"
        )

    mapping = work_cached.select(
        *[
            F.col(old_cols[i]).alias(f"__old_{i}")
            for i in range(len(parent_cols))
        ],
        *[
            F.col(parent_cols[i]).alias(f"__new_{i}")
            for i in range(len(parent_cols))
        ],
        F.col("__synthetic_pos"),
    )

    partition_cols = [F.col(f"__old_{i}") for i in range(len(parent_cols))]
    w = Window.partitionBy(*partition_cols).orderBy(F.col("__synthetic_pos"))

    mapping = mapping.withColumn(
        "__candidate_rank",
        F.row_number().over(w).cast("long"),
    )

    counts = mapping.groupBy(
        *[F.col(f"__old_{i}") for i in range(len(parent_cols))]
    ).agg(
        F.count(F.lit(1)).cast("long").alias("__candidate_count")
    )

    mapping = mapping.join(
        counts,
        on=[f"__old_{i}" for i in range(len(parent_cols))],
        how="left",
    )

    return _persist(mapping, storage_level)


def _fk_join_condition(
    left_df: DataFrame,
    left_cols: List[str],
    right_df: DataFrame,
    right_cols: List[str],
):
    conditions = [
        left_df[left_cols[i]].eqNullSafe(right_df[right_cols[i]])
        for i in range(len(left_cols))
    ]
    return reduce(lambda a, b: a & b, conditions)


def _apply_fk_mapping(
    work: DataFrame,
    fk: ForeignKeySpec,
    mapping: DataFrame,
    *,
    seed: int,
    broadcast_fk_counts: bool,
) -> DataFrame:
    """
    Remapeia FK do filho para valores sintéticos existentes no pai.
    """
    fk_tag = (
        f"__fk_{fk.parent_table}_"
        f"{_stable_seed(seed, fk.parent_table, fk.columns, fk.parent_columns)}"
    )

    n = len(fk.columns)

    counts = mapping.select(
        *[
            F.col(f"__old_{i}").alias(f"{fk_tag}_old_{i}")
            for i in range(n)
        ],
        F.col("__candidate_count").alias(f"{fk_tag}_count"),
    ).dropDuplicates(
        [f"{fk_tag}_old_{i}" for i in range(n)]
    )

    count_old_cols = [f"{fk_tag}_old_{i}" for i in range(n)]
    cond_counts = _fk_join_condition(
        work,
        list(fk.columns),
        counts,
        count_old_cols,
    )

    if broadcast_fk_counts:
        work = work.join(F.broadcast(counts), cond_counts, "left")
    else:
        work = work.join(counts, cond_counts, "left")

    work = work.withColumn(
        f"{fk_tag}_rank",
        F.when(
            F.col(f"{fk_tag}_count").isNull(),
            F.lit(None).cast("long"),
        ).otherwise(
            F.floor(
                F.rand(_stable_seed(seed, fk_tag, "rank"))
                * F.col(f"{fk_tag}_count")
            ).cast("long") + F.lit(1)
        ),
    )

    m = mapping.select(
        *[
            F.col(f"__old_{i}").alias(f"{fk_tag}_map_old_{i}")
            for i in range(n)
        ],
        *[
            F.col(f"__new_{i}").alias(f"{fk_tag}_new_{i}")
            for i in range(n)
        ],
        F.col("__candidate_rank").alias(f"{fk_tag}_map_rank"),
    )

    map_old_cols = [f"{fk_tag}_map_old_{i}" for i in range(n)]

    cond_map_key = _fk_join_condition(
        work,
        list(fk.columns),
        m,
        map_old_cols,
    )

    cond_map = cond_map_key & (
        work[f"{fk_tag}_rank"] == m[f"{fk_tag}_map_rank"]
    )

    work = work.join(m, cond_map, "left")

    for i, child_col in enumerate(fk.columns):
        child_type = _get_field_type(work, child_col)
        work = work.withColumn(
            child_col,
            F.col(f"{fk_tag}_new_{i}").cast(child_type),
        )

    drop_cols = (
        [f"{fk_tag}_old_{i}" for i in range(n)]
        + [f"{fk_tag}_map_old_{i}" for i in range(n)]
        + [f"{fk_tag}_new_{i}" for i in range(n)]
        + [
            f"{fk_tag}_count",
            f"{fk_tag}_rank",
            f"{fk_tag}_map_rank",
        ]
    )

    return work.drop(*drop_cols)


# ============================================================
# 7. Validações de resultado
# ============================================================

def validate_primary_keys(
    tables: Mapping[str, DataFrame],
    specs: Mapping[str, TableSpec],
) -> DataFrame:
    """
    Retorna diagnóstico de PK por tabela.
    """
    if not tables:
        raise ValueError("`tables` está vazio.")

    spark = next(iter(tables.values())).sparkSession
    rows = []

    for name, spec in specs.items():
        df = tables[name]

        total_rows = df.count()
        distinct_pk = df.select(*spec.pk_cols).dropDuplicates().count()

        null_condition = reduce(
            lambda a, b: a | b,
            [F.col(c).isNull() for c in spec.pk_cols],
        )

        null_pk_rows = df.where(null_condition).count()
        duplicate_pk_rows = total_rows - distinct_pk

        rows.append(
            (
                name,
                ",".join(spec.pk_cols),
                int(total_rows),
                int(distinct_pk),
                int(null_pk_rows),
                int(duplicate_pk_rows),
            )
        )

    schema = (
        "table string, pk_cols string, total_rows long, distinct_pk long, "
        "null_pk_rows long, duplicate_pk_rows long"
    )

    return spark.createDataFrame(rows, schema=schema)


def _filter_child_fk_for_validation(
    child_df: DataFrame,
    fk: ForeignKeySpec,
    nullable_fk_policy: NullableFkPolicy,
) -> DataFrame:
    """
    Define como FKs nulas devem ser tratadas na validação.

    allow_any_null:
        Se qualquer coluna da FK for nula, ignora essa linha na validação.

    allow_all_null:
        Só ignora se todas as colunas da FK forem nulas.
        FK parcialmente nula será validada e tende a ser inválida.

    invalid_null:
        Não ignora nulos. Nulos podem aparecer como inválidos.
    """
    if nullable_fk_policy == "invalid_null":
        return child_df

    any_null = reduce(
        lambda a, b: a | b,
        [F.col(c).isNull() for c in fk.columns],
    )

    all_null = reduce(
        lambda a, b: a & b,
        [F.col(c).isNull() for c in fk.columns],
    )

    if nullable_fk_policy == "allow_any_null":
        return child_df.where(~any_null)

    if nullable_fk_policy == "allow_all_null":
        return child_df.where(~all_null)

    raise ValueError(f"nullable_fk_policy inválida: {nullable_fk_policy}")


def validate_foreign_keys(
    tables: Mapping[str, DataFrame],
    specs: Mapping[str, TableSpec],
    *,
    nullable_fk_policy: NullableFkPolicy = "allow_any_null",
) -> DataFrame:
    """
    Retorna diagnóstico de FK por relacionamento.
    """
    if not tables:
        raise ValueError("`tables` está vazio.")

    spark = next(iter(tables.values())).sparkSession
    rows = []

    for child_name, child_spec in specs.items():
        child_df_raw = tables[child_name]

        for fk in child_spec.foreign_keys:
            parent_df = tables[fk.parent_table]

            child_df = _filter_child_fk_for_validation(
                child_df_raw,
                fk,
                nullable_fk_policy,
            )

            child_keys = child_df.select(*fk.columns).dropDuplicates()

            parent_keys = parent_df.select(
                *[
                    F.col(parent_col).alias(child_col)
                    for child_col, parent_col in zip(fk.columns, fk.parent_columns)
                ]
            ).dropDuplicates()

            invalid = child_keys.join(
                parent_keys,
                on=list(fk.columns),
                how="left_anti",
            ).count()

            total_distinct = child_keys.count()

            rows.append(
                (
                    child_name,
                    ",".join(fk.columns),
                    fk.parent_table,
                    ",".join(fk.parent_columns),
                    int(total_distinct),
                    int(invalid),
                )
            )

    schema = (
        "child_table string, fk_cols string, parent_table string, parent_cols string, "
        "distinct_child_fk long, invalid_fk long"
    )

    return spark.createDataFrame(rows, schema=schema)


def _run_validation_or_raise(
    result: Mapping[str, DataFrame],
    specs: Mapping[str, TableSpec],
    *,
    nullable_fk_policy: NullableFkPolicy,
) -> None:
    pk_report = validate_primary_keys(result, specs)
    fk_report = validate_foreign_keys(
        result,
        specs,
        nullable_fk_policy=nullable_fk_policy,
    )

    pk_problems = pk_report.where(
        "null_pk_rows > 0 OR duplicate_pk_rows > 0"
    ).count()

    fk_problems = fk_report.where(
        "invalid_fk > 0"
    ).count()

    if pk_problems or fk_problems:
        print(">>> FALHA NA VALIDAÇÃO DE PK/FK")
        print(">>> PRIMARY KEYS")
        pk_report.show(truncate=False)

        print(">>> FOREIGN KEYS")
        fk_report.show(truncate=False)

        raise RuntimeError(
            f"Validação falhou: {pk_problems} tabela(s) com PK inválida, "
            f"{fk_problems} relação(ões) com FK inválida."
        )


def run_validation_or_raise(
    tables: Mapping[str, DataFrame],
    specs: Mapping[str, TableSpec],
    *,
    nullable_fk_policy: NullableFkPolicy = "allow_any_null",
) -> None:
    """
    Wrapper público para validar PK/FK e lançar erro se houver problema.
    """
    _run_validation_or_raise(
        result=tables,
        specs=specs,
        nullable_fk_policy=nullable_fk_policy,
    )


# ============================================================
# 8. Função principal do gerador
# ============================================================

def synthesize_multitable_spark(
    tables: Mapping[str, DataFrame],
    specs: Mapping[str, TableSpec],
    n_rows_by_table: Optional[Mapping[str, int]] = None,
    *,
    seed: int = 42,
    append_after_max_pk: bool = True,
    validate_mode: ValidateMode = "full",
    nullable_fk_policy: NullableFkPolicy = "allow_any_null",
    broadcast_fk_counts: bool = False,
    storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    verbose: bool = False,
) -> Dict[str, DataFrame]:
    """
    Gera dados sintéticos multi-tabela.

    Args:
        tables:
            Dict nome_tabela -> DataFrame original.
        specs:
            Dict nome_tabela -> TableSpec.
        n_rows_by_table:
            Dict nome_tabela -> número de linhas sintéticas.
            Se None, usa o volume original.
        seed:
            Semente global.
        append_after_max_pk:
            Se True, PK nova começa após max(PK original).
        validate_mode:
            "full" ou "none".
        nullable_fk_policy:
            Política para validar FKs nulas.
        broadcast_fk_counts:
            Se True, força broadcast da tabela de contagem de candidatos.
            Recomendado False para tabelas grandes.
        storage_level:
            Nível de persistência Spark.
        verbose:
            Se True, imprime progresso.

    Returns:
        Dict nome_tabela -> DataFrame sintético.
    """
    if validate_mode not in ("none", "full"):
        raise ValueError("validate_mode deve ser 'none' ou 'full'.")

    if nullable_fk_policy not in ("allow_any_null", "allow_all_null", "invalid_null"):
        raise ValueError(
            "nullable_fk_policy deve ser 'allow_any_null', "
            "'allow_all_null' ou 'invalid_null'."
        )

    _validate_specs(tables, specs)

    n_rows_by_table = dict(n_rows_by_table or {})
    order = _topological_order(specs)
    parent_refs = _referenced_parent_columns(specs)

    result: Dict[str, DataFrame] = {}
    mappings: Dict[Tuple[str, Tuple[str, ...]], DataFrame] = {}
    intermediates: List[DataFrame] = []

    try:
        for table_name in order:
            source = tables[table_name]
            spec = specs[table_name]
            spark = source.sparkSession
            original_cols = source.columns

            target_n_raw = n_rows_by_table.get(table_name)

            ref_col_sets = parent_refs.get(table_name, set())

            ref_cols = sorted(
                set(c for cols in ref_col_sets for c in cols)
                | set(spec.pk_cols)
            )

            if spec.static:
                src_count = source.count()

                if target_n_raw is not None and int(target_n_raw) != src_count:
                    warnings.warn(
                        f"Tabela `{table_name}` é static=True. "
                        f"n_rows_by_table={target_n_raw} será ignorado. "
                        f"Serão copiadas {src_count} linhas."
                    )

                if verbose:
                    print(f"[{table_name}] STATIC | copiando {src_count} linhas")

                work = (
                    _with_contiguous_row_id(source, "__synthetic_pos")
                    .withColumn("__orig_src_row_id", F.col("__synthetic_pos"))
                )

                for c in ref_cols:
                    work = work.withColumn(f"__old__{c}", F.col(c))

                if spec.postprocess is not None:
                    work = spec.postprocess(work, result)

                work = _persist(work, storage_level)
                work.count()
                intermediates.append(work)

            else:
                src_indexed = _with_contiguous_row_id(source, "__src_row_id")
                src_indexed = _persist(src_indexed, storage_level)
                src_count = src_indexed.count()
                intermediates.append(src_indexed)

                target_n = int(target_n_raw if target_n_raw is not None else src_count)
                keep_all = table_name in parent_refs

                if verbose:
                    role = "PAI" if keep_all else "FILHO"
                    print(
                        f"[{table_name}] {role} | origem={src_count:,} "
                        f"-> sintético={target_n:,}"
                    )

                work = _bootstrap_rows_exact(
                    src_indexed,
                    target_n,
                    src_count=src_count,
                    seed=_stable_seed(seed, table_name, "bootstrap"),
                    spark=spark,
                    keep_all_source_rows=keep_all,
                )

                for c in ref_cols:
                    work = work.withColumn(f"__old__{c}", F.col(c))

                work = _generate_pk_columns(
                    work,
                    src_indexed,
                    spec,
                    append_after_max=append_after_max_pk,
                    target_n=target_n,
                )

                for fk in spec.foreign_keys:
                    key = (fk.parent_table, tuple(fk.parent_columns))

                    if key not in mappings:
                        raise ValueError(
                            f"Mapping não encontrado para FK "
                            f"{table_name}.{fk.columns} -> "
                            f"{fk.parent_table}.{fk.parent_columns}"
                        )

                    work = _apply_fk_mapping(
                        work,
                        fk,
                        mappings[key],
                        seed=_stable_seed(
                            seed,
                            table_name,
                            fk.parent_table,
                            fk.columns,
                            fk.parent_columns,
                        ),
                        broadcast_fk_counts=broadcast_fk_counts,
                    )

                if spec.postprocess is not None:
                    work = spec.postprocess(work, result)

                work = _persist(work, storage_level)
                work.count()
                intermediates.append(work)

            if table_name in parent_refs:
                for cols in parent_refs[table_name]:
                    mapping_df = _build_mapping_for_parent_cols(
                        work,
                        tuple(cols),
                        storage_level=storage_level,
                    )
                    mapping_df.count()
                    mappings[(table_name, tuple(cols))] = mapping_df
                    intermediates.append(mapping_df)

            synth = work.select(*original_cols)
            synth = _persist(synth, storage_level)
            synth.count()

            result[table_name] = synth

        if validate_mode == "full":
            if verbose:
                print("Validando PKs e FKs...")

            _run_validation_or_raise(
                result,
                specs,
                nullable_fk_policy=nullable_fk_policy,
            )

            if verbose:
                print("Validação concluída: PKs e FKs OK.")

        return result

    except Exception:
        for df in result.values():
            _safe_unpersist(df)
        raise

    finally:
        for df in intermediates:
            _safe_unpersist(df)


# ============================================================
# 9. Camada genérica de configuração, leitura, execução e escrita
# ============================================================

def _as_tuple(
    value: Any,
    *,
    field_name: str,
    table_name: str,
    required: bool = True,
) -> Tuple[str, ...]:
    """
    Normaliza campos de configuração para tupla de strings.

    Aceita:
        "COLUNA"
        ["COLUNA_1", "COLUNA_2"]
        ("COLUNA_1", "COLUNA_2")
    """
    if value is None:
        if required:
            raise ValueError(
                f"Campo `{field_name}` é obrigatório na tabela `{table_name}`."
            )
        return tuple()

    if isinstance(value, str):
        value = value.strip()

        if not value and required:
            raise ValueError(
                f"Campo `{field_name}` está vazio na tabela `{table_name}`."
            )

        return (value,) if value else tuple()

    if isinstance(value, (list, tuple)):
        out: List[str] = []

        for item in value:
            if item is None:
                raise ValueError(
                    f"Campo `{field_name}` possui valor None na tabela `{table_name}`."
                )

            item_str = str(item).strip()

            if not item_str:
                raise ValueError(
                    f"Campo `{field_name}` possui valor vazio na tabela `{table_name}`."
                )

            out.append(item_str)

        if required and not out:
            raise ValueError(
                f"Campo `{field_name}` está vazio na tabela `{table_name}`."
            )

        return tuple(out)

    raise TypeError(
        f"Campo `{field_name}` da tabela `{table_name}` deve ser string, list ou tuple. "
        f"Recebido: {type(value).__name__}"
    )


def _as_bool(value: Any, *, default: bool = False) -> bool:
    """
    Converte valores comuns de configuração para bool.
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value_norm = value.strip().lower()

        if value_norm in {"true", "1", "yes", "y", "sim", "s"}:
            return True

        if value_norm in {"false", "0", "no", "n", "nao", "não"}:
            return False

        raise ValueError(f"Valor booleano inválido: {value!r}")

    return bool(value)


def _get_parent_pk_from_config(
    specs_config: SpecConfig,
    *,
    parent_table: str,
    child_table: str,
) -> Tuple[str, ...]:
    """
    Permite omitir parent_columns na FK.

    Se parent_columns não for informado, usa pk_cols da tabela pai.
    """
    if parent_table not in specs_config:
        raise ValueError(
            f"A tabela filha `{child_table}` referencia `{parent_table}`, "
            f"mas `{parent_table}` não existe em specs_config."
        )

    parent_raw = specs_config[parent_table]

    if "pk_cols" not in parent_raw:
        raise ValueError(
            f"Não foi possível inferir parent_columns da FK em `{child_table}`. "
            f"A tabela pai `{parent_table}` não possui `pk_cols` em specs_config."
        )

    return _as_tuple(
        parent_raw["pk_cols"],
        field_name="pk_cols",
        table_name=parent_table,
        required=True,
    )


def _build_foreign_keys_from_config(
    table_name: str,
    raw_fks: Any,
    specs_config: SpecConfig,
) -> Tuple[ForeignKeySpec, ...]:
    """
    Cria ForeignKeySpec a partir de configuração declarativa.

    Formato aceito:
        foreign_keys=[
            {
                "columns": ["COLUNA_FK"],
                "parent_table": "tabela_pai",
                "parent_columns": ["COLUNA_PK_PAI"]
            }
        ]

    Se parent_columns for omitido, usa pk_cols da tabela pai.
    """
    if raw_fks is None:
        return tuple()

    if isinstance(raw_fks, ABCMapping):
        raw_fks = [raw_fks]

    if not isinstance(raw_fks, (list, tuple)):
        raise TypeError(
            f"`foreign_keys` da tabela `{table_name}` deve ser lista/tupla de dicts."
        )

    fks: List[ForeignKeySpec] = []

    for i, raw_fk in enumerate(raw_fks):
        if isinstance(raw_fk, ForeignKeySpec):
            fks.append(raw_fk)
            continue

        if not isinstance(raw_fk, ABCMapping):
            raise TypeError(
                f"FK #{i} da tabela `{table_name}` deve ser dict ou ForeignKeySpec."
            )

        parent_table = str(raw_fk.get("parent_table", "")).strip()

        if not parent_table:
            raise ValueError(
                f"FK #{i} da tabela `{table_name}` precisa informar `parent_table`."
            )

        columns = _as_tuple(
            raw_fk.get("columns"),
            field_name="foreign_keys.columns",
            table_name=table_name,
            required=True,
        )

        parent_columns_raw = raw_fk.get("parent_columns")

        if parent_columns_raw is None:
            parent_columns = _get_parent_pk_from_config(
                specs_config,
                parent_table=parent_table,
                child_table=table_name,
            )
        else:
            parent_columns = _as_tuple(
                parent_columns_raw,
                field_name="foreign_keys.parent_columns",
                table_name=table_name,
                required=True,
            )

        fks.append(
            ForeignKeySpec(
                columns=columns,
                parent_table=parent_table,
                parent_columns=parent_columns,
            )
        )

    return tuple(fks)


def build_specs_from_config(
    specs_config: SpecConfig,
    *,
    postprocessors: Optional[Mapping[str, PostProcessor]] = None,
    default_static: bool = False,
) -> Dict[str, TableSpec]:
    """
    Monta specs de qualquer conjunto de tabelas, sem hardcode.

    Exemplo esperado:
        specs_config = {
            "nome_da_tabela": {
                "pk_cols": ["id"],
                "static": False,
                "foreign_keys": [
                    {
                        "columns": ["fk_id"],
                        "parent_table": "tabela_pai",
                        "parent_columns": ["id"]
                    }
                ]
            }
        }
    """
    if not specs_config:
        raise ValueError("`specs_config` está vazio.")

    postprocessors = postprocessors or {}

    specs: Dict[str, TableSpec] = {}

    for table_name, raw_spec in specs_config.items():
        if not isinstance(raw_spec, ABCMapping):
            raise TypeError(
                f"Configuração da tabela `{table_name}` deve ser um dict."
            )

        pk_cols = _as_tuple(
            raw_spec.get("pk_cols"),
            field_name="pk_cols",
            table_name=table_name,
            required=True,
        )

        foreign_keys = _build_foreign_keys_from_config(
            table_name=table_name,
            raw_fks=raw_spec.get("foreign_keys"),
            specs_config=specs_config,
        )

        static = _as_bool(
            raw_spec.get("static"),
            default=default_static,
        )

        specs[table_name] = TableSpec(
            name=table_name,
            pk_cols=pk_cols,
            foreign_keys=foreign_keys,
            static=static,
            postprocess=postprocessors.get(table_name),
        )

    return specs


def read_table_generic(
    spark: SparkSession,
    path: str,
    *,
    file_format: str = "parquet",
    options: Optional[Mapping[str, Any]] = None,
) -> DataFrame:
    """
    Lê uma tabela sem assumir nome, schema ou formato fixo.

    Formatos comuns:
        parquet
        csv
        delta
        json
        orc
    """
    fmt = file_format.strip().lower()
    read_options = dict(options or {})

    if fmt == "csv":
        read_options.setdefault("header", True)
        read_options.setdefault("inferSchema", True)

    reader = spark.read

    for key, value in read_options.items():
        reader = reader.option(key, value)

    return reader.format(fmt).load(path)


def read_tables_from_paths(
    spark: SparkSession,
    table_paths: TablePaths,
    *,
    default_format: str = "parquet",
    table_formats: Optional[Mapping[str, str]] = None,
    default_options: Optional[Mapping[str, Any]] = None,
    table_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, DataFrame]:
    """
    Lê N tabelas a partir de um dict nome_tabela -> path.
    """
    if not table_paths:
        raise ValueError("`table_paths` está vazio.")

    table_formats = table_formats or {}
    table_options = table_options or {}
    default_options = default_options or {}

    tables: Dict[str, DataFrame] = {}

    for table_name, path in table_paths.items():
        fmt = table_formats.get(table_name, default_format)

        options = dict(default_options)
        options.update(table_options.get(table_name, {}))

        tables[table_name] = read_table_generic(
            spark,
            path,
            file_format=fmt,
            options=options,
        )

    return tables


def build_n_rows_by_table_generic(
    tables: Mapping[str, DataFrame],
    *,
    fixed_n_rows_by_table: Optional[Mapping[str, int]] = None,
    scale_by_table: Optional[Mapping[str, float]] = None,
    default_scale: float = 1.0,
) -> Optional[Dict[str, int]]:
    """
    Monta n_rows_by_table sem hardcode.

    Regras:
        - Se fixed_n_rows_by_table tiver valor para a tabela, usa esse valor.
        - Caso contrário, usa count(original) * escala.
        - Se não houver fixed nem escala diferente de 1.0, retorna None,
          deixando synthesize_multitable_spark usar o volume original.
    """
    fixed_n_rows_by_table = dict(fixed_n_rows_by_table or {})
    scale_by_table = dict(scale_by_table or {})

    if not fixed_n_rows_by_table and not scale_by_table and float(default_scale) == 1.0:
        return None

    n_rows: Dict[str, int] = {}

    for table_name, df in tables.items():
        if table_name in fixed_n_rows_by_table:
            target = int(fixed_n_rows_by_table[table_name])
        else:
            scale = float(scale_by_table.get(table_name, default_scale))
            source_count = df.count()
            target = int(round(source_count * scale))

        if target < 0:
            raise ValueError(
                f"n_rows calculado para `{table_name}` ficou negativo: {target}."
            )

        n_rows[table_name] = target

    return n_rows


def save_tables_generic(
    tables: Mapping[str, DataFrame],
    base_path: str,
    *,
    output_format: str = "parquet",
    mode: str = "overwrite",
    coalesce: Optional[int] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Salva qualquer quantidade de tabelas sem nomes hardcoded.
    """
    fmt = output_format.strip().lower()
    write_options = dict(options or {})

    if fmt == "csv":
        write_options.setdefault("header", True)

    for table_name, df in tables.items():
        out_df = df.coalesce(coalesce) if coalesce is not None else df

        writer = out_df.write.mode(mode)

        for key, value in write_options.items():
            writer = writer.option(key, value)

        writer.format(fmt).save(f"{base_path}/{table_name}")


def run_synthesis_from_tables(
    tables: Mapping[str, DataFrame],
    specs_config: SpecConfig,
    *,
    postprocessors: Optional[Mapping[str, PostProcessor]] = None,
    n_rows_by_table: Optional[Mapping[str, int]] = None,
    scale_by_table: Optional[Mapping[str, float]] = None,
    default_scale: float = 1.0,
    seed: int = 42,
    append_after_max_pk: bool = True,
    validate_mode: ValidateMode = "full",
    nullable_fk_policy: NullableFkPolicy = "allow_any_null",
    broadcast_fk_counts: bool = False,
    storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    verbose: bool = False,
    show_diagnostics: bool = True,
) -> Dict[str, DataFrame]:
    """
    Executa o gerador para qualquer conjunto de DataFrames já carregados.
    """
    specs = build_specs_from_config(
        specs_config,
        postprocessors=postprocessors,
    )

    effective_n_rows = build_n_rows_by_table_generic(
        tables,
        fixed_n_rows_by_table=n_rows_by_table,
        scale_by_table=scale_by_table,
        default_scale=default_scale,
    )

    synthetic = synthesize_multitable_spark(
        tables=tables,
        specs=specs,
        n_rows_by_table=effective_n_rows,
        seed=seed,
        append_after_max_pk=append_after_max_pk,
        validate_mode=validate_mode,
        nullable_fk_policy=nullable_fk_policy,
        broadcast_fk_counts=broadcast_fk_counts,
        storage_level=storage_level,
        verbose=verbose,
    )

    if show_diagnostics:
        print("\n>>> Diagnóstico PRIMARY KEYS")
        validate_primary_keys(synthetic, specs).show(truncate=False)

        print("\n>>> Diagnóstico FOREIGN KEYS")
        validate_foreign_keys(
            synthetic,
            specs,
            nullable_fk_policy=nullable_fk_policy,
        ).show(truncate=False)

    return synthetic


def run_synthesis_from_paths(
    spark: SparkSession,
    table_paths: TablePaths,
    specs_config: SpecConfig,
    *,
    default_input_format: str = "parquet",
    table_formats: Optional[Mapping[str, str]] = None,
    default_read_options: Optional[Mapping[str, Any]] = None,
    table_read_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
    postprocessors: Optional[Mapping[str, PostProcessor]] = None,
    n_rows_by_table: Optional[Mapping[str, int]] = None,
    scale_by_table: Optional[Mapping[str, float]] = None,
    default_scale: float = 1.0,
    seed: int = 42,
    append_after_max_pk: bool = True,
    validate_mode: ValidateMode = "full",
    nullable_fk_policy: NullableFkPolicy = "allow_any_null",
    broadcast_fk_counts: bool = False,
    storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    verbose: bool = False,
    show_diagnostics: bool = True,
    save_path: Optional[str] = None,
    save_format: str = "parquet",
    save_mode: str = "overwrite",
    save_coalesce: Optional[int] = None,
) -> Dict[str, DataFrame]:
    """
    Lê, sintetiza, valida e opcionalmente salva N tabelas.
    """
    tables = read_tables_from_paths(
        spark,
        table_paths,
        default_format=default_input_format,
        table_formats=table_formats,
        default_options=default_read_options,
        table_options=table_read_options,
    )

    synthetic = run_synthesis_from_tables(
        tables=tables,
        specs_config=specs_config,
        postprocessors=postprocessors,
        n_rows_by_table=n_rows_by_table,
        scale_by_table=scale_by_table,
        default_scale=default_scale,
        seed=seed,
        append_after_max_pk=append_after_max_pk,
        validate_mode=validate_mode,
        nullable_fk_policy=nullable_fk_policy,
        broadcast_fk_counts=broadcast_fk_counts,
        storage_level=storage_level,
        verbose=verbose,
        show_diagnostics=show_diagnostics,
    )

    if save_path is not None:
        save_tables_generic(
            synthetic,
            save_path,
            output_format=save_format,
            mode=save_mode,
            coalesce=save_coalesce,
        )

    return synthetic


__all__ = [
    "ForeignKeySpec",
    "TableSpec",
    "NullableFkPolicy",
    "ValidateMode",
    "SpecConfig",
    "TablePaths",
    "build_specs_from_config",
    "read_table_generic",
    "read_tables_from_paths",
    "build_n_rows_by_table_generic",
    "save_tables_generic",
    "validate_primary_keys",
    "validate_foreign_keys",
    "run_validation_or_raise",
    "synthesize_multitable_spark",
    "run_synthesis_from_tables",
    "run_synthesis_from_paths",
]