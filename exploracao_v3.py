"""
synthetic_multitable_spark_v3.py
================================

Gerador de dados sintéticos MULTI-TABELA em PySpark.

Engine (synthesize_multitable_spark): inalterado e validado.

NOVIDADE v3 — runner robusto a partir de caminhos + dicionário de specs:
    run_synthesis_from_paths(spark, table_paths, specs_config, ...)

    Corrige o problema de "relacionamento não encontrado":
      - converte fielmente cada FK declarada em specs_config -> ForeignKeySpec;
      - mantém `tables` e `specs` indexados pelas MESMAS chaves;
      - faz pré-checagem clara: aponta exatamente qual parent_table/coluna
        não casa, em vez de uma falha genérica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Literal
from collections.abc import Mapping as ABCMapping
import warnings
import zlib

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


NullableFkPolicy = Literal["allow_any_null", "allow_all_null", "invalid_null"]
ValidateMode = Literal["none", "full"]


@dataclass(frozen=True)
class ForeignKeySpec:
    columns: Tuple[str, ...]
    parent_table: str
    parent_columns: Tuple[str, ...]


PostProcessor = Callable[[DataFrame, Mapping[str, DataFrame]], DataFrame]


@dataclass(frozen=True)
class TableSpec:
    name: str
    pk_cols: Tuple[str, ...]
    foreign_keys: Tuple[ForeignKeySpec, ...] = field(default_factory=tuple)
    static: bool = False
    postprocess: Optional[PostProcessor] = None


def _stable_seed(base_seed: int, *parts: object) -> int:
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


def _validate_specs(tables, specs) -> None:
    if not specs:
        raise ValueError("`specs` está vazio.")
    for name, spec in specs.items():
        if name not in tables:
            raise ValueError(f"Tabela `{name}` está em specs, mas não está em tables.")
        if spec.name != name:
            raise ValueError(f"Inconsistência: chave specs=`{name}`, mas TableSpec.name=`{spec.name}`.")
        if not spec.pk_cols:
            raise ValueError(f"Tabela `{name}` precisa ter pelo menos uma coluna de PK.")
        df_cols = set(tables[name].columns)
        for pk in spec.pk_cols:
            if pk not in df_cols:
                raise ValueError(f"PK col `{pk}` não existe na tabela `{name}`.")
        seen_fk_cols: set = set()
        for fk in spec.foreign_keys:
            if not fk.columns:
                raise ValueError(f"FK vazia declarada na tabela `{name}`.")
            if len(fk.columns) != len(fk.parent_columns):
                raise ValueError(f"FK inválida em `{name}`: tamanhos diferentes.")
            if fk.parent_table == name:
                raise ValueError(f"Self-reference não suportado: `{name}`.")
            if fk.parent_table not in specs:
                raise ValueError(f"FK em `{name}` referencia `{fk.parent_table}` ausente em specs.")
            if fk.parent_table not in tables:
                raise ValueError(f"FK em `{name}` referencia `{fk.parent_table}` ausente em tables.")
            for c in fk.columns:
                if c not in df_cols:
                    raise ValueError(f"FK col `{c}` não existe na filha `{name}`.")
                if c in seen_fk_cols:
                    raise ValueError(f"Coluna `{c}` em `{name}` participa de mais de uma FK.")
                seen_fk_cols.add(c)
            parent_cols = set(tables[fk.parent_table].columns)
            for pc in fk.parent_columns:
                if pc not in parent_cols:
                    raise ValueError(f"FK em `{name}` referencia `{pc}` ausente no pai `{fk.parent_table}`.")


def _topological_order(specs) -> List[str]:
    remaining = set(specs.keys())
    done: set = set()
    order: List[str] = []
    while remaining:
        ready = [n for n in remaining
                 if {fk.parent_table for fk in specs[n].foreign_keys}.issubset(done)]
        if not ready:
            unresolved = {t: [fk.parent_table for fk in specs[t].foreign_keys] for t in remaining}
            raise ValueError(f"Ciclo/self-ref/pai ausente. Pendências: {unresolved}")
        for name in sorted(ready):
            order.append(name); done.add(name); remaining.remove(name)
    return order


def _referenced_parent_columns(specs) -> Dict[str, set]:
    refs: Dict[str, set] = {}
    for child_spec in specs.values():
        for fk in child_spec.foreign_keys:
            refs.setdefault(fk.parent_table, set()).add(tuple(fk.parent_columns))
    return refs


def _with_contiguous_row_id(df: DataFrame, id_col: str) -> DataFrame:
    spark = df.sparkSession
    schema_with_id = T.StructType(df.schema.fields + [T.StructField(id_col, T.LongType(), False)])
    indexed_rdd = df.rdd.zipWithIndex().map(lambda r: tuple(r[0]) + (r[1],))
    return spark.createDataFrame(indexed_rdd, schema=schema_with_id)


def _bootstrap_rows_exact(src_indexed, n_rows, *, src_count, seed, spark, keep_all_source_rows):
    if n_rows < 0:
        raise ValueError("n_rows deve ser >= 0.")
    src_cols = [c for c in src_indexed.columns if c != "__src_row_id"]
    if n_rows == 0:
        empty_schema = T.StructType(
            [T.StructField("__synthetic_pos", T.LongType(), False),
             T.StructField("__orig_src_row_id", T.LongType(), True)]
            + [f for f in src_indexed.schema.fields if f.name in src_cols])
        return spark.createDataFrame([], schema=empty_schema)
    if src_count == 0:
        raise ValueError("Fonte vazia mas n_rows > 0.")
    if keep_all_source_rows:
        if n_rows < src_count:
            raise ValueError(f"Pai precisa n_rows >= src_count. n_rows={n_rows}, src_count={src_count}.")
        base_keep = (src_indexed
                     .withColumn("__synthetic_pos", F.col("__src_row_id"))
                     .withColumn("__orig_src_row_id", F.col("__src_row_id"))
                     .select("__synthetic_pos", "__orig_src_row_id", *src_cols))
        extra_n = n_rows - src_count
        if extra_n == 0:
            return base_keep
        extra_positions = (spark.range(src_count, n_rows)
                           .withColumnRenamed("id", "__synthetic_pos")
                           .withColumn("__lookup_src_row_id",
                                       F.floor(F.rand(seed) * F.lit(src_count)).cast("long")))
        extra = (extra_positions
                 .join(src_indexed,
                       extra_positions["__lookup_src_row_id"] == src_indexed["__src_row_id"], "left")
                 .withColumn("__orig_src_row_id", F.col("__src_row_id"))
                 .select("__synthetic_pos", "__orig_src_row_id", *src_cols))
        return base_keep.unionByName(extra)
    positions = (spark.range(0, n_rows)
                 .withColumnRenamed("id", "__synthetic_pos")
                 .withColumn("__lookup_src_row_id",
                             F.floor(F.rand(seed) * F.lit(src_count)).cast("long")))
    return (positions
            .join(src_indexed, positions["__lookup_src_row_id"] == src_indexed["__src_row_id"], "left")
            .withColumn("__orig_src_row_id", F.col("__src_row_id"))
            .select("__synthetic_pos", "__orig_src_row_id", *src_cols))


_INT_TYPE_LIMITS = ((T.ByteType, 127), (T.ShortType, 32_767), (T.IntegerType, 2_147_483_647))


def _max_pk_value(df_cached, pk):
    row = df_cached.agg(F.max(F.col(pk)).alias("max_pk")).collect()[0]
    return int(row["max_pk"]) if row["max_pk"] is not None else None


def _set_unique_pk_column(work, source_cached, pk, *, append_after_max, target_n, offset=0):
    dt = _get_field_type(source_cached, pk)
    if _is_integer_type(dt):
        start = (_max_pk_value(source_cached, pk) or 0) + 1 if append_after_max else 1
        highest = start + target_n - 1 + offset
        for type_cls, limit in _INT_TYPE_LIMITS:
            if isinstance(dt, type_cls) and highest > limit:
                raise OverflowError(f"PK `{pk}` {type_cls.__name__} estoura limite {limit:,} (max {highest:,}).")
        return work.withColumn(pk, (F.col("__synthetic_pos") + F.lit(start + offset)).cast(dt))
    if _is_string_type(dt):
        return work.withColumn(pk, F.concat(F.lit(f"SYN_{pk}_"),
                               F.lpad((F.col("__synthetic_pos") + F.lit(offset)).cast("string"), 14, "0")).cast(dt))
    raise TypeError(f"PK `{pk}` tipo {dt!r} sem estratégia segura.")


def _generate_pk_columns(work, source_cached, spec, *, append_after_max, target_n):
    if len(spec.pk_cols) == 1:
        return _set_unique_pk_column(work, source_cached, spec.pk_cols[0],
                                     append_after_max=append_after_max, target_n=target_n, offset=0)
    last_pk = spec.pk_cols[-1]
    last_type = _get_field_type(source_cached, last_pk)
    if not _is_safe_pk_type(last_type):
        raise TypeError(f"PK composta `{spec.name}` última col `{last_pk}` tipo {last_type!r} inseguro.")
    return _set_unique_pk_column(work, source_cached, last_pk,
                                 append_after_max=append_after_max, target_n=target_n, offset=0)


def _build_mapping_for_parent_cols(work_cached, parent_cols, storage_level):
    old_cols = [f"__old__{c}" for c in parent_cols]
    missing_old = [c for c in old_cols if c not in work_cached.columns]
    if missing_old:
        raise ValueError(f"Mapping: colunas antigas ausentes: {missing_old}")
    mapping = work_cached.select(
        *[F.col(old_cols[i]).alias(f"__old_{i}") for i in range(len(parent_cols))],
        *[F.col(parent_cols[i]).alias(f"__new_{i}") for i in range(len(parent_cols))],
        F.col("__synthetic_pos"))
    partition_cols = [F.col(f"__old_{i}") for i in range(len(parent_cols))]
    w = Window.partitionBy(*partition_cols).orderBy(F.col("__synthetic_pos"))
    mapping = mapping.withColumn("__candidate_rank", F.row_number().over(w).cast("long"))
    counts = mapping.groupBy(*[F.col(f"__old_{i}") for i in range(len(parent_cols))]).agg(
        F.count(F.lit(1)).cast("long").alias("__candidate_count"))
    mapping = mapping.join(counts, on=[f"__old_{i}" for i in range(len(parent_cols))], how="left")
    return _persist(mapping, storage_level)


def _fk_join_condition(left_df, left_cols, right_df, right_cols):
    conditions = [left_df[left_cols[i]].eqNullSafe(right_df[right_cols[i]]) for i in range(len(left_cols))]
    return reduce(lambda a, b: a & b, conditions)


def _apply_fk_mapping(work, fk, mapping, *, seed, broadcast_fk_counts, fk_index=0):
    fk_tag = f"__fk{fk_index}_{fk.parent_table}_{_stable_seed(seed, fk.parent_table, fk.columns, fk.parent_columns)}"
    n = len(fk.columns)
    counts = mapping.select(
        *[F.col(f"__old_{i}").alias(f"{fk_tag}_old_{i}") for i in range(n)],
        F.col("__candidate_count").alias(f"{fk_tag}_count"),
    ).dropDuplicates([f"{fk_tag}_old_{i}" for i in range(n)])
    count_old_cols = [f"{fk_tag}_old_{i}" for i in range(n)]
    cond_counts = _fk_join_condition(work, list(fk.columns), counts, count_old_cols)
    if broadcast_fk_counts:
        work = work.join(F.broadcast(counts), cond_counts, "left")
    else:
        work = work.join(counts, cond_counts, "left")
    work = work.withColumn(f"{fk_tag}_rank",
        F.when(F.col(f"{fk_tag}_count").isNull(), F.lit(None).cast("long")).otherwise(
            F.floor(F.rand(_stable_seed(seed, fk_tag, "rank")) * F.col(f"{fk_tag}_count")).cast("long") + F.lit(1)))
    m = mapping.select(
        *[F.col(f"__old_{i}").alias(f"{fk_tag}_map_old_{i}") for i in range(n)],
        *[F.col(f"__new_{i}").alias(f"{fk_tag}_new_{i}") for i in range(n)],
        F.col("__candidate_rank").alias(f"{fk_tag}_map_rank"))
    map_old_cols = [f"{fk_tag}_map_old_{i}" for i in range(n)]
    cond_map_key = _fk_join_condition(work, list(fk.columns), m, map_old_cols)
    cond_map = cond_map_key & (work[f"{fk_tag}_rank"] == m[f"{fk_tag}_map_rank"])
    work = work.join(m, cond_map, "left")
    for i, child_col in enumerate(fk.columns):
        child_type = _get_field_type(work, child_col)
        work = work.withColumn(child_col, F.col(f"{fk_tag}_new_{i}").cast(child_type))
    drop_cols = ([f"{fk_tag}_old_{i}" for i in range(n)]
                 + [f"{fk_tag}_map_old_{i}" for i in range(n)]
                 + [f"{fk_tag}_new_{i}" for i in range(n)]
                 + [f"{fk_tag}_count", f"{fk_tag}_rank", f"{fk_tag}_map_rank"])
    return work.drop(*drop_cols)


def validate_primary_keys(tables, specs):
    spark = next(iter(tables.values())).sparkSession
    rows = []
    for name, spec in specs.items():
        df = tables[name]
        total_rows = df.count()
        distinct_pk = df.select(*spec.pk_cols).dropDuplicates().count()
        null_condition = reduce(lambda a, b: a | b, [F.col(c).isNull() for c in spec.pk_cols])
        null_pk_rows = df.where(null_condition).count()
        rows.append((name, ",".join(spec.pk_cols), int(total_rows), int(distinct_pk),
                     int(null_pk_rows), int(total_rows - distinct_pk)))
    schema = ("table string, pk_cols string, total_rows long, distinct_pk long, "
              "null_pk_rows long, duplicate_pk_rows long")
    return spark.createDataFrame(rows, schema=schema)


def _filter_child_fk_for_validation(child_df, fk, nullable_fk_policy):
    if nullable_fk_policy == "invalid_null":
        return child_df
    any_null = reduce(lambda a, b: a | b, [F.col(c).isNull() for c in fk.columns])
    all_null = reduce(lambda a, b: a & b, [F.col(c).isNull() for c in fk.columns])
    if nullable_fk_policy == "allow_any_null":
        return child_df.where(~any_null)
    if nullable_fk_policy == "allow_all_null":
        return child_df.where(~all_null)
    raise ValueError(f"nullable_fk_policy inválida: {nullable_fk_policy}")


def validate_foreign_keys(tables, specs, *, nullable_fk_policy="allow_any_null"):
    spark = next(iter(tables.values())).sparkSession
    rows = []
    for child_name, child_spec in specs.items():
        child_df_raw = tables[child_name]
        for fk in child_spec.foreign_keys:
            parent_df = tables[fk.parent_table]
            child_df = _filter_child_fk_for_validation(child_df_raw, fk, nullable_fk_policy)
            child_keys = child_df.select(*fk.columns).dropDuplicates()
            parent_keys = parent_df.select(
                *[F.col(parent_col).alias(child_col)
                  for child_col, parent_col in zip(fk.columns, fk.parent_columns)]).dropDuplicates()
            invalid = child_keys.join(parent_keys, on=list(fk.columns), how="left_anti").count()
            total_distinct = child_keys.count()
            rows.append((child_name, ",".join(fk.columns), fk.parent_table,
                         ",".join(fk.parent_columns), int(total_distinct), int(invalid)))
    schema = ("child_table string, fk_cols string, parent_table string, parent_cols string, "
              "distinct_child_fk long, invalid_fk long")
    return spark.createDataFrame(rows, schema=schema)


def _run_validation_or_raise(result, specs, *, nullable_fk_policy):
    pk_report = validate_primary_keys(result, specs)
    fk_report = validate_foreign_keys(result, specs, nullable_fk_policy=nullable_fk_policy)
    pk_problems = pk_report.where("null_pk_rows > 0 OR duplicate_pk_rows > 0").count()
    fk_problems = fk_report.where("invalid_fk > 0").count()
    if pk_problems or fk_problems:
        print(">>> FALHA NA VALIDAÇÃO"); pk_report.show(truncate=False); fk_report.show(truncate=False)
        raise RuntimeError(f"Validação falhou: {pk_problems} PK, {fk_problems} FK.")


def synthesize_multitable_spark(tables, specs, n_rows_by_table=None, *, seed=42,
                                append_after_max_pk=True, validate_mode="full",
                                nullable_fk_policy="allow_any_null", broadcast_fk_counts=False,
                                storage_level=StorageLevel.MEMORY_AND_DISK, verbose=False):
    if validate_mode not in ("none", "full"):
        raise ValueError("validate_mode deve ser 'none' ou 'full'.")
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
            ref_cols = sorted(set(c for cols in ref_col_sets for c in cols) | set(spec.pk_cols))
            if spec.static:
                src_count = source.count()
                if target_n_raw is not None and int(target_n_raw) != src_count:
                    warnings.warn(f"`{table_name}` static; n_rows ignorado.")
                if verbose:
                    print(f"[{table_name}] STATIC | {src_count} linhas")
                work = (_with_contiguous_row_id(source, "__synthetic_pos")
                        .withColumn("__orig_src_row_id", F.col("__synthetic_pos")))
                for c in ref_cols:
                    work = work.withColumn(f"__old__{c}", F.col(c))
                work = _persist(work, storage_level); work.count(); intermediates.append(work)
            else:
                src_indexed = _with_contiguous_row_id(source, "__src_row_id")
                src_indexed = _persist(src_indexed, storage_level)
                src_count = src_indexed.count(); intermediates.append(src_indexed)
                target_n = int(target_n_raw if target_n_raw is not None else src_count)
                keep_all = table_name in parent_refs
                if verbose:
                    print(f"[{table_name}] {'PAI' if keep_all else 'FILHO'} | {src_count}->{target_n}")
                work = _bootstrap_rows_exact(src_indexed, target_n, src_count=src_count,
                    seed=_stable_seed(seed, table_name, "bootstrap"), spark=spark, keep_all_source_rows=keep_all)
                for c in ref_cols:
                    work = work.withColumn(f"__old__{c}", F.col(c))
                work = _generate_pk_columns(work, src_indexed, spec,
                    append_after_max=append_after_max_pk, target_n=target_n)
                for fk in spec.foreign_keys:
                    key = (fk.parent_table, tuple(fk.parent_columns))
                    if key not in mappings:
                        raise ValueError(f"Mapping ausente para {table_name}.{fk.columns}->{key}")
                    work = _apply_fk_mapping(work, fk, mappings[key],
                        seed=_stable_seed(seed, table_name, fk.parent_table, fk.columns, fk.parent_columns),
                        broadcast_fk_counts=broadcast_fk_counts)
                if spec.postprocess is not None:
                    work = spec.postprocess(work, result)
                work = _persist(work, storage_level); work.count(); intermediates.append(work)
            if table_name in parent_refs:
                for cols in parent_refs[table_name]:
                    mapping_df = _build_mapping_for_parent_cols(work, tuple(cols), storage_level=storage_level)
                    mapping_df.count()
                    mappings[(table_name, tuple(cols))] = mapping_df
                    intermediates.append(mapping_df)
            synth = work.select(*original_cols)
            synth = _persist(synth, storage_level); synth.count()
            result[table_name] = synth
        if validate_mode == "full":
            if verbose:
                print("Validando...")
            _run_validation_or_raise(result, specs, nullable_fk_policy=nullable_fk_policy)
            if verbose:
                print("Validação OK.")
        return result
    except Exception:
        for df in result.values():
            _safe_unpersist(df)
        raise
    finally:
        for df in intermediates:
            _safe_unpersist(df)


# ============================================================
# 9. Postprocess de exemplo (base CDB) — opcional
# ============================================================

def postprocess_instrumento(work: DataFrame, generated: Mapping[str, DataFrame]) -> DataFrame:
    """
    Recompõe COD_IF e COD_ISIN da tabela instrumento a partir do novo NUM_IF.
    Use via postprocess_by_table={"instrumento": postprocess_instrumento}.
    """
    if "tipo_if" not in generated:
        raise ValueError("postprocess_instrumento requer a tabela `tipo_if` já gerada.")
    tipo_if = generated["tipo_if"]
    if not {"NUM_TIPO_IF", "COD_TIPO_IF"}.issubset(set(tipo_if.columns)):
        raise ValueError("tipo_if precisa conter NUM_TIPO_IF e COD_TIPO_IF.")
    if "NUM_TIPO_IF" not in work.columns:
        raise ValueError("instrumento não contém NUM_TIPO_IF.")

    tipo = tipo_if.select("NUM_TIPO_IF", "COD_TIPO_IF").dropDuplicates(["NUM_TIPO_IF"])
    if "COD_TIPO_IF" in work.columns:
        work = work.drop("COD_TIPO_IF")
    work = work.join(F.broadcast(tipo), on="NUM_TIPO_IF", how="left")

    if "COD_IF" in work.columns:
        work = work.withColumn(
            "COD_IF",
            F.concat(F.coalesce(F.col("COD_TIPO_IF"), F.lit("UNK")),
                     F.lpad(F.col("NUM_IF").cast("string"), 6, "0")),
        )
    if "COD_ISIN" in work.columns:
        work = work.withColumn(
            "COD_ISIN",
            F.concat(F.lit("BR"), F.lpad(F.col("NUM_IF").cast("string"), 10, "0")),
        )
    return work.drop("COD_TIPO_IF")


# ============================================================
# 10. Construção de specs a partir de dicionário (FIX do bug)
# ============================================================

def _build_specs_from_config(
    specs_config: Mapping[str, Mapping],
    postprocess_by_table: Optional[Mapping[str, PostProcessor]] = None,
) -> Dict[str, TableSpec]:
    """
    Converte o dicionário declarativo em specs tipados.

    Aceita, por tabela:
        {
            "pk_cols": [...],                      # obrigatório
            "static": bool,                        # opcional (default False)
            "foreign_keys": [                      # opcional
                {"columns": [...],
                 "parent_table": "...",
                 "parent_columns": [...]},
                ...
            ],
        }

    Este passo é onde o relacionamento costumava "sumir": se as FKs não
    forem convertidas em ForeignKeySpec (com tuplas), o motor não enxerga
    o vínculo. Aqui a conversão é explícita e validada.
    """
    if not isinstance(specs_config, ABCMapping) or not specs_config:
        raise ValueError("`specs_config` deve ser um dicionário não vazio.")

    postprocess_by_table = dict(postprocess_by_table or {})
    specs: Dict[str, TableSpec] = {}

    for name, cfg in specs_config.items():
        if not isinstance(cfg, ABCMapping):
            raise TypeError(f"Config da tabela `{name}` deve ser um dict, recebido {type(cfg)!r}.")

        pk_cols = cfg.get("pk_cols")
        if not pk_cols:
            raise ValueError(f"Tabela `{name}`: `pk_cols` é obrigatório e não pode ser vazio.")
        if isinstance(pk_cols, str):
            pk_cols = [pk_cols]

        raw_fks = cfg.get("foreign_keys") or cfg.get("fks") or []
        if isinstance(raw_fks, ABCMapping):
            raw_fks = [raw_fks]

        fks: List[ForeignKeySpec] = []
        for i, fk in enumerate(raw_fks):
            if not isinstance(fk, ABCMapping):
                raise ValueError(
                    f"Tabela `{name}`: FK #{i} deve ser um dict com "
                    f"`columns`, `parent_table`, `parent_columns`. Recebido: {fk!r}"
                )
            try:
                cols = fk["columns"]
                parent_table = fk["parent_table"]
                parent_cols = fk["parent_columns"]
            except KeyError as e:
                raise ValueError(
                    f"Tabela `{name}`: FK #{i} está faltando a chave {e}. "
                    f"Use exatamente `columns`, `parent_table`, `parent_columns`."
                ) from None

            if isinstance(cols, str):
                cols = [cols]
            if isinstance(parent_cols, str):
                parent_cols = [parent_cols]
            parent_table = str(parent_table).strip()

            fks.append(ForeignKeySpec(
                columns=tuple(cols),
                parent_table=parent_table,
                parent_columns=tuple(parent_cols),
            ))

        specs[name] = TableSpec(
            name=name,
            pk_cols=tuple(pk_cols),
            foreign_keys=tuple(fks),
            static=bool(cfg.get("static", False)),
            postprocess=postprocess_by_table.get(name),
        )

    return specs


def _preflight_relationships(
    table_paths: Mapping[str, str],
    specs: Mapping[str, TableSpec],
) -> None:
    """
    Checagem amigável ANTES de ler/processar: garante que todo relacionamento
    declarado aponta para uma tabela pai conhecida e que há caminho para ela.
    Gera mensagem que diz EXATAMENTE qual vínculo está quebrado.
    """
    faltam_caminho = [n for n in specs if n not in table_paths]
    if faltam_caminho:
        raise ValueError(
            "Faltam caminhos em `table_paths` para tabelas declaradas em "
            f"specs_config: {faltam_caminho}."
        )

    sobra_caminho = [n for n in table_paths if n not in specs]
    if sobra_caminho:
        warnings.warn(
            f"`table_paths` contém tabelas sem spec (serão ignoradas): {sobra_caminho}."
        )

    conhecidas = sorted(specs.keys())
    problemas: List[str] = []
    for name, spec in specs.items():
        for fk in spec.foreign_keys:
            if fk.parent_table not in specs:
                problemas.append(
                    f"  - `{name}` --{list(fk.columns)}--> `{fk.parent_table}`: "
                    f"a tabela pai `{fk.parent_table}` NÃO está em specs_config."
                )

    if problemas:
        raise ValueError(
            "Relacionamento(s) apontando para tabela pai inexistente:\n"
            + "\n".join(problemas)
            + f"\n\nTabelas declaradas: {conhecidas}\n"
            + "Confira se `parent_table` casa EXATAMENTE (inclusive maiúsc./minúsc.) "
            + "com a chave usada em specs_config / table_paths."
        )


def _validate_relationship_columns(
    tables: Mapping[str, DataFrame],
    specs: Mapping[str, TableSpec],
) -> None:
    """
    Confirma que PK/FK declaradas existem de fato no schema lido.
    Causa comum de 'relacionamento não encontrado': nome de coluna diferente
    do que está no Parquet (acento, caixa, espaço).
    """
    problemas: List[str] = []
    for name, spec in specs.items():
        child_cols = set(tables[name].columns)
        for pk in spec.pk_cols:
            if pk not in child_cols:
                problemas.append(
                    f"  - PK `{pk}` não existe em `{name}`. Colunas: {sorted(child_cols)}"
                )
        for fk in spec.foreign_keys:
            for c in fk.columns:
                if c not in child_cols:
                    problemas.append(
                        f"  - FK `{c}` (`{name}` -> `{fk.parent_table}`) não existe em `{name}`. "
                        f"Colunas: {sorted(child_cols)}"
                    )
            parent_cols = set(tables[fk.parent_table].columns)
            for pc in fk.parent_columns:
                if pc not in parent_cols:
                    problemas.append(
                        f"  - parent_col `{pc}` (`{name}` -> `{fk.parent_table}`) não existe "
                        f"no pai `{fk.parent_table}`. Colunas: {sorted(parent_cols)}"
                    )
            if len(fk.columns) != len(fk.parent_columns):
                problemas.append(
                    f"  - `{name}` -> `{fk.parent_table}`: nº de colunas difere "
                    f"{list(fk.columns)} vs {list(fk.parent_columns)}."
                )
    if problemas:
        raise ValueError(
            "Colunas de relacionamento não encontradas nos dados lidos:\n"
            + "\n".join(problemas)
        )


# ============================================================
# 11. Leitura e runner principal a partir de caminhos
# ============================================================

def _read_table(
    spark: SparkSession,
    path: str,
    fmt: str,
    options: Optional[Mapping[str, object]] = None,
) -> DataFrame:
    options = dict(options or {})
    reader = spark.read
    for k, v in options.items():
        reader = reader.option(k, v)

    fmt = (fmt or "parquet").lower()
    if fmt == "parquet":
        return reader.parquet(path)
    if fmt == "orc":
        return reader.orc(path)
    if fmt == "csv":
        return (reader
                .option("header", options.get("header", True))
                .option("inferSchema", options.get("inferSchema", True))
                .csv(path))
    return reader.format(fmt).load(path)


def run_synthesis_from_paths(
    spark: SparkSession,
    table_paths: Mapping[str, str],
    specs_config: Mapping[str, Mapping],
    *,
    default_input_format: str = "parquet",
    input_options: Optional[Mapping[str, object]] = None,
    n_rows_by_table: Optional[Mapping[str, int]] = None,
    scale_factor: Optional[float] = None,
    seed: int = 42,
    append_after_max_pk: bool = True,
    validate_mode: ValidateMode = "full",
    nullable_fk_policy: NullableFkPolicy = "allow_any_null",
    broadcast_fk_counts: bool = False,
    storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    postprocess_by_table: Optional[Mapping[str, PostProcessor]] = None,
    save_path: Optional[str] = None,
    save_format: Literal["csv", "parquet"] = "parquet",
    verbose: bool = True,
) -> Dict[str, DataFrame]:
    """
    Lê tabelas a partir de `table_paths`, monta as specs a partir de
    `specs_config` (com FKs!) e sintetiza preservando os relacionamentos.

    Args principais:
        table_paths:        {nome_tabela: caminho}
        specs_config:       {nome_tabela: {"pk_cols": [...],
                                           "static": bool,
                                           "foreign_keys": [ {...} ]}}
        scale_factor:       multiplicador de volume para tabelas não estáticas
                            (ex.: 10 gera 10x). Ignorado se n_rows_by_table for dado.
        postprocess_by_table: {nome_tabela: funcao_postprocess} (opcional)

    Retorna: {nome_tabela: DataFrame sintético}.
    """
    # 1) dict -> specs tipadas (converte as FKs declaradas)
    specs = _build_specs_from_config(specs_config, postprocess_by_table)

    # 2) pré-checagem de relacionamentos (mensagens claras)
    _preflight_relationships(table_paths, specs)

    # 3) ordem topológica (detecta ciclos/pais ausentes cedo)
    order = _topological_order(specs)
    if verbose:
        print("Ordem topológica:", " -> ".join(order))

    # 4) leitura — MESMAS chaves de specs_config
    tables: Dict[str, DataFrame] = {
        name: _read_table(spark, table_paths[name], default_input_format, input_options)
        for name in specs
    }

    # 5) valida nomes de colunas de PK/FK contra o schema real
    _validate_relationship_columns(tables, specs)

    # 6) volume por tabela
    if n_rows_by_table is None:
        n_rows_by_table = {}
        for name in specs:
            base = tables[name].count()
            if specs[name].static:
                n_rows_by_table[name] = base
            elif scale_factor:
                n_rows_by_table[name] = int(round(base * scale_factor))
            else:
                n_rows_by_table[name] = base
    if verbose:
        print("n_rows_by_table:", dict(n_rows_by_table))

    # 7) sintetização (motor inalterado)
    synthetic = synthesize_multitable_spark(
        tables=tables,
        specs=specs,
        n_rows_by_table=n_rows_by_table,
        seed=seed,
        append_after_max_pk=append_after_max_pk,
        validate_mode=validate_mode,
        nullable_fk_policy=nullable_fk_policy,
        broadcast_fk_counts=broadcast_fk_counts,
        storage_level=storage_level,
        verbose=verbose,
    )

    # 8) gravação opcional
    if save_path:
        for name, df in synthetic.items():
            writer = df.write.mode("overwrite")
            fmt = (save_format or "parquet").lower()
            if fmt == "csv":
                writer.option("header", True).csv(f"{save_path}/{name}")
            elif fmt == "parquet":
                writer.parquet(f"{save_path}/{name}")
            else:
                writer.format(fmt).save(f"{save_path}/{name}")
        if verbose:
            print("Dados sintéticos salvos em:", save_path)

    return synthetic
