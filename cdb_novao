from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import sys
import time
import warnings
import zlib
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import reduce
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REQUIRED_ENV_VARS = (
    "DATAGEN_RAW_BASE_URI",
    "DATAGEN_SYNTHETIC_BASE_URI",
    "DATAGEN_SPECS_URI",
)
DEFAULT_SCALE_FACTOR = 2.0
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Filtro de domínio: CDB simplificado.
#
# Toda tabela de origem que possuir a coluna NUM_TIPO_IF é filtrada para
# NUM_TIPO_IF == 49 ANTES da síntese, garantindo que o modelo seja gerado
# usando apenas o CDB simplificado. Tabelas que NÃO possuem a coluna passam
# intactas. O filtro é aplicado por `_aplica_filtro_tipo_if` (logo após
# `read_parquet`) nos dois pontos de leitura da fonte de síntese:
#   1. referential_sample  (caminho --limit);
#   2. engorda             (caminho sem --limit).
#
# IMPORTANTE: a leitura de max(pk) em compute_pk_maxes NÃO usa este filtro
# de propósito — ela precisa do max real da tabela inteira para que as PKs
# sintéticas não colidam com linhas de produção de OUTROS NUM_TIPO_IF.
# ---------------------------------------------------------------------------
FILTRO_TIPO_IF_COLUMN = "NUM_TIPO_IF"
FILTRO_TIPO_IF_VALUE = 49 #filtro cdb simplificado

# ---------------------------------------------------------------------------
# Regras de engorda por coluna.
#
# Data de engorda = instante em que este script começa a executar. A mesma data
# é reutilizada para todas as tabelas do run, evitando pequenas diferenças de
# timestamp entre componentes ou ações Spark.
#
# Regras aplicadas quando a coluna existir na tabela:
#   DAT_INCLUSAO              -> data/hora da engorda (timestamp)
#   DAT_ALTERACAO             -> mesma data/hora de DAT_INCLUSAO (timestamp)
#   DT_EMISSAO                -> data da engorda, sem timestamp
#   DT_VENCIMENTO             -> data da engorda + prazo, sem timestamp
#
# NUM_ID_CERTIFICACAO_CETIP NÃO entra aqui: ela é PK e a geração de PK
# (compute_pk_maxes + _set_unique_pk_column) já a faz incremental acima do max
# real da tabela inteira, com a folga do --pk-safety-band. Tratá-la de novo
# desfazia a folga e relia o max sem necessidade.
#
# Para DT_VENCIMENTO, se não for informado um prazo fixo por tabela, o código
# preserva o prazo original da linha bootstrapada: DT_VENCIMENTO - DT_EMISSAO.
# Se esse prazo não existir, for inválido ou <= 0, usa 365 dias por segurança.
# ---------------------------------------------------------------------------
ENGORDA_COL_DAT_INCLUSAO = "DAT_INCLUSAO"
ENGORDA_COL_DAT_ALTERACAO = "DAT_ALTERACAO"
ENGORDA_COL_DT_VENCIMENTO = "DT_VENCIMENTO"
ENGORDA_COL_DT_EMISSAO = "DT_EMISSAO"
DEFAULT_DT_VENCIMENTO_PRAZO_DIAS = 30
MIN_DT_VENCIMENTO_PRAZO_DIAS = 1

NullableFkPolicy = Literal["allow_any_null", "allow_all_null", "invalid_null"]


ValidateMode = Literal["none", "full"]


RelationshipPolicy = Literal["warn_and_skip", "raise"]


SaveErrorPolicy = Literal["warn_and_continue", "raise"]


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


def _is_float_type(dt: T.DataType) -> bool:
    """FloatType/DoubleType. Comuns quando CSV é lido com inferSchema=True."""
    return isinstance(dt, (T.FloatType, T.DoubleType))


def _is_decimal_type(dt: T.DataType) -> bool:
    return isinstance(dt, T.DecimalType)


def _is_numeric_pk_type(dt: T.DataType) -> bool:
    return _is_integer_type(dt) or _is_float_type(dt) or _is_decimal_type(dt)


def _is_string_type(dt: T.DataType) -> bool:
    return isinstance(dt, T.StringType)


def _is_safe_pk_type(dt: T.DataType) -> bool:
    return _is_numeric_pk_type(dt) or _is_string_type(dt)


def _get_field_type(df: DataFrame, col_name: str) -> T.DataType:
    for f in df.schema.fields:
        if f.name == col_name:
            return f.dataType
    raise ValueError(f"Coluna `{col_name}` não existe no DataFrame.")


def _has_column(df: DataFrame, col_name: str) -> bool:
    return col_name in df.columns


def _normalize_engorda_ts(value: Optional[datetime]) -> datetime:
    """Retorna o timestamp único do run de engorda."""
    if value is None:
        return datetime.now().replace(microsecond=0)

    if isinstance(value, datetime):
        return value.replace(microsecond=0)

    raise TypeError("engorda_ts deve ser datetime ou None.")


def _engorda_date(value: datetime) -> date:
    return value.date()


def _timestamp_literal_for_type(value: datetime, dt: T.DataType):
    """Literal de timestamp respeitando o tipo físico da coluna."""
    if isinstance(dt, T.StringType):
        return F.date_format(F.lit(value).cast("timestamp"), "yyyy-MM-dd HH:mm:ss")
    return F.lit(value).cast(dt)


def _date_literal_for_type(value: date, dt: T.DataType):
    """Literal de data sem hora respeitando o tipo físico da coluna."""
    if isinstance(dt, T.StringType):
        return F.date_format(F.lit(value).cast("date"), "yyyy-MM-dd")
    if isinstance(dt, T.TimestampType):
        # Sem timestamp/hora útil: grava a data à meia-noite se a coluna física
        # for TimestampType no metastore/origem.
        return F.lit(value).cast("timestamp").cast(dt)
    return F.lit(value).cast(dt)


def _date_expression_for_type(expr, dt: T.DataType):
    """Expressão de data sem hora respeitando o tipo físico da coluna."""
    if isinstance(dt, T.StringType):
        return F.date_format(expr.cast("date"), "yyyy-MM-dd")
    if isinstance(dt, T.TimestampType):
        return expr.cast("date").cast("timestamp").cast(dt)
    return expr.cast(dt)


def _apply_engorda_business_rules(
    work: DataFrame,
    *,
    engorda_ts: datetime,
    dt_vencimento_prazo_dias: Optional[int] = None,
    default_dt_vencimento_prazo_dias: int = DEFAULT_DT_VENCIMENTO_PRAZO_DIAS,
) -> DataFrame:
    """
    Aplica as regras de DATA do engorda às colunas existentes na tabela.

    Regras (aplicadas só quando a coluna existir; tipo físico preservado):
      DAT_INCLUSAO  -> timestamp da engorda
      DAT_ALTERACAO -> mesmo timestamp de DAT_INCLUSAO
      DT_EMISSAO    -> data da engorda, sem hora
      DT_VENCIMENTO -> data da engorda + prazo, sem hora

    NUM_ID_CERTIFICACAO_CETIP NÃO é tratada aqui: é PK, e a geração de PK
    (compute_pk_maxes + _set_unique_pk_column) já a faz incremental acima do max
    real da tabela inteira, com a folga do --pk-safety-band. Reescrevê-la aqui
    desfazia a folga e relia o max desnecessariamente.

    A função é tolerante: se uma coluna não existir, não altera a tabela.
    """
    engorda_dt = _engorda_date(engorda_ts)

    # 1) Calcula prazo de vencimento ANTES de sobrescrever DT_EMISSAO.
    tmp_prazo_col = "__engorda_prazo_dias"
    while tmp_prazo_col in work.columns:
        tmp_prazo_col = f"_{tmp_prazo_col}"

    has_vencimento = ENGORDA_COL_DT_VENCIMENTO in work.columns
    has_emissao = ENGORDA_COL_DT_EMISSAO in work.columns

    if has_vencimento:
        if dt_vencimento_prazo_dias is not None:
            prazo_expr = F.lit(int(dt_vencimento_prazo_dias)).cast("int")
        elif has_emissao:
            prazo_expr = F.datediff(
                F.to_date(F.col(ENGORDA_COL_DT_VENCIMENTO)),
                F.to_date(F.col(ENGORDA_COL_DT_EMISSAO)),
            ).cast("int")
        else:
            prazo_expr = F.lit(int(default_dt_vencimento_prazo_dias)).cast("int")

        prazo_expr = F.coalesce(
            prazo_expr,
            F.lit(int(default_dt_vencimento_prazo_dias)).cast("int"),
        )
        prazo_expr = F.when(
            prazo_expr < F.lit(MIN_DT_VENCIMENTO_PRAZO_DIAS),
            F.lit(int(default_dt_vencimento_prazo_dias)).cast("int"),
        ).otherwise(prazo_expr)

        work = work.withColumn(tmp_prazo_col, prazo_expr)

    # 2) DAT_INCLUSAO e DAT_ALTERACAO usam exatamente o mesmo timestamp.
    for col_name in (ENGORDA_COL_DAT_INCLUSAO, ENGORDA_COL_DAT_ALTERACAO):
        if col_name in work.columns:
            work = work.withColumn(
                col_name,
                _timestamp_literal_for_type(
                    engorda_ts,
                    _get_field_type(work, col_name),
                ),
            )

    # 3) DT_EMISSAO = data da engorda sem timestamp.
    if has_emissao:
        work = work.withColumn(
            ENGORDA_COL_DT_EMISSAO,
            _date_literal_for_type(
                engorda_dt,
                _get_field_type(work, ENGORDA_COL_DT_EMISSAO),
            ),
        )

    # 4) DT_VENCIMENTO = data da engorda + prazo, sem timestamp.
    if has_vencimento:
        venc_expr = F.expr(
            f"date_add(DATE '{engorda_dt.isoformat()}', CAST({tmp_prazo_col} AS INT))"
        )
        work = work.withColumn(
            ENGORDA_COL_DT_VENCIMENTO,
            _date_expression_for_type(
                venc_expr,
                _get_field_type(work, ENGORDA_COL_DT_VENCIMENTO),
            ),
        ).drop(tmp_prazo_col)

    return work


def _persist(df: DataFrame, storage_level: StorageLevel) -> DataFrame:
    return df.persist(storage_level)


def _materialize(
    df: DataFrame, storage_level: StorageLevel, truncate_lineage: bool
) -> DataFrame:
    """Materialize a stage. When truncate_lineage is set, use an eager
    localCheckpoint so the LOGICAL plan is replaced by a shallow RDD leaf —
    a persist keeps the cached plan tree, which the analyzer still traverses on
    every dependent query, so deep FK-mapping chains compound until the driver
    OOMs analyzing them. Used for --limit, where the referential-sample chain
    makes the plans deep; the full run keeps persist (recomputable on
    executor loss, which localCheckpoint is not)."""
    if truncate_lineage:
        return df.localCheckpoint(eager=True)
    out = _persist(df, storage_level)
    out.count()
    return out


def _safe_unpersist(df: Optional[DataFrame]) -> None:
    if df is None:
        return
    try:
        df.unpersist()
    except Exception:
        pass


def _warn_or_raise(message: str, *, policy: RelationshipPolicy = "warn_and_skip") -> None:
    """
    Centraliza a política para relacionamento inválido.

    policy="raise": mantém comportamento estrito.
    policy="warn_and_skip": emite warning e deixa o processamento continuar.
    """
    if policy == "raise":
        raise ValueError(message)

    if policy == "warn_and_skip":
        warnings.warn(message, UserWarning, stacklevel=2)
        return

    raise ValueError(f"relationship_policy inválida: {policy!r}")


def _format_fk(child_table: str, fk: ForeignKeySpec) -> str:
    return (
        f"{child_table}.{list(fk.columns)} -> "
        f"{fk.parent_table}.{list(fk.parent_columns)}"
    )


def _sanitize_specs_against_known_tables(
    specs: Mapping[str, TableSpec],
    known_tables: Mapping[str, Any],
    *,
    relationship_policy: RelationshipPolicy = "warn_and_skip",
) -> Dict[str, TableSpec]:
    """
    Remove FKs que apontam para parent_table inexistente em specs/known_tables.

    Usada antes de ler/processar dados, principalmente em run_synthesis_from_paths.
    Não valida colunas, pois os DataFrames ainda podem não ter sido lidos.
    """
    if relationship_policy not in ("warn_and_skip", "raise"):
        raise ValueError("relationship_policy deve ser 'warn_and_skip' ou 'raise'.")

    sanitized: Dict[str, TableSpec] = {}

    for name, spec in specs.items():
        valid_fks: List[ForeignKeySpec] = []

        for fk in spec.foreign_keys:
            problems: List[str] = []

            if fk.parent_table == name:
                problems.append("self-reference não é suportado")

            if fk.parent_table not in specs:
                problems.append(
                    f"parent_table `{fk.parent_table}` não existe em specs_config/specs"
                )

            if fk.parent_table not in known_tables:
                problems.append(
                    f"parent_table `{fk.parent_table}` não existe em table_paths/tables"
                )

            if len(fk.columns) != len(fk.parent_columns):
                problems.append(
                    f"quantidade de columns {list(fk.columns)} difere de "
                    f"parent_columns {list(fk.parent_columns)}"
                )

            if problems:
                _warn_or_raise(
                    "Relacionamento ignorado: "
                    f"{_format_fk(name, fk)}. Motivo(s): "
                    + "; ".join(problems)
                    + ". As tabelas serão geradas sem preservar essa FK.",
                    policy=relationship_policy,
                )
                continue

            valid_fks.append(fk)

        sanitized[name] = TableSpec(
            name=spec.name,
            pk_cols=spec.pk_cols,
            foreign_keys=tuple(valid_fks),
            static=spec.static,
            postprocess=spec.postprocess,
        )

    return sanitized


def _fk_has_data_problem(
    tables: Mapping[str, DataFrame],
    child_table: str,
    fk: ForeignKeySpec,
    *,
    nullable_fk_policy: NullableFkPolicy = "allow_any_null",
) -> Optional[str]:
    """
    Verifica se a FK declarada existe logicamente nos dados de entrada.

    Retorna:
        None se o relacionamento parece válido.
        Uma string com o motivo se deve ser ignorado.

    Regras:
        - Se não houver nenhuma chave FK para validar, não considera problema.
        - Se houver zero matches com o pai, ignora a relação.
        - Se houver valores órfãos, ignora a relação para evitar falha posterior.
    """
    child_df_raw = tables[child_table]
    parent_df = tables[fk.parent_table]

    child_df = _filter_child_fk_for_validation(
        child_df_raw,
        fk,
        nullable_fk_policy,
    )

    child_keys = child_df.select(*fk.columns).dropDuplicates()

    total_child_keys = child_keys.count()
    if total_child_keys == 0:
        return None

    parent_keys = parent_df.select(
        *[
            F.col(parent_col).alias(child_col)
            for child_col, parent_col in zip(fk.columns, fk.parent_columns)
        ]
    ).dropDuplicates()

    matched_keys = child_keys.join(
        parent_keys,
        on=list(fk.columns),
        how="inner",
    ).count()

    if matched_keys == 0:
        return (
            f"nenhum valor da FK {list(fk.columns)} da tabela `{child_table}` "
            f"encontrou correspondência no pai `{fk.parent_table}` "
            f"pelas colunas {list(fk.parent_columns)}"
        )

    invalid_keys = child_keys.join(
        parent_keys,
        on=list(fk.columns),
        how="left_anti",
    ).count()

    if invalid_keys > 0:
        return (
            f"existem {invalid_keys} chave(s) FK órfã(s) em `{child_table}` "
            f"para o pai `{fk.parent_table}`"
        )

    return None


def _sanitize_specs_for_available_relationships(
    tables: Mapping[str, DataFrame],
    specs: Mapping[str, TableSpec],
    *,
    relationship_policy: RelationshipPolicy = "warn_and_skip",
    nullable_fk_policy: NullableFkPolicy = "allow_any_null",
    check_relationship_values: bool = True,
) -> Dict[str, TableSpec]:
    """
    Remove FKs inválidas sem impedir a geração das tabelas.

    O que continua sendo erro fatal:
        - specs vazio;
        - tabela declarada em specs inexistente em tables;
        - PK inexistente.

    O que vira warning + FK ignorada:
        - parent_table ausente;
        - coluna FK ausente;
        - parent_column ausente;
        - self-reference;
        - tamanhos diferentes de FK;
        - mesma coluna usada em mais de uma FK;
        - FK sem match com o pai;
        - FK com órfãos.
    """
    if not specs:
        raise ValueError("`specs` está vazio.")

    if relationship_policy not in ("warn_and_skip", "raise"):
        raise ValueError("relationship_policy deve ser 'warn_and_skip' ou 'raise'.")

    sanitized: Dict[str, TableSpec] = {}

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
                    "Sem PK válida não é seguro gerar a tabela."
                )

        seen_fk_cols: set = set()
        valid_fks: List[ForeignKeySpec] = []

        for fk in spec.foreign_keys:
            problems: List[str] = []

            if not fk.columns:
                problems.append("FK vazia")

            if len(fk.columns) != len(fk.parent_columns):
                problems.append(
                    f"quantidade de columns {list(fk.columns)} difere de "
                    f"parent_columns {list(fk.parent_columns)}"
                )

            if fk.parent_table == name:
                problems.append("self-reference não é suportado")

            if fk.parent_table not in specs:
                problems.append(
                    f"parent_table `{fk.parent_table}` não existe em specs"
                )

            if fk.parent_table not in tables:
                problems.append(
                    f"parent_table `{fk.parent_table}` não existe em tables"
                )

            for c in fk.columns:
                if c not in df_cols:
                    problems.append(
                        f"coluna FK `{c}` não existe na tabela filha `{name}`"
                    )

                if c in seen_fk_cols:
                    problems.append(
                        f"coluna `{c}` participa de mais de uma FK; "
                        "remapeamento ambíguo"
                    )

            if fk.parent_table in tables:
                parent_cols = set(tables[fk.parent_table].columns)
                for pc in fk.parent_columns:
                    if pc not in parent_cols:
                        problems.append(
                            f"parent_column `{pc}` não existe no pai `{fk.parent_table}`"
                        )

            if not problems and check_relationship_values:
                data_problem = _fk_has_data_problem(
                    tables,
                    name,
                    fk,
                    nullable_fk_policy=nullable_fk_policy,
                )
                if data_problem:
                    problems.append(data_problem)

            if problems:
                _warn_or_raise(
                    "Relacionamento ignorado: "
                    f"{_format_fk(name, fk)}. Motivo(s): "
                    + "; ".join(problems)
                    + ". As tabelas serão geradas sem preservar essa FK.",
                    policy=relationship_policy,
                )
                continue

            for c in fk.columns:
                seen_fk_cols.add(c)

            valid_fks.append(fk)

        sanitized[name] = TableSpec(
            name=spec.name,
            pk_cols=spec.pk_cols,
            foreign_keys=tuple(valid_fks),
            static=spec.static,
            postprocess=spec.postprocess,
        )

    return sanitized


def _validate_relationship_columns(
    tables: Mapping[str, DataFrame],
    specs: Mapping[str, TableSpec],
    *,
    relationship_policy: RelationshipPolicy = "warn_and_skip",
    nullable_fk_policy: NullableFkPolicy = "allow_any_null",
    check_relationship_values: bool = True,
) -> Dict[str, TableSpec]:
    """
    Confirma PK/FK declaradas contra schemas reais.

    Agora retorna specs saneadas. FKs inválidas viram warning e são ignoradas.
    PK inválida continua sendo erro fatal.
    """
    return _sanitize_specs_for_available_relationships(
        tables,
        specs,
        relationship_policy=relationship_policy,
        nullable_fk_policy=nullable_fk_policy,
        check_relationship_values=check_relationship_values,
    )


def _validate_specs(
    tables: Mapping[str, DataFrame],
    specs: Mapping[str, TableSpec],
) -> None:
    """
    Validação estrita das specs já saneadas.

    Esta função mantém o nome original, mas agora deve receber specs sem FKs
    inválidas. A sanitização acontece antes dela dentro de synthesize_multitable_spark.
    """
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
                raise ValueError(
                    f"FK em `{name}` referencia `{fk.parent_table}` ausente em specs."
                )

            if fk.parent_table not in tables:
                raise ValueError(
                    f"FK em `{name}` referencia `{fk.parent_table}` ausente em tables."
                )

            for c in fk.columns:
                if c not in df_cols:
                    raise ValueError(f"FK col `{c}` não existe na filha `{name}`.")

                if c in seen_fk_cols:
                    raise ValueError(f"Coluna `{c}` em `{name}` participa de mais de uma FK.")

                seen_fk_cols.add(c)

            parent_cols = set(tables[fk.parent_table].columns)
            for pc in fk.parent_columns:
                if pc not in parent_cols:
                    raise ValueError(
                        f"FK em `{name}` referencia `{pc}` ausente no pai `{fk.parent_table}`."
                    )


def _toposort_break_cycles(
    deps: Mapping[str, set],
    *,
    on_cycle: Optional[Callable[[set, Mapping[str, set], set], None]] = None,
) -> List[str]:
    """Order nodes so every node follows all of its `deps`, returning each node
    exactly once.

    Cycles are not fatal: when no node is ready, the cycle is broken by forcing
    the node with the fewest still-unresolved deps (ties broken by name) so the
    result is deterministic. `on_cycle(remaining, deps, done)` is invoked the
    first time a break is forced, for callers that want to warn.
    """
    remaining = set(deps)
    done: set = set()
    order: List[str] = []
    cycle_reported = False

    while remaining:
        ready = sorted(n for n in remaining if deps[n] <= done)

        if not ready:
            if on_cycle is not None and not cycle_reported:
                on_cycle(remaining, deps, done)
                cycle_reported = True
            ready = [min(remaining, key=lambda n: (len(deps[n] - done), n))]

        for name in ready:
            order.append(name)
            done.add(name)
            remaining.discard(name)

    return order


def _topological_order(specs: Mapping[str, TableSpec]) -> List[str]:
    """Parents before children. Shares topo_order_tables' cycle policy: cycles
    are broken (with a warning) rather than raised, since sanitize/validate have
    already removed self-refs and missing parents by the time this runs."""
    deps: Dict[str, set] = {
        name: {
            fk.parent_table
            for fk in spec.foreign_keys
            if fk.parent_table != name and fk.parent_table in specs
        }
        for name, spec in specs.items()
    }

    def _warn(remaining: set, deps: Mapping[str, set], done: set) -> None:
        unresolved = {t: sorted(deps[t] - done) for t in sorted(remaining) if deps[t] - done}
        warnings.warn(
            "Ciclo de FK detectado; quebrando arbitrariamente para ordenar. "
            f"Pendências: {unresolved}",
            UserWarning,
            stacklevel=2,
        )

    return _toposort_break_cycles(deps, on_cycle=_warn)


def _referenced_parent_columns(specs: Mapping[str, TableSpec]) -> Dict[str, set]:
    refs: Dict[str, set] = {}

    for child_spec in specs.values():
        for fk in child_spec.foreign_keys:
            refs.setdefault(fk.parent_table, set()).add(tuple(fk.parent_columns))

    return refs


def _with_contiguous_row_id(df: DataFrame, id_col: str) -> DataFrame:
    """
    Adiciona um identificador contíguo 0..N-1 de forma paralela.

    Substitui a versão anterior que usava Window.orderBy() sem partitionBy,
    o que forçava toda a tabela em uma única tarefa (single-task sort). Em
    tabelas de 600M+ linhas isso era um gargalo serial intransponível.

    Algoritmo:
        1. mid = monotonically_increasing_id() — ordem determinística por
           partição (codifica (partition_id, counter) nos bits altos/baixos).
        2. part = spark_partition_id() — id da partição de origem.
        3. part_row = row_number() over (partitionBy part orderBy mid) —
           contador local, totalmente paralelo (sem shuffle entre partições).
        4. sizes = groupBy(part).agg(count(*)) — uma linha por partição.
           O map-side combine reduz N linhas a ~num_partições linhas antes
           do shuffle, então a etapa é barota mas leve.
        5. offset = soma cumulativa de sizes, calculada no DRIVER. `sizes` tem
           uma linha por partição e já é pequeno; coletá-lo e fazer o prefix-sum
           em Python evita um Window sem partitionBy, que o Spark executa como
           Exchange SinglePartition (serial, trava em tabelas grandes).
        6. id_col = offset + part_row - 1, com offset trazido por broadcast.

    Equivalência com a versão anterior:
        monotonically_increasing_id() ordena por (partition_id, counter).
        A ordenação global anterior era: partição 0 em ordem de counter,
        depois partição 1, etc. Esta versão reproduz exatamente essa ordem
        mas sem mover dados entre partições — cada partição calcula seu
        row_number localmente e recebe apenas seu offset por broadcast.

    Determinismo:
        A leitura Parquet é determinística (ordem de linhas por arquivo é
        estável), então mid_col tem a mesma ordem nas duas materializações
        (uma para sizes, outra para o join final). part_row é consistente
        porque depende apenas da ordem de mid dentro de cada partição.
    """
    part_col = f"__{id_col}_part"
    while part_col in df.columns:
        part_col = f"_{part_col}"

    part_row_col = f"__{id_col}_prow"
    while part_row_col in df.columns:
        part_row_col = f"_{part_row_col}"

    part_size_col = f"__{id_col}_psize"
    while part_size_col in df.columns:
        part_size_col = f"_{part_size_col}"

    offset_col = f"__{id_col}_poff"
    while offset_col in df.columns:
        offset_col = f"_{offset_col}"

    mid_col = f"__{id_col}_mid"
    while mid_col in df.columns:
        mid_col = f"_{mid_col}"

    df = (
        df
        .withColumn(mid_col, F.monotonically_increasing_id())
        .withColumn(part_col, F.spark_partition_id())
    )

    w_part = Window.partitionBy(part_col).orderBy(F.col(mid_col))
    df = df.withColumn(part_row_col, F.row_number().over(w_part))

    sizes = (
        df.groupBy(part_col)
        .agg(F.count(F.lit(1)).cast("long").alias(part_size_col))
    )

    # Prefix-sum no driver: uma linha por partição. Coletar é o mesmo custo do
    # broadcast a seguir e evita o Window sem partitionBy (SinglePartition).
    spark = df.sparkSession
    ordered_sizes = sorted(
        ((row[part_col], row[part_size_col]) for row in sizes.collect()),
        key=lambda pair: pair[0],
    )
    running = 0
    offset_rows: List[Tuple[int, int]] = []
    for part_value, size in ordered_sizes:
        offset_rows.append((part_value, running))
        running += size

    offset_schema = T.StructType(
        [
            T.StructField(part_col, T.IntegerType(), False),
            T.StructField(offset_col, T.LongType(), False),
        ]
    )
    offsets = spark.createDataFrame(offset_rows, schema=offset_schema)

    df = df.join(F.broadcast(offsets), on=part_col, how="left")

    df = df.withColumn(
        id_col,
        (F.col(offset_col) + F.col(part_row_col) - F.lit(1)).cast("long"),
    )

    return df.drop(mid_col, part_col, part_row_col, offset_col)


def _bootstrap_rows_exact(
    src_indexed: DataFrame,
    n_rows: int,
    *,
    src_count: int,
    seed: int,
    spark: SparkSession,
    keep_all_source_rows: bool,
) -> DataFrame:
    if n_rows < 0:
        raise ValueError("n_rows deve ser >= 0.")

    src_cols = [c for c in src_indexed.columns if c != "__src_row_id"]

    if n_rows == 0:
        # Evita spark.createDataFrame([], schema=...), que pode acionar
        # cloudpickle em algumas versões do PySpark/Python.
        return src_indexed.limit(0).select(
            F.lit(None).cast("long").alias("__synthetic_pos"),
            F.lit(None).cast("long").alias("__orig_src_row_id"),
            *[F.col(c) for c in src_cols],
        )

    if src_count == 0:
        raise ValueError("Fonte vazia mas n_rows > 0.")

    if keep_all_source_rows:
        if n_rows < src_count:
            raise ValueError(
                f"Pai precisa n_rows >= src_count. n_rows={n_rows}, src_count={src_count}."
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


_INT_TYPE_LIMITS = (
    (T.ByteType, 127),
    (T.ShortType, 32_767),
    (T.IntegerType, 2_147_483_647),
)


_FLOAT_EXACT_INT_LIMIT = 16_777_216            # 2^24 (float 32 bits)


_DOUBLE_EXACT_INT_LIMIT = 9_007_199_254_740_992  # 2^53 (double 64 bits)


def _max_pk_value(df_cached: DataFrame, pk: str) -> Optional[int]:
    """
    Retorna o maior valor atual da PK como int.

    v4: também funciona para PK double/float/decimal (caso típico de CSV lido
    com inferSchema=True). Valores NaN são ignorados via floor seguro.
    """
    row = df_cached.agg(F.max(F.col(pk)).alias("max_pk")).collect()[0]
    value = row["max_pk"]

    if value is None:
        return None

    value_f = float(value)

    # NaN não é comparável; trata como inexistente para não propagar lixo.
    if math.isnan(value_f):
        return None

    return int(math.floor(value_f))


def _set_unique_pk_column(
    work: DataFrame,
    source_cached: DataFrame,
    pk: str,
    *,
    append_after_max: bool,
    target_n: int,
    offset: int = 0,
    pk_max_override: Optional[int] = None,
) -> DataFrame:
    # When pk_max_override is given, append after THIS max instead of the one
    # observed in source_cached. Used so a --limit'd (sampled) source still gets
    # PKs above the table's TRUE max, computed from the full Parquet by engorda.
    dt = _get_field_type(source_cached, pk)

    if _is_integer_type(dt):
        observed_max = (
            pk_max_override if pk_max_override is not None
            else _max_pk_value(source_cached, pk)
        )
        start = (observed_max or 0) + 1 if append_after_max else 1
        highest = start + target_n - 1 + offset

        for type_cls, limit in _INT_TYPE_LIMITS:
            if isinstance(dt, type_cls) and highest > limit:
                raise OverflowError(
                    f"PK `{pk}` {type_cls.__name__} estoura limite {limit:,} "
                    f"(max {highest:,})."
                )

        return work.withColumn(
            pk,
            (F.col("__synthetic_pos") + F.lit(start + offset)).cast(dt),
        )

    # ---- NOVO na v4: PK em ponto flutuante (double/float) -----------------
    # Cenário típico: CSV lido com inferSchema=True infere IDs como double.
    # Estratégia: gerar a mesma sequência inteira e castar para o tipo
    # original, garantindo que os valores fiquem na faixa de inteiros
    # representáveis de forma exata (2^53 para double, 2^24 para float).
    if _is_float_type(dt):
        observed_max = (
            pk_max_override if pk_max_override is not None
            else _max_pk_value(source_cached, pk)
        )
        start = (observed_max or 0) + 1 if append_after_max else 1
        highest = start + target_n - 1 + offset

        exact_limit = (
            _DOUBLE_EXACT_INT_LIMIT
            if isinstance(dt, T.DoubleType)
            else _FLOAT_EXACT_INT_LIMIT
        )

        if highest > exact_limit:
            raise OverflowError(
                f"PK `{pk}` ({type(dt).__name__}) atingiria {highest:,}, acima do "
                f"limite de inteiro exato {exact_limit:,}. Acima disso valores "
                "consecutivos colidem e a PK deixaria de ser única. "
                "Sugestão: converta a coluna para LongType na leitura."
            )

        return work.withColumn(
            pk,
            (F.col("__synthetic_pos") + F.lit(start + offset)).cast(dt),
        )

    # ---- NOVO na v4: PK decimal -------------------------------------------
    if _is_decimal_type(dt):
        observed_max = (
            pk_max_override if pk_max_override is not None
            else _max_pk_value(source_cached, pk)
        )
        start = (observed_max or 0) + 1 if append_after_max else 1
        highest = start + target_n - 1 + offset

        # Dígitos inteiros disponíveis = precision - scale.
        int_digits = dt.precision - dt.scale
        decimal_limit = (10 ** int_digits) - 1 if int_digits > 0 else 0

        if highest > decimal_limit:
            raise OverflowError(
                f"PK `{pk}` Decimal({dt.precision},{dt.scale}) estoura o limite "
                f"de {decimal_limit:,} (max {highest:,})."
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
        f"PK `{pk}` tipo {dt!r} sem estratégia segura. "
        "Tipos suportados: inteiro, double, float, decimal e string. "
        "Sugestão: faça cast da coluna para um desses tipos antes da síntese."
    )


def _generate_pk_columns(
    work: DataFrame,
    source_cached: DataFrame,
    spec: TableSpec,
    *,
    append_after_max: bool,
    target_n: int,
    pk_max_override: Optional[int] = None,
) -> DataFrame:
    if len(spec.pk_cols) == 1:
        return _set_unique_pk_column(
            work,
            source_cached,
            spec.pk_cols[0],
            append_after_max=append_after_max,
            target_n=target_n,
            pk_max_override=pk_max_override,
        )

    last_pk = spec.pk_cols[-1]
    last_type = _get_field_type(source_cached, last_pk)

    if not _is_safe_pk_type(last_type):
        raise TypeError(
            f"PK composta `{spec.name}` última col `{last_pk}` tipo {last_type!r} inseguro. "
            "Tipos suportados: inteiro, double, float, decimal e string."
        )

    return _set_unique_pk_column(
        work,
        source_cached,
        last_pk,
        append_after_max=append_after_max,
        target_n=target_n,
        pk_max_override=pk_max_override,
    )


def _build_mapping_for_parent_cols(
    work_cached: DataFrame,
    parent_cols: Tuple[str, ...],
    storage_level: StorageLevel,
) -> DataFrame:
    old_cols = [f"__old__{c}" for c in parent_cols]
    missing_old = [c for c in old_cols if c not in work_cached.columns]

    if missing_old:
        raise ValueError(f"Mapping: colunas antigas ausentes: {missing_old}")

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
    fk_index: int = 0,
) -> DataFrame:
    fk_tag = (
        f"__fk{fk_index}_{fk.parent_table}_"
        f"{_stable_seed(seed, fk.parent_table, fk.columns, fk.parent_columns)}"
    )
    n = len(fk.columns)

    counts = mapping.select(
        *[
            F.col(f"__old_{i}").alias(f"{fk_tag}_old_{i}")
            for i in range(n)
        ],
        F.col("__candidate_count").alias(f"{fk_tag}_count"),
    ).dropDuplicates([f"{fk_tag}_old_{i}" for i in range(n)])

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
    cond_map_key = _fk_join_condition(work, list(fk.columns), m, map_old_cols)
    cond_map = cond_map_key & (work[f"{fk_tag}_rank"] == m[f"{fk_tag}_map_rank"])

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
        + [f"{fk_tag}_count", f"{fk_tag}_rank", f"{fk_tag}_map_rank"]
    )

    return work.drop(*drop_cols)


def _rows_to_spark_df(
    spark: SparkSession,
    rows: List[Tuple[Any, ...]],
    columns: List[Tuple[str, str]],
) -> DataFrame:
    """
    Cria um DataFrame pequeno de diagnóstico sem spark.createDataFrame(rows).

    Motivo:
        Em alguns ambientes, spark.createDataFrame(lista_python) pode acionar
        cloudpickle.dumps e falhar com:

            IndexError: tuple index out of range
            Could not serialize object

        Para evitar isso, montamos cada linha usando spark.range(1).select(lit(...)).
        Assim não serializamos função Python nem lista de Row para os executores.

    Args:
        spark: SparkSession.
        rows: lista de tuplas com os valores.
        columns: lista de pares (nome_coluna, tipo_spark_sql), ex.:
                 [("table", "string"), ("total_rows", "long")].
    """
    select_exprs_empty = [
        F.lit(None).cast(dtype).alias(name)
        for name, dtype in columns
    ]

    if not rows:
        return spark.range(0).select(*select_exprs_empty)

    df_out: Optional[DataFrame] = None

    for row in rows:
        if len(row) != len(columns):
            raise ValueError(
                f"Linha de diagnóstico possui {len(row)} valores, mas eram esperados "
                f"{len(columns)}: {row!r}"
            )

        exprs = [
            F.lit(value).cast(dtype).alias(name)
            for value, (name, dtype) in zip(row, columns)
        ]
        one = spark.range(1).select(*exprs)
        df_out = one if df_out is None else df_out.unionByName(one)

    return df_out


def validate_primary_keys(
    tables: Mapping[str, DataFrame],
    specs: Mapping[str, TableSpec],
) -> DataFrame:
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

        rows.append(
            (
                name,
                ",".join(spec.pk_cols),
                int(total_rows),
                int(distinct_pk),
                int(null_pk_rows),
                int(total_rows - distinct_pk),
            )
        )

    return _rows_to_spark_df(
        spark,
        rows,
        columns=[
            ("table", "string"),
            ("pk_cols", "string"),
            ("total_rows", "long"),
            ("distinct_pk", "long"),
            ("null_pk_rows", "long"),
            ("duplicate_pk_rows", "long"),
        ],
    )


def _filter_child_fk_for_validation(
    child_df: DataFrame,
    fk: ForeignKeySpec,
    nullable_fk_policy: NullableFkPolicy,
) -> DataFrame:
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
    relationship_policy: RelationshipPolicy = "warn_and_skip",
) -> DataFrame:
    """
    Valida FKs que ainda estão ativas em specs.

    Se por algum motivo uma FK inválida chegar aqui e relationship_policy for
    warn_and_skip, ela entra no relatório como invalid_fk=-1 e gera warning,
    sem quebrar a execução.
    """
    spark = next(iter(tables.values())).sparkSession
    rows = []

    for child_name, child_spec in specs.items():
        child_df_raw = tables[child_name]

        for fk in child_spec.foreign_keys:
            if fk.parent_table not in tables:
                _warn_or_raise(
                    "Validação FK ignorada: "
                    f"{_format_fk(child_name, fk)}. Pai não existe em tables.",
                    policy=relationship_policy,
                )
                rows.append(
                    (
                        child_name,
                        ",".join(fk.columns),
                        fk.parent_table,
                        ",".join(fk.parent_columns),
                        0,
                        -1,
                    )
                )
                continue

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

    return _rows_to_spark_df(
        spark,
        rows,
        columns=[
            ("child_table", "string"),
            ("fk_cols", "string"),
            ("parent_table", "string"),
            ("parent_cols", "string"),
            ("distinct_child_fk", "long"),
            ("invalid_fk", "long"),
        ],
    )


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
        relationship_policy="warn_and_skip",
    )

    pk_problems = pk_report.where(
        "null_pk_rows > 0 OR duplicate_pk_rows > 0"
    ).count()

    fk_problems = fk_report.where(
        "invalid_fk > 0"
    ).count()

    if pk_problems or fk_problems:
        print(">>> FALHA NA VALIDAÇÃO")
        pk_report.show(truncate=False)
        fk_report.show(truncate=False)
        raise RuntimeError(
            f"Validação falhou: {pk_problems} PK, {fk_problems} FK."
        )


def run_validation_or_raise(
    result: Mapping[str, DataFrame],
    specs: Mapping[str, TableSpec],
    *,
    nullable_fk_policy: NullableFkPolicy = "allow_any_null",
) -> None:
    _run_validation_or_raise(
        result,
        specs,
        nullable_fk_policy=nullable_fk_policy,
    )


def synthesize_multitable_spark(
    tables: Mapping[str, DataFrame],
    specs: Mapping[str, TableSpec],
    n_rows_by_table: Optional[Mapping[str, int]] = None,
    *,
    seed: int = 42,
    append_after_max_pk: bool = True,
    pk_max_by_table: Optional[Mapping[str, int]] = None,
    engorda_ts: Optional[datetime] = None,
    dt_vencimento_prazo_dias_by_table: Optional[Mapping[str, int]] = None,
    default_dt_vencimento_prazo_dias: int = DEFAULT_DT_VENCIMENTO_PRAZO_DIAS,
    validate_mode: ValidateMode = "full",
    nullable_fk_policy: NullableFkPolicy = "allow_any_null",
    broadcast_fk_counts: bool = False,
    storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    verbose: bool = False,
    relationship_policy: RelationshipPolicy = "warn_and_skip",
    check_relationship_values: bool = True,
    truncate_lineage: bool = False,
) -> Dict[str, DataFrame]:
    """
    Gera dados sintéticos multi-tabela.

    Novo comportamento:
        FKs inválidas/ausentes são ignoradas com warning quando
        relationship_policy="warn_and_skip".

    Para recuperar comportamento estrito antigo:
        relationship_policy="raise"
    """
    if validate_mode not in ("none", "full"):
        raise ValueError("validate_mode deve ser 'none' ou 'full'.")

    if relationship_policy not in ("warn_and_skip", "raise"):
        raise ValueError("relationship_policy deve ser 'warn_and_skip' ou 'raise'.")

    engorda_ts = _normalize_engorda_ts(engorda_ts)
    dt_vencimento_prazo_dias_by_table = dict(dt_vencimento_prazo_dias_by_table or {})

    # Saneia specs antes da validação/topologia/mapping.
    active_specs = _sanitize_specs_for_available_relationships(
        tables,
        specs,
        relationship_policy=relationship_policy,
        nullable_fk_policy=nullable_fk_policy,
        check_relationship_values=check_relationship_values,
    )

    _validate_specs(tables, active_specs)

    n_rows_by_table = dict(n_rows_by_table or {})
    order = _topological_order(active_specs)
    parent_refs = _referenced_parent_columns(active_specs)

    result: Dict[str, DataFrame] = {}
    mappings: Dict[Tuple[str, Tuple[str, ...]], DataFrame] = {}
    intermediates: List[DataFrame] = []

    # How many child tables still need each parent mapping. Once a parent's last
    # consumer is synthesized, its mapping is unpersisted instead of being held
    # for the whole component — large components otherwise keep every table's
    # bootstrapped frames + mappings cached at once, which is the memory wall.
    mapping_consumers: Dict[Tuple[str, Tuple[str, ...]], int] = {}
    for child_name in order:
        for fk in active_specs[child_name].foreign_keys:
            key = (fk.parent_table, tuple(fk.parent_columns))
            mapping_consumers[key] = mapping_consumers.get(key, 0) + 1

    def _release_mapping_consumer(key: Tuple[str, Tuple[str, ...]]) -> None:
        remaining = mapping_consumers.get(key, 0) - 1
        mapping_consumers[key] = remaining
        if remaining <= 0:
            _safe_unpersist(mappings.pop(key, None))

    if verbose:
        print("Specs ativas após saneamento de relacionamentos:")
        for table_name, spec in active_specs.items():
            if spec.foreign_keys:
                for fk in spec.foreign_keys:
                    print("  OK:", _format_fk(table_name, fk))
            else:
                print(f"  {table_name}: sem FK ativa")

    try:
        for table_name in order:
            source = tables[table_name]
            spec = active_specs[table_name]
            spark = source.sparkSession
            original_cols = source.columns
            target_n_raw = n_rows_by_table.get(table_name)

            ref_col_sets = parent_refs.get(table_name, set())
            ref_cols = sorted(
                set(c for cols in ref_col_sets for c in cols)
                | set(spec.pk_cols)
            )

            src_indexed: Optional[DataFrame] = None

            if spec.static:
                src_count = source.count()

                if target_n_raw is not None and int(target_n_raw) != src_count:
                    warnings.warn(
                        f"`{table_name}` static; n_rows ignorado.",
                        UserWarning,
                        stacklevel=2,
                    )

                if verbose:
                    print(f"[{table_name}] STATIC | {src_count} linhas")

                work = (
                    _with_contiguous_row_id(source, "__synthetic_pos")
                    .withColumn("__orig_src_row_id", F.col("__synthetic_pos"))
                )

                for c in ref_cols:
                    work = work.withColumn(f"__old__{c}", F.col(c))

                if spec.postprocess is not None:
                    work = spec.postprocess(work, result)

                work = _apply_engorda_business_rules(
                    work,
                    engorda_ts=engorda_ts,
                    dt_vencimento_prazo_dias=dt_vencimento_prazo_dias_by_table.get(table_name),
                    default_dt_vencimento_prazo_dias=default_dt_vencimento_prazo_dias,
                )

                work = _materialize(work, storage_level, truncate_lineage)
                intermediates.append(work)

            else:
                src_indexed = _with_contiguous_row_id(source, "__src_row_id")
                src_indexed = _persist(src_indexed, storage_level)
                src_count = src_indexed.count()
                intermediates.append(src_indexed)

                target_n = int(target_n_raw if target_n_raw is not None else src_count)
                keep_all = table_name in parent_refs

                if verbose:
                    print(
                        f"[{table_name}] {'PAI' if keep_all else 'FILHO'} | "
                        f"{src_count}->{target_n}"
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
                    pk_max_override=(pk_max_by_table or {}).get(table_name),
                )

                for fk_idx, fk in enumerate(spec.foreign_keys):
                    key = (fk.parent_table, tuple(fk.parent_columns))

                    if key not in mappings:
                        _warn_or_raise(
                            "Mapping ausente para relacionamento ativo: "
                            f"{_format_fk(table_name, fk)}. "
                            "A FK será mantida sem remapeamento nesta tabela.",
                            policy=relationship_policy,
                        )
                        _release_mapping_consumer(key)
                        continue

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
                        fk_index=fk_idx,
                    )
                    _release_mapping_consumer(key)

                if spec.postprocess is not None:
                    work = spec.postprocess(work, result)

                work = _apply_engorda_business_rules(
                    work,
                    engorda_ts=engorda_ts,
                    dt_vencimento_prazo_dias=dt_vencimento_prazo_dias_by_table.get(table_name),
                    default_dt_vencimento_prazo_dias=default_dt_vencimento_prazo_dias,
                )

                work = _materialize(work, storage_level, truncate_lineage)
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

            # synth and any parent mapping are now materialized with their own
            # cached blocks, so the bulky bootstrapped frames for this table are
            # no longer needed — free them now instead of at end-of-component.
            _safe_unpersist(work)
            _safe_unpersist(src_indexed)

        if validate_mode == "full":
            if verbose:
                print("Validando...")

            _run_validation_or_raise(
                result,
                active_specs,
                nullable_fk_policy=nullable_fk_policy,
            )

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


def _normalize_cols(value: Any, *, field_name: str, table_name: str) -> Tuple[str, ...]:
    if value is None:
        raise ValueError(f"Tabela `{table_name}`: `{field_name}` é obrigatório.")

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"Tabela `{table_name}`: `{field_name}` deve ser string/list/tuple. "
            f"Recebido: {type(value).__name__}."
        )

    out = tuple(str(c).strip() for c in value if str(c).strip())

    if not out:
        raise ValueError(f"Tabela `{table_name}`: `{field_name}` não pode ser vazio.")

    return out


def _try_normalize_cols(
    value: Any,
    *,
    field_name: str,
    table_name: str,
    fk_index: Optional[int] = None,
) -> Optional[Tuple[str, ...]]:
    """
    Versão tolerante de _normalize_cols.

    Retorna None quando o campo não existe ou está vazio.
    Isso permite avisar e ignorar apenas a FK problemática sem parar o código.
    """
    try:
        return _normalize_cols(value, field_name=field_name, table_name=table_name)
    except Exception as exc:
        suffix = f" FK #{fk_index}" if fk_index is not None else ""
        warnings.warn(
            f"Configuração de relacionamento ignorada em `{table_name}`{suffix}: "
            f"campo `{field_name}` inválido. Motivo: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None


def _infer_parent_table_from_config(
    specs_config: Mapping[str, Mapping],
    *,
    child_table: str,
    parent_columns: Optional[Tuple[str, ...]],
    child_fk_columns: Optional[Tuple[str, ...]],
) -> Tuple[Optional[str], Optional[Tuple[str, ...]], str]:
    """
    Tenta inferir parent_table quando ele não foi informado na FK.

    Estratégia:
        1. Se parent_columns foi informado, procura uma tabela cuja pk_cols seja
           exatamente igual a parent_columns.
        2. Se parent_columns não foi informado, usa child_fk_columns como pista
           e procura uma tabela cuja pk_cols seja exatamente igual a child_fk_columns.
        3. Se houver match único, retorna a tabela inferida.
        4. Se houver zero ou múltiplos matches, retorna None e motivo amigável.

    Observação:
        A inferência é conservadora de propósito. Se ficar ambígua, a FK é ignorada
        com warning para evitar relacionamento errado.
    """
    candidates: List[str] = []
    target_cols = parent_columns or child_fk_columns

    if not target_cols:
        return None, None, "não foi possível inferir: parent_table e parent_columns ausentes"

    for candidate_table, cfg in specs_config.items():
        if candidate_table == child_table:
            continue
        if not isinstance(cfg, ABCMapping):
            continue
        raw_pk = cfg.get("pk_cols")
        if raw_pk is None:
            continue
        try:
            pk_cols = _normalize_cols(
                raw_pk,
                field_name="pk_cols",
                table_name=str(candidate_table),
            )
        except Exception:
            continue
        if tuple(pk_cols) == tuple(target_cols):
            candidates.append(str(candidate_table))

    if len(candidates) == 1:
        inferred_parent_table = candidates[0]
        inferred_parent_columns = tuple(target_cols)
        return (
            inferred_parent_table,
            inferred_parent_columns,
            f"parent_table inferido automaticamente como `{inferred_parent_table}` "
            f"porque pk_cols={list(inferred_parent_columns)}",
        )

    if not candidates:
        return (
            None,
            None,
            "parent_table ausente e nenhuma tabela candidata foi encontrada "
            f"com pk_cols={list(target_cols)}",
        )

    return (
        None,
        None,
        "parent_table ausente e a inferência ficou ambígua; "
        f"candidatos com pk_cols={list(target_cols)}: {candidates}",
    )


def _build_specs_from_config(
    specs_config: Mapping[str, Mapping],
    postprocess_by_table: Optional[Mapping[str, PostProcessor]] = None,
    *,
    relationship_policy: RelationshipPolicy = "warn_and_skip",
) -> Dict[str, TableSpec]:
    """
    Converte dicionário declarativo em specs tipados.

    Regras desta versão:
        - pk_cols continua obrigatório por tabela.
        - foreign_keys é opcional.
        - Dentro de cada FK, parent_table NÃO é mais obrigatório.
        - Se parent_table não vier, o código tenta inferir pelo pk_cols do pai.
        - Se não conseguir inferir, avisa e ignora somente aquela FK.
        - Se coluna FK/parent_column não existir depois no schema, avisa e ignora
          somente aquela FK na etapa de saneamento.

    Formatos aceitos:
        {
            "tabela_filha": {
                "pk_cols": ["ID_FILHO"],
                "foreign_keys": [
                    {
                        "columns": ["ID_PAI"],
                        # parent_table e parent_columns sao opcionais
                        "parent_table": "tabela_pai",
                        "parent_columns": ["ID_PAI"]
                    }
                ]
            }
        }
    """
    if not isinstance(specs_config, ABCMapping) or not specs_config:
        raise ValueError("`specs_config` deve ser um dicionário não vazio.")

    if relationship_policy not in ("warn_and_skip", "raise"):
        raise ValueError("relationship_policy deve ser 'warn_and_skip' ou 'raise'.")

    postprocess_by_table = dict(postprocess_by_table or {})
    specs: Dict[str, TableSpec] = {}

    for name, cfg in specs_config.items():
        name = str(name).strip()

        if not isinstance(cfg, ABCMapping):
            raise TypeError(
                f"Config da tabela `{name}` deve ser um dict, recebido {type(cfg)!r}."
            )

        # PK é estrutural para a geração. Sem PK não é seguro sintetizar.
        pk_cols = _normalize_cols(
            cfg.get("pk_cols"),
            field_name="pk_cols",
            table_name=name,
        )

        raw_fks = cfg.get("foreign_keys") or cfg.get("fks") or []

        if isinstance(raw_fks, ABCMapping):
            raw_fks = [raw_fks]

        if not isinstance(raw_fks, (list, tuple)):
            _warn_or_raise(
                f"Tabela `{name}`: `foreign_keys` deveria ser lista/tupla de dicts, "
                f"mas veio {type(raw_fks).__name__}. Todas as FKs dessa tabela serão ignoradas.",
                policy=relationship_policy,
            )
            raw_fks = []

        fks: List[ForeignKeySpec] = []

        for i, fk in enumerate(raw_fks):
            if isinstance(fk, ForeignKeySpec):
                # Se vier objeto pronto e completo, mantém.
                # Se estiver incompleto, saneamento posterior trata.
                fks.append(fk)
                continue

            if not isinstance(fk, ABCMapping):
                _warn_or_raise(
                    f"Relacionamento ignorado em `{name}` FK #{i}: esperado dict, recebido {fk!r}.",
                    policy=relationship_policy,
                )
                continue

            # columns continua necessário para saber qual coluna da filha participaria da FK.
            cols = _try_normalize_cols(
                fk.get("columns"),
                field_name="foreign_keys.columns",
                table_name=name,
                fk_index=i,
            )
            if not cols:
                _warn_or_raise(
                    f"Relacionamento ignorado em `{name}` FK #{i}: `columns` não foi informado. "
                    "A tabela será sintetizada sem essa FK.",
                    policy=relationship_policy,
                )
                continue

            raw_parent_table = fk.get("parent_table")
            parent_table = str(raw_parent_table).strip() if raw_parent_table is not None else ""

            parent_cols = _try_normalize_cols(
                fk.get("parent_columns"),
                field_name="foreign_keys.parent_columns",
                table_name=name,
                fk_index=i,
            ) if fk.get("parent_columns") is not None else None

            # Se parent_columns não foi informado, mas parent_table existe, usa pk_cols do pai.
            if parent_cols is None and parent_table:
                parent_cfg = specs_config.get(parent_table)
                if isinstance(parent_cfg, ABCMapping) and parent_cfg.get("pk_cols") is not None:
                    parent_cols = _normalize_cols(
                        parent_cfg.get("pk_cols"),
                        field_name="pk_cols",
                        table_name=parent_table,
                    )
                    warnings.warn(
                        f"Relacionamento `{name}` FK #{i}: `parent_columns` não informado. "
                        f"Usando pk_cols da tabela pai `{parent_table}`: {list(parent_cols)}.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    _warn_or_raise(
                        f"Relacionamento ignorado em `{name}` FK #{i}: `parent_columns` ausente "
                        f"e não foi possível obter pk_cols do parent_table `{parent_table}`.",
                        policy=relationship_policy,
                    )
                    continue

            # Se parent_table não foi informado, tenta inferir por parent_columns ou columns.
            if not parent_table:
                inferred_parent, inferred_parent_cols, reason = _infer_parent_table_from_config(
                    specs_config,
                    child_table=name,
                    parent_columns=parent_cols,
                    child_fk_columns=cols,
                )

                if inferred_parent and inferred_parent_cols:
                    parent_table = inferred_parent
                    parent_cols = inferred_parent_cols
                    warnings.warn(
                        f"Relacionamento `{name}` FK #{i}: `parent_table` não informado. {reason}.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    _warn_or_raise(
                        f"Relacionamento ignorado em `{name}` FK #{i}: {reason}. "
                        "A tabela será sintetizada sem essa FK.",
                        policy=relationship_policy,
                    )
                    continue

            if parent_cols is None:
                _warn_or_raise(
                    f"Relacionamento ignorado em `{name}` FK #{i}: `parent_columns` não informado "
                    "e não foi possível inferir. A tabela será sintetizada sem essa FK.",
                    policy=relationship_policy,
                )
                continue

            fks.append(
                ForeignKeySpec(
                    columns=tuple(cols),
                    parent_table=parent_table,
                    parent_columns=tuple(parent_cols),
                )
            )

        specs[name] = TableSpec(
            name=name,
            pk_cols=pk_cols,
            foreign_keys=tuple(fks),
            static=bool(cfg.get("static", False)),
            postprocess=postprocess_by_table.get(name),
        )

    return specs


def build_specs_from_config(
    specs_config: Mapping[str, Mapping],
    postprocess_by_table: Optional[Mapping[str, PostProcessor]] = None,
    *,
    relationship_policy: RelationshipPolicy = "warn_and_skip",
) -> Dict[str, TableSpec]:
    return _build_specs_from_config(
        specs_config,
        postprocess_by_table,
        relationship_policy=relationship_policy,
    )


def _normalize_save_path(save_path: str) -> str:
    """
    Normaliza o caminho de saída:
        - expande "~";
        - remove "/" final para evitar caminhos com "//";
        - mantém esquemas remotos (oci://, s3://, hdfs://, dbfs:/) intactos.
    """
    path = str(save_path).strip()

    has_scheme = "://" in path or path.startswith("dbfs:/")

    if not has_scheme:
        path = os.path.expanduser(path)

    while len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    return path


def _is_local_path(path: str) -> bool:
    return "://" not in path and not path.startswith("dbfs:/")


_INVALID_COL_CHARS_PATTERN = re.compile(r"[ ,;{}()\n\t=]")


def _sanitize_columns_for_save(df: DataFrame, table_name: str) -> DataFrame:
    """
    Renomeia colunas com caracteres inválidos para escrita em Parquet
    (espaço, vírgula, ponto-e-vírgula, chaves, parênteses, '=', tab, newline).

    Cada caractere inválido vira "_". Se houver colisão de nomes após o
    rename, adiciona sufixo numérico. Emite warning listando os renames,
    para o rename ficar auditável.
    """
    renames: List[Tuple[str, str]] = []
    new_names: List[str] = []
    used: set = set()

    for col_name in df.columns:
        new_name = _INVALID_COL_CHARS_PATTERN.sub("_", col_name)

        if not new_name.strip():
            new_name = "col"

        base = new_name
        suffix = 1
        while new_name in used:
            new_name = f"{base}_{suffix}"
            suffix += 1

        used.add(new_name)
        new_names.append(new_name)

        if new_name != col_name:
            renames.append((col_name, new_name))

    if not renames:
        return df

    warnings.warn(
        f"Tabela `{table_name}`: colunas renomeadas para gravação por conterem "
        f"caracteres inválidos para Parquet: {renames}.",
        UserWarning,
        stacklevel=2,
    )

    return df.toDF(*new_names)


def _save_hint_for_error(exc: Exception, out_path: str) -> str:
    """
    Gera dica prática conforme o tipo de erro de gravação.
    """
    msg = str(exc)
    hints: List[str] = []

    lowered = msg.lower()

    if "permission" in lowered or "denied" in lowered or "errno 13" in lowered:
        hints.append(
            "Parece falta de permissão de escrita. Caminhos como '/csv/' apontam "
            "para a RAIZ do filesystem; use um caminho relativo (ex.: './csv') ou "
            "absoluto dentro do seu usuário (ex.: '~/csv' ou '/home/usuario/csv')."
        )

    if "invalid character" in lowered or "attribute name" in lowered:
        hints.append(
            "Nome de coluna inválido para o formato. A sanitização automática "
            "deveria ter tratado; verifique se há colunas com caracteres exóticos."
        )

    if "already exists" in lowered:
        hints.append(
            "O destino já existe e o modo de escrita não permitiu sobrescrever."
        )

    if "winutils" in lowered or "hadoop_home" in lowered:
        hints.append(
            "Ambiente Windows sem winutils.exe/HADOOP_HOME configurado. "
            "Configure o winutils compatível com a versão do Hadoop do Spark."
        )

    if not hints:
        hints.append(
            f"Verifique se o diretório pai de `{out_path}` existe e se o processo "
            "do Spark tem permissão de escrita nele."
        )

    return " ".join(hints)


def run_synthesis_from_tables(
    tables: Mapping[str, DataFrame],
    specs_config: Mapping[str, Mapping],
    *,
    n_rows_by_table: Optional[Mapping[str, int]] = None,
    scale_factor: Optional[float] = None,
    seed: int = 42,
    append_after_max_pk: bool = True,
    pk_max_by_table: Optional[Mapping[str, int]] = None,
    engorda_ts: Optional[datetime] = None,
    dt_vencimento_prazo_dias_by_table: Optional[Mapping[str, int]] = None,
    default_dt_vencimento_prazo_dias: int = DEFAULT_DT_VENCIMENTO_PRAZO_DIAS,
    validate_mode: ValidateMode = "full",
    nullable_fk_policy: NullableFkPolicy = "allow_any_null",
    broadcast_fk_counts: bool = False,
    storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    postprocess_by_table: Optional[Mapping[str, PostProcessor]] = None,
    save_path: Optional[str] = None,
    save_format: Literal["csv", "parquet"] = "parquet",
    save_options: Optional[Mapping[str, object]] = None,
    save_single_file: bool = False,
    save_error_policy: SaveErrorPolicy = "warn_and_continue",
    save_mode: str = "overwrite",
    verbose: bool = True,
    relationship_policy: RelationshipPolicy = "warn_and_skip",
    check_relationship_values: bool = True,
    truncate_lineage: bool = False,
) -> Dict[str, DataFrame]:
    """
    Runner para quando os DataFrames já estão carregados.

    v4: agora também aceita save_path/save_format (e opções de gravação),
    com gravação robusta igual à de run_synthesis_from_paths.

    v5: aceita `save_mode` e `oci`. Se `oci` for um dict (ex.: {"auth": "config_file"}),
    configura o conector OCI no Spark antes de gravar. Para Data Flow/resource
    principal já ativo, pode passar oci={"auth": "none"} ou simplesmente omitir.
    """
    specs = _build_specs_from_config(
        specs_config,
        postprocess_by_table,
        relationship_policy=relationship_policy,
    )

    specs = _sanitize_specs_against_known_tables(
        specs,
        tables,
        relationship_policy=relationship_policy,
    )

    specs = _validate_relationship_columns(
        tables,
        specs,
        relationship_policy=relationship_policy,
        nullable_fk_policy=nullable_fk_policy,
        check_relationship_values=check_relationship_values,
    )

    if n_rows_by_table is None:
        effective_n_rows: Dict[str, int] = {}
        for name in specs:
            base = tables[name].count()
            if specs[name].static:
                effective_n_rows[name] = base
            elif scale_factor:
                effective_n_rows[name] = int(round(base * scale_factor))
            else:
                effective_n_rows[name] = base
    else:
        effective_n_rows = dict(n_rows_by_table)

    if verbose:
        print("Ordem topológica:", " -> ".join(_topological_order(specs)))
        print("n_rows_by_table:", effective_n_rows)

    synthetic = synthesize_multitable_spark(
        tables=tables,
        specs=specs,
        n_rows_by_table=effective_n_rows,
        seed=seed,
        append_after_max_pk=append_after_max_pk,
        pk_max_by_table=pk_max_by_table,
        engorda_ts=engorda_ts,
        dt_vencimento_prazo_dias_by_table=dt_vencimento_prazo_dias_by_table,
        default_dt_vencimento_prazo_dias=default_dt_vencimento_prazo_dias,
        validate_mode=validate_mode,
        nullable_fk_policy=nullable_fk_policy,
        broadcast_fk_counts=broadcast_fk_counts,
        storage_level=storage_level,
        verbose=verbose,
        relationship_policy=relationship_policy,
        check_relationship_values=False,
        truncate_lineage=truncate_lineage,
    )

    if save_path:
        save_synthetic_tables(
            synthetic,
            save_path,
            save_format=save_format,
            save_options=save_options,
            save_single_file=save_single_file,
            save_error_policy=save_error_policy,
            save_mode=save_mode,          # <-- ADD THIS LINE (upstream omits it)
            verbose=verbose,
        )

    return synthetic


def save_synthetic_tables(
    synthetic: Mapping[str, DataFrame],
    save_path: str,
    *,
    save_format: Literal["csv", "parquet"] = "parquet",
    save_options: Optional[Mapping[str, object]] = None,
    save_single_file: bool = False,
    save_error_policy: SaveErrorPolicy = "warn_and_continue",
    save_mode: str = "overwrite",
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Grava as tabelas sintéticas em disco de forma resiliente.

    save_mode (NOVO na v5): modo de escrita do Spark por tabela.
        "overwrite" (default) substitui o diretório da tabela.
        "append" acrescenta; "ignore" não grava se já existir; "errorifexists".
        Obs.: o `existing_data_behavior="overwrite_or_ignore"` do pyarrow não tem
        equivalente exato no Spark — o mais próximo de "sobrescrever sempre" é
        "overwrite". Funciona com destinos locais e oci:// igualmente.

    Comportamento:
        - Normaliza o caminho e cria o diretório base se for filesystem local.
        - Sanitiza nomes de coluna inválidos para Parquet (warning auditável).
        - Cada tabela é gravada dentro de try/except: a falha de UMA tabela não
          impede a gravação das demais.
        - Ao final, se houve falhas:
            save_error_policy="warn_and_continue": emite warning com resumo;
            save_error_policy="raise": levanta RuntimeError com resumo.

    Retorna:
        dict {tabela: caminho_gravado} apenas com as tabelas gravadas com sucesso.
    """
    if save_error_policy not in ("warn_and_continue", "raise"):
        raise ValueError(
            "save_error_policy deve ser 'warn_and_continue' ou 'raise'."
        )

    fmt = (save_format or "parquet").lower()
    base_path = _normalize_save_path(save_path)
    options = dict(save_options or {})

    # Em filesystem local, garante que o diretório base exista e detecta
    # problemas de permissão ANTES de disparar jobs Spark.
    if _is_local_path(base_path):
        try:
            os.makedirs(base_path, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Não foi possível criar o diretório de saída `{base_path}`: {exc}. "
                + _save_hint_for_error(exc, base_path)
            ) from exc

    saved: Dict[str, str] = {}
    failures: List[Tuple[str, str, str]] = []  # (tabela, caminho, erro+dica)

    for name, df in synthetic.items():
        out_path = f"{base_path}/{name}"

        try:
            # CSV aceita espaço/parênteses no header; só sanitiza nos formatos
            # que proíbem (parquet/orc/etc.), preservando os nomes originais
            # do metadado na saída CSV.
            df_out = df if fmt == "csv" else _sanitize_columns_for_save(df, name)

            if save_single_file:
                df_out = df_out.coalesce(1)

            writer = df_out.write.mode(save_mode)

            for k, v in options.items():
                writer = writer.option(k, v)

            if fmt == "csv":
                writer.option("header", options.get("header", True)).csv(out_path)
            elif fmt == "parquet":
                writer.parquet(out_path)
            else:
                writer.format(fmt).save(out_path)

            saved[name] = out_path

            if verbose:
                print(f"[salvo] {name} -> {out_path} ({fmt})")

        except Exception as exc:
            hint = _save_hint_for_error(exc, out_path)
            failures.append((name, out_path, f"{exc} | Dica: {hint}"))

            if verbose:
                print(f"[FALHA ao salvar] {name} -> {out_path}: {exc}")

    if failures:
        resumo = "; ".join(
            f"`{name}` em `{path}`: {erro}" for name, path, erro in failures
        )
        mensagem = (
            f"Falha ao gravar {len(failures)} de {len(synthetic)} tabela(s) "
            f"em `{base_path}` (formato {fmt}): {resumo}"
        )

        if save_error_policy == "raise":
            raise RuntimeError(mensagem)

        warnings.warn(mensagem, UserWarning, stacklevel=2)

    elif verbose and saved:
        print(f"Dados sintéticos salvos em: {base_path} ({len(saved)} tabela(s), formato {fmt})")

    return saved



def table_path_name(table: str) -> str:
    return table.split(".", 1)[1] if "." in table else table


def raw_path(config: dict[str, str], table: str) -> str:
    parts = [config["DATAGEN_RAW_BASE_URI"]]
    if config.get("DATAGEN_RAW_PREFIX"):
        parts.append(config["DATAGEN_RAW_PREFIX"])
    parts.append(table_path_name(table))
    return "/".join(parts)


def synthetic_base_path(config: dict[str, str]) -> str:
    base = config["DATAGEN_SYNTHETIC_BASE_URI"]
    prefix = config.get("DATAGEN_SYNTHETIC_PREFIX")
    return f"{base}/{prefix}" if prefix else base


def get_engorda_env() -> dict[str, str]:
    config: dict[str, str] = {}
    missing = []
    for name in REQUIRED_ENV_VARS:
        value = os.environ.get(name)
        if not value:
            missing.append(name)
        else:
            config[name] = value.rstrip("/")
    if missing:
        logger.error("Missing required environment variable(s): %s", ", ".join(missing))
        sys.exit(1)
    config["DATAGEN_RAW_PREFIX"] = os.environ.get("DATAGEN_RAW_PREFIX", "").strip("/")
    config["DATAGEN_SYNTHETIC_PREFIX"] = os.environ.get(
        "DATAGEN_SYNTHETIC_PREFIX", ""
    ).strip("/")
    return config


def normalize_specs(specs: dict) -> dict:
    out: dict = {}
    for raw_name, cfg in specs.items():
        name = table_path_name(str(raw_name))
        if name in out:
            raise ValueError(
                f"Spec key collision after schema stripping: `{raw_name}` reduces to "
                f"`{name}`, which is already present."
            )
        new_cfg = copy.deepcopy(dict(cfg))
        for fk_key in ("foreign_keys", "fks"):
            fks = new_cfg.get(fk_key)
            if not isinstance(fks, (list, tuple)):
                continue
            for fk in fks:
                if isinstance(fk, dict) and fk.get("parent_table"):
                    fk["parent_table"] = table_path_name(str(fk["parent_table"]))
        out[name] = new_cfg
    return out


def connected_components(specs: dict) -> list[list[str]]:
    parent: dict[str, str] = {t: t for t in specs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for table, cfg in specs.items():
        for fk_key in ("foreign_keys", "fks"):
            for fk in cfg.get(fk_key) or []:
                if not isinstance(fk, dict):
                    continue
                p = fk.get("parent_table")
                if p in specs:
                    union(table, p)

    groups: dict[str, list[str]] = {}
    for table in specs:
        groups.setdefault(find(table), []).append(table)
    return [sorted(g) for g in groups.values()]


def _fk_parent_tables(specs: dict) -> set[str]:
    parents: set[str] = set()
    for cfg in specs.values():
        for fk_key in ("foreign_keys", "fks"):
            for fk in cfg.get(fk_key) or []:
                if isinstance(fk, dict) and fk.get("parent_table") in specs:
                    parents.add(fk["parent_table"])
    return parents


def _fk_list(cfg: dict) -> list[dict]:
    fks = cfg.get("foreign_keys")
    if not isinstance(fks, (list, tuple)):
        fks = cfg.get("fks")
    return [fk for fk in (fks or []) if isinstance(fk, dict)]


def _fk_is_whole_pk(pk_cols: list[str], fk: dict) -> bool:
    """True when a FK's columns are exactly the child's primary key.

    These are 1:1 "shared-key" extension tables (e.g. JUROS_FLUTUANTE keyed by
    NUM_CONDICAO_IF, which is also its FK to CONDICAO_IF). The synthesizer's FK
    remap can leave such a column NULL or non-unique; bind_shared_key_children
    rebinds it to distinct parent keys instead.
    """
    cols = fk.get("columns") or []
    return bool(pk_cols) and bool(cols) and sorted(cols) == sorted(pk_cols)


def topo_order_tables(comp_specs: dict) -> list[str]:
    """Order a component's tables so every parent comes before its children.

    Used by referential sampling (sample parents first, then keep only children
    whose FK lands in the sampled parents). Self-references are ignored; cycles
    are broken arbitrarily so the function always returns every table once.
    """
    deps: dict[str, set[str]] = {t: set() for t in comp_specs}
    for table, cfg in comp_specs.items():
        for fk in _fk_list(cfg):
            parent = fk.get("parent_table")
            if parent in comp_specs and parent != table:
                deps[table].add(parent)

    return _toposort_break_cycles(deps)


def effective_n_rows(
    specs: dict, source_counts: dict[str, int], scale_factor: float
) -> dict[str, int]:
    parents = _fk_parent_tables(specs)
    targets: dict[str, int] = {}
    for table, cfg in specs.items():
        count = int(source_counts[table])
        static = bool(cfg.get("static", False))
        override = cfg.get("n_rows")
        if count == 0:
            target = 0
        elif static:
            target = count  # static is terminal; override ignored (see warn in engorda)
        elif override is not None:
            target = int(override)
        else:
            target = int(round(count * scale_factor))
        if not static and count > 0 and table in parents:
            target = max(target, count)  # parent floor: keep_all_source_rows needs target >= count
        targets[table] = target
    return targets


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic relational Parquet from ingested raw Parquet."
    )
    parser.add_argument("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR,
                        help="Global row-count multiplier for non-static tables.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Synthesis seed.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue with remaining components after a failure, "
                             "then exit non-zero.")
    parser.add_argument("--limit", type=positive_int, default=None,
                        help="Sample at most this many rows per table for a fast test run. "
                             "Sampling is referential (parents first, children kept only when "
                             "their FK lands in a sampled parent), so FKs stay consistent — but "
                             "child counts come out smaller than the limit. Omit for full data.")
    parser.add_argument("--pk-offset", type=positive_int, default=None,
                        help="Floor for synthetic PK starts. By default engorda reads each table's "
                             "TRUE max(pk) from the full Parquet and generates PKs as true_max+1, "
                             "etc (safe with --limit, collision-free vs the real table). Pass "
                             "--pk-offset N to start at max(true_max, N) instead, e.g. to reserve "
                             "a band well above all real PKs. FKs are remapped to match.")
    parser.add_argument("--pk-safety-band", type=positive_int, default=None,
                        help="Safety gap added above each table's true max(pk): synthetic PKs "
                             "start at true_max + band + 1. Leaves headroom so the real table can "
                             "grow between the max read and the load without colliding. Default: "
                             "no gap (start right after true_max).")
    parser.add_argument("--dt-vencimento-prazo-dias", type=positive_int, default=None,
                        help="Prazo fixo em dias para DT_VENCIMENTO = data da engorda + X. "
                             "Se omitido, preserva o prazo original da linha quando possível; "
                             "se o prazo original for inválido, usa 365 dias.")
    parser.add_argument("--specs", default=None,
                        help="Override DATAGEN_SPECS_URI (URI of a single specs.json object).")
    return parser.parse_args()


def read_parquet(spark: SparkSession, path: str, limit: int | None = None) -> DataFrame:
    df = spark.read.parquet(path)
    return df.limit(limit) if limit is not None else df


def _aplica_filtro_tipo_if(df: DataFrame) -> DataFrame:
    """Filtra a fonte de síntese para o CDB simplificado (NUM_TIPO_IF == 49).

    Aplica o filtro APENAS quando a coluna NUM_TIPO_IF existe no DataFrame;
    tabelas sem a coluna passam intactas. Linhas com NUM_TIPO_IF NULL são
    descartadas (não casam com == 49), o que é o comportamento desejado.

    Usado somente na leitura da FONTE de síntese (referential_sample e o
    caminho sem --limit de engorda). A leitura de max(pk) em compute_pk_maxes
    é intencionalmente NÃO filtrada: ela precisa do max real da tabela inteira
    para evitar colisão das PKs sintéticas com linhas de outros NUM_TIPO_IF.
    """
    if FILTRO_TIPO_IF_COLUMN in df.columns:
        return df.where(F.col(FILTRO_TIPO_IF_COLUMN) == F.lit(FILTRO_TIPO_IF_VALUE))
    return df


def _read_pk_max(spark, path: str, pk_col: str):
    """max(pk_col) from the full Parquet at `path` (footer-fast with pushdown)."""
    row = read_parquet(spark, path).agg(F.max(F.col(pk_col))).first()
    return row[0] if row is not None else None


def _pk_capacity(spark, path: str, pk_col: str):
    """Largest integer the PK column's type can hold (None for string/unknown)."""
    dt = read_parquet(spark, path).schema[pk_col].dataType
    if isinstance(dt, T.DecimalType):
        int_digits = dt.precision - dt.scale
        return (10 ** int_digits) - 1 if int_digits > 0 else 0
    if isinstance(dt, T.ByteType):
        return 127
    if isinstance(dt, T.ShortType):
        return 32_767
    if isinstance(dt, T.IntegerType):
        return 2**31 - 1
    if isinstance(dt, T.LongType):
        return 2**63 - 1
    if isinstance(dt, T.DoubleType):
        return 2**53
    if isinstance(dt, T.FloatType):
        return 2**24
    return None


def compute_pk_maxes(spark, config, comp_specs, floor: int = 0, band: int = 0,
                     n_rows: dict | None = None) -> dict[str, int]:
    """Per-table starting max for synthetic PKs, read from the FULL Parquet.

    For each non-static numeric-PK table the start is ``max(true_max + band, floor)``:
      - ``true_max`` = max(pk) from the full Parquet (footer-fast with
        spark.sql.parquet.aggregatePushdown=true), so synthetic PKs land above
        the real max even under --limit;
      - ``band`` = safety gap added above true_max, leaving room for the real
        table to grow between this read and the load without colliding;
      - ``floor`` = absolute minimum (the --pk-offset reserved band).

    The band/floor are then CLAMPED to the PK column's domain so a tight type
    (e.g. Decimal(3,0), max 999) can't overflow: start is capped at
    ``capacity - n_rows`` (and never below true_max). A table whose own growth
    already exceeds its PK domain is warned about (mark it static / scale down).

    Tables that are static, PK-less, or whose max is unreadable/non-numeric are
    omitted; the synthesizer falls back to append_after_max on the data.
    """
    n_rows = n_rows or {}
    out: dict[str, int] = {}
    for table, cfg in comp_specs.items():
        if cfg.get("static"):
            continue
        pk_cols = cfg.get("pk_cols") or []
        if not pk_cols:
            continue
        pk_col = pk_cols[-1]  # the synthesizer generates the last PK column
        try:
            # NB: max(pk) é lido da tabela INTEIRA, sem o filtro NUM_TIPO_IF==46,
            # de propósito. As PKs sintéticas precisam ficar acima do max real de
            # TODOS os NUM_TIPO_IF para não colidirem com linhas de produção de
            # outros tipos de IF (ver validate_collision_producao). Filtrar aqui
            # reduziria o max e poderia gerar PKs que colidem com dados reais.
            raw_max = _read_pk_max(spark, raw_path(config, table), pk_col)
            true_max = int(raw_max) if raw_max is not None else None
            cap = _pk_capacity(spark, raw_path(config, table), pk_col)
        except Exception as exc:
            logger.warning("Could not read max(%s) for %s: %s", pk_col, table, exc)
            true_max, cap = None, None
        if true_max is None:
            continue
        start = max(true_max + band, floor)
        if cap is not None:
            headroom = cap - int(n_rows.get(table, 0))  # max start so start + n_rows <= cap
            if headroom < true_max:
                logger.warning(
                    "Table %s: PK domain (max %d) cannot hold %s new row(s) above %d; "
                    "mark it static or reduce scale.",
                    table, cap, n_rows.get(table, 0), true_max)
            elif start > headroom:
                logger.info("Table %s: clamping synthetic PK start %d -> %d to fit PK domain (%d)",
                            table, start, headroom, cap)
            start = max(true_max, min(start, headroom))
        out[table] = start
    return out


def referential_sample(spark, config, comp_specs, limit: int | None) -> dict:
    """Subset referencial pais-antes-de-filhos, mantendo a FK consistente.

    Percorre o componente pais-antes-de-filhos: lê cada tabela, aplica o filtro
    de domínio CDB simplificado (NUM_TIPO_IF == 49) onde a coluna existe, e
    mantém só as linhas-filhas cuja FK cai num pai já subsetado (ou é NULL).

    Isto PROPAGA o filtro de domínio PARA BAIXO na árvore de FK, PELA CHAVE:
    uma filha SEM a coluna NUM_TIPO_IF (ex.: CARTEIRA_COMITENTE) fica restrita
    ao universo CDB porque seu NUM_IF precisa casar com uma linha de
    INSTRUMENTO_FINANCEIRO que sobreviveu ao filtro de tipo no pai. Sem essa
    propagação, o pai encolhe para o tipo 49, a filha mantém TODAS as linhas,
    toda chave-filha vira órfã e null_orphan_fks zera a FK inteira — o sintoma
    "CARTEIRA_COMITENTE.NUM_IF 100% nulo" observado no caminho full antigo.

    limit:
        int  -> também limita cada tabela a `limit` linhas (teste rápido). Os
                conjuntos de chave do pai são pequenos -> broadcast é seguro.
        None -> run COMPLETO: sem cap de linhas. Os conjuntos de chave do pai
                podem ser grandes -> o join de chave NÃO usa F.broadcast (deixa
                o AQE decidir) para evitar OOM no driver.
    """
    order = topo_order_tables(comp_specs)
    sampled: dict = {}
    broadcast_keys = limit is not None
    for table in order:
        # Filtra para o CDB simplificado (NUM_TIPO_IF == 49) ANTES da propagação
        # referencial, para que a consistência de FK seja calculada sobre o
        # subconjunto 49. Tabelas sem a coluna passam intactas.
        df = _aplica_filtro_tipo_if(read_parquet(spark, raw_path(config, table)))
        for fk in _fk_list(comp_specs[table]):
            parent = fk.get("parent_table")
            cols = list(fk.get("columns") or [])
            pcols = list(fk.get("parent_columns") or [])
            if parent == table or parent not in sampled or not cols or len(cols) != len(pcols):
                continue  # self-ref / out-of-component / malformed -> handled by null pass
            keys = (sampled[parent]
                    .select(*[F.col(pc).alias(f"__k{i}") for i, pc in enumerate(pcols)])
                    .dropna().distinct())
            cond = reduce(lambda a, b: a & b,
                          [df[cols[i]] == keys[f"__k{i}"] for i in range(len(cols))])
            # Broadcast só no caminho --limit (chaves pequenas). No full, as
            # chaves distintas de um pai grande estouram o broadcast -> deixa o
            # AQE escolher a estratégia de join.
            keys_side = F.broadcast(keys) if broadcast_keys else keys
            joined = df.join(keys_side, cond, "left")
            all_fk_null = reduce(lambda a, b: a & b, [F.col(c).isNull() for c in cols])
            df = (joined
                  .where(F.col("__k0").isNotNull() | all_fk_null)
                  .drop(*[f"__k{i}" for i in range(len(pcols))]))
        if limit is not None:
            df = df.limit(limit)
        # localCheckpoint (eager) nos dois caminhos: cada filho abaixo referencia
        # sampled[parent], então sem truncar o plano lógico a cadeia de joins se
        # acumula por toda a ordem topológica e todo consumidor downstream
        # (validação de FK, síntese) reanalisa um plano fundo o bastante para
        # OOMar o driver -> "SparkContext shutdown". persist() não ajuda — o
        # analyzer ainda percorre a árvore de plano cacheada; localCheckpoint a
        # substitui por uma folha RDD rasa.
        sampled[table] = df.localCheckpoint(eager=True)
    return sampled


def bind_shared_key_children(synthetic: dict, comp_specs: dict) -> dict:
    """Rebind 1:1 shared-key children (PK == FK) to distinct synthetic parent keys.

    For a table whose primary key IS its FK to a parent (e.g. JUROS_FLUTUANTE /
    RESGATE keyed by NUM_CONDICAO_IF -> CONDICAO_IF), the synthesizer's FK remap
    can leave the column NULL or non-unique — fatal because it's a NOT NULL PK.
    Here we overwrite those columns with a distinct slice of the parent's
    synthetic keys, guaranteeing valid, unique, non-null keys. Child rows beyond
    the number of parent keys are dropped (1:1 cardinality).
    """
    for child, cfg in comp_specs.items():
        child_df = synthetic.get(child)
        if child_df is None:
            continue
        pk_cols = list(cfg.get("pk_cols") or [])
        for fk in _fk_list(cfg):
            cols = list(fk.get("columns") or [])
            pcols = list(fk.get("parent_columns") or [])
            parent = fk.get("parent_table")
            parent_df = synthetic.get(parent)
            if (parent_df is None or parent == child or len(cols) != len(pcols)
                    or not _fk_is_whole_pk(pk_cols, fk)):
                continue

            # Numbering em PARALELO via _with_contiguous_row_id em vez de
            # row_number() sobre Window sem partitionBy. O padrão antigo virava
            # Exchange SinglePartition (sort serial numa única task) e estourava
            # num pai grande tipo CONDICAO_IF; o lado do filho, com
            # orderBy(monotonically_increasing_id()), ainda era NÃO-determinístico
            # (mid é reavaliado por estágio), variando o pareamento entre runs.
            # _with_contiguous_row_id dá id contíguo 0..N-1 bijetivo dos dois
            # lados; o inner join pareia 1:1 e descarta filhos acima do nº de
            # chaves do pai (mesma cardinalidade documentada), sem sort global.
            keys = (parent_df
                    .select(*[F.col(pc).alias(f"__np{i}") for i, pc in enumerate(pcols)])
                    .dropna().distinct())
            keys = _with_contiguous_row_id(keys, "__bind_rn")

            numbered = _with_contiguous_row_id(child_df, "__bind_rn")

            joined = numbered.join(keys, "__bind_rn", "inner")
            for i, c in enumerate(cols):
                joined = joined.withColumn(c, F.col(f"__np{i}"))
            child_df = joined.drop("__bind_rn", *[f"__np{i}" for i in range(len(pcols))])
        synthetic[child] = child_df
    return synthetic


def null_orphan_fks(synthetic: dict, comp_specs: dict) -> dict:
    """Set any FK value with no matching parent PK to NULL (nullable columns).

    Catches references the synthesizer could not remap — self-references,
    relationships it ignored (source orphans / parent absent), and residual
    sampling orphans — so the load doesn't hit a parent-key-not-found violation.

    Whole-PK FKs (every FK col is also a PK col) are handled by
    bind_shared_key_children. For a PARTIAL overlap (some FK cols are PK, some
    aren't) we still neutralize the orphan by nulling only the NON-PK columns:
    under Oracle MATCH SIMPLE a single NULL column disables the composite-FK
    check, and the PK columns stay intact (NOT NULL). Only safe for nullable
    FK columns, which the non-PK ones are.
    """
    for child, cfg in comp_specs.items():
        child_df = synthetic.get(child)
        if child_df is None:
            continue
        pk_set = set(cfg.get("pk_cols") or [])
        for fk in _fk_list(cfg):
            parent = fk.get("parent_table")
            parent_df = synthetic.get(parent)
            cols = list(fk.get("columns") or [])
            pcols = list(fk.get("parent_columns") or [])
            if parent_df is None or not cols or len(cols) != len(pcols):
                continue

            # Colunas da FK que NÃO são PK. Se TODAS forem PK -> território do
            # bind_shared_key_children, pula. Antes, qualquer interseção
            # (set(cols) & pk_set) pulava a FK inteira e deixava órfãos de uma
            # sobreposição PARCIAL passarem direto pro load (ORA-02291). Agora
            # anulamos só as não-PK, o que basta para desligar a checagem.
            nullable_cols = [c for c in cols if c not in pk_set]
            if not nullable_cols:
                continue

            keys = (parent_df
                    .select(*[F.col(pc).alias(f"__pk{i}") for i, pc in enumerate(pcols)])
                    .dropna().distinct()
                    .withColumn("__match", F.lit(True)))
            cond = reduce(lambda a, b: a & b,
                          [child_df[cols[i]] == keys[f"__pk{i}"] for i in range(len(cols))])
            # Sem F.broadcast: as chaves distintas de um pai sintético grande
            # estouram o threshold de broadcast (OOM no driver / fallback
            # silencioso). O AQE ainda faz broadcast automático quando o build
            # side é de fato pequeno.
            joined = child_df.join(keys, cond, "left")
            any_fk_set = reduce(lambda a, b: a | b, [F.col(c).isNotNull() for c in cols])
            is_orphan = F.col("__match").isNull() & any_fk_set
            for c in nullable_cols:
                joined = joined.withColumn(
                    c, F.when(is_orphan, F.lit(None).cast(child_df.schema[c].dataType))
                        .otherwise(F.col(c)))
            child_df = joined.drop("__match", *[f"__pk{i}" for i in range(len(pcols))])
        synthetic[child] = child_df
    return synthetic


def release(*dataframes) -> None:
    for df in dataframes:
        if df is None:
            continue
        try:
            df.unpersist()
        except Exception:
            pass


def _delete_path(spark: SparkSession, path: str) -> None:
    """Recursively delete exactly `path` via the Hadoop FileSystem API.

    Scoped to a single table prefix. Used instead of Spark's mode("overwrite"),
    whose delete-before-write removes the shared parent prefix on the OCI HDFS
    connector and clobbers sibling tables.
    """
    jvm = spark._jvm
    hadoop_conf = spark._jsc.hadoopConfiguration()
    jpath = jvm.org.apache.hadoop.fs.Path(path)
    fs = jpath.getFileSystem(hadoop_conf)
    if fs.exists(jpath):
        fs.delete(jpath, True)


def write_synthetic_table(spark: SparkSession, df: DataFrame, out_path: str) -> None:
    """Write one synthetic table to its own prefix without touching siblings.

    Delete only this table's prefix, then append. Equivalent to per-table
    overwrite, but the destructive step is scoped to exactly `out_path`.
    """
    table_name = out_path.rstrip("/").rsplit("/", 1)[-1]
    df_out = _sanitize_columns_for_save(df, table_name)
    _delete_path(spark, out_path)
    df_out.write.mode("append").parquet(out_path)


def load_specs(spark: SparkSession, specs_uri: str) -> dict:
    records = spark.sparkContext.wholeTextFiles(specs_uri).collect()
    if len(records) != 1:
        raise ValueError(
            f"Expected exactly one specs object at `{specs_uri}`, found {len(records)}. "
            "DATAGEN_SPECS_URI must point at a single specs.json file, not a prefix."
        )
    try:
        parsed = json.loads(records[0][1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"specs.json at `{specs_uri}` is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(f"specs.json at `{specs_uri}` must be a non-empty object.")
    return normalize_specs(parsed)


def engorda(spark, config, specs, scale_factor, seed, continue_on_error,
            limit=None, pk_offset=None, pk_safety_band=None,
            dt_vencimento_prazo_dias=None) -> None:
    components = connected_components(specs)
    save_base = synthetic_base_path(config)
    total = len(components)
    if limit is not None:
        logger.info("Input limit active: reading at most %d row(s) per raw table", limit)
    if pk_offset is not None:
        logger.info("PK offset floor active: synthetic PKs start at >= %d", pk_offset)
    if pk_safety_band is not None:
        logger.info("PK safety band active: synthetic PKs start at true_max + %d", pk_safety_band)
    logger.info("Loaded %d table(s) in %d component(s)", len(specs), total)
    run_started = time.perf_counter()
    failures: list[str] = []
    engorda_ts = _normalize_engorda_ts(None)
    logger.info("Data engorda do run: %s", engorda_ts.strftime("%Y-%m-%d %H:%M:%S"))

    for index, comp in enumerate(sorted(components, key=lambda c: sorted(c)[0]), start=1):
        comp_specs = {t: specs[t] for t in comp}
        label = ",".join(sorted(comp))
        comp_tables = {}
        synthetic = {}
        try:
            started = time.perf_counter()
            if limit is not None:
                # Referential sampling: parent rows first, then keep only children
                # whose FK lands in a sampled parent -> FK-consistent under --limit.
                comp_tables = referential_sample(spark, config, comp_specs, limit)
            else:
                # Run COMPLETO: MESMA propagação referencial do filtro de domínio,
                # agora sem cap de linhas. Filtra INSTRUMENTO_FINANCEIRO por
                # NUM_TIPO_IF==46 e propaga o universo CDB pela árvore de FK, de
                # modo que filhas SEM a coluna (CARTEIRA_COMITENTE, CARTEIRA_
                # PARTICIPANTE, CREDITO, OPERACAO, ...) fiquem restritas ao tipo
                # 46 via NUM_IF. Antes o filtro era por tabela isolada: o pai
                # encolhia, as filhas passavam inteiras e viravam órfãs ->
                # null_orphan_fks zerava a FK (NUM_IF 100% nulo).
                comp_tables = referential_sample(spark, config, comp_specs, None)
            counts = {t: comp_tables[t].count() for t in comp}
            for t in comp:
                if comp_specs[t].get("static") and comp_specs[t].get("n_rows") is not None:
                    logger.warning("Table %s is static; ignoring n_rows override", t)
            n_rows = effective_n_rows(comp_specs, counts, scale_factor)
            logger.info("[%d/%d] Component {%s}: n_rows=%s", index, total, label, n_rows)
            pk_max = compute_pk_maxes(spark, config, comp_specs,
                                      floor=(pk_offset or 0), band=(pk_safety_band or 0),
                                      n_rows=n_rows)
            if pk_max:
                logger.info("[%d/%d] true PK max per table: %s", index, total, pk_max)
            prazo_vencimento_por_tabela: dict[str, int] = {}
            for t, cfg in comp_specs.items():
                cfg_prazo = cfg.get("dt_vencimento_prazo_dias")
                if cfg_prazo is not None:
                    prazo_vencimento_por_tabela[t] = int(cfg_prazo)
                elif dt_vencimento_prazo_dias is not None:
                    prazo_vencimento_por_tabela[t] = int(dt_vencimento_prazo_dias)
            # Synthesize (validate_mode="none": we make FKs load-safe ourselves
            # via null_orphan_fks instead of failing the whole component on an
            # orphan), then write each table with a scoped delete (Spark's
            # overwrite clobbers siblings on the OCI connector).
            synthetic = run_synthesis_from_tables(
                comp_tables, comp_specs,
                n_rows_by_table=n_rows, seed=seed,
                pk_max_by_table=pk_max,
                engorda_ts=engorda_ts,
                dt_vencimento_prazo_dias_by_table=prazo_vencimento_por_tabela,
                validate_mode="none", verbose=False,
                # Under --limit the referential-sample chain makes synthesis plans
                # deep enough to OOM the driver; truncate work lineage via eager
                # localCheckpoint. The full run keeps persist (recomputable).
                truncate_lineage=(limit is not None),
            )
            synthetic = bind_shared_key_children(synthetic, comp_specs)
            synthetic = null_orphan_fks(synthetic, comp_specs)
            for name, df in synthetic.items():
                out_path = f"{save_base}/{name}"
                logger.info("[%d/%d] writing %s -> %s", index, total, name, out_path)
                write_synthetic_table(spark, df, out_path)
            logger.info("[%d/%d] Component {%s} done in %.1fs",
                        index, total, label, time.perf_counter() - started)
        except Exception as exc:
            logger.exception("[%d/%d] Component {%s} failed: %s", index, total, label, exc)
            failures.append(label)
            if not continue_on_error:
                raise
        finally:
            release(*comp_tables.values(), *synthetic.values())
            try:
                spark.catalog.clearCache()
            except Exception:
                pass

    logger.info("Finished: %d/%d component(s) in %.1fs",
                total - len(failures), total, time.perf_counter() - run_started)
    if failures:
        logger.error("Failed component(s): %s", "; ".join(failures))
        sys.exit(1)


# Workload-level Spark settings, independent of cluster shape. Driver/executor
# OCPU, memory and count are tuned via the OCI Data Flow UI, not here.
_STATIC_SPARK_CONF = {
    "spark.sql.parquet.datetimeRebaseModeInWrite": "CORRECTED",
    "spark.sql.parquet.int96RebaseModeInWrite": "CORRECTED",
    # Answer max(pk) from Parquet footer stats (metadata only, no scan) so
    # computing each table's true max PK stays fast even under --limit.
    "spark.sql.parquet.aggregatePushdown": "true",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    # Tolerate long GC pauses on large (fat) executors instead of declaring them
    # lost — losing 1 of few executors triggers expensive recompute cascades.
    "spark.network.timeout": "600s",
    "spark.executor.heartbeatInterval": "30s",
    # Survive transient shuffle-block unavailability (a GC pause makes an executor
    # briefly unreachable) instead of failing the fetch -> with few executors a
    # FetchFailed forces a full map-stage recompute, and 4 of them abort the job.
    "spark.shuffle.io.maxRetries": "10",
    "spark.shuffle.io.retryWait": "15s",
    # Overhead as a fraction of executor memory (Spark 3.3+), so it auto-scales
    # with whatever shape is picked in the Data Flow UI. 0.2 (~20%) suits PySpark
    # + shuffle-heavy work; the 0.1 default gets containers RM-killed at scale.
    # NOTE: do NOT also set the absolute spark.executor.memoryOverhead in the UI
    # — the absolute wins over the factor and would pin overhead to one shape.
    "spark.executor.memoryOverheadFactor": "0.2",
}

# Adaptive Query Execution + shuffle sizing. These are runtime SQL confs, so we
# also re-apply them to an already-active session: on Data Flow the context may
# be created by the platform before this runs, which would ignore builder confs.
_RUNTIME_SPARK_CONF = {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    # AQE coalesces post-shuffle partitions toward this target size, so the
    # high partition count below never lands as giant reducer tasks.
    "spark.sql.adaptive.advisoryPartitionSizeInBytes": "256m",
    # Fewer, larger MAP tasks (default is 128m) -> fewer map outputs. Total
    # shuffle blocks = map_tasks x shuffle.partitions, and a huge block count is
    # what causes FetchFailedException at scale. 512m cuts map tasks ~4x.
    "spark.sql.files.maxPartitionBytes": "512m",
    # Initial/max REDUCE partition count. Balances two forces: large enough that
    # no partition is oversized (AQE only MERGES, never SPLITS outside skewed
    # joins), but not so large that map_tasks x this explodes the block count and
    # triggers FetchFailed. ~0.5-1GB partitions on 128GB executors; AQE coalesces
    # small components back down via the advisory size above.
    "spark.sql.shuffle.partitions": "8000",
}


def create_spark_session(app_name: str) -> SparkSession:
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name)
    for key, value in {**_STATIC_SPARK_CONF, **_RUNTIME_SPARK_CONF}.items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()

    # Guarantee the AQE/shuffle confs apply even if the session pre-existed.
    for key, value in _RUNTIME_SPARK_CONF.items():
        spark.conf.set(key, value)
    return spark


def main() -> None:
    args = parse_arguments()
    config = get_engorda_env()
    spark = create_spark_session("DataGenEngordaTables")
    try:
        specs_uri = args.specs or config["DATAGEN_SPECS_URI"]
        specs = load_specs(spark, specs_uri)
        engorda(spark, config, specs, args.scale_factor, args.seed,
                args.continue_on_error, args.limit, args.pk_offset, args.pk_safety_band,
                args.dt_vencimento_prazo_dias)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
