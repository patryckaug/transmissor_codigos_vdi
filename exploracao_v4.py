"""
synthetic_multitable_spark_v4.py
================================

Gerador de dados sintéticos MULTI-TABELA em PySpark.

Objetivo:
    Gerar dados sintéticos preservando, quando possível:
      - estrutura relacional;
      - primary keys únicas;
      - foreign keys válidas;
      - distribuições por bootstrap de linhas inteiras;
      - relacionamentos entre tabelas via remapeamento old_key -> synthetic_key.

Comportamento importante desta versão:
    - Se uma FK declarada em specs_config/specs NÃO existir de forma estrutural
      ou lógica, o código emite AVISO e IGNORA apenas esse relacionamento.
    - Mesmo com relacionamento inválido/ausente, as tabelas continuam sendo
      sintetizadas.
    - Relacionamentos válidos continuam sendo preservados.

NOVIDADES DA v4 (em relação à v3):
    1. PK numérica não-inteira agora é suportada:
         - DoubleType / FloatType / DecimalType passam a ter estratégia segura
           de geração de PK (sequência inteira exata castada para o tipo
           original, com checagem de limite de representação exata).
         - Isso resolve o erro:
               TypeError: PK `NUM_ID_ENTIDADE` tipo DoubleType() sem estratégia segura.
           que acontece quando o CSV é lido com inferSchema=True e o Spark
           infere colunas de ID como double.
    2. Gravação (save_path) muito mais robusta:
         - Normalização do caminho (remove "/" final, expande "~").
         - Sanitização automática de nomes de coluna inválidos para Parquet
           (espaço, vírgula, ";{}()=\\n\\t"), com warning informando o rename.
         - Escrita por tabela protegida com try/except: uma tabela que falhar
           não derruba as demais; ao final, um resumo de falhas é emitido
           (warning ou exceção, conforme save_error_policy).
         - Mensagens de erro com dicas práticas (ex.: permissão negada ao
           gravar em "/csv/" na raiz do filesystem).
         - Opção save_single_file=True para gerar um único arquivo por tabela
           (coalesce(1)) — útil para CSV de inspeção manual.
    3. Nenhuma mudança na forma de chamar run_synthesis_from_paths /
       run_synthesis_from_tables: todos os parâmetros novos têm default e a
       chamada existente do notebook continua funcionando igual.

Exemplos de relacionamento ignorado com warning:
    - parent_table declarado não está em specs/tables/table_paths;
    - parent_table não foi informado e não pôde ser inferido;
    - coluna FK não existe na tabela filha;
    - parent_column não existe na tabela pai;
    - FK e parent_columns têm tamanhos diferentes;
    - self-reference;
    - FK tem valores órfãos em relação ao pai original;
    - FK não tem nenhum match com a tabela pai.

Observação:
    Este arquivo mantém os nomes das funções já existentes no seu .py.
    Foram adicionados parâmetros opcionais, com default seguro, mas os nomes
    das funções principais foram mantidos.
"""

from __future__ import annotations

import math
import os
import re

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


# ============================================================
# 2. Utilitários de tipo, seed, persistência e warnings
# ============================================================

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


def _persist(df: DataFrame, storage_level: StorageLevel) -> DataFrame:
    return df.persist(storage_level)


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


# ============================================================
# 3. Validação das specs e ordenação topológica
# ============================================================

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


def _topological_order(specs: Mapping[str, TableSpec]) -> List[str]:
    remaining = set(specs.keys())
    done: set = set()
    order: List[str] = []

    while remaining:
        ready = [
            n
            for n in remaining
            if {fk.parent_table for fk in specs[n].foreign_keys}.issubset(done)
        ]

        if not ready:
            unresolved = {
                t: [fk.parent_table for fk in specs[t].foreign_keys]
                for t in remaining
            }
            raise ValueError(f"Ciclo/self-ref/pai ausente. Pendências: {unresolved}")

        for name in sorted(ready):
            order.append(name)
            done.add(name)
            remaining.remove(name)

    return order


def _referenced_parent_columns(specs: Mapping[str, TableSpec]) -> Dict[str, set]:
    refs: Dict[str, set] = {}

    for child_spec in specs.values():
        for fk in child_spec.foreign_keys:
            refs.setdefault(fk.parent_table, set()).add(tuple(fk.parent_columns))

    return refs


# ============================================================
# 4. Indexação e bootstrap
# ============================================================

def _with_contiguous_row_id(df: DataFrame, id_col: str) -> DataFrame:
    """
    Adiciona um identificador contíguo 0..N-1 sem usar RDD/lambda.

    Motivo:
        Algumas combinações de PySpark + Python geram erro em cloudpickle.dumps
        quando usamos df.rdd.zipWithIndex().map(lambda ...), por exemplo:

            IndexError: tuple index out of range
            Could not serialize object

        Esta implementação usa apenas expressões Spark SQL/DataFrame, evitando
        serialização de função Python para os executores.

    Observação:
        Window.orderBy(...) cria uma ordenação global. É mais segura para
        compatibilidade do que a versão RDD, embora possa ser mais custosa em
        tabelas muito grandes.
    """
    ordering_col = f"__{id_col}_order_tmp"

    while ordering_col in df.columns:
        ordering_col = f"_{ordering_col}"

    w = Window.orderBy(F.col(ordering_col))

    return (
        df
        .withColumn(ordering_col, F.monotonically_increasing_id())
        .withColumn(id_col, (F.row_number().over(w) - F.lit(1)).cast("long"))
        .drop(ordering_col)
    )


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


# ============================================================
# 5. Geração de PK
# ============================================================

_INT_TYPE_LIMITS = (
    (T.ByteType, 127),
    (T.ShortType, 32_767),
    (T.IntegerType, 2_147_483_647),
)

# Maior inteiro representável de forma EXATA em cada tipo de ponto flutuante.
# Acima desses limites, inteiros consecutivos colidem (perda de precisão),
# o que quebraria a unicidade da PK.
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
) -> DataFrame:
    dt = _get_field_type(source_cached, pk)

    if _is_integer_type(dt):
        start = (_max_pk_value(source_cached, pk) or 0) + 1 if append_after_max else 1
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
        start = (_max_pk_value(source_cached, pk) or 0) + 1 if append_after_max else 1
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
        start = (_max_pk_value(source_cached, pk) or 0) + 1 if append_after_max else 1
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
) -> DataFrame:
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
            f"PK composta `{spec.name}` última col `{last_pk}` tipo {last_type!r} inseguro. "
            "Tipos suportados: inteiro, double, float, decimal e string."
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


# ============================================================
# 7. Validações de resultado
# ============================================================

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


# Alias público opcional sem mudar o nome interno existente.
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
    relationship_policy: RelationshipPolicy = "warn_and_skip",
    check_relationship_values: bool = True,
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


# ============================================================
# 9. Configuração, leitura e runners genéricos
# ============================================================

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
                        "parent_table": "tabela_pai",          # opcional
                        "parent_columns": ["ID_PAI"]           # opcional se parent_table tiver pk_cols
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
                # Se vier objeto pronto e completo, mantém. Se estiver incompleto, saneamento posterior trata.
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


# Alias público sem alterar o nome interno existente.
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


def _preflight_relationships(
    table_paths: Mapping[str, str],
    specs: Mapping[str, TableSpec],
    *,
    relationship_policy: RelationshipPolicy = "warn_and_skip",
) -> Dict[str, TableSpec]:
    """
    Checagem amigável antes de ler/processar.

    Agora retorna specs saneadas, removendo FKs cujo parent_table não exista.
    Tabelas declaradas sem path continuam sendo erro fatal, pois não há dados
    para sintetizar.
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
            f"`table_paths` contém tabelas sem spec (serão ignoradas): {sobra_caminho}.",
            UserWarning,
            stacklevel=2,
        )

    return _sanitize_specs_against_known_tables(
        specs,
        table_paths,
        relationship_policy=relationship_policy,
    )


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
        return (
            reader
            .option("header", options.get("header", True))
            .option("inferSchema", options.get("inferSchema", True))
            .csv(path)
        )

    return reader.format(fmt).load(path)


# ============================================================
# 9.1 Gravação robusta (NOVO na v4)
# ============================================================

# Caracteres proibidos em nomes de coluna no Parquet (e problemáticos em geral).
_INVALID_COL_CHARS_PATTERN = re.compile(r"[ ,;{}()\n\t=]")


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


def save_synthetic_tables(
    synthetic: Mapping[str, DataFrame],
    save_path: str,
    *,
    save_format: Literal["csv", "parquet"] = "parquet",
    save_options: Optional[Mapping[str, object]] = None,
    save_single_file: bool = False,
    save_error_policy: SaveErrorPolicy = "warn_and_continue",
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Grava as tabelas sintéticas em disco de forma resiliente.

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

            writer = df_out.write.mode("overwrite")

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


def run_synthesis_from_tables(
    tables: Mapping[str, DataFrame],
    specs_config: Mapping[str, Mapping],
    *,
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
    save_options: Optional[Mapping[str, object]] = None,
    save_single_file: bool = False,
    save_error_policy: SaveErrorPolicy = "warn_and_continue",
    verbose: bool = True,
    relationship_policy: RelationshipPolicy = "warn_and_skip",
    check_relationship_values: bool = True,
) -> Dict[str, DataFrame]:
    """
    Runner para quando os DataFrames já estão carregados.

    v4: agora também aceita save_path/save_format (e opções de gravação),
    com gravação robusta igual à de run_synthesis_from_paths.
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
        validate_mode=validate_mode,
        nullable_fk_policy=nullable_fk_policy,
        broadcast_fk_counts=broadcast_fk_counts,
        storage_level=storage_level,
        verbose=verbose,
        relationship_policy=relationship_policy,
        check_relationship_values=False,
    )

    if save_path:
        save_synthetic_tables(
            synthetic,
            save_path,
            save_format=save_format,
            save_options=save_options,
            save_single_file=save_single_file,
            save_error_policy=save_error_policy,
            verbose=verbose,
        )

    return synthetic


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
    save_options: Optional[Mapping[str, object]] = None,
    save_single_file: bool = False,
    save_error_policy: SaveErrorPolicy = "warn_and_continue",
    verbose: bool = True,
    relationship_policy: RelationshipPolicy = "warn_and_skip",
    check_relationship_values: bool = True,
) -> Dict[str, DataFrame]:
    """
    Lê tabelas a partir de table_paths, monta specs a partir de specs_config
    e sintetiza preservando apenas os relacionamentos válidos.

    Se algum relacionamento declarado não existir, ele gera warning e segue.

    v4:
        - PKs do tipo double/float/decimal (caso comum em CSV com
          inferSchema=True) agora têm estratégia segura de geração.
        - Gravação resiliente: caminho normalizado, colunas sanitizadas para
          Parquet, escrita por tabela protegida contra falha individual e
          mensagens de erro com dica prática.
        - Parâmetros novos opcionais: save_options, save_single_file,
          save_error_policy. A chamada antiga continua funcionando igual.
    """
    # 1) dict -> specs tipadas
    specs = _build_specs_from_config(
        specs_config,
        postprocess_by_table,
        relationship_policy=relationship_policy,
    )

    # 2) pré-checagem de relacionamentos por nome de tabela
    specs = _preflight_relationships(
        table_paths,
        specs,
        relationship_policy=relationship_policy,
    )

    # 3) ordem topológica já sem FKs com parent_table ausente
    order = _topological_order(specs)
    if verbose:
        print("Ordem topológica:", " -> ".join(order))

    # 4) leitura — mesmas chaves de specs_config saneado
    tables: Dict[str, DataFrame] = {
        name: _read_table(
            spark,
            table_paths[name],
            default_input_format,
            input_options,
        )
        for name in specs
    }

    # 5) valida/saneia colunas e valores de relacionamento contra schema real
    specs = _validate_relationship_columns(
        tables,
        specs,
        relationship_policy=relationship_policy,
        nullable_fk_policy=nullable_fk_policy,
        check_relationship_values=check_relationship_values,
    )

    # 6) volume por tabela
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
        print("n_rows_by_table:", effective_n_rows)

    # 7) sintetização
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
        relationship_policy=relationship_policy,
        # Já checamos antes; evita repetir joins caros de diagnóstico.
        check_relationship_values=False,
    )

    # 8) gravação opcional (v4: resiliente)
    if save_path:
        save_synthetic_tables(
            synthetic,
            save_path,
            save_format=save_format,
            save_options=save_options,
            save_single_file=save_single_file,
            save_error_policy=save_error_policy,
            verbose=verbose,
        )

    return synthetic
