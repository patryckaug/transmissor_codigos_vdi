"""
specs_from_metadata.py
======================

Reconstrói o `specs_config` (consumido pelo synthetic_multitable_spark_v4 E
pelo sdv_metrics) APENAS a partir do metadado, em pandas puro — sem Spark,
sem JAR oci-hdfs, sem ler os Parquet, sem risco de OOM.

Premissa: o metadado (METADADO_CDB_SIMPLIFICADO_PK_E_FK.csv/.xlsx) é confiável
e está preenchido com PK/FK. A validação de integridade contra os dados é
pulada aqui de propósito — o próprio sintetizador a refaz em tempo de execução
(_sanitize_specs_for_available_relationships com check_relationship_values=True),
ignorando com warning qualquer FK órfã/sem match.

Reaproveita a Camada 1 do builder do colega (build_candidate_specs_from_metadata),
que já é pandas puro, e adiciona a montagem final no formato esperado:

    {
      "TABELA": {
        "pk_cols": ["COL_PK"],
        "foreign_keys": [
          {"columns": ["COL_FK"], "parent_table": "PAI", "parent_columns": ["PK_PAI"]},
          ...
        ],
        # "static": True   # opcional, p/ tabelas estáticas
      },
      ...
    }

Uso
---
    from specs_from_metadata import build_specs_config_from_metadata

    specs_config = build_specs_config_from_metadata(
        meta="METADADO_CDB_SIMPLIFICADO_PK_E_FK.csv",
        static_tables=["TIPO_IF"],   # ajuste conforme suas tabelas estáticas
    )

    # pluga direto no sdv_metrics:
    run_comparison_report(..., specs_config=specs_config, ...)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd


# ============================================================
# 0. Helpers de normalização (idênticos aos do builder do colega)
# ============================================================

_TRUE_TOKENS = {"y", "yes", "s", "sim", "true", "t", "1", "1.0"}


def _truthy(v: Any) -> bool:
    """Interpreta Y/N, S/N, YES/NO, True/False, 1/0 (string ou número) como bool."""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        try:
            return float(v) != 0.0
        except (TypeError, ValueError):
            return False
    return str(v).strip().lower() in _TRUE_TOKENS


def _clean(v: Any) -> Optional[str]:
    """Retorna string limpa ou None (trata NaN/vazio)."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    return s


def read_metadata(path: str) -> pd.DataFrame:
    """Lê o metadado a partir de .csv ou .xlsx/.xls."""
    low = path.lower()
    if low.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    # CSV: tenta utf-8-sig e cai para latin-1 (metadados BR às vezes vêm assim)
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def _normalize_meta(meta: Any) -> pd.DataFrame:
    """Padroniza o DataFrame de metadado (nomes de coluna em UPPER, strings limpas)."""
    if isinstance(meta, str):
        meta = read_metadata(meta)
    if not isinstance(meta, pd.DataFrame):
        raise TypeError("`meta` deve ser um caminho (str) ou um pandas.DataFrame.")

    df = meta.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]

    for required in ("TABLE_NAME", "COLUMN_NAME"):
        if required not in df.columns:
            raise ValueError(
                f"Metadado precisa da coluna `{required}`. "
                f"Colunas vistas: {list(df.columns)}"
            )

    for optional in (
        "IS_PRIMARY_KEY", "IS_FOREIGN_KEY", "FK_REF_TABLE", "FK_REF_COLUMN",
        "COLUMN_ID", "IS_INDEXED", "DATA_TYPE",
    ):
        if optional not in df.columns:
            df[optional] = None

    for str_col in (
        "TABLE_NAME", "COLUMN_NAME", "FK_REF_TABLE", "FK_REF_COLUMN", "DATA_TYPE"
    ):
        df[str_col] = df[str_col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    return df


# ============================================================
# 1. Metadado -> candidatas (Camada 1, pandas puro)
# ============================================================

def build_candidate_specs_from_metadata(meta: Any) -> Dict[str, Dict[str, Any]]:
    """
    Extrai PKs e FKs candidatas do metadado, por tabela.

    Retorna:
        {
          "TABELA": {
             "pk_cols": [...],
             "columns": [...],
             "_fk_candidates": [
                 {"columns": ["COL"], "parent_table": "PAI"|None,
                  "parent_columns": ["REF"]|None}
             ]
          }
        }
    """
    df = _normalize_meta(meta)
    specs: Dict[str, Dict[str, Any]] = {}

    for table in dict.fromkeys(df["TABLE_NAME"].tolist()):
        if not _clean(table):
            continue

        sub = df[df["TABLE_NAME"] == table].copy()
        if "COLUMN_ID" in sub.columns:
            sub = sub.sort_values("COLUMN_ID", na_position="last")
        rows = sub.to_dict("records")

        pk_cols = [
            _clean(r["COLUMN_NAME"])
            for r in rows
            if _truthy(r.get("IS_PRIMARY_KEY")) and _clean(r["COLUMN_NAME"])
        ]

        fk_candidates: List[Dict[str, Any]] = []
        for r in rows:
            if not _truthy(r.get("IS_FOREIGN_KEY")):
                continue
            child_col = _clean(r["COLUMN_NAME"])
            if not child_col:
                continue
            ref_table = _clean(r.get("FK_REF_TABLE"))
            ref_col = _clean(r.get("FK_REF_COLUMN"))
            fk_candidates.append(
                {
                    "columns": [child_col],
                    "parent_table": ref_table,
                    "parent_columns": [ref_col] if ref_col else None,
                }
            )

        specs[table] = {
            "pk_cols": pk_cols,
            "columns": [
                _clean(r["COLUMN_NAME"]) for r in rows if _clean(r["COLUMN_NAME"])
            ],
            "_fk_candidates": fk_candidates,
        }

    return specs


# ============================================================
# 2. Montagem final do specs_config (formato do sintetizador/sdv_metrics)
# ============================================================

def build_specs_config_from_metadata(
    meta: Any,
    *,
    static_tables: Optional[Sequence[str]] = None,
    drop_tables_without_pk: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Monta o specs_config final a partir do metadado, em pandas puro.

    Regras (espelham o que o sintetizador exige):
        - PK é obrigatória. Tabela sem PK no metadado é descartada (com aviso)
          se drop_tables_without_pk=True; caso contrário levanta erro.
        - FK só entra se tiver parent_table E parent_columns. Se o metadado
          marcou IS_FOREIGN_KEY mas não preencheu o pai, a FK é ignorada com
          aviso (o sintetizador não conseguiria remapeá-la mesmo).
        - parent_table que não existe como tabela no metadado -> FK ignorada.
        - Uma coluna que apareça em mais de uma FK -> mantém a primeira, ignora
          as demais (remapeamento ambíguo, igual ao sintetizador).
        - Nomes preservados EXATAMENTE como no metadado (case-sensitive), pois
          precisam casar com as pastas no Object Storage.

    Parâmetros:
        meta           : caminho (.csv/.xlsx) ou pandas.DataFrame do metadado.
        static_tables  : nomes de tabelas a marcar com static=True.
        drop_tables_without_pk : descarta (True) ou erra (False) p/ tabela sem PK.

    Retorna:
        dict specs_config pronto para run_comparison_report / run_synthesis_*.
    """
    static_set = {s.strip() for s in (static_tables or [])}
    candidates = build_candidate_specs_from_metadata(meta)

    if not candidates:
        raise ValueError("Nenhuma tabela encontrada no metadado.")

    known_tables = set(candidates.keys())
    # mapa case-insensitive para resolver parent_table com grafia divergente
    lower_to_real = {t.lower(): t for t in known_tables}

    specs_config: Dict[str, Dict[str, Any]] = {}
    avisos: List[str] = []

    for table, cand in candidates.items():
        pk_cols = [c for c in cand["pk_cols"] if c]

        if not pk_cols:
            msg = f"`{table}`: sem PK no metadado."
            if drop_tables_without_pk:
                avisos.append(msg + " Tabela descartada do specs_config.")
                continue
            raise ValueError(
                msg + " Defina a PK no metadado ou use drop_tables_without_pk=True."
            )

        seen_fk_cols: set = set()
        final_fks: List[Dict[str, Any]] = []

        for fk in cand["_fk_candidates"]:
            cols = [c for c in fk["columns"] if c]
            parent = _clean(fk.get("parent_table"))
            pcols = fk.get("parent_columns")

            if not cols:
                continue

            if not parent:
                avisos.append(
                    f"`{table}`.{cols}: FK marcada no metadado sem FK_REF_TABLE. "
                    "Ignorada."
                )
                continue

            # resolve grafia do pai contra as tabelas conhecidas
            parent_real = parent if parent in known_tables else lower_to_real.get(
                parent.lower()
            )
            if not parent_real:
                avisos.append(
                    f"`{table}`.{cols}: parent_table `{parent}` não existe no "
                    "metadado. FK ignorada."
                )
                continue

            # parent_columns: usa o do metadado; se faltar, cai na PK do pai
            if not pcols:
                parent_pk = [c for c in candidates[parent_real]["pk_cols"] if c]
                if not parent_pk:
                    avisos.append(
                        f"`{table}`.{cols}: FK_REF_COLUMN ausente e pai "
                        f"`{parent_real}` não tem PK. FK ignorada."
                    )
                    continue
                pcols = parent_pk
                avisos.append(
                    f"`{table}`.{cols}: FK_REF_COLUMN ausente; usando PK do pai "
                    f"`{parent_real}` = {pcols}."
                )

            pcols = [c for c in pcols if c]
            if len(cols) != len(pcols):
                avisos.append(
                    f"`{table}`.{cols} -> `{parent_real}`.{pcols}: tamanhos "
                    "diferentes de columns/parent_columns. FK ignorada."
                )
                continue

            if any(c in seen_fk_cols for c in cols):
                avisos.append(
                    f"`{table}`.{cols}: coluna já usada em outra FK "
                    "(remapeamento ambíguo). FK ignorada."
                )
                continue

            seen_fk_cols.update(cols)
            final_fks.append(
                {
                    "columns": cols,
                    "parent_table": parent_real,
                    "parent_columns": pcols,
                }
            )

        entry: Dict[str, Any] = {"pk_cols": pk_cols}
        if final_fks:
            entry["foreign_keys"] = final_fks
        if table in static_set or table.lower() in {s.lower() for s in static_set}:
            entry["static"] = True

        specs_config[table] = entry

    if verbose:
        n_fks = sum(len(v.get("foreign_keys", [])) for v in specs_config.values())
        print(
            f"specs_config montado: {len(specs_config)} tabela(s), "
            f"{n_fks} FK(s)."
        )
        if avisos:
            print(f"\n{len(avisos)} aviso(s):")
            for a in avisos:
                print("  -", a)

    return specs_config


# ============================================================
# 3. Inspeção rápida do specs_config
# ============================================================

def print_specs_config(specs_config: Mapping[str, Mapping[str, Any]]) -> None:
    """Imprime o specs_config de forma legível para conferência."""
    for table, cfg in specs_config.items():
        static = " [STATIC]" if cfg.get("static") else ""
        print(f"\n{table}{static}")
        print(f"  PK: {cfg.get('pk_cols')}")
        for fk in cfg.get("foreign_keys", []):
            print(
                f"  FK: {fk['columns']} -> "
                f"{fk['parent_table']}.{fk['parent_columns']}"
            )


from spec_build_trust_metadata import build_specs_config_from_metadata, print_specs_config

specs_config = build_specs_config_from_metadata(
    meta="METADADO_CDB_SIMPLIFICADO_PK_E_FK.csv",   # ou .xlsx
    static_tables=[...],   # se houver tabelas estáticas; senão omita
)

print_specs_config(specs_config)   # confira PKs e FKs antes de sintetizar
