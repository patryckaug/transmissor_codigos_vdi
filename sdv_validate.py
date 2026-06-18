# -*- coding: utf-8 -*-
"""
sdv_validate.py
===============

Validação ESTRUTURAL e RELACIONAL de dados sintéticos contra os originais —
para decidir, com alta confiança, se a sintetização preservou TIPO DE DADO,
CHAVES PRIMÁRIAS e CHAVES ESTRANGEIRAS.

Dois modos (combináveis)
------------------------
1. DATA-DRIVEN (padrão, specs_config=None): não confia em nenhuma declaração.
   A "verdade" é inferida dos PRÓPRIOS DADOS, dos dois lados:
       Toda propriedade estrutural verdadeira no ORIGINAL tem que continuar
       verdadeira no SINTÉTICO. Qualquer divergência é uma INCONSISTÊNCIA.

2. DECLARADO (specs_config fornecido): valida diretamente as PK/FK que você
   declarou (inclusive COMPOSTAS), marcadas como fonte="declarado". Mesmo aqui
   a declaração é checada contra os dados — uma PK declarada que não é única no
   próprio original, ou uma FK declarada com órfãos no original, vira ALERTA.
   A descoberta de relacionamentos NÃO declarados continua ativa por padrão
   (discover_undeclared=True) — é o ponto cego que a validação por specs_config
   sozinha não cobre.

Concretamente:
  • TIPO   — o dtype real (lido do schema Parquet) de cada coluna do original
             tem que bater com o do sintético.
  • PK     — coluna(s) ÚNICA(S) E NÃO-NULA(S) no original. Declarada (do
             specs_config) ou descoberta. Se quebra no sintético → inconsistência.
  • FK     — relação de INCLUSÃO (valores do filho ⊆ chave do pai). Declarada
             (do specs_config) ou descoberta. Se a inclusão cai do original para
             o sintético (surgem órfãos) → inconsistência.

Limitações honestas
-------------------
  • FK por inclusão DESCOBERTA é heurística (falso positivo de domínios que se
    sobrepõem). FK DECLARADA não tem esse problema (você informou a relação).
  • Amostragem (max_rows) ORFANA relacionamentos artificialmente. Para veredito
    de produção, rode com max_rows=None sobre o dado completo.
  • Este módulo SÓ LÊ. Não escreve nada.

Uso
---
    from sdv_validate import run_structural_validation, display_validation_report

    resultado = run_structural_validation(
        original_path  = "oci://bkt@ns/onprem-export",
        synthetic_path = "oci://bkt@ns/synthetic",
        fmt            = "parquet",
        max_rows       = None,
        specs_config   = specs_config,   # opcional: passa PK/FK declaradas
    )
    resultado["veredito"]            # "APROVADO" | "REPROVADO"
    resultado["inconsistencias"]     # DataFrame com todo problema achado
"""

from __future__ import annotations

import io
import os
import warnings as _warnings
from collections.abc import Mapping as ABCMapping
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ============================================================
# 0. Acesso a OCI Object Storage (somente leitura)
# ============================================================

_OCI_FS = None


def _is_oci(path: str) -> bool:
    return str(path).startswith("oci://")


def _get_oci_fs():
    global _OCI_FS
    if _OCI_FS is None:
        try:
            from ocifs import OCIFileSystem
        except ImportError as exc:
            raise ImportError(
                "Caminho oci:// detectado mas 'ocifs' não está instalado. "
                "Rode: pip install ocifs"
            ) from exc
        with _warnings.catch_warnings():
            _warnings.filterwarnings(
                "ignore", category=FutureWarning, module=r"urllib3.*"
            )
            _warnings.filterwarnings("ignore", message=r".*'strict' parameter.*")
            _OCI_FS = OCIFileSystem()
    return _OCI_FS


def _join_path(base: str, child: str) -> str:
    if _is_oci(base):
        return base.rstrip("/") + "/" + str(child).lstrip("/")
    return os.path.join(os.path.expanduser(str(base).rstrip("/")), str(child))


def _list_tables(base: str, fmt: str) -> List[str]:
    """
    Descobre os NOMES de tabela sob um prefixo (uma "pasta" por tabela).
    Funciona em oci:// (via ocifs) e local. Retorna nomes de primeiro nível.
    """
    fmt = (fmt or "parquet").lower()
    if _is_oci(base):
        fs = _get_oci_fs()
        base_norm = base.rstrip("/")
        try:
            todos = fs.find(base_norm)
        except FileNotFoundError:
            return []
        nativo = base_norm[len("oci://"):]
        nomes = set()
        for f in todos:
            rel = f[len(nativo):].lstrip("/") if f.startswith(nativo) else f
            if "/" in rel:
                nomes.add(rel.split("/")[0])
        return sorted(nomes)
    base_norm = os.path.expanduser(base.rstrip("/"))
    if not os.path.isdir(base_norm):
        return []
    return sorted(
        d for d in os.listdir(base_norm)
        if os.path.isdir(os.path.join(base_norm, d))
    )


# ============================================================
# 1. Leitura com dtypes REAIS preservados
# ============================================================

def _read_parquet_parts(path: str, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    """Lê Parquet (pasta de parts OU arquivo único) preservando dtypes nativos."""
    if _is_oci(path):
        fs = _get_oci_fs()
        path_n = str(path).rstrip("/")
        try:
            enc = fs.find(path_n)
        except FileNotFoundError:
            enc = []
        parts = [f for f in enc if f.lower().endswith(".parquet")]
        if not parts and (path_n.lower().endswith(".parquet") or fs.isfile(path_n)):
            parts = [path_n]
        if not parts:
            raise FileNotFoundError(f"Nenhum .parquet em: {path_n}")
        dfs, acc = [], 0
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=FutureWarning, module=r"urllib3.*")
            _warnings.filterwarnings("ignore", message=r".*'strict' parameter.*")
            for f in sorted(parts):
                part = pd.read_parquet(io.BytesIO(fs.cat_file(f)))
                if max_rows:
                    falta = max_rows - acc
                    if falta <= 0:
                        break
                    if len(part) > falta:
                        part = part.sample(n=falta, random_state=seed)
                dfs.append(part)
                acc += len(part)
                if max_rows and acc >= max_rows:
                    break
        return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    p = os.path.expanduser(str(path).rstrip("/"))
    df = pd.read_parquet(p)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df


def _read_table(path: str, fmt: str, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    fmt = (fmt or "parquet").lower()
    if fmt != "parquet":
        raise ValueError(
            "Esta validação estrutural exige Parquet (carrega os dtypes reais). "
            "CSV perde tipagem (tudo vira string/inferido) e tornaria a checagem "
            "de TIPO não-confiável. Converta para Parquet antes de validar."
        )
    return _read_parquet_parts(path, max_rows, seed)


# ============================================================
# 2. Normalização de dtype para comparação justa
# ============================================================

def _canonical_dtype(series: pd.Series) -> str:
    """
    Reduz o dtype a uma classe canônica comparável, tolerando diferenças
    cosméticas que NÃO são erro de sintetização:
        int8/16/32/64/Int64 -> "integer"; float16/32/64 -> "float";
        bool/boolean -> "boolean"; datetime64[*] -> "datetime";
        object/string/category -> "string".
    Mantém a distinção que IMPORTA: integer != float != string != datetime !=
    boolean. Um ID que era integer e virou string é flagado.
    """
    dt = series.dtype
    if pd.api.types.is_bool_dtype(dt):
        return "boolean"
    if pd.api.types.is_integer_dtype(dt):
        return "integer"
    if pd.api.types.is_float_dtype(dt):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(dt):
        return "datetime"
    return "string"


# ============================================================
# 3. Descoberta de PK candidata (única + não-nula)
# ============================================================

def _is_unique_notnull(df: pd.DataFrame, cols: List[str]) -> Tuple[bool, int, int]:
    """Retorna (é_pk, n_nulos, n_duplicados) para um conjunto de colunas."""
    n = len(df)
    if n == 0:
        return False, 0, 0
    sub = df[cols]
    nulos = int(sub.isna().any(axis=1).sum())
    distintos = int(len(sub.drop_duplicates()))
    dup = n - distintos
    return (nulos == 0 and dup == 0), nulos, dup


def _discover_single_col_pks(df: pd.DataFrame) -> List[str]:
    """Colunas que sozinhas são únicas e não-nulas no DataFrame dado."""
    pks = []
    for c in df.columns:
        ok, _, _ = _is_unique_notnull(df, [c])
        if ok:
            pks.append(c)
    return pks


# ============================================================
# 4. Descoberta de FK candidata por INCLUSÃO de valores
# ============================================================

def _value_set(series: pd.Series) -> set:
    """Conjunto de valores não-nulos, normalizado para string."""
    s = series.dropna()
    if len(s) == 0:
        return set()
    return set(s.astype(str).unique())


def _value_set_cols(df: pd.DataFrame, cols: List[str]) -> Optional[set]:
    """
    Conjunto de valores de uma ou mais colunas (chave composta), como string.
    Linhas com qualquer coluna nula são descartadas (chave parcial não conta).
    Retorna None se alguma coluna não existir (caller decide o que fazer).
    """
    if any(c not in df.columns for c in cols):
        return None
    sub = df[cols].dropna()
    if len(sub) == 0:
        return set()
    if len(cols) == 1:
        return set(sub[cols[0]].astype(str).unique())
    return set(sub.astype(str).agg("||".join, axis=1).unique())


def _inclusion_ratio(child_vals: set, parent_vals: set) -> float:
    """% dos valores distintos do filho que existem no pai. 1.0 = inclusão total."""
    if not child_vals:
        return np.nan
    contidos = len(child_vals & parent_vals)
    return contidos / len(child_vals)


def _fk_inclusion(
    child_df: pd.DataFrame, child_cols: List[str],
    parent_df: pd.DataFrame, parent_cols: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Inclusão de uma FK (simples OU composta) entre dois DataFrames.
    Retorna dict(incl, orphans, n_child, coverage) ou None se faltar coluna.
    """
    cvals = _value_set_cols(child_df, child_cols)
    pvals = _value_set_cols(parent_df, parent_cols)
    if cvals is None or pvals is None:
        return None
    if not cvals:
        return {"incl": np.nan, "orphans": 0, "n_child": 0, "coverage": 0.0}
    inter = len(cvals & pvals)
    return {
        "incl": inter / len(cvals),
        "orphans": len(cvals - pvals),
        "n_child": len(cvals),
        "coverage": (inter / len(pvals)) if pvals else 0.0,
    }


def _discover_inclusions(
    tables: Dict[str, pd.DataFrame],
    pk_candidates: Dict[str, List[str]],
    *,
    min_inclusion: float = 1.0,
    min_distinct_child: int = 5,
    min_parent_coverage: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Descobre FKs candidatas: para cada coluna (filho), testa se seus valores
    estão contidos nos de uma PK candidata de OUTRA tabela (pai). Conservador:
    pai = coluna única+não-nula no original. Marca 'confianca' para revisão.
    """
    candidatas: List[Dict[str, Any]] = []
    parent_sets: Dict[Tuple[str, str], set] = {}
    for ptab, pcols in pk_candidates.items():
        for pcol in pcols:
            parent_sets[(ptab, pcol)] = _value_set(tables[ptab][pcol])

    own_pks = {t: set(cols) for t, cols in pk_candidates.items()}

    for child, cdf in tables.items():
        for ccol in cdf.columns:
            if ccol in own_pks.get(child, set()):
                continue
            cvals = _value_set(cdf[ccol])
            if len(cvals) < min_distinct_child:
                continue
            for (ptab, pcol), pvals in parent_sets.items():
                if ptab == child and pcol == ccol:
                    continue
                if not pvals:
                    continue
                ratio = _inclusion_ratio(cvals, pvals)
                if ratio is np.nan or ratio < min_inclusion:
                    continue
                cobertura = len(cvals & pvals) / len(pvals) if pvals else 0.0
                if cobertura < min_parent_coverage:
                    continue
                if cobertura >= 0.5 and len(cvals) >= 20:
                    conf = "alta"
                elif cobertura >= 0.2:
                    conf = "media"
                else:
                    conf = "baixa"
                candidatas.append({
                    "child_table": child, "child_col": ccol,
                    "parent_table": ptab, "parent_col": pcol,
                    "incl_original": round(float(ratio), 4),
                    "n_distinct_child": len(cvals),
                    "parent_coverage": round(float(cobertura), 4),
                    "confianca": conf,
                })
    return candidatas


# ============================================================
# 4b. Leitura de PK/FK declaradas (specs_config) — opcional
# ============================================================

def _norm_cols(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, (list, tuple)):
        return [str(c).strip() for c in value if str(c).strip()]
    return []


def _pks_from_config(specs_config: Mapping[str, Mapping]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name, cfg in specs_config.items():
        if isinstance(cfg, ABCMapping):
            pk = _norm_cols(cfg.get("pk_cols"))
            if pk:
                out[str(name)] = pk
    return out


def _fks_from_config(specs_config: Mapping[str, Mapping]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name, cfg in specs_config.items():
        if not isinstance(cfg, ABCMapping):
            continue
        raw = cfg.get("foreign_keys") or cfg.get("fks") or []
        if isinstance(raw, ABCMapping):
            raw = [raw]
        for fk in (raw or []):
            if not isinstance(fk, ABCMapping):
                continue
            cols  = _norm_cols(fk.get("columns"))
            ptab  = str(fk.get("parent_table")).strip() if fk.get("parent_table") else ""
            pcols = _norm_cols(fk.get("parent_columns"))
            if not pcols and ptab and isinstance(specs_config.get(ptab), ABCMapping):
                pcols = _norm_cols(specs_config[ptab].get("pk_cols"))
            if cols and ptab and pcols and len(cols) == len(pcols):
                out.append({
                    "child_table": str(name), "child_cols": cols,
                    "parent_table": ptab, "parent_cols": pcols,
                })
    return out


# ============================================================
# 5. Validação — compara estrutura ORIGINAL vs SINTÉTICO
# ============================================================

def run_structural_validation(
    *,
    original_path: "Union[str, Mapping[str, str]]",
    synthetic_path: "Union[str, Mapping[str, str]]",
    fmt: str = "parquet",
    max_rows: Optional[int] = None,
    fk_max_cols: int = 1,
    min_inclusion: float = 1.0,
    min_distinct_child: int = 5,
    min_parent_coverage: float = 0.05,
    specs_config: Optional[Mapping[str, Mapping]] = None,
    discover_undeclared: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Valida estrutura (tipo) e relacionamentos (PK/FK) do sintético contra o
    original. Retorna dict com veredito e DataFrames.

    Parâmetros (novos)
    ------------------
    specs_config        : opcional. Se fornecido, valida as PK/FK DECLARADAS
                          (inclusive compostas), marcadas fonte="declarado".
                          Sem ele, o módulo segue 100% data-driven.
    discover_undeclared : se True (padrão), também descobre PK/FK pelos dados
                          (fonte="descoberto"), mesmo com specs_config — para
                          pegar relacionamentos não declarados. Ponha False para
                          validar SOMENTE o que você declarou.

    fk_max_cols : 1 = descoberta só de FKs de coluna única (composta DESCOBERTA
                  não implementada). FK DECLARADA composta funciona normalmente.

    Retorna
    -------
    dict com: veredito, inconsistencias, tipos, pks, fks, tabelas, amostrado,
              n_criticos. As tabelas pks/fks têm coluna 'fonte' (declarado|descoberto).
    """
    if fk_max_cols != 1 and discover_undeclared:
        raise NotImplementedError(
            "fk_max_cols>1 (descoberta de FK composta) não implementado. "
            "FK composta DECLARADA (via specs_config) funciona; para descoberta, "
            "mantenha fk_max_cols=1."
        )

    # ── 1. descobrir e carregar tabelas (dos dois lados) ──
    if isinstance(original_path, str):
        orig_paths = {t: _join_path(original_path, t) for t in _list_tables(original_path, fmt)}
    else:
        orig_paths = dict(original_path)
    if isinstance(synthetic_path, str):
        synth_paths = {t: _join_path(synthetic_path, t) for t in _list_tables(synthetic_path, fmt)}
    else:
        synth_paths = dict(synthetic_path)

    if verbose:
        print(f"Tabelas no original:  {sorted(orig_paths)}")
        print(f"Tabelas no sintético: {sorted(synth_paths)}")
        if specs_config:
            print(f"Modo DECLARADO ativo (specs_config) | descoberta={'ON' if discover_undeclared else 'OFF'}")

    inconsist: List[Dict[str, Any]] = []

    # ── 2. presença de tabelas ──
    so, ss = set(orig_paths), set(synth_paths)
    tab_rows: List[Dict[str, Any]] = []
    for t in sorted(so | ss):
        no_orig, no_synth = t in so, t in ss
        if no_orig and not no_synth:
            inconsist.append({"severidade": "CRÍTICO", "tipo": "TABELA_AUSENTE",
                              "objeto": t, "detalhe": "tabela existe no original mas NÃO no sintético"})
        elif no_synth and not no_orig:
            inconsist.append({"severidade": "ALERTA", "tipo": "TABELA_EXTRA",
                              "objeto": t, "detalhe": "tabela existe no sintético mas NÃO no original"})
        tab_rows.append({"tabela": t, "no_original": no_orig, "no_sintetico": no_synth})

    comuns = sorted(so & ss)
    if not comuns:
        return {
            "veredito": "REPROVADO", "inconsistencias": pd.DataFrame(inconsist),
            "tipos": pd.DataFrame(), "pks": pd.DataFrame(), "fks": pd.DataFrame(),
            "tabelas": pd.DataFrame(tab_rows), "amostrado": bool(max_rows), "n_criticos": 0,
        }

    real: Dict[str, pd.DataFrame] = {}
    synth: Dict[str, pd.DataFrame] = {}
    for t in comuns:
        real[t]  = _read_table(orig_paths[t], fmt, max_rows, seed)
        synth[t] = _read_table(synth_paths[t], fmt, max_rows, seed)
        if verbose:
            print(f"  {t}: original {len(real[t]):,}×{len(real[t].columns)} | "
                  f"sintético {len(synth[t]):,}×{len(synth[t].columns)}")

    # ── 3. TIPO de dado por coluna ──
    tipo_rows: List[Dict[str, Any]] = []
    for t in comuns:
        rcols, scols = list(real[t].columns), list(synth[t].columns)
        for c in rcols:
            if c not in scols:
                inconsist.append({"severidade": "CRÍTICO", "tipo": "COLUNA_AUSENTE",
                                  "objeto": f"{t}.{c}", "detalhe": "coluna existe no original mas NÃO no sintético"})
                tipo_rows.append({"tabela": t, "coluna": c, "tipo_original": _canonical_dtype(real[t][c]),
                                  "tipo_sintetico": "—", "status": "AUSENTE"})
                continue
            to = _canonical_dtype(real[t][c])
            ts = _canonical_dtype(synth[t][c])
            ok = (to == ts)
            if not ok:
                inconsist.append({"severidade": "CRÍTICO", "tipo": "TIPO_DIVERGENTE",
                                  "objeto": f"{t}.{c}", "detalhe": f"tipo original={to} mas sintético={ts}"})
            tipo_rows.append({"tabela": t, "coluna": c, "tipo_original": to,
                              "tipo_sintetico": ts, "status": "OK" if ok else "DIVERGENTE"})
        for c in scols:
            if c not in rcols:
                inconsist.append({"severidade": "ALERTA", "tipo": "COLUNA_EXTRA",
                                  "objeto": f"{t}.{c}", "detalhe": "coluna existe no sintético mas NÃO no original"})
                tipo_rows.append({"tabela": t, "coluna": c, "tipo_original": "—",
                                  "tipo_sintetico": _canonical_dtype(synth[t][c]), "status": "EXTRA"})

    # ── 4. PK: declarada (se specs_config) + descoberta no ORIGINAL ──
    pk_candidates_orig: Dict[str, List[str]] = {t: _discover_single_col_pks(real[t]) for t in comuns}
    declared_pks = _pks_from_config(specs_config) if specs_config else {}
    pk_rows: List[Dict[str, Any]] = []
    seen_pk: set = set()

    def _registra_pk(t: str, cols: List[str], fonte: str) -> None:
        chave = (t, ",".join(cols))
        if chave in seen_pk:
            return
        seen_pk.add(chave)
        existe_orig = all(c in real[t].columns for c in cols)
        ok_o = _is_unique_notnull(real[t], cols)[0] if existe_orig else False
        if any(c not in synth[t].columns for c in cols):
            pk_rows.append({"tabela": t, "coluna_pk": ",".join(cols), "fonte": fonte,
                            "pk_no_original": ok_o, "pk_no_sintetico": False,
                            "nulos_synth": -1, "duplicados_synth": -1, "status": "QUEBRADA"})
            return
        ok_s, nulos_s, dup_s = _is_unique_notnull(synth[t], cols)
        if not ok_s and ok_o:  # só é "quebra" se ERA PK no original
            inconsist.append({"severidade": "CRÍTICO", "tipo": "PK_QUEBRADA",
                              "objeto": f"{t}.{','.join(cols)}",
                              "detalhe": (f"é PK ({fonte}) no original (única+não-nula) mas no "
                                          f"sintético tem {nulos_s} nulo(s) e {dup_s} duplicado(s)")})
        pk_rows.append({"tabela": t, "coluna_pk": ",".join(cols), "fonte": fonte,
                        "pk_no_original": ok_o, "pk_no_sintetico": ok_s,
                        "nulos_synth": nulos_s, "duplicados_synth": dup_s,
                        "status": "OK" if ok_s else "QUEBRADA"})

    # 4a. PKs DECLARADAS (checadas também no original como sanidade)
    for t, cols in declared_pks.items():
        if t not in comuns:
            continue
        if any(c not in real[t].columns for c in cols):
            inconsist.append({"severidade": "ALERTA", "tipo": "PK_DECLARADA_COLUNA_AUSENTE",
                              "objeto": f"{t}.{','.join(cols)}",
                              "detalhe": "PK declarada referencia coluna inexistente no original"})
            continue
        ok_o, nul_o, dup_o = _is_unique_notnull(real[t], cols)
        if not ok_o:
            inconsist.append({"severidade": "ALERTA", "tipo": "PK_DECLARADA_INVALIDA_ORIGINAL",
                              "objeto": f"{t}.{','.join(cols)}",
                              "detalhe": (f"PK declarada NÃO é única+não-nula no próprio original "
                                          f"({nul_o} nulo(s), {dup_o} duplicado(s)); declaração ou "
                                          "dado de origem inconsistente")})
        _registra_pk(t, cols, "declarado")

    # 4b. PKs DESCOBERTAS (não declaradas)
    if discover_undeclared:
        for t in comuns:
            for c in pk_candidates_orig[t]:
                _registra_pk(t, [c], "descoberto")

    # ── 5. FK: declarada (se specs_config) + descoberta por inclusão ──
    fk_rows: List[Dict[str, Any]] = []
    declared_fk_keys: set = set()

    # 5a. FKs DECLARADAS (inclusão no original e no sintético; composta OK)
    if specs_config:
        for d in _fks_from_config(specs_config):
            child, ccols = d["child_table"], d["child_cols"]
            ptab,  pcols = d["parent_table"], d["parent_cols"]
            label = f"{child}.{','.join(ccols)} -> {ptab}.{','.join(pcols)}"
            if child not in comuns or ptab not in comuns:
                inconsist.append({"severidade": "ALERTA", "tipo": "FK_DECLARADA_TABELA_AUSENTE",
                                  "objeto": label, "detalhe": "FK declarada referencia tabela ausente nos dados"})
                continue
            base = _fk_inclusion(real[child], ccols, real[ptab], pcols)
            synv = _fk_inclusion(synth[child], ccols, synth[ptab], pcols)
            if base is None or synv is None:
                inconsist.append({"severidade": "ALERTA", "tipo": "FK_DECLARADA_COLUNA_AUSENTE",
                                  "objeto": label, "detalhe": "FK declarada referencia coluna ausente em um dos lados"})
                continue
            incl_o, incl_s = base["incl"], synv["incl"]
            if not np.isnan(incl_o) and incl_o < 1.0 - 1e-9:
                inconsist.append({"severidade": "ALERTA", "tipo": "FK_DECLARADA_ORFAOS_ORIGINAL",
                                  "objeto": label,
                                  "detalhe": (f"FK declarada já tem inclusão {incl_o:.4f} no próprio "
                                              f"original ({base['orphans']} órfão(s)); pode ser FK "
                                              "nullable ou declaração imprecisa")})
            quebrou = bool(np.isnan(incl_s)) or (not np.isnan(incl_o) and incl_s < incl_o - 1e-9)
            if quebrou:
                _is = f"{incl_s:.4f}" if not np.isnan(incl_s) else "nan"
                _io = f"{incl_o:.4f}" if not np.isnan(incl_o) else "nan"
                inconsist.append({"severidade": "CRÍTICO", "tipo": "FK_DEGRADADA",
                                  "objeto": label,
                                  "detalhe": f"inclusão caiu de {_io} (original) para {_is} (sintético): {synv['orphans']} órfão(s)"})
            fk_rows.append({
                "fk": label,
                "incl_original": (round(float(incl_o), 4) if not np.isnan(incl_o) else np.nan),
                "incl_sintetico": (round(float(incl_s), 4) if not np.isnan(incl_s) else np.nan),
                "valores_orfaos_synth": synv["orphans"],
                "n_distinct_child": synv["n_child"],
                "parent_coverage": round(float(synv["coverage"]), 4),
                "confianca": "declarado", "fonte": "declarado",
                "status": "DEGRADADA" if quebrou else "OK",
            })
            declared_fk_keys.add((child, tuple(ccols), ptab, tuple(pcols)))

    # 5b. FKs DESCOBERTAS por inclusão (não declaradas)
    if discover_undeclared:
        for inc in _discover_inclusions(real, pk_candidates_orig,
                                        min_inclusion=min_inclusion,
                                        min_distinct_child=min_distinct_child,
                                        min_parent_coverage=min_parent_coverage):
            child, ccol = inc["child_table"], inc["child_col"]
            ptab, pcol  = inc["parent_table"], inc["parent_col"]
            if (child, (ccol,), ptab, (pcol,)) in declared_fk_keys:
                continue
            if ccol not in synth.get(child, pd.DataFrame()).columns:
                continue
            if pcol not in synth.get(ptab, pd.DataFrame()).columns:
                continue
            cvals_s = _value_set(synth[child][ccol])
            pvals_s = _value_set(synth[ptab][pcol])
            ratio_s = _inclusion_ratio(cvals_s, pvals_s)
            orfaos = (len(cvals_s - pvals_s) if cvals_s else 0)
            quebrou = (ratio_s is np.nan) or (ratio_s < inc["incl_original"] - 1e-9)
            if quebrou:
                inconsist.append({"severidade": "CRÍTICO", "tipo": "FK_DEGRADADA",
                                  "objeto": f"{child}.{ccol} -> {ptab}.{pcol}",
                                  "detalhe": (f"inclusão caiu de {inc['incl_original']:.4f} (original) "
                                              f"para {ratio_s if ratio_s is not np.nan else float('nan'):.4f} "
                                              f"(sintético): {orfaos} órfão(s)")})
            fk_rows.append({
                "fk": f"{child}.{ccol} -> {ptab}.{pcol}",
                "incl_original": inc["incl_original"],
                "incl_sintetico": (round(float(ratio_s), 4) if ratio_s is not np.nan else np.nan),
                "valores_orfaos_synth": orfaos,
                "n_distinct_child": inc["n_distinct_child"],
                "parent_coverage": inc["parent_coverage"],
                "confianca": inc["confianca"], "fonte": "descoberto",
                "status": "DEGRADADA" if quebrou else "OK",
            })

    # ── 6. veredito ──
    df_inc = pd.DataFrame(inconsist)
    criticos = int((df_inc["severidade"] == "CRÍTICO").sum()) if len(df_inc) else 0
    veredito = "REPROVADO" if criticos > 0 else "APROVADO"

    return {
        "veredito": veredito,
        "inconsistencias": df_inc,
        "tipos": pd.DataFrame(tipo_rows),
        "pks": pd.DataFrame(pk_rows),
        "fks": pd.DataFrame(fk_rows),
        "tabelas": pd.DataFrame(tab_rows),
        "amostrado": bool(max_rows),
        "n_criticos": criticos,
    }


# ============================================================
# 6. Exibição inline no notebook
# ============================================================

def display_validation_report(resultado: Mapping[str, Any]) -> None:
    """Renderiza o resultado de run_structural_validation inline no Jupyter."""
    from IPython.display import display, Markdown

    _NAVY, _GREEN, _RED, _AMBER = "#1E2761", "#15803D", "#B91C1C", "#B45309"

    def _sec(t, e=""):
        display(Markdown(f"---\n### {e} {t}"))

    def _style(df, status_col="status"):
        if df is None or not len(df):
            return None
        sty = df.style.set_table_styles([
            {"selector": "th", "props": f"background:{_NAVY};color:white;padding:6px 8px;text-align:left"},
            {"selector": "td", "props": "padding:5px 8px"},
        ])
        if status_col and status_col in df.columns:
            def _bg(v):
                up = str(v).upper()
                c = _GREEN if up == "OK" else (_AMBER if up in ("EXTRA",) else _RED)
                return f"background-color:{c};color:white;font-weight:bold"
            sty = sty.map(_bg, subset=[status_col])
        return sty

    display(Markdown("# 🔒 Validação Estrutural & Relacional"))

    veredito = resultado["veredito"]
    cor = _GREEN if veredito == "APROVADO" else _RED
    n_crit = resultado.get("n_criticos", 0)
    display(Markdown(
        f"<div style='background:{cor};color:white;padding:14px 18px;"
        f"border-radius:8px;font-size:20px;font-weight:bold'>"
        f"VEREDITO: {veredito} &nbsp;·&nbsp; {n_crit} inconsistência(s) crítica(s)</div>"
    ))

    if resultado.get("amostrado"):
        display(Markdown(
            "> ⚠️ **AMOSTRADO (max_rows definido).** O veredito de PK/FK é "
            "**INDICATIVO, não conclusivo**: amostrar tabelas independentemente "
            "orfana FKs que existem no dado completo. **Para decidir o append em "
            "produção, rode com `max_rows=None`.**"
        ))

    _sec("Inconsistências encontradas", "🚨")
    df_inc = resultado["inconsistencias"]
    if df_inc is not None and len(df_inc):
        ordem = {"CRÍTICO": 0, "ALERTA": 1}
        df_inc = df_inc.assign(_o=df_inc["severidade"].map(ordem)).sort_values(["_o", "tipo"]).drop(columns="_o")
        def _bg_sev(v):
            c = _RED if str(v) == "CRÍTICO" else _AMBER
            return f"background-color:{c};color:white;font-weight:bold"
        sty = df_inc.style.set_table_styles([
            {"selector": "th", "props": f"background:{_NAVY};color:white;padding:6px 8px;text-align:left"},
            {"selector": "td", "props": "padding:5px 8px"},
        ]).map(_bg_sev, subset=["severidade"])
        display(sty)
    else:
        display(Markdown(
            f"<span style='color:{_GREEN};font-weight:bold'>Nenhuma inconsistência. "
            "Estrutura e relacionamentos (declarados e/ou inferidos) foram "
            "preservados no sintético.</span>"
        ))

    _sec("Tipos de dado (original vs sintético)", "🧬")
    df_t = resultado["tipos"]
    if df_t is not None and len(df_t):
        ruins = df_t[df_t["status"] != "OK"]
        display(Markdown(f"{len(df_t)} coluna(s) comparada(s) · **{len(ruins)}** divergente(s)/ausente(s)/extra(s)."))
        display(_style(ruins if len(ruins) else df_t, "status"))
    else:
        display(Markdown("_Sem colunas comparadas._"))

    _sec("Chaves primárias (declaradas + descobertas)", "🔑")
    df_pk = resultado["pks"]
    if df_pk is not None and len(df_pk):
        display(Markdown(
            "`fonte`: **declarado** (do specs_config) ou **descoberto** (única+"
            "não-nula no original). `pk_no_sintetico=False` = o sintetizador "
            "**quebrou** uma chave que valia no original."
        ))
        display(_style(df_pk, "status"))
    else:
        display(Markdown("_Nenhuma PK detectada/declarada._"))

    _sec("Chaves estrangeiras (declaradas + descobertas por inclusão)", "🔗")
    df_fk = resultado["fks"]
    if df_fk is not None and len(df_fk):
        display(Markdown(
            "`fonte`: **declarado** (do specs_config — confiável) ou **descoberto** "
            "(inclusão inferida — heurística, pode ter falso positivo). "
            "`incl_sintetico < incl_original` = relacionamento **degradado** (órfãos)."
        ))
        display(_style(df_fk, "status"))
    else:
        display(Markdown("_Nenhuma FK declarada/detectada._"))


# from sdv_validate import run_structural_validation, display_validation_report
#
# resultado = run_structural_validation(
#     original_path  = "oci://oci-st-blc-engordai-qab-n@gr97zovfhcmu/onprem-export",
#     synthetic_path = "oci://oci-st-blc-engordai-qab-n@gr97zovfhcmu/synthetic",
#     fmt            = "parquet",
#     max_rows       = None,           # veredito de produção
#     specs_config   = specs_config,   # opcional: passa PK/FK declaradas
# )
# display_validation_report(resultado)
