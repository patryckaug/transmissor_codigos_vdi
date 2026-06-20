# -*- coding: utf-8 -*-
"""
check_fks_specs.py
==================

Varre TODAS as FKs declaradas no specs_config e verifica, nos dados
SINTÉTICOS, se cada uma está íntegra (todo valor do filho tem pai).
Os órfãos são exatamente o que o Oracle rejeitaria com ORA-02291.

Escopo e limite (leia antes de concluir)
-----------------------------------------
Este verificador checa SOMENTE as FKs que estão no specs_config — ou seja,
as que o sintetizador CONHECIA. Se todas derem OK, isso prova que a síntese
preservou bem o que foi declarado. NÃO prova que o append é seguro: uma FK
REAL do banco que ficou de fora do specs_config (como a JUR_FLUT_CONDICAO_IF_FK,
que você confirmou não estar declarada) não é vista aqui e pode ainda quebrar
o append. Para cobrir esse ponto cego, seria preciso a lista completa de FKs
do Oracle (all_constraints).

Lê só as tabelas envolvidas em FK, uma vez cada (com cache). Não escreve nada.

Uso
---
    import json
    from check_fks_specs import check_all_fks

    with open("spec_config.json") as f:
        specs_config = json.load(f)

    rel = check_all_fks(
        synthetic_path = "oci://oci-st-blc-engordai-qab-n@gr97zovfhcmu/synthetic",
        specs_config   = specs_config,
    )
    rel["todas_ok"]      # True se nenhuma FK declarada tem órfão
    rel["relatorio"]     # DataFrame: uma linha por FK, com órfãos e status
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from sdv_validate import _join_path, _read_table


# ── normalização de chave robusta a int-vs-float (.0) ──
def _norm_key_series(s: pd.Series) -> set:
    s = s.dropna()
    if len(s) == 0:
        return set()
    out = set()
    if pd.api.types.is_numeric_dtype(s):
        for v in s.unique():
            f = float(v)
            out.add(str(int(f)) if f.is_integer() else str(f))
    else:
        for v in s.unique():
            txt = str(v).strip()
            try:
                f = float(txt)
                txt = str(int(f)) if f.is_integer() else str(f)
            except (ValueError, TypeError):
                pass
            out.add(txt)
    return out


def _norm_one(v) -> str:
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (ValueError, TypeError):
        return str(v).strip()


def _normalize_cols_cfg(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(c) for c in v]


def check_all_fks(
    *,
    synthetic_path: str,
    specs_config: Mapping[str, Mapping],
    fmt: str = "parquet",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Verifica todas as FKs do specs_config nos dados sintéticos.

    Retorna dict com:
      "todas_ok"  : bool
      "relatorio" : DataFrame (uma linha por FK)
      "n_fks"     : nº de FKs verificadas
      "n_quebradas": nº de FKs com órfão
    """
    # ── coletar todas as FKs declaradas ──
    fks: List[Dict[str, Any]] = []
    for child, cfg in specs_config.items():
        for fk in (cfg.get("foreign_keys") or cfg.get("fks") or []):
            cols  = _normalize_cols_cfg(fk.get("columns"))
            pcols = _normalize_cols_cfg(fk.get("parent_columns"))
            ptab  = fk.get("parent_table")
            if not cols or not pcols or not ptab:
                continue
            fks.append({
                "child_table": child, "child_cols": cols,
                "parent_table": str(ptab), "parent_cols": pcols,
            })

    if verbose:
        print(f"{len(fks)} FK(s) declarada(s) no specs_config para verificar.\n",
              flush=True)

    # ── cache de leitura: cada tabela lida uma única vez ──
    cache: Dict[str, pd.DataFrame] = {}

    def _get(tab: str) -> Optional[pd.DataFrame]:
        if tab in cache:
            return cache[tab]
        try:
            if verbose:
                print(f"  lendo {tab}...", flush=True)
            df = _read_table(_join_path(synthetic_path, tab), fmt, None, seed)
        except Exception as exc:
            if verbose:
                print(f"  ⚠️ falha ao ler {tab}: {exc}", flush=True)
            df = None
        cache[tab] = df
        return df

    linhas: List[Dict[str, Any]] = []

    for i, fk in enumerate(fks, start=1):
        child, ccols = fk["child_table"], fk["child_cols"]
        ptab,  pcols = fk["parent_table"], fk["parent_cols"]
        rotulo = f"{child}.{'+'.join(ccols)} -> {ptab}.{'+'.join(pcols)}"

        if verbose:
            print(f"[{i}/{len(fks)}] {rotulo}", flush=True)

        cdf = _get(child)
        pdf = _get(ptab)

        # tabela ausente / ilegível
        if cdf is None or pdf is None:
            linhas.append({"fk": rotulo, "status": "ERRO",
                           "motivo": "tabela não lida",
                           "orfaos": None, "match_rate": None})
            continue

        # FK composta: comparar por tupla de colunas concatenadas
        falta_c = [c for c in ccols if c not in cdf.columns]
        falta_p = [c for c in pcols if c not in pdf.columns]
        if falta_c or falta_p:
            linhas.append({"fk": rotulo, "status": "ERRO",
                           "motivo": f"colunas ausentes child={falta_c} parent={falta_p}",
                           "orfaos": None, "match_rate": None})
            continue

        if len(ccols) == 1:
            cvals = _norm_key_series(cdf[ccols[0]])
            pvals = _norm_key_series(pdf[pcols[0]])
            n_fk_nula = int(cdf[ccols[0]].isna().sum())
        else:
            # composta: chave = tupla normalizada, ignora linhas com qualquer nulo
            c_sub = cdf[ccols].dropna()
            p_sub = pdf[pcols].dropna()
            cvals = set(
                tuple(_norm_one(r[c]) for c in ccols)
                for _, r in c_sub.iterrows()
            )
            pvals = set(
                tuple(_norm_one(r[c]) for c in pcols)
                for _, r in p_sub.iterrows()
            )
            n_fk_nula = len(cdf) - len(c_sub)

        orfaos = cvals - pvals
        n_orf = len(orfaos)
        n_dist = len(cvals)
        match = (n_dist - n_orf) / n_dist if n_dist else float("nan")
        exemplos = list(orfaos)[:8]

        linhas.append({
            "fk": rotulo,
            "distintos_filho": n_dist,
            "distintos_pai": len(pvals),
            "fk_nula": n_fk_nula,
            "orfaos": n_orf,
            "match_rate": round(match, 4) if n_dist else None,
            "exemplos_orfaos": ", ".join(map(str, exemplos)) + (" ..." if n_orf > 8 else ""),
            "status": "OK" if n_orf == 0 else "QUEBRADA",
        })
        if verbose:
            tag = "✅ OK" if n_orf == 0 else f"❌ QUEBRADA ({n_orf} órfão(s))"
            print(f"        {tag}", flush=True)

    rel = pd.DataFrame(linhas)
    n_quebradas = int((rel["status"] == "QUEBRADA").sum()) if len(rel) else 0
    n_erros = int((rel["status"] == "ERRO").sum()) if len(rel) else 0
    todas_ok = (n_quebradas == 0 and n_erros == 0)

    if verbose:
        print()
        print("=" * 60)
        print(f"FKs verificadas: {len(fks)} · quebradas: {n_quebradas} · "
              f"erros de leitura: {n_erros}")
        if todas_ok:
            print("✅ Todas as FKs DECLARADAS estão íntegras no sintético.")
        else:
            print("❌ Há FK(s) quebrada(s) ou não verificável(is) — ver relatório.")
        print("⚠️  Lembrete: isto cobre só o que está no specs_config. Uma FK real "
              "do banco fora da config (não declarada) NÃO é vista aqui e ainda "
              "pode causar ORA-02291 no append.")
        print("=" * 60)

    return {
        "todas_ok": todas_ok,
        "relatorio": rel,
        "n_fks": len(fks),
        "n_quebradas": n_quebradas,
        "n_erros": n_erros,
    }

import json
from check_fks_specs import check_all_fks

with open("spec_config.json") as f:
    specs_config = json.load(f)

rel = check_all_fks(
    synthetic_path = "oci://oci-st-blc-engordai-qab-n@gr97zovfhcmu/synthetic",
    specs_config   = specs_config,
)
display(rel["relatorio"])   # uma linha por FK, com órfãos e status
