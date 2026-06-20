# -*- coding: utf-8 -*-
"""
check_fk_juros_condicao.py
==========================

Verifica, NOS DADOS SINTÉTICOS, se a FK que quebrou no append está íntegra:

    JUROS_FLUTUANTE.NUM_CONDICAO_IF  ->  CONDICAO_IF.NUM_CONDICAO_IF
    (constraint Oracle: JUR_FLUT_CONDICAO_IF_FK)

Ou seja: todo valor de NUM_CONDICAO_IF presente em JUROS_FLUTUANTE existe como
chave em CONDICAO_IF? Os valores que NÃO existem são exatamente os que o Oracle
rejeitou com ORA-02291 (parent key not found).

Lê só essas duas tabelas, inteiras, direto do Object Storage. Não escreve nada.

Uso
---
    from check_fk_juros_condicao import check_fk

    rel = check_fk(
        synthetic_path = "oci://oci-st-blc-engordai-qab-n@gr97zovfhcmu/synthetic",
        # parametrizável caso os nomes/colunas mudem:
        child_table   = "JUROS_FLUTUANTE",
        child_col     = "NUM_CONDICAO_IF",
        parent_table  = "CONDICAO_IF",
        parent_col    = "NUM_CONDICAO_IF",
    )
    rel["ok"]              # True se nenhum órfão
    rel["n_orfaos"]        # nº de valores distintos sem pai
    rel["exemplos"]        # alguns valores órfãos
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from sdv_validate import _join_path, _read_table


def _norm_key_series(s: pd.Series) -> set:
    """
    Conjunto de valores não-nulos normalizado para comparação de chaves,
    robusto ao problema do '.0': quando uma coluna tem nulos, o pandas
    promove inteiros a float (10 -> 10.0), e o outro lado (sem nulos) fica
    int (10). Comparar como string daria '10.0' != '10' e geraria falsos
    órfãos. Aqui, valores numéricos inteiros são canonizados para o int
    correspondente antes de virar string.
    """
    s = s.dropna()
    if len(s) == 0:
        return set()
    out = set()
    if pd.api.types.is_numeric_dtype(s):
        for v in s.unique():
            f = float(v)
            # inteiro exato (10.0) -> "10"; senão mantém o número como veio
            out.add(str(int(f)) if f.is_integer() else str(f))
    else:
        for v in s.unique():
            txt = str(v).strip()
            # string que é "10.0" mas representa inteiro -> "10"
            try:
                f = float(txt)
                txt = str(int(f)) if f.is_integer() else str(f)
            except (ValueError, TypeError):
                pass
            out.add(txt)
    return out


def check_fk(
    *,
    synthetic_path: str,
    child_table: str = "JUROS_FLUTUANTE",
    child_col: str = "NUM_CONDICAO_IF",
    parent_table: str = "CONDICAO_IF",
    parent_col: str = "NUM_CONDICAO_IF",
    fmt: str = "parquet",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Confere a integridade referencial da FK no conjunto SINTÉTICO.

    Compara normalizando para string, para não gerar falso órfão por diferença
    de dtype (ex.: pai lido como int e filho como string). Considera apenas
    valores NÃO-nulos do filho (FK nula = órfão intencional da política do
    sintetizador, não é violação de parent key).
    """
    child_path  = _join_path(synthetic_path, child_table)
    parent_path = _join_path(synthetic_path, parent_table)

    if verbose:
        print(f"Lendo filho  {child_table} ...", flush=True)
    cdf = _read_table(child_path, fmt, None, seed)
    if verbose:
        print(f"  {child_table}: {len(cdf):,} linhas", flush=True)
        print(f"Lendo pai    {parent_table} ...", flush=True)
    pdf = _read_table(parent_path, fmt, None, seed)
    if verbose:
        print(f"  {parent_table}: {len(pdf):,} linhas", flush=True)

    # validações de existência das colunas (erro claro se nome estiver errado)
    if child_col not in cdf.columns:
        raise KeyError(
            f"Coluna `{child_col}` não existe em {child_table}. "
            f"Colunas: {list(cdf.columns)}"
        )
    if parent_col not in pdf.columns:
        raise KeyError(
            f"Coluna `{parent_col}` não existe em {parent_table}. "
            f"Colunas: {list(pdf.columns)}"
        )

    # contagem de FK nula no filho (órfão intencional, não conta como violação)
    n_linhas_filho = len(cdf)
    n_fk_nula = int(cdf[child_col].isna().sum())

    # conjuntos de valores normalizados (robusto a int-vs-float .0), só não-nulos
    cvals = _norm_key_series(cdf[child_col])   # distintos não-nulos do filho
    pvals = _norm_key_series(pdf[parent_col])  # distintos não-nulos do pai

    orfaos = sorted(cvals - pvals)
    n_orfaos = len(orfaos)
    n_distintos_filho = len(cvals)
    match_rate = (
        (n_distintos_filho - n_orfaos) / n_distintos_filho
        if n_distintos_filho else float("nan")
    )

    # quantas LINHAS do filho seriam rejeitadas (não só valores distintos)
    if n_orfaos:
        orfaos_set = set(orfaos)
        nao_nulos = cdf[child_col].dropna()
        def _norm_one(v):
            try:
                f = float(v)
                return str(int(f)) if f.is_integer() else str(f)
            except (ValueError, TypeError):
                return str(v).strip()
        linhas_rejeitadas = int(
            nao_nulos.map(_norm_one).isin(orfaos_set).sum()
        )
    else:
        linhas_rejeitadas = 0

    ok = (n_orfaos == 0)

    if verbose:
        print()
        print(f"FK: {child_table}.{child_col} -> {parent_table}.{parent_col}")
        print(f"  valores distintos no filho (não-nulos): {n_distintos_filho:,}")
        print(f"  valores distintos no pai   (não-nulos): {len(pvals):,}")
        print(f"  FK nula no filho (órfão intencional)  : {n_fk_nula:,} "
              f"de {n_linhas_filho:,} linhas")
        print(f"  valores SEM pai (violam a FK)         : {n_orfaos:,}")
        print(f"  LINHAS do filho que seriam rejeitadas : {linhas_rejeitadas:,}")
        print(f"  match_rate                            : {match_rate:.4%}")
        print()
        if ok:
            print("✅ FK ÍNTEGRA no sintético: todo NUM_CONDICAO_IF do filho tem "
                  "pai em CONDICAO_IF.")
        else:
            print("❌ FK QUEBRADA no sintético: há valores sem pai — são exatamente "
                  "os que o Oracle rejeita com ORA-02291.")
            print(f"   Exemplos de valores órfãos: {orfaos[:10]}")
        print()
        print("Interpretação: como esta FK NÃO estava no specs_config, o "
              "sintetizador não remapeou NUM_CONDICAO_IF para as PKs sintéticas "
              "de CONDICAO_IF. Corrigir = declarar esta FK no specs_config e "
              "re-sintetizar.")

    return {
        "ok": ok,
        "fk": f"{child_table}.{child_col} -> {parent_table}.{parent_col}",
        "n_distintos_filho": n_distintos_filho,
        "n_distintos_pai": len(pvals),
        "n_fk_nula": n_fk_nula,
        "n_linhas_filho": n_linhas_filho,
        "n_orfaos": n_orfaos,
        "linhas_rejeitadas": linhas_rejeitadas,
        "match_rate": match_rate,
        "exemplos": orfaos[:20],
    }

from check_fk_juros_condicao import check_fk

rel = check_fk(
    synthetic_path = "oci://oci-st-blc-engordai-qab-n@gr97zovfhcmu/synthetic",
)
