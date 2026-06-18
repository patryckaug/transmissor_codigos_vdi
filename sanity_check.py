# -*- coding: utf-8 -*-
"""
preappend_check.py
==================

Pré-checagem de SEGURANÇA antes de appendar o conjunto sintético no banco de
PRODUÇÃO REAL. Lê SOMENTE o Object Storage (não escreve nada, não toca
produção).

Contexto e premissas (confirmados)
----------------------------------
  • Destino do append = banco de PRODUÇÃO REAL (NÃO o onprem-export; este é só
    uma cópia de exploração).
  • Conjunto sintético é AUTOCONTIDO: pais e filhos entram juntos no mesmo
    append. Logo, toda FK deve resolver DENTRO do próprio sintético.
  • Regra inviolável: PK e FK completamente NOVAS no destino. PROIBIDO duplicar.

O que ESTE script garante (necessário, mas NÃO suficiente)
----------------------------------------------------------
Verifica, dentro do próprio sintético:
  1. PK_DUP_INTERNA  — nenhuma PK candidata pode ter duplicata/nulo no sintético
                       (se duplica internamente, o INSERT já viola unique).
  2. FK_SEM_PAI      — toda relação de inclusão observada no original deve
                       resolver dentro do sintético autocontido (pai presente).
  3. PK_RANGE        — min/max de cada PK numérica no sintético, para o
                       engenheiro comparar com max(PK) de PRODUÇÃO.

O que ESTE script NÃO pode garantir (só produção responde)
----------------------------------------------------------
  • Se as PKs sintéticas COLIDEM com as já existentes em PRODUÇÃO. Isso é
    impossível de verificar a partir do Object Storage — produção é um banco
    vivo. A peça 2 (SELECT em produção, no protocolo abaixo) cobre isso, e é
    OBRIGATÓRIA. Validar colisão contra o onprem-export daria GO FALSO.

Uso
---
    from preappend_check import run_preappend_check

    rel = run_preappend_check(
        synthetic_path = "oci://bkt@ns/synthetic",
        original_path  = "oci://bkt@ns/onprem-export",  # só p/ descobrir PKs/FKs
        fmt            = "parquet",
        max_rows       = None,    # OBRIGATÓRIO None aqui — ver nota.
    )
    rel["go_nogo"]        # "GO (pendente produção)" | "NO-GO"
    rel["pk_interna"]     # DataFrame
    rel["fk_interna"]     # DataFrame
    rel["pk_ranges"]      # DataFrame -> levar para comparar com produção
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from sdv_validate import (
    _join_path, _list_tables, _read_table,
    _discover_single_col_pks, _is_unique_notnull,
    _discover_inclusions, _value_set,
)


def run_preappend_check(
    *,
    synthetic_path: "Union[str, Mapping[str, str]]",
    original_path: "Union[str, Mapping[str, str]]",
    fmt: str = "parquet",
    max_rows: Optional[int] = None,
    min_inclusion: float = 1.0,
    min_distinct_child: int = 5,
    min_parent_coverage: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Pré-checagem de consistência interna do sintético + ranges de PK.

    IMPORTANTE: rode com max_rows=None. Amostrar orfana FK artificialmente
    (FK_SEM_PAI falso) e calcula range de PK errado — inútil para esta decisão.
    """
    if max_rows is not None:
        raise ValueError(
            "max_rows deve ser None nesta checagem: amostra orfana FK e falseia "
            "o range de PK. Para decidir append em produção, leia o dado completo."
        )

    # ── descobrir tabelas ──
    if isinstance(synthetic_path, str):
        synth_paths = {t: _join_path(synthetic_path, t)
                       for t in _list_tables(synthetic_path, fmt)}
    else:
        synth_paths = dict(synthetic_path)
    if isinstance(original_path, str):
        orig_paths = {t: _join_path(original_path, t)
                      for t in _list_tables(original_path, fmt)}
    else:
        orig_paths = dict(original_path)

    comuns = sorted(set(synth_paths) & set(orig_paths))
    if verbose:
        print(f"Tabelas a checar: {comuns}")

    # ── carregar ──
    synth: Dict[str, pd.DataFrame] = {}
    real:  Dict[str, pd.DataFrame] = {}
    for t in comuns:
        synth[t] = _read_table(synth_paths[t], fmt, None, seed)
        real[t]  = _read_table(orig_paths[t], fmt, None, seed)
        if verbose:
            print(f"  {t}: sintético {len(synth[t]):,} linhas")

    bloqueios: List[str] = []

    # ── 1. PK candidata (descoberta no original) sem dup/nulo no sintético ──
    pk_candidates = {t: _discover_single_col_pks(real[t]) for t in comuns}
    pk_rows: List[Dict[str, Any]] = []
    for t in comuns:
        for c in pk_candidates[t]:
            if c not in synth[t].columns:
                continue
            ok, nulos, dup = _is_unique_notnull(synth[t], [c])
            if not ok:
                bloqueios.append(
                    f"{t}.{c}: PK com {nulos} nulo(s) e {dup} duplicado(s) DENTRO "
                    "do sintético (viola unique já no próprio conjunto)."
                )
            pk_rows.append({
                "tabela": t, "pk": c,
                "linhas": len(synth[t]),
                "nulos": nulos, "duplicados": dup,
                "status": "OK" if ok else "BLOQUEIO",
            })

    # ── 2. FK (inclusão do original) resolve dentro do sintético autocontido ──
    inclusions = _discover_inclusions(
        real, pk_candidates,
        min_inclusion=min_inclusion, min_distinct_child=min_distinct_child,
        min_parent_coverage=min_parent_coverage,
    )
    fk_rows: List[Dict[str, Any]] = []
    for inc in inclusions:
        child, ccol = inc["child_table"], inc["child_col"]
        ptab, pcol  = inc["parent_table"], inc["parent_col"]
        if ccol not in synth.get(child, pd.DataFrame()).columns:
            continue
        if pcol not in synth.get(ptab, pd.DataFrame()).columns:
            continue
        cvals = _value_set(synth[child][ccol])
        pvals = _value_set(synth[ptab][pcol])
        orfaos = sorted(cvals - pvals)
        n_orf = len(orfaos)
        if n_orf > 0 and inc["confianca"] == "alta":
            bloqueios.append(
                f"{child}.{ccol} -> {ptab}.{pcol}: {n_orf} valor(es) de FK sem pai "
                f"DENTRO do sintético (conf.=alta). Conjunto não é "
                "autocontido para essa relação."
            )
        fk_rows.append({
            "fk": f"{child}.{ccol} -> {ptab}.{pcol}",
            "confianca": inc["confianca"],
            "parent_coverage": inc["parent_coverage"],
            "valores_orfaos": n_orf,
            "exemplos_orfaos": ", ".join(map(str, orfaos[:5])) + (" ..." if n_orf > 5 else ""),
            "status": "OK" if n_orf == 0 else (
                "BLOQUEIO" if inc["confianca"] == "alta" else "REVISAR"
            ),
        })

    # ── 3. range de PK por tabela (numéricas) — levar para comparar com prod ──
    range_rows: List[Dict[str, Any]] = []
    for t in comuns:
        for c in pk_candidates[t]:
            if c not in synth[t].columns:
                continue
            col = synth[t][c]
            if pd.api.types.is_numeric_dtype(col):
                range_rows.append({
                    "tabela": t, "pk": c,
                    "min_synth": col.min(), "max_synth": col.max(),
                    "n_distintos": int(col.nunique()),
                    "acao": "comparar com SELECT max(%s) em PRODUÇÃO" % c,
                })
            else:
                range_rows.append({
                    "tabela": t, "pk": c,
                    "min_synth": "(não-numérica)", "max_synth": "(não-numérica)",
                    "n_distintos": int(col.nunique()),
                    "acao": "verificar interseção de conjunto com PRODUÇÃO",
                })

    go = "NO-GO" if bloqueios else "GO (pendente verificação em produção)"

    return {
        "go_nogo": go,
        "bloqueios": bloqueios,
        "pk_interna": pd.DataFrame(pk_rows),
        "fk_interna": pd.DataFrame(fk_rows),
        "pk_ranges": pd.DataFrame(range_rows),
        "tabelas": comuns,
    }


def display_preappend_report(rel: Mapping[str, Any]) -> None:
    """Renderiza a pré-checagem inline no notebook."""
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
                c = _GREEN if up == "OK" else (_AMBER if up == "REVISAR" else _RED)
                return f"background-color:{c};color:white;font-weight:bold"
            sty = sty.map(_bg, subset=[status_col])
        return sty

    display(Markdown("# 🚦 Pré-checagem de Append em Produção"))

    go = rel["go_nogo"]
    cor = _GREEN if go.startswith("GO") else _RED
    display(Markdown(
        f"<div style='background:{cor};color:white;padding:14px 18px;"
        f"border-radius:8px;font-size:19px;font-weight:bold'>{go}</div>"
    ))

    display(Markdown(
        "> ⚠️ **Esta checagem é NECESSÁRIA mas NÃO SUFICIENTE.** Ela só prova "
        "consistência **interna** do sintético (lendo o Object Storage). A "
        "colisão de PK com o banco de **produção real** NÃO pode ser verificada "
        "aqui — produção é um banco vivo. Mesmo com **GO**, o append só é seguro "
        "após a verificação contra produção descrita no protocolo abaixo."
    ))

    if rel["bloqueios"]:
        _sec("Bloqueios (impedem o append)", "⛔")
        for b in rel["bloqueios"]:
            display(Markdown(f"- ❌ {b}"))

    _sec("PK — unicidade interna no sintético", "🔑")
    display(_style(rel["pk_interna"]) or Markdown("_sem PK detectada_"))

    _sec("FK — resolução interna (conjunto autocontido)", "🔗")
    display(Markdown(
        "Cada FK observada no original deve ter **todos os valores com pai "
        "dentro do próprio sintético**. `valores_orfaos>0` em confiança "
        "**alta** é bloqueio; em **média/baixa**, REVISAR — pode ser falso "
        "positivo de inclusão (domínios numéricos que se sobrepõem por acaso)."
    ))
    display(_style(rel["fk_interna"]) or Markdown("_sem FK detectada_"))

    _sec("Ranges de PK — LEVAR PARA COMPARAR COM PRODUÇÃO", "📏")
    display(Markdown(
        "Estes são os ranges no **sintético**. O engenheiro deve confirmar em "
        "**produção** que cada range sintético não intersecta o existente."
    ))
    display(_style(rel["pk_ranges"], status_col=None) or Markdown("_sem PK numérica_"))

    # protocolo de produção
    _sec("Protocolo OBRIGATÓRIO contra produção (peça 2)", "📋")
    display(Markdown(
        "Antes do INSERT, **dentro da mesma transação**, para cada tabela:\n\n"
        "1. `SELECT max(<PK>) FROM <tabela>;` em **produção**.\n"
        "2. Confirmar que `min_synth > max(PK) produção` (range sintético todo "
        "acima) **ou** que não há interseção de conjunto (PK não-numérica).\n"
        "3. Se houver QUALQUER interseção → **abortar** e re-gerar as PKs com "
        "offset acima do max de produção.\n"
        "4. Inserir em **ordem topológica** (tabela-pai antes da tabela-filha), "
        "a mesma ordem que o sintetizador usa (`_topological_order`).\n"
        "5. Tudo em **transação com rollback**: se qualquer passo falhar, "
        "desfazer o append inteiro. Nada de append parcial em produção.\n"
        "6. Se produção recebe escrita concorrente, garantir lock ou offset de "
        "PK reservado, senão o max(PK) pode mudar entre o SELECT e o INSERT."
    ))
