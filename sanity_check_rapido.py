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


def _normalize_cols_cfg(v: Any) -> List[str]:
    """Aceita string única, lista ou None; retorna lista de nomes de coluna."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(c) for c in v]


def _pks_from_specs(specs_config: Mapping[str, Mapping]) -> Dict[str, List[str]]:
    """Extrai PKs de coluna única declaradas no specs_config, por tabela."""
    out: Dict[str, List[str]] = {}
    for t, cfg in specs_config.items():
        pks = _normalize_cols_cfg(cfg.get("pk_cols"))
        # esta checagem trata PK de coluna única (caso dominante de append)
        out[t] = [p for p in pks] if len(pks) == 1 else ([pks[0]] if pks else [])
    return out


def _fks_from_specs(specs_config: Mapping[str, Mapping]) -> List[Dict[str, str]]:
    """Extrai FKs de coluna única declaradas no specs_config."""
    fks: List[Dict[str, str]] = []
    for t, cfg in specs_config.items():
        for fk in (cfg.get("foreign_keys") or cfg.get("fks") or []):
            cols  = _normalize_cols_cfg(fk.get("columns"))
            pcols = _normalize_cols_cfg(fk.get("parent_columns"))
            ptab  = fk.get("parent_table")
            if len(cols) == 1 and len(pcols) == 1 and ptab:
                fks.append({
                    "child_table": t, "child_col": cols[0],
                    "parent_table": str(ptab), "parent_col": pcols[0],
                })
    return fks


def run_preappend_check(
    *,
    synthetic_path: "Union[str, Mapping[str, str]]",
    original_path: "Union[str, Mapping[str, str]]",
    specs_config: Optional[Mapping[str, Mapping]] = None,
    fmt: str = "parquet",
    max_rows: Optional[int] = None,
    original_sample_rows: int = 500_000,
    discovery_safety_net: bool = True,
    min_inclusion: float = 1.0,
    min_distinct_child: int = 5,
    min_parent_coverage: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Pré-checagem de consistência interna do sintético + ranges de PK.

    DESEMPENHO (tabelas originais de 100M+ linhas)
    ----------------------------------------------
    • specs_config (se fornecido) define o ESCOPO: quais colunas testar como
      PK e quais pares como FK. Evita varrer todas as colunas do original.
      NÃO é veredito — a PK/FK declarada é VERIFICADA nos dados; se não se
      sustentar, aparece como falha.
    • O ORIGINAL é lido por AMOSTRA (original_sample_rows) — basta para
      confirmar tipos e rodar a rede de segurança de descoberta. O veredito
      NÃO depende do original.
    • O SINTÉTICO é lido INTEIRO (é pequeno, ~100k) — toda a verificação de
      PK duplicada / FK órfã / range acontece sobre o sintético completo, que
      é onde a confiabilidade importa. Amostrar o sintético seria perigoso
      (uma PK duplicada pode estar em qualquer linha), por isso NÃO se amostra.
    • discovery_safety_net=True roda a descoberta automática SÓ na amostra do
      original e sinaliza FK forte que NÃO está no specs_config (rede contra
      specs_config incompleto). Rápido, pois é só amostra.

    max_rows: mantido como None obrigatório — refere-se ao SINTÉTICO, que nunca
    é amostrado. Use original_sample_rows para controlar a amostra do original.
    """
    if max_rows is not None:
        raise ValueError(
            "max_rows deve ser None: o sintético é lido inteiro (é onde está o "
            "veredito; amostrá-lo esconderia PK duplicada). Para acelerar a "
            "leitura do ORIGINAL, use original_sample_rows."
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

    # ── carregar: SINTÉTICO inteiro, ORIGINAL só amostra ──
    synth: Dict[str, pd.DataFrame] = {}
    real:  Dict[str, pd.DataFrame] = {}
    for t in comuns:
        synth[t] = _read_table(synth_paths[t], fmt, None, seed)            # inteiro
        real[t]  = _read_table(orig_paths[t], fmt, original_sample_rows, seed)  # amostra
        if verbose:
            print(f"  {t}: sintético {len(synth[t]):,} (inteiro) | "
                  f"original ~{len(real[t]):,} (amostra p/ descoberta)")

    bloqueios: List[str] = []

    # ── definir PKs candidatas: do specs_config se houver, senão descobrir ──
    if specs_config is not None:
        pk_candidates = _pks_from_specs(specs_config)
        # restringe às tabelas em comum e a colunas que existem
        pk_candidates = {
            t: [c for c in cols if t in synth and c in synth[t].columns]
            for t, cols in pk_candidates.items() if t in comuns
        }
        origem_pk = "specs_config (verificado nos dados)"
    else:
        pk_candidates = {t: _discover_single_col_pks(real[t]) for t in comuns}
        origem_pk = "descoberta automática (amostra do original)"
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

    # ── 2. FK: valida as declaradas no sintético; descoberta = rede de segurança ──
    fk_rows: List[Dict[str, Any]] = []

    def _testar_fk_no_sintetico(child, ccol, ptab, pcol, origem, confianca):
        """Verifica, NO SINTÉTICO, se a FK resolve (filho ⊆ pai). Registra linha."""
        if child not in synth or ptab not in synth:
            return
        if ccol not in synth[child].columns or pcol not in synth[ptab].columns:
            return
        cvals = _value_set(synth[child][ccol])
        pvals = _value_set(synth[ptab][pcol])
        orfaos = sorted(cvals - pvals)
        n_orf = len(orfaos)
        # FK DECLARADA no specs_config com órfão = bloqueio (é o que viemos checar).
        # FK só DESCOBERTA (não declarada) = aviso, mesmo com órfão (pode ser
        # falso positivo de inclusão; não bloqueia o append por conta dela).
        if n_orf > 0 and origem == "specs_config":
            bloqueios.append(
                f"{child}.{ccol} -> {ptab}.{pcol}: {n_orf} valor(es) de FK sem pai "
                "DENTRO do sintético (FK declarada). Conjunto não é autocontido "
                "para essa relação."
            )
            status = "BLOQUEIO"
        elif n_orf > 0:
            status = "REVISAR (não declarada)"
        else:
            status = "OK"
        fk_rows.append({
            "fk": f"{child}.{ccol} -> {ptab}.{pcol}",
            "origem": origem,
            "confianca": confianca,
            "valores_orfaos": n_orf,
            "exemplos_orfaos": ", ".join(map(str, orfaos[:5])) + (" ..." if n_orf > 5 else ""),
            "status": status,
        })
        return (child, ccol, ptab, pcol)

    testadas: set = set()

    # 2a. FKs declaradas no specs_config (o veredito que interessa)
    if specs_config is not None:
        for fk in _fks_from_specs(specs_config):
            chave = _testar_fk_no_sintetico(
                fk["child_table"], fk["child_col"],
                fk["parent_table"], fk["parent_col"],
                origem="specs_config", confianca="declarada",
            )
            if chave:
                testadas.add(chave)

    # 2b. rede de segurança: descobre FK forte na AMOSTRA do original e, se não
    #     estava declarada, sinaliza para revisão (cobre specs_config incompleto)
    if discovery_safety_net:
        inclusions = _discover_inclusions(
            real, pk_candidates,
            min_inclusion=min_inclusion, min_distinct_child=min_distinct_child,
            min_parent_coverage=min_parent_coverage,
        )
        for inc in inclusions:
            chave = (inc["child_table"], inc["child_col"],
                     inc["parent_table"], inc["parent_col"])
            if chave in testadas:
                continue  # já coberta pelo specs_config
            if inc["confianca"] != "alta":
                continue  # só sinaliza descoberta de ALTA confiança (evita ruído)
            _testar_fk_no_sintetico(
                inc["child_table"], inc["child_col"],
                inc["parent_table"], inc["parent_col"],
                origem="descoberta", confianca=inc["confianca"],
            )

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
        "origem_pk": origem_pk,
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
                c = _GREEN if up == "OK" else (_AMBER if "REVISAR" in up else _RED)
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
    display(Markdown(f"_Escopo das PKs: {rel.get('origem_pk', 'n/d')}._"))
    display(_style(rel["pk_interna"]) or Markdown("_sem PK detectada_"))

    _sec("FK — resolução interna (conjunto autocontido)", "🔗")
    display(Markdown(
        "FK **declarada no specs_config** com `valores_orfaos>0` é **bloqueio** "
        "(é o que viemos checar). FK apenas **descoberta** (não declarada) com "
        "órfão vai para **REVISAR** — pode ser falso positivo de inclusão e não "
        "bloqueia o append. A coluna `origem` diz qual é qual."
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

# import json
# from preappend_check import run_preappend_check, display_preappend_report

# with open("spec_config.json") as f:
#     specs_config = json.load(f)

# rel = run_preappend_check(
#     synthetic_path = "oci://oci-st-blc-engordai-qab-n@gr97zovfhcmu/synthetic",
#     original_path  = "oci://oci-st-blc-engordai-qab-n@gr97zovfhcmu/onprem-export",
#     specs_config   = specs_config,
#     fmt            = "parquet",
#     max_rows       = None,            # refere-se ao sintético: sempre inteiro
#     original_sample_rows = 500_000,   # amostra do original p/ descoberta
# )
# display_preappend_report(rel)
