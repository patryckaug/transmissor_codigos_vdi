# -*- coding: utf-8 -*-
"""
sdv_validate.py
===============

Validação ESTRUTURAL e RELACIONAL, 100% data-driven, de dados sintéticos
contra os originais — para decidir, com alta confiança, se a sintetização
preservou TIPO DE DADO, CHAVES PRIMÁRIAS e CHAVES ESTRANGEIRAS.

Princípio central (sem leakage)
-------------------------------
Este validador NÃO recebe specs_config nem metadado. Ele não confia em
nenhuma declaração de PK/FK. A "verdade" é inferida dos PRÓPRIOS DADOS, dos
DOIS lados, e a comparação é feita assim:

    Toda propriedade estrutural verdadeira no ORIGINAL tem que continuar
    verdadeira no SINTÉTICO. Qualquer divergência é uma INCONSISTÊNCIA.

Concretamente:
  • TIPO   — o dtype real (lido do schema Parquet) de cada coluna do original
             tem que bater com o do sintético. Coluna a mais/a menos, ou tipo
             trocado (int viva como string, etc.) → inconsistência.
  • PK     — toda coluna (ou par de colunas) que é ÚNICA E NÃO-NULA no original
             é uma PK candidata. Se ela não for única/não-nula no sintético →
             inconsistência (o sintetizador quebrou uma chave).
  • FK     — toda relação de INCLUSÃO observada no original (os valores de uma
             coluna do filho estão 100% contidos numa coluna-chave de outra
             tabela) é uma FK candidata. Se essa mesma inclusão quebra no
             sintético (surgem valores órfãos) → inconsistência.

Por que comparativo, e não absoluto
------------------------------------
"Existe FK?" é indecidível só pelos dados (coincidência de domínio gera falso
positivo). Mas "uma inclusão que valia no original parou de valer no sintético"
é um SINAL DETERMINÍSTICO de que a sintetização degradou um relacionamento —
mesmo um relacionamento que ninguém declarou no specs_config. É exatamente o
ponto cego que a validação baseada em specs_config não cobre.

Limitações honestas (lidas com atenção antes de decidir append)
---------------------------------------------------------------
  • FK por inclusão é heurística: dá falso positivo (duas colunas de código que
    compartilham domínio sem relação real). Por isso a saída marca confiança e
    você revisa as candidatas, não as toma como dogma.
  • Amostragem (max_rows) ORFANA relacionamentos artificialmente. Para um
    veredito confiável de PK/FK, rode com max_rows=None sobre o dado completo.
    Com amostra, o resultado é indicativo, não conclusivo — e o relatório diz
    isso em destaque.
  • Este módulo SÓ LÊ. Não escreve nada em lugar nenhum.

Uso
---
    from sdv_validate import run_structural_validation

    resultado = run_structural_validation(
        original_path  = "oci://bkt@ns/onprem-export",   # prefixo OU dict
        synthetic_path = "oci://bkt@ns/synthetic",
        fmt            = "parquet",
        max_rows       = None,      # None p/ veredito de produção (recomendado)
        fk_max_cols    = 1,         # 1 = só FKs de coluna única (rápido)
    )

    resultado["veredito"]            # "APROVADO" | "REPROVADO"
    resultado["inconsistencias"]     # DataFrame com todo problema achado
"""

from __future__ import annotations

import io
import os
import warnings as _warnings
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
        # prefixo relativo ao base, em forma nativa do ocifs (sem oci://)
        nativo = base_norm[len("oci://"):]
        nomes = set()
        for f in todos:
            rel = f[len(nativo):].lstrip("/") if f.startswith(nativo) else f
            if "/" in rel:
                nomes.add(rel.split("/")[0])
        return sorted(nomes)
    # local
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
    fmt = "parquet"
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
    # local
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
        int8/int16/int32/int64/Int64  -> "integer"
        float16/32/64                 -> "float"
        bool/boolean                  -> "boolean"
        datetime64[*]                 -> "datetime"
        object/string                 -> "string"
        category                      -> "string" (categoria é representação)
    Mantém a distinção que IMPORTA para append: integer != float != string !=
    datetime != boolean. Um ID que era integer e virou string é flagado.
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
    # object, string, category -> string
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
    """Conjunto de valores não-nulos, normalizado para string para comparar."""
    s = series.dropna()
    if len(s) == 0:
        return set()
    return set(s.astype(str).unique())


def _inclusion_ratio(child_vals: set, parent_vals: set) -> float:
    """% dos valores distintos do filho que existem no pai. 1.0 = inclusão total."""
    if not child_vals:
        return np.nan
    contidos = len(child_vals & parent_vals)
    return contidos / len(child_vals)


def _discover_inclusions(
    tables: Dict[str, pd.DataFrame],
    pk_candidates: Dict[str, List[str]],
    *,
    min_inclusion: float = 1.0,
    min_distinct_child: int = 5,
    min_parent_coverage: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Descobre FKs candidatas: para cada coluna de cada tabela (filho), testa se
    seus valores estão contidos nos valores de uma PK candidata de OUTRA tabela
    (pai). Inclusão >= min_inclusion vira candidata.

    Defesas contra falso positivo (domínios numéricos que se sobrepõem por
    acaso, ex.: COD_X ⊆ COD_Y só porque ambos são 1..N):
      • a coluna-filho NÃO pode ser ela mesma uma PK única da própria tabela
        (uma PK própria raramente é FK de outra);
      • exige cobertura mínima do pai: a fração de valores DISTINTOS do pai que
        aparecem no filho (relacionamento real costuma cobrir parte relevante
        do pai; 1-2 valores coincidentes não);
      • marca uma 'confianca' qualitativa para revisão humana.

    Conservador de propósito: só considera pai = coluna que é PK candidata
    (única+não-nula) no original.
    """
    candidatas: List[Dict[str, Any]] = []
    parent_sets: Dict[Tuple[str, str], set] = {}
    for ptab, pcols in pk_candidates.items():
        for pcol in pcols:
            parent_sets[(ptab, pcol)] = _value_set(tables[ptab][pcol])

    # conjunto de PKs próprias por tabela (para não tratá-las como filho-FK)
    own_pks = {t: set(cols) for t, cols in pk_candidates.items()}

    for child, cdf in tables.items():
        for ccol in cdf.columns:
            # uma PK única da própria tabela dificilmente é FK de outra
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
                # cobertura do pai: quantos valores do pai o filho referencia
                cobertura = len(cvals & pvals) / len(pvals) if pvals else 0.0
                if cobertura < min_parent_coverage:
                    continue
                # confiança qualitativa
                if cobertura >= 0.5 and len(cvals) >= 20:
                    conf = "alta"
                elif cobertura >= 0.2:
                    conf = "media"
                else:
                    conf = "baixa"
                candidatas.append({
                    "child_table": child,
                    "child_col":   ccol,
                    "parent_table": ptab,
                    "parent_col":  pcol,
                    "incl_original": round(float(ratio), 4),
                    "n_distinct_child": len(cvals),
                    "parent_coverage": round(float(cobertura), 4),
                    "confianca": conf,
                })
    return candidatas


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
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Valida estrutura (tipo) e relacionamentos (PK/FK) do sintético contra o
    original, sem usar specs_config. Retorna dict com veredito e DataFrames.

    Parâmetros
    ----------
    original_path / synthetic_path : prefixo base (str) — descobre as tabelas
        automaticamente — OU dict {tabela: caminho}.
    fmt          : "parquet" (exigido; ver _read_table).
    max_rows     : None para veredito de produção (recomendado). Com amostra,
        o resultado de PK/FK é INDICATIVO, não conclusivo.
    fk_max_cols  : 1 = só FKs de coluna única (padrão, rápido). >1 não
        implementado aqui (FK composta multiplica o custo combinatório).
    min_inclusion: limiar de inclusão para considerar uma FK candidata no
        original (1.0 = inclusão total).
    min_distinct_child : ignora colunas com menos distintos que isso ao inferir
        FK (evita ruído de colunas de flag/status).

    Retorna
    -------
    {
      "veredito": "APROVADO" | "REPROVADO",
      "inconsistencias": DataFrame,   # cada linha = 1 problema
      "tipos": DataFrame,             # comparação de dtype por coluna
      "pks": DataFrame,               # PK candidata: vale no orig? e no synth?
      "fks": DataFrame,               # FK candidata: inclusão orig vs synth
      "tabelas": DataFrame,           # presença/contagem por tabela
      "amostrado": bool,
    }
    """
    if fk_max_cols != 1:
        raise NotImplementedError(
            "fk_max_cols>1 (FK composta) não implementado nesta versão. "
            "FK de coluna única cobre o caso dominante; composta exige busca "
            "combinatória cara e deve ser tratada à parte."
        )

    # ── 1. descobrir e carregar tabelas (dos dois lados) ──
    if isinstance(original_path, str):
        orig_tabs = _list_tables(original_path, fmt)
        orig_paths = {t: _join_path(original_path, t) for t in orig_tabs}
    else:
        orig_paths = dict(original_path)

    if isinstance(synthetic_path, str):
        synth_tabs = _list_tables(synthetic_path, fmt)
        synth_paths = {t: _join_path(synthetic_path, t) for t in synth_tabs}
    else:
        synth_paths = dict(synthetic_path)

    if verbose:
        print(f"Tabelas no original:  {sorted(orig_paths)}")
        print(f"Tabelas no sintético: {sorted(synth_paths)}")

    inconsist: List[Dict[str, Any]] = []

    # ── 2. presença de tabelas (estrutural de alto nível) ──
    so, ss = set(orig_paths), set(synth_paths)
    tab_rows: List[Dict[str, Any]] = []
    for t in sorted(so | ss):
        no_orig, no_synth = t in so, t in ss
        if no_orig and not no_synth:
            inconsist.append({
                "severidade": "CRÍTICO", "tipo": "TABELA_AUSENTE",
                "objeto": t,
                "detalhe": "tabela existe no original mas NÃO no sintético",
            })
        elif no_synth and not no_orig:
            inconsist.append({
                "severidade": "ALERTA", "tipo": "TABELA_EXTRA",
                "objeto": t,
                "detalhe": "tabela existe no sintético mas NÃO no original",
            })
        tab_rows.append({"tabela": t, "no_original": no_orig, "no_sintetico": no_synth})

    comuns = sorted(so & ss)
    if not comuns:
        df_inc = pd.DataFrame(inconsist)
        return {
            "veredito": "REPROVADO",
            "inconsistencias": df_inc,
            "tipos": pd.DataFrame(), "pks": pd.DataFrame(),
            "fks": pd.DataFrame(),
            "tabelas": pd.DataFrame(tab_rows),
            "amostrado": bool(max_rows),
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
                inconsist.append({
                    "severidade": "CRÍTICO", "tipo": "COLUNA_AUSENTE",
                    "objeto": f"{t}.{c}",
                    "detalhe": "coluna existe no original mas NÃO no sintético",
                })
                tipo_rows.append({"tabela": t, "coluna": c,
                                  "tipo_original": _canonical_dtype(real[t][c]),
                                  "tipo_sintetico": "—", "status": "AUSENTE"})
                continue
            to = _canonical_dtype(real[t][c])
            ts = _canonical_dtype(synth[t][c])
            ok = (to == ts)
            if not ok:
                inconsist.append({
                    "severidade": "CRÍTICO", "tipo": "TIPO_DIVERGENTE",
                    "objeto": f"{t}.{c}",
                    "detalhe": f"tipo original={to} mas sintético={ts}",
                })
            tipo_rows.append({"tabela": t, "coluna": c, "tipo_original": to,
                              "tipo_sintetico": ts,
                              "status": "OK" if ok else "DIVERGENTE"})
        for c in scols:
            if c not in rcols:
                inconsist.append({
                    "severidade": "ALERTA", "tipo": "COLUNA_EXTRA",
                    "objeto": f"{t}.{c}",
                    "detalhe": "coluna existe no sintético mas NÃO no original",
                })
                tipo_rows.append({"tabela": t, "coluna": c, "tipo_original": "—",
                                  "tipo_sintetico": _canonical_dtype(synth[t][c]),
                                  "status": "EXTRA"})

    # ── 4. PK descoberta no ORIGINAL, re-testada no SINTÉTICO ──
    pk_candidates_orig: Dict[str, List[str]] = {
        t: _discover_single_col_pks(real[t]) for t in comuns
    }
    pk_rows: List[Dict[str, Any]] = []
    for t in comuns:
        for c in pk_candidates_orig[t]:
            if c not in synth[t].columns:
                continue  # já reportado como COLUNA_AUSENTE
            ok_s, nulos_s, dup_s = _is_unique_notnull(synth[t], [c])
            if not ok_s:
                inconsist.append({
                    "severidade": "CRÍTICO", "tipo": "PK_QUEBRADA",
                    "objeto": f"{t}.{c}",
                    "detalhe": (f"coluna é PK no original (única+não-nula) mas no "
                                f"sintético tem {nulos_s} nulo(s) e {dup_s} "
                                f"duplicado(s)"),
                })
            pk_rows.append({
                "tabela": t, "coluna_pk": c,
                "pk_no_original": True,
                "pk_no_sintetico": ok_s,
                "nulos_synth": nulos_s, "duplicados_synth": dup_s,
                "status": "OK" if ok_s else "QUEBRADA",
            })

    # ── 5. FK descoberta por inclusão no ORIGINAL, re-testada no SINTÉTICO ──
    inclusions = _discover_inclusions(
        real, pk_candidates_orig,
        min_inclusion=min_inclusion, min_distinct_child=min_distinct_child,
        min_parent_coverage=min_parent_coverage,
    )
    fk_rows: List[Dict[str, Any]] = []
    for inc in inclusions:
        child, ccol = inc["child_table"], inc["child_col"]
        ptab, pcol  = inc["parent_table"], inc["parent_col"]
        # precisa existir dos dois lados no sintético
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
            inconsist.append({
                "severidade": "CRÍTICO", "tipo": "FK_DEGRADADA",
                "objeto": f"{child}.{ccol} -> {ptab}.{pcol}",
                "detalhe": (f"inclusão caiu de {inc['incl_original']:.4f} (original) "
                            f"para {ratio_s if ratio_s is not np.nan else float('nan'):.4f} "
                            f"(sintético): {orfaos} valor(es) órfão(s) no filho"),
            })
        fk_rows.append({
            "fk": f"{child}.{ccol} -> {ptab}.{pcol}",
            "incl_original": inc["incl_original"],
            "incl_sintetico": (round(float(ratio_s), 4)
                               if ratio_s is not np.nan else np.nan),
            "valores_orfaos_synth": orfaos,
            "n_distinct_child": inc["n_distinct_child"],
            "parent_coverage": inc["parent_coverage"],
            "confianca": inc["confianca"],
            "status": "DEGRADADA" if quebrou else "OK",
        })

    # ── 6. veredito ──
    df_inc = pd.DataFrame(inconsist)
    criticos = (
        int((df_inc["severidade"] == "CRÍTICO").sum())
        if len(df_inc) else 0
    )
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

    display(Markdown("# 🔒 Validação Estrutural & Relacional (data-driven)"))

    veredito = resultado["veredito"]
    cor = _GREEN if veredito == "APROVADO" else _RED
    n_crit = resultado.get("n_criticos", 0)
    display(Markdown(
        f"<div style='background:{cor};color:white;padding:14px 18px;"
        f"border-radius:8px;font-size:20px;font-weight:bold'>"
        f"VEREDITO: {veredito} &nbsp;·&nbsp; {n_crit} inconsistência(s) crítica(s)"
        f"</div>"
    ))

    if resultado.get("amostrado"):
        display(Markdown(
            "> ⚠️ **AMOSTRADO (max_rows definido).** O veredito de PK/FK é "
            "**INDICATIVO, não conclusivo**: amostrar tabelas independentemente "
            "orfana FKs que existem no dado completo, gerando FK_DEGRADADA falsa. "
            "**Para decidir o append em produção, rode com `max_rows=None`.**"
        ))

    # inconsistências primeiro — é o que importa
    _sec("Inconsistências encontradas", "🚨")
    df_inc = resultado["inconsistencias"]
    if df_inc is not None and len(df_inc):
        ordem = {"CRÍTICO": 0, "ALERTA": 1}
        df_inc = df_inc.assign(_o=df_inc["severidade"].map(ordem)).sort_values(
            ["_o", "tipo"]).drop(columns="_o")
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
            "Estrutura e relacionamentos inferidos do original foram preservados "
            "no sintético.</span>"
        ))

    _sec("Tipos de dado (original vs sintético)", "🧬")
    df_t = resultado["tipos"]
    if df_t is not None and len(df_t):
        ruins = df_t[df_t["status"] != "OK"]
        display(Markdown(
            f"{len(df_t)} coluna(s) comparada(s) · "
            f"**{len(ruins)}** divergente(s)/ausente(s)/extra(s)."
        ))
        mostrar = ruins if len(ruins) else df_t
        display(_style(mostrar, "status"))
    else:
        display(Markdown("_Sem colunas comparadas._"))

    _sec("Chaves primárias descobertas", "🔑")
    df_pk = resultado["pks"]
    if df_pk is not None and len(df_pk):
        display(Markdown(
            "Cada linha é uma coluna que é **única + não-nula no ORIGINAL** "
            "(logo, PK de fato). `pk_no_sintetico=False` significa que o "
            "sintetizador **quebrou** uma chave que existia."
        ))
        display(_style(df_pk, "status"))
    else:
        display(Markdown("_Nenhuma PK de coluna única detectada no original._"))

    _sec("Chaves estrangeiras descobertas (por inclusão)", "🔗")
    df_fk = resultado["fks"]
    if df_fk is not None and len(df_fk):
        display(Markdown(
            "Cada linha é uma relação de **inclusão total observada no ORIGINAL** "
            "(valores do filho ⊆ chave do pai) — uma FK de fato, mesmo que "
            "ninguém a tenha declarado. `incl_sintetico < incl_original` = a "
            "sintetização **degradou** o relacionamento (surgiram órfãos)."
        ))
        display(Markdown(
            "> ℹ️ Inclusão é heurística: pode haver falso positivo (colunas de "
            "código que compartilham domínio). Revise as candidatas; o que importa "
            "é nenhuma cair do original para o sintético."
        ))
        display(_style(df_fk, "status"))
    else:
        display(Markdown("_Nenhuma relação de inclusão detectada no original._"))
