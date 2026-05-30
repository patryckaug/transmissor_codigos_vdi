"""
specs_config_builder.py
=======================

Gera, valida e blinda o `specs_config` consumido pelo
`synthetic_multitable_spark_v3.py` (Sintetizador Relacional v3).

Filosofia de projeto
---------------------
O objetivo é montar um `specs_config` em que TODO relacionamento foi validado
contra os dados reais — não apenas declarado no metadado. Para isso usamos um
pipeline em camadas, da fonte mais confiável para a menos confiável:

    1. METADADO (fonte primária, determinística)
       Lê METADADO_CDB_SIMPLIFICADO_PK_E_FK.csv (ou .xlsx) e extrai:
         - PKs   -> colunas com IS_PRIMARY_KEY verdadeiro
         - FKs   -> colunas com IS_FOREIGN_KEY verdadeiro + FK_REF_TABLE/FK_REF_COLUMN

    2. SCHEMA REAL (determinístico)
       Confere, contra o DataFrame Spark de cada tabela, se as colunas de PK/FK
       realmente existem.

    3. INTEGRIDADE NOS DADOS (determinístico — o "show")
       Para cada PK valida unicidade/nulos. Para cada FK valida integridade
       referencial real (órfãos, taxa de match) com join no Spark.
       >>> Esta camada é a ÚNICA fonte de verdade sobre "o relacionamento existe?".

    4. LLM (opcional, só para os buracos — gerador de HIPÓTESES)
       Onde o metadado é omisso/ambíguo (FK sem pai, ou relacionamento não
       declarado que o nome denuncia), o LLM (OCI Generative AI) PROPÕE pais
       candidatos. Cada proposta volta para a camada 3 e só entra se passar.
       O LLM nunca decide validade; ele só sugere o que testar.

    5. MONTAGEM
       Emite o dict `specs_config` final (apenas relacionamentos aprovados),
       resolve conflitos de coluna em >1 FK, e devolve um relatório pandas.

Por que o LLM não valida
-------------------------
Um `left_anti` join responde com certeza se há órfãos. Delegar isso a um LLM
seria mais frágil e mais caro. O LLM é ótimo para inferir intenção a partir de
nomes ("ID_CLIENTE -> CLIENTE.ID_CLIENTE"); a verdade sobre os dados vem do Spark.

Uso típico
----------
    from specs_config_builder import build_validated_specs_config, load_spark_tables

    tables = load_spark_tables(
        spark,
        {
            "CLIENTE":  "s3://.../cliente",
            "CONTRATO": "s3://.../contrato",
            "PARCELA":  "s3://.../parcela",
        },
        fmt="parquet",
    )

    result = build_validated_specs_config(
        meta="METADADO_CDB_SIMPLIFICADO_PK_E_FK.csv",
        tables=tables,
        show_samples=True,
        use_llm=True,
        compartment_id=COMPARTMENT_ID,   # mesmos do seu conexao-genai.ipynb
        model_id=MODEL_ID,
        llm_config_path="new_oci/config",
    )

    result.report               # pandas DataFrame com tudo que foi decidido
    specs_config = result.specs_config

    # Plugue direto no sintetizador:
    synthetic = run_synthesis_from_tables(tables, specs_config, scale_factor=1.0)

Premissas
---------
- Cada coluna FK do metadado é tratada como uma FK de coluna única
  (columns=[col], parent_columns=[ref_col]). FKs compostas reais devem ser
  passadas via `extra_relationships` ou confirmadas pelo LLM.
- `tables` é um dict {nome_tabela: DataFrame Spark} já carregado. As chaves
  precisam casar com TABLE_NAME do metadado (case-insensitive é tratado).
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


# ============================================================
# 0. Helpers de normalização de flags e strings
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
    return pd.read_csv(path)


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
                f"Metadado precisa da coluna `{required}`. Colunas vistas: {list(df.columns)}"
            )

    for optional in (
        "IS_PRIMARY_KEY", "IS_FOREIGN_KEY", "FK_REF_TABLE", "FK_REF_COLUMN",
        "COLUMN_ID", "IS_INDEXED", "DATA_TYPE",
    ):
        if optional not in df.columns:
            df[optional] = None

    for str_col in ("TABLE_NAME", "COLUMN_NAME", "FK_REF_TABLE", "FK_REF_COLUMN", "DATA_TYPE"):
        df[str_col] = df[str_col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    return df


# ============================================================
# 1. Camada 1 — Metadado -> specs candidatas (determinístico)
# ============================================================

def build_candidate_specs_from_metadata(meta: Any) -> Dict[str, Dict[str, Any]]:
    """
    Extrai PKs e FKs candidatas do metadado, por tabela.

    Retorna:
        {
          "TABELA": {
             "pk_cols": [...],
             "columns": [...],            # todas as colunas conhecidas no metadado
             "_fk_candidates": [
                 {"columns": ["COL"], "parent_table": "PAI" | None,
                  "parent_columns": ["REF"] | None, "source": "metadata"}
             ]
          }
        }
    """
    df = _normalize_meta(meta)
    specs: Dict[str, Dict[str, Any]] = {}

    # dict.fromkeys preserva ordem e remove duplicatas
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
                    "source": "metadata",
                }
            )

        specs[table] = {
            "pk_cols": pk_cols,
            "columns": [_clean(r["COLUMN_NAME"]) for r in rows if _clean(r["COLUMN_NAME"])],
            "_fk_candidates": fk_candidates,
        }

    return specs


# ============================================================
# 2. Carregamento e inspeção ("show") das tabelas
# ============================================================

def load_spark_tables(
    spark: SparkSession,
    table_paths: Mapping[str, str],
    *,
    fmt: str = "parquet",
    options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, DataFrame]:
    """Carrega tabelas Spark a partir de caminhos (mesmo espírito do sintetizador)."""
    options = dict(options or {})
    fmt = (fmt or "parquet").lower()
    out: Dict[str, DataFrame] = {}

    for name, path in table_paths.items():
        reader = spark.read
        for k, v in options.items():
            reader = reader.option(k, v)
        if fmt == "parquet":
            out[name] = reader.parquet(path)
        elif fmt == "orc":
            out[name] = reader.orc(path)
        elif fmt == "csv":
            out[name] = (
                reader.option("header", options.get("header", True))
                .option("inferSchema", options.get("inferSchema", True))
                .csv(path)
            )
        else:
            out[name] = reader.format(fmt).load(path)
    return out


def _resolve_table_name(name: str, tables: Mapping[str, DataFrame]) -> Optional[str]:
    """Casa o nome do metadado com a chave real em `tables` (case-insensitive)."""
    if name in tables:
        return name
    lower_map = {k.lower(): k for k in tables}
    return lower_map.get(name.lower())


def inspect_tables(
    tables: Mapping[str, DataFrame],
    *,
    show_n: int = 5,
    do_show: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Imprime schema + amostra de cada tabela e devolve metadados estruturais.

    O `df.show()` é só para inspeção humana; as decisões de validade vêm dos
    checks programáticos das próximas funções.
    """
    info: Dict[str, Dict[str, Any]] = {}
    for name, df in tables.items():
        cols = df.columns
        if do_show:
            print(f"\n{'=' * 70}\nTABELA: {name}  | colunas: {len(cols)}\n{'=' * 70}")
            df.printSchema()
            df.show(show_n, truncate=False)
        info[name] = {"columns": cols, "dtypes": dict(df.dtypes)}
    return info


# ============================================================
# 3. Camada 3 — Validação de integridade nos DADOS
#    (esta é a fonte de verdade sobre os relacionamentos)
# ============================================================

def validate_pk(df: DataFrame, pk_cols: Sequence[str]) -> Dict[str, Any]:
    """Valida unicidade e ausência de nulos de uma PK (simples ou composta)."""
    missing = [c for c in pk_cols if c not in df.columns]
    if missing:
        return {"is_valid_pk": False, "reason": f"colunas de PK ausentes: {missing}",
                "total_rows": None, "distinct_pk": None, "null_pk_rows": None,
                "duplicate_pk_rows": None}

    total = df.count()
    distinct = df.select(*pk_cols).dropDuplicates().count()
    null_cond = reduce(lambda a, b: a | b, [F.col(c).isNull() for c in pk_cols])
    null_rows = df.where(null_cond).count()

    return {
        "total_rows": int(total),
        "distinct_pk": int(distinct),
        "null_pk_rows": int(null_rows),
        "duplicate_pk_rows": int(total - distinct),
        "is_valid_pk": bool(distinct == total and null_rows == 0 and total > 0),
        "reason": None,
    }


def validate_fk(
    child_df: DataFrame,
    child_cols: Sequence[str],
    parent_df: DataFrame,
    parent_cols: Sequence[str],
    *,
    ignore_null: bool = True,
    max_orphan_ratio: float = 0.0,
) -> Dict[str, Any]:
    """
    Valida integridade referencial REAL de uma FK contra os dados de origem.

    Critério de validade:
        - existe pelo menos 1 match com o pai, E
        - a proporção de chaves órfãs <= max_orphan_ratio (default 0 = sem órfãos).

    `max_orphan_ratio=0.0` reproduz o comportamento do sintetizador, que ignora
    a FK se houver qualquer órfão.
    """
    missing_child = [c for c in child_cols if c not in child_df.columns]
    missing_parent = [c for c in parent_cols if c not in parent_df.columns]
    if missing_child or missing_parent:
        return {
            "valid": False,
            "reason": f"colunas ausentes child={missing_child} parent={missing_parent}",
            "distinct_child_keys": None, "matched": None, "orphans": None,
            "match_rate": 0.0, "orphan_ratio": 1.0,
        }

    sel = child_df.select(*child_cols)
    if ignore_null:
        not_all_null = reduce(lambda a, b: a | b, [F.col(c).isNotNull() for c in child_cols])
        sel = sel.where(not_all_null)
    child_keys = sel.dropDuplicates()

    total_keys = child_keys.count()
    if total_keys == 0:
        return {
            "valid": False,
            "reason": "nenhuma chave FK não-nula para validar",
            "distinct_child_keys": 0, "matched": 0, "orphans": 0,
            "match_rate": 0.0, "orphan_ratio": 0.0,
        }

    parent_keys = parent_df.select(
        *[F.col(p).alias(c) for c, p in zip(child_cols, parent_cols)]
    ).dropDuplicates()

    orphans = child_keys.join(parent_keys, on=list(child_cols), how="left_anti").count()
    matched = total_keys - orphans
    match_rate = matched / total_keys
    orphan_ratio = orphans / total_keys

    valid = bool(matched > 0 and orphan_ratio <= max_orphan_ratio)
    reason = None
    if matched == 0:
        reason = "nenhum match com o pai (relacionamento provavelmente inexistente)"
    elif orphan_ratio > max_orphan_ratio:
        reason = (
            f"{orphans} chave(s) órfã(s) de {total_keys} "
            f"(orphan_ratio={orphan_ratio:.3f} > {max_orphan_ratio})"
        )

    return {
        "valid": valid,
        "reason": reason,
        "distinct_child_keys": int(total_keys),
        "matched": int(matched),
        "orphans": int(orphans),
        "match_rate": float(round(match_rate, 4)),
        "orphan_ratio": float(round(orphan_ratio, 4)),
    }


def _infer_pk_candidates(df: DataFrame, *, prefer_substr: str = "ID") -> List[str]:
    """
    Inferência de PK a partir dos dados quando o metadado não traz PK.

    Retorna colunas que são únicas E não-nulas. Prioriza nomes contendo `prefer_substr`.
    Custo: 1 distinct + 1 count de null por coluna (use com parcimônia em tabelas largas).
    """
    total = df.count()
    if total == 0:
        return []
    candidates: List[str] = []
    for c in df.columns:
        distinct = df.select(c).dropDuplicates().count()
        if distinct != total:
            continue
        nulls = df.where(F.col(c).isNull()).count()
        if nulls == 0:
            candidates.append(c)
    candidates.sort(key=lambda c: (prefer_substr.lower() not in c.lower(), c))
    return candidates


# ============================================================
# 4. Camada 4 — LLM (OCI GenAI) como gerador de hipóteses
# ============================================================

def _extract_text_from_oci_response(response: Any) -> str:
    """
    Extrai texto da resposta do OCI GenAI, cobrindo os dois formatos:
      - GenericChatRequest -> data.chat_response.choices[0].message.content[i].text
      - CohereChatRequest  -> data.chat_response.text
    """
    data = getattr(response, "data", response)
    chat_response = getattr(data, "chat_response", None)
    if chat_response is None:
        return str(data)

    choices = getattr(chat_response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content:
            parts = [getattr(c, "text", None) for c in content]
            parts = [p for p in parts if p]
            if parts:
                return "\n".join(parts)

    text = getattr(chat_response, "text", None)
    if text:
        return text
    return str(chat_response)


def _make_oci_client(config_path: str = "new_oci/config"):
    """Cria o GenerativeAIInferenceClient (import lazy: só exige `oci` se usar LLM)."""
    import oci
    from oci.generative_ai_inference import GenerativeAIInferenceClient

    config = oci.config.from_file(file_location=config_path)
    return GenerativeAIInferenceClient(config)


def _llm_chat(
    client,
    prompt: str,
    *,
    compartment_id: str,
    model_id: str,
    max_tokens: int = 2000,
    temperature: float = 0.0,
) -> str:
    """Chamada de chat seguindo o seu padrão OCI (GenericChatRequest)."""
    from oci.generative_ai_inference.models import (
        ChatDetails, OnDemandServingMode, Message, TextContent, GenericChatRequest,
    )

    chat_request = GenericChatRequest(
        messages=[Message(role="USER", content=[TextContent(text=prompt)])],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
    )
    chat_details = ChatDetails(
        compartment_id=compartment_id,
        serving_mode=OnDemandServingMode(model_id=model_id),
        chat_request=chat_request,
    )
    response = client.chat(chat_details)
    return _extract_text_from_oci_response(response)


def _parse_json_block(text: str) -> Optional[dict]:
    """Faz parse robusto de JSON, removendo cercas markdown e texto ao redor."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start : end + 1])
        except Exception:
            return None
    return None


def _build_llm_prompt(schema: dict, unresolved: List[dict]) -> str:
    return (
        "Você é um especialista em modelagem de dados relacionais.\n"
        "Recebe o SCHEMA (tabelas, colunas e suas primary keys) e uma lista de "
        "COLUNAS_FK_NAO_RESOLVIDAS (colunas marcadas como FK mas sem tabela/coluna pai).\n\n"
        "Tarefa:\n"
        "1) Para cada item de COLUNAS_FK_NAO_RESOLVIDAS, indique a tabela e a coluna PAI "
        "mais provável, usando convenções de nome (ex.: ID_CLIENTE -> CLIENTE.ID_CLIENTE), "
        "compatibilidade de tipo e a PK do pai.\n"
        "2) Você PODE sugerir relacionamentos não declarados se o padrão de nome for forte.\n"
        "3) O pai sugerido deve existir no SCHEMA e a coluna pai deve ser PK desse pai "
        "(ou claramente uma chave).\n"
        "4) Se não houver candidato confiável, NÃO invente: omita o item.\n\n"
        f"SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"COLUNAS_FK_NAO_RESOLVIDAS:\n{json.dumps(unresolved, ensure_ascii=False)}\n\n"
        "Responda APENAS com JSON válido, sem texto antes ou depois, no formato:\n"
        '{"relationships":[{"child_table":"T","child_columns":["c"],'
        '"parent_table":"P","parent_columns":["pk"],"confidence":0.0,"reason":"..."}]}'
    )


def llm_propose_relationships(
    schema: dict,
    unresolved: List[dict],
    *,
    config_path: str,
    compartment_id: str,
    model_id: str,
    max_tokens: int = 2000,
) -> List[dict]:
    """
    Pede ao LLM para propor pais de FKs não resolvidas. Robusto a falhas:
    qualquer erro retorna lista vazia (o pipeline segue só com o determinístico).
    """
    if not unresolved:
        return []
    try:
        client = _make_oci_client(config_path)
        prompt = _build_llm_prompt(schema, unresolved)
        raw = _llm_chat(
            client, prompt,
            compartment_id=compartment_id, model_id=model_id,
            max_tokens=max_tokens, temperature=0.0,
        )
        parsed = _parse_json_block(raw)
        if not parsed or "relationships" not in parsed:
            warnings.warn("LLM não retornou JSON utilizável; seguindo sem sugestões.", UserWarning)
            return []
        rels = parsed.get("relationships", [])
        return rels if isinstance(rels, list) else []
    except Exception as exc:  # falha de rede/credencial/SDK não pode derrubar o build
        warnings.warn(f"Falha ao consultar o LLM ({exc}); seguindo sem sugestões.", UserWarning)
        return []


# ============================================================
# 5. Orquestrador — monta e valida o specs_config
# ============================================================

@dataclass
class SpecsBuildResult:
    specs_config: Dict[str, Dict[str, Any]]
    report: pd.DataFrame
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    llm_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    pk_inferred: Dict[str, List[str]] = field(default_factory=dict)

    def summary(self) -> str:
        n_tables = len(self.specs_config)
        n_fks = sum(len(v.get("foreign_keys", [])) for v in self.specs_config.values())
        return (
            f"specs_config: {n_tables} tabela(s), {n_fks} FK(s) validada(s) | "
            f"{len(self.skipped)} relacionamento(s) descartado(s) | "
            f"{len(self.llm_suggestions)} sugestão(ões) do LLM avaliada(s)"
        )


def build_validated_specs_config(
    meta: Any,
    tables: Mapping[str, DataFrame],
    *,
    show_samples: bool = True,
    show_n: int = 5,
    ignore_null_fk: bool = True,
    max_orphan_ratio: float = 0.0,
    infer_missing_pk: bool = True,
    static_tables: Optional[Sequence[str]] = None,
    extra_relationships: Optional[List[dict]] = None,
    use_llm: bool = False,
    llm_config_path: str = "new_oci/config",
    compartment_id: Optional[str] = None,
    model_id: Optional[str] = None,
    llm_min_confidence: float = 0.55,
    verbose: bool = True,
) -> SpecsBuildResult:
    """
    Constrói um `specs_config` 100% validado contra os dados.

    Parâmetros principais
    ---------------------
    meta : caminho do metadado (.csv/.xlsx) ou pandas.DataFrame.
    tables : dict {nome_tabela: DataFrame Spark} já carregado.
    max_orphan_ratio : tolerância de órfãos por FK (0.0 = exige integridade total,
        igual ao sintetizador). Use, p.ex., 0.001 para dados levemente sujos.
    infer_missing_pk : se True, tenta inferir PK pelos dados quando o metadado
        não trouxer PK para uma tabela.
    static_tables : tabelas a marcar como static=True no specs_config.
    extra_relationships : lista de FKs adicionais a testar, no formato
        {"child_table","child_columns","parent_table","parent_columns"}.
        Útil para FKs compostas que o metadado coluna-a-coluna não expressa.
    use_llm : ativa a camada de sugestão via OCI GenAI para FKs não resolvidas.

    Retorna
    -------
    SpecsBuildResult com .specs_config (pronto p/ run_synthesis_from_tables),
    .report (pandas), .skipped, .llm_suggestions, .pk_inferred.
    """
    static_tables = set(static_tables or [])
    report_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    llm_suggestions: List[Dict[str, Any]] = []
    pk_inferred: Dict[str, List[str]] = {}

    # ---- Camada 1: metadado -> candidatas
    candidates = build_candidate_specs_from_metadata(meta)

    # ---- Escopo: interseção metadado x tabelas carregadas
    name_map: Dict[str, str] = {}  # nome_metadado -> chave_real_em_tables
    for meta_name in candidates:
        real = _resolve_table_name(meta_name, tables)
        if real is not None:
            name_map[meta_name] = real

    in_meta_not_loaded = [t for t in candidates if t not in name_map]
    loaded_not_in_meta = [t for t in tables if t.lower() not in {m.lower() for m in candidates}]
    if in_meta_not_loaded:
        warnings.warn(
            f"Tabelas no metadado sem DataFrame carregado (ignoradas): {in_meta_not_loaded}",
            UserWarning,
        )
    if loaded_not_in_meta:
        warnings.warn(
            f"Tabelas carregadas sem entrada no metadado (ignoradas): {loaded_not_in_meta}",
            UserWarning,
        )

    if not name_map:
        raise ValueError("Nenhuma tabela em comum entre o metadado e `tables`.")

    # Daqui para frente usamos SEMPRE o nome canônico = a chave real de `tables`.
    # Isso garante que as chaves de specs_config (e os parent_table das FKs) casem
    # exatamente com `tables`, que é o que o sintetizador exige (case-sensitive).
    cand_by_real: Dict[str, Dict[str, Any]] = {
        name_map[meta_name]: candidates[meta_name] for meta_name in name_map
    }
    scope = list(cand_by_real.keys())  # nomes canônicos (chaves reais de tables)

    # ---- "show" das tabelas em escopo
    scoped_tables = {real: tables[real] for real in scope}
    if show_samples:
        inspect_tables(scoped_tables, show_n=show_n, do_show=True)

    # ---- Camada 3a: validação de PK (e inferência se faltar)
    validated_pk: Dict[str, List[str]] = {}
    for meta_name in scope:
        df = scoped_tables[meta_name]
        pk_cols = [c for c in cand_by_real[meta_name]["pk_cols"] if c]

        if not pk_cols and infer_missing_pk:
            inferred = _infer_pk_candidates(df)
            if inferred:
                pk_cols = [inferred[0]]
                pk_inferred[meta_name] = inferred
                if verbose:
                    print(f"[PK] `{meta_name}` sem PK no metadado. Inferida pelos dados: {pk_cols} "
                          f"(outras candidatas: {inferred[1:] or 'nenhuma'})")

        if not pk_cols:
            skipped.append({"type": "table", "table": meta_name,
                            "reason": "sem PK no metadado e não foi possível inferir"})
            report_rows.append({"table": meta_name, "kind": "PK", "columns": "",
                                "status": "BLOQUEADO", "reason": "sem PK"})
            warnings.warn(
                f"Tabela `{meta_name}` excluída do specs_config: sem PK válida.",
                UserWarning,
            )
            continue

        pk_check = validate_pk(df, pk_cols)
        status = "ok" if pk_check["is_valid_pk"] else "alerta"
        if not pk_check["is_valid_pk"] and verbose:
            print(f"[PK] `{meta_name}` PK {pk_cols} não é única/sem nulos "
                  f"(distinct={pk_check['distinct_pk']} de {pk_check['total_rows']}, "
                  f"nulos={pk_check['null_pk_rows']}). Mantida (o sintetizador regenera a PK).")
        validated_pk[meta_name] = pk_cols
        report_rows.append({
            "table": meta_name, "kind": "PK", "columns": ",".join(pk_cols),
            "parent_table": "", "parent_columns": "",
            "total_rows": pk_check["total_rows"], "distinct_or_matched": pk_check["distinct_pk"],
            "nulls_or_orphans": pk_check["null_pk_rows"], "match_rate": "",
            "status": status, "reason": pk_check["reason"] or "",
        })

    # tabelas que sobreviveram (têm PK)
    valid_tables = set(validated_pk.keys())

    def _fk_check_and_record(child, child_cols, parent, parent_cols, source):
        """Roda validate_fk e registra no relatório. Retorna o dict de métricas ou None."""
        if parent not in valid_tables:
            skipped.append({"type": "fk", "child_table": child, "child_columns": child_cols,
                            "parent_table": parent, "parent_columns": parent_cols,
                            "source": source, "reason": "pai ausente/sem PK no escopo"})
            return None
        metrics = validate_fk(
            scoped_tables[child], child_cols, scoped_tables[parent], parent_cols,
            ignore_null=ignore_null_fk, max_orphan_ratio=max_orphan_ratio,
        )
        report_rows.append({
            "table": child, "kind": f"FK({source})", "columns": ",".join(child_cols),
            "parent_table": parent, "parent_columns": ",".join(parent_cols),
            "total_rows": metrics.get("distinct_child_keys"),
            "distinct_or_matched": metrics.get("matched"),
            "nulls_or_orphans": metrics.get("orphans"),
            "match_rate": metrics.get("match_rate"),
            "status": "ok" if metrics["valid"] else "descartado",
            "reason": metrics["reason"] or "",
        })
        if not metrics["valid"]:
            skipped.append({"type": "fk", "child_table": child, "child_columns": child_cols,
                            "parent_table": parent, "parent_columns": parent_cols,
                            "source": source, "reason": metrics["reason"]})
            return None
        return metrics

    # ---- Camada 3b: validação das FKs declaradas + coleta das não resolvidas
    accepted_fks: Dict[str, List[dict]] = {t: [] for t in valid_tables}
    unresolved_for_llm: List[dict] = []

    for child in valid_tables:
        df = scoped_tables[child]
        child_dtypes = dict(df.dtypes)
        for fk in cand_by_real[child]["_fk_candidates"]:
            child_cols = [c for c in fk["columns"] if c in df.columns]
            if not child_cols:
                skipped.append({"type": "fk", "child_table": child,
                                "child_columns": fk["columns"], "source": "metadata",
                                "reason": "coluna FK não existe no schema real"})
                continue

            parent = _clean(fk.get("parent_table"))
            parent_cols = fk.get("parent_columns")

            # resolve pai pelo nome do metadado -> escopo
            if parent:
                parent = parent if parent in valid_tables else (
                    next((m for m in valid_tables if m.lower() == parent.lower()), None)
                )

            # se o pai existe mas não veio parent_columns, usa a PK do pai
            if parent and not parent_cols:
                parent_cols = validated_pk.get(parent)

            if parent and parent_cols:
                metrics = _fk_check_and_record(child, child_cols, parent, list(parent_cols), "metadata")
                if metrics:
                    accepted_fks[child].append(
                        {"columns": child_cols, "parent_table": parent,
                         "parent_columns": list(parent_cols), "match_rate": metrics["match_rate"],
                         "orphans": metrics["orphans"]}
                    )
            else:
                # buraco: vai para o LLM (se habilitado)
                unresolved_for_llm.append({
                    "child_table": child,
                    "child_columns": child_cols,
                    "child_dtype": child_dtypes.get(child_cols[0]),
                    "metadata_ref_table": fk.get("parent_table"),
                    "metadata_ref_column": (fk.get("parent_columns") or [None])[0],
                })

    # FKs compostas / extras informadas manualmente também passam pela validação
    for fk in (extra_relationships or []):
        child = _resolve_table_name(str(fk["child_table"]), {t: None for t in valid_tables}) \
            or next((m for m in valid_tables if m.lower() == str(fk["child_table"]).lower()), None)
        parent = next((m for m in valid_tables if m.lower() == str(fk["parent_table"]).lower()), None)
        if not child or not parent:
            skipped.append({"type": "fk", "source": "extra", "reason": "tabela fora do escopo",
                            **fk})
            continue
        cc = list(fk["child_columns"])
        pc = list(fk["parent_columns"])
        metrics = _fk_check_and_record(child, cc, parent, pc, "extra")
        if metrics:
            accepted_fks[child].append({"columns": cc, "parent_table": parent,
                                        "parent_columns": pc, "match_rate": metrics["match_rate"],
                                        "orphans": metrics["orphans"]})

    # ---- Camada 4: LLM propõe pais para o que ficou sem resolução; re-valida tudo
    if use_llm and unresolved_for_llm:
        if not (compartment_id and model_id):
            warnings.warn("use_llm=True mas compartment_id/model_id não informados; pulando LLM.",
                          UserWarning)
        else:
            schema = {
                t: {"columns": scoped_tables[t].columns, "pk": validated_pk[t]}
                for t in valid_tables
            }
            proposals = llm_propose_relationships(
                schema, unresolved_for_llm,
                config_path=llm_config_path, compartment_id=compartment_id, model_id=model_id,
            )
            for prop in proposals:
                try:
                    child = next((m for m in valid_tables
                                  if m.lower() == str(prop["child_table"]).lower()), None)
                    parent = next((m for m in valid_tables
                                   if m.lower() == str(prop["parent_table"]).lower()), None)
                    conf = float(prop.get("confidence", 0.0))
                except Exception:
                    continue

                record = {**prop, "accepted": False,
                          "note": "abaixo da confiança mínima" if conf < llm_min_confidence else ""}
                if not child or not parent or conf < llm_min_confidence:
                    if not child or not parent:
                        record["note"] = "tabela sugerida fora do escopo"
                    llm_suggestions.append(record)
                    continue

                cc = list(prop["child_columns"])
                pc = list(prop["parent_columns"])
                metrics = _fk_check_and_record(child, cc, parent, pc, "llm")
                if metrics:
                    accepted_fks[child].append({"columns": cc, "parent_table": parent,
                                                "parent_columns": pc,
                                                "match_rate": metrics["match_rate"],
                                                "orphans": metrics["orphans"]})
                    record["accepted"] = True
                    record["note"] = f"validada nos dados (match_rate={metrics['match_rate']})"
                else:
                    record["note"] = "rejeitada na validação dos dados"
                llm_suggestions.append(record)

    # ---- Camada 5: resolve conflito de coluna em >1 FK e monta o dict final
    specs_config: Dict[str, Dict[str, Any]] = {}
    for table in valid_tables:
        # ordena por melhor match_rate / menos órfãos; uma coluna só pode estar em 1 FK
        ranked = sorted(accepted_fks[table], key=lambda x: (-x["match_rate"], x["orphans"]))
        used_cols: set = set()
        final_fks: List[Dict[str, Any]] = []
        for fk in ranked:
            if any(c in used_cols for c in fk["columns"]):
                skipped.append({"type": "fk", "child_table": table,
                                "child_columns": fk["columns"],
                                "parent_table": fk["parent_table"],
                                "reason": "coluna já usada em outra FK (conflito resolvido)"})
                continue
            used_cols.update(fk["columns"])
            final_fks.append({
                "columns": fk["columns"],
                "parent_table": fk["parent_table"],
                "parent_columns": fk["parent_columns"],
            })

        entry: Dict[str, Any] = {"pk_cols": validated_pk[table]}
        if final_fks:
            entry["foreign_keys"] = final_fks
        if table in static_tables or any(s.lower() == table.lower() for s in static_tables):
            entry["static"] = True
        specs_config[table] = entry

    report = pd.DataFrame(report_rows)
    result = SpecsBuildResult(
        specs_config=specs_config,
        report=report,
        skipped=skipped,
        llm_suggestions=llm_suggestions,
        pk_inferred=pk_inferred,
    )

    if verbose:
        print("\n" + "=" * 70)
        print(result.summary())
        print("=" * 70)
        if skipped:
            print(f"\n{len(skipped)} relacionamento(s) descartado(s). Veja result.skipped / result.report.")

    return result


# ============================================================
# Exemplo de uso (não executa na importação)
# ============================================================
#
# from specs_config_builder import build_validated_specs_config, load_spark_tables
#
# COMPARTMENT_ID = "ocid1.compartment.oc1..xxxx"
# MODEL_ID       = "ocid1.generativeaimodel.oc1.sa-saopaulo-1.xxxx"
#
# tables = load_spark_tables(spark, {
#     "CLIENTE":  "caminho/cliente",
#     "CONTRATO": "caminho/contrato",
#     "PARCELA":  "caminho/parcela",
# }, fmt="parquet")
#
# result = build_validated_specs_config(
#     meta="METADADO_CDB_SIMPLIFICADO_PK_E_FK.csv",
#     tables=tables,
#     show_samples=True,
#     max_orphan_ratio=0.0,        # exige integridade total (igual ao sintetizador)
#     use_llm=True,
#     compartment_id=COMPARTMENT_ID,
#     model_id=MODEL_ID,
#     llm_config_path="new_oci/config",
# )
#
# result.report          # tabela com tudo que foi validado/descartado
# specs_config = result.specs_config
#
# # plugue direto no sintetizador:
# synthetic = run_synthesis_from_tables(tables, specs_config, scale_factor=1.0)
