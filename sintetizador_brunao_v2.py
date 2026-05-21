# ============================================================
# 9. Configuração genérica — sem tabelas hardcoded
# ============================================================

SpecConfig = Mapping[str, Mapping[str, Any]]
TablePaths = Mapping[str, str]


def _as_tuple(
    value: Any,
    *,
    field_name: str,
    table_name: str,
    required: bool = True,
) -> Tuple[str, ...]:
    """
    Normaliza campos de configuração para tupla de strings.

    Aceita:
        "COLUNA"
        ["COLUNA_1", "COLUNA_2"]
        ("COLUNA_1", "COLUNA_2")
    """
    if value is None:
        if required:
            raise ValueError(
                f"Campo `{field_name}` é obrigatório na tabela `{table_name}`."
            )
        return tuple()

    if isinstance(value, str):
        value = value.strip()
        if not value and required:
            raise ValueError(
                f"Campo `{field_name}` está vazio na tabela `{table_name}`."
            )
        return (value,) if value else tuple()

    if isinstance(value, (list, tuple)):
        out = []

        for item in value:
            if item is None:
                raise ValueError(
                    f"Campo `{field_name}` possui valor None na tabela `{table_name}`."
                )

            item_str = str(item).strip()

            if not item_str:
                raise ValueError(
                    f"Campo `{field_name}` possui valor vazio na tabela `{table_name}`."
                )

            out.append(item_str)

        if required and not out:
            raise ValueError(
                f"Campo `{field_name}` está vazio na tabela `{table_name}`."
            )

        return tuple(out)

    raise TypeError(
        f"Campo `{field_name}` da tabela `{table_name}` deve ser string, list ou tuple. "
        f"Recebido: {type(value).__name__}"
    )


def _as_bool(value: Any, *, default: bool = False) -> bool:
    """
    Converte valores comuns de configuração para bool.
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value_norm = value.strip().lower()

        if value_norm in {"true", "1", "yes", "y", "sim", "s"}:
            return True

        if value_norm in {"false", "0", "no", "n", "nao", "não"}:
            return False

        raise ValueError(f"Valor booleano inválido: {value!r}")

    return bool(value)


def _get_parent_pk_from_config(
    specs_config: SpecConfig,
    *,
    parent_table: str,
    child_table: str,
) -> Tuple[str, ...]:
    """
    Permite omitir parent_columns na FK.
    Se parent_columns não for informado, usa pk_cols da tabela pai.
    """
    if parent_table not in specs_config:
        raise ValueError(
            f"A tabela filha `{child_table}` referencia `{parent_table}`, "
            f"mas `{parent_table}` não existe em specs_config."
        )

    parent_raw = specs_config[parent_table]

    if "pk_cols" not in parent_raw:
        raise ValueError(
            f"Não foi possível inferir parent_columns da FK em `{child_table}`. "
            f"A tabela pai `{parent_table}` não possui `pk_cols` em specs_config."
        )

    return _as_tuple(
        parent_raw["pk_cols"],
        field_name="pk_cols",
        table_name=parent_table,
        required=True,
    )


def _build_foreign_keys_from_config(
    table_name: str,
    raw_fks: Any,
    specs_config: SpecConfig,
) -> Tuple[ForeignKeySpec, ...]:
    """
    Cria ForeignKeySpec a partir de configuração declarativa.

    Formatos aceitos:

    foreign_keys=[
        {
            "columns": ["COLUNA_FK"],
            "parent_table": "tabela_pai",
            "parent_columns": ["COLUNA_PK_PAI"]
        }
    ]

    Se parent_columns for omitido, usa pk_cols da tabela pai.
    """
    if raw_fks is None:
        return tuple()

    if isinstance(raw_fks, ABCMapping):
        raw_fks = [raw_fks]

    if not isinstance(raw_fks, (list, tuple)):
        raise TypeError(
            f"`foreign_keys` da tabela `{table_name}` deve ser lista/tupla de dicts."
        )

    fks: List[ForeignKeySpec] = []

    for i, raw_fk in enumerate(raw_fks):
        if isinstance(raw_fk, ForeignKeySpec):
            fks.append(raw_fk)
            continue

        if not isinstance(raw_fk, ABCMapping):
            raise TypeError(
                f"FK #{i} da tabela `{table_name}` deve ser dict ou ForeignKeySpec."
            )

        parent_table = str(raw_fk.get("parent_table", "")).strip()

        if not parent_table:
            raise ValueError(
                f"FK #{i} da tabela `{table_name}` precisa informar `parent_table`."
            )

        columns = _as_tuple(
            raw_fk.get("columns"),
            field_name="foreign_keys.columns",
            table_name=table_name,
            required=True,
        )

        parent_columns_raw = raw_fk.get("parent_columns")

        if parent_columns_raw is None:
            parent_columns = _get_parent_pk_from_config(
                specs_config,
                parent_table=parent_table,
                child_table=table_name,
            )
        else:
            parent_columns = _as_tuple(
                parent_columns_raw,
                field_name="foreign_keys.parent_columns",
                table_name=table_name,
                required=True,
            )

        fks.append(
            ForeignKeySpec(
                columns=columns,
                parent_table=parent_table,
                parent_columns=parent_columns,
            )
        )

    return tuple(fks)


def build_specs_from_config(
    specs_config: SpecConfig,
    *,
    postprocessors: Optional[Mapping[str, PostProcessor]] = None,
    default_static: bool = False,
) -> Dict[str, TableSpec]:
    """
    Monta specs de qualquer conjunto de tabelas, sem hardcode.

    specs_config esperado:

    {
        "nome_da_tabela": {
            "pk_cols": ["id"],
            "static": False,
            "foreign_keys": [
                {
                    "columns": ["fk_id"],
                    "parent_table": "tabela_pai",
                    "parent_columns": ["id"]
                }
            ]
        }
    }

    Observação:
        postprocessors continua existindo, mas fica fora do motor genérico.
        Se uma tabela específica precisar recompor campos derivados, você injeta
        a função por fora via postprocessors={"nome_tabela": funcao}.
    """
    if not specs_config:
        raise ValueError("`specs_config` está vazio.")

    postprocessors = postprocessors or {}

    specs: Dict[str, TableSpec] = {}

    for table_name, raw_spec in specs_config.items():
        if not isinstance(raw_spec, ABCMapping):
            raise TypeError(
                f"Configuração da tabela `{table_name}` deve ser um dict."
            )

        pk_cols = _as_tuple(
            raw_spec.get("pk_cols"),
            field_name="pk_cols",
            table_name=table_name,
            required=True,
        )

        foreign_keys = _build_foreign_keys_from_config(
            table_name=table_name,
            raw_fks=raw_spec.get("foreign_keys"),
            specs_config=specs_config,
        )

        static = _as_bool(
            raw_spec.get("static"),
            default=default_static,
        )

        specs[table_name] = TableSpec(
            name=table_name,
            pk_cols=pk_cols,
            foreign_keys=foreign_keys,
            static=static,
            postprocess=postprocessors.get(table_name),
        )

    return specs


# ============================================================
# 10. Leitura genérica de tabelas
# ============================================================

def read_table_generic(
    spark: SparkSession,
    path: str,
    *,
    file_format: str = "parquet",
    options: Optional[Mapping[str, Any]] = None,
) -> DataFrame:
    """
    Lê uma tabela sem assumir nome, schema ou formato fixo.

    Formatos comuns:
        parquet
        csv
        delta
        json
        orc
    """
    fmt = file_format.strip().lower()
    read_options = dict(options or {})

    if fmt == "csv":
        read_options.setdefault("header", True)
        read_options.setdefault("inferSchema", True)

    reader = spark.read

    for key, value in read_options.items():
        reader = reader.option(key, value)

    return reader.format(fmt).load(path)


def read_tables_from_paths(
    spark: SparkSession,
    table_paths: TablePaths,
    *,
    default_format: str = "parquet",
    table_formats: Optional[Mapping[str, str]] = None,
    default_options: Optional[Mapping[str, Any]] = None,
    table_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, DataFrame]:
    """
    Lê N tabelas a partir de um dict nome_tabela -> path.

    Não existe nenhuma tabela hardcoded aqui.
    """
    if not table_paths:
        raise ValueError("`table_paths` está vazio.")

    table_formats = table_formats or {}
    table_options = table_options or {}
    default_options = default_options or {}

    tables: Dict[str, DataFrame] = {}

    for table_name, path in table_paths.items():
        fmt = table_formats.get(table_name, default_format)

        options = dict(default_options)
        options.update(table_options.get(table_name, {}))

        tables[table_name] = read_table_generic(
            spark,
            path,
            file_format=fmt,
            options=options,
        )

    return tables


# ============================================================
# 11. Volume genérico por tabela
# ============================================================

def build_n_rows_by_table_generic(
    tables: Mapping[str, DataFrame],
    *,
    fixed_n_rows_by_table: Optional[Mapping[str, int]] = None,
    scale_by_table: Optional[Mapping[str, float]] = None,
    default_scale: float = 1.0,
) -> Optional[Dict[str, int]]:
    """
    Monta n_rows_by_table sem hardcode.

    Regras:
        - Se fixed_n_rows_by_table tiver valor para a tabela, usa esse valor.
        - Caso contrário, usa count(original) * escala.
        - Se não houver fixed nem escala diferente de 1.0, retorna None,
          deixando synthesize_multitable_spark usar o volume original.
    """
    fixed_n_rows_by_table = dict(fixed_n_rows_by_table or {})
    scale_by_table = dict(scale_by_table or {})

    if not fixed_n_rows_by_table and not scale_by_table and float(default_scale) == 1.0:
        return None

    n_rows: Dict[str, int] = {}

    for table_name, df in tables.items():
        if table_name in fixed_n_rows_by_table:
            target = int(fixed_n_rows_by_table[table_name])
        else:
            scale = float(scale_by_table.get(table_name, default_scale))
            source_count = df.count()
            target = int(round(source_count * scale))

        if target < 0:
            raise ValueError(
                f"n_rows calculado para `{table_name}` ficou negativo: {target}."
            )

        n_rows[table_name] = target

    return n_rows


# ============================================================
# 12. Escrita genérica
# ============================================================

def save_tables_generic(
    tables: Mapping[str, DataFrame],
    base_path: str,
    *,
    output_format: str = "parquet",
    mode: str = "overwrite",
    coalesce: Optional[int] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Salva qualquer quantidade de tabelas sem nomes hardcoded.
    """
    fmt = output_format.strip().lower()
    options = dict(options or {})

    if fmt == "csv":
        options.setdefault("header", True)

    for table_name, df in tables.items():
        out_df = df.coalesce(coalesce) if coalesce is not None else df

        writer = out_df.write.mode(mode)

        for key, value in options.items():
            writer = writer.option(key, value)

        writer.format(fmt).save(f"{base_path}/{table_name}")


# ============================================================
# 13. Runner genérico para DataFrames já carregados
# ============================================================

def run_synthesis_from_tables(
    tables: Mapping[str, DataFrame],
    specs_config: SpecConfig,
    *,
    postprocessors: Optional[Mapping[str, PostProcessor]] = None,
    n_rows_by_table: Optional[Mapping[str, int]] = None,
    scale_by_table: Optional[Mapping[str, float]] = None,
    default_scale: float = 1.0,
    seed: int = 42,
    append_after_max_pk: bool = True,
    validate_mode: ValidateMode = "full",
    nullable_fk_policy: NullableFkPolicy = "allow_any_null",
    broadcast_fk_counts: bool = False,
    storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    verbose: bool = False,
    show_diagnostics: bool = True,
) -> Dict[str, DataFrame]:
    """
    Executa o gerador para qualquer conjunto de DataFrames.

    Esse é o runner principal quando você já tem as tabelas carregadas.
    """
    specs = build_specs_from_config(
        specs_config,
        postprocessors=postprocessors,
    )

    effective_n_rows = build_n_rows_by_table_generic(
        tables,
        fixed_n_rows_by_table=n_rows_by_table,
        scale_by_table=scale_by_table,
        default_scale=default_scale,
    )

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
    )

    if show_diagnostics:
        print("\n>>> Diagnóstico PRIMARY KEYS")
        validate_primary_keys(synthetic, specs).show(truncate=False)

        print("\n>>> Diagnóstico FOREIGN KEYS")
        validate_foreign_keys(
            synthetic,
            specs,
            nullable_fk_policy=nullable_fk_policy,
        ).show(truncate=False)

    return synthetic


# ============================================================
# 14. Runner genérico para paths
# ============================================================

def run_synthesis_from_paths(
    spark: SparkSession,
    table_paths: TablePaths,
    specs_config: SpecConfig,
    *,
    default_input_format: str = "parquet",
    table_formats: Optional[Mapping[str, str]] = None,
    default_read_options: Optional[Mapping[str, Any]] = None,
    table_read_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
    postprocessors: Optional[Mapping[str, PostProcessor]] = None,
    n_rows_by_table: Optional[Mapping[str, int]] = None,
    scale_by_table: Optional[Mapping[str, float]] = None,
    default_scale: float = 1.0,
    seed: int = 42,
    append_after_max_pk: bool = True,
    validate_mode: ValidateMode = "full",
    nullable_fk_policy: NullableFkPolicy = "allow_any_null",
    broadcast_fk_counts: bool = False,
    storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    verbose: bool = False,
    show_diagnostics: bool = True,
    save_path: Optional[str] = None,
    save_format: str = "parquet",
    save_mode: str = "overwrite",
    save_coalesce: Optional[int] = None,
) -> Dict[str, DataFrame]:
    """
    Lê, sintetiza, valida e opcionalmente salva N tabelas.

    Não possui:
        - nome de tabela hardcoded;
        - nome de coluna hardcoded;
        - quantidade fixa de tabelas;
        - formato fixo obrigatório.
    """
    tables = read_tables_from_paths(
        spark,
        table_paths,
        default_format=default_input_format,
        table_formats=table_formats,
        default_options=default_read_options,
        table_options=table_read_options,
    )

    synthetic = run_synthesis_from_tables(
        tables=tables,
        specs_config=specs_config,
        postprocessors=postprocessors,
        n_rows_by_table=n_rows_by_table,
        scale_by_table=scale_by_table,
        default_scale=default_scale,
        seed=seed,
        append_after_max_pk=append_after_max_pk,
        validate_mode=validate_mode,
        nullable_fk_policy=nullable_fk_policy,
        broadcast_fk_counts=broadcast_fk_counts,
        storage_level=storage_level,
        verbose=verbose,
        show_diagnostics=show_diagnostics,
    )

    if save_path is not None:
        save_tables_generic(
            synthetic,
            save_path,
            output_format=save_format,
            mode=save_mode,
            coalesce=save_coalesce,
        )

    return synthetic