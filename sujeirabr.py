from synthetic_multitable_spark_v3 import (
    run_synthesis_from_paths,
    validate_primary_keys,
    validate_foreign_keys,
    postprocess_instrumento,   # opcional: recompõe COD_IF/COD_ISIN do instrumento
    build_specs_for_base,      # (apenas se quiser as specs prontas da base CDB)
)


table_paths = {
    "tipo_if": "/caminho/tipo_if",
    "instrumento": "/caminho/instrumento",
    "operacao": "/caminho/operacao",
}

from pyspark import StorageLevel

synthetic = run_synthesis_from_paths(
    spark=spark,
    table_paths=table_paths,
    specs_config=specs_config,
    default_input_format="parquet",
    scale_factor=10,                       # ou use n_rows_by_table={...}
    seed=42,
    append_after_max_pk=True,
    validate_mode="full",
    nullable_fk_policy="allow_any_null",
    broadcast_fk_counts=False,
    storage_level=StorageLevel.MEMORY_AND_DISK,
    postprocess_by_table={"instrumento": postprocess_instrumento},
    save_path="/caminho/saida",           # opcional; remova p/ não salvar
    save_format="parquet",
    verbose=True,
)


# n_rows_by_table = {
#     "tipo_if": 5,
#     "instrumento": 10_000,
#     "operacao": 500_000,
# }
# synthetic = run_synthesis_from_paths(spark, table_paths, specs_config,
#     n_rows_by_table=n_rows_by_table, validate_mode="full", verbose=True)
tipo_if_synth     = synthetic["tipo_if"]
instrumento_synth = synthetic["instrumento"]
operacao_synth    = synthetic["operacao"]
