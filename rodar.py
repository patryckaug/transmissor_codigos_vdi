import importlib
import teste_sintetizador_v1
importlib.reload(teste_sintetizador_v1)

df_output = teste_sintetizador_v1.synthesize_table_v1(
    sdf=df,
    n_rows=30000,
    key_cols=[
        'NUM_CARTEIRA_COMITENTE','NUM_IF','NUM_ID_ENTIDADE','NUM_ID_ENTIDADE',
        'COD_TIPO_POSICAO_CARTEIRA','NUM_SISTEMA','NUM_CONTA_PARTICIPANTE','NUM_CONTA'
    ],
    seed=42,
    spark=spark,
)

df_output.show(10)
print(f"Total: {df_output.count()} linhas")