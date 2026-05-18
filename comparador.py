import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import datetime as dt
from typing import Optional, Sequence, Tuple, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ===========================================================
# Paleta (consistente com o deck)
# ===========================================================
C_ORIG  = "#0F2A47"   # navy — Original
C_SYNT  = "#06B6D4"   # cyan — Sintética
C_INK   = "#1E3A5F"
C_MUTED = "#64748B"
C_LINE  = "#E2E8F0"
C_BG    = "#FFFFFF"


# ===========================================================
# Helpers de tipo
# ===========================================================
def _kind(dt_):
    if isinstance(dt_, (T.StringType, T.BooleanType)):
        return "categorical"
    if isinstance(dt_, (T.ByteType, T.ShortType, T.IntegerType, T.LongType,
                        T.FloatType, T.DoubleType, T.DecimalType)):
        return "numeric"
    if isinstance(dt_, (T.DateType, T.TimestampType)):
        return "datetime"
    return "other"


# ===========================================================
# Agregações em Spark (não trazem dados, só agregados)
# ===========================================================
def _top_k(sdf: DataFrame, col: str, k: int = 8):
    nn = sdf.where(F.col(col).isNotNull())
    total = nn.count()
    if total == 0:
        return [], 0
    rows = (nn.groupBy(col).count()
              .orderBy(F.desc("count"))
              .limit(k).collect())
    return [(str(r[col]), r["count"] / total) for r in rows], total


def _freq_for_values(sdf: DataFrame, col: str, values: List[str]):
    """Frequência relativa das categorias `values` em sdf (mesmo col)."""
    nn = sdf.where(F.col(col).isNotNull())
    total = nn.count()
    if total == 0 or not values:
        return {v: 0.0 for v in values}
    rows = (nn.where(F.col(col).cast("string").isin(values))
              .groupBy(col).count().collect())
    return {str(r[col]): r["count"] / total for r in rows}


def _hist_numeric(sdf: DataFrame, col: str, n_bins: int,
                  shared_range: Optional[Tuple[float, float]] = None):
    nn = sdf.where(F.col(col).isNotNull())
    if shared_range:
        mn, mx = shared_range
    else:
        s = nn.select(F.min(col).alias("mn"), F.max(col).alias("mx")).collect()[0]
        mn, mx = s["mn"], s["mx"]
    if mn is None or mx is None or mn == mx:
        return np.array([0, 1]), np.zeros(n_bins)
    mn, mx = float(mn), float(mx)
    width = (mx - mn) / n_bins
    bucketed = nn.select(
        F.least(
            F.greatest(F.floor((F.col(col).cast("double") - F.lit(mn)) / F.lit(width)),
                       F.lit(0)),
            F.lit(n_bins - 1)
        ).alias("bucket")
    )
    counts = np.zeros(n_bins, dtype=float)
    for r in bucketed.groupBy("bucket").count().collect():
        counts[int(r["bucket"])] = r["count"]
    edges = np.linspace(mn, mx, n_bins + 1)
    return edges, counts


def _hist_datetime(sdf: DataFrame, col: str, n_bins: int,
                   shared_range: Optional[Tuple[float, float]] = None):
    epoch = sdf.where(F.col(col).isNotNull()) \
               .select(F.col(col).cast("timestamp").cast("long").alias("epoch"))
    if shared_range:
        mn, mx = shared_range
    else:
        s = epoch.select(F.min("epoch").alias("mn"), F.max("epoch").alias("mx")).collect()[0]
        mn, mx = s["mn"], s["mx"]
    if mn is None or mx is None or mn == mx:
        return np.array([0, 1]), np.zeros(n_bins)
    mn, mx = float(mn), float(mx)
    width = (mx - mn) / n_bins
    bucketed = epoch.select(
        F.least(
            F.greatest(F.floor((F.col("epoch") - F.lit(mn)) / F.lit(width)), F.lit(0)),
            F.lit(n_bins - 1)
        ).alias("bucket")
    )
    counts = np.zeros(n_bins, dtype=float)
    for r in bucketed.groupBy("bucket").count().collect():
        counts[int(r["bucket"])]
        
        
        
import importlib, comparador  # se você salvar como comparador.py
importlib.reload(comparador)

fig = comparador.compare_tables(
    sdf_orig=df,                 # tabela original
    sdf_synt=df_output,          # saída do synthesize_table_v1
    max_cols_to_plot=6,          # cap pro grid 2×3
    title="EngordAI · CDB · Original vs Sintética v1",
)

# pra salvar pro deck/print:
fig.savefig("comparacao_v1.png", dpi=180, bbox_inches="tight", facecolor="white")