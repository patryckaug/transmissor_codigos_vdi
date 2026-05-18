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
def kind(dt):
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
    """Frequência relativa das categorias values em sdf (mesmo col)."""
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
        counts[int(r["bucket"])] = r["count"]
    edges = np.linspace(mn, mx, n_bins + 1)
    return edges, counts


def _null_pct(sdf: DataFrame, col: str, total: int) -> float:
    if total == 0:
        return 0.0
    return sdf.where(F.col(col).isNull()).count() / total


# ===========================================================
# Plots por tipo
# ===========================================================
def _plot_categorical(ax, col, sdf_o, sdf_s, k=8):
    top_o, _ = _top_k(sdf_o, col, k)
    if not top_o:
        ax.text(0.5, 0.5, "(sem dados)", ha="center", va="center", color=C_MUTED)
        return
    values  = [v for v, _ in top_o]
    freqs_o = [f for _, f in top_o]
    freq_s_map = _freq_for_values(sdf_s, col, values)
    freqs_s = [freq_s_map.get(v, 0.0) for v in values]

    x, w = np.arange(len(values)), 0.4
    ax.bar(x - w/2, freqs_o, w, color=C_ORIG, label="Original", edgecolor="none")
    ax.bar(x + w/2, freqs_s, w, color=C_SYNT, label="Sintética", edgecolor="none")

    labels = [v if len(v) <= 12 else v[:10] + "…" for v in values]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8, color=C_INK)
    ax.set_ylabel("Frequência", fontsize=8, color=C_MUTED)


def _plot_distribution(ax, col, sdf_o, sdf_s, n_bins=25, is_dt=False):
    fn = _hist_datetime if is_dt else _hist_numeric
    e_o, _ = fn(sdf_o, col, n_bins)
    e_s, _ = fn(sdf_s, col, n_bins)
    if e_o[-1] == e_o[0] and e_s[-1] == e_s[0]:
        ax.text(0.5, 0.5, "(constante)", ha="center", va="center", color=C_MUTED)
        return
    mn = min(e_o[0],  e_s[0])
    mx = max(e_o[-1], e_s[-1])
    edges, counts_o = fn(sdf_o, col, n_bins, (mn, mx))
    _,     counts_s = fn(sdf_s, col, n_bins, (mn, mx))

    counts_o = counts_o / max(counts_o.sum(), 1.0)
    counts_s = counts_s / max(counts_s.sum(), 1.0)

    centers = (edges[:-1] + edges[1:]) / 2
    width   = (edges[1] - edges[0]) * 0.95

    ax.bar(centers, counts_o, width, color=C_ORIG, alpha=0.70, edgecolor="none", label="Original")
    ax.bar(centers, counts_s, width, color=C_SYNT, alpha=0.60, edgecolor="none", label="Sintética")

    if is_dt:
        def fmt(x, _):
            try:    return dt.datetime.fromtimestamp(x).strftime("%Y-%m")
            except: return ""
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax.set_ylabel("Frequência", fontsize=8, color=C_MUTED)


# ===========================================================
# Função principal
# ===========================================================
def compare_tables(
    sdf_orig: DataFrame,
    sdf_synt: DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    max_cols_to_plot: int = 6,
    figsize: Tuple[float, float] = (14, 9),
    title: str = "Comparação · Original vs Sintética",
):
    """
    Compara visualmente tabela original e sintética.
    - Header com volume e variação
    - Grid de até 6 colunas: categóricas (bars lado a lado), numéricas/datas (histogramas sobrepostos)
    - Cada subplot mostra null% original / sintética
    Tudo em Spark; só agregados pequenos vão pro driver.
    """
    schema_o = {f.name: f.dataType for f in sdf_orig.schema.fields}

    # ---- contagens ----
    n_o = sdf_orig.count()
    n_s = sdf_synt.count()

    # ---- seleção de colunas (round-robin entre tipos) ----
    if columns is None:
        common = [c for c in sdf_orig.columns if c in sdf_synt.columns]
        buckets = {"categorical": [], "numeric": [], "datetime": []}
        for c in common:
            k = _kind(schema_o[c])
            if k in buckets:
                buckets[k].append(c)
        columns, i = [], 0
        while len(columns) < max_cols_to_plot:
            added = False
            for k in ("categorical", "numeric", "datetime"):
                if i < len(buckets[k]) and len(columns) < max_cols_to_plot:
                    columns.append(buckets[k][i]); added = True
            i += 1
            if not added: break
    columns = list(columns)[:max_cols_to_plot]
    n = len(columns)
    if n == 0:
        raise ValueError("Nenhuma coluna em comum para comparar.")

    # ---- layout ----
    ncols = 2 if n > 2 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=C_BG)
    axes = np.array(axes).flatten() if n > 1 else [axes]

    # ---- header ----
    delta = (n_s - n_o) / n_o * 100 if n_o else 0
    sign  = "+" if delta >= 0 else ""
    fig.suptitle(title, fontsize=17, fontweight="bold", color=C_INK,
                 x=0.05, y=0.975, ha="left")
    fig.text(0.05, 0.935,
             f"Original: {n_o:,} linhas   ·   Sintética: {n_s:,} linhas "
             f"({sign}{delta:.1f}%)   ·   {n} colunas em comparação",
             fontsize=10.5, color=C_MUTED, ha="left")

    # legenda única
    fig.legend(handles=[Patch(facecolor=C_ORIG, label="Original"),
                        Patch(facecolor=C_SYNT, label="Sintética")],
               loc="upper right", bbox_to_anchor=(0.97, 0.965),
               frameon=False, fontsize=10, ncol=2)

    # ---- plots ----
    for i, col in enumerate(columns):
        ax = axes[i]
        kind = _kind(schema_o[col])
        if   kind == "categorical": _plot_categorical(ax, col, sdf_orig, sdf_synt)
        elif kind == "numeric":     _plot_distribution(ax, col, sdf_orig, sdf_synt, is_dt=False)
        elif kind == "datetime":    _plot_distribution(ax, col, sdf_orig, sdf_synt, is_dt=True)
        else:
            ax.text(0.5, 0.5, f"tipo: {kind}", ha="center", va="center", color=C_MUTED)

        null_o = _null_pct(sdf_orig, col, n_o)
        null_s = _null_pct(sdf_synt, col, n_s)

        ax.set_title(col, fontsize=11, fontweight="bold", color=C_INK, loc="left", pad=8)
        ax.text(1.0, 1.02, f"null: {null_o:.0%} → {null_s:.0%}",
                transform=ax.transAxes, fontsize=8, color=C_MUTED, ha="right")

        # estilo enxuto
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(C_LINE)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        ax.grid(axis="y", color=C_LINE, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    # esconde axes vazios
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    return fig