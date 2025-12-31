# dashboard.py
import base64
import io
import re
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

from config.settings import DB_PATH, PROVIDER_NAME
from services.proxmox import list_vms, list_nodes, get_node_uptime, list_tasks


# ============================================================
# Utilidades / Segurança de cálculo
# ============================================================

def _read_sql(query: str, params=None) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=params)


def pct(used: pd.Series, total: pd.Series) -> pd.Series:
    """Percentual seguro (trata total=0, NaN, inf)."""
    used = pd.to_numeric(used, errors="coerce").astype(float)
    total = pd.to_numeric(total, errors="coerce").astype(float)
    out = np.where(total > 0, (used / total) * 100.0, 0.0)
    return pd.Series(out).replace([np.inf, -np.inf], 0).fillna(0).round(2)


def empty_figure(title: str, subtitle: str = "") -> go.Figure:
    """Figura vazia (sem Plotly Express) para evitar bug de template do PX."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=30, r=30, t=70, b=40),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        font=dict(family="Segoe UI, Roboto, sans-serif", size=14),
    )
    if subtitle:
        fig.add_annotation(
            text=subtitle,
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14),
        )
    return fig


def _parse_num_with_unit(x):
    """
    Converte números vindos do Excel/CSV sem quebrar:
    - Se já for número (int/float), retorna float.
    - Se for string com ',' decimal ou '.' decimal, detecta corretamente.
    - Converte 'PiB'/'TiB'/'GiB' para TB (decimal).
    """
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).replace("\xa0", " ").strip()
    if s in ("*", ""):
        return np.nan

    m = re.search(r"([0-9\.,]+)\s*([A-Za-z%]+)?", s)
    if not m:
        return np.nan

    num = m.group(1)
    unit = (m.group(2) or "").lower().strip()

    if "," in num and "." in num:
        num = num.replace(".", "").replace(",", ".")
    elif "," in num and "." not in num:
        num = num.replace(",", ".")
    else:
        parts = num.split(".")
        if len(parts) > 1 and len(parts[-1]) == 3 and all(p.isdigit() for p in parts):
            num = "".join(parts)
        else:
            num = num

    try:
        val = float(num)
    except Exception:
        return np.nan

    if unit in ("%", "pct", "percent"):
        return val

    # unidade -> TB (decimal)
    if unit == "pib":
        val *= 1125.899906842624
    elif unit == "tib":
        val *= 1.099511627776
    elif unit == "gib":
        val *= 0.001073741824

    return val


# ============================================================
# Dados (SQLite) — Proxmox cluster
# ============================================================

def load_month_data(year: int, month: int) -> pd.DataFrame:
    """Carrega dados do mês/ano e calcula percentuais por cluster (snapshot)."""
    prefix = f"{year:04d}-{month:02d}"
    query = """
        SELECT
            ts, cluster_name,
            cpu_used_cores, cpu_total_cores,
            mem_used_gib, mem_total_gib,
            storage_used_tib, storage_total_tib
        FROM cluster_capacity_units
        WHERE substr(ts, 1, 7) = ?
    """
    df = _read_sql(query, params=(prefix,))
    if df.empty:
        return df

    df["cpu_pct"] = pct(df["cpu_used_cores"], df["cpu_total_cores"])
    df["mem_pct"] = pct(df["mem_used_gib"], df["mem_total_gib"])
    df["storage_pct"] = pct(df["storage_used_tib"], df["storage_total_tib"])
    return df


def aggregate_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega média e pico por cluster (com base nos snapshots do mês)."""
    if df.empty:
        return df
    agg = df.groupby("cluster_name").agg(
        cpu_avg=("cpu_pct", "mean"),
        cpu_max=("cpu_pct", "max"),
        mem_avg=("mem_pct", "mean"),
        mem_max=("mem_pct", "max"),
        storage_avg=("storage_pct", "mean"),
        storage_max=("storage_pct", "max"),
    ).reset_index()
    return agg.round(1)


def get_available_years() -> list[int]:
    """Lista anos disponíveis no banco (mais leve)."""
    df = _read_sql("SELECT DISTINCT substr(ts,1,4) AS y FROM cluster_capacity ORDER BY y")
    if df.empty:
        return [datetime.today().year]
    years = [int(x) for x in df["y"].dropna().tolist()]
    return years or [datetime.today().year]


def get_available_months_for_year(year: int) -> list[int]:
    """Lista meses disponíveis para um ano específico (mais leve)."""
    df = _read_sql(
        """
        SELECT DISTINCT CAST(substr(ts,6,2) AS INTEGER) AS m
        FROM cluster_capacity
        WHERE substr(ts,1,4) = ?
        ORDER BY m
        """,
        params=(f"{int(year):04d}",),
    )
    if df.empty:
        return [datetime.today().month]
    months = df["m"].dropna().astype(int).tolist()
    return months or [datetime.today().month]


def get_available_month_keys() -> list[str]:
    """Lista meses disponíveis no formato YYYY-MM (ordenado, mais leve)."""
    df = _read_sql("SELECT DISTINCT substr(ts,1,7) AS ym FROM cluster_capacity ORDER BY ym")
    if df.empty:
        return [datetime.today().strftime("%Y-%m")]
    keys = df["ym"].dropna().tolist()
    return keys or [datetime.today().strftime("%Y-%m")]


def load_months_data(month_keys: list[str]) -> pd.DataFrame:
    """Carrega dados para uma lista de meses (YYYY-MM) e calcula percentuais por cluster."""
    if not month_keys:
        return pd.DataFrame()

    placeholders = ",".join(["?"] * len(month_keys))
    query = f"""
        SELECT
            ts, cluster_name,
            cpu_used_cores, cpu_total_cores,
            mem_used_gib, mem_total_gib,
            storage_used_tib, storage_total_tib
        FROM cluster_capacity_units
        WHERE substr(ts, 1, 7) IN ({placeholders})
    """
    df = _read_sql(query, params=month_keys)

    if df.empty:
        return df

    df["month"] = pd.to_datetime(df["ts"], errors="coerce").dt.strftime("%Y-%m")
    df["cpu_pct"] = pct(df["cpu_used_cores"], df["cpu_total_cores"])
    df["mem_pct"] = pct(df["mem_used_gib"], df["mem_total_gib"])
    df["storage_pct"] = pct(df["storage_used_tib"], df["storage_total_tib"])
    return df


def aggregate_by_cluster_and_month(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega por (mês, cluster): médias e picos do mês (percentual por cluster)."""
    if df.empty:
        return df
    agg = df.groupby(["month", "cluster_name"]).agg(
        cpu_avg=("cpu_pct", "mean"),
        cpu_max=("cpu_pct", "max"),
        mem_avg=("mem_pct", "mean"),
        mem_max=("mem_pct", "max"),
        storage_avg=("storage_pct", "mean"),
        storage_max=("storage_pct", "max"),
    ).reset_index()
    return agg.round(1)


# ============================================================
# Ambiente (ponderado) — soma used/total por snapshot
# ============================================================

def compute_env_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula % do AMBIENTE por snapshot (ponderado):
    soma used/total de todos os clusters para cada ts.
    Retorna colunas: month, ts, cpu_pct, mem_pct, storage_pct
    """
    if df.empty:
        return df

    df = df.copy()
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["ts"], errors="coerce").dt.strftime("%Y-%m")

    cpu_used_col = "cpu_used_cores" if "cpu_used_cores" in df.columns else "cpu_used_ghz"
    cpu_total_col = "cpu_total_cores" if "cpu_total_cores" in df.columns else "cpu_total_ghz"
    mem_used_col = "mem_used_gib" if "mem_used_gib" in df.columns else "mem_used_gb"
    mem_total_col = "mem_total_gib" if "mem_total_gib" in df.columns else "mem_total_gb"
    sto_used_col = "storage_used_tib" if "storage_used_tib" in df.columns else "storage_used_tb"
    sto_total_col = "storage_total_tib" if "storage_total_tib" in df.columns else "storage_total_tb"

    snap = df.groupby(["month", "ts"], as_index=False).agg(
        cpu_used=(cpu_used_col, "sum"),
        cpu_total=(cpu_total_col, "sum"),
        mem_used=(mem_used_col, "sum"),
        mem_total=(mem_total_col, "sum"),
        storage_used=(sto_used_col, "sum"),
        storage_total=(sto_total_col, "sum"),
    )

    snap["cpu_pct"] = pct(snap["cpu_used"], snap["cpu_total"])
    snap["mem_pct"] = pct(snap["mem_used"], snap["mem_total"])
    snap["storage_pct"] = pct(snap["storage_used"], snap["storage_total"])
    return snap


def aggregate_env_weighted_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega o AMBIENTE (ponderado) por mês:
    - avg = média dos snapshots do mês
    - max = pico (máximo) entre os snapshots do mês
    """
    if df.empty:
        return df

    snap = compute_env_snapshots(df)
    if snap.empty:
        return pd.DataFrame()

    overall = snap.groupby("month", as_index=False).agg(
        cpu_avg=("cpu_pct", "mean"),
        cpu_max=("cpu_pct", "max"),
        mem_avg=("mem_pct", "mean"),
        mem_max=("mem_pct", "max"),
        storage_avg=("storage_pct", "mean"),
        storage_max=("storage_pct", "max"),
    )
    return overall.round(1)


def get_env_weighted_for_month(df: pd.DataFrame) -> dict:
    """Retorna métricas do ambiente ponderado para 1 mês (média e pico dos snapshots)."""
    zeros = {"cpu_avg": 0.0, "cpu_max": 0.0, "mem_avg": 0.0, "mem_max": 0.0, "storage_avg": 0.0, "storage_max": 0.0}
    if df.empty:
        return zeros

    overall = aggregate_env_weighted_by_month(df)
    if overall.empty:
        return zeros

    row = overall.iloc[0].to_dict()
    return {k: float(row.get(k, 0.0)) for k in zeros.keys()}


# ============================================================
# Gráficos (Plotly)
# ============================================================

def build_powerbi_style_storage_chart(agg: pd.DataFrame):
    """Gráfico estilo PowerBI para Storage médio por cluster (sem cortar labels)."""
    fig = go.Figure()

    y_max = float(agg["storage_avg"].max()) if not agg.empty else 0.0
    y_top = max(100.0, y_max * 1.25)

    fig.add_trace(
        go.Bar(
            x=agg["cluster_name"],
            y=agg["storage_avg"],
            marker=dict(color="#40E0D0"),
            text=[f"{v:.1f}%" for v in agg["storage_avg"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Storage médio: %{y:.1f}%<extra></extra>",
            opacity=0.85,
        )
    )

    fig.update_layout(
        title="Uso médio de Storage por Cluster (%)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=30, r=30, t=70, b=80),
        xaxis=dict(title="Cluster", tickangle=-35, showgrid=False, zeroline=False),
        yaxis=dict(
            title="Storage (%)",
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            range=[0, y_top],
        ),
        font=dict(family="Segoe UI, Roboto, sans-serif", size=14),
        bargap=0.28,
    )
    return fig


def build_gauge(
    value: float,
    title: str = "",
    yellow_from: float = 50.0,
    red_from: float = 80.0,
    show_number: bool = False,
    bar_color: str = "#6C5CE7",
):
    """
    Gauge (verde/amarelo/vermelho) para %.
    - Escala fixa 0/20/40/60/80/100 (consistente em todos)
    - yellow_from/red_from configuráveis (ex: Storage 80/95)
    - show_number=True mostra o valor (%) dentro do gauge
    """
    value = float(value)
    yellow_from = float(yellow_from)
    red_from = float(red_from)

    mode = "gauge+number" if show_number else "gauge"

    fig = go.Figure(
        go.Indicator(
            mode=mode,
            value=value,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": title},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 20, 40, 60, 80, 100],
                    "ticktext": ["0", "20", "40", "60", "80", "100"],
                },
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, yellow_from], "color": "#d4edda"},
                    {"range": [yellow_from, red_from], "color": "#fff3cd"},
                    {"range": [red_from, 100], "color": "#f8d7da"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.75, "value": red_from},
            },
        )
    )
    fig.update_layout(
        height=170,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Segoe UI, Roboto, sans-serif", size=12),
    )
    return fig


def build_compare_bar_by_cluster(agg_x: pd.DataFrame, agg_y: pd.DataFrame, month_x: str, month_y: str):
    """Barra agrupada: Storage médio por cluster (Mês X vs Mês Y)."""
    if agg_x.empty or agg_y.empty:
        return empty_figure("Sem dados para comparação")

    x = agg_x[["cluster_name", "storage_avg"]].rename(columns={"storage_avg": "storage_x"})
    y = agg_y[["cluster_name", "storage_avg"]].rename(columns={"storage_avg": "storage_y"})
    m = pd.merge(x, y, on="cluster_name", how="outer").fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=m["cluster_name"],
        y=m["storage_x"],
        name=month_x,
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Storage médio: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=m["cluster_name"],
        y=m["storage_y"],
        name=month_y,
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Storage médio: %{y:.1f}%<extra></extra>",
    ))

    y_max = float(max(m["storage_x"].max(), m["storage_y"].max()))
    fig.update_layout(
        barmode="group",
        title=f"Comparação Storage médio por Cluster (%) — {month_x} vs {month_y}",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=30, r=30, t=70, b=90),
        xaxis=dict(title="Cluster", tickangle=-30, showgrid=False, zeroline=False),
        yaxis=dict(
            title="Storage (%)",
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            range=[0, max(100.0, y_max * 1.25)],
        ),
        font=dict(family="Segoe UI, Roboto, sans-serif", size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================
# Gauges por cluster
# ============================================================

# Se quiser limitar a quantidade exibida (por performance), ajuste aqui:
CLUSTER_GAUGES_DEFAULT_TOPN = None  # None = mostra todos; ou ex: 12

# Padroniza barra roxa (igual o gauge do topo)
CLUSTER_BAR_COLOR = "#40E0D0"


def build_cluster_gauge_cards(agg: pd.DataFrame, metric: str = "avg", topn: int | None = CLUSTER_GAUGES_DEFAULT_TOPN) -> list:
    """
    Cards por cluster com 3 gauges (CPU/MEM/Storage) + rótulo + porcentagem.
    - CPU/MEM: amarelo>=50, vermelho>=80
    - Storage: amarelo>=80, vermelho>=95
    - Barra do gauge: roxa
    """
    if agg.empty:
        return []

    metric = (metric or "avg").lower().strip()
    suffix = "_avg" if metric == "avg" else "_max"

    d = agg.copy()
    order_col = f"storage{suffix}"
    if order_col in d.columns:
        d = d.sort_values(order_col, ascending=False)

    if isinstance(topn, int) and topn > 0:
        d = d.head(topn)

    def _metric_block(label: str, value: float, yellow_from: float, red_from: float):
        value = float(value)
        return html.Div(
            [
                html.Div(
                    [
                        html.Span(label, className="fw-semibold"),
                        html.Span(f" — {value:.1f}%", className="text-muted ms-1"),
                    ],
                    style={"fontSize": "13px", "marginBottom": "4px"},
                ),
                dcc.Graph(
                    figure=build_gauge(
                        value,
                        title="",
                        show_number=True,
                        bar_color=CLUSTER_BAR_COLOR,
                        yellow_from=yellow_from,
                        red_from=red_from,
                    ),
                    config={"displayModeBar": False},
                ),
            ]
        )

    cards = []
    for _, r in d.iterrows():
        name = str(r.get("cluster_name", "")).strip()

        cpu_v = float(r.get(f"cpu{suffix}", 0.0))
        mem_v = float(r.get(f"mem{suffix}", 0.0))
        sto_v = float(r.get(f"storage{suffix}", 0.0))

        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Div(name, className="fw-semibold"),
                        html.Div(f"{PROVIDER_NAME} — {('Média' if metric == 'avg' else 'Pico')}", className="text-muted", style={"fontSize": "12px"}),
                        dbc.Row([
                            dbc.Col(_metric_block("CPU", cpu_v, yellow_from=50.0, red_from=80.0), md=4),
                            dbc.Col(_metric_block("MEM", mem_v, yellow_from=50.0, red_from=80.0), md=4),
                            dbc.Col(_metric_block("Storage", sto_v, yellow_from=80.0, red_from=90.0), md=4),
                        ], className="g-2"),
                    ]),
                    className="shadow-sm",
                ),
                md=6,
                className="mb-3",
            )
        )
    return cards


# ============================================================
# Upload manual — Backup (somente)
# ============================================================

def parse_upload(contents: str, filename: str) -> pd.DataFrame:
    """Suporta CSV e XLSX/XLS. Excel: primeira aba. CSV: tenta ';' e depois ','."""
    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    name = (filename or "").lower()

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(decoded))

    if name.endswith(".csv"):
        try:
            return pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=";")
        except Exception:
            return pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=",")

    raise ValueError("Formato não suportado. Envie CSV ou XLSX.")


def normalize_backup_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza o arquivo de backup sem quebrar floats (Excel) e calcula Used% se faltar."""
    if df.empty:
        return df

    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    rename = {}
    for c in d.columns:
        cl = c.lower().strip()
        if cl == "backup":
            rename[c] = "Backup"
        elif cl == "total":
            rename[c] = "Total"
        elif cl == "used":
            rename[c] = "Used"
        elif cl in ("used%", "used_pct"):
            rename[c] = "Used%"
    d = d.rename(columns=rename)

    if "Backup" not in d.columns:
        d.insert(0, "Backup", "")

    if "Total" not in d.columns:
        d["Total"] = np.nan
    if "Used" not in d.columns:
        d["Used"] = np.nan

    d["Total"] = d["Total"].apply(_parse_num_with_unit)
    d["Used"] = d["Used"].apply(_parse_num_with_unit)

    if "Used%" in d.columns and d["Used%"].notna().any():
        d["Used%"] = d["Used%"].apply(_parse_num_with_unit).fillna(0.0)
    else:
        d["Used%"] = pct(d["Used"].fillna(0), d["Total"].fillna(0))

    d["Total"] = d["Total"].round(2)
    d["Used"] = d["Used"].round(2)
    d["Used%"] = d["Used%"].round(2)

    return d


def split_backup_sections(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Separa detalhe vs linhas-resumo (Total/Nodes)."""
    if df.empty:
        return df, {}

    d = normalize_backup_df(df).copy()

    summary = {}
    special_keys = [
        "Total Backup (JDF)",
        "Total Backup (SBC)",
        "Nodes (SBC)",
        "Nodes (JDF)",
    ]

    for k in special_keys:
        row = d[d["Backup"].astype(str).str.strip().str.lower() == k.lower()]
        if not row.empty:
            r = row.iloc[0]
            val = r.get("Total", np.nan)
            if pd.isna(val):
                val = r.get("Used", np.nan)
            summary[k] = val

    mask_special = d["Backup"].astype(str).str.lower().isin([s.lower() for s in special_keys])
    detail = d[~mask_special].copy()

    if not detail.empty and "Used%" in detail.columns:
        detail = detail.sort_values("Used%", ascending=False)

    return detail, summary


def summarize_backup_cards(detail: pd.DataFrame, summary: dict) -> list:
    """Cards: top crítico, média, e (se existir) totais/nodes."""
    cards = []

    if not detail.empty:
        top = detail.sort_values("Used%", ascending=False).iloc[0]
        top_name = str(top.get("Backup", ""))
        top_used = float(top.get("Used%", 0.0))
        avg_used = float(detail["Used%"].mean()) if "Used%" in detail.columns else 0.0

        cards.extend([
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    html.Div("Mais Utilizado (Used%)", className="text-muted"),
                    html.H3(f"{top_used:.1f}%", className="fw-bold"),
                    html.Div(top_name, className="text-muted"),
                ]), className="shadow-sm"),
                md=3, className="mb-3",
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    html.Div("Média de utilização (Used%)", className="text-muted"),
                    html.H3(f"{avg_used:.1f}%", className="fw-bold"),
                    html.Div("Somente itens de detalhe", className="text-muted"),
                ]), className="shadow-sm"),
                md=3, className="mb-3",
            ),
        ])

    def _fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        try:
            fv = float(v)
            if fv.is_integer():
                return f"{int(fv)}"
            return f"{fv:.2f}"
        except Exception:
            return str(v)

    for label in ["Total Backup (SBC)", "Total Backup (JDF)", "Nodes (SBC)", "Nodes (JDF)"]:
        if label in summary:
            cards.append(
                dbc.Col(
                    dbc.Card(dbc.CardBody([
                        html.Div(label, className="text-muted"),
                        html.H3(_fmt(summary[label]), className="fw-bold"),
                    ]), className="shadow-sm"),
                    md=3, className="mb-3",
                )
            )

    return cards


def backup_simple_fig(detail: pd.DataFrame) -> go.Figure:
    """Gráfico simples e legível: Used% por backup (somente detalhe)."""
    if detail.empty:
        return empty_figure("Backup dados", "Envie um arquivo para visualizar")

    d = detail.copy()
    fig = go.Figure(go.Bar(
        x=d["Backup"],
        y=d["Used%"],
        text=[f"{v:.1f}%" for v in d["Used%"]],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Used%: %{y:.1f}%<extra></extra>",
        opacity=0.85,
    ))
    fig.update_layout(
        title="Backup — Utilização (Used%)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=30, r=30, t=70, b=110),
        xaxis=dict(title="", tickangle=-30, showgrid=False, zeroline=False),
        yaxis=dict(title="Used (%)", gridcolor="rgba(0,0,0,0.08)", zeroline=False, range=[0, 100]),
        font=dict(family="Segoe UI, Roboto, sans-serif", size=14),
    )
    return fig


def datatable_component(df: pd.DataFrame) -> html.Div:
    if df.empty:
        return html.Div(dbc.Alert("Sem dados.", color="warning"))

    d = df.copy()
    columns = [{"name": c, "id": c} for c in d.columns]

    return html.Div(
        dash_table.DataTable(
            columns=columns,
            data=d.to_dict("records"),
            page_action="none",
            fixed_rows={"headers": True},
            style_table={
                "height": "420px",
                "overflowY": "auto",
                "overflowX": "auto",
                "borderRadius": "10px",
                "border": "1px solid rgba(0,0,0,0.08)",
            },
            style_header={
                "backgroundColor": "#f8f9fa",
                "fontWeight": "700",
                "padding": "12px",
                "textAlign": "center",
                "fontFamily": "Segoe UI, Roboto, sans-serif",
                "fontSize": "13px",
                "whiteSpace": "normal",
                "height": "auto",
            },
            style_cell={
                "padding": "10px",
                "fontFamily": "Segoe UI, Roboto, sans-serif",
                "fontSize": "13px",
                "textAlign": "center",
                "whiteSpace": "nowrap",
                "minWidth": "120px",
                "width": "120px",
                "maxWidth": "320px",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "rgba(0,0,0,0.02)"},
                {"if": {"state": "active"}, "backgroundColor": "rgba(108,92,231,0.10)"},
            ],
        )
    )


# ============================================================
# Upload manual — Storage (somente)
# ============================================================

def _parse_pib(x):
    """Converte números/textos da planilha de Storage em PiB (float)."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).replace("\xa0", " ").strip()
    if s in ("*", ""):
        return np.nan

    m = re.search(r"([0-9\.,]+)", s)
    if not m:
        return np.nan

    num = m.group(1)
    if "," in num and "." in num:
        num = num.replace(".", "").replace(",", ".")
    elif "," in num and "." not in num:
        num = num.replace(",", ".")

    try:
        return float(num)
    except Exception:
        return np.nan


def normalize_storage_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o arquivo de Storage:
    - Padroniza nomes de colunas
    - Converte Total/Allocated em PiB (float)
    - Calcula Used% = Allocated/Total * 100 (para os 2 blocos)
    """
    if df.empty:
        return df

    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    colmap = {}
    for c in d.columns:
        cl = str(c).strip().lower()
        if cl == "storage system":
            colmap[c] = "Storage System"
        elif cl == "location":
            colmap[c] = "Location"
        elif cl == "total (pib)":
            colmap[c] = "Total_1_PiB"
        elif cl == "allocated (pib)":
            colmap[c] = "Allocated_1_PiB"
        elif cl == "total (pib).1":
            colmap[c] = "Total_2_PiB"
        elif cl == "allocated (pib).1":
            colmap[c] = "Allocated_2_PiB"

    d = d.rename(columns=colmap)

    if "Storage System" not in d.columns:
        d["Storage System"] = ""
    if "Location" not in d.columns:
        d["Location"] = ""

    for c in ["Total_1_PiB", "Allocated_1_PiB", "Total_2_PiB", "Allocated_2_PiB"]:
        if c not in d.columns:
            d[c] = np.nan

    d["Total_1_PiB"] = d["Total_1_PiB"].apply(_parse_pib)
    d["Allocated_1_PiB"] = d["Allocated_1_PiB"].apply(_parse_pib)
    d["Total_2_PiB"] = d["Total_2_PiB"].apply(_parse_pib)
    d["Allocated_2_PiB"] = d["Allocated_2_PiB"].apply(_parse_pib)

    d["Used_1_%"] = pct(d["Allocated_1_PiB"].fillna(0), d["Total_1_PiB"].fillna(0))
    d["Used_2_%"] = pct(d["Allocated_2_PiB"].fillna(0), d["Total_2_PiB"].fillna(0))

    for c in ["Total_1_PiB", "Allocated_1_PiB", "Total_2_PiB", "Allocated_2_PiB"]:
        d[c] = d[c].round(2)
    d["Used_1_%"] = d["Used_1_%"].round(2)
    d["Used_2_%"] = d["Used_2_%"].round(2)

    d = d.sort_values("Used_1_%", ascending=False)
    return d


def summarize_storage_cards(df: pd.DataFrame) -> list:
    """Cards: top/média bloco 1 e (se existir) top/média bloco 2."""
    if df.empty:
        return []

    cards = []

    def _mk(title, value, subtitle=""):
        return dbc.Col(
            dbc.Card(dbc.CardBody([
                html.Div(title, className="text-muted"),
                html.H3(value, className="fw-bold"),
                html.Div(subtitle, className="text-muted") if subtitle else html.Div(),
            ]), className="shadow-sm"),
            md=3, className="mb-3"
        )

    top1 = df.sort_values("Used_1_%", ascending=False).iloc[0]
    cards.append(_mk(
        "Storage — Maior Used% (Bloco 1)",
        f"{float(top1['Used_1_%']):.1f}%",
        f"{top1.get('Storage System','')} / {top1.get('Location','')}"
    ))
    cards.append(_mk(
        "Storage — Média Used% (Bloco 1)",
        f"{float(df['Used_1_%'].mean()):.1f}%",
        "Allocated_1 / Total_1"
    ))

    has2 = df["Total_2_PiB"].notna().any() and (df["Total_2_PiB"].fillna(0).sum() > 0)
    if has2:
        top2 = df.sort_values("Used_2_%", ascending=False).iloc[0]
        cards.append(_mk(
            "Storage — Maior Used% (Bloco 2)",
            f"{float(top2['Used_2_%']):.1f}%",
            f"{top2.get('Storage System','')} / {top2.get('Location','')}"
        ))
        cards.append(_mk(
            "Storage — Média Used% (Bloco 2)",
            f"{float(df['Used_2_%'].mean()):.1f}%",
            "Allocated_2 / Total_2"
        ))

    return cards


def storage_simple_fig(df: pd.DataFrame) -> go.Figure:
    """Gráfico simples: Used_1_% por Location (com Storage System no hover)."""
    if df.empty:
        return empty_figure("Storage dados", "Envie um arquivo para visualizar")

    d = df.copy()
    fig = go.Figure(go.Bar(
        x=d["Location"],
        y=d["Used_1_%"],
        text=[f"{v:.1f}%" for v in d["Used_1_%"]],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Used_1_%: %{y:.1f}%<br>Storage System: %{customdata}<extra></extra>",
        customdata=[f"{ss}" for ss in d["Storage System"]],
        opacity=0.85,
    ))
    fig.update_layout(
        title="Storage — Utilização (Used% Bloco 1) por Location",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=30, r=30, t=70, b=110),
        xaxis=dict(title="", tickangle=-30, showgrid=False, zeroline=False),
        yaxis=dict(title="Used (%)", gridcolor="rgba(0,0,0,0.08)", zeroline=False, range=[0, 100]),
        font=dict(family="Segoe UI, Roboto, sans-serif", size=14),
    )
    return fig


# ============================================================
# App Dash
# ============================================================

def create_app():
    years = get_available_years()
    current_year = years[-1]
    months = get_available_months_for_year(current_year)
    current_month = months[-1]

    month_keys = get_available_month_keys()
    month_key_options = [{"label": k, "value": k} for k in month_keys]
    default_y = month_keys[-1]
    default_x = month_keys[-2] if len(month_keys) >= 2 else month_keys[-1]

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title=f"Portal de Capacity {PROVIDER_NAME}",
        suppress_callback_exceptions=True,
    )

    # ==============================
    # TAB 1 (Visão do mês)
    # ==============================

    overview_layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Ano", className="fw-semibold"),
                            dcc.Dropdown(
                                id="year-dropdown",
                                options=[{"label": y, "value": y} for y in years],
                                value=current_year,
                                clearable=False,
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            html.Label("Mês", className="fw-semibold"),
                            dcc.Dropdown(
                                id="month-dropdown",
                                options=[{"label": f"{m:02d}", "value": m} for m in months],
                                value=current_month,
                                clearable=False,
                            ),
                        ],
                        md=2,
                    ),
                ],
                className="mb-3",
            ),

            dbc.Row(id="summary-cards", className="mb-4"),

            # Gauges por cluster (novo)
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.Div("Gauges por Cluster (CPU/MEM/Storage)", className="fw-semibold"),
                                                md=8
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="cluster-gauge-metric",
                                                    options=[
                                                        {"label": "Média do mês", "value": "avg"},
                                                        {"label": "Pico do mês", "value": "max"},
                                                    ],
                                                    value="avg",
                                                    clearable=False,
                                                ),
                                                md=4,
                                            ),
                                        ],
                                        className="g-2 align-items-center",
                                    )
                                ),
                                dbc.CardBody(dbc.Row(id="cluster-gauges-row", className="g-2")),
                            ],
                            className="shadow-sm",
                        ),
                        md=12,
                        className="mb-4",
                    )
                ]
            ),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Uso médio de Storage por Cluster (%)"),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="storage-graph",
                                        config={"displayModeBar": False},
                                        style={"height": "480px"},
                                    )
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        md=5,
                        className="mb-4",
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Resumo por Cluster (média e pico)"),
                                dbc.CardBody(
                                    dash_table.DataTable(
                                        id="cluster-table",
                                        columns=[
                                            {"name": "Cluster", "id": "cluster_name"},
                                            {"name": "CPU média (%)", "id": "cpu_avg"},
                                            {"name": "CPU pico (%)", "id": "cpu_max"},
                                            {"name": "Mem média (%)", "id": "mem_avg"},
                                            {"name": "Mem pico (%)", "id": "mem_max"},
                                            {"name": "Storage médio (%)", "id": "storage_avg"},
                                            {"name": "Storage pico (%)", "id": "storage_max"},
                                        ],
                                        data=[],
                                        page_action="none",
                                        fixed_rows={"headers": True},
                                        style_table={
                                            "height": "420px",
                                            "overflowY": "auto",
                                            "overflowX": "auto",
                                            "borderRadius": "10px",
                                            "border": "1px solid rgba(0,0,0,0.08)",
                                        },
                                        style_header={
                                            "backgroundColor": "#f8f9fa",
                                            "fontWeight": "700",
                                            "padding": "12px",
                                            "textAlign": "center",
                                            "fontFamily": "Segoe UI, Roboto, sans-serif",
                                            "fontSize": "13px",
                                            "whiteSpace": "normal",
                                            "height": "auto",
                                        },
                                        style_cell={
                                            "padding": "10px",
                                            "fontFamily": "Segoe UI, Roboto, sans-serif",
                                            "fontSize": "13px",
                                            "textAlign": "center",
                                            "whiteSpace": "nowrap",
                                            "minWidth": "120px",
                                            "width": "120px",
                                            "maxWidth": "160px",
                                        },
                                        style_cell_conditional=[
                                            {
                                                "if": {"column_id": "cluster_name"},
                                                "textAlign": "left",
                                                "minWidth": "260px",
                                                "width": "260px",
                                                "maxWidth": "360px",
                                            }
                                        ],
                                        style_data_conditional=[
                                            {"if": {"row_index": "odd"}, "backgroundColor": "rgba(0,0,0,0.02)"},
                                            {"if": {"state": "active"}, "backgroundColor": "rgba(108,92,231,0.10)"},
                                        ],
                                    ),
                                    style={"padding": "12px"},
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        md=7,
                        className="mb-4",
                    ),
                ]
            ),
        ],
    )

    # ==============================
    # TAB 2 (Comparar meses) — sem botões 3/6/12 etc.
    # ==============================

    compare_layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Mês X (YYYY-MM)", className="fw-semibold"),
                            dcc.Dropdown(
                                id="compare-month-x",
                                options=month_key_options,
                                value=default_x,
                                clearable=False,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Mês Y (YYYY-MM)", className="fw-semibold"),
                            dcc.Dropdown(
                                id="compare-month-y",
                                options=month_key_options,
                                value=default_y,
                                clearable=False,
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="mb-3",
            ),

            dbc.Row(id="compare-summary-cards", className="mb-4"),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(html.Div(id="compare-title", className="fw-semibold")),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="compare-graph",
                                        config={"displayModeBar": False},
                                        style={"height": "520px"},
                                    )
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        md=12,
                        className="mb-4",
                    ),
                ]
            ),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Tabela da comparação"),
                                dbc.CardBody(
                                    dash_table.DataTable(
                                        id="compare-table",
                                        columns=[],
                                        data=[],
                                        page_action="none",
                                        fixed_rows={"headers": True},
                                        style_table={
                                            "height": "420px",
                                            "overflowY": "auto",
                                            "overflowX": "auto",
                                            "borderRadius": "10px",
                                            "border": "1px solid rgba(0,0,0,0.08)",
                                        },
                                        style_header={
                                            "backgroundColor": "#f8f9fa",
                                            "fontWeight": "700",
                                            "padding": "12px",
                                            "textAlign": "center",
                                            "fontFamily": "Segoe UI, Roboto, sans-serif",
                                            "fontSize": "13px",
                                            "whiteSpace": "normal",
                                            "height": "auto",
                                        },
                                        style_cell={
                                            "padding": "10px",
                                            "fontFamily": "Segoe UI, Roboto, sans-serif",
                                            "fontSize": "13px",
                                            "textAlign": "center",
                                            "whiteSpace": "nowrap",
                                            "minWidth": "120px",
                                            "width": "120px",
                                            "maxWidth": "220px",
                                        },
                                        style_data_conditional=[
                                            {"if": {"row_index": "odd"}, "backgroundColor": "rgba(0,0,0,0.02)"},
                                            {"if": {"state": "active"}, "backgroundColor": "rgba(108,92,231,0.10)"},
                                        ],
                                    ),
                                    style={"padding": "12px"},
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        md=12,
                    )
                ]
            )
        ],
    )

    # ==============================
    # TAB 3 (Backup dados)
    # ==============================

    backup_layout = dbc.Container(
        fluid=True,
        children=[
            html.Div(
                [
                    html.H5("Backup dados", className="fw-bold"),
                    html.Div("Upload de planilha/CSV pelo time de Backup. ", className="text-muted"),
                ],
                className="mb-3",
            ),

            dcc.Store(id="manual-backup-store", data=None),
            dbc.Row(id="manual-backup-cards", className="mb-2"),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Upload — Backup"),
                                dbc.CardBody(
                                    [
                                        dcc.Upload(
                                            id="upload-backup",
                                            children=html.Div(["Arraste e solte ou ", html.A("selecione o arquivo (CSV/XLSX)")]),
                                            style={
                                                "width": "100%",
                                                "height": "90px",
                                                "lineHeight": "90px",
                                                "borderWidth": "1px",
                                                "borderStyle": "dashed",
                                                "borderRadius": "10px",
                                                "textAlign": "center",
                                                "cursor": "pointer",
                                            },
                                            multiple=False,
                                        ),
                                        html.Div(
                                            id="upload-backup-status",
                                            className="text-muted mt-2",
                                            style={"fontSize": "12px"},
                                        ),
                                    ]
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        md=12,
                        className="mb-3",
                    ),
                ]
            ),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Backup — Visão (simples)"),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="manual-backup-graph",
                                        config={"displayModeBar": False},
                                        style={"height": "420px"},
                                    )
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        md=12,
                        className="mb-4",
                    )
                ]
            ),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Backup — Tabela"),
                                dbc.CardBody(html.Div(id="manual-backup-table-wrap")),
                            ],
                            className="shadow-sm",
                        ),
                        md=12,
                        className="mb-4",
                    ),
                ]
            ),
        ],
    )

    # ==============================
    # TAB 4 (Storage dados)
    # ==============================

    storage_layout = dbc.Container(
        fluid=True,
        children=[
            html.Div(
                [
                    html.H5("Storage dados", className="fw-bold"),
                    html.Div("Upload de planilha/CSV pelo time de Storage.", className="text-muted"),
                ],
                className="mb-3",
            ),

            dcc.Store(id="manual-storage-store", data=None),
            dbc.Row(id="manual-storage-cards", className="mb-2"),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Upload — Storage"),
                                dbc.CardBody(
                                    [
                                        dcc.Upload(
                                            id="upload-storage",
                                            children=html.Div(["Arraste e solte ou ", html.A("selecione o arquivo (CSV/XLSX)")]),
                                            style={
                                                "width": "100%",
                                                "height": "90px",
                                                "lineHeight": "90px",
                                                "borderWidth": "1px",
                                                "borderStyle": "dashed",
                                                "borderRadius": "10px",
                                                "textAlign": "center",
                                                "cursor": "pointer",
                                            },
                                            multiple=False,
                                        ),
                                        html.Div(
                                            id="upload-storage-status",
                                            className="text-muted mt-2",
                                            style={"fontSize": "12px"},
                                        ),
                                    ]
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        md=12,
                        className="mb-3",
                    ),
                ]
            ),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Storage — Visão (simples)"),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="manual-storage-graph",
                                        config={"displayModeBar": False},
                                        style={"height": "420px"},
                                    )
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        md=12,
                        className="mb-4",
                    )
                ]
            ),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Storage — Tabela"),
                                dbc.CardBody(html.Div(id="manual-storage-table-wrap")),
                            ],
                            className="shadow-sm",
                        ),
                        md=12,
                        className="mb-4",
                    ),
                ]
            ),
        ],
    )

    inventory_layout = dbc.Container(
        fluid=True,
        children=[
            html.Div([html.H5("Inventário (VM/CT)", className="fw-bold"), html.Div("Lista de VMs e Containers com uso e status.", className="text-muted")], className="mb-3"),
            html.Div(id="inventory-status", className="text-muted mb-2", style={"fontSize": "12px"}),
            dbc.Row(
                [
                    dbc.Col(dcc.Dropdown(id="inv-type", options=[{"label": "Todos", "value": "all"}, {"label": "QEMU (VM)", "value": "qemu"}, {"label": "LXC (CT)", "value": "lxc"}], value="all", clearable=False), md=3),
                    dbc.Col(dcc.Dropdown(id="inv-status", options=[{"label": "Todos", "value": "all"}, {"label": "Rodando", "value": "running"}, {"label": "Parado", "value": "stopped"}], value="all", clearable=False), md=3),
                    dbc.Col(dcc.Input(id="inv-search", placeholder="Buscar por nome/VMID", type="text"), md=4),
                    dbc.Col(dbc.Button("Atualizar", id="inv-refresh", color="secondary"), md=2),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Inventário — Tabela"),
                                dbc.CardBody(
                                    dash_table.DataTable(
                                        id="inventory-table",
                                        columns=[
                                            {"name": "VMID", "id": "vmid"},
                                            {"name": "Nome", "id": "name"},
                                            {"name": "Tipo", "id": "type"},
                                            {"name": "Nó", "id": "node"},
                                            {"name": "Status", "id": "status"},
                                            {"name": "CPU (%)", "id": "cpu_pct"},
                                            {"name": "Mem (GiB)", "id": "mem_gib"},
                                            {"name": "Mem (%)", "id": "mem_pct"},
                                            {"name": "Disco (GiB)", "id": "disk_gib"},
                                            {"name": "Disco (%)", "id": "disk_pct"},
                                        ],
                                        data=[],
                                        page_action="none",
                                        fixed_rows={"headers": True},
                                        style_table={"height": "520px", "overflowY": "auto", "overflowX": "auto", "borderRadius": "10px", "border": "1px solid rgba(0,0,0,0.08)"},
                                        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "700", "padding": "12px", "textAlign": "center", "fontFamily": "Segoe UI, Roboto, sans-serif", "fontSize": "13px"},
                                        style_cell={"padding": "10px", "fontFamily": "Segoe UI, Roboto, sans-serif", "fontSize": "13px", "textAlign": "center", "whiteSpace": "nowrap", "minWidth": "120px", "width": "120px", "maxWidth": "220px"},
                                        style_cell_conditional=[{"if": {"column_id": "name"}, "textAlign": "left", "minWidth": "200px", "width": "260px", "maxWidth": "360px"}],
                                        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "rgba(0,0,0,0.02)"}],
                                    )
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        md=12,
                    )
                ]
            ),
        ],
    )

    nodes_layout = dbc.Container(
        fluid=True,
        children=[
            html.Div([html.H5("Nós do Cluster", className="fw-bold"), html.Div("Status, uptime e utilização por nó.", className="text-muted")], className="mb-3"),
            html.Div(id="nodes-status", className="text-muted mb-2", style={"fontSize": "12px"}),
            dbc.Row([dbc.Col(dbc.Button("Atualizar", id="nodes-refresh", color="secondary"), md=2)], className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Nós — Tabela"),
                                dbc.CardBody(
                                    dash_table.DataTable(
                                        id="nodes-table",
                                        columns=[
                                            {"name": "Nó", "id": "node"},
                                            {"name": "Status", "id": "status"},
                                            {"name": "Uptime (h)", "id": "uptime_h"},
                                            {"name": "CPU (%)", "id": "cpu_pct"},
                                            {"name": "Mem (GiB)", "id": "mem_gib"},
                                            {"name": "Mem (%)", "id": "mem_pct"},
                                        ],
                                        data=[],
                                        page_action="none",
                                        fixed_rows={"headers": True},
                                        style_table={"height": "520px", "overflowY": "auto", "overflowX": "auto", "borderRadius": "10px", "border": "1px solid rgba(0,0,0,0.08)"},
                                        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "700", "padding": "12px", "textAlign": "center", "fontFamily": "Segoe UI, Roboto, sans-serif", "fontSize": "13px"},
                                        style_cell={"padding": "10px", "fontFamily": "Segoe UI, Roboto, sans-serif", "fontSize": "13px", "textAlign": "center", "whiteSpace": "nowrap", "minWidth": "120px", "width": "120px", "maxWidth": "220px"},
                                        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "rgba(0,0,0,0.02)"}],
                                    )
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        md=12,
                    )
                ]
            ),
        ],
    )

    tasks_layout = dbc.Container(
        fluid=True,
        children=[
            html.Div([html.H5("Tarefas do Cluster", className="fw-bold"), html.Div("Operações recentes e status.", className="text-muted")], className="mb-3"),
            html.Div(id="tasks-status", className="text-muted mb-2", style={"fontSize": "12px"}),
            dbc.Row(
                [
                    dbc.Col(dcc.Dropdown(id="tasks-status", options=[{"label": "Todos", "value": "all"}, {"label": "OK", "value": "OK"}, {"label": "Falha", "value": "error"}], value="all", clearable=False), md=3),
                    dbc.Col(dcc.Input(id="tasks-type", placeholder="Filtrar por tipo (ex: backup, migrate)", type="text"), md=5),
                    dbc.Col(dbc.Button("Atualizar", id="tasks-refresh", color="secondary"), md=2),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Tarefas — Tabela"),
                                dbc.CardBody(
                                    dash_table.DataTable(
                                        id="tasks-table",
                                        columns=[
                                            {"name": "Hora", "id": "time"},
                                            {"name": "Nó", "id": "node"},
                                            {"name": "Tipo", "id": "type"},
                                            {"name": "Status", "id": "status"},
                                            {"name": "Usuário", "id": "user"},
                                        ],
                                        data=[],
                                        page_action="none",
                                        fixed_rows={"headers": True},
                                        style_table={"height": "520px", "overflowY": "auto", "overflowX": "auto", "borderRadius": "10px", "border": "1px solid rgba(0,0,0,0.08)"},
                                        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "700", "padding": "12px", "textAlign": "center", "fontFamily": "Segoe UI, Roboto, sans-serif", "fontSize": "13px"},
                                        style_cell={"padding": "10px", "fontFamily": "Segoe UI, Roboto, sans-serif", "fontSize": "13px", "textAlign": "center", "whiteSpace": "nowrap", "minWidth": "120px", "width": "120px", "maxWidth": "220px"},
                                        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "rgba(0,0,0,0.02)"}],
                                    )
                                ),
                            ],
                            className="shadow-sm",
                        ),
                        md=12,
                    )
                ]
            ),
        ],
    )
    # ==============================
    # Layout principal (Tabs)
    # ==============================

    app.layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Navbar(
                dbc.Container(
                    [
                        dbc.NavbarBrand(f"Portal de Capacity {PROVIDER_NAME}", className="fw-bold"),
                        html.Span("v2.0", className="text-muted ms-2"),
                    ]
                ),
                dark=True,
                className="mb-4",
                style={"backgroundColor": "#40E0D0"},
            ),

            dcc.Tabs(
                id="tabs",
                value="tab-overview",
                children=[
                    dcc.Tab(label="Visão do mês", value="tab-overview"),
                    dcc.Tab(label="Comparar meses", value="tab-compare"),
                    dcc.Tab(label="Inventário", value="tab-inventory"),
                    dcc.Tab(label="Nós", value="tab-nodes"),
                    dcc.Tab(label="Tarefas", value="tab-tarefas"),
                ],
            ),
            html.Div(id="tabs-content", className="mt-3"),
        ],
    )

    # ==============================
    # Render tabs
    # ==============================

    @app.callback(
        Output("tabs-content", "children"),
        Input("tabs", "value"),
    )
    def render_tab(tab):
        if tab == "tab-compare":
            return compare_layout
        if tab == "tab-inventory":
            return inventory_layout
        if tab == "tab-nodes":
            return nodes_layout
        if tab == "tab-tarefas":
            return tasks_layout
        return overview_layout

    # ============================================================
    # Callback: atualizar dropdown de meses quando muda o ano
    # ============================================================

    @app.callback(
        Output("month-dropdown", "options"),
        Output("month-dropdown", "value"),
        Input("year-dropdown", "value"),
    )
    def update_month_options(selected_year):
        months2 = get_available_months_for_year(int(selected_year))
        opts = [{"label": f"{m:02d}", "value": m} for m in months2]
        val = months2[-1]
        return opts, val

    # ==============================
    # Callback - Visão do mês
    # ==============================

    @app.callback(
        Output("storage-graph", "figure"),
        Output("cluster-table", "data"),
        Output("summary-cards", "children"),
        Output("cluster-gauges-row", "children"),
        Input("year-dropdown", "value"),
        Input("month-dropdown", "value"),
        Input("cluster-gauge-metric", "value"),
    )
    def update_overview(year, month, cluster_metric):
        df = load_month_data(int(year), int(month))
        agg = aggregate_by_cluster(df)

        if df.empty or agg.empty:
            cards = [dbc.Col(dbc.Alert("Nenhum dado encontrado para este mês/ano.", color="warning"), md=12)]
            return empty_figure("Nenhum dado para o período selecionado"), [], cards, []

        fig = build_powerbi_style_storage_chart(agg)
        table_data = agg.to_dict("records")

        # Gauges gerais do ambiente (ponderado)
        env = get_env_weighted_for_month(df)
        cpu_env = env["cpu_avg"]
        mem_env = env["mem_avg"]
        storage_env = env["storage_avg"]

        def _env_block(label: str, value: float, yellow_from: float, red_from: float):
            value = float(value)
            return dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Span(f"{label} — Ambiente (ponderado)", className="text-muted"),
                            html.Span(f" — {value:.1f}%", className="text-muted ms-1"),
                        ],
                        style={"fontSize": "13px", "marginBottom": "4px"},
                    ),
                    dcc.Graph(
                        figure=build_gauge(
                            value,
                            title="",
                            show_number=True,
                            bar_color=CLUSTER_BAR_COLOR,
                            yellow_from=yellow_from,
                            red_from=red_from,
                        ),
                        config={"displayModeBar": False},
                    ),
                ]
            )

        # 100% consistente: Storage geral também 80/95
        cards = [
            dbc.Col(dbc.Card(_env_block("CPU", cpu_env, yellow_from=50.0, red_from=80.0), className="shadow-sm"), md=4, className="mb-3"),
            dbc.Col(dbc.Card(_env_block("Memória", mem_env, yellow_from=50.0, red_from=80.0), className="shadow-sm"), md=4, className="mb-3"),
            dbc.Col(dbc.Card(_env_block("Storage", storage_env, yellow_from=80.0, red_from=90.0), className="shadow-sm"), md=4, className="mb-3"),
        ]

        cluster_cards = build_cluster_gauge_cards(agg, metric=cluster_metric or "avg", topn=CLUSTER_GAUGES_DEFAULT_TOPN)
        return fig, table_data, cards, cluster_cards

    @app.callback(
        Output("inventory-table", "data"),
        Output("inventory-status", "children"),
        Input("inv-refresh", "n_clicks"),
        State("inv-type", "value"),
        State("inv-status", "value"),
        State("inv-search", "value"),
        prevent_initial_call=True,
    )
    def update_inventory(n_clicks, inv_type, inv_status, inv_search):
        try:
            items = list_vms()
            rows = []
            q = (inv_search or "").strip().lower()
            for it in items:
                t = str(it.get("type") or "")
                st = str(it.get("status") or "")
                if inv_type and inv_type != "all" and t != inv_type:
                    continue
                if inv_status and inv_status != "all" and st != inv_status:
                    continue
                name = str(it.get("name") or "")
                vmid = str(it.get("vmid") or "")
                if q and not (q in name.lower() or q in vmid.lower()):
                    continue
                cpu_ratio = float(it.get("cpu") or 0.0)
                maxmem = float(it.get("maxmem") or 0.0)
                mem = float(it.get("mem") or 0.0)
                maxdisk = float(it.get("maxdisk") or 0.0)
                disk = float(it.get("disk") or 0.0)
                rows.append(
                    {
                        "vmid": vmid,
                        "name": name,
                        "type": t.upper(),
                        "node": str(it.get("node") or ""),
                        "status": st,
                        "cpu_pct": round(cpu_ratio * 100.0, 1),
                        "mem_gib": round(mem / (1024 ** 3), 2),
                        "mem_pct": round((mem / maxmem) * 100.0, 1) if maxmem > 0 else 0.0,
                        "disk_gib": round(disk / (1024 ** 3), 2),
                        "disk_pct": round((disk / maxdisk) * 100.0, 1) if maxdisk > 0 else 0.0,
                    }
                )
            rows = sorted(rows, key=lambda r: (r["status"] != "running", -r["cpu_pct"]))
            return rows, ""
        except Exception as e:
            return [], f"Erro ao consultar inventário: {e}"

    @app.callback(
        Output("nodes-table", "data"),
        Output("nodes-status", "children"),
        Input("nodes-refresh", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_nodes(n_clicks):
        try:
            nodes = list_nodes()
            rows = []
            for n in nodes:
                node_name = str(n.get("node") or "")
                cpu_ratio = float(n.get("cpu") or 0.0)
                maxmem = float(n.get("maxmem") or 0.0)
                mem = float(n.get("mem") or 0.0)
                uptime = get_node_uptime(node_name) if node_name else 0
                rows.append(
                    {
                        "node": node_name,
                        "status": str(n.get("status") or ""),
                        "uptime_h": round((uptime or 0) / 3600.0, 1),
                        "cpu_pct": round(cpu_ratio * 100.0, 1),
                        "mem_gib": round(mem / (1024 ** 3), 2),
                        "mem_pct": round((mem / maxmem) * 100.0, 1) if maxmem > 0 else 0.0,
                    }
                )
            rows = sorted(rows, key=lambda r: (-r["cpu_pct"], -r["mem_pct"]))
            return rows, ""
        except Exception as e:
            return [], f"Erro ao consultar nós: {e}"

    @app.callback(
        Output("tasks-table", "data"),
        Output("tasks-status", "children"),
        Input("tasks-refresh", "n_clicks"),
        State("tasks-status", "value"),
        State("tasks-type", "value"),
        prevent_initial_call=True,
    )
    def update_tasks(n_clicks, status_filter, type_filter):
        try:
            tasks = list_tasks()
            rows = []
            tf = (type_filter or "").strip().lower()
            for t in tasks:
                st = str(t.get("status") or "")
                tp = str(t.get("type") or "")
                if status_filter and status_filter != "all":
                    if status_filter == "error" and "error" not in st.lower():
                        continue
                    if status_filter == "OK" and "error" in st.lower():
                        continue
                if tf and tf not in tp.lower():
                    continue
                endtime = int(t.get("endtime") or 0)
                ts = datetime.fromtimestamp(endtime).strftime("%Y-%m-%d %H:%M:%S") if endtime else ""
                rows.append(
                    {
                        "time": ts,
                        "node": str(t.get("node") or ""),
                        "type": tp,
                        "status": st,
                        "user": str(t.get("user") or ""),
                    }
                )
            rows = sorted(rows, key=lambda r: r["time"], reverse=True)[:500]
            return rows, ""
        except Exception as e:
            return [], f"Erro ao consultar tarefas: {e}"

    # ==============================
    # Callback - Comparação (X vs Y)
    # ==============================

    @app.callback(
        Output("compare-graph", "figure"),
        Output("compare-table", "columns"),
        Output("compare-table", "data"),
        Output("compare-title", "children"),
        Output("compare-summary-cards", "children"),
        Input("compare-month-x", "value"),
        Input("compare-month-y", "value"),
    )
    def update_compare(month_x, month_y):
        df = load_months_data([month_x, month_y])
        if df.empty:
            return (
                empty_figure("Nenhum dado para os meses selecionados"),
                [],
                [],
                f"Comparação {month_x} vs {month_y} — sem dados",
                [dbc.Col(dbc.Alert("Nenhum dado para os meses selecionados.", color="warning"), md=12)],
            )

        agg = aggregate_by_cluster_and_month(df)
        agg_x = agg[agg["month"] == month_x].drop(columns=["month"])
        agg_y = agg[agg["month"] == month_y].drop(columns=["month"])

        fig = build_compare_bar_by_cluster(agg_x, agg_y, month_x, month_y)
        title = f"Comparação — {month_x} vs {month_y}"

        x = agg_x.rename(columns={"cpu_avg": "cpu_avg_x", "mem_avg": "mem_avg_x", "storage_avg": "storage_avg_x"})
        y = agg_y.rename(columns={"cpu_avg": "cpu_avg_y", "mem_avg": "mem_avg_y", "storage_avg": "storage_avg_y"})
        m = pd.merge(
            x[["cluster_name", "cpu_avg_x", "mem_avg_x", "storage_avg_x"]],
            y[["cluster_name", "cpu_avg_y", "mem_avg_y", "storage_avg_y"]],
            on="cluster_name", how="outer"
        ).fillna(0)

        m["cpu_delta"] = (m["cpu_avg_y"] - m["cpu_avg_x"]).round(1)
        m["mem_delta"] = (m["mem_avg_y"] - m["mem_avg_x"]).round(1)
        m["storage_delta"] = (m["storage_avg_y"] - m["storage_avg_x"]).round(1)
        m = m.sort_values("storage_avg_y", ascending=False)

        columns = [
            {"name": "Cluster", "id": "cluster_name"},
            {"name": f"CPU média {month_x}", "id": "cpu_avg_x"},
            {"name": f"CPU média {month_y}", "id": "cpu_avg_y"},
            {"name": "Δ CPU (Y-X)", "id": "cpu_delta"},
            {"name": f"Mem média {month_x}", "id": "mem_avg_x"},
            {"name": f"Mem média {month_y}", "id": "mem_avg_y"},
            {"name": "Δ Mem (Y-X)", "id": "mem_delta"},
            {"name": f"Storage médio {month_x}", "id": "storage_avg_x"},
            {"name": f"Storage médio {month_y}", "id": "storage_avg_y"},
            {"name": "Δ Storage (Y-X)", "id": "storage_delta"},
        ]
        data = m.to_dict("records")

        df_x = df[df["month"] == month_x].copy()
        df_y = df[df["month"] == month_y].copy()
        env_x = get_env_weighted_for_month(df_x)
        env_y = get_env_weighted_for_month(df_y)

        def _cmp_card(label: str, v_x: float, v_y: float, yellow_from: float, red_from: float):
            v_x = float(v_x)
            v_y = float(v_y)
            return dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Div(
                            [
                                html.Span(f"{label} — Ambiente (ponderado)", className="text-muted"),
                                html.Span(f" — {month_x}: {v_x:.1f}% → {month_y}: {v_y:.1f}%", className="text-muted ms-1"),
                            ],
                            style={"fontSize": "13px", "marginBottom": "4px"},
                        ),
                        dcc.Graph(
                            figure=build_gauge(
                                v_y,
                                title="",
                                show_number=True,
                                bar_color=CLUSTER_BAR_COLOR,
                                yellow_from=yellow_from,
                                red_from=red_from,
                            ),
                            config={"displayModeBar": False},
                        ),
                    ]),
                    className="shadow-sm",
                ),
                md=4,
                className="mb-3",
            )

        # 100% consistente: Storage também 80/95 aqui
        cards = [
            _cmp_card("CPU", env_x["cpu_avg"], env_y["cpu_avg"], yellow_from=50.0, red_from=80.0),
            _cmp_card("Memória", env_x["mem_avg"], env_y["mem_avg"], yellow_from=50.0, red_from=80.0),
            _cmp_card("Storage", env_x["storage_avg"], env_y["storage_avg"], yellow_from=80.0, red_from=90.0),
        ]

        return fig, columns, data, title, cards

    # ============================================================
    # Backup dados — callbacks
    # ============================================================

    @app.callback(
        Output("manual-backup-store", "data"),
        Output("upload-backup-status", "children"),
        Input("upload-backup", "contents"),
        State("upload-backup", "filename"),
        prevent_initial_call=True,
    )
    def on_upload_backup(contents, filename):
        if not contents:
            return None, "Nenhum arquivo carregado."
        try:
            df = parse_upload(contents, filename)
            df = normalize_backup_df(df)
            payload = df.to_json(date_format="iso", orient="split")
            return payload, f"Backup carregado: {filename} ({len(df)} linhas)"
        except Exception as e:
            return None, dbc.Alert(f"Erro ao carregar Backup: {e}", color="danger")

    @app.callback(
        Output("manual-backup-graph", "figure"),
        Output("manual-backup-table-wrap", "children"),
        Output("manual-backup-cards", "children"),
        Input("manual-backup-store", "data"),
    )
    def render_manual_backup(data_json):
        if not data_json:
            return (
                empty_figure("Backup dados", "Envie um arquivo para visualizar"),
                dbc.Alert("Envie o arquivo de Backup (CSV/XLSX).", color="info"),
                [],
            )

        df = pd.read_json(data_json, orient="split")
        detail, summary = split_backup_sections(df)

        fig = backup_simple_fig(detail)
        table = datatable_component(detail)
        cards = summarize_backup_cards(detail, summary)
        return fig, table, cards

    # ============================================================
    # Storage dados — callbacks
    # ============================================================

    @app.callback(
        Output("manual-storage-store", "data"),
        Output("upload-storage-status", "children"),
        Input("upload-storage", "contents"),
        State("upload-storage", "filename"),
        prevent_initial_call=True,
    )
    def on_upload_storage(contents, filename):
        if not contents:
            return None, "Nenhum arquivo carregado."
        try:
            df = parse_upload(contents, filename)
            df = normalize_storage_df(df)
            payload = df.to_json(date_format="iso", orient="split")
            return payload, f"Storage carregado: {filename} ({len(df)} linhas)"
        except Exception as e:
            return None, dbc.Alert(f"Erro ao carregar Storage: {e}", color="danger")

    @app.callback(
        Output("manual-storage-graph", "figure"),
        Output("manual-storage-table-wrap", "children"),
        Output("manual-storage-cards", "children"),
        Input("manual-storage-store", "data"),
    )
    def render_manual_storage(data_json):
        if not data_json:
            return (
                empty_figure("Storage dados", "Envie um arquivo para visualizar"),
                dbc.Alert("Envie o arquivo de Storage (CSV/XLSX).", color="info"),
                [],
            )

        df = pd.read_json(data_json, orient="split")
        fig = storage_simple_fig(df)
        table = datatable_component(df)
        cards = summarize_storage_cards(df)
        return fig, table, cards

    return app
