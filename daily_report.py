import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =========================
# 参数
# =========================
START_DATE = "2026-04-01"
END_DATE = "2026-04-30"
TOP_N = 5
LOOKBACK_DAYS = 5


# =========================
# 数据加载
# =========================
def load_data():
    df = pd.read_csv("sector_pct.csv")
    df["date"] = pd.to_datetime(df["date"])

    df = df[
        (df["date"] >= START_DATE) &
        (df["date"] <= END_DATE)
    ].copy()

    return df


# =========================
# 主线识别
# =========================
def detect_mainline(df):
    df["rank"] = df.groupby("date")["pct_chg"].rank(
        ascending=False,
        method="first"
    )

    df_top = df[df["rank"] <= TOP_N]

    latest_date = df["date"].max()
    recent_dates = sorted(df["date"].unique())[-LOOKBACK_DAYS:]

    df_recent = df_top[df_top["date"].isin(recent_dates)]

    stats = df_recent.groupby("sector").agg(
        appear_days=("date", "count"),
        avg_pct=("pct_chg", "mean"),
        best_rank=("rank", "min")
    ).reset_index()

    stats = stats.sort_values(
        by=["appear_days", "avg_pct"],
        ascending=False
    )

    return stats, df_top, latest_date


# =========================
# 新主线
# =========================
def detect_new_mainline(df_top, latest_date):
    today = df_top[df_top["date"] == latest_date]
    prev = df_top[df_top["date"] < latest_date]

    new = set(today["sector"]) - set(prev["sector"])
    return list(new)


# =========================
# 图1：热力图
# =========================
def heatmap_fig(df):
    pivot = df.pivot_table(
        index="sector",
        columns="date",
        values="pct_chg"
    )

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
        title="板块热力图（主线一眼看）"
    )

    return fig


# =========================
# 图2：排名趋势
# =========================
def rank_fig(df):
    df["rank"] = df.groupby("date")["pct_chg"].rank(
        ascending=False,
        method="first"
    )

    pivot = df.pivot_table(
        index="date",
        columns="sector",
        values="rank"
    )

    fig = go.Figure()

    for col in pivot.columns:
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[col],
                mode="lines",
                name=col
            )
        )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="板块排名趋势（主线轨迹）")

    return fig


# =========================
# 图3：每日最强
# =========================
def top_bar_fig(df):
    top = df.loc[df.groupby("date")["pct_chg"].idxmax()]

    fig = px.bar(
        top,
        x="date",
        y="pct_chg",
        color="sector",
        title="每日最强板块"
    )

    return fig


# =========================
# 主线面板HTML
# =========================
def build_panel(stats, new_sectors, latest_date):

    html = f"""
    <h2>📅 日期：{latest_date.date()}</h2>

    <h3>🔥 主线板块</h3>
    <ul>
    """

    for _, row in stats.head(5).iterrows():
        html += f"""
        <li>
        {row['sector']} |
        持续:{int(row['appear_days'])}天 |
        平均涨幅:{row['avg_pct']:.2f}% |
        最好排名:{int(row['best_rank'])}
        </li>
        """

    html += "</ul>"

    html += "<h3>⚡ 新主线</h3><ul>"

    if new_sectors:
        for s in new_sectors:
            html += f"<li>{s}</li>"
    else:
        html += "<li>无</li>"

    html += "</ul>"

    return html


# =========================
# 生成HTML报告
# =========================
def generate_report():

    df = load_data()

    stats, df_top, latest_date = detect_mainline(df)
    new_sectors = detect_new_mainline(df_top, latest_date)

    panel_html = build_panel(stats, new_sectors, latest_date)

    fig1 = heatmap_fig(df)
    fig2 = rank_fig(df)
    fig3 = top_bar_fig(df)

    with open("daily_report.html", "w", encoding="utf-8") as f:

        f.write("<html><head><meta charset='utf-8'></head><body>")

        f.write("<h1>📊 A股板块复盘报告</h1>")

        f.write(panel_html)

        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
        f.write(fig3.to_html(full_html=False, include_plotlyjs=False))

        f.write("</body></html>")

    print("🎉 报告已生成：daily_report.html")


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    generate_report()