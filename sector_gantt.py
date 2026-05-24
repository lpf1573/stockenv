import pandas as pd
import plotly.express as px

print("🚀 开始生成甘特图...")

df = pd.read_csv("sector_pct.csv")
df["date"] = pd.to_datetime(df["date"])

print("数据预览：")
print(df.head())

# ===== 核心：每一天选 Top3 板块 =====
df["rank"] = df.groupby("date")["pct_chg"].rank(
    ascending=False,
    method="first"
)

top_df = df[df["rank"] <= 3]

print("\n筛选后数据：")
print(top_df.head())

# ===== 构造甘特图数据 =====
gantt_rows = []

for sector, group in top_df.groupby("sector"):
    group = group.sort_values("date")

    start = group["date"].min()
    end = group["date"].max()

    gantt_rows.append({
        "sector": sector,
        "start": start,
        "end": end,
        "avg_pct": group["pct_chg"].mean()
    })

gantt_df = pd.DataFrame(gantt_rows)

print("\n甘特数据：")
print(gantt_df)

# ===== 画图 =====
fig = px.timeline(
    gantt_df,
    x_start="start",
    x_end="end",
    y="sector",
    color="avg_pct",
    title="板块轮动甘特图（测试版）"
)

fig.update_yaxes(autorange="reversed")

fig.write_html("gantt.html")

print("🎉 甘特图已生成：gantt.html")