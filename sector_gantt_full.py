import pandas as pd
import plotly.express as px
import random
from datetime import datetime, timedelta

# ==============================
# 参数区（你以后主要改这里）
# ==============================

MODE = "APRIL"   # "APRIL" 或 "FULL"

TOP_N = 5        # 每天选前N板块


# ==============================
# 1️⃣ 生成 mock 数据（1-4月）
# ==============================

def generate_data():
    sectors = [
        "机器人", "半导体", "军工", "算力", "新能源",
        "医药", "消费电子", "低空经济", "证券", "游戏"
    ]

    start_date = datetime(2026, 1, 1)
    rows = []

    for i in range(120):  # 4个月
        date = start_date + timedelta(days=i)

        # 每5天切换一个热点板块
        hot_sector = sectors[(i // 5) % len(sectors)]

        for sector in sectors:
            pct = random.uniform(-2, 3)

            # 热点板块加权
            if sector == hot_sector:
                pct += random.uniform(3, 6)

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "sector": sector,
                "pct_chg": round(pct, 2)
            })

    df = pd.DataFrame(rows)
    df.to_csv("sector_pct.csv", index=False, encoding="utf-8-sig")

    print("✅ 已生成数据：sector_pct.csv")


# ==============================
# 2️⃣ 生成甘特图
# ==============================

def build_gantt():
    print("🚀 开始生成甘特图...")

    df = pd.read_csv("sector_pct.csv")
    df["date"] = pd.to_datetime(df["date"])

    # ===== 时间过滤 =====
    if MODE == "APRIL":
        start = "2026-04-01"
        end = "2026-04-30"
    else:
        start = "2026-01-01"
        end = "2026-04-30"

    df = df[
        (df["date"] >= start) &
        (df["date"] <= end)
    ]

    print(f"📅 当前区间：{start} → {end}")

    # ===== 排名 =====
    df["rank"] = df.groupby("date")["pct_chg"].rank(
        ascending=False,
        method="first"
    )

    top_df = df[df["rank"] <= TOP_N]

    # ===== 构造甘特数据 =====
    gantt_rows = []

    for sector, group in top_df.groupby("sector"):
        group = group.sort_values("date")

        start_date = None
        prev_date = None
        pct_list = []

        for _, row in group.iterrows():
            current_date = row["date"]

            if start_date is None:
                start_date = current_date
                prev_date = current_date
                pct_list = [row["pct_chg"]]
                continue

            # 连续判断
            if (current_date - prev_date).days <= 3:
                prev_date = current_date
                pct_list.append(row["pct_chg"])
            else:
                gantt_rows.append({
                    "sector": sector,
                    "start": start_date,
                    "end": prev_date + pd.Timedelta(days=1),
                    "avg_pct": sum(pct_list) / len(pct_list)
                })

                start_date = current_date
                prev_date = current_date
                pct_list = [row["pct_chg"]]

        # 收尾
        if start_date is not None:
            gantt_rows.append({
                "sector": sector,
                "start": start_date,
                "end": prev_date + pd.Timedelta(days=1),
                "avg_pct": sum(pct_list) / len(pct_list)
            })

    gantt_df = pd.DataFrame(gantt_rows)

    if gantt_df.empty:
        print("❌ 没有数据，检查时间区间或TOP_N")
        return

    # ===== 画图 =====
    fig = px.timeline(
        gantt_df,
        x_start="start",
        x_end="end",
        y="sector",
        color="avg_pct",
        title=f"板块轮动甘特图（{MODE}）",
        hover_data=["avg_pct"]
    )

    fig.update_yaxes(autorange="reversed")

    fig.write_html("sector_gantt.html")

    print("🎉 甘特图已生成：sector_gantt.html")


# ==============================
# 主程序
# ==============================

if __name__ == "__main__":
    generate_data()
    build_gantt()