import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 参数
# =========================
START_DATE = "2026-04-01"
END_DATE = "2026-04-30"

# =========================
# 读取数据
# =========================
def load_data():
    df = pd.read_csv("sector_pct.csv")
    df["date"] = pd.to_datetime(df["date"])

    df = df[
        (df["date"] >= START_DATE) &
        (df["date"] <= END_DATE)
    ]

    return df


# =========================
# 1️⃣ 热力图（最重要）
# =========================
def draw_heatmap(df):
    pivot = df.pivot_table(
        index="sector",
        columns="date",
        values="pct_chg"
    )

    plt.figure(figsize=(16, 6))

    sns.heatmap(
        pivot,
        cmap="RdYlGn_r",
        center=0
    )

    plt.title("🔥 板块热力图（主线一眼看）")
    plt.xlabel("时间")
    plt.ylabel("板块")

    plt.tight_layout()
    plt.show()


# =========================
# 2️⃣ 板块排名趋势图（核心）
# =========================
def draw_rank_line(df):

    df["rank"] = df.groupby("date")["pct_chg"].rank(
        ascending=False,
        method="first"
    )

    pivot = df.pivot_table(
        index="date",
        columns="sector",
        values="rank"
    )

    plt.figure(figsize=(16, 6))

    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], label=col)

    plt.gca().invert_yaxis()

    plt.title("📈 板块排名趋势（主线轨迹）")
    plt.xlabel("时间")
    plt.ylabel("排名（越小越强）")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


# =========================
# 3️⃣ 每日最强板块柱状图
# =========================
def draw_top_bar(df):

    top = df.loc[df.groupby("date")["pct_chg"].idxmax()]

    plt.figure(figsize=(16, 5))

    plt.bar(top["date"].astype(str), top["pct_chg"])

    plt.xticks(rotation=45)

    plt.title("🏆 每日最强板块")
    plt.xlabel("日期")
    plt.ylabel("涨幅")

    plt.tight_layout()
    plt.show()


# =========================
# 主程序
# =========================
if __name__ == "__main__":

    df = load_data()

    print("数据加载完成：", df.shape)

    draw_heatmap(df)
    draw_rank_line(df)
    draw_top_bar(df)