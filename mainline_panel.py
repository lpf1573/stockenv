import pandas as pd

# =============================
# 参数
# =============================
TOP_N = 5
LOOKBACK_DAYS = 5   # 看最近几天持续性


# =============================
# 读取数据
# =============================
def load_data():
    df = pd.read_csv("sector_pct.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


# =============================
# 主线识别
# =============================
def detect_mainline(df):

    # 排名
    df["rank"] = df.groupby("date")["pct_chg"].rank(
        ascending=False,
        method="first"
    )

    # 只看前N
    df_top = df[df["rank"] <= TOP_N].copy()

    # 最近日期
    latest_date = df["date"].max()

    recent_dates = sorted(df["date"].unique())[-LOOKBACK_DAYS:]

    df_recent = df_top[df_top["date"].isin(recent_dates)]

    # =============================
    # 统计
    # =============================
    stats = df_recent.groupby("sector").agg(
        appear_days=("date", "count"),
        avg_pct=("pct_chg", "mean"),
        best_rank=("rank", "min")
    ).reset_index()

    # 排序逻辑（核心）
    stats = stats.sort_values(
        by=["appear_days", "avg_pct"],
        ascending=False
    )

    return stats, latest_date, df_top


# =============================
# 新主线识别
# =============================
def detect_new_mainline(df_top, latest_date):

    today_df = df_top[df_top["date"] == latest_date]
    prev_df = df_top[df_top["date"] < latest_date]

    today_sectors = set(today_df["sector"])
    old_sectors = set(prev_df["sector"])

    new_sectors = today_sectors - old_sectors

    return list(new_sectors)


# =============================
# 输出面板
# =============================
def print_panel(stats, new_sectors, latest_date):

    print("\n==============================")
    print(f"📅 日期：{latest_date.date()}")
    print("==============================\n")

    print("🔥 主线板块（按强度排序）：\n")

    for i, row in stats.head(5).iterrows():
        print(
            f"{row['sector']:<8} | "
            f"持续:{int(row['appear_days'])}天 | "
            f"平均涨幅:{row['avg_pct']:.2f}% | "
            f"最好排名:{int(row['best_rank'])}"
        )

    print("\n⚡ 新主线（今天首次出现）：\n")

    if new_sectors:
        for s in new_sectors:
            print(f"👉 {s}")
    else:
        print("无")

    print("\n==============================\n")


# =============================
# 主程序
# =============================
if __name__ == "__main__":

    df = load_data()

    stats, latest_date, df_top = detect_mainline(df)

    new_sectors = detect_new_mainline(df_top, latest_date)

    print_panel(stats, new_sectors, latest_date)