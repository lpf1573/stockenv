import pandas as pd
import time

# =========================
# 参数
# =========================
TOP_N_SECTORS = 3        # 取前3个主线板块
TOP_N_STOCKS = 5         # 每个板块取前5只
MAX_PCT = 9              # 过滤：避免当天涨太多（防追高）
MODE = "CSV"             # "CSV" 或 "LIVE"

# =========================
# 1️⃣ 读取板块数据（你已有）
# =========================
def load_sector_data():
    df = pd.read_csv("sector_pct.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

# =========================
# 2️⃣ 主线识别（复用你之前逻辑）
# =========================
def detect_mainline(df):
    df["rank"] = df.groupby("date")["pct_chg"].rank(ascending=False)
    df_top = df[df["rank"] <= 5]

    latest = df["date"].max()
    recent = sorted(df["date"].unique())[-5:]

    stats = df_top[df_top["date"].isin(recent)].groupby("sector").agg(
        appear_days=("date","count"),
        avg_pct=("pct_chg","mean")
    ).reset_index()

    stats = stats.sort_values(
        by=["appear_days","avg_pct"],
        ascending=False
    )

    return stats.head(TOP_N_SECTORS)["sector"].tolist()

# =========================
# 3️⃣ 获取板块个股（两种模式）
# =========================
def get_sector_stocks(sector):

    if MODE == "CSV":
        # 👉 你需要准备这个文件（见下说明）
        df = pd.read_csv("stock_data.csv")
        return df[df["sector"] == sector]

    else:
        import akshare as ak
        try:
            df = ak.stock_board_industry_cons_em(symbol=sector)
            df = df[["代码","名称","涨跌幅","成交额","换手率"]]
            df.columns = ["code","name","pct_chg","amount","turnover"]
            return df
        except:
            return pd.DataFrame()

# =========================
# 4️⃣ 打分模型（核心）
# =========================
def score(df):

    if df.empty:
        return df

    df["rank_pct"] = df["pct_chg"].rank(ascending=False)
    df["rank_amount"] = df["amount"].rank(ascending=False)
    df["rank_turnover"] = df["turnover"].rank(ascending=False)

    df["score"] = (
        (1/df["rank_pct"])*50 +
        (1/df["rank_amount"])*30 +
        (1/df["rank_turnover"])*20
    ) * 100

    return df.sort_values("score", ascending=False)

# =========================
# 5️⃣ 候选股生成
# =========================
def pick_candidates():

    sector_df = load_sector_data()
    main_sectors = detect_mainline(sector_df)

    print("\n🔥 主线板块：", main_sectors)

    result = []

    for sec in main_sectors:

        df = get_sector_stocks(sec)

        if df.empty:
            continue

        df = score(df)

        # 👉 过滤：避免涨太多（追高）
        df = df[df["pct_chg"] < MAX_PCT]

        top = df.head(TOP_N_STOCKS)
        top["sector"] = sec

        result.append(top)

        time.sleep(1)

    if result:
        final = pd.concat(result)
        print("\n📊 明日候选股：\n")
        print(final[["sector","code","name","pct_chg","score"]])
    else:
        print("❌ 没有数据")

# =========================
# 主程序
# =========================
if __name__ == "__main__":
    pick_candidates()