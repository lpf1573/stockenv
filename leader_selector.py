import akshare as ak
import pandas as pd
import time

# ====== 参数 ======
TOP_N_PER_SECTOR = 3


# ====== 获取板块成分股 ======
def get_sector_stocks(sector_name):
    try:
        df = ak.stock_board_industry_cons_em(symbol=sector_name)
        df = df[["代码", "名称", "涨跌幅", "成交额", "换手率"]]
        df.columns = ["code", "name", "pct_chg", "amount", "turnover"]
        return df
    except Exception as e:
        print(f"❌ 获取板块 {sector_name} 失败：{e}")
        return pd.DataFrame()


# ====== 龙头评分模型 ======
def score_stocks(df):
    if df.empty:
        return df

    # 排名（越靠前越好）
    df["rank_pct"] = df["pct_chg"].rank(ascending=False)
    df["rank_amount"] = df["amount"].rank(ascending=False)
    df["rank_turnover"] = df["turnover"].rank(ascending=False)

    # 标准化
    df["score"] = (
        (1 / df["rank_pct"]) * 50 +
        (1 / df["rank_amount"]) * 30 +
        (1 / df["rank_turnover"]) * 20
    ) * 100

    return df.sort_values("score", ascending=False)


# ====== 主函数：板块 → 龙头 ======
def get_sector_leaders(sectors):
    result = []

    for sector in sectors:
        print(f"\n🔍 处理板块：{sector}")

        df = get_sector_stocks(sector)

        if df.empty:
            continue

        df = score_stocks(df)

        top_df = df.head(TOP_N_PER_SECTOR)

        top_df["sector"] = sector

        result.append(top_df)

        time.sleep(1)  # 防止接口炸

    if result:
        return pd.concat(result)
    else:
        return pd.DataFrame()


# ====== 测试入口 ======
if __name__ == "__main__":
    # 👉 这里替换成你甘特图识别出来的主线板块
    hot_sectors = ["机器人", "半导体", "算力"]

    leaders = get_sector_leaders(hot_sectors)

    if not leaders.empty:
        print("\n🔥 主线龙头股：\n")
        print(leaders[["sector", "code", "name", "pct_chg", "score"]])
    else:
        print("❌ 没有获取到数据")