"""
filter_stock_pool.py

股票池清洗器：自动筛掉“垃圾票”

功能：
1. 读取已有数据集 output/a_share_labeled_dataset_with_auto_manual_label.csv
2. 按最近 N 天统计每只股票质量
3. 剔除：
   - 数据太少
   - 最近成交额太低
   - 最近换手太低
   - 波动太低，长期横盘
   - 最近跌幅过大
   - 主力行为长期太弱
4. 输出：
   output/clean_stock_pool.csv
   output/removed_stock_pool.csv

运行：
    python3 filter_stock_pool.py
"""

import os
import pandas as pd
import numpy as np


DATASET_PATH = "output/a_share_labeled_dataset_with_auto_manual_label.csv"
CLEAN_POOL_PATH = "output/clean_stock_pool.csv"
REMOVED_POOL_PATH = "output/removed_stock_pool.csv"


# =========================
# 参数区：你可以以后微调
# =========================

LOOKBACK_DAYS = 60

MIN_HISTORY_ROWS = 180

# 最近60日平均成交额，低于这个剔除
MIN_AVG_AMOUNT = 200_000_000

# 最近60日平均换手率，低于这个剔除
MIN_AVG_TURNOVER = 1.0

# 最近60日平均波动率，低于这个剔除
MIN_AVG_VOLATILITY = 0.015

# 最近60日最大涨跌幅区间太小，说明横盘太久
MIN_PRICE_RANGE_60D = 0.18

# 最近20日跌幅过大，暂时剔除
MAX_RECENT_20D_DROP = -0.25

# 最近主力行为平均分太低，剔除
MIN_AVG_MAIN_FORCE_SCORE = 5

# 最近趋势均值过低，剔除
MIN_AVG_RAINBOW_SCORE = 10


def normalize_symbol(x):
    return str(x).replace(".0", "").zfill(6)


def evaluate_symbol(symbol: str, g: pd.DataFrame) -> dict:
    g = g.sort_values("date").copy()

    result = {
        "symbol": symbol,
        "keep": True,
        "remove_reason": "",
    }

    if len(g) < MIN_HISTORY_ROWS:
        result["keep"] = False
        result["remove_reason"] = f"历史数据不足({len(g)})"
        return result

    recent = g.tail(LOOKBACK_DAYS).copy()

    avg_amount = recent["amount"].mean()
    avg_turnover = recent["turnover"].mean()
    avg_volatility = recent["return_1d"].std()

    high_60 = recent["high"].max()
    low_60 = recent["low"].min()
    price_range_60 = high_60 / low_60 - 1 if low_60 > 0 else 0

    close_now = recent["close"].iloc[-1]
    close_20d_ago = recent["close"].iloc[-20] if len(recent) >= 20 else recent["close"].iloc[0]
    recent_20d_return = close_now / close_20d_ago - 1 if close_20d_ago > 0 else 0

    avg_main_force = recent["main_force_score"].mean() if "main_force_score" in recent.columns else 0
    avg_rainbow = recent["rainbow_score"].mean() if "rainbow_score" in recent.columns else 0

    latest_close = recent["close"].iloc[-1]
    latest_date = recent["date"].iloc[-1]

    result.update(
        {
            "latest_date": latest_date,
            "latest_close": latest_close,
            "avg_amount": avg_amount,
            "avg_turnover": avg_turnover,
            "avg_volatility": avg_volatility,
            "price_range_60d": price_range_60,
            "recent_20d_return": recent_20d_return,
            "avg_main_force_score": avg_main_force,
            "avg_rainbow_score": avg_rainbow,
        }
    )

    reasons = []

    if avg_amount < MIN_AVG_AMOUNT:
        reasons.append(f"成交额低({avg_amount:.0f})")

    if avg_turnover < MIN_AVG_TURNOVER:
        reasons.append(f"换手率低({avg_turnover:.2f})")

    if avg_volatility < MIN_AVG_VOLATILITY:
        reasons.append(f"波动太低({avg_volatility:.4f})")

    if price_range_60 < MIN_PRICE_RANGE_60D:
        reasons.append(f"60日横盘({price_range_60:.2%})")

    if recent_20d_return < MAX_RECENT_20D_DROP:
        reasons.append(f"近20日跌幅过大({recent_20d_return:.2%})")

    if avg_main_force < MIN_AVG_MAIN_FORCE_SCORE:
        reasons.append(f"主力行为长期弱({avg_main_force:.1f})")

    if avg_rainbow < MIN_AVG_RAINBOW_SCORE:
        reasons.append(f"趋势长期弱({avg_rainbow:.1f})")

    if reasons:
        result["keep"] = False
        result["remove_reason"] = "；".join(reasons)

    return result


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"找不到数据集：{DATASET_PATH}")

    print("读取数据集...")
    df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig", dtype={"symbol": str})
    df["symbol"] = df["symbol"].apply(normalize_symbol)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    needed_cols = ["amount", "turnover", "return_1d", "high", "low", "close"]
    missing = [c for c in needed_cols if c not in df.columns]

    if missing:
        raise ValueError(f"数据集缺少必要字段：{missing}")

    records = []

    for symbol, g in df.groupby("symbol"):
        records.append(evaluate_symbol(symbol, g))

    result_df = pd.DataFrame(records)

    clean_df = result_df[result_df["keep"] == True].copy()
    removed_df = result_df[result_df["keep"] == False].copy()

    clean_df = clean_df.sort_values(
        ["avg_amount", "avg_turnover", "avg_rainbow_score"],
        ascending=False,
    )

    os.makedirs("output", exist_ok=True)

    clean_df.to_csv(CLEAN_POOL_PATH, index=False, encoding="utf-8-sig")
    removed_df.to_csv(REMOVED_POOL_PATH, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("股票池清洗完成")
    print(f"原始股票数：{df['symbol'].nunique()}")
    print(f"保留股票数：{len(clean_df)}")
    print(f"剔除股票数：{len(removed_df)}")
    print(f"保留股票池：{CLEAN_POOL_PATH}")
    print(f"剔除明细：{REMOVED_POOL_PATH}")

    if len(clean_df) > 0:
        print("\n保留股票 Top 20：")
        print(clean_df.head(20))

    if len(removed_df) > 0:
        print("\n剔除原因 Top 20：")
        print(removed_df[["symbol", "remove_reason"]].head(20))


if __name__ == "__main__":
    main()
