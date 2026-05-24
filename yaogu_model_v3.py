"""
yaogu_model_v3.py

妖股模型 v3：短线实战版

核心思想：
1. 不追求每天都有票
2. 只筛“趋势 + 主力行为 + 短线结构 + 风险不过高”的票
3. 输出更干净的短线候选池

输入：
    output/a_share_labeled_dataset_with_auto_manual_label.csv
    output/latest_stock_signals.csv

输出：
    output/yaogu_model_v3_signals.csv

运行：
    python3 yaogu_model_v3.py
"""

import os
import numpy as np
import pandas as pd


DATASET_PATH = "output/a_share_labeled_dataset_with_auto_manual_label.csv"
LATEST_SIGNAL_PATH = "output/latest_stock_signals.csv"
OUTPUT_PATH = "output/yaogu_model_v3_signals.csv"


def safe_num(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def add_short_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()

    g["high_10d"] = g["high"].rolling(10).max()
    g["high_20d"] = g["high"].rolling(20).max()
    g["high_60d"] = g["high"].rolling(60).max()

    g["low_20d"] = g["low"].rolling(20).min()
    g["low_60d"] = g["low"].rolling(60).min()

    g["amount_ma20"] = g["amount"].rolling(20).mean()
    g["volume_ma20"] = g["volume"].rolling(20).mean()
    g["volume_ratio_v3"] = g["volume"] / g["volume_ma20"]

    g["pct_3d"] = g["close"].pct_change(3) * 100
    g["pct_5d"] = g["close"].pct_change(5) * 100
    g["pct_10d"] = g["close"].pct_change(10) * 100
    g["pct_20d"] = g["close"].pct_change(20) * 100

    g["is_red"] = (g["close"] > g["open"]).astype(int)
    g["red_count_3d"] = g["is_red"].rolling(3).sum()
    g["red_count_5d"] = g["is_red"].rolling(5).sum()

    g["is_big_up"] = (g["pct_change"] >= 5).astype(int)
    g["is_limit_like"] = (g["pct_change"] >= 9.5).astype(int)
    g["big_up_count_10d"] = g["is_big_up"].rolling(10).sum()
    g["limit_like_count_20d"] = g["is_limit_like"].rolling(20).sum()

    g["break_20d_high"] = (g["close"] >= g["high_20d"].shift(1) * 0.99).astype(int)
    g["break_60d_high"] = (g["close"] >= g["high_60d"].shift(1) * 0.99).astype(int)

    g["position_60d"] = (g["close"] - g["low_60d"]) / (g["high_60d"] - g["low_60d"])

    g["upper_shadow_ratio_v3"] = (
        g["high"] - g[["open", "close"]].max(axis=1)
    ) / g["open"]

    g["lower_shadow_ratio_v3"] = (
        g[["open", "close"]].min(axis=1) - g["low"]
    ) / g["open"]

    g["real_body_ratio_v3"] = (g["close"] - g["open"]).abs() / g["open"]

    # 缩量回踩：靠近20日线，同时量比不高
    if "ma20" in g.columns:
        g["near_ma20"] = ((g["close"] / g["ma20"] - 1).abs() <= 0.03).astype(int)
    else:
        g["near_ma20"] = 0

    g["shrink_pullback"] = (
        (g["near_ma20"] == 1)
        & (g["volume_ratio_v3"] <= 0.8)
        & (g["close"] > g["open"])
    ).astype(int)

    return g


def score_yaogu_v3(row: pd.Series) -> pd.Series:
    score = 0
    risk = 0
    reasons = []

    rainbow = safe_num(row.get("rainbow_score"))
    main_force = safe_num(row.get("main_force_score"))
    model_score = safe_num(row.get("model_score"))
    prob_up = safe_num(row.get("prob_up"))
    prob_down = safe_num(row.get("prob_down"))
    pred_label = safe_num(row.get("pred_label"))

    pct_change = safe_num(row.get("pct_change"))
    pct_5d = safe_num(row.get("pct_5d"))
    pct_10d = safe_num(row.get("pct_10d"))
    pct_20d = safe_num(row.get("pct_20d"))

    turnover = safe_num(row.get("turnover"))
    volume_ratio = safe_num(row.get("volume_ratio_v3"))
    amount_ma20 = safe_num(row.get("amount_ma20"))

    red_count_3d = safe_num(row.get("red_count_3d"))
    red_count_5d = safe_num(row.get("red_count_5d"))
    big_up_count_10d = safe_num(row.get("big_up_count_10d"))
    limit_like_count_20d = safe_num(row.get("limit_like_count_20d"))

    break_20d = safe_num(row.get("break_20d_high"))
    break_60d = safe_num(row.get("break_60d_high"))
    position_60d = safe_num(row.get("position_60d"))

    upper_shadow = safe_num(row.get("upper_shadow_ratio_v3"))
    lower_shadow = safe_num(row.get("lower_shadow_ratio_v3"))
    shrink_pullback = safe_num(row.get("shrink_pullback"))

    # =========================
    # 1. 趋势基础
    # =========================

    if rainbow >= 90:
        score += 18
        reasons.append("彩虹趋势极强")
    elif rainbow >= 70:
        score += 12
        reasons.append("彩虹趋势较强")
    elif rainbow >= 50:
        score += 6
        reasons.append("趋势初步转强")

    # =========================
    # 2. 主力行为
    # =========================

    if main_force >= 80:
        score += 22
        reasons.append("主力行为强")
    elif main_force >= 60:
        score += 16
        reasons.append("主力行为活跃")
    elif main_force >= 30:
        score += 8
        reasons.append("主力行为初现")

    # =========================
    # 3. 模型辅助
    # =========================

    if model_score >= 60 and prob_up >= 0.6:
        score += 18
        reasons.append("模型强看多")
    elif model_score >= 30 and prob_up >= 0.3:
        score += 8
        reasons.append("模型轻度看多")

    if pred_label == 1:
        score += 6
        reasons.append("模型标签偏多")

    # =========================
    # 4. 放量结构
    # =========================

    if volume_ratio >= 3:
        score += 18
        reasons.append("爆量")
    elif volume_ratio >= 2:
        score += 13
        reasons.append("明显放量")
    elif volume_ratio >= 1.5:
        score += 7
        reasons.append("温和放量")

    if amount_ma20 >= 500_000_000:
        score += 8
        reasons.append("成交额充足")
    elif amount_ma20 >= 200_000_000:
        score += 4
        reasons.append("成交额尚可")

    # =========================
    # 5. 突破结构
    # =========================

    if break_60d == 1:
        score += 20
        reasons.append("突破60日高点")
    elif break_20d == 1:
        score += 14
        reasons.append("突破20日高点")

    # =========================
    # 6. 短线情绪
    # =========================

    if pct_change >= 9.5:
        score += 18
        reasons.append("涨停特征")
    elif pct_change >= 5:
        score += 12
        reasons.append("大阳线")
    elif pct_change >= 2:
        score += 5
        reasons.append("日内走强")

    if red_count_3d >= 3:
        score += 8
        reasons.append("三连阳")
    elif red_count_5d >= 4:
        score += 6
        reasons.append("五日多阳")

    if big_up_count_10d >= 2:
        score += 8
        reasons.append("近期多次大涨")

    if limit_like_count_20d >= 1:
        score += 8
        reasons.append("近期有涨停基因")

    if lower_shadow >= 0.03 and pct_change > 0:
        score += 6
        reasons.append("下影承接")

    if shrink_pullback == 1:
        score += 10
        reasons.append("缩量回踩")

    # =========================
    # 7. 低位启动加分
    # =========================

    if 0.25 <= position_60d <= 0.65 and pct_20d < 25:
        score += 10
        reasons.append("低中位启动")
    elif position_60d < 0.25 and pct_change > 2:
        score += 6
        reasons.append("低位异动")

    # =========================
    # 风险扣分
    # =========================

    if prob_down >= 0.55:
        risk += 18
        reasons.append("模型下跌概率偏高")

    if pred_label == -1:
        risk += 12
        reasons.append("模型标签偏空")

    if pct_change <= -3:
        risk += 25
        reasons.append("当日明显走弱")

    if upper_shadow >= 0.06:
        risk += 15
        reasons.append("上影线过长")

    if pct_5d >= 25:
        risk += 15
        reasons.append("5日涨幅过大")

    if pct_10d >= 45:
        risk += 20
        reasons.append("10日涨幅过大")

    if turnover >= 25:
        risk += 12
        reasons.append("换手过高")

    if position_60d >= 0.9 and pct_20d >= 40:
        risk += 18
        reasons.append("高位加速风险")

    final_score = max(0, min(score - risk, 100))

    if final_score >= 80 and risk <= 20:
        level = "S"
        action = "重点观察"
    elif final_score >= 65 and risk <= 30:
        level = "A"
        action = "观察"
    elif final_score >= 50:
        level = "B"
        action = "轻观察"
    else:
        level = "C"
        action = "忽略"

    return pd.Series(
        {
            "yaogu_v3_score": final_score,
            "yaogu_v3_raw_score": score,
            "yaogu_v3_risk": risk,
            "yaogu_v3_level": level,
            "yaogu_v3_action": action,
            "yaogu_v3_reason": "；".join(reasons),
        }
    )


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"找不到数据集：{DATASET_PATH}")

    print("读取训练数据集...")
    df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig", dtype={"symbol": str})
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    groups = []
    for symbol, g in df.groupby("symbol"):
        groups.append(add_short_features(g))

    df = pd.concat(groups, ignore_index=True)

    latest_signal = None
    if os.path.exists(LATEST_SIGNAL_PATH):
        latest_signal = pd.read_csv(LATEST_SIGNAL_PATH, encoding="utf-8-sig", dtype={"symbol": str})
        latest_signal["symbol"] = latest_signal["symbol"].astype(str).str.zfill(6)

        keep_cols = ["symbol", "prob_up", "prob_down", "model_score", "pred_label"]
        keep_cols = [c for c in keep_cols if c in latest_signal.columns]

        df = df.merge(
            latest_signal[keep_cols],
            on="symbol",
            how="left",
            suffixes=("", "_latest"),
        )

    else:
        df["prob_up"] = 0
        df["prob_down"] = 0
        df["model_score"] = 0
        df["pred_label"] = 0

    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()

    print(f"最新日期：{latest_date}")
    print(f"最新股票数量：{len(latest)}")

    score_df = latest.apply(score_yaogu_v3, axis=1)
    result = pd.concat([latest, score_df], axis=1)

    output_cols = [
        "symbol",
        "date",
        "close",
        "pct_change",
        "turnover",
        "rainbow_score",
        "main_force_score",
        "model_score",
        "prob_up",
        "prob_down",
        "pred_label",
        "volume_ratio_v3",
        "pct_5d",
        "pct_10d",
        "pct_20d",
        "break_20d_high",
        "break_60d_high",
        "red_count_3d",
        "big_up_count_10d",
        "limit_like_count_20d",
        "position_60d",
        "yaogu_v3_score",
        "yaogu_v3_raw_score",
        "yaogu_v3_risk",
        "yaogu_v3_level",
        "yaogu_v3_action",
        "yaogu_v3_reason",
    ]

    output_cols = [c for c in output_cols if c in result.columns]

    result = result.sort_values(
        ["yaogu_v3_score", "yaogu_v3_raw_score"],
        ascending=False,
    )

    os.makedirs("output", exist_ok=True)
    result[output_cols].to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    strong = result[result["yaogu_v3_level"].isin(["S", "A"])]

    print("\n==============================")
    print("妖股模型 v3 扫描完成")
    print(f"输出文件：{OUTPUT_PATH}")
    print(f"S/A 候选数量：{len(strong)}")

    if len(strong) > 0:
        print("\nTop 候选：")
        print(strong[output_cols].head(20))
    else:
        print("\n今天没有 S/A 级候选，建议空仓观察。")


if __name__ == "__main__":
    main()
