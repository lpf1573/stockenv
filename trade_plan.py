"""
trade_plan.py

自动交易计划模块

读取：
    output/latest_stock_signals.csv
    output/yaogu_model_v3_signals.csv
    output/clean_stock_pool.csv
    output/stock_meta_map.csv

输出：
    output/trade_plan.csv
    output/trade_plan.md

运行：
    python3 trade_plan.py
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd


SIGNAL_PATH = "output/latest_stock_signals.csv"
YAOGU_PATH = "output/yaogu_model_v3_signals.csv"
CLEAN_POOL_PATH = "output/clean_stock_pool.csv"
META_PATH = "output/stock_meta_map.csv"

OUTPUT_CSV = "output/trade_plan.csv"
OUTPUT_MD = "output/trade_plan.md"


def safe_read(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str)


def normalize_symbol(s):
    return (
        s.astype(str)
        .str.replace("sh.", "", regex=False)
        .str.replace("sz.", "", regex=False)
        .str.replace(".0", "", regex=False)
        .str.zfill(6)
    )


def to_numeric(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_meta(df):
    df = df.copy()

    if "symbol" not in df.columns:
        if "code" in df.columns:
            df["symbol"] = df["code"]
        elif "股票代码" in df.columns:
            df["symbol"] = df["股票代码"]
        else:
            raise ValueError(f"数据缺少 symbol/code 列：{df.columns.tolist()}")

    df["symbol"] = normalize_symbol(df["symbol"])

    meta = safe_read(META_PATH)
    if meta is None or meta.empty:
        df["stock_name"] = ""
        df["industry"] = "未知"
        df["board"] = "未知"
        df["themes"] = "其他"
        df["display_name"] = df["symbol"]
        return df

    if "symbol" not in meta.columns:
        if "code" in meta.columns:
            meta["symbol"] = meta["code"]
        else:
            meta["symbol"] = ""

    meta["symbol"] = normalize_symbol(meta["symbol"])

    keep_cols = ["symbol", "stock_name", "industry", "board", "themes"]
    keep_cols = [c for c in keep_cols if c in meta.columns]
    meta = meta[keep_cols].drop_duplicates("symbol")

    df = df.merge(meta, on="symbol", how="left")

    for col, default in [
        ("stock_name", ""),
        ("industry", "未知"),
        ("board", "未知"),
        ("themes", "其他"),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    df["display_name"] = df["symbol"] + " " + df["stock_name"].astype(str)
    return df


def load_clean_symbols():
    clean_df = safe_read(CLEAN_POOL_PATH)

    if clean_df is None or clean_df.empty:
        return set()

    if "symbol" not in clean_df.columns:
        if "code" in clean_df.columns:
            clean_df["symbol"] = clean_df["code"]
        else:
            return set()

    clean_df["symbol"] = normalize_symbol(clean_df["symbol"])
    return set(clean_df["symbol"].tolist())


def prepare_data():
    signal_df = safe_read(SIGNAL_PATH)

    if signal_df is None or signal_df.empty:
        raise FileNotFoundError(f"找不到：{SIGNAL_PATH}")

    signal_df = add_meta(signal_df)

    signal_num_cols = [
        "close",
        "pct_change",
        "rainbow_score",
        "main_force_score",
        "pred_label",
        "prob_up",
        "prob_down",
        "model_score",
    ]
    signal_df = to_numeric(signal_df, signal_num_cols)

    if "date" in signal_df.columns:
        signal_df["date"] = pd.to_datetime(signal_df["date"], errors="coerce")

    yaogu_df = safe_read(YAOGU_PATH)

    if yaogu_df is not None and not yaogu_df.empty:
        if "symbol" not in yaogu_df.columns:
            if "code" in yaogu_df.columns:
                yaogu_df["symbol"] = yaogu_df["code"]
            else:
                yaogu_df["symbol"] = ""

        yaogu_df["symbol"] = normalize_symbol(yaogu_df["symbol"])

        yaogu_num_cols = [
            "yaogu_v3_score",
            "yaogu_v3_raw_score",
            "yaogu_v3_risk",
            "prob_up",
            "prob_down",
            "model_score",
            "rainbow_score",
            "main_force_score",
            "pct_change",
            "turnover",
            "close",
            "pct_5d",
            "pct_10d",
            "pct_20d",
            "volume_ratio_v3",
        ]
        yaogu_df = to_numeric(yaogu_df, yaogu_num_cols)

        keep_cols = [
            "symbol",
            "yaogu_v3_score",
            "yaogu_v3_raw_score",
            "yaogu_v3_risk",
            "yaogu_v3_level",
            "yaogu_v3_action",
            "yaogu_v3_reason",
            "volume_ratio_v3",
            "pct_5d",
            "pct_10d",
            "pct_20d",
        ]
        keep_cols = [c for c in keep_cols if c in yaogu_df.columns]

        signal_df = signal_df.merge(
            yaogu_df[keep_cols].drop_duplicates("symbol"),
            on="symbol",
            how="left",
        )
    else:
        signal_df["yaogu_v3_score"] = np.nan
        signal_df["yaogu_v3_raw_score"] = np.nan
        signal_df["yaogu_v3_risk"] = np.nan
        signal_df["yaogu_v3_level"] = ""
        signal_df["yaogu_v3_action"] = ""
        signal_df["yaogu_v3_reason"] = ""

    clean_symbols = load_clean_symbols()
    signal_df["is_clean_pool"] = signal_df["symbol"].isin(clean_symbols) if clean_symbols else False

    return signal_df


def judge_market(df):
    avg_score = df["model_score"].mean()
    risk_count = int(
        (
            (df["pred_label"] == -1)
            | (df["prob_down"] >= 0.5)
            | (df["pct_change"] <= -3)
        ).sum()
    )

    strong_count = int(
        (
            (df["model_score"] >= 60)
            & (df["prob_up"] >= 0.6)
            & (df["rainbow_score"] >= 80)
            & (df["main_force_score"] >= 60)
            & (df["pred_label"] == 1)
        ).sum()
    )

    risk_ratio = risk_count / max(len(df), 1)

    if avg_score < 20 or risk_ratio >= 0.5:
        return {
            "market_status": "风险偏高",
            "mode": "防守",
            "position_advice": "0%~10%",
            "advice": "不建议主动开仓，只看不追。",
            "avg_score": avg_score,
            "risk_count": risk_count,
            "risk_ratio": risk_ratio,
            "strong_count": strong_count,
        }

    if avg_score >= 50 and strong_count >= 3 and risk_ratio < 0.3:
        return {
            "market_status": "市场偏强",
            "mode": "进攻",
            "position_advice": "30%~60%",
            "advice": "可以关注强候选，但必须等待盘中确认。",
            "avg_score": avg_score,
            "risk_count": risk_count,
            "risk_ratio": risk_ratio,
            "strong_count": strong_count,
        }

    return {
        "market_status": "空仓观察",
        "mode": "试错",
        "position_advice": "0%~20%",
        "advice": "只允许小仓位观察，优先低吸，不追高。",
        "avg_score": avg_score,
        "risk_count": risk_count,
        "risk_ratio": risk_ratio,
        "strong_count": strong_count,
    }


def safe_float(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def calc_buy_score(row):
    score = 0
    reasons = []

    model_score = safe_float(row.get("model_score"))
    prob_up = safe_float(row.get("prob_up"))
    prob_down = safe_float(row.get("prob_down"))
    rainbow = safe_float(row.get("rainbow_score"))
    main_force = safe_float(row.get("main_force_score"))
    pct_change = safe_float(row.get("pct_change"))
    yaogu = safe_float(row.get("yaogu_v3_score"))
    yaogu_risk = safe_float(row.get("yaogu_v3_risk"))
    pred_label = safe_float(row.get("pred_label"))
    is_clean_pool = bool(row.get("is_clean_pool", False))

    if is_clean_pool:
        score += 15
        reasons.append("优质股票池")

    if model_score >= 60:
        score += 20
        reasons.append("模型分较强")
    elif model_score >= 40:
        score += 10
        reasons.append("模型分尚可")

    if prob_up >= 0.6:
        score += 15
        reasons.append("上涨概率高")
    elif prob_up >= 0.35:
        score += 8
        reasons.append("上涨概率一般")

    if rainbow >= 80:
        score += 15
        reasons.append("趋势强")
    elif rainbow >= 60:
        score += 8
        reasons.append("趋势尚可")

    if main_force >= 80:
        score += 20
        reasons.append("主力行为强")
    elif main_force >= 60:
        score += 12
        reasons.append("主力行为活跃")
    elif main_force >= 40:
        score += 6
        reasons.append("主力初现")

    if yaogu >= 80:
        score += 20
        reasons.append("妖股v3强")
    elif yaogu >= 65:
        score += 12
        reasons.append("妖股v3较强")
    elif yaogu >= 50:
        score += 6
        reasons.append("妖股v3一般")

    if -2 <= pct_change <= 3:
        score += 10
        reasons.append("适合低吸区间")
    elif 3 < pct_change <= 6:
        score += 6
        reasons.append("偏强但不宜追高")
    elif pct_change > 8:
        score -= 20
        reasons.append("涨幅过高，防追高")

    if prob_down >= 0.5:
        score -= 25
        reasons.append("下跌概率偏高")

    if yaogu_risk >= 30:
        score -= 20
        reasons.append("妖股风险扣分高")

    if pred_label == -1:
        score -= 20
        reasons.append("模型标签偏空")

    return max(0, min(score, 100)), "；".join(reasons)


def classify_trade_type(row):
    pct_change = safe_float(row.get("pct_change"))
    yaogu = safe_float(row.get("yaogu_v3_score"))
    rainbow = safe_float(row.get("rainbow_score"))
    main_force = safe_float(row.get("main_force_score"))

    if pct_change <= 3 and rainbow >= 60:
        return "低吸观察"

    if 3 < pct_change <= 7 and main_force >= 60:
        return "突破观察"

    if yaogu >= 75 and main_force >= 70:
        return "强势观察"

    return "仅观察"


def make_trade_plan(df, market):
    rows = []

    for _, row in df.iterrows():
        buy_score, reasons = calc_buy_score(row)
        trade_type = classify_trade_type(row)

        pct_change = safe_float(row.get("pct_change"))
        prob_down = safe_float(row.get("prob_down"))
        yaogu_risk = safe_float(row.get("yaogu_v3_risk"))
        close = safe_float(row.get("close"), np.nan)

        avoid = False
        avoid_reason = ""

        if pct_change > 8:
            avoid = True
            avoid_reason = "当日涨幅过高，避免追高"
        elif prob_down >= 0.55:
            avoid = True
            avoid_reason = "下跌概率偏高"
        elif yaogu_risk >= 40:
            avoid = True
            avoid_reason = "妖股风险过高"

        if market["mode"] == "防守":
            action = "不买，仅观察"
        elif avoid:
            action = "回避"
        elif buy_score >= 80:
            action = "重点观察"
        elif buy_score >= 65:
            action = "小仓试错"
        elif buy_score >= 50:
            action = "只观察"
        else:
            action = "忽略"

        stop_loss = ""
        take_profit = ""

        if not pd.isna(close):
            stop_loss = f"{close * 0.97:.2f}"
            take_profit = f"{close * 1.05:.2f} / {close * 1.08:.2f}"

        if trade_type == "低吸观察":
            open_condition = "次日平开/小低开，且不跌破昨收，可观察低吸"
        elif trade_type == "突破观察":
            open_condition = "次日放量突破前高才考虑，不追高开"
        elif trade_type == "强势观察":
            open_condition = "只做强势确认，冲高不追，回落承接再看"
        else:
            open_condition = "无明确买点，观察为主"

        rows.append(
            {
                "symbol": row.get("symbol", ""),
                "stock_name": row.get("stock_name", ""),
                "display_name": row.get("display_name", ""),
                "industry": row.get("industry", ""),
                "board": row.get("board", ""),
                "themes": row.get("themes", ""),
                "date": row.get("date", ""),
                "close": row.get("close", np.nan),
                "pct_change": row.get("pct_change", np.nan),
                "model_score": row.get("model_score", np.nan),
                "rainbow_score": row.get("rainbow_score", np.nan),
                "main_force_score": row.get("main_force_score", np.nan),
                "prob_up": row.get("prob_up", np.nan),
                "prob_down": row.get("prob_down", np.nan),
                "yaogu_v3_score": row.get("yaogu_v3_score", np.nan),
                "yaogu_v3_risk": row.get("yaogu_v3_risk", np.nan),
                "yaogu_v3_level": row.get("yaogu_v3_level", ""),
                "trade_type": trade_type,
                "buy_score": buy_score,
                "action": action,
                "open_condition": open_condition,
                "stop_loss_price": stop_loss,
                "take_profit_price": take_profit,
                "reason": reasons,
                "avoid_reason": avoid_reason,
            }
        )

    plan = pd.DataFrame(rows)

    for col in ["buy_score", "yaogu_v3_score", "model_score"]:
        if col in plan.columns:
            plan[col] = pd.to_numeric(plan[col], errors="coerce").fillna(0)

    return plan.sort_values(["buy_score", "yaogu_v3_score", "model_score"], ascending=False)


def save_markdown(plan, market):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# 明日交易计划")
    lines.append("")
    lines.append(f"生成时间：{now}")
    lines.append("")
    lines.append("## 市场状态")
    lines.append("")
    lines.append(f"- 市场判断：{market['market_status']}")
    lines.append(f"- 模式：{market['mode']}")
    lines.append(f"- 建议仓位：{market['position_advice']}")
    lines.append(f"- 平均模型分：{market['avg_score']:.2f}")
    lines.append(f"- 强候选数量：{market['strong_count']}")
    lines.append(f"- 风险票数量：{market['risk_count']}")
    lines.append(f"- 风险比例：{market['risk_ratio']:.2%}")
    lines.append(f"- 建议：{market['advice']}")
    lines.append("")
    lines.append("## Top 关注")
    lines.append("")

    focus = plan[plan["action"].isin(["重点观察", "小仓试错"])].head(10)

    if focus.empty:
        lines.append("今天没有值得出手的股票，建议空仓观察。")
    else:
        for _, row in focus.iterrows():
            lines.append(f"### {row['display_name']}")
            lines.append(f"- 行业/板块：{row['industry']} / {row['board']}")
            lines.append(f"- 题材：{row['themes']}")
            lines.append(f"- 买入评分：{row['buy_score']}")
            lines.append(f"- 操作：{row['action']}")
            lines.append(f"- 类型：{row['trade_type']}")
            lines.append(f"- 开仓条件：{row['open_condition']}")
            lines.append(f"- 止损价：{row['stop_loss_price']}")
            lines.append(f"- 止盈参考：{row['take_profit_price']}")
            lines.append(f"- 理由：{row['reason']}")
            lines.append("")

    lines.append("## 风险提醒")
    lines.append("")
    lines.append("- 本计划只用于辅助筛选，不构成买卖建议。")
    lines.append("- 不满足开仓条件，不买。")
    lines.append("- 单票亏损约 -3% 必须执行止损。")
    lines.append("- 高开过多不追。")
    lines.append("- 市场风险偏高时，以空仓为主。")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs("output", exist_ok=True)

    print("读取数据...")
    df = prepare_data()

    market = judge_market(df)

    print("市场判断：", market["market_status"])
    print("建议仓位：", market["position_advice"])

    plan = make_trade_plan(df, market)

    plan.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    save_markdown(plan, market)

    print("\n==============================")
    print("自动交易计划生成完成")
    print(f"CSV：{OUTPUT_CSV}")
    print(f"Markdown：{OUTPUT_MD}")

    focus = plan[plan["action"].isin(["重点观察", "小仓试错"])].head(10)

    if focus.empty:
        print("\n今天没有值得出手的股票，建议空仓观察。")
    else:
        print("\nTop 关注：")
        print(focus[["display_name", "buy_score", "action", "trade_type", "open_condition"]])


if __name__ == "__main__":
    main()
