"""
review_predictions.py

次日/多日预测复盘工具。
作用：
1. 读取 history/daily_picks/YYYY-MM-DD_picks.csv
2. 用当前 output/latest_stock_signals.csv 或指定行情文件验证收益
3. 写入 history/prediction_review.csv

运行：
    python3 review_predictions.py 2026-05-22
    python3 review_predictions.py 2026-05-22 output/latest_stock_signals.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prediction_archive import normalize_symbol, safe_read, prepare_signal

HISTORY_DIR = Path("history")
DAILY_PICK_DIR = HISTORY_DIR / "daily_picks"
REVIEW_PATH = HISTORY_DIR / "prediction_review.csv"
SIGNAL_PATH = Path("output/latest_stock_signals.csv")


def _get_price_col(df: pd.DataFrame) -> str:
    for c in ["close", "收盘价", "current_close", "price"]:
        if c in df.columns:
            return c
    raise ValueError(f"行情文件缺少 close/收盘价 列，当前列名：{list(df.columns)}")


def load_picks(pick_date: str) -> pd.DataFrame:
    path = DAILY_PICK_DIR / f"{pick_date}_picks.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到推荐存档：{path}。请先在 app 里点击“存档今日推荐”。")
    df = safe_read(path)
    if df is None or df.empty:
        raise ValueError(f"推荐存档为空：{path}")
    return normalize_symbol(df)


def load_current_signal(current_path: str | Path = SIGNAL_PATH) -> pd.DataFrame:
    df = safe_read(current_path)
    if df is None or df.empty:
        raise FileNotFoundError(f"找不到当前行情/信号文件：{current_path}")
    return prepare_signal(df)


def review_pick_date(pick_date: str, current_path: str | Path = SIGNAL_PATH) -> pd.DataFrame:
    picks = load_picks(pick_date)
    current = load_current_signal(current_path)

    price_col_pick = _get_price_col(picks)
    price_col_now = _get_price_col(current)

    picks["entry_close"] = pd.to_numeric(picks[price_col_pick], errors="coerce")
    current["review_close"] = pd.to_numeric(current[price_col_now], errors="coerce")

    now_cols = ["symbol", "review_close"]
    for c in ["date", "pct_change", "rainbow_score", "main_force_score", "model_score", "prob_up", "prob_down", "signal_status"]:
        if c in current.columns and c not in now_cols:
            now_cols.append(c)

    merged = picks.merge(current[now_cols], on="symbol", how="left", suffixes=("", "_review"))

    merged["review_date"] = pd.to_datetime(merged.get("date_review", merged.get("date", pd.Timestamp.today())), errors="coerce")
    if "review_date" in merged.columns:
        merged["review_date"] = merged["review_date"].dt.strftime("%Y-%m-%d")

    merged["next_return_pct"] = (merged["review_close"] / merged["entry_close"] - 1) * 100
    merged["hit"] = np.where(merged["next_return_pct"] > 0, 1, 0)
    merged["big_win"] = np.where(merged["next_return_pct"] >= 5, 1, 0)
    merged["big_loss"] = np.where(merged["next_return_pct"] <= -5, 1, 0)

    def result_tag(x):
        if pd.isna(x):
            return "未匹配"
        if x >= 7:
            return "大肉"
        if x >= 3:
            return "有效"
        if x > 0:
            return "小赚"
        if x > -3:
            return "震荡"
        return "失败"

    merged["review_result"] = merged["next_return_pct"].apply(result_tag)
    merged["pick_date"] = pick_date

    # 去重更新：同一个 pick_date + symbol 只保留最新
    old = safe_read(REVIEW_PATH)
    if old is not None and not old.empty:
        old = normalize_symbol(old)
        old = old[~((old["pick_date"].astype(str) == str(pick_date)) & (old["symbol"].isin(merged["symbol"])))].copy()
        out = pd.concat([old, merged], ignore_index=True)
    else:
        out = merged.copy()

    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")
    return merged


def review_summary(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["next_return_pct"]).copy()
    if valid.empty:
        return {"count": 0, "hit_rate": 0, "avg_return": 0, "big_win": 0, "big_loss": 0}
    return {
        "count": int(len(valid)),
        "hit_rate": float(valid["hit"].mean() * 100),
        "avg_return": float(valid["next_return_pct"].mean()),
        "big_win": int(valid["big_win"].sum()),
        "big_loss": int(valid["big_loss"].sum()),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python3 review_predictions.py YYYY-MM-DD [current_signal_csv]")
        raise SystemExit(1)
    pick_date = sys.argv[1]
    current = sys.argv[2] if len(sys.argv) >= 3 else SIGNAL_PATH
    reviewed = review_pick_date(pick_date, current)
    s = review_summary(reviewed)
    print(f"已复盘 {pick_date}: 样本{s['count']}，胜率{s['hit_rate']:.1f}%，平均收益{s['avg_return']:.2f}%")
    print(f"大赚: {s['big_win']}，大亏: {s['big_loss']}")
