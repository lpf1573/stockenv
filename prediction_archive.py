"""
prediction_archive.py

每日预测存档工具。
作用：
1. 把 output/latest_stock_signals.csv 备份到 history/daily_signals/YYYY-MM-DD_signals.csv
2. 从信号里挑出 TopN 推荐，保存到 history/daily_picks/YYYY-MM-DD_picks.csv
3. 给 app 和复盘脚本复用

运行：
    python3 prediction_archive.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SIGNAL_PATH = Path("output/latest_stock_signals.csv")
YAOGU_PATH = Path("output/yaogu_model_v3_signals.csv")
CLEAN_POOL_PATH = Path("output/clean_stock_pool.csv")
META_PATH = Path("output/stock_meta_map.csv")

HISTORY_DIR = Path("history")
DAILY_SIGNAL_DIR = HISTORY_DIR / "daily_signals"
DAILY_PICK_DIR = HISTORY_DIR / "daily_picks"

for p in [HISTORY_DIR, DAILY_SIGNAL_DIR, DAILY_PICK_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def safe_read(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str)


def normalize_symbol(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "symbol" not in df.columns:
        if "code" in df.columns:
            df["symbol"] = df["code"]
        elif "股票代码" in df.columns:
            df["symbol"] = df["股票代码"]
        else:
            raise ValueError(f"缺少 symbol/code/股票代码 列，当前列名：{list(df.columns)}")

    df["symbol"] = (
        df["symbol"]
        .astype(str)
        .str.replace("sh.", "", regex=False)
        .str.replace("sz.", "", regex=False)
        .str.replace(".0", "", regex=False)
        .str.extract(r"(\d+)", expand=False)
        .fillna("")
        .str.zfill(6)
    )
    return df


def to_number(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def merge_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_symbol(df)
    meta = safe_read(META_PATH)
    if meta is None or meta.empty:
        for col, default in [("stock_name", ""), ("industry", "未知"), ("board", "未知"), ("themes", "其他")]:
            if col not in df.columns:
                df[col] = default
            df[col] = df[col].fillna(default)
        df["display_name"] = df["symbol"] + " " + df.get("stock_name", "").astype(str)
        return df

    meta = normalize_symbol(meta)
    keep = [c for c in ["symbol", "stock_name", "industry", "board", "themes"] if c in meta.columns]
    meta = meta[keep].drop_duplicates("symbol")
    df = df.merge(meta, on="symbol", how="left", suffixes=("", "_meta"))

    for col, default in [("stock_name", ""), ("industry", "未知"), ("board", "未知"), ("themes", "其他")]:
        meta_col = f"{col}_meta"
        if meta_col in df.columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[meta_col])
            else:
                df[col] = df[meta_col]
            df = df.drop(columns=[meta_col])
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    df["display_name"] = df["symbol"] + " " + df["stock_name"].astype(str)
    return df


def load_clean_symbols() -> set[str]:
    clean = safe_read(CLEAN_POOL_PATH)
    if clean is None or clean.empty:
        return set()
    clean = normalize_symbol(clean)
    return set(clean["symbol"].tolist())


def prepare_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = merge_meta(df)
    numeric_cols = [
        "close", "pct_change", "rainbow_score", "main_force_score", "pred_label",
        "prob_up", "prob_down", "model_score", "turnover", "volume_ratio_v3",
        "pct_5d", "pct_10d", "pct_20d", "yaogu_v3_score", "yaogu_v3_raw_score", "yaogu_v3_risk",
    ]
    df = to_number(df, numeric_cols)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.Timestamp.today().normalize()

    clean_symbols = load_clean_symbols()
    df["is_clean_pool"] = df["symbol"].isin(clean_symbols) if clean_symbols else False

    if "signal_status" not in df.columns:
        def status(row):
            if (
                row.get("model_score", 0) >= 60
                and row.get("prob_up", 0) >= 0.6
                and row.get("rainbow_score", 0) >= 75
                and row.get("main_force_score", 0) >= 55
                and row.get("pred_label", 0) == 1
            ):
                return "强买观察"
            if row.get("pred_label", 0) == -1 or row.get("prob_down", 0) >= 0.5 or row.get("pct_change", 0) <= -3:
                return "风险"
            if row.get("model_score", 0) >= 30 and row.get("rainbow_score", 0) >= 55:
                return "观察"
            return "弱"
        df["signal_status"] = df.apply(status, axis=1)

    score_col = "yaogu_v3_score" if "yaogu_v3_score" in df.columns else "model_score"
    if score_col in df.columns:
        df = df.sort_values(score_col, ascending=False)
    return df.reset_index(drop=True)


def load_best_signal() -> pd.DataFrame:
    """优先读取妖股 v3，没有则读取 latest_stock_signals。"""
    yaogu = safe_read(YAOGU_PATH)
    sig = safe_read(SIGNAL_PATH)

    if yaogu is not None and not yaogu.empty:
        df = prepare_signal(yaogu)
    elif sig is not None and not sig.empty:
        df = prepare_signal(sig)
    else:
        raise FileNotFoundError("找不到 output/latest_stock_signals.csv 或 output/yaogu_model_v3_signals.csv")
    return df


def pick_candidates(df: pd.DataFrame, top_n: int = 20, only_clean: bool = False) -> pd.DataFrame:
    df = df.copy()
    if only_clean and "is_clean_pool" in df.columns:
        df = df[df["is_clean_pool"] == True]

    # 综合分：优先妖股分，其次模型分，并叠加趋势/主力。
    for c in ["yaogu_v3_score", "model_score", "rainbow_score", "main_force_score", "prob_up"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["final_pick_score"] = (
        df["yaogu_v3_score"].where(df["yaogu_v3_score"] > 0, df["model_score"])
        + df["rainbow_score"] * 0.15
        + df["main_force_score"] * 0.15
        + df["prob_up"] * 20
        + np.where(df.get("is_clean_pool", False), 5, 0)
    )

    keep_cols = [
        "symbol", "display_name", "stock_name", "industry", "board", "themes", "date", "close", "pct_change",
        "model_score", "prob_up", "prob_down", "rainbow_score", "main_force_score", "pred_label", "signal_status",
        "yaogu_v3_score", "yaogu_v3_raw_score", "yaogu_v3_risk", "yaogu_v3_level", "yaogu_v3_action", "yaogu_v3_reason",
        "is_clean_pool", "final_pick_score",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df.sort_values("final_pick_score", ascending=False).head(top_n)[keep_cols].reset_index(drop=True)


def infer_trade_date(df: pd.DataFrame) -> str:
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce").max()
        if pd.notna(d):
            return d.strftime("%Y-%m-%d")
    return datetime.now().strftime("%Y-%m-%d")


def archive_today(top_n: int = 20, only_clean: bool = False) -> tuple[Path, Path, pd.DataFrame]:
    df = load_best_signal()
    trade_date = infer_trade_date(df)

    signal_path = DAILY_SIGNAL_DIR / f"{trade_date}_signals.csv"
    pick_path = DAILY_PICK_DIR / f"{trade_date}_picks.csv"

    df.to_csv(signal_path, index=False, encoding="utf-8-sig")
    picks = pick_candidates(df, top_n=top_n, only_clean=only_clean)
    picks.to_csv(pick_path, index=False, encoding="utf-8-sig")
    return signal_path, pick_path, picks


if __name__ == "__main__":
    signal_path, pick_path, picks = archive_today(top_n=20, only_clean=False)
    print(f"已保存全量信号: {signal_path}")
    print(f"已保存推荐清单: {pick_path}")
    print(picks[["symbol", "display_name", "final_pick_score"]].head(10).to_string(index=False))
