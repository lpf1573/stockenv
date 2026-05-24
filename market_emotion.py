"""
market_emotion.py

市场情绪与主线识别。
输入：output/latest_stock_signals.csv 或妖股/全市场信号文件
输出：output/market_emotion.csv, output/sector_strength.csv

运行：
    python3 market_emotion.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prediction_archive import safe_read, prepare_signal

SIGNAL_PATH = Path("output/latest_stock_signals.csv")
YAOGU_PATH = Path("output/yaogu_model_v3_signals.csv")
EMOTION_PATH = Path("output/market_emotion.csv")
SECTOR_PATH = Path("output/sector_strength.csv")


def load_signal() -> pd.DataFrame:
    df = safe_read(SIGNAL_PATH)
    if df is None or df.empty:
        df = safe_read(YAOGU_PATH)
    if df is None or df.empty:
        raise FileNotFoundError("找不到 output/latest_stock_signals.csv 或 output/yaogu_model_v3_signals.csv")
    return prepare_signal(df)


def calc_sector_strength(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["pct_change", "model_score", "rainbow_score", "main_force_score", "prob_up", "yaogu_v3_score"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "industry" not in df.columns:
        df["industry"] = "未知"

    g = df.groupby("industry", dropna=False).agg(
        stock_count=("symbol", "count"),
        avg_pct=("pct_change", "mean"),
        strong_count=("pct_change", lambda x: int((x >= 5).sum())),
        limit_like_count=("pct_change", lambda x: int((x >= 9.5).sum())),
        avg_model=("model_score", "mean"),
        avg_trend=("rainbow_score", "mean"),
        avg_force=("main_force_score", "mean"),
        avg_prob_up=("prob_up", "mean"),
        avg_yaogu=("yaogu_v3_score", "mean"),
    ).reset_index()

    g["sector_strength"] = (
        g["avg_pct"] * 5
        + g["strong_count"] * 3
        + g["limit_like_count"] * 8
        + g["avg_model"] * 0.25
        + g["avg_trend"] * 0.2
        + g["avg_force"] * 0.2
        + g["avg_prob_up"] * 20
        + g["avg_yaogu"] * 0.2
    )
    g = g.sort_values("sector_strength", ascending=False).reset_index(drop=True)
    return g


def calc_market_emotion(df: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["pct_change", "model_score", "rainbow_score", "main_force_score", "prob_up", "prob_down"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    total = max(len(df), 1)
    up_ratio = float((df["pct_change"] > 0).mean())
    strong_ratio = float((df["pct_change"] >= 5).mean())
    risk_ratio = float(((df["pct_change"] <= -3) | (df["prob_down"] >= 0.5)).mean())
    avg_model = float(df["model_score"].mean())
    avg_trend = float(df["rainbow_score"].mean())
    avg_force = float(df["main_force_score"].mean())

    top_sector = sector.iloc[0]["industry"] if not sector.empty else "未知"
    top_sector_strength = float(sector.iloc[0]["sector_strength"]) if not sector.empty else 0

    emotion_score = (
        up_ratio * 30
        + strong_ratio * 40
        + avg_model * 0.15
        + avg_trend * 0.15
        + avg_force * 0.15
        - risk_ratio * 25
        + min(top_sector_strength / 10, 15)
    )
    emotion_score = max(0, min(100, emotion_score))

    if emotion_score >= 70:
        phase = "主升/进攻"
        advice = "可围绕主线小仓试错，但拒绝无脑追高。"
    elif emotion_score >= 55:
        phase = "修复/轮动"
        advice = "适合做强板块前排，节奏比胆子重要。"
    elif emotion_score >= 40:
        phase = "震荡/观察"
        advice = "降低仓位，等主线更清楚。"
    else:
        phase = "退潮/防守"
        advice = "少动手，现金也是仓位。"

    out = pd.DataFrame([{
        "date": pd.to_datetime(df.get("date", pd.Series([pd.Timestamp.today()])).max(), errors="coerce"),
        "stock_count": total,
        "up_ratio": up_ratio,
        "strong_ratio": strong_ratio,
        "risk_ratio": risk_ratio,
        "avg_model": avg_model,
        "avg_trend": avg_trend,
        "avg_force": avg_force,
        "top_sector": top_sector,
        "top_sector_strength": top_sector_strength,
        "emotion_score": emotion_score,
        "phase": phase,
        "advice": advice,
    }])
    return out


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_signal()
    sector = calc_sector_strength(df)
    emotion = calc_market_emotion(df, sector)
    EMOTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    sector.to_csv(SECTOR_PATH, index=False, encoding="utf-8-sig")
    emotion.to_csv(EMOTION_PATH, index=False, encoding="utf-8-sig")
    print(f"已生成：{SECTOR_PATH}")
    print(f"已生成：{EMOTION_PATH}")
    print(emotion[["phase", "emotion_score", "top_sector", "advice"]].to_string(index=False))
    return emotion, sector


if __name__ == "__main__":
    main()
