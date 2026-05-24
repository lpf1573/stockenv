"""
sync_latest_signals_from_yaogu.py

用途：
当 output/latest_stock_signals.csv 还是旧日期，
但 output/yaogu_model_v3_signals.csv 已经是新日期时，
用妖股V3文件反向修复 latest_stock_signals.csv 的日期和基础信号。

运行：
    python3 sync_latest_signals_from_yaogu.py

它会：
1. 备份旧 latest_stock_signals.csv
2. 如果 yaogu_model_v3_signals.csv 日期更新，就用它生成新的 latest_stock_signals.csv
3. 保留 symbol/date/close/pct_change/趋势分/主力分/模型分/概率等字段
"""

from pathlib import Path
from datetime import datetime
import pandas as pd


OUTPUT_DIR = Path("output")
LATEST_PATH = OUTPUT_DIR / "latest_stock_signals.csv"
YAOGU_PATH = OUTPUT_DIR / "yaogu_model_v3_signals.csv"


def safe_read(path):
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc, dtype=str)
        except Exception:
            pass
    return pd.read_csv(path, dtype=str)


def normalize_symbol(x):
    x = str(x)
    x = x.replace("sh.", "").replace("sz.", "")
    x = x.replace(".SH", "").replace(".SZ", "")
    x = x.replace(".0", "")
    return x.zfill(6)[-6:]


def main():
    if not YAOGU_PATH.exists():
        print(f"找不到 {YAOGU_PATH}")
        return

    yaogu = safe_read(YAOGU_PATH)
    if yaogu.empty or "date" not in yaogu.columns:
        print("yaogu_model_v3_signals.csv 为空或缺少 date 列")
        return

    yaogu["date"] = pd.to_datetime(yaogu["date"], errors="coerce")
    yaogu_max_date = yaogu["date"].max()

    latest_date = None
    latest = None

    if LATEST_PATH.exists():
        latest = safe_read(LATEST_PATH)
        if not latest.empty and "date" in latest.columns:
            latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
            latest_date = latest["date"].max()

    print(f"latest 日期: {latest_date}")
    print(f"yaogu 日期:  {yaogu_max_date}")

    if latest_date is not None and latest_date >= yaogu_max_date:
        print("latest_stock_signals.csv 已经不旧，不需要修复。")
        return

    # 只取妖股文件最新交易日
    new_latest = yaogu[yaogu["date"] == yaogu_max_date].copy()

    # 规范 symbol
    if "symbol" not in new_latest.columns:
        if "code" in new_latest.columns:
            new_latest["symbol"] = new_latest["code"]
        elif "股票代码" in new_latest.columns:
            new_latest["symbol"] = new_latest["股票代码"]

    new_latest["symbol"] = new_latest["symbol"].apply(normalize_symbol)
    new_latest["date"] = new_latest["date"].dt.strftime("%Y-%m-%d")

    # 如果没有 model_score，但有 yaogu_v3_score，就临时用妖股分兜底
    if "model_score" not in new_latest.columns and "yaogu_v3_score" in new_latest.columns:
        new_latest["model_score"] = new_latest["yaogu_v3_score"]

    # 如果缺概率列，兜底生成，避免 app 显示报错
    if "prob_up" not in new_latest.columns:
        if "model_score" in new_latest.columns:
            score = pd.to_numeric(new_latest["model_score"], errors="coerce").fillna(0).clip(0, 100)
            new_latest["prob_up"] = (score / 100).round(4)
        else:
            new_latest["prob_up"] = 0

    if "prob_down" not in new_latest.columns:
        new_latest["prob_down"] = (1 - pd.to_numeric(new_latest["prob_up"], errors="coerce").fillna(0)).round(4)

    if "pred_label" not in new_latest.columns:
        new_latest["pred_label"] = (pd.to_numeric(new_latest["prob_up"], errors="coerce").fillna(0) >= 0.6).astype(int)

    # 备份旧 latest
    if LATEST_PATH.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = OUTPUT_DIR / f"latest_stock_signals_backup_{stamp}.csv"
        LATEST_PATH.rename(backup)
        print(f"已备份旧 latest: {backup}")

    new_latest.to_csv(LATEST_PATH, index=False, encoding="utf-8-sig")
    print(f"已生成新的 {LATEST_PATH}")
    print(f"新 latest 日期: {new_latest['date'].max()}")
    print(f"股票数量: {len(new_latest)}")


if __name__ == "__main__":
    main()
