
"""
us_daily_report_module.py

美股每日复盘模块，可被 Streamlit app.py 一键调用。

依赖：
    pip install yfinance pandas

用法：
    from us_daily_report_module import generate_us_daily_report
"""

from pathlib import Path
from datetime import datetime

import pandas as pd


REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


INDEX_ETFS = {
    "SPY": "标普500",
    "QQQ": "纳斯达克100",
    "DIA": "道琼斯",
    "IWM": "罗素2000",
}

SECTOR_ETFS = {
    "XLK": "科技",
    "XLF": "金融",
    "XLV": "医疗",
    "XLE": "能源",
    "XLY": "可选消费",
    "XLP": "必选消费",
    "XLI": "工业",
    "XLC": "通信",
    "XLU": "公用事业",
    "XLB": "材料",
    "XLRE": "房地产",
    "SMH": "半导体",
    "ARKK": "高成长",
}

MEGA_CAPS = {
    "AAPL": "苹果",
    "MSFT": "微软",
    "NVDA": "英伟达",
    "GOOGL": "谷歌A",
    "AMZN": "亚马逊",
    "META": "Meta",
    "TSLA": "特斯拉",
    "AVGO": "博通",
    "AMD": "AMD",
    "NFLX": "奈飞",
    "PLTR": "Palantir",
    "SMCI": "超微电脑",
}


def _download_prices(tickers, period="10d"):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("缺少 yfinance，请先运行：pip install yfinance") from exc

    data = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    rows = []

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = data.copy()
            else:
                df = data[ticker].copy()

            df = df.dropna()
            if len(df) < 2:
                continue

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            close = float(latest["Close"])
            prev_close = float(prev["Close"])
            pct = (close / prev_close - 1) * 100
            volume = float(latest.get("Volume", 0))

            rows.append({
                "ticker": ticker,
                "date": df.index[-1].strftime("%Y-%m-%d"),
                "close": close,
                "pct_change": pct,
                "volume": volume,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def _add_names(df, name_map):
    df = df.copy()
    if df.empty:
        return df
    df["name"] = df["ticker"].map(name_map).fillna(df["ticker"])
    return df


def _market_stage(index_df, sector_df):
    if index_df.empty:
        return "未知", "数据不足，无法判断。"

    def pct(ticker):
        row = index_df[index_df["ticker"] == ticker]
        if row.empty:
            return 0.0
        return float(row["pct_change"].iloc[0])

    qqq = pct("QQQ")
    spy = pct("SPY")
    iwm = pct("IWM")

    sector_strong_count = 0
    if not sector_df.empty:
        sector_strong_count = int((sector_df["pct_change"] > 0).sum())

    avg = (qqq + spy + iwm) / 3

    if qqq > 1 and spy > 0.7:
        return "风险偏好上升", "科技成长方向占优，重点观察 AI、半导体、大型科技。"
    if avg > 0.3 and sector_strong_count >= 6:
        return "全面修复", "指数和板块扩散同步，市场承接较好。"
    if avg > 0.2:
        return "温和修复", "指数偏强，但要观察是否扩散到小盘和周期。"
    if qqq < -1 or spy < -0.8:
        return "风险释放", "控制仓位，避免追高高估值科技股。"
    return "震荡观察", "方向不够明确，等待指数和板块共振。"


def _format_table(df, cols):
    if df.empty:
        return "暂无数据\n"

    show = df.copy()
    for c in ["close", "pct_change", "volume"]:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce")

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, r in show.iterrows():
        row = []
        for c in cols:
            if c == "pct_change":
                row.append(f"{r[c]:.2f}%")
            elif c == "close":
                row.append(f"{r[c]:.2f}")
            elif c == "volume":
                row.append(f"{r[c] / 1e6:.1f}M")
            else:
                row.append(str(r.get(c, "")))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def generate_us_daily_report():
    index_df = _add_names(_download_prices(INDEX_ETFS.keys()), INDEX_ETFS)
    sector_df = _add_names(_download_prices(SECTOR_ETFS.keys()), SECTOR_ETFS)
    mega_df = _add_names(_download_prices(MEGA_CAPS.keys()), MEGA_CAPS)

    if not index_df.empty:
        index_df = index_df.sort_values("pct_change", ascending=False)
    if not sector_df.empty:
        sector_df = sector_df.sort_values("pct_change", ascending=False)
    if not mega_df.empty:
        mega_df = mega_df.sort_values("pct_change", ascending=False)

    date_text = index_df["date"].iloc[0] if not index_df.empty else datetime.now().strftime("%Y-%m-%d")
    stage, advice = _market_stage(index_df, sector_df)

    strongest_sector = sector_df.iloc[0]["name"] if not sector_df.empty else "未知"
    weakest_sector = sector_df.iloc[-1]["name"] if not sector_df.empty else "未知"
    strongest_stock = mega_df.iloc[0]["name"] if not mega_df.empty else "未知"
    weakest_stock = mega_df.iloc[-1]["name"] if not mega_df.empty else "未知"

    lines = []
    lines.append(f"# 美股每日复盘 - {date_text}")
    lines.append("")
    lines.append("## 一、市场状态")
    lines.append(f"- 市场阶段：{stage}")
    lines.append(f"- 简要判断：{advice}")
    lines.append(f"- 最强板块：{strongest_sector}")
    lines.append(f"- 最弱板块：{weakest_sector}")
    lines.append(f"- 最强大型科技股：{strongest_stock}")
    lines.append(f"- 最弱大型科技股：{weakest_stock}")
    lines.append("")
    lines.append("## 二、主要指数")
    lines.append(_format_table(index_df, ["ticker", "name", "close", "pct_change", "volume"]))
    lines.append("")
    lines.append("## 三、板块 ETF 强弱")
    lines.append(_format_table(sector_df, ["ticker", "name", "close", "pct_change", "volume"]))
    lines.append("")
    lines.append("## 四、大型科技股表现")
    lines.append(_format_table(mega_df, ["ticker", "name", "close", "pct_change", "volume"]))
    lines.append("")
    lines.append("## 五、明日观察")
    lines.append("- 观察 QQQ 是否继续强于 SPY。")
    lines.append("- 如果 SMH、XLK 继续领涨，AI/半导体仍是主线。")
    lines.append("- 如果 IWM 明显走弱，说明风险偏好不足。")
    lines.append("- 如果 XLU、XLP 走强，说明资金可能转向防御。")
    lines.append("")
    lines.append("> 仅用于研究和复盘，不构成投资建议。")

    report_text = "\n".join(lines)

    md_path = REPORT_DIR / f"us_daily_report_{date_text}.md"
    md_path.write_text(report_text, encoding="utf-8")

    index_df.to_csv(REPORT_DIR / f"us_index_{date_text}.csv", index=False, encoding="utf-8-sig")
    sector_df.to_csv(REPORT_DIR / f"us_sector_{date_text}.csv", index=False, encoding="utf-8-sig")
    mega_df.to_csv(REPORT_DIR / f"us_mega_caps_{date_text}.csv", index=False, encoding="utf-8-sig")

    return md_path, report_text, index_df, sector_df, mega_df


if __name__ == "__main__":
    path, *_ = generate_us_daily_report()
    print(f"已生成：{path}")
