"""
us_daily_report.py

美股每日交易日报自动生成器

功能：
1. 自动抓取美股主要指数 ETF / 科技股 / 板块 ETF
2. 统计日涨跌幅、成交量、强弱排序
3. 自动生成 Markdown 日报
4. 输出到 reports/us_daily_report_YYYY-MM-DD.md

运行：
    pip install yfinance pandas
    python3 us_daily_report.py

说明：
    - 使用 yfinance，需要网络
    - 默认分析最近 7 个交易日
"""

from pathlib import Path
from datetime import datetime

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


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
}

MEGA_CAPS = {
    "AAPL": "苹果",
    "MSFT": "微软",
    "NVDA": "英伟达",
    "GOOGL": "谷歌",
    "AMZN": "亚马逊",
    "META": "Meta",
    "TSLA": "特斯拉",
    "AVGO": "博通",
    "AMD": "AMD",
    "NFLX": "奈飞",
}


def download_prices(tickers, period="10d"):
    if yf is None:
        raise ImportError("缺少 yfinance，请先运行：pip install yfinance")

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


def add_names(df, name_map):
    df = df.copy()
    df["name"] = df["ticker"].map(name_map).fillna(df["ticker"])
    return df


def market_stage(index_df):
    if index_df.empty:
        return "未知", "数据不足"

    qqq = index_df[index_df["ticker"] == "QQQ"]
    spy = index_df[index_df["ticker"] == "SPY"]
    iwm = index_df[index_df["ticker"] == "IWM"]

    qqq_pct = float(qqq["pct_change"].iloc[0]) if not qqq.empty else 0
    spy_pct = float(spy["pct_change"].iloc[0]) if not spy.empty else 0
    iwm_pct = float(iwm["pct_change"].iloc[0]) if not iwm.empty else 0

    avg = (qqq_pct + spy_pct + iwm_pct) / 3

    if qqq_pct > 1 and spy_pct > 0.7:
        return "风险偏好上升", "科技成长方向占优，可关注 AI/半导体/大型科技。"
    if avg > 0.3:
        return "温和修复", "指数整体偏强，但需要观察成交量和板块扩散。"
    if qqq_pct < -1 or spy_pct < -0.8:
        return "风险释放", "控制仓位，避免追高高估值科技股。"
    return "震荡观察", "市场方向不明确，优先等待指数和板块共振。"


def format_table(df, cols):
    if df.empty:
        return "暂无数据\n"

    show = df.copy()
    for c in ["close", "pct_change", "volume"]:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce")

    lines = []
    headers = cols
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

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


def generate_report():
    all_index = download_prices(INDEX_ETFS.keys())
    all_sector = download_prices(SECTOR_ETFS.keys())
    all_mega = download_prices(MEGA_CAPS.keys())

    index_df = add_names(all_index, INDEX_ETFS).sort_values("pct_change", ascending=False)
    sector_df = add_names(all_sector, SECTOR_ETFS).sort_values("pct_change", ascending=False)
    mega_df = add_names(all_mega, MEGA_CAPS).sort_values("pct_change", ascending=False)

    date_text = index_df["date"].iloc[0] if not index_df.empty else datetime.now().strftime("%Y-%m-%d")

    stage, advice = market_stage(index_df)

    strongest_sector = sector_df.iloc[0]["name"] if not sector_df.empty else "未知"
    weakest_sector = sector_df.iloc[-1]["name"] if not sector_df.empty else "未知"

    strongest_stock = mega_df.iloc[0]["name"] if not mega_df.empty else "未知"
    weakest_stock = mega_df.iloc[-1]["name"] if not mega_df.empty else "未知"

    lines = []
    lines.append(f"# 美股每日交易日报 - {date_text}")
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
    lines.append(format_table(index_df, ["ticker", "name", "close", "pct_change", "volume"]))
    lines.append("")
    lines.append("## 三、板块 ETF 强弱")
    lines.append(format_table(sector_df, ["ticker", "name", "close", "pct_change", "volume"]))
    lines.append("")
    lines.append("## 四、大型科技股表现")
    lines.append(format_table(mega_df, ["ticker", "name", "close", "pct_change", "volume"]))
    lines.append("")
    lines.append("## 五、明日观察")
    lines.append("- 观察 QQQ 是否继续强于 SPY。")
    lines.append("- 如果科技板块继续领涨，AI/半导体仍是重点方向。")
    lines.append("- 如果小盘 IWM 明显走弱，说明风险偏好不足。")
    lines.append("- 若防御板块走强，说明资金可能转向避险。")
    lines.append("")
    lines.append("> 仅用于研究和复盘，不构成投资建议。")

    report = "\n".join(lines)

    path = REPORT_DIR / f"us_daily_report_{date_text}.md"
    path.write_text(report, encoding="utf-8")

    print(f"已生成：{path}")
    return path


if __name__ == "__main__":
    generate_report()
