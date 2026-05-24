"""
update_market_dataset.py

只更新全部688科创板股票，带防卡死机制。

运行：
    python3 update_market_dataset.py

依赖：
    pip install baostock pandas numpy
"""

import os
import time
import signal
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import baostock as bs
except ImportError:
    bs = None


# =========================
# 配置区
# =========================

OUTPUT_DIR = "output"

DATASET_PATH = "output/a_share_labeled_dataset_with_auto_manual_label.csv"

# 单独使用688股票池，避免旧的300股票池污染
STOCK_POOL_PATH = "output/stock_pool_688.csv"

FAILED_PATH = "output/update_failed_symbols.csv"

FULL_START_DATE = "2024-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# 只跑688
MARKET_MODE = "star688"

# None = 全部688
MAX_STOCKS = None

SLEEP_BETWEEN_STOCKS = 0

# 单只股票 BaoStock 请求超时时间
REQUEST_TIMEOUT_SECONDS = 20

FUTURE_DAYS = 5
UP_THRESHOLD = 0.03
DOWN_THRESHOLD = -0.03

MIN_HISTORY_ROWS = 180

# 最近20日平均成交额低于2亿就过滤
MIN_AMOUNT_MEAN = 200_000_000


# =========================
# 基础工具
# =========================

def timeout_handler(signum, frame):
    raise TimeoutError("BaoStock 请求超时")


def ensure_baostock():
    if bs is None:
        raise ImportError("请先安装 baostock：pip install baostock")


def login_baostock():
    ensure_baostock()
    print("登录 BaoStock...")
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败：{lg.error_msg}")


def logout_baostock():
    try:
        bs.logout()
        print("BaoStock 已登出。")
    except Exception:
        pass


def to_bs_code(symbol: str) -> str:
    symbol = str(symbol).replace(".0", "").zfill(6)
    if symbol.startswith(("600", "601", "603", "605", "688")):
        return "sh." + symbol
    return "sz." + symbol


def normalize_symbol(code: str) -> str:
    return str(code).replace("sh.", "").replace("sz.", "").replace(".0", "").zfill(6)


def next_day(date_value) -> str:
    dt = pd.to_datetime(date_value)
    return (dt + timedelta(days=1)).strftime("%Y-%m-%d")


# =========================
# 股票池：只取全部688
# =========================

def get_stock_pool() -> list[str]:
    if os.path.exists(STOCK_POOL_PATH):
        print(f"读取本地688股票池：{STOCK_POOL_PATH}")
        pool_df = pd.read_csv(STOCK_POOL_PATH, dtype={"code": str})

        codes = pool_df["code"].astype(str).tolist()
        codes = [
            c if c.startswith(("sh.", "sz.")) else to_bs_code(c)
            for c in codes
        ]

        if MAX_STOCKS is not None:
            codes = codes[:MAX_STOCKS]

        print(f"688股票池数量：{len(codes)}")
        return codes

    print("从 BaoStock 获取全部688股票池...")
    rs = bs.query_stock_basic()

    if rs.error_code != "0":
        raise RuntimeError(f"获取股票池失败：{rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    stock_df = pd.DataFrame(rows, columns=rs.fields)

    if stock_df.empty:
        raise ValueError("股票池为空")

    if "type" in stock_df.columns:
        stock_df = stock_df[stock_df["type"] == "1"]

    if "status" in stock_df.columns:
        stock_df = stock_df[stock_df["status"] == "1"]

    stock_df["code"] = stock_df["code"].astype(str)

    # 核心：只保留 sh.688
    stock_df = stock_df[
        stock_df["code"].str.startswith("sh.688")
    ].copy()

    stock_df = stock_df.sort_values("code").reset_index(drop=True)

    if MAX_STOCKS is not None:
        stock_df = stock_df.head(MAX_STOCKS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stock_df.to_csv(STOCK_POOL_PATH, index=False, encoding="utf-8-sig")

    symbols = stock_df["code"].tolist()

    print(f"688股票池数量：{len(symbols)}")
    print(f"688股票池已保存：{STOCK_POOL_PATH}")

    return symbols


# =========================
# 数据获取：带防卡死
# =========================

def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(REQUEST_TIMEOUT_SECONDS)

    try:
        fields = (
            "date,code,open,high,low,close,preclose,volume,amount,"
            "adjustflag,turn,tradestatus,pctChg,isST"
        )

        rs = bs.query_history_k_data_plus(
            symbol,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",
        )

        if rs.error_code != "0":
            raise RuntimeError(f"BaoStock错误：{rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        df = pd.DataFrame(rows, columns=rs.fields)

    finally:
        signal.alarm(0)

    if df.empty:
        return pd.DataFrame()

    df = df.rename(
        columns={
            "code": "symbol",
            "turn": "turnover",
            "pctChg": "pct_change",
        }
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "open", "high", "low", "close", "preclose",
        "volume", "amount", "turnover", "pct_change",
        "tradestatus", "isST",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["tradestatus"] == 1].copy()
    df = df[df["isST"] == 0].copy()

    if df.empty:
        return pd.DataFrame()

    df["change"] = df["close"] - df["preclose"]
    df["amplitude"] = (df["high"] - df["low"]) / df["preclose"] * 100

    df["symbol"] = df["symbol"].apply(normalize_symbol)

    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================
# 特征工程
# =========================

def add_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()

    g["return_1d"] = g["close"].pct_change()

    for ma in [5, 10, 20, 60, 120]:
        g[f"ma{ma}"] = g["close"].rolling(ma).mean()

    g["close_ma5_ratio"] = g["close"] / g["ma5"] - 1
    g["close_ma20_ratio"] = g["close"] / g["ma20"] - 1
    g["close_ma60_ratio"] = g["close"] / g["ma60"] - 1

    g["vol_ma5"] = g["volume"].rolling(5).mean()
    g["vol_ma20"] = g["volume"].rolling(20).mean()
    g["volume_ratio"] = g["volume"] / g["vol_ma20"]

    g["volatility_10d"] = g["return_1d"].rolling(10).std()
    g["volatility_20d"] = g["return_1d"].rolling(20).std()

    g["body_ratio"] = (g["close"] - g["open"]) / g["open"]

    g["upper_shadow_ratio"] = (
        g["high"] - g[["open", "close"]].max(axis=1)
    ) / g["open"]

    g["lower_shadow_ratio"] = (
        g[["open", "close"]].min(axis=1) - g["low"]
    ) / g["open"]

    g["high_20d"] = g["high"].rolling(20).max()
    g["low_20d"] = g["low"].rolling(20).min()

    delta = g["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain_14 = gain.rolling(14).mean()
    avg_loss_14 = loss.rolling(14).mean()
    rs_14 = avg_gain_14 / avg_loss_14
    g["rsi_14"] = 100 - (100 / (1 + rs_14))

    avg_gain_6 = gain.rolling(6).mean()
    avg_loss_6 = loss.rolling(6).mean()
    rs_6 = avg_gain_6 / avg_loss_6
    g["rsi_6"] = 100 - (100 / (1 + rs_6))

    ema12 = g["close"].ewm(span=12, adjust=False).mean()
    ema26 = g["close"].ewm(span=26, adjust=False).mean()

    g["macd"] = ema12 - ema26
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
    g["macd_diff"] = g["macd"] - g["macd_signal"]

    mid = g["close"].rolling(20).mean()
    std = g["close"].rolling(20).std()

    g["boll_high"] = mid + 2 * std
    g["boll_low"] = mid - 2 * std
    g["boll_width"] = (g["boll_high"] - g["boll_low"]) / g["close"]

    g["boll_position"] = (
        (g["close"] - g["boll_low"]) / (g["boll_high"] - g["boll_low"])
    )

    return g


def add_extra_short_term_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()

    g["high_10d"] = g["high"].rolling(10).max()
    g["high_20d_prev"] = g["high"].rolling(20).max().shift(1)
    g["break_20d_high"] = (g["close"] >= g["high_20d_prev"] * 0.99).astype(int)

    g["pct_change_3d"] = g["close"].pct_change(3) * 100
    g["pct_change_5d"] = g["close"].pct_change(5) * 100
    g["amount_ma20"] = g["amount"].rolling(20).mean()

    return g


def add_rainbow_score(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()

    valid = g[["ma5", "ma10", "ma20", "ma60"]].notna().all(axis=1)

    score = pd.Series(0, index=g.index, dtype="float")

    score += (
        (g["ma5"] > g["ma10"])
        & (g["ma10"] > g["ma20"])
        & (g["ma20"] > g["ma60"])
    ) * 40

    score += (g["close"] > g["ma20"]) * 20
    score += (g["close"] > g["ma60"]) * 20
    score += ((g["ma5"] / g["ma60"] - 1) > 0.05) * 20

    g["rainbow_score"] = score.clip(upper=100)
    g.loc[~valid, "rainbow_score"] = np.nan

    return g


def add_main_force_score(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()

    score = pd.Series(0, index=g.index, dtype="float")

    cond_volume = g["volume_ratio"] > 2
    cond_big_up = g["pct_change"] > 5
    cond_turnover = g["turnover"] > 5
    cond_yang = g["close"] > g["open"]

    if "break_20d_high" in g.columns:
        cond_break = g["break_20d_high"] == 1
    else:
        cond_break = pd.Series(False, index=g.index)

    score += cond_volume * 30
    score += cond_big_up * 30
    score += cond_turnover * 20
    score += cond_yang * 20
    score += cond_break * 20

    g["main_force_score"] = score.clip(upper=100)
    g.loc[g["volume_ratio"].isna(), "main_force_score"] = np.nan

    signal_df = pd.DataFrame(index=g.index)
    signal_df["放量"] = np.where(cond_volume, "放量", "")
    signal_df["大涨"] = np.where(cond_big_up, "大涨", "")
    signal_df["换手活跃"] = np.where(cond_turnover, "换手活跃", "")
    signal_df["阳线"] = np.where(cond_yang, "阳线", "")
    signal_df["突破20日高点"] = np.where(cond_break, "突破20日高点", "")

    g["main_force_signal"] = signal_df.apply(
        lambda row: "；".join([x for x in row if x]),
        axis=1,
    )

    return g


# =========================
# 标签
# =========================

def add_labels(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()

    g["future_close"] = g["close"].shift(-FUTURE_DAYS)
    g["future_return"] = g["future_close"] / g["close"] - 1

    g["label"] = 0
    g.loc[g["future_return"] >= UP_THRESHOLD, "label"] = 1
    g.loc[g["future_return"] <= DOWN_THRESHOLD, "label"] = -1

    rainbow_score = g["rainbow_score"].fillna(0)
    main_force_score = g["main_force_score"].fillna(0)
    pct_change = g["pct_change"].fillna(0)
    future_return = g["future_return"].fillna(0)

    g["manual_label"] = 0

    g.loc[
        (pct_change >= 5) & (future_return < 0),
        "manual_label"
    ] = -1

    g.loc[
        future_return <= DOWN_THRESHOLD,
        "manual_label"
    ] = -1

    g.loc[
        (rainbow_score >= 80) & (main_force_score >= 60),
        "manual_label"
    ] = 1

    g.loc[
        (pct_change >= 5) & (main_force_score >= 60) & (future_return > 0),
        "manual_label"
    ] = 1

    g["final_label"] = g["manual_label"]

    return g


# =========================
# 重新计算整只股票特征
# =========================

def recalc_symbol_dataset(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    g = g.drop_duplicates(["symbol", "date"], keep="last")

    g = add_features(g)
    g = add_extra_short_term_features(g)
    g = add_rainbow_score(g)
    g = add_main_force_score(g)
    g = add_labels(g)

    g = g.dropna(subset=["rainbow_score", "main_force_score"])

    return g


# =========================
# 增量更新
# =========================

def load_existing_dataset() -> pd.DataFrame | None:
    if not os.path.exists(DATASET_PATH):
        return None

    print(f"读取已有数据集：{DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig", dtype={"symbol": str})
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 只保留688，防止旧数据里有300/301混进来
    df = df[df["symbol"].str.startswith("688")].copy()

    print(f"已有688样本数：{len(df)}")

    if not df.empty:
        print(f"已有688日期范围：{df['date'].min()} -> {df['date'].max()}")

    return df


def update_one_symbol(bs_symbol: str, existing_df: pd.DataFrame | None) -> pd.DataFrame:
    symbol = normalize_symbol(bs_symbol)

    if existing_df is None or existing_df.empty:
        start_date = FULL_START_DATE
        old_g = pd.DataFrame()
    else:
        old_g = existing_df[existing_df["symbol"] == symbol].copy()

        if old_g.empty:
            start_date = FULL_START_DATE
        else:
            last_date = old_g["date"].max()
            start_date = next_day(last_date)

    if start_date > END_DATE:
        print(f"{symbol} 已是最新，直接复用旧数据")
        if old_g.empty:
            return pd.DataFrame()
        return old_g

    print(f"{symbol} 更新区间：{start_date} -> {END_DATE}")

    new_g = fetch_stock_data(
        bs_symbol,
        start_date=start_date,
        end_date=END_DATE,
    )

    if new_g.empty:
        print(f"{symbol} 没有新数据，直接复用旧数据")
        if old_g.empty:
            return pd.DataFrame()
        return old_g

    if old_g.empty:
        combined = new_g
    else:
        base_cols = [
            "symbol", "date", "open", "high", "low", "close", "preclose",
            "volume", "amount", "turnover", "pct_change", "change", "amplitude",
        ]

        keep_old_cols = [c for c in base_cols if c in old_g.columns]
        old_base = old_g[keep_old_cols].copy()

        combined = pd.concat([old_base, new_g], ignore_index=True)

    combined["symbol"] = combined["symbol"].astype(str).str.zfill(6)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.drop_duplicates(["symbol", "date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)

    if len(combined) < MIN_HISTORY_ROWS:
        raise ValueError(f"历史数据太少：{len(combined)}")

    amount_mean = combined["amount"].tail(20).mean()

    if pd.isna(amount_mean) or amount_mean < MIN_AMOUNT_MEAN:
        raise ValueError(f"成交额过低：{amount_mean:.0f}")

    return recalc_symbol_dataset(combined)


# =========================
# 主流程
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_start_time = time.time()

    existing_df = load_existing_dataset()

    login_baostock()

    all_data = []
    failed_records = []

    try:
        symbols = get_stock_pool()
        total_symbols = len(symbols)

        print("\n==============================")
        print(f"本次准备处理全部688股票：{total_symbols} 只")
        print("==============================")

        for idx, bs_symbol in enumerate(symbols, start=1):
            one_start_time = time.time()

            print("\n" + "=" * 60)
            print(f"[{idx}/{total_symbols}] 处理 {bs_symbol}")

            try:
                g = update_one_symbol(bs_symbol, existing_df)

                if not g.empty:
                    all_data.append(g)
                    print(f"{bs_symbol} 完成，样本数：{len(g)}")
                else:
                    print(f"{bs_symbol} 无有效数据")

            except TimeoutError as e:
                failed_records.append(
                    {
                        "symbol": bs_symbol,
                        "reason": str(e),
                    }
                )
                print(f"{bs_symbol} 超时跳过：{e}")

            except Exception as e:
                failed_records.append(
                    {
                        "symbol": bs_symbol,
                        "reason": str(e),
                    }
                )
                print(f"{bs_symbol} 失败：{e}")

            one_cost = time.time() - one_start_time
            print(f"{bs_symbol} 耗时：{one_cost:.2f} 秒")

            if SLEEP_BETWEEN_STOCKS > 0:
                time.sleep(SLEEP_BETWEEN_STOCKS)

    finally:
        logout_baostock()

    if failed_records:
        pd.DataFrame(failed_records).to_csv(
            FAILED_PATH,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"\n失败记录已保存：{FAILED_PATH}")

    if not all_data:
        raise ValueError("没有任何688股票成功，无法更新数据集。")

    print("\n正在合并688数据...")

    dataset = pd.concat(all_data, ignore_index=True)

    dataset["symbol"] = dataset["symbol"].astype(str).str.zfill(6)
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")

    dataset = dataset[dataset["symbol"].str.startswith("688")].copy()

    dataset = dataset.drop_duplicates(["symbol", "date"], keep="last")
    dataset = dataset.sort_values(["date", "symbol"]).reset_index(drop=True)

    dataset.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")

    total_cost = time.time() - total_start_time

    print("\n==============================")
    print("全部688增量更新完成")
    print(f"数据集路径：{DATASET_PATH}")
    print(f"总样本数：{len(dataset)}")
    print(f"688股票数：{dataset['symbol'].nunique()}")
    print(f"日期范围：{dataset['date'].min()} -> {dataset['date'].max()}")
    print(f"总耗时：{total_cost:.2f} 秒")

    print("\nfinal_label 分布：")
    print(dataset["final_label"].value_counts())
    print(dataset["final_label"].value_counts(normalize=True))

    print("\n下一步运行：")
    print("python3 train_model.py")
    print("python3 yaogu_model_v3.py")
    print("streamlit run app.py")


if __name__ == "__main__":
    main()