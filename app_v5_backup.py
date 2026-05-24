"""
app_trading_terminal_v5_1_industry_fix.py
单文件版 A股交易终端 V5

运行：
    streamlit run app_trading_terminal_v5_core.py

说明：
    这个版本不再依赖 prediction_archive.py / review_predictions.py / market_emotion.py，避免模块导入失败。
"""

import os
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

OUTPUT_DIR = "output"
HISTORY_DIR = "history"
DAILY_SIGNAL_DIR = os.path.join(HISTORY_DIR, "daily_signals")
REVIEW_DIR = os.path.join(HISTORY_DIR, "reviews")

SIGNAL_PATH = os.path.join(OUTPUT_DIR, "latest_stock_signals.csv")
YAOGU_PATH = os.path.join(OUTPUT_DIR, "yaogu_model_v3_signals.csv")
CLEAN_POOL_PATH = os.path.join(OUTPUT_DIR, "clean_stock_pool.csv")
REMOVED_POOL_PATH = os.path.join(OUTPUT_DIR, "removed_stock_pool.csv")
META_PATH = os.path.join(OUTPUT_DIR, "stock_meta_map.csv")
REVIEW_RESULT_PATH = os.path.join(HISTORY_DIR, "prediction_review.csv")

TRAIN_SCRIPT = "train_model.py"
YAOGU_SCRIPT = "yaogu_model_v3.py"
FILTER_SCRIPT = "filter_stock_pool.py"


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(DAILY_SIGNAL_DIR, exist_ok=True)
    os.makedirs(REVIEW_DIR, exist_ok=True)


def safe_read(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8", dtype=str)


def normalize_symbol(df):
    df = df.copy()
    if "symbol" not in df.columns:
        if "code" in df.columns:
            df["symbol"] = df["code"]
        elif "股票代码" in df.columns:
            df["symbol"] = df["股票代码"]
        else:
            st.error(f"文件缺少 symbol/code/股票代码 列，当前列名：{list(df.columns)}")
            st.stop()
    df["symbol"] = (
        df["symbol"].astype(str)
        .str.replace("sh.", "", regex=False)
        .str.replace("sz.", "", regex=False)
        .str.replace(".0", "", regex=False)
        .str.extract(r"(\d+)", expand=False)
        .fillna(df["symbol"].astype(str))
        .str.zfill(6)
    )
    return df


def to_number(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _standardize_meta_columns(meta):
    """兼容不同来源的股票基础信息字段名。"""
    meta = meta.copy()
    alias_map = {
        "code": "symbol",
        "股票代码": "symbol",
        "证券代码": "symbol",
        "name": "stock_name",
        "股票名称": "stock_name",
        "证券简称": "stock_name",
        "名称": "stock_name",
        "所属行业": "industry",
        "行业名称": "industry",
        "申万行业": "industry",
        "申万一级行业": "industry",
        "东财行业": "industry",
        "通达信行业": "industry",
        "概念板块": "themes",
        "题材概念": "themes",
        "概念": "themes",
        "题材": "themes",
        "市场板块": "board",
        "上市板块": "board",
    }
    for old_name, new_name in alias_map.items():
        if old_name in meta.columns and new_name not in meta.columns:
            meta[new_name] = meta[old_name]
    return meta


def _fill_placeholder_with_meta(df, col, default):
    meta_col = f"{col}_meta"
    bad_values = {"", "未知", "其他", "None", "nan", "NaN", "--", "-"}
    if col not in df.columns:
        df[col] = default
    if meta_col in df.columns:
        left = df[col].astype(str)
        bad_mask = df[col].isna() | left.isin(bad_values)
        df.loc[bad_mask, col] = df.loc[bad_mask, meta_col]
    df[col] = df[col].replace(list(bad_values), np.nan).fillna(default)
    return df


def merge_meta(df):
    df = normalize_symbol(df)

    # 先兼容信号文件自身可能已经带有中文字段
    df = _standardize_meta_columns(df)

    meta = safe_read(META_PATH)
    if meta is not None and not meta.empty:
        meta = _standardize_meta_columns(meta)
        meta = normalize_symbol(meta)
        keep = [c for c in ["symbol", "stock_name", "industry", "board", "themes"] if c in meta.columns]
        if "symbol" in keep:
            meta = meta[keep].drop_duplicates("symbol")
            df = df.merge(meta, on="symbol", how="left", suffixes=("", "_meta"))

    for col, default in [("stock_name", ""), ("industry", "未知"), ("board", "未知"), ("themes", "其他")]:
        df = _fill_placeholder_with_meta(df, col, default)

    # 如果 board 缺失，根据代码做一个最基础的市场板块判断
    board_bad = df["board"].astype(str).isin(["", "未知", "其他", "None", "nan", "NaN", "--", "-"])
    df.loc[board_bad & df["symbol"].str.startswith("688"), "board"] = "科创板"
    df.loc[board_bad & df["symbol"].str.startswith("300"), "board"] = "创业板"
    df.loc[board_bad & df["symbol"].str.startswith("60"), "board"] = "沪市主板"
    df.loc[board_bad & df["symbol"].str.startswith("00"), "board"] = "深市主板"

    df["display_name"] = (df["symbol"].astype(str) + " " + df["stock_name"].astype(str)).str.strip()
    return df


def load_clean_symbols():
    df = safe_read(CLEAN_POOL_PATH)
    if df is None or df.empty:
        return set()
    return set(normalize_symbol(df)["symbol"].tolist())


def format_percent(x):
    try:
        v = float(x)
        if pd.isna(v):
            return "0.0%"
        if abs(v) <= 1:
            v *= 100
        return f"{v:.1f}%"
    except Exception:
        return "0.0%"


def prepare_signal_df(df):
    df = merge_meta(df)
    nums = ["close", "pct_change", "rainbow_score", "main_force_score", "pred_label", "prob_up", "prob_down", "model_score"]
    df = to_number(df, nums)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    clean = load_clean_symbols()
    df["is_clean_pool"] = df["symbol"].isin(clean) if clean else False

    def status(r):
        if r.get("model_score", 0) >= 60 and r.get("prob_up", 0) >= 0.6 and r.get("rainbow_score", 0) >= 80 and r.get("main_force_score", 0) >= 60 and r.get("pred_label", 0) == 1:
            return "强买观察"
        if r.get("pred_label", 0) == -1 or r.get("prob_down", 0) >= 0.5 or r.get("pct_change", 0) <= -3:
            return "风险"
        if r.get("model_score", 0) >= 30 and r.get("rainbow_score", 0) >= 60:
            return "观察"
        return "弱"

    df["signal_status"] = df.apply(status, axis=1)
    if "model_score" in df.columns:
        df = df.sort_values("model_score", ascending=False)
    return df.reset_index(drop=True)


def prepare_yaogu_df(df):
    df = merge_meta(df)
    nums = ["close", "pct_change", "turnover", "rainbow_score", "main_force_score", "model_score", "prob_up", "prob_down", "pred_label", "volume_ratio_v3", "pct_5d", "pct_10d", "pct_20d", "yaogu_v3_score", "yaogu_v3_raw_score", "yaogu_v3_risk"]
    df = to_number(df, nums)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    clean = load_clean_symbols()
    df["is_clean_pool"] = df["symbol"].isin(clean) if clean else False
    if "yaogu_v3_score" in df.columns:
        df = df.sort_values("yaogu_v3_score", ascending=False)
    return df.reset_index(drop=True)


def compute_market_emotion(df):
    if df is None or df.empty:
        return {"market_phase": "无数据", "emotion_score": 0, "advice": "先更新数据", "strong_count": 0, "risk_count": 0, "top_industry": "未知"}
    df = to_number(df, ["model_score", "rainbow_score", "main_force_score", "pct_change", "prob_down"])
    total = max(len(df), 1)
    avg_model = float(df.get("model_score", pd.Series([0])).mean())
    avg_rainbow = float(df.get("rainbow_score", pd.Series([0])).mean())
    avg_force = float(df.get("main_force_score", pd.Series([0])).mean())
    strong = int(((df.get("model_score", 0) >= 60) & (df.get("rainbow_score", 0) >= 70) & (df.get("main_force_score", 0) >= 60)).sum())
    risk = int(((df.get("pct_change", 0) <= -3) | (df.get("prob_down", 0) >= 0.5)).sum())
    score = avg_model * 0.35 + avg_rainbow * 0.35 + avg_force * 0.30
    risk_ratio = risk / total
    strong_ratio = strong / total
    if risk_ratio >= 0.45 or score < 25:
        phase, advice = "退潮/防守", "控制仓位，别硬上。现金也是仓位。"
    elif strong_ratio >= 0.08 and score >= 55:
        phase, advice = "主升/进攻", "关注强主线核心票，但不要无脑追高。"
    elif score >= 40:
        phase, advice = "修复/观察", "轻仓试错，等板块和个股共振。"
    else:
        phase, advice = "冰点/等待", "信号弱，等市场自己把牌亮出来。"
    top_industry = "未知"
    if "industry" in df.columns and "model_score" in df.columns:
        s = df.groupby("industry")["model_score"].mean().sort_values(ascending=False)
        if not s.empty:
            top_industry = str(s.index[0])
    return {"market_phase": phase, "emotion_score": round(score, 2), "advice": advice, "strong_count": strong, "risk_count": risk, "top_industry": top_industry}


def industry_heatmap_data(df, top_n=20):
    if df is None or df.empty or "industry" not in df.columns:
        return pd.DataFrame()
    df = to_number(df, ["model_score", "rainbow_score", "main_force_score", "pct_change"])
    g = df.groupby("industry").agg(
        股票数=("symbol", "count"),
        平均模型分=("model_score", "mean"),
        平均趋势分=("rainbow_score", "mean"),
        平均主力分=("main_force_score", "mean"),
        平均涨跌幅=("pct_change", "mean"),
    ).reset_index()
    g["热度分"] = g["平均模型分"] * 0.4 + g["平均趋势分"] * 0.3 + g["平均主力分"] * 0.3
    return g.sort_values("热度分", ascending=False).head(top_n).round(2)


def choose_recommendations(signal_df, yaogu_df=None, top_n=30, only_clean=False):
    if yaogu_df is not None and not yaogu_df.empty:
        df = yaogu_df.copy()
        sort_col = "yaogu_v3_score" if "yaogu_v3_score" in df.columns else "model_score"
    else:
        df = signal_df.copy()
        sort_col = "model_score"
    if only_clean and "is_clean_pool" in df.columns:
        df = df[df["is_clean_pool"]]
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)
    return df.head(top_n).copy()


def archive_today_predictions(signal_df, yaogu_df=None, top_n=30, only_clean=False):
    ensure_dirs()
    rec = choose_recommendations(signal_df, yaogu_df, top_n, only_clean)
    if rec.empty:
        raise ValueError("没有可存档的推荐")
    date_value = rec["date"].max() if "date" in rec.columns else datetime.today()
    try:
        trade_date = pd.to_datetime(date_value).strftime("%Y-%m-%d")
    except Exception:
        trade_date = datetime.today().strftime("%Y-%m-%d")
    rec["archive_date"] = trade_date
    rec["archive_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rec["recommend_rank"] = range(1, len(rec) + 1)
    path = os.path.join(DAILY_SIGNAL_DIR, f"{trade_date}_predictions.csv")
    rec.to_csv(path, index=False, encoding="utf-8-sig")
    return path, rec


def list_prediction_files():
    ensure_dirs()
    return sorted([os.path.join(DAILY_SIGNAL_DIR, f) for f in os.listdir(DAILY_SIGNAL_DIR) if f.endswith("_predictions.csv")], reverse=True)


def review_prediction_file(prediction_file, current_df):
    ensure_dirs()
    pred = safe_read(prediction_file)
    if pred is None or pred.empty:
        raise FileNotFoundError(f"预测文件为空或不存在：{prediction_file}")
    pred = normalize_symbol(pred)
    current = current_df.copy()
    for col in ["close", "pct_change", "model_score", "yaogu_v3_score"]:
        if col in pred.columns:
            pred[col] = pd.to_numeric(pred[col], errors="coerce")
        if col in current.columns:
            current[col] = pd.to_numeric(current[col], errors="coerce")
    keep = [c for c in ["symbol", "date", "close", "pct_change", "model_score", "rainbow_score", "main_force_score", "prob_up", "prob_down", "signal_status"] if c in current.columns]
    cur = current[keep].drop_duplicates("symbol").rename(columns={"date": "review_date", "close": "review_close", "pct_change": "next_pct_change", "model_score": "review_model_score", "rainbow_score": "review_rainbow_score", "main_force_score": "review_main_force_score", "prob_up": "review_prob_up", "prob_down": "review_prob_down", "signal_status": "review_signal_status"})
    result = pred.merge(cur, on="symbol", how="left")
    if "close" in result.columns and "review_close" in result.columns:
        result["entry_close"] = pd.to_numeric(result["close"], errors="coerce")
        result["review_close"] = pd.to_numeric(result["review_close"], errors="coerce")
        result["real_return_pct"] = (result["review_close"] / result["entry_close"] - 1) * 100
    elif "next_pct_change" in result.columns:
        result["real_return_pct"] = pd.to_numeric(result["next_pct_change"], errors="coerce")
    else:
        result["real_return_pct"] = np.nan
    result["hit"] = result["real_return_pct"] > 0
    result["big_win"] = result["real_return_pct"] >= 5
    result["big_loss"] = result["real_return_pct"] <= -5
    result["review_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = os.path.join(REVIEW_DIR, os.path.basename(prediction_file).replace("_predictions.csv", "_review.csv"))
    result.to_csv(out, index=False, encoding="utf-8-sig")
    if os.path.exists(REVIEW_RESULT_PATH):
        old = safe_read(REVIEW_RESULT_PATH)
        all_df = pd.concat([old, result], ignore_index=True) if old is not None else result
        subset = [c for c in ["archive_date", "symbol", "review_date"] if c in all_df.columns]
        if subset:
            all_df = all_df.drop_duplicates(subset=subset, keep="last")
    else:
        all_df = result
    all_df.to_csv(REVIEW_RESULT_PATH, index=False, encoding="utf-8-sig")
    return out, result


def summarize_review(df):
    if df is None or df.empty:
        return {"total": 0, "hit_rate": 0, "avg_return": 0, "big_win_count": 0, "big_loss_count": 0}
    r = pd.to_numeric(df.get("real_return_pct"), errors="coerce")
    return {"total": len(df), "hit_rate": round(float((r > 0).mean() * 100), 2), "avg_return": round(float(r.mean()), 2), "big_win_count": int((r >= 5).sum()), "big_loss_count": int((r <= -5).sum())}



def find_latest_prediction_before(current_date_text=None):
    files = list_prediction_files()
    if not files:
        return None
    if current_date_text is None:
        return files[0]
    try:
        cur = pd.to_datetime(current_date_text)
    except Exception:
        return files[0]
    candidates = []
    for f in files:
        name = os.path.basename(f).replace('_predictions.csv','')
        try:
            d = pd.to_datetime(name)
            if d < cur:
                candidates.append((d, f))
        except Exception:
            pass
    if not candidates:
        return files[0]
    return sorted(candidates, reverse=True)[0][1]


def run_script(script):
    if not os.path.exists(script):
        return script, 1, '', f'找不到 {script}'
    p = subprocess.run(f'python3 {script}', shell=True, capture_output=True, text=True)
    return script, p.returncode, p.stdout, p.stderr


def auto_update_pipeline(run_filter=True, run_archive=True, run_review=True, top_n=30, only_clean=False):
    logs = []
    for script in ([FILTER_SCRIPT] if run_filter else []):
        logs.append(run_script(script))
    for script in [TRAIN_SCRIPT, YAOGU_SCRIPT]:
        logs.append(run_script(script))
    archive_info = None
    review_info = None
    latest = safe_read(SIGNAL_PATH)
    if latest is not None and not latest.empty:
        sig = prepare_signal_df(latest)
        yg_raw = safe_read(YAOGU_PATH)
        yg = prepare_yaogu_df(yg_raw) if yg_raw is not None and not yg_raw.empty else pd.DataFrame()
        if run_archive:
            try:
                archive_info = archive_today_predictions(sig, yg if not yg.empty else None, top_n, only_clean)[0]
            except Exception as e:
                archive_info = f'存档失败：{e}'
        if run_review:
            try:
                d = sig['date'].max() if 'date' in sig.columns else None
                d_text = pd.to_datetime(d).strftime('%Y-%m-%d') if d is not None else None
                old_file = find_latest_prediction_before(d_text)
                if old_file:
                    review_info = review_prediction_file(old_file, sig)[0]
            except Exception as e:
                review_info = f'复盘失败：{e}'
    return logs, archive_info, review_info


def market_breadth(df):
    if df is None or df.empty:
        return {'上涨占比':0, '强势占比':0, '风险占比':0, '平均涨跌幅':0}
    x = to_number(df, ['pct_change','model_score','rainbow_score','main_force_score','prob_down'])
    pct = pd.to_numeric(x.get('pct_change'), errors='coerce') if 'pct_change' in x.columns else pd.Series(dtype=float)
    strong = ((x.get('model_score',0)>=60)&(x.get('rainbow_score',0)>=70)&(x.get('main_force_score',0)>=60))
    risk = ((x.get('prob_down',0)>=0.5)|(x.get('pct_change',0)<=-3))
    n=max(len(x),1)
    return {'上涨占比':round(float((pct>0).mean()*100),2) if len(pct) else 0, '强势占比':round(float(strong.sum()/n*100),2), '风险占比':round(float(risk.sum()/n*100),2), '平均涨跌幅':round(float(pct.mean()),2) if len(pct) else 0}


def leading_stock_data(df, top_n=20):
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    x = to_number(x, ['model_score','rainbow_score','main_force_score','pct_change','prob_up','prob_down','yaogu_v3_score'])
    base = x['model_score'] if 'model_score' in x.columns else 0
    trend = x['rainbow_score'] if 'rainbow_score' in x.columns else 0
    force = x['main_force_score'] if 'main_force_score' in x.columns else 0
    pct = x['pct_change'] if 'pct_change' in x.columns else 0
    yaogu = x['yaogu_v3_score'] if 'yaogu_v3_score' in x.columns else base
    x['leader_score'] = base*0.28 + trend*0.24 + force*0.24 + yaogu*0.16 + np.clip(pct, -10, 10)*0.8
    x['leader_tag'] = np.where((x.get('rainbow_score',0)>=85)&(x.get('main_force_score',0)>=75), '龙头候选', np.where(x.get('pct_change',0)>=5, '加速候选', '观察'))
    return x.sort_values('leader_score', ascending=False).head(top_n).round(2)


def style_detection(df):
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    x = to_number(x, ['model_score','rainbow_score','main_force_score','pct_change'])
    def board_of(s):
        s=str(s)
        if s.startswith('688'): return '科创板688'
        if s.startswith('300'): return '创业板300'
        if s.startswith('60'): return '沪主板60'
        if s.startswith('00'): return '深主板00'
        return '其他'
    if 'board' not in x.columns or x['board'].isna().all():
        x['style_board'] = x['symbol'].apply(board_of)
    else:
        x['style_board'] = x['board'].fillna(x['symbol'].apply(board_of))
    g=x.groupby('style_board').agg(股票数=('symbol','count'),平均模型分=('model_score','mean'),平均趋势分=('rainbow_score','mean'),平均主力分=('main_force_score','mean'),平均涨跌幅=('pct_change','mean')).reset_index()
    g['风格强度']=g['平均模型分']*0.35+g['平均趋势分']*0.30+g['平均主力分']*0.25+g['平均涨跌幅']*1.0
    return g.sort_values('风格强度', ascending=False).round(2)


def risk_radar(df):
    if df is None or df.empty:
        return []
    x=to_number(df, ['pct_change','prob_down','rainbow_score','main_force_score','model_score'])
    alerts=[]
    n=max(len(x),1)
    down3=int((x.get('pct_change',0)<=-3).sum())
    high_down=int((x.get('prob_down',0)>=0.5).sum())
    weak_trend=int((x.get('rainbow_score',0)<30).sum())
    if down3/n>=0.15: alerts.append(('高风险', f'大跌票占比 {down3/n*100:.1f}%，盘面偏弱，别急着抄底。'))
    if high_down/n>=0.25: alerts.append(('偏空', f'下跌概率较高股票占比 {high_down/n*100:.1f}%，模型整体偏谨慎。'))
    if weak_trend/n>=0.35: alerts.append(('趋势差', f'趋势分低于30的股票占比 {weak_trend/n*100:.1f}%，说明赚钱效应不足。'))
    if not alerts: alerts.append(('正常', '暂未发现系统性风险，但个股追高仍要管住手。'))
    return alerts


def review_group_stats(df):
    if df is None or df.empty or 'real_return_pct' not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    x=df.copy()
    x['real_return_pct']=pd.to_numeric(x['real_return_pct'], errors='coerce')
    if 'industry' in x.columns:
        by_ind=x.groupby('industry').agg(样本数=('symbol','count'),命中率=('real_return_pct', lambda s: round((s>0).mean()*100,2)),平均收益=('real_return_pct','mean')).reset_index().sort_values('平均收益', ascending=False).round(2)
    else:
        by_ind=pd.DataFrame()
    score_col='model_score' if 'model_score' in x.columns else None
    by_score=pd.DataFrame()
    if score_col:
        x[score_col]=pd.to_numeric(x[score_col], errors='coerce')
        x['模型分区间']=pd.cut(x[score_col], bins=[-1,40,60,80,100,999], labels=['<=40','40-60','60-80','80-100','>100'])
        by_score=x.groupby('模型分区间', observed=False).agg(样本数=('symbol','count'),命中率=('real_return_pct', lambda s: round((s>0).mean()*100,2)),平均收益=('real_return_pct','mean')).reset_index().round(2)
    return by_ind, by_score

def display_table(df):
    show = df.copy()
    if "date" in show.columns:
        show["date"] = pd.to_datetime(show["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["prob_up", "prob_down"]:
        if col in show.columns:
            show[col] = show[col].apply(format_percent)
    if "is_clean_pool" in show.columns:
        show["is_clean_pool"] = np.where(show["is_clean_pool"], "✅", "")
    for col in show.columns:
        if col in ["close", "pct_change", "rainbow_score", "main_force_score", "model_score", "yaogu_v3_score", "yaogu_v3_raw_score", "yaogu_v3_risk", "real_return_pct"]:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(2)
    rename = {"display_name": "股票", "is_clean_pool": "优质池", "industry": "行业", "board": "板块", "themes": "题材", "date": "日期", "close": "收盘价", "pct_change": "涨跌幅%", "rainbow_score": "趋势分", "main_force_score": "主力分", "model_score": "模型分", "prob_up": "上涨概率", "prob_down": "下跌概率", "pred_label": "预测标签", "signal_status": "状态", "yaogu_v3_score": "妖股v3分", "yaogu_v3_raw_score": "原始分", "yaogu_v3_risk": "风险扣分", "yaogu_v3_level": "等级", "yaogu_v3_action": "操作建议", "yaogu_v3_reason": "入选原因", "real_return_pct": "验证收益%", "hit": "命中"}
    preferred = ["display_name", "is_clean_pool", "industry", "board", "themes", "date", "close", "pct_change", "model_score", "rainbow_score", "main_force_score", "prob_up", "prob_down", "pred_label", "signal_status", "yaogu_v3_score", "yaogu_v3_raw_score", "yaogu_v3_risk", "yaogu_v3_level", "yaogu_v3_action", "yaogu_v3_reason", "real_return_pct", "hit"]
    cols = [c for c in preferred if c in show.columns]
    return show[cols].rename(columns=rename)


def inject_css():
    st.markdown("""
    <style>
    .stApp{background:#080d14;color:#e5e7eb}.block-container{padding-top:1.2rem;max-width:1500px}[data-testid="stSidebar"]{background:#0b1220}.hero{background:linear-gradient(135deg,#111827,#0f172a 55%,#312e81);border:1px solid rgba(148,163,184,.25);padding:26px 30px;border-radius:26px;margin-bottom:18px;box-shadow:0 18px 48px rgba(0,0,0,.35)}.hero h1{margin:0;font-size:38px;color:white}.hero p{color:#cbd5e1}.card{background:#0f172a;border:1px solid rgba(148,163,184,.22);border-radius:20px;padding:18px;box-shadow:0 12px 28px rgba(0,0,0,.28);min-height:126px}.k{font-size:13px;color:#94a3b8;font-weight:700}.v{font-size:30px;font-weight:900;color:#fff;margin-top:8px}.n{font-size:13px;color:#94a3b8;margin-top:6px}.hot{color:#fb7185}.good{color:#22c55e}.gold{color:#fbbf24}.purple{color:#a78bfa}</style>
    """, unsafe_allow_html=True)


def metric_card(label, value, note, cls=""):
    st.markdown(f"<div class='card'><div class='k'>{label}</div><div class='v {cls}'>{value}</div><div class='n'>{note}</div></div>", unsafe_allow_html=True)


def top_card(row, rank):
    score = row.get("yaogu_v3_score", row.get("model_score", 0))
    try:
        score = float(score)
    except Exception:
        score = 0
    action = row.get("yaogu_v3_action", row.get("signal_status", "观察"))
    level = row.get("yaogu_v3_level", "")
    st.markdown(f"""
    <div class='card' style='min-height:260px'>
      <div class='k'>#{rank} 核心作战卡</div>
      <div class='v gold'>{row.get('display_name', row.get('symbol',''))}</div>
      <div class='n'>行业：{row.get('industry','未知')} ｜ 板块：{row.get('board','未知')}</div>
      <hr style='border-color:rgba(148,163,184,.18)'>
      <div class='n'>综合分：<b class='hot'>{score:.2f}</b> ｜ 收盘：{row.get('close','')}</div>
      <div class='n'>涨跌幅：{row.get('pct_change','')}% ｜ 趋势：{row.get('rainbow_score','')} ｜ 主力：{row.get('main_force_score','')}</div>
      <div class='n'>题材：{row.get('themes','其他')}</div>
      <div class='n'>建议：<b class='purple'>{action}</b> {level}</div>
    </div>""", unsafe_allow_html=True)


st.set_page_config(page_title="A股交易终端 V5.1 行业修复版", page_icon="🚀", layout="wide")
inject_css()
ensure_dirs()

st.sidebar.markdown("## 🚀 控制台")
if st.sidebar.button("🔄 一键更新+存档+复盘", use_container_width=True):
    logs, archive_info, review_info = auto_update_pipeline(run_filter=True, run_archive=True, run_review=True, top_n=30, only_clean=False)
    with st.expander("运行日志", expanded=True):
        for script, code, out, err in logs:
            st.success(f"{script} 完成") if code == 0 else st.error(f"{script} 失败")
            if out:
                st.code(out[-3000:])
            if err:
                st.code(err[-3000:])
        if archive_info:
            st.info(f"存档：{archive_info}")
        if review_info:
            st.info(f"复盘：{review_info}")
    st.rerun()

if st.sidebar.button("🧹 刷新优质股票池", use_container_width=True):
    p = subprocess.run(f"python3 {FILTER_SCRIPT}", shell=True, capture_output=True, text=True) if os.path.exists(FILTER_SCRIPT) else None
    if p and p.returncode == 0:
        st.sidebar.success("完成")
    else:
        st.sidebar.error("失败或找不到脚本")
    st.rerun()

uploaded = st.sidebar.file_uploader("上传 latest_stock_signals.csv", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔍 筛选")
only_clean = st.sidebar.checkbox("只看优质股票池", False)
min_model = st.sidebar.slider("最低模型分", 0, 100, 0)
min_rainbow = st.sidebar.slider("最低趋势分", 0, 100, 0)
min_force = st.sidebar.slider("最低主力分", 0, 100, 0)
keyword = st.sidebar.text_input("搜索股票/行业/题材", "")

raw = pd.read_csv(uploaded, dtype=str) if uploaded is not None else safe_read(SIGNAL_PATH)
source = "上传文件" if uploaded is not None else SIGNAL_PATH
if raw is None or raw.empty:
    st.error("找不到 output/latest_stock_signals.csv，请先运行 train_model.py")
    st.stop()

signal_df = prepare_signal_df(raw)
yaogu_raw = safe_read(YAOGU_PATH)
yaogu_df = prepare_yaogu_df(yaogu_raw) if yaogu_raw is not None and not yaogu_raw.empty else pd.DataFrame()

filtered = signal_df.copy()
filtered = filtered[(filtered.get("model_score", 0) >= min_model) & (filtered.get("rainbow_score", 0) >= min_rainbow) & (filtered.get("main_force_score", 0) >= min_force)]
if only_clean and "is_clean_pool" in filtered.columns:
    filtered = filtered[filtered["is_clean_pool"]]
if keyword.strip():
    k = keyword.strip()
    mask = False
    for c in ["display_name", "industry", "board", "themes", "symbol"]:
        if c in filtered.columns:
            mask = mask | filtered[c].astype(str).str.contains(k, na=False)
    filtered = filtered[mask]

last_date = signal_df["date"].max() if "date" in signal_df.columns else ""
try:
    last_date_text = pd.to_datetime(last_date).strftime("%Y-%m-%d")
except Exception:
    last_date_text = str(last_date)
emotion = compute_market_emotion(signal_df)

st.markdown(f"""
<div class='hero'><h1>🚀 A股交易终端 V5.1 行业修复版</h1>
<p>数据来源：{source} ｜ 最新交易日：{last_date_text} ｜ 预测存档 + 次日复盘 + 市场情绪 ｜ 不构成买卖建议</p></div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card("市场阶段", emotion["market_phase"], emotion["advice"], "hot")
with c2:
    metric_card("情绪分", emotion["emotion_score"], "模型/趋势/主力综合", "gold")
with c3:
    metric_card("强候选", emotion["strong_count"], "模型趋势主力共振", "good")
with c4:
    metric_card("风险票", emotion["risk_count"], "偏空或大跌", "hot")
with c5:
    metric_card("最热行业", emotion["top_industry"], "按平均模型分", "purple")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏆 今日作战", "📊 主线热度", "💾 存档复盘", "🧠 核心模块", "🔥 妖股V3", "📋 全部数据"])
with tab1:
    st.subheader("🏆 Top3 核心作战卡")
    top = choose_recommendations(filtered, yaogu_df if not yaogu_df.empty else None, 3, only_clean)
    if top.empty:
        st.info("暂无推荐")
    else:
        cols = st.columns(min(3, len(top)))
        for i, (_, row) in enumerate(top.iterrows()):
            with cols[i]:
                top_card(row, i + 1)
    st.subheader("📈 Top 15 模型分")
    chart = filtered.sort_values("model_score", ascending=False).head(15) if "model_score" in filtered.columns else filtered.head(15)
    if not chart.empty and "model_score" in chart.columns:
        st.bar_chart(chart.set_index("display_name")["model_score"])

with tab2:
    st.subheader("🔥 行业/板块热度")
    heat = industry_heatmap_data(signal_df, 30)
    if heat.empty:
        st.info("缺少 industry 字段")
    else:
        st.dataframe(heat, use_container_width=True, hide_index=True)
        st.bar_chart(heat.set_index("industry")["热度分"])

with tab3:
    st.subheader("💾 今日预测存档")
    top_n = st.slider("存档推荐数量", 3, 100, 30)
    if st.button("保存今日推荐", use_container_width=True):
        path, rec = archive_today_predictions(signal_df, yaogu_df if not yaogu_df.empty else None, top_n, only_clean)
        st.success(f"已保存：{path}")
        st.dataframe(display_table(rec), use_container_width=True, hide_index=True)
    st.subheader("✅ 次日复盘验证")
    files = list_prediction_files()
    if not files:
        st.info("暂无历史推荐文件，先保存今日推荐。")
    else:
        selected = st.selectbox("选择要验证的推荐文件", files, format_func=os.path.basename)
        if st.button("用当前行情验证", use_container_width=True):
            out, result = review_prediction_file(selected, signal_df)
            s = summarize_review(result)
            a, b, c, d, e = st.columns(5)
            with a:
                metric_card("样本数", s["total"], "验证股票数")
            with b:
                metric_card("命中率", f'{s["hit_rate"]:.2f}%', "收益>0")
            with c:
                metric_card("平均收益", f'{s["avg_return"]:.2f}%', "按收盘价验证")
            with d:
                metric_card("大赚", s["big_win_count"], ">=5%")
            with e:
                metric_card("大亏", s["big_loss_count"], "<=-5%")
            st.success(f"复盘文件：{out}")
            st.dataframe(display_table(result), use_container_width=True, hide_index=True)
    long = safe_read(REVIEW_RESULT_PATH)
    if long is not None and not long.empty:
        st.subheader("📚 长期复盘记录")
        st.dataframe(display_table(long.tail(300)), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("🧠 核心模块总览")
    b = market_breadth(signal_df)
    a1,a2,a3,a4 = st.columns(4)
    with a1: metric_card("上涨占比", f"{b['上涨占比']:.2f}%", "红盘股票比例", "good")
    with a2: metric_card("强势占比", f"{b['强势占比']:.2f}%", "模型/趋势/主力共振", "gold")
    with a3: metric_card("风险占比", f"{b['风险占比']:.2f}%", "偏空或大跌", "hot")
    with a4: metric_card("平均涨跌", f"{b['平均涨跌幅']:.2f}%", "全池平均", "purple")

    st.markdown("### 🐉 龙头候选识别")
    leader_source = yaogu_df if not yaogu_df.empty else signal_df
    leaders = leading_stock_data(leader_source, 30)
    st.dataframe(display_table(leaders), use_container_width=True, hide_index=True)

    st.markdown("### 🧭 市场风格识别")
    style_df = style_detection(signal_df)
    if not style_df.empty:
        st.dataframe(style_df, use_container_width=True, hide_index=True)
        st.bar_chart(style_df.set_index("style_board")["风格强度"])

    st.markdown("### ⚠️ 风险雷达")
    for level, msg in risk_radar(signal_df):
        if level in ["高风险", "偏空", "趋势差"]:
            st.warning(f"{level}：{msg}")
        else:
            st.success(f"{level}：{msg}")

    long_df = safe_read(REVIEW_RESULT_PATH)
    if long_df is not None and not long_df.empty:
        st.markdown("### 📈 长期复盘：行业 / 分数区间")
        by_ind, by_score = review_group_stats(long_df)
        c_left, c_right = st.columns(2)
        with c_left:
            st.caption("按行业统计")
            if not by_ind.empty:
                st.dataframe(by_ind.head(30), use_container_width=True, hide_index=True)
        with c_right:
            st.caption("按模型分区间统计")
            if not by_score.empty:
                st.dataframe(by_score, use_container_width=True, hide_index=True)

with tab5:
    if yaogu_df.empty:
        st.info("暂无妖股 v3 数据，请先运行 yaogu_model_v3.py")
    else:
        st.dataframe(display_table(yaogu_df), use_container_width=True, hide_index=True)

with tab6:
    st.subheader("全部筛选结果")
    st.dataframe(display_table(filtered), use_container_width=True, hide_index=True)
    clean_df = safe_read(CLEAN_POOL_PATH)
    if clean_df is not None and not clean_df.empty:
        with st.expander("🧹 优质股票池"):
            st.dataframe(merge_meta(clean_df).head(300), use_container_width=True, hide_index=True)
    removed_df = safe_read(REMOVED_POOL_PATH)
    if removed_df is not None and not removed_df.empty:
        with st.expander("🗑️ 被剔除股票"):
            st.dataframe(merge_meta(removed_df).head(300), use_container_width=True, hide_index=True)

st.caption("说明：本系统用于交易研究和复盘，不构成买卖建议。真正能救命的是仓位管理，不是花里胡哨的分数。")
