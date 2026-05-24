import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="A股量化交易驾驶舱 Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# 路径配置：保持你原来的文件名
# =========================
SIGNAL_PATH = "output/latest_stock_signals.csv"
YAOGU_PATH = "output/yaogu_model_v3_signals.csv"
CLEAN_POOL_PATH = "output/clean_stock_pool.csv"
REMOVED_POOL_PATH = "output/removed_stock_pool.csv"
META_PATH = "output/stock_meta_map.csv"

TRAIN_SCRIPT = "train_model.py"
YAOGU_SCRIPT = "yaogu_model_v3.py"
FILTER_SCRIPT = "filter_stock_pool.py"


# =========================
# 全局样式：深色交易终端风格
# =========================
def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #172554 0, #020617 34%, #020617 100%);
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
            border-right: 1px solid rgba(148,163,184,0.18);
        }
        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.25);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.82));
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 28px;
            padding: 26px 30px;
            margin-bottom: 18px;
            box-shadow: 0 24px 80px rgba(0,0,0,0.35);
        }
        .hero h1 {
            margin: 0;
            font-size: 38px;
            line-height: 1.15;
            color: #f8fafc;
        }
        .hero p {
            margin: 10px 0 0 0;
            color: #cbd5e1;
            font-size: 15px;
        }
        .section-title {
            font-size: 22px;
            font-weight: 900;
            margin: 24px 0 12px 0;
            color: #f8fafc;
        }
        .terminal-card {
            background: rgba(15,23,42,0.72);
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 22px;
            padding: 18px 20px;
            min-height: 140px;
            box-shadow: 0 18px 50px rgba(0,0,0,0.28);
        }
        .card-title {
            color: #94a3b8;
            font-size: 13px;
            font-weight: 800;
            letter-spacing: .04em;
            text-transform: uppercase;
        }
        .card-value {
            color: #f8fafc;
            font-size: 30px;
            font-weight: 950;
            margin-top: 8px;
        }
        .card-note {
            color: #cbd5e1;
            font-size: 13px;
            margin-top: 8px;
        }
        .battle-card {
            background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(2,6,23,0.96));
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 24px;
            padding: 18px 18px;
            min-height: 380px;
            box-shadow: 0 22px 70px rgba(0,0,0,0.35);
        }
        .battle-rank {
            color: #facc15;
            font-weight: 950;
            font-size: 15px;
        }
        .battle-name {
            color: #f8fafc;
            font-weight: 950;
            font-size: 23px;
            margin-top: 6px;
        }
        .battle-meta {
            color: #94a3b8;
            font-size: 13px;
            margin-top: 5px;
        }
        .big-score {
            color: #facc15;
            font-size: 42px;
            font-weight: 1000;
            line-height: 1;
        }
        .badge {
            display: inline-block;
            padding: 5px 9px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 800;
            margin: 4px 4px 0 0;
        }
        .badge-gold { background: rgba(250,204,21,.15); color: #fde68a; border:1px solid rgba(250,204,21,.35); }
        .badge-red { background: rgba(239,68,68,.15); color: #fecaca; border:1px solid rgba(239,68,68,.35); }
        .badge-green { background: rgba(34,197,94,.15); color: #bbf7d0; border:1px solid rgba(34,197,94,.35); }
        .badge-purple { background: rgba(168,85,247,.15); color: #e9d5ff; border:1px solid rgba(168,85,247,.35); }
        .risk-box {
            background: rgba(127,29,29,0.18);
            border: 1px solid rgba(248,113,113,0.25);
            border-radius: 18px;
            padding: 14px 16px;
            color: #fecaca;
        }
        .ok-box {
            background: rgba(20,83,45,0.18);
            border: 1px solid rgba(74,222,128,0.25);
            border-radius: 18px;
            padding: 14px 16px;
            color: #bbf7d0;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# 基础工具函数
# =========================
@st.cache_data(show_spinner=False)
def safe_read(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk", dtype=str)


def normalize(df):
    df = df.copy()
    if "symbol" not in df.columns:
        if "code" in df.columns:
            df["symbol"] = df["code"]
        elif "股票代码" in df.columns:
            df["symbol"] = df["股票代码"]
        else:
            st.error(f"文件缺少 symbol/code 列，当前列名：{list(df.columns)}")
            st.stop()

    df["symbol"] = (
        df["symbol"]
        .astype(str)
        .str.replace("sh.", "", regex=False)
        .str.replace("sz.", "", regex=False)
        .str.replace(".0", "", regex=False)
        .str.zfill(6)
    )
    return df


def to_number(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def merge_meta(df):
    df = normalize(df)
    meta = safe_read(META_PATH)

    if meta is None or meta.empty:
        df["stock_name"] = ""
        df["industry"] = "未知"
        df["board"] = "未知"
        df["themes"] = "其他"
        df["display_name"] = df["symbol"]
        return df

    meta = normalize(meta)
    keep_cols = [c for c in ["symbol", "stock_name", "industry", "board", "themes"] if c in meta.columns]
    meta = meta[keep_cols].drop_duplicates("symbol")
    df = df.merge(meta, on="symbol", how="left")

    for col, default in [("stock_name", ""), ("industry", "未知"), ("board", "未知"), ("themes", "其他")]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    df["display_name"] = df["symbol"] + " " + df["stock_name"].astype(str)
    return df


def load_clean_symbols():
    clean_df = safe_read(CLEAN_POOL_PATH)
    if clean_df is None or clean_df.empty:
        return set()
    clean_df = normalize(clean_df)
    return set(clean_df["symbol"].tolist())


def get_num(row, col, default=0):
    try:
        value = row.get(col, default)
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def format_percent(x):
    try:
        x = float(x)
        if pd.isna(x):
            return "0.0%"
        if x <= 1:
            x *= 100
        return f"{x:.1f}%"
    except Exception:
        return "0.0%"


def date_text(value):
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


# =========================
# 数据准备
# =========================
def prepare_signal_df(df):
    df = merge_meta(df)
    numeric_cols = [
        "close", "pct_change", "rainbow_score", "main_force_score", "pred_label",
        "prob_up", "prob_down", "model_score", "turnover",
    ]
    df = to_number(df, numeric_cols)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    clean_symbols = load_clean_symbols()
    df["is_clean_pool"] = df["symbol"].isin(clean_symbols) if clean_symbols else False

    def status(row):
        model_score = get_num(row, "model_score")
        prob_up = get_num(row, "prob_up")
        prob_down = get_num(row, "prob_down")
        rainbow = get_num(row, "rainbow_score")
        main_force = get_num(row, "main_force_score")
        pred = get_num(row, "pred_label")
        pct = get_num(row, "pct_change")

        if model_score >= 70 and prob_up >= 0.62 and rainbow >= 80 and main_force >= 65 and pred == 1:
            return "进攻候选"
        if model_score >= 55 and prob_up >= 0.55 and rainbow >= 65:
            return "观察增强"
        if pred == -1 or prob_down >= 0.5 or pct <= -3:
            return "风险"
        if model_score >= 30 and rainbow >= 55:
            return "普通观察"
        return "弱"

    df["signal_status"] = df.apply(status, axis=1)
    return df.sort_values("model_score", ascending=False).reset_index(drop=True)


def prepare_yaogu_df(df):
    df = merge_meta(df)
    numeric_cols = [
        "close", "pct_change", "turnover", "rainbow_score", "main_force_score",
        "model_score", "prob_up", "prob_down", "pred_label", "volume_ratio_v3",
        "pct_5d", "pct_10d", "pct_20d", "yaogu_v3_score", "yaogu_v3_raw_score", "yaogu_v3_risk",
    ]
    df = to_number(df, numeric_cols)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    clean_symbols = load_clean_symbols()
    df["is_clean_pool"] = df["symbol"].isin(clean_symbols) if clean_symbols else False
    if "yaogu_v3_score" in df.columns:
        df = df.sort_values("yaogu_v3_score", ascending=False)
    return df.reset_index(drop=True)


# =========================
# 决策逻辑
# =========================
def calc_market(signal_df):
    total = max(len(signal_df), 1)
    risk_count = int((signal_df["signal_status"] == "风险").sum())
    attack_count = int((signal_df["signal_status"] == "进攻候选").sum())
    observe_count = int((signal_df["signal_status"].isin(["观察增强", "普通观察"])).sum())
    clean_count = int(signal_df["is_clean_pool"].sum())
    avg_score = float(signal_df["model_score"].mean()) if "model_score" in signal_df.columns else 0
    risk_ratio = risk_count / total

    if risk_ratio >= 0.50 or avg_score < 22:
        stage = "防守 / 退潮"
        advice = "控制仓位，少追高，优先等分歧释放。"
        color = "red"
    elif attack_count >= 5 and avg_score >= 50:
        stage = "进攻 / 主升"
        advice = "可重点看强主线，但仍要等盘口确认。"
        color = "green"
    elif attack_count >= 2 or avg_score >= 38:
        stage = "修复 / 试错"
        advice = "小仓位试错，重点看主线持续性。"
        color = "gold"
    else:
        stage = "混沌 / 观察"
        advice = "信号不够一致，先看不动手也算交易。"
        color = "gray"

    return {
        "stage": stage,
        "advice": advice,
        "color": color,
        "risk_count": risk_count,
        "attack_count": attack_count,
        "observe_count": observe_count,
        "clean_count": clean_count,
        "avg_score": avg_score,
        "risk_ratio": risk_ratio,
    }


def build_sector_rank(df, score_col="model_score"):
    if df.empty or "industry" not in df.columns:
        return pd.DataFrame()
    rank = (
        df.groupby("industry", dropna=False)
        .agg(
            平均分=(score_col, "mean"),
            股票数=("symbol", "count"),
            强候选=("signal_status", lambda x: int((x == "进攻候选").sum()) if "signal_status" in df.columns else 0),
            平均涨跌幅=("pct_change", "mean") if "pct_change" in df.columns else ("symbol", "count"),
        )
        .reset_index()
        .rename(columns={"industry": "板块/行业"})
    )
    rank["热度"] = rank["平均分"] * 0.65 + rank["强候选"] * 6 + rank["股票数"].clip(upper=10) * 1.2
    return rank.sort_values("热度", ascending=False).head(15)


def action_text(row):
    score = get_num(row, "yaogu_v3_score", get_num(row, "model_score"))
    risk = get_num(row, "yaogu_v3_risk", 0)
    pct = get_num(row, "pct_change", 0)
    rainbow = get_num(row, "rainbow_score", 0)
    main_force = get_num(row, "main_force_score", 0)
    prob_up = get_num(row, "prob_up", 0)

    if risk >= 25 or pct <= -4:
        return "只观察，不开新仓"
    if score >= 80 and rainbow >= 75 and main_force >= 65 and prob_up >= 0.58:
        return "重点盯竞价，强则低吸"
    if score >= 65:
        return "放入自选，等回踩确认"
    return "普通观察"


def risk_text(row):
    risks = []
    if get_num(row, "yaogu_v3_risk", 0) >= 20:
        risks.append("风险扣分偏高")
    if get_num(row, "pct_change", 0) >= 7:
        risks.append("当日涨幅偏大，别无脑追")
    if get_num(row, "prob_down", 0) >= 0.45:
        risks.append("下跌概率不低")
    if get_num(row, "main_force_score", 0) < 50:
        risks.append("主力行为分不足")
    if not risks:
        risks.append("暂无明显模型风险")
    return "；".join(risks)


# =========================
# 展示组件
# =========================
def render_metric_card(label, value, note, tone="gray"):
    border = {
        "green": "rgba(34,197,94,.38)",
        "red": "rgba(239,68,68,.38)",
        "gold": "rgba(250,204,21,.38)",
        "purple": "rgba(168,85,247,.38)",
        "gray": "rgba(148,163,184,.22)",
    }.get(tone, "rgba(148,163,184,.22)")
    st.markdown(
        f"""
        <div class="terminal-card" style="border-color:{border};">
            <div class="card-title">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_battle_card(row, rank):
    display_name = row.get("display_name", row.get("symbol", ""))
    industry = row.get("industry", "未知")
    board = row.get("board", "未知")
    themes = row.get("themes", "其他")
    date = date_text(row.get("date", ""))

    score = get_num(row, "yaogu_v3_score", get_num(row, "model_score", 0))
    close = get_num(row, "close", 0)
    pct_change = get_num(row, "pct_change", 0)
    rainbow = get_num(row, "rainbow_score", 0)
    main_force = get_num(row, "main_force_score", 0)
    prob_up = row.get("prob_up", 0)
    prob_down = row.get("prob_down", 0)
    level = row.get("yaogu_v3_level", "")
    action = row.get("yaogu_v3_action", action_text(row))
    is_clean = bool(row.get("is_clean_pool", False))

    pct_color = "#fecaca" if pct_change < 0 else "#bbf7d0"
    clean_badge = '<span class="badge badge-green">优质池</span>' if is_clean else '<span class="badge badge-red">非优质池</span>'
    level_badge = f'<span class="badge badge-purple">{level}</span>' if level else ''

    st.markdown(
        f"""
        <div class="battle-card">
            <div class="battle-rank">#{rank} 作战卡</div>
            <div class="battle-name">{display_name}</div>
            <div class="battle-meta">{industry} ｜ {board} ｜ {date}</div>
            <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-top:18px;">
                <div>
                    <div style="color:#94a3b8;font-size:12px;font-weight:800;">综合强度</div>
                    <div class="big-score">{score:.1f}</div>
                </div>
                <div style="text-align:right;">
                    <div style="color:#94a3b8;font-size:12px;">收盘价</div>
                    <div style="color:#f8fafc;font-size:22px;font-weight:900;">{close:.2f}</div>
                    <div style="color:{pct_color};font-size:14px;font-weight:800;">{pct_change:.2f}%</div>
                </div>
            </div>
            <div style="margin-top:12px;">
                {clean_badge}
                <span class="badge badge-gold">{action}</span>
                {level_badge}
            </div>
            <div style="margin-top:15px;color:#cbd5e1;font-size:13px;line-height:1.7;">
                <b>题材：</b>{themes}<br>
                <b>策略：</b>{action_text(row)}<br>
                <b>风险：</b>{risk_text(row)}
            </div>
            <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                <div class="ok-box">上涨概率<br><b>{format_percent(prob_up)}</b></div>
                <div class="risk-box">下跌概率<br><b>{format_percent(prob_down)}</b></div>
            </div>
            <div style="margin-top:12px;color:#94a3b8;font-size:12px;">
                彩虹趋势分：{rainbow:.1f} ｜ 主力行为分：{main_force:.1f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_yaogu_table(df):
    show = df.copy()
    if "date" in show.columns:
        show["date"] = pd.to_datetime(show["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["prob_up", "prob_down"]:
        if col in show.columns:
            show[col] = show[col].apply(format_percent)
    if "is_clean_pool" in show.columns:
        show["is_clean_pool"] = np.where(show["is_clean_pool"], "✅", "")
    round_cols = [
        "close", "pct_change", "turnover", "rainbow_score", "main_force_score", "model_score",
        "volume_ratio_v3", "pct_5d", "pct_10d", "pct_20d", "yaogu_v3_score",
        "yaogu_v3_raw_score", "yaogu_v3_risk",
    ]
    for col in round_cols:
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(2)
    cols = [
        "display_name", "is_clean_pool", "industry", "board", "themes", "date", "close", "pct_change",
        "turnover", "rainbow_score", "main_force_score", "model_score", "prob_up", "prob_down",
        "pred_label", "volume_ratio_v3", "pct_5d", "pct_10d", "pct_20d", "yaogu_v3_score",
        "yaogu_v3_raw_score", "yaogu_v3_risk", "yaogu_v3_level", "yaogu_v3_action", "yaogu_v3_reason",
    ]
    cols = [c for c in cols if c in show.columns]
    rename = {
        "display_name": "股票", "is_clean_pool": "优质池", "industry": "行业", "board": "板块", "themes": "题材",
        "date": "日期", "close": "收盘价", "pct_change": "涨跌幅%", "turnover": "换手率",
        "rainbow_score": "趋势分", "main_force_score": "主力分", "model_score": "模型分",
        "prob_up": "上涨概率", "prob_down": "下跌概率", "pred_label": "预测标签", "volume_ratio_v3": "量比v3",
        "pct_5d": "5日%", "pct_10d": "10日%", "pct_20d": "20日%", "yaogu_v3_score": "妖股v3分",
        "yaogu_v3_raw_score": "原始分", "yaogu_v3_risk": "风险扣分", "yaogu_v3_level": "等级",
        "yaogu_v3_action": "操作建议", "yaogu_v3_reason": "入选原因",
    }
    return show[cols].rename(columns=rename)


def display_signal_table(df):
    show = df.copy()
    if "date" in show.columns:
        show["date"] = pd.to_datetime(show["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["prob_up", "prob_down"]:
        if col in show.columns:
            show[col] = show[col].apply(format_percent)
    if "is_clean_pool" in show.columns:
        show["is_clean_pool"] = np.where(show["is_clean_pool"], "✅", "")
    for col in ["close", "pct_change", "rainbow_score", "main_force_score", "model_score", "turnover"]:
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(2)
    cols = [
        "display_name", "is_clean_pool", "industry", "board", "themes", "date", "close", "pct_change",
        "turnover", "rainbow_score", "main_force_score", "model_score", "prob_up", "prob_down",
        "pred_label", "signal_status",
    ]
    cols = [c for c in cols if c in show.columns]
    rename = {
        "display_name": "股票", "is_clean_pool": "优质池", "industry": "行业", "board": "板块", "themes": "题材",
        "date": "日期", "close": "收盘价", "pct_change": "涨跌幅%", "turnover": "换手率",
        "rainbow_score": "趋势分", "main_force_score": "主力分", "model_score": "模型分",
        "prob_up": "上涨概率", "prob_down": "下跌概率", "pred_label": "预测标签", "signal_status": "状态",
    }
    return show[cols].rename(columns=rename)


# =========================
# 主程序
# =========================
inject_css()

st.sidebar.markdown("## 🚀 交易控制台")

if st.sidebar.button("🔄 一键更新模型 + v3信号", use_container_width=True):
    logs = []
    with st.spinner("正在训练模型并生成 v3 信号..."):
        for name, script in [("模型训练", TRAIN_SCRIPT), ("妖股模型 v3", YAOGU_SCRIPT)]:
            if os.path.exists(script):
                code, out, err = run_command(f"python3 {script}")
                logs.append((name, code, out, err))
            else:
                logs.append((name, 1, "", f"找不到 {script}"))
    with st.expander("查看运行日志", expanded=True):
        for name, code, out, err in logs:
            st.success(f"{name} 完成") if code == 0 else st.error(f"{name} 失败")
            if out:
                st.code(out[-3000:])
            if err:
                st.code(err[-3000:])
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("🧹 刷新优质股票池", use_container_width=True):
    with st.spinner("正在刷新优质股票池..."):
        if os.path.exists(FILTER_SCRIPT):
            code, out, err = run_command(f"python3 {FILTER_SCRIPT}")
        else:
            code, out, err = 1, "", f"找不到 {FILTER_SCRIPT}"
    with st.expander("查看清洗日志", expanded=True):
        st.success("优质股票池刷新完成") if code == 0 else st.error("优质股票池刷新失败")
        if out:
            st.code(out[-3000:])
        if err:
            st.code(err[-3000:])
    st.cache_data.clear()
    st.rerun()

uploaded = st.sidebar.file_uploader("上传 latest_stock_signals.csv", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔍 筛选器")

only_clean_pool = st.sidebar.checkbox("只看优质股票池", value=False)
min_model_score = st.sidebar.slider("最低模型分", 0, 100, 0)
min_rainbow_score = st.sidebar.slider("最低趋势分", 0, 100, 0)
min_main_force_score = st.sidebar.slider("最低主力分", 0, 100, 0)
status_filter = st.sidebar.multiselect(
    "信号状态",
    ["进攻候选", "观察增强", "普通观察", "风险", "弱"],
    default=[],
)
keyword = st.sidebar.text_input("搜索股票/行业/题材", "")

if uploaded is not None:
    raw_signal = pd.read_csv(uploaded, dtype=str)
    data_source = "上传文件"
else:
    raw_signal = safe_read(SIGNAL_PATH)
    data_source = SIGNAL_PATH

if raw_signal is None or raw_signal.empty:
    st.error("找不到 output/latest_stock_signals.csv，请先运行 train_model.py")
    st.stop()

signal_df = prepare_signal_df(raw_signal)

yaogu_raw = safe_read(YAOGU_PATH)
yaogu_df = prepare_yaogu_df(yaogu_raw) if yaogu_raw is not None and not yaogu_raw.empty else pd.DataFrame()

filtered = signal_df.copy()
filtered = filtered[
    (filtered["model_score"] >= min_model_score)
    & (filtered["rainbow_score"] >= min_rainbow_score)
    & (filtered["main_force_score"] >= min_main_force_score)
]
if only_clean_pool:
    filtered = filtered[filtered["is_clean_pool"]]
if status_filter:
    filtered = filtered[filtered["signal_status"].isin(status_filter)]
if keyword.strip():
    key = keyword.strip()
    mask = False
    for col in ["display_name", "industry", "board", "themes", "symbol"]:
        if col in filtered.columns:
            mask = mask | filtered[col].astype(str).str.contains(key, case=False, na=False)
    filtered = filtered[mask]

last_date_text = date_text(signal_df["date"].max()) if "date" in signal_df.columns else "未知"
market = calc_market(signal_df)

st.markdown(
    f"""
    <div class="hero">
        <h1>🚀 A股量化交易驾驶舱 Pro</h1>
        <p>数据来源：{data_source} ｜ 最新交易日：{last_date_text} ｜ 妖股模型 v3 + 优质池过滤 ｜ 交易信号只做辅助，不替你背锅</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    render_metric_card("市场阶段", market["stage"], market["advice"], market["color"])
with m2:
    render_metric_card("平均模型分", f"{market['avg_score']:.2f}", "候选池整体强度", "purple")
with m3:
    render_metric_card("进攻候选", market["attack_count"], "模型+趋势+主力共振", "green")
with m4:
    render_metric_card("风险票", market["risk_count"], f"风险占比 {market['risk_ratio']:.1%}", "red")
with m5:
    render_metric_card("优质池命中", market["clean_count"], "成交额/换手/波动过滤", "gold")

st.markdown('<div class="section-title">🏆 今日核心作战区</div>', unsafe_allow_html=True)
if not yaogu_df.empty:
    top_df = yaogu_df.copy()
    if only_clean_pool:
        top_df = top_df[top_df["is_clean_pool"]]
    top_df = top_df.sort_values("yaogu_v3_score", ascending=False) if "yaogu_v3_score" in top_df.columns else top_df
    top3 = top_df.head(3)
    st.caption("排序依据：妖股 v3 分数优先")
else:
    top3 = filtered.sort_values("model_score", ascending=False).head(3)
    st.caption("排序依据：模型分")

if top3.empty:
    st.info("当前没有可展示的核心票。")
else:
    cols = st.columns(min(3, len(top3)))
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            render_battle_card(row, i + 1)

st.markdown('<div class="section-title">🔥 主线强度与风险雷达</div>', unsafe_allow_html=True)
left, right = st.columns([1.4, 1])

with left:
    sector_rank = build_sector_rank(filtered if not filtered.empty else signal_df)
    st.markdown("#### 板块/行业热度 Top15")
    if sector_rank.empty:
        st.info("暂无板块数据")
    else:
        st.bar_chart(sector_rank.set_index("板块/行业")["热度"])
        st.dataframe(sector_rank.round(2), use_container_width=True, hide_index=True)

with right:
    st.markdown("#### 信号状态分布")
    status_counts = signal_df["signal_status"].value_counts()
    st.bar_chart(status_counts)

    st.markdown("#### 风险提示")
    if market["risk_ratio"] >= 0.5:
        st.markdown('<div class="risk-box">风险票占比过高：今天适合少动，不适合上头。</div>', unsafe_allow_html=True)
    elif market["attack_count"] >= 5:
        st.markdown('<div class="ok-box">进攻候选数量不错：可以重点观察主线持续性。</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-box">市场没有明显合力：先看主线，别急着梭哈。</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">📊 多维分析面板</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["🔥 妖股模型 v3", "📋 全部筛选结果", "🧹 优质股票池", "🗑️ 剔除原因"])

with tab1:
    if yaogu_df.empty:
        st.info("暂无妖股 v3 数据，请先运行 yaogu_model_v3.py")
    else:
        yaogu_show = yaogu_df.copy()
        if only_clean_pool:
            yaogu_show = yaogu_show[yaogu_show["is_clean_pool"]]
        st.dataframe(display_yaogu_table(yaogu_show), use_container_width=True, hide_index=True, height=560)

with tab2:
    st.dataframe(display_signal_table(filtered), use_container_width=True, hide_index=True, height=620)

with tab3:
    clean_df = safe_read(CLEAN_POOL_PATH)
    if clean_df is None or clean_df.empty:
        st.info("暂无 clean_stock_pool.csv，请先运行 filter_stock_pool.py")
    else:
        clean_df = merge_meta(clean_df)
        st.dataframe(clean_df.head(300), use_container_width=True, hide_index=True, height=560)

with tab4:
    removed_df = safe_read(REMOVED_POOL_PATH)
    if removed_df is None or removed_df.empty:
        st.info("暂无 removed_stock_pool.csv")
    else:
        removed_df = merge_meta(removed_df)
        cols = ["display_name", "industry", "board", "themes", "remove_reason"]
        cols = [c for c in cols if c in removed_df.columns]
        st.dataframe(removed_df[cols], use_container_width=True, hide_index=True, height=560)

st.caption("说明：本系统用于复盘、筛选和辅助决策，不构成买卖建议。真正下单前，看竞价、成交量、板块联动和风险承受能力。")
