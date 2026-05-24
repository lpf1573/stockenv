"""
app_v6_12345.py

A股量化交易终端 V6：核心模块增强版

新增：
1. 一键更新流水线
2. 主线识别系统
3. 龙头识别系统
4. 连板梯队雏形
5. 自动交易日报

运行：
    streamlit run app.py

建议：
    下载后改名为 app.py，替换旧版。
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# 基础配置
# =========================

st.set_page_config(
    page_title="A股量化交易终端 V6",
    page_icon="🚀",
    layout="wide",
)

OUTPUT_DIR = Path("output")
HISTORY_DIR = Path("history")
REPORT_DIR = Path("reports")
DAILY_SIGNAL_DIR = HISTORY_DIR / "daily_signals"
REVIEW_PATH = HISTORY_DIR / "prediction_review.csv"

SIGNAL_PATH = OUTPUT_DIR / "latest_stock_signals.csv"
YAOGU_PATH = OUTPUT_DIR / "yaogu_model_v3_signals.csv"
CLEAN_POOL_PATH = OUTPUT_DIR / "clean_stock_pool.csv"
META_PATH = OUTPUT_DIR / "stock_meta_map.csv"
TRADE_PLAN_PATH = OUTPUT_DIR / "trade_plan.csv"

UPDATE_SCRIPT = "update_market_dataset.py"
TRAIN_SCRIPT = "train_model.py"
YAOGU_SCRIPT = "yaogu_model_v3.py"
FILTER_SCRIPT = "filter_stock_pool.py"


# =========================
# 样式
# =========================

def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:#070b12;
            color:#e5e7eb;
        }
        [data-testid="stSidebar"] {
            background:#0b1220;
        }
        .hero {
            background:linear-gradient(135deg,#111827,#1e1b4b,#312e81);
            border:1px solid #334155;
            border-radius:28px;
            padding:30px 34px;
            margin-bottom:22px;
            box-shadow:0 18px 45px rgba(0,0,0,0.35);
        }
        .hero-title {
            font-size:40px;
            font-weight:950;
            color:#f8fafc;
        }
        .hero-sub {
            color:#cbd5e1;
            margin-top:10px;
            font-size:15px;
        }
        .metric-card {
            background:#0f172a;
            border:1px solid #26344f;
            border-radius:22px;
            padding:22px 24px;
            min-height:145px;
            box-shadow:0 10px 30px rgba(0,0,0,0.28);
        }
        .metric-label {
            color:#94a3b8;
            font-weight:800;
            font-size:15px;
        }
        .metric-value {
            font-size:34px;
            font-weight:950;
            margin-top:12px;
        }
        .metric-note {
            color:#94a3b8;
            margin-top:10px;
            font-size:14px;
        }
        .battle-card {
            background:#0f172a;
            border:1px solid #2f4265;
            border-radius:24px;
            padding:26px 28px;
            min-height:330px;
            box-shadow:0 14px 38px rgba(0,0,0,0.33);
        }
        .rank {
            color:#94a3b8;
            font-weight:900;
        }
        .stock-title {
            color:#fbbf24;
            font-size:30px;
            font-weight:950;
            margin:16px 0 14px 0;
        }
        .section-card {
            background:#0f172a;
            border:1px solid #26344f;
            border-radius:22px;
            padding:20px 22px;
            margin-bottom:16px;
        }
        .muted {
            color:#94a3b8;
        }
        .score {
            color:#fb7185;
            font-weight:900;
        }
        .good {
            color:#4ade80;
            font-weight:900;
        }
        .bad {
            color:#fb7185;
            font-weight:900;
        }
        .warn {
            color:#facc15;
            font-weight:900;
        }
        hr {
            border-color:#1f2937;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =========================
# 通用函数
# =========================

def safe_read(path):
    path = Path(path)
    if not path.exists():
        return None
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc, dtype=str)
        except Exception:
            pass
    try:
        return pd.read_csv(path, dtype=str)
    except Exception:
        return None


def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def normalize_symbol(s):
    s = str(s)
    s = s.replace("sh.", "").replace("sz.", "")
    s = s.replace(".SH", "").replace(".SZ", "")
    s = s.replace(".0", "")
    return s.zfill(6)[-6:]


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize(df):
    df = df.copy()
    col = pick_col(df, ["symbol", "code", "股票代码", "证券代码", "ts_code"])
    if col is None:
        st.error(f"文件缺少股票代码列，当前列名：{list(df.columns)}")
        st.stop()
    df["symbol"] = df[col].apply(normalize_symbol)
    return df


def to_number(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def merge_meta(df):
    df = normalize(df)

    rename = {}
    name_col = pick_col(df, ["stock_name", "股票名称", "名称", "name", "证券简称"])
    industry_col = pick_col(df, ["industry", "所属行业", "申万行业", "行业", "sw_industry"])
    board_col = pick_col(df, ["board", "板块", "市场板块", "上市板", "market"])
    themes_col = pick_col(df, ["themes", "概念板块", "题材", "概念", "concepts"])

    if name_col and name_col != "stock_name":
        rename[name_col] = "stock_name"
    if industry_col and industry_col != "industry":
        rename[industry_col] = "industry"
    if board_col and board_col != "board":
        rename[board_col] = "board"
    if themes_col and themes_col != "themes":
        rename[themes_col] = "themes"

    if rename:
        df = df.rename(columns=rename)

    meta = safe_read(META_PATH)
    if meta is not None and not meta.empty:
        meta = normalize(meta)

        mrename = {}
        m_name = pick_col(meta, ["stock_name", "股票名称", "名称", "name", "证券简称"])
        m_industry = pick_col(meta, ["industry", "所属行业", "申万行业", "行业", "sw_industry"])
        m_board = pick_col(meta, ["board", "板块", "市场板块", "上市板", "market"])
        m_themes = pick_col(meta, ["themes", "概念板块", "题材", "概念", "concepts"])

        if m_name and m_name != "stock_name":
            mrename[m_name] = "stock_name"
        if m_industry and m_industry != "industry":
            mrename[m_industry] = "industry"
        if m_board and m_board != "board":
            mrename[m_board] = "board"
        if m_themes and m_themes != "themes":
            mrename[m_themes] = "themes"

        if mrename:
            meta = meta.rename(columns=mrename)

        keep = [c for c in ["symbol", "stock_name", "industry", "board", "themes"] if c in meta.columns]
        meta = meta[keep].drop_duplicates("symbol")

        df = df.merge(meta, on="symbol", how="left", suffixes=("", "_meta"))

        for col in ["stock_name", "industry", "board", "themes"]:
            meta_col = f"{col}_meta"
            if col not in df.columns:
                df[col] = np.nan
            if meta_col in df.columns:
                bad = df[col].isna() | df[col].astype(str).isin(["", "未知", "其他", "nan", "None"])
                df.loc[bad, col] = df.loc[bad, meta_col]
                df = df.drop(columns=[meta_col])

    defaults = {
        "stock_name": "",
        "industry": "其他",
        "board": "未知",
        "themes": "其他",
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str)
        df.loc[df[col].isin(["", "nan", "None"]), col] = default

    df["display_name"] = df["symbol"] + " " + df["stock_name"].astype(str)
    return df


def load_clean_symbols():
    clean = safe_read(CLEAN_POOL_PATH)
    if clean is None or clean.empty:
        return set()
    clean = normalize(clean)
    return set(clean["symbol"].tolist())


def get_num(row, col, default=0):
    try:
        v = row.get(col, default)
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def format_percent(x):
    try:
        x = float(x)
        if pd.isna(x):
            return "0.0%"
        if abs(x) <= 1:
            x *= 100
        return f"{x:.1f}%"
    except Exception:
        return "0.0%"


# =========================
# 数据准备
# =========================

def prepare_signal_df(raw):
    df = merge_meta(raw)
    num_cols = [
        "close", "pct_change", "rainbow_score", "main_force_score",
        "pred_label", "prob_up", "prob_down", "model_score",
        "turnover", "volume_ratio_v3", "pct_5d", "pct_10d", "pct_20d",
        "limit_up_days", "连板数", "涨停天数",
    ]
    df = to_number(df, num_cols)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.Timestamp.today().normalize()

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

        if model_score >= 60 and prob_up >= 0.6 and rainbow >= 80 and main_force >= 60 and pred == 1:
            return "强买观察"
        if pred == -1 or prob_down >= 0.5 or pct <= -3:
            return "风险"
        if model_score >= 30 and rainbow >= 60:
            return "观察"
        return "弱"

    df["signal_status"] = df.apply(status, axis=1)
    return df.sort_values("model_score", ascending=False).reset_index(drop=True)


def prepare_yaogu_df(raw):
    df = merge_meta(raw)
    num_cols = [
        "close", "pct_change", "turnover", "rainbow_score", "main_force_score",
        "model_score", "prob_up", "prob_down", "pred_label",
        "volume_ratio_v3", "pct_5d", "pct_10d", "pct_20d",
        "yaogu_v3_score", "yaogu_v3_raw_score", "yaogu_v3_risk",
        "limit_up_days", "连板数", "涨停天数",
    ]
    df = to_number(df, num_cols)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.Timestamp.today().normalize()

    clean_symbols = load_clean_symbols()
    df["is_clean_pool"] = df["symbol"].isin(clean_symbols) if clean_symbols else False

    if "yaogu_v3_score" in df.columns:
        df = df.sort_values("yaogu_v3_score", ascending=False)

    return df.reset_index(drop=True)


def build_core_score(df):
    df = df.copy()
    for c in ["model_score", "rainbow_score", "main_force_score", "pct_change", "yaogu_v3_score"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["core_score"] = (
        df["model_score"].clip(0, 100) * 0.30
        + df["rainbow_score"].clip(0, 100) * 0.25
        + df["main_force_score"].clip(0, 100) * 0.20
        + df["yaogu_v3_score"].clip(0, 100) * 0.20
        + df["pct_change"].clip(-10, 20) * 0.05
    )
    return df.sort_values("core_score", ascending=False).reset_index(drop=True)


# =========================
# 核心模块 1：一键流水线
# =========================

def run_pipeline(run_update=True, run_train=True, run_yaogu=True, run_filter=False):
    jobs = []
    if run_update:
        jobs.append(("更新行情", UPDATE_SCRIPT))
    if run_train:
        jobs.append(("训练模型", TRAIN_SCRIPT))
    if run_yaogu:
        jobs.append(("生成妖股V3", YAOGU_SCRIPT))
    if run_filter:
        jobs.append(("刷新优质池", FILTER_SCRIPT))

    logs = []
    for name, script in jobs:
        if not os.path.exists(script):
            logs.append((name, 1, "", f"找不到 {script}"))
            continue
        code, out, err = run_command(f"python3 {script}")
        logs.append((name, code, out, err))
    return logs


# =========================
# 核心模块 2：主线识别
# =========================

def identify_mainlines(df):
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    for c in ["model_score", "rainbow_score", "main_force_score", "pct_change", "yaogu_v3_score", "core_score"]:
        if c not in work.columns:
            work[c] = 0
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)

    group_col = "industry"
    grouped = work.groupby(group_col, dropna=False).agg(
        股票数量=("symbol", "count"),
        平均综合分=("core_score", "mean"),
        平均模型分=("model_score", "mean"),
        平均趋势分=("rainbow_score", "mean"),
        平均主力分=("main_force_score", "mean"),
        平均涨跌幅=("pct_change", "mean"),
        强候选数=("signal_status", lambda s: (s == "强买观察").sum() if s is not None else 0),
        风险数=("signal_status", lambda s: (s == "风险").sum() if s is not None else 0),
    ).reset_index().rename(columns={group_col: "主线"})

    grouped["主线强度"] = (
        grouped["平均综合分"] * 0.45
        + grouped["平均趋势分"] * 0.20
        + grouped["平均主力分"] * 0.20
        + grouped["强候选数"] * 3
        - grouped["风险数"] * 1.5
    )

    def stage(row):
        if row["主线强度"] >= 75 and row["强候选数"] >= 2:
            return "主升"
        if row["主线强度"] >= 55:
            return "加强"
        if row["主线强度"] >= 40:
            return "轮动"
        return "弱"

    grouped["状态"] = grouped.apply(stage, axis=1)
    grouped = grouped.sort_values("主线强度", ascending=False)
    return grouped.round(2)


# =========================
# 核心模块 3：龙头识别
# =========================

def identify_leaders(df):
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    for c in ["core_score", "model_score", "rainbow_score", "main_force_score", "pct_change", "yaogu_v3_score"]:
        if c not in work.columns:
            work[c] = 0
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)

    work["leader_score"] = (
        work["core_score"] * 0.45
        + work["yaogu_v3_score"] * 0.25
        + work["main_force_score"] * 0.15
        + work["pct_change"].clip(-10, 20) * 0.15
    )

    def leader_type(row):
        symbol = str(row.get("symbol", ""))
        pct = get_num(row, "pct_change")
        yaogu = get_num(row, "yaogu_v3_score")
        main_force = get_num(row, "main_force_score")
        rainbow = get_num(row, "rainbow_score")

        if symbol.startswith("688") or symbol.startswith("300"):
            if yaogu >= 80:
                return "20cm妖股候选"
            return "20cm趋势候选"
        if pct >= 9.5:
            return "涨停龙头候选"
        if rainbow >= 85 and main_force >= 75:
            return "趋势龙头候选"
        return "核心候选"

    work["龙头类型"] = work.apply(leader_type, axis=1)
    return work.sort_values("leader_score", ascending=False).reset_index(drop=True)


# =========================
# 核心模块 4：连板梯队雏形
# =========================

def build_limit_up_ladder(df):
    if df.empty:
        return pd.DataFrame()

    work = df.copy()

    # 如果数据里有真实连板字段，优先使用；否则用涨跌幅估算
    ladder_col = None
    for c in ["limit_up_days", "连板数", "涨停天数"]:
        if c in work.columns:
            ladder_col = c
            break

    if ladder_col:
        work["ladder"] = pd.to_numeric(work[ladder_col], errors="coerce").fillna(0).astype(int)
    else:
        pct = pd.to_numeric(work.get("pct_change", 0), errors="coerce").fillna(0)
        work["ladder"] = np.where(pct >= 19.5, 1, np.where(pct >= 9.5, 1, 0))

    hot = work[work["ladder"] > 0].copy()
    if hot.empty:
        return pd.DataFrame(columns=["梯队", "股票数量", "代表股票"])

    out = (
        hot.groupby("ladder")
        .agg(
            股票数量=("symbol", "count"),
            代表股票=("display_name", lambda s: "、".join(s.astype(str).head(8))),
        )
        .reset_index()
        .sort_values("ladder", ascending=False)
    )
    out["梯队"] = out["ladder"].astype(int).astype(str) + "板"
    return out[["梯队", "股票数量", "代表股票"]]


# =========================
# 核心模块 5：自动日报
# =========================

def generate_daily_report(signal_df, leader_df, mainline_df, ladder_df, emotion):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    date_text = pd.to_datetime(signal_df["date"].max()).strftime("%Y-%m-%d") if not signal_df.empty else datetime.now().strftime("%Y-%m-%d")
    path_md = REPORT_DIR / f"{date_text}_daily_report.md"

    top_mainlines = mainline_df.head(5) if not mainline_df.empty else pd.DataFrame()
    top_leaders = leader_df.head(10) if not leader_df.empty else pd.DataFrame()

    lines = []
    lines.append(f"# A股交易日报 - {date_text}")
    lines.append("")
    lines.append("## 一、市场状态")
    lines.append(f"- 市场阶段：{emotion.get('stage', '未知')}")
    lines.append(f"- 情绪分：{emotion.get('score', 0)}")
    lines.append(f"- 强候选数量：{emotion.get('strong', 0)}")
    lines.append(f"- 风险票数量：{emotion.get('risk', 0)}")
    lines.append(f"- 最热行业：{emotion.get('hot_industry', '未知')}")
    lines.append(f"- 建议：{emotion.get('advice', '')}")
    lines.append("")

    lines.append("## 二、主线方向")
    if top_mainlines.empty:
        lines.append("- 暂无主线数据")
    else:
        for _, r in top_mainlines.iterrows():
            lines.append(
                f"- {r['主线']}：状态={r['状态']}，主线强度={r['主线强度']}，"
                f"强候选={r['强候选数']}，平均涨跌幅={r['平均涨跌幅']}%"
            )
    lines.append("")

    lines.append("## 三、龙头候选")
    if top_leaders.empty:
        lines.append("- 暂无龙头候选")
    else:
        for _, r in top_leaders.iterrows():
            lines.append(
                f"- {r.get('display_name','')}：{r.get('龙头类型','')}，"
                f"leader_score={get_num(r, 'leader_score'):.2f}，"
                f"行业={r.get('industry','')}，涨跌幅={get_num(r, 'pct_change'):.2f}%"
            )
    lines.append("")

    lines.append("## 四、连板梯队")
    if ladder_df.empty:
        lines.append("- 暂无涨停/连板数据；如果需要精确连板，需要接入涨停板历史数据。")
    else:
        for _, r in ladder_df.iterrows():
            lines.append(f"- {r['梯队']}：{r['股票数量']}只，代表：{r['代表股票']}")
    lines.append("")

    lines.append("## 五、明日观察")
    lines.append("- 优先观察主线强度是否延续。")
    lines.append("- 高分个股只做低吸观察，不追一致高开。")
    lines.append("- 如果风险票明显增加，降低仓位。")
    lines.append("")
    lines.append("> 仅用于研究和复盘，不构成投资建议。")

    path_md.write_text("\n".join(lines), encoding="utf-8")
    return path_md


# =========================
# 复盘存档
# =========================

def save_today_predictions(df, limit=30):
    DAILY_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return None
    last_date = pd.to_datetime(df["date"].max()).strftime("%Y-%m-%d")
    path = DAILY_SIGNAL_DIR / f"{last_date}_predictions.csv"
    df.head(limit).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def available_archives():
    if not DAILY_SIGNAL_DIR.exists():
        return []
    return sorted(DAILY_SIGNAL_DIR.glob("*_predictions.csv"), reverse=True)


def review_archive(archive_path, current_df):
    old = safe_read(archive_path)
    if old is None or old.empty or current_df.empty:
        return pd.DataFrame()

    old = merge_meta(old)
    cur = normalize(current_df)
    cur = to_number(cur, ["close", "pct_change"])

    old = to_number(old, ["close", "model_score", "rainbow_score", "main_force_score", "yaogu_v3_score", "core_score"])
    cur_keep = [c for c in ["symbol", "close", "pct_change", "date"] if c in cur.columns]
    cur = cur[cur_keep].drop_duplicates("symbol", keep="last")

    merged = old.merge(cur, on="symbol", how="left", suffixes=("_pred", "_now"))

    if "close_pred" in merged.columns and "close_now" in merged.columns:
        merged["next_return_pct"] = (
            pd.to_numeric(merged["close_now"], errors="coerce")
            / pd.to_numeric(merged["close_pred"], errors="coerce")
            - 1
        ) * 100
    else:
        merged["next_return_pct"] = np.nan

    merged["hit"] = merged["next_return_pct"] > 0
    merged["big_win"] = merged["next_return_pct"] >= 5
    merged["big_loss"] = merged["next_return_pct"] <= -5

    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    old_date = archive_path.name.replace("_predictions.csv", "")
    merged["prediction_date"] = old_date
    merged["review_date"] = datetime.now().strftime("%Y-%m-%d")
    merged.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")

    return merged


# =========================
# 市场情绪
# =========================

def market_emotion(df):
    if df.empty:
        return {
            "stage": "无数据",
            "score": 0,
            "strong": 0,
            "risk": 0,
            "hot_industry": "无",
            "advice": "请先生成信号数据。",
        }

    avg_model = float(df["model_score"].mean()) if "model_score" in df.columns else 0
    avg_rainbow = float(df["rainbow_score"].mean()) if "rainbow_score" in df.columns else 0
    avg_main = float(df["main_force_score"].mean()) if "main_force_score" in df.columns else 0

    strong = int((df["signal_status"] == "强买观察").sum()) if "signal_status" in df.columns else 0
    risk = int((df["signal_status"] == "风险").sum()) if "signal_status" in df.columns else 0

    score = round(avg_model * 0.45 + avg_rainbow * 0.35 + avg_main * 0.20, 2)

    if score >= 65 and strong >= 5:
        stage = "进攻期"
        advice = "可以关注核心主线低吸机会，但不能追一致高潮。"
    elif score >= 45:
        stage = "修复/观察"
        advice = "轻仓试错，等待板块和个股共振。"
    else:
        stage = "防守期"
        advice = "控制仓位，优先保护本金。"

    hot_industry = "其他"
    if "industry" in df.columns and "model_score" in df.columns:
        tmp = df.groupby("industry")["model_score"].mean().sort_values(ascending=False)
        if not tmp.empty:
            hot_industry = str(tmp.index[0])

    return {
        "stage": stage,
        "score": score,
        "strong": strong,
        "risk": risk,
        "hot_industry": hot_industry,
        "advice": advice,
    }


# =========================
# 展示函数
# =========================

def render_metric_card(label, value, note, color="#f8fafc"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_battle_card(row, rank):
    name = row.get("display_name", row.get("symbol", ""))
    industry = row.get("industry", "其他")
    board = row.get("board", "未知")
    themes = row.get("themes", "其他")

    score = get_num(row, "core_score")
    leader_score = get_num(row, "leader_score")
    close = get_num(row, "close")
    pct = get_num(row, "pct_change")
    rainbow = get_num(row, "rainbow_score")
    main_force = get_num(row, "main_force_score")
    model_score = get_num(row, "model_score")
    yaogu_score = get_num(row, "yaogu_v3_score")
    action = row.get("yaogu_v3_action", row.get("signal_status", "观察"))
    leader_type = row.get("龙头类型", "核心候选")

    st.markdown(
        f"""
        <div class="battle-card">
            <div class="rank">#{rank} 核心作战卡 ｜ {leader_type}</div>
            <div class="stock-title">{name}</div>
            <div class="muted">行业：{industry} ｜ 板块：{board}</div>
            <hr>
            <div class="muted">综合分：<span class="score">{score:.2f}</span> ｜ 龙头分：<span class="warn">{leader_score:.2f}</span></div>
            <div class="muted" style="margin-top:10px;">模型：{model_score:.2f} ｜ 妖股：{yaogu_score:.2f}</div>
            <div class="muted" style="margin-top:10px;">收盘：{close:.2f} ｜ 涨跌幅：<span class="good">{pct:.2f}%</span></div>
            <div class="muted" style="margin-top:10px;">趋势：{rainbow:.1f} ｜ 主力：{main_force:.1f}</div>
            <div class="muted" style="margin-top:10px;">题材：{themes}</div>
            <div class="muted" style="margin-top:14px;">建议：<span class="warn">{action}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_table(df):
    show = df.copy()
    if "date" in show.columns:
        show["date"] = pd.to_datetime(show["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["prob_up", "prob_down"]:
        if col in show.columns:
            show[col] = show[col].apply(format_percent)
    if "is_clean_pool" in show.columns:
        show["is_clean_pool"] = np.where(show["is_clean_pool"], "✅", "")

    round_cols = [
        "close", "pct_change", "turnover", "rainbow_score", "main_force_score",
        "model_score", "volume_ratio_v3", "pct_5d", "pct_10d", "pct_20d",
        "yaogu_v3_score", "yaogu_v3_raw_score", "yaogu_v3_risk",
        "core_score", "leader_score", "next_return_pct",
    ]
    for col in round_cols:
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(2)

    cols = [
        "display_name", "is_clean_pool", "龙头类型", "industry", "board", "themes", "date",
        "close", "pct_change", "turnover", "rainbow_score", "main_force_score",
        "model_score", "core_score", "leader_score",
        "prob_up", "prob_down", "pred_label", "signal_status",
        "volume_ratio_v3", "pct_5d", "pct_10d", "pct_20d",
        "yaogu_v3_score", "yaogu_v3_raw_score", "yaogu_v3_risk",
        "yaogu_v3_level", "yaogu_v3_action", "yaogu_v3_reason",
        "next_return_pct", "hit", "big_win", "big_loss",
    ]
    cols = [c for c in cols if c in show.columns]

    rename = {
        "display_name": "股票",
        "is_clean_pool": "优质池",
        "industry": "行业",
        "board": "板块",
        "themes": "题材",
        "date": "日期",
        "close": "收盘价",
        "pct_change": "涨跌幅%",
        "turnover": "换手率",
        "rainbow_score": "趋势分",
        "main_force_score": "主力分",
        "model_score": "模型分",
        "core_score": "综合分",
        "leader_score": "龙头分",
        "prob_up": "上涨概率",
        "prob_down": "下跌概率",
        "pred_label": "预测标签",
        "signal_status": "状态",
        "volume_ratio_v3": "量比v3",
        "pct_5d": "5日涨跌%",
        "pct_10d": "10日涨跌%",
        "pct_20d": "20日涨跌%",
        "yaogu_v3_score": "妖股v3分",
        "yaogu_v3_raw_score": "原始分",
        "yaogu_v3_risk": "风险扣分",
        "yaogu_v3_level": "等级",
        "yaogu_v3_action": "操作建议",
        "yaogu_v3_reason": "入选原因",
        "next_return_pct": "验证收益%",
        "hit": "命中",
        "big_win": "大赚",
        "big_loss": "大亏",
    }
    return show[cols].rename(columns=rename)


# =========================
# 侧边栏
# =========================

st.sidebar.markdown("## 🚀 V6 控制台")

with st.sidebar.expander("🔄 一键流水线", expanded=True):
    run_update = st.checkbox("更新行情", value=True)
    run_train = st.checkbox("训练模型", value=True)
    run_yaogu = st.checkbox("生成妖股V3", value=True)
    run_filter = st.checkbox("刷新优质池", value=False)

    if st.button("执行一键更新", use_container_width=True):
        logs = run_pipeline(run_update, run_train, run_yaogu, run_filter)
        with st.expander("运行日志", expanded=True):
            for name, code, out, err in logs:
                st.success(f"{name} 完成") if code == 0 else st.error(f"{name} 失败")
                if out:
                    st.code(out[-4000:])
                if err:
                    st.code(err[-4000:])
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔍 筛选")
only_clean_pool = st.sidebar.checkbox("只看优质池", value=False)
min_model_score = st.sidebar.slider("最低模型分", 0, 100, 0)
min_rainbow_score = st.sidebar.slider("最低趋势分", 0, 100, 0)
min_main_force_score = st.sidebar.slider("最低主力分", 0, 100, 0)
keyword = st.sidebar.text_input("搜索股票/行业/题材", "")


# =========================
# 数据加载
# =========================

raw_signal = safe_read(SIGNAL_PATH)
if raw_signal is None or raw_signal.empty:
    st.error("找不到 output/latest_stock_signals.csv，请先运行 train_model.py")
    st.stop()

signal_df = prepare_signal_df(raw_signal)

yaogu_raw = safe_read(YAOGU_PATH)
yaogu_df = prepare_yaogu_df(yaogu_raw) if yaogu_raw is not None and not yaogu_raw.empty else pd.DataFrame()

if not yaogu_df.empty:
    base_df = yaogu_df.copy()
    if "signal_status" in signal_df.columns:
        sig_keep = signal_df[["symbol", "signal_status"]].drop_duplicates("symbol")
        base_df = base_df.merge(sig_keep, on="symbol", how="left")
else:
    base_df = signal_df.copy()

base_df = build_core_score(base_df)
leader_df = identify_leaders(base_df)
mainline_df = identify_mainlines(leader_df)
ladder_df = build_limit_up_ladder(leader_df)
emotion = market_emotion(signal_df)

filtered = leader_df.copy()

for col, threshold in [
    ("model_score", min_model_score),
    ("rainbow_score", min_rainbow_score),
    ("main_force_score", min_main_force_score),
]:
    if col in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered[col], errors="coerce").fillna(0) >= threshold]

if only_clean_pool and "is_clean_pool" in filtered.columns:
    filtered = filtered[filtered["is_clean_pool"] == True]

if keyword.strip():
    key = keyword.strip()
    mask = (
        filtered["display_name"].astype(str).str.contains(key, case=False, na=False)
        | filtered["industry"].astype(str).str.contains(key, case=False, na=False)
        | filtered["board"].astype(str).str.contains(key, case=False, na=False)
        | filtered["themes"].astype(str).str.contains(key, case=False, na=False)
        | filtered["symbol"].astype(str).str.contains(key, case=False, na=False)
    )
    filtered = filtered[mask]

last_date = pd.to_datetime(signal_df["date"].max()).strftime("%Y-%m-%d")


# =========================
# 顶部
# =========================

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">🚀 A股量化交易终端 V6</div>
        <div class="hero-sub">
            最新交易日：{last_date} ｜ 一键流水线 + 主线识别 + 龙头识别 + 连板梯队 + 自动日报
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_metric_card("市场阶段", emotion["stage"], emotion["advice"], "#fb7185")
with c2:
    render_metric_card("情绪分", emotion["score"], "模型/趋势/主力综合", "#facc15")
with c3:
    render_metric_card("强候选", emotion["strong"], "模型趋势主力共振", "#4ade80")
with c4:
    render_metric_card("风险票", emotion["risk"], "偏空或大跌", "#fb7185")
with c5:
    render_metric_card("最热行业", emotion["hot_industry"], "按平均模型分", "#a78bfa")


# =========================
# 主体
# =========================

tabs = st.tabs([
    "🏆 今日作战",
    "📊 主线识别",
    "👑 龙头识别",
    "🔥 连板梯队",
    "💾 存档复盘",
    "📝 自动日报",
    "📋 全部数据",
])

with tabs[0]:
    st.subheader("🏆 Top3 核心作战卡")
    top3 = filtered.head(3)
    if top3.empty:
        st.info("当前没有符合条件的候选。")
    else:
        cols = st.columns(3)
        for i, (_, row) in enumerate(top3.iterrows()):
            with cols[i]:
                render_battle_card(row, i + 1)

    st.subheader("🎯 今日核心候选池")
    st.dataframe(display_table(filtered.head(30)), use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("📊 主线识别系统")
    if mainline_df.empty:
        st.info("暂无主线数据。")
    else:
        st.dataframe(mainline_df, use_container_width=True, hide_index=True)
        st.bar_chart(mainline_df.set_index("主线").head(15)["主线强度"])

with tabs[2]:
    st.subheader("👑 龙头识别系统")
    st.dataframe(display_table(filtered.head(50)), use_container_width=True, hide_index=True)

    if not filtered.empty:
        st.markdown("#### 龙头类型分布")
        st.bar_chart(filtered["龙头类型"].value_counts())

with tabs[3]:
    st.subheader("🔥 连板梯队")
    if ladder_df.empty:
        st.info("当前没有可识别的涨停/连板数据。精确连板需要接入真实涨停板历史数据。")
    else:
        st.dataframe(ladder_df, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("💾 今日预测存档")
    save_count = st.slider("存档推荐数量", 5, 100, 30)
    if st.button("保存今日推荐", use_container_width=True):
        path = save_today_predictions(filtered, save_count)
        if path:
            st.success(f"已保存：{path}")
        else:
            st.error("没有可保存的数据。")

    st.markdown("---")
    st.subheader("✅ 次日复盘验证")

    archives = available_archives()
    if not archives:
        st.info("暂无历史推荐存档。")
    else:
        selected = st.selectbox("选择要验证的预测日期", archives, format_func=lambda p: p.name.replace("_predictions.csv", ""))
        if st.button("用当前行情验证这一天推荐", use_container_width=True):
            review_df = review_archive(selected, signal_df)
            if review_df.empty:
                st.error("复盘失败，数据为空。")
            else:
                hit_rate = review_df["hit"].mean() * 100
                avg_ret = review_df["next_return_pct"].mean()
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("命中率", f"{hit_rate:.1f}%")
                k2.metric("平均收益", f"{avg_ret:.2f}%")
                k3.metric("大赚数量", int(review_df["big_win"].sum()))
                k4.metric("大亏数量", int(review_df["big_loss"].sum()))
                st.dataframe(display_table(review_df), use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("📝 自动交易日报")

    if st.button("生成今日交易日报", use_container_width=True):
        path = generate_daily_report(signal_df, filtered, mainline_df, ladder_df, emotion)
        st.success(f"已生成：{path}")

    latest_report = None
    if REPORT_DIR.exists():
        reports = sorted(REPORT_DIR.glob("*_daily_report.md"), reverse=True)
        if reports:
            latest_report = reports[0]

    if latest_report:
        st.markdown(f"#### 最新日报：{latest_report.name}")
        st.markdown(latest_report.read_text(encoding="utf-8"))
    else:
        st.info("暂无日报，点击上方按钮生成。")

with tabs[6]:
    st.subheader("📋 全部筛选结果")
    st.dataframe(display_table(filtered), use_container_width=True, hide_index=True)


st.caption("说明：本系统仅用于研究、复盘和交易辅助，不构成投资建议。")
