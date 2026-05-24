"""
fix_stock_meta_map.py

修复 / 增强 output/stock_meta_map.csv

解决问题：
1. app 里行业、题材大量显示“其他”
2. 主线识别无法准确分行业
3. 龙头作战卡缺少行业/题材信息

运行：
    python3 fix_stock_meta_map.py

输入：
    output/latest_stock_signals.csv
    output/yaogu_model_v3_signals.csv
    output/stock_meta_map.csv  可选

输出：
    output/stock_meta_map.csv
    output/stock_meta_map_backup_时间戳.csv
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


OUTPUT_DIR = Path("output")

SIGNAL_PATH = OUTPUT_DIR / "latest_stock_signals.csv"
YAOGU_PATH = OUTPUT_DIR / "yaogu_model_v3_signals.csv"
META_PATH = OUTPUT_DIR / "stock_meta_map.csv"


# =========================
# 手工增强映射
# 先把最近高频出现的科创板候选补上
# 后面你可以继续往里面加
# =========================

MANUAL_META = {
    "688700": {
        "stock_name": "东威科技",
        "industry": "专用设备",
        "board": "科创板",
        "themes": "PCB设备;复合铜箔;新能源设备;半导体设备",
    },
    "688512": {
        "stock_name": "慧智微",
        "industry": "半导体",
        "board": "科创板",
        "themes": "射频芯片;5G;AI芯片;国产替代",
    },
    "688507": {
        "stock_name": "索辰科技",
        "industry": "软件服务",
        "board": "科创板",
        "themes": "工业软件;AI仿真;军工信息化;CAE",
    },
    "688630": {
        "stock_name": "芯碁微装",
        "industry": "半导体设备",
        "board": "科创板",
        "themes": "光刻设备;PCB设备;泛半导体;先进封装",
    },
    "688360": {
        "stock_name": "德马科技",
        "industry": "智能物流",
        "board": "科创板",
        "themes": "机器人;智能物流;工业自动化;专用设备",
    },
    "688257": {
        "stock_name": "新锐股份",
        "industry": "有色金属",
        "board": "科创板",
        "themes": "硬质合金;钨;高端制造;新材料",
    },
    "688010": {
        "stock_name": "福光股份",
        "industry": "光学光电子",
        "board": "科创板",
        "themes": "光学镜头;军工;机器视觉;安防",
    },
}


# =========================
# 关键词规则
# 根据股票名称 / 题材文本推断行业
# =========================

KEYWORD_RULES = [
    ("芯片", "半导体", "芯片;国产替代;半导体"),
    ("微", "半导体", "芯片;半导体;国产替代"),
    ("半导体", "半导体", "半导体;国产替代"),
    ("光刻", "半导体设备", "光刻设备;半导体设备"),
    ("设备", "专用设备", "高端装备;专用设备"),
    ("科技", "科技成长", "科技成长;高端制造"),
    ("软件", "软件服务", "工业软件;人工智能"),
    ("信息", "软件服务", "信创;数据要素;人工智能"),
    ("智能", "人工智能", "人工智能;机器人;智能制造"),
    ("机器人", "机器人", "机器人;智能制造"),
    ("光电", "光学光电子", "光学光电子;机器视觉"),
    ("股份", "其他", "其他"),
]


def safe_read(path: Path):
    if not path.exists():
        return None

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


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_df(df):
    df = df.copy()

    symbol_col = pick_col(df, ["symbol", "code", "股票代码", "证券代码", "ts_code"])
    if symbol_col is None:
        raise ValueError(f"缺少股票代码列，当前列名：{list(df.columns)}")

    df["symbol"] = df[symbol_col].apply(normalize_symbol)

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

    for col, default in [
        ("stock_name", ""),
        ("industry", "其他"),
        ("board", "未知"),
        ("themes", "其他"),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str)
        df.loc[df[col].isin(["", "nan", "None", "未知"]), col] = default

    return df[["symbol", "stock_name", "industry", "board", "themes"]].drop_duplicates("symbol")


def infer_board(symbol):
    symbol = str(symbol)
    if symbol.startswith("688"):
        return "科创板"
    if symbol.startswith("300"):
        return "创业板"
    if symbol.startswith("60"):
        return "沪主板"
    if symbol.startswith("00"):
        return "深主板"
    if symbol.startswith("8") or symbol.startswith("4"):
        return "北交所"
    return "未知"


def infer_from_text(symbol, name, old_industry="", old_themes=""):
    text = f"{name} {old_industry} {old_themes}"

    # 手工映射优先
    if symbol in MANUAL_META:
        return MANUAL_META[symbol]

    for kw, industry, themes in KEYWORD_RULES:
        if kw in text:
            return {
                "stock_name": name,
                "industry": industry,
                "board": infer_board(symbol),
                "themes": old_themes if old_themes not in ["", "其他", "未知", "nan", "None"] else themes,
            }

    return {
        "stock_name": name,
        "industry": old_industry if old_industry not in ["", "其他", "未知", "nan", "None"] else "其他",
        "board": infer_board(symbol),
        "themes": old_themes if old_themes not in ["", "其他", "未知", "nan", "None"] else "其他",
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    frames = []

    for path in [SIGNAL_PATH, YAOGU_PATH, META_PATH]:
        df = safe_read(path)
        if df is None or df.empty:
            continue
        try:
            frames.append(normalize_df(df))
        except Exception as e:
            print(f"跳过 {path}: {e}")

    if not frames:
        print("没有找到可用数据，请确认 output/latest_stock_signals.csv 是否存在。")
        return

    all_meta = pd.concat(frames, ignore_index=True)

    # 同一 symbol 多来源合并，优先保留非“其他”的字段
    rows = []
    for symbol, g in all_meta.groupby("symbol"):
        stock_name = ""
        industry = "其他"
        board = infer_board(symbol)
        themes = "其他"

        for _, r in g.iterrows():
            n = str(r.get("stock_name", ""))
            ind = str(r.get("industry", "其他"))
            bd = str(r.get("board", "未知"))
            th = str(r.get("themes", "其他"))

            if stock_name in ["", "nan", "None"] and n not in ["", "nan", "None"]:
                stock_name = n
            if industry in ["", "其他", "未知", "nan", "None"] and ind not in ["", "其他", "未知", "nan", "None"]:
                industry = ind
            if board in ["", "未知", "其他", "nan", "None"] and bd not in ["", "未知", "其他", "nan", "None"]:
                board = bd
            if themes in ["", "其他", "未知", "nan", "None"] and th not in ["", "其他", "未知", "nan", "None"]:
                themes = th

        inferred = infer_from_text(symbol, stock_name, industry, themes)

        rows.append({
            "symbol": symbol,
            "stock_name": inferred["stock_name"],
            "industry": inferred["industry"],
            "board": inferred["board"],
            "themes": inferred["themes"],
        })

    out = pd.DataFrame(rows).drop_duplicates("symbol")

    # 备份旧文件
    if META_PATH.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = OUTPUT_DIR / f"stock_meta_map_backup_{stamp}.csv"
        META_PATH.rename(backup_path)
        print(f"已备份旧映射：{backup_path}")

    out = out.sort_values("symbol")
    out.to_csv(META_PATH, index=False, encoding="utf-8-sig")

    print(f"已生成：{META_PATH}")
    print(f"股票数量：{len(out)}")
    print("行业分布 Top20：")
    print(out["industry"].value_counts().head(20))


if __name__ == "__main__":
    main()
