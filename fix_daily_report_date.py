
"""
fix_daily_report_date.py

修复 app.py 自动日报日期显示旧文件的问题。

功能：
1. 自动备份 app.py
2. 强制日报读取 latest_stock_signals.csv 的最大 date
3. 生成日报后优先显示最新交易日对应的日报
4. 如果 reports/ 里有旧日报，不再误显示旧文件

运行：
    python3 fix_daily_report_date.py
"""

from pathlib import Path
from datetime import datetime


APP_PATH = Path("app.py")


PATCH_FUNCTION = '''
def get_current_trade_date_from_signal():
    """
    从 output/latest_stock_signals.csv 读取当前最新交易日。
    避免 reports 目录里旧日报被误认为最新。
    """
    try:
        df = pd.read_csv(SIGNAL_PATH, encoding="utf-8-sig", dtype=str)
    except Exception:
        try:
            df = pd.read_csv(SIGNAL_PATH, dtype=str)
        except Exception:
            return datetime.now().strftime("%Y-%m-%d")

    if "date" not in df.columns or df.empty:
        return datetime.now().strftime("%Y-%m-%d")

    return pd.to_datetime(df["date"], errors="coerce").max().strftime("%Y-%m-%d")


def get_current_daily_report_path():
    """
    只读取当前交易日对应的 A股日报。
    """
    date_text = get_current_trade_date_from_signal()
    return REPORT_DIR / f"{date_text}_daily_report.md"
'''


OLD_BLOCK = '''    latest_report = None
    if REPORT_DIR.exists():
        reports = sorted(REPORT_DIR.glob("*_daily_report.md"), reverse=True)
        if reports:
            latest_report = reports[0]

    if latest_report:
        st.markdown(f"#### 最新日报：{latest_report.name}")
        st.markdown(latest_report.read_text(encoding="utf-8"))
    else:
        st.info("暂无日报，点击上方按钮生成。")'''


NEW_BLOCK = '''    latest_report = get_current_daily_report_path()

    if latest_report.exists():
        st.markdown(f"#### 当前交易日日报：{latest_report.name}")
        st.markdown(latest_report.read_text(encoding="utf-8"))
    else:
        current_date = get_current_trade_date_from_signal()
        st.info(f"当前交易日 {current_date} 暂无日报，点击上方按钮生成。")'''


def main():
    if not APP_PATH.exists():
        print("找不到 app.py，请在 stockenv 项目根目录运行。")
        return

    text = APP_PATH.read_text(encoding="utf-8")

    backup = Path(f"app_backup_before_daily_report_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
    backup.write_text(text, encoding="utf-8")
    print(f"已备份：{backup}")

    if "def get_current_trade_date_from_signal()" not in text:
        marker = "# =========================\n# 展示函数"
        if marker in text:
            text = text.replace(marker, PATCH_FUNCTION + "\n\n" + marker, 1)
        else:
            marker2 = "def display_table("
            if marker2 in text:
                text = text.replace(marker2, PATCH_FUNCTION + "\n\n" + marker2, 1)
            else:
                text += "\n\n" + PATCH_FUNCTION

    if OLD_BLOCK in text:
        text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    else:
        print("没有精确找到旧的 latest_report 代码块。")
        print("请把 app.py 里 自动日报 tab 中 reports glob 读取最新日报的逻辑手动改为 get_current_daily_report_path()。")

    APP_PATH.write_text(text, encoding="utf-8")
    print("修复完成。")
    print("请重新运行：streamlit run app.py")
    print("然后点击：自动日报 -> 生成今日交易日报")


if __name__ == "__main__":
    main()
