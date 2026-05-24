
"""
patch_app_add_us_report.py

自动给现有 app.py 增加“美股日报”Tab 和一键生成按钮。

使用：
    1. 把 us_daily_report_module.py 放到项目根目录
    2. 把本文件放到项目根目录
    3. 运行：
        python3 patch_app_add_us_report.py

说明：
    - 会自动备份 app.py 为 app_backup_before_us_report.py
    - 适配当前 V6 app.py 的 tabs 结构
"""

from pathlib import Path


APP_PATH = Path("app.py")
BACKUP_PATH = Path("app_backup_before_us_report.py")

IMPORT_LINE = "from us_daily_report_module import generate_us_daily_report\n"

US_TAB_BLOCK = '''
with tabs[6]:
    st.subheader("🇺🇸 美股每日复盘")

    st.info("点击按钮后会联网抓取美股指数、板块 ETF 和大型科技股数据，并生成 Markdown 日报。")

    if st.button("一键生成美股复盘日报", use_container_width=True):
        with st.spinner("正在生成美股复盘日报..."):
            try:
                us_path, us_text, us_index_df, us_sector_df, us_mega_df = generate_us_daily_report()
                st.success(f"已生成：{us_path}")

                c1, c2, c3 = st.columns(3)
                if not us_index_df.empty:
                    qqq = us_index_df[us_index_df["ticker"] == "QQQ"]
                    spy = us_index_df[us_index_df["ticker"] == "SPY"]
                    iwm = us_index_df[us_index_df["ticker"] == "IWM"]

                    c1.metric("QQQ", f"{float(qqq['pct_change'].iloc[0]):.2f}%" if not qqq.empty else "无")
                    c2.metric("SPY", f"{float(spy['pct_change'].iloc[0]):.2f}%" if not spy.empty else "无")
                    c3.metric("IWM", f"{float(iwm['pct_change'].iloc[0]):.2f}%" if not iwm.empty else "无")

                st.markdown("### 主要指数")
                st.dataframe(us_index_df, use_container_width=True, hide_index=True)

                st.markdown("### 板块 ETF 强弱")
                st.dataframe(us_sector_df, use_container_width=True, hide_index=True)

                st.markdown("### 大型科技股")
                st.dataframe(us_mega_df, use_container_width=True, hide_index=True)

                st.markdown("### 复盘日报")
                st.markdown(us_text)

            except Exception as e:
                st.error(f"美股日报生成失败：{e}")
                st.warning("如果提示缺少 yfinance，请先运行：pip install yfinance")

    st.markdown("---")
    st.caption("美股数据来自 yfinance，仅用于研究和复盘。")

'''


def main():
    if not APP_PATH.exists():
        print("找不到 app.py，请把本脚本放到项目根目录运行。")
        return

    text = APP_PATH.read_text(encoding="utf-8")

    if "generate_us_daily_report" in text:
        print("app.py 里已经包含美股日报模块，不重复修改。")
        return

    BACKUP_PATH.write_text(text, encoding="utf-8")
    print(f"已备份：{BACKUP_PATH}")

    marker = "import streamlit as st\n"
    if marker in text:
        text = text.replace(marker, marker + IMPORT_LINE, 1)
    else:
        text = IMPORT_LINE + text

    if '"📋 全部数据",' in text:
        text = text.replace('"📋 全部数据",', '"🇺🇸 美股日报",\n    "📋 全部数据",', 1)
    else:
        print("没有找到 tabs 列表里的“📋 全部数据”，可能需要手动添加 Tab。")

    old_block = 'with tabs[6]:\n    st.subheader("📋 全部筛选结果")'
    if old_block in text:
        text = text.replace(
            old_block,
            US_TAB_BLOCK + '\nwith tabs[7]:\n    st.subheader("📋 全部筛选结果")',
            1,
        )
    else:
        print("没有找到全部数据 Tab 的插入位置，可能需要手动添加美股日报代码块。")

    APP_PATH.write_text(text, encoding="utf-8")
    print("已完成：app.py 已增加美股日报 Tab。")
    print("下一步运行：streamlit run app.py")


if __name__ == "__main__":
    main()
