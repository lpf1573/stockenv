import akshare as ak
import pandas as pd
from datetime import datetime

print("🚀 开始获取板块数据...")

try:
    df = ak.stock_board_industry_name_em()
    print("✅ 数据获取成功")

    df = df[["板块名称", "涨跌幅"]]
    df.columns = ["sector", "pct_chg"]

    df["date"] = datetime.now().strftime("%Y-%m-%d")

    print(df.head())

    df.to_csv("sector_pct.csv", index=False)
    print("🎉 文件已生成：sector_pct.csv")

except Exception as e:
    print("❌ 出错了：", e)