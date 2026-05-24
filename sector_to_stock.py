import pandas as pd
import random

def get_mock_sector_stocks():
    """
    模拟：板块 -> 股票映射
    后面可以替换成真实数据（akshare / tushare）
    """
    return {
        "机器人": ["埃斯顿", "绿的谐波", "鸣志电器", "中大力德"],
        "半导体": ["中芯国际", "寒武纪", "长电科技", "北方华创"],
        "算力": ["中际旭创", "浪潮信息", "紫光股份"],
        "新能源": ["宁德时代", "阳光电源", "隆基绿能"],
        "军工": ["中航沈飞", "航发动力", "中航光电"]
    }


def score_stock():
    """
    模拟打分（后面可替换为真实指标）
    """
    return round(random.uniform(60, 100), 2)


def pick_leaders_from_sector(sectors):
    """
    输入：主线板块列表
    输出：每个板块最强3只股票
    """
    mapping = get_mock_sector_stocks()

    result = []

    for sector in sectors:
        stocks = mapping.get(sector, [])

        scored = []
        for s in stocks:
            scored.append({
                "sector": sector,
                "stock": s,
                "score": score_stock()
            })

        # 排序选Top3
        scored = sorted(scored, key=lambda x: x["score"], reverse=True)[:3]

        result.extend(scored)

    return pd.DataFrame(result)


if __name__ == "__main__":
    # 👉 模拟你甘特图识别出的主线
    hot_sectors = ["机器人", "半导体", "算力"]

    df = pick_leaders_from_sector(hot_sectors)

    print("\n🔥 主线板块 → 龙头股：\n")
    print(df)