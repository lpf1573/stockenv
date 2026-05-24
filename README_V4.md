# A股交易研究终端 V4

## 文件说明

把这 4 个文件复制到你的项目根目录，和 `train_model.py` 同级：

- `app_trading_terminal_v4.py`：主界面
- `prediction_archive.py`：每日推荐存档
- `review_predictions.py`：次日复盘验证
- `market_emotion.py`：市场情绪和主线板块识别

## 运行

```bash
streamlit run app_trading_terminal_v4.py
```

## 每天使用流程

### 今天收盘后

1. 运行你的数据更新 / 训练脚本，生成：
   - `output/latest_stock_signals.csv`
   - `output/yaogu_model_v3_signals.csv`，如果有
2. 打开 V4 app。
3. 点击：`存档今日推荐`。

系统会生成：

```text
history/daily_signals/YYYY-MM-DD_signals.csv
history/daily_picks/YYYY-MM-DD_picks.csv
```

### 明天收盘后

1. 再次更新行情和模型。
2. 打开 V4 app。
3. 选择昨天的推荐日期。
4. 点击：`用当前行情验证这一天推荐`。

系统会生成长期复盘表：

```text
history/prediction_review.csv
```

## 单独命令

```bash
python3 prediction_archive.py
python3 review_predictions.py 2026-05-22
python3 market_emotion.py
```

## 注意

- 复盘验证使用推荐日收盘价和当前收盘价计算收益。
- 这不是自动交易，不构成买卖建议。
- 真正有价值的是长期统计，不是某一天 Top3 灵不灵。
