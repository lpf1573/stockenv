"""
train_model.py

功能：
1. 读取标注数据
2. 清洗数据
3. 训练随机森林模型
4. 输出模型 + 特征重要性
5. 生成 latest_stock_signals.csv

运行：
    python3 train_model.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_PATH = "output/a_share_labeled_dataset_with_auto_manual_label.csv"
MODEL_PATH = "output/a_share_random_forest_model.joblib"
SIGNAL_PATH = "output/latest_stock_signals.csv"

TEST_RATIO = 0.2


def load_data():
    print("📥 读取数据...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])

    print("原始数据量：", len(df))
    return df


def clean_data(df):
    print("🧹 清洗数据...")

    df = df.dropna()

    # 只保留有意义标签
    df = df[df["final_label"].isin([-1, 0, 1])]

    print("清洗后数据量：", len(df))
    print("标签分布：")
    print(df["final_label"].value_counts())

    return df


def split_data(df):
    print("📊 切分训练集 / 测试集...")

    df = df.sort_values("date")

    split_index = int(len(df) * (1 - TEST_RATIO))

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    print("训练集：", len(train_df))
    print("测试集：", len(test_df))

    return train_df, test_df


# def get_features(df):
#     ignore_cols = [
#         "symbol", "date",
#         "future_close", "future_return",
#         "label", "manual_label", "final_label"
#     ]

#     features = [col for col in df.columns if col not in ignore_cols]

#     return features
def get_features(df):
    ignore_cols = [
        "symbol",
        "date",
        "future_close",
        "future_return",
        "label",
        "manual_label",
        "final_label",
        "main_force_signal",
        "adjustflag",
    ]

    features = []

    for col in df.columns:
        if col in ignore_cols:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)

    print("使用特征数量：", len(features))
    print("使用特征：")
    print(features)

    return features

def train_model(train_df, features):
    print("🤖 训练模型...")

    X = train_df[features]
    y = train_df["final_label"]

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    print("✅ 模型训练完成")

    return model


def evaluate_model(model, test_df, features):
    print("📈 简单评估...")

    X_test = test_df[features]
    y_test = test_df["final_label"]

    pred = model.predict(X_test)

    acc = (pred == y_test).mean()

    print("测试集准确率：", round(acc, 4))


def save_model(model):
    os.makedirs("output", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("💾 模型已保存：", MODEL_PATH)


def generate_signals(model, df, features):
    print("🚀 生成最新信号...")

    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date].copy()

    X = latest_df[features]

    probs = model.predict_proba(X)

    latest_df["prob_down"] = probs[:, 0]
    latest_df["prob_mid"] = probs[:, 1]
    latest_df["prob_up"] = probs[:, 2]

    latest_df["pred_label"] = model.predict(X)

    latest_df["model_score"] = latest_df["prob_up"] * 100

    latest_df = latest_df.sort_values("model_score", ascending=False)

    output_cols = [
        "symbol", "date", "close", "pct_change",
        "rainbow_score", "main_force_score",
        "pred_label", "prob_up", "prob_down",
        "model_score"
    ]

    latest_df[output_cols].to_csv(
        SIGNAL_PATH, index=False, encoding="utf-8-sig"
    )

    print("📊 最新信号已生成：", SIGNAL_PATH)


def main():
    df = load_data()
    df = clean_data(df)

    train_df, test_df = split_data(df)

    features = get_features(df)

    model = train_model(train_df, features)

    evaluate_model(model, test_df, features)

    save_model(model)

    generate_signals(model, df, features)

    print("\n🎯 完成：模型 + 信号 已更新")


if __name__ == "__main__":
    main()