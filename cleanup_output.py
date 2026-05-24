"""
cleanup_output.py

安全整理 output 目录：
1. 核心文件保留在 output/
2. 旧数据移动到 backup/output_backup/
3. 日志移动到 logs/
4. 不直接永久删除，避免误删

运行：
    python3 cleanup_output.py
"""

from pathlib import Path
import shutil
from datetime import datetime


OUTPUT_DIR = Path("output")
BACKUP_DIR = Path("backup/output_backup")
LOG_DIR = Path("logs")

KEEP_FILES = {
    "latest_stock_signals.csv",
    "yaogu_model_v3_signals.csv",
    "stock_meta_map.csv",
    "clean_stock_pool.csv",
    "a_share_random_forest_model.joblib",
    "feature_importance.csv",
    "trade_plan.csv",
    "trade_plan.md",
}

MOVE_TO_LOGS = {
    "failed_symbols.csv",
    "update_failed_symbols.csv",
}

MOVE_TO_BACKUP = {
    "a_share_labeled_dataset.csv",
    "a_share_labeled_dataset_with_auto_manual_label.csv",
    "stock_pool.csv",
    "stock_pool_688.csv",
    "removed_stock_pool.csv",
    "yaogu_radar_v2_signals.csv",
}


def move_file(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if dst.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = dst_dir / f"{src.stem}_{stamp}{src.suffix}"

    shutil.move(str(src), str(dst))
    print(f"移动: {src} -> {dst}")


def main():
    if not OUTPUT_DIR.exists():
        print("找不到 output/ 目录")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for file in OUTPUT_DIR.iterdir():
        if not file.is_file():
            continue

        name = file.name

        if name in KEEP_FILES:
            print(f"保留: {file}")
        elif name in MOVE_TO_LOGS:
            move_file(file, LOG_DIR)
        elif name in MOVE_TO_BACKUP:
            move_file(file, BACKUP_DIR)
        else:
            # 未识别文件不删除，先移动到备份，避免误删
            move_file(file, BACKUP_DIR)

    print("\n整理完成。")
    print("核心文件仍在 output/")
    print("旧数据在 backup/output_backup/")
    print("日志在 logs/")


if __name__ == "__main__":
    main()
