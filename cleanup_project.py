"""
cleanup_project.py

整理当前交易系统项目根目录：
1. 删除明显重复/旧版本文件
2. 删除可重新生成的 HTML 报告
3. 删除旧研究/废弃脚本
4. 保留核心运行脚本
5. 默认不会删除 output/history/logs 等数据目录

运行：
    python3 cleanup_project.py

安全说明：
    - 默认 DRY_RUN = True，只预览不删除
    - 确认无误后，把 DRY_RUN 改成 False，再运行一次
"""

from pathlib import Path
import shutil
from datetime import datetime


# =========================
# 安全开关
# =========================
DRY_RUN = False

PROJECT_ROOT = Path(".").resolve()

# 明确建议删除的重复/旧版本文件
DELETE_FILES = {
    # 旧 app
    "app_optimized.py",
    "app_optimized_review.py",
    "app_trading_terminal_v4.py",
    "app_trading_terminal_v4_single.py",
    "app_trading_terminal_v4 copy.py",
    "app_trading_terminal_v5_core.py",
    "app_trading_terminal_v5_1_industry_fix.py",

    # copy 文件
    "market_emotion copy.py",
    "prediction_archive copy.py",
    "review_predictions copy.py",
    "README_V4 copy.md",

    # 旧研究/废弃脚本
    "build_market_dataset.py",
    "build_large_dataset_clean.py",
    "update_market_dataset2.py",
    "yaogu_radar_v2.py",
    "mock_sector_data.py",

    # 可重新生成的 HTML 报告
    "daily_report.html",
    "sector_gantt.html",
    "sector_rotation_gantt.html",
}

# 可选删除：如果你不用 Jupyter，可以删除
OPTIONAL_DELETE_FILES = {
    "1.ipynb",
    "start.ipynb",
}

# 明确保留的核心文件，仅用于打印提醒，不会强制处理
KEEP_FILES = {
    "app.py",
    "train_model.py",
    "update_market_dataset.py",
    "yaogu_model_v3.py",
    "filter_stock_pool.py",
    "cleanup_output.py",
    "market_emotion.py",
    "prediction_archive.py",
    "review_predictions.py",
    "candidate_picker.py",
    "leader_selector.py",
    "mainline_panel.py",
    "visual_panel.py",
    "daily_report.py",
    "sector_gantt.py",
    "sector_gantt_full.py",
    "gen_sector_data.py",
    "sector_to_stock.py",
    "pyvenv.cfg",
}

# 永远不要动这些目录
PROTECTED_DIRS = {
    "output",
    "history",
    "logs",
    "backup",
    "include",
    "lib",
    "share",
    "__pycache__",
    ".git",
    ".venv",
    "venv",
}


def safe_delete_file(path: Path):
    if not path.exists():
        return

    if DRY_RUN:
        print(f"[预览删除] {path.name}")
    else:
        path.unlink()
        print(f"[已删除] {path.name}")


def move_to_backup(path: Path):
    backup_dir = PROJECT_ROOT / "backup" / "project_cleanup_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    dst = backup_dir / path.name
    if dst.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = backup_dir / f"{path.stem}_{stamp}{path.suffix}"

    if DRY_RUN:
        print(f"[预览备份移动] {path.name} -> {dst}")
    else:
        shutil.move(str(path), str(dst))
        print(f"[已移动备份] {path.name} -> {dst}")


def main():
    print("=" * 60)
    print("交易系统项目目录清理工具")
    print(f"当前目录: {PROJECT_ROOT}")
    print(f"DRY_RUN: {DRY_RUN}")
    print("=" * 60)

    existing_files = {p.name: p for p in PROJECT_ROOT.iterdir() if p.is_file()}

    print("\n一、将删除的旧版本/重复文件：")
    found_delete = False
    for name in sorted(DELETE_FILES):
        path = existing_files.get(name)
        if path:
            found_delete = True
            safe_delete_file(path)
    if not found_delete:
        print("没有找到需要删除的旧版本文件。")

    print("\n二、可选删除的 Jupyter 文件：")
    found_optional = False
    for name in sorted(OPTIONAL_DELETE_FILES):
        path = existing_files.get(name)
        if path:
            found_optional = True
            print(f"[可选] {name}  如果不用 Jupyter，可以手动删除。")
    if not found_optional:
        print("没有找到 Jupyter 可选删除文件。")

    print("\n三、核心文件检查：")
    for name in sorted(KEEP_FILES):
        if name in existing_files:
            print(f"[保留] {name}")

    print("\n四、未识别的根目录文件：")
    known = DELETE_FILES | OPTIONAL_DELETE_FILES | KEEP_FILES
    unknown = []
    for path in PROJECT_ROOT.iterdir():
        if path.is_dir():
            continue
        if path.name not in known:
            unknown.append(path.name)

    if unknown:
        for name in sorted(unknown):
            print(f"[未识别，未处理] {name}")
    else:
        print("没有未识别文件。")

    print("\n五、目录保护：")
    for d in sorted(PROTECTED_DIRS):
        if (PROJECT_ROOT / d).exists():
            print(f"[保护目录] {d}/")

    print("\n" + "=" * 60)
    if DRY_RUN:
        print("当前是预览模式，没有真正删除。")
        print("确认列表没问题后，打开 cleanup_project.py，把 DRY_RUN = True 改成 False，再运行：")
        print("    python3 cleanup_project.py")
    else:
        print("清理完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
