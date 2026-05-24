import xmind
from pathlib import Path

# 你的 md 文件
md_file = Path("~/stockenv/reports/us_daily_report_2026-05-22.md").expanduser()

# 创建 xmind
workbook = xmind.load("us_report.xmind")
sheet = workbook.getPrimarySheet()
sheet.setTitle("美股日报")

root = sheet.getRootTopic()
root.setTitle("美股日报")

# 读取 markdown
text = md_file.read_text(encoding="utf-8")

# 简单按行导入
for line in text.splitlines():
    line = line.strip()

    if line.startswith("## "):
        topic = root.addSubTopic()
        topic.setTitle(line.replace("## ", ""))

    elif line.startswith("- "):
        child = topic.addSubTopic()
        child.setTitle(line.replace("- ", ""))

# 保存
xmind.save(workbook, "us_daily_report.xmind")

print("已生成 us_daily_report.xmind")