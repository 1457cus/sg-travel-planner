import pandas as pd
from pathlib import Path

def clean_attractions():
    input_path = Path("sg_attractions.csv")
    output_path = Path("sg_attractions_cleaned.csv")

    try:
        # === 修复1：指定 quotechar 和 escapechar ===
        df = pd.read_csv(
            input_path,
            encoding="utf-8-sig",
            quotechar='"',        # 定义引号字符
            escapechar='\\',      # 定义转义字符
            on_bad_lines="warn"   # 跳过错误行（可选）
        )

        # === 修复2：处理字段中的逗号 ===
        # 替换字段内的逗号为中文逗号（或移除）
        df["开放时间段"] = df["开放时间段"].str.replace(",", "，")
        df["景点特色说明"] = df["景点特色说明"].str.replace(",", "，")

        # === 修复3：处理双引号冲突 ===
        df["景点特色说明"] = df["景点特色说明"].str.replace('"', "'")

        # === 清洗逻辑（其他步骤保持不变） ===
        # 处理门票字段（示例）
        df["门票最低(元)"] = df["门票(元)"].apply(
            lambda x: int(x.split("-")[0]) if "-" in str(x) else 0 if "免费" in str(x) else None
        )
        df["门票最高(元)"] = df["门票(元)"].apply(
            lambda x: int(x.split("-")[1]) if "-" in str(x) else None
        )

        # 保存清洗结果
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ 清洗完成！生成文件：{output_path}")

    except pd.errors.ParserError as e:
        print(f"❌ CSV 解析失败：{str(e)}")
        print("请检查文件中是否存在未转义的逗号或引号")

if __name__ == "__main__":
    clean_attractions()

if __name__ == "__main__":
    clean_attractions()