"""
韶关旅游数据唯一标识符生成脚本 - 增强版
生成规则：SG-[类型代码][子类代码]-[哈希特征码]-[序号]
适配文件：sg_attractions.csv / sg_food_cleaned.csv / sg_culture_cleaned.csv
"""

import pandas as pd
from hashlib import blake2b
from pathlib import Path
import os

# ====================== 配置区 ======================
# 路径配置
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = Path("D:/plan/data")        # 数据文件存放目录
OUTPUT_DIR = BASE_DIR / "processed"   # 输出目录
# ===================================================

ENCODING = "utf-8-sig"  # 处理含 BOM 头的 UTF-8 文件

# 类型映射配置 (统一定义)
TYPE_CONFIG = {
    "attractions": {
        "file": "sg_attractions_cleaned.csv",
        "type_code": "A",
        "subtype_field": "主类型",
        "subtype_map": {
            "自然": "N",
            "历史": "H",
            "亲子": "K",
            "自然/历史": "NH"
        }
    },
    "food": {
        "file": "sg_food_cleaned.csv",
        "type_code": "F",
        "subtype_field": "类型",
        "subtype_map": {
            "农家菜": "N",
            "西餐": "W",
            "粤菜": "Y",
            "早茶": "Z",
            "茶餐厅": "C",
            "日式火锅": "R",
            "素食": "S",
            "炖品": "D",
            "火锅": "H",
            "烧烤": "B"
        }
    },
    "culture": {
        "file": "sg_culture_cleaned.csv",
        "type_code": "C",
        "subtype_field": "类别",
        "subtype_map": {
            "民俗": "M",
            "传统戏剧": "X",
            "传统技艺": "J",
            "传统舞蹈": "W",
            "手工艺": "S"
        }
    
    }
}
# ===================================================

def generate_id(row, data_type):
    """生成带分类结构的唯一标识符"""
    config = TYPE_CONFIG[data_type]
    
    # 获取子类代码
    raw_subtype = row[config["subtype_field"]]
    subtype = config["subtype_map"].get(raw_subtype.split("/")[0], "O")  # 取首个子类
    
    # 生成特征哈希码（名称前10字符的BLAKE2哈希）
    name_part = row["名称"][:10] if data_type != "food" else row["店名"][:10]
    hash_hex = blake2b(name_part.encode(), digest_size=2).hexdigest().upper()
    
    # 生成序号（按文件内顺序）
    global counter
    counter += 1
    return f"SG-{config['type_code']}{subtype}-{hash_hex}-{counter:04d}"

def process_data(data_type):
    """处理指定类型数据"""
    global counter
    counter = 0  # 重置计数器
    
    config = TYPE_CONFIG[data_type]
    input_path = DATA_DIR / config["file"]
    output_path = OUTPUT_DIR / f"{data_type}_with_id.csv"
    
    # ==== 文件存在性检查 ====
    if not input_path.exists():
        print(f"⛔ 文件 {input_path.name} 未找到，请检查：")
        print(f"   - 文件名是否包含隐藏字符（如 .csv.txt）")
        print(f"   - 是否执行过数据清洗")
        return
    
    try:
        # 读取数据
        df = pd.read_csv(input_path, encoding=ENCODING)
        
        if data_type == "attractions":
           df = df.rename(columns={"景点特色说明": "景点特色说明"})  # 确保列名一致
           required_cols = ["名称", "类型", "门票(元)", "景点特色说明"]
        elif data_type == "food":
           required_cols = ["店名", "人均消费", "特色菜"]
        elif data_type == "culture":
           required_cols = ["名称", "类别", "级别"]
        
        # 生成唯一编码
        id_field = "店名" if data_type == "food" else "名称"
        df["唯一编码"] = df.apply(lambda row: generate_id(row, data_type), axis=1)
        
        # 保存结果
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding=ENCODING)
        print(f"✅ {data_type} 数据处理完成，生成 {len(df)} 条编码")
        
    except Exception as e:
        print(f"❌ {data_type} 数据处理失败：{str(e)}")

if __name__ == "__main__":
    # 按顺序处理所有数据类型
    print("="*40)
    process_data("attractions")
    process_data("food")
    process_data("culture")
    print("="*40)