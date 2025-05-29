"""
韶关餐饮数据清洗脚本 - 自动生成 sg_food_cleaned.csv
"""

import pandas as pd
from pathlib import Path

def clean_food():
    # 配置路径
    input_path = Path("sg_food.csv")       # 原始数据文件
    output_path = Path("sg_food_cleaned.csv")  # 清洗后文件
    
    try:
        # 读取原始数据
        df = pd.read_csv(input_path, encoding="utf-8")
        
        # 清洗规则
        df['店名'] = df['店名'].str.replace(r'\(.*店\)', '', regex=True)  # 移除分店信息
        df['人均消费'] = df['人均消费'].str.replace('¥', '').astype(int)
        df['特色菜'] = df['特色菜'].str.replace(' ', '、')  # 空格替换为顿号
        
        # 生成价格区间 (规则：原价×0.8-原价×1.8)
        df['人均最低(元)'] = (df['人均消费'] * 0.8).round().astype(int)
        df['人均最高(元)'] = (df['人均消费'] * 1.8).round().astype(int)
        
        # 分类标记
        df['类型'] = df.apply(lambda x: 
            "火锅" if "火锅" in x['店名'] else
            "西餐" if "西餐" in x['店名'] else
            "粤菜", axis=1)
        
        # 保存结果
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ 清洗完成！生成文件：{output_path}")
        
    except Exception as e:
        print(f"❌ 清洗失败：{str(e)}")
        print("请检查：")
        print("1. 当前目录是否存在 sg_food.csv")
        print("2. 文件内容是否与示例格式一致")

if __name__ == "__main__":
    clean_food()
