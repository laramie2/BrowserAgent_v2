import pandas as pd
import json


# 设置Pandas显示选项
pd.set_option('display.max_colwidth', None)  # 显示完整列内容
pd.set_option('display.max_rows', None)       # 显示所有行
pd.set_option('display.width', None)          # 自动检测宽度
pd.set_option('display.max_seq_items', None)  # 显示序列的所有元素

# 逐行读取并解析
data = []
with open('/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft_step-opsrc.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data.append(json.loads(line.strip()))
        except json.JSONDecodeError as e:
            print(f"解析错误: {e}")
            print(f"问题行: {line[:100]}...")

# 转换为DataFrame
df = pd.DataFrame(data)

# 查看数据
print(df.head())
print(df.shape)
print(df.columns)