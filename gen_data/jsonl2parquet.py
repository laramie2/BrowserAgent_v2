import pandas as pd
import json

def jsonl_to_parquet_pandas(jsonl_file_path, parquet_file_path):
    """
    使用Pandas将JSONL转换为Parquet
    
    Args:
        jsonl_file_path: 输入的JSONL文件路径
        parquet_file_path: 输出的Parquet文件路径
    """
    print(f"读取JSONL文件: {jsonl_file_path}")
    
    # 读取JSONL文件
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"跳过无效行 {line_num}: {e}")
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    print(f"共读取 {len(df)} 条记录")
    print(f"列名: {list(df.columns)}")
    
    # 转换为Parquet
    df.to_parquet(parquet_file_path, index=False, engine='pyarrow')
    print(f"已保存到: {parquet_file_path}")
    
    return df


if __name__ == "__main__":
    # 使用示例
    jsonl_path = "/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/RL/dataset/train/train_hotpot500_nq500.jsonl"
    parquet_path = "/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/RL/dataset/train/train_hotpot500_nq500.parquet"
    jsonl_to_parquet_pandas(jsonl_path, parquet_path)