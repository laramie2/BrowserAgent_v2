# 用于统计模型生成轨迹的成功率
# python check.py

import jsonlines
import json
from pathlib import Path

def count_success_tasks(jsonl_file_path):
    """
    统计jsonl文件中成功任务的数量
    
    Args:
        jsonl_file_path: jsonl文件路径
    
    Returns:
        dict: 包含统计信息的字典
    """
    total = 0
    success_count = 0
    failed_count = 0
    
    try:
        with jsonlines.open(jsonl_file_path) as reader:
            for item in reader:
                total += 1
                # 检查success字段
                if item.get('success', False):
                    success_count += 1
                else:
                    failed_count += 1
                    
    except FileNotFoundError:
        print(f"错误：文件 {jsonl_file_path} 不存在")
        return None
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return None
    
    return {
        'total': total,
        'success': success_count,
        'failed': failed_count,
        'success_rate': success_count / total * 100 if total > 0 else 0
    }

# 使用示例
if __name__ == "__main__":
    # 方法1：直接指定文件路径
    file_path = '/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_seq/results/Qwen2.5-VL-7B-Instruct_task-opsrc-sft-5e-5lr-freeze_false-2epoch/hotpot_test_results.jsonl'  # 替换为你的文件路径
    stats = count_success_tasks(file_path)
    
    if stats:
        print(f"总任务数: {stats['total']}")
        print(f"成功任务数: {stats['success']}")
        print(f"失败任务数: {stats['failed']}")
        print(f"成功率: {stats['success_rate']:.2f}%")
    
    # # 方法2：使用命令行参数
    # import argparse
    # parser = argparse.ArgumentParser(description='统计jsonl文件中成功任务的数量')
    # parser.add_argument('--input_file', type=str, required=True, help='输入的jsonl文件路径')
    # args = parser.parse_args()
    
    # stats = count_success_tasks(args.input_file)
    # if stats:
    #     print(f"\n文件: {args.input_file}")
    #     print(f"总任务数: {stats['total']}")
    #     print(f"成功任务数: {stats['success']}")
    #     print(f"失败任务数: {stats['failed']}")
    #     print(f"成功率: {stats['success_rate']:.2f}%")