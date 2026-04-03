import json
import argparse
from pathlib import Path

def extract_jsonl_lines(input_file, output_file, num_lines):
    """
    从JSONL文件中顺序提取指定数量的行
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        num_lines: 要提取的行数
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    if num_lines <= 0:
        raise ValueError("提取行数必须大于0")
    
    extracted_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(infile, 1):
            if i > num_lines:
                break
            
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            try:
                # 验证JSON格式是否正确
                json.loads(line)
                outfile.write(line + '\n')
                extracted_count += 1
            except json.JSONDecodeError as e:
                print(f"警告: 第{i}行不是有效的JSON格式，已跳过。错误: {e}")
    
    print(f"成功提取 {extracted_count} 行到 {output_file}")
    return extracted_count

def main():
    parser = argparse.ArgumentParser(description='从JSONL文件中顺序提取指定数量的行')
    parser.add_argument('input_file', help='输入JSONL文件路径')
    parser.add_argument('output_file', help='输出JSONL文件路径')
    parser.add_argument('-n', '--num-lines', type=int, required=True, 
                       help='要提取的行数')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='显示详细信息')
    
    args = parser.parse_args()
    
    try:
        extracted = extract_jsonl_lines(
            args.input_file, 
            args.output_file, 
            args.num_lines
        )
        
        if args.verbose:
            print(f"处理完成: 从 {args.input_file} 提取了前 {extracted} 行")
            
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # 使用示例
    # 方式1: 直接调用函数
    # extract_jsonl_lines('input.jsonl', 'output.jsonl', 100)
    
    # 方式2: 命令行方式
    # python extract_jsonl.py input.jsonl output.jsonl -n 100
    # python extract_jsonl.py input.jsonl output.jsonl -n 100 -v
    
    main()