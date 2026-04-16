# 用于将v1的数据中的```对转换成<action></action>

import json
import re

def convert_dataset_format(input_file: str, output_file: str):
    # 正则表达式：匹配 </think> 之后，可能包含空白符或换行，接着是 ```...``` 的内容
    # 使用 re.DOTALL 使得 (.*?) 可以匹配包含换行符的内容
    pattern = re.compile(r'</think>\s*```(.*?)```', re.DOTALL)
    
    processed_count = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
         
        for line in fin:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # 遍历所有 messages 进行替换
            for msg in data.get("messages", []):
                # 1. 转换 assistant 的输出格式
                if msg["role"] == "assistant":
                    # 提取匹配内容并替换为 <action> 标签
                    msg["content"] = pattern.sub(r'</think>\n<action>\1</action>', msg["content"])
                
                # 2. 同步转换 system prompt 中的规则与示例 (极其重要，否则 SFT 阶段会产生认知冲突)
                elif msg["role"] == "system":
                    # 替换 Rule 6 中的示例格式
                    msg["content"] = pattern.sub(r'</think>\n   <action>\1</action>', msg["content"])
                    
                    # 替换 Rule 6 中的文字描述说明
                    msg["content"] = msg["content"].replace(
                        "enclosed in code fences", 
                        "enclosed in <action></action> tags"
                    )
                    
                    # 替换 Rule 8 中的格式定义说明
                    msg["content"] = msg["content"].replace(
                        "```command [parameters]```",
                        "<action>command [parameters]</action>"
                    )
                    
            # 将处理后的 JSON 对象写回新文件
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
            processed_count += 1
            
    print(f"转换完成！共处理了 {processed_count} 条数据。")
    print(f"输出文件已保存至: {output_file}")

# 执行转换
if __name__ == "__main__":
    INPUT_FILE_PATH = "sft_task-opsrc.jsonl"   # 替换为你的原始输入文件路径
    OUTPUT_FILE_PATH = "output.jsonl" # 替换为你想保存的输出文件路径
    
    convert_dataset_format(INPUT_FILE_PATH, OUTPUT_FILE_PATH)