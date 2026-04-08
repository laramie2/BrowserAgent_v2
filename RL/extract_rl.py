import pandas as pd
import json
import os
import ast
import numpy as np
import argparse

# --- 新增的全局 SYSTEM PROMPT ---
NEW_SYSTEM_PROMPT = """You are a browser interaction assistant designed to execute step-by-step browser operations efficiently and precisely to complete the user's task. You are provided with specific tasks and webpage-related information, and you need to output accurate actions to accomplish the user's task.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
The open tabs: These are the tabs you have open.
The previous actions: There are the actions you just performed. It may be helpful to track your progress.
Information already found: Information related to the current query that has been identified in historical actions. You need to integrate and supplement this information.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id] [content]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the ""Enter"" key is pressed after typing unless press_enter_after is set to 0.
`hover [id] [content]`: Hover over an element with id.
`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
`scroll [down|up]`: Scroll the page up or down.

Tab Management Actions:
`new_tab`: Open a new, empty browser tab.
`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.
`close_tab`: Close the currently active tab.

URL Navigation Actions:
`goto [url]`: Navigate to a specific URL.
`go_back`: Navigate to the previously viewed page.
`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as ""N/A"" in the bracket.

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation.
2. You should only issue one action at a time.
3. You should follow the examples to reason step by step and then issue the next action.
4. You should refer to historical actions when issue an action and try not to make repetitive actions
5. All reasoning must be inside `<think></think>` tags, and there must be no output before `<think></think>`.
6. After `<think></think>`, only the action should be generated in the correct format, enclosed in code fences. For example:
   <think>This button looks relevant to my goal. Clicking it should take me to the next step.</think>
   ```click [id] [content]```
7. Issue the stop action when you think you have achieved the objective. Don’t generate anything after stop.
8. Always format actions correctly: 
```command [parameters]```
For example, if searching for ""death row inmates in the US"" in a search field with ID `21`, correctly format it as:
```type [21] [death row inmates in the US] [1]```
Avoid incorrect formats that omit brackets around parameters or numeric values.
9.Between `<think></think>`, you need to use `<conclusion></conclusion>` to enclose the information obtained in this round that is relevant to the current query. Note that if there is no valid information, this part is not required. The enclosed information must be directly usable to answer the original query. And the final anwser should be as simple as possible."""


# 自定义 JSON 编码器，专门用来处理 NumPy 格式的数据 (作为备用备查)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def load_excluded_questions(json_path):
    """
    从提供的 JSON 文件中加载需要排除的问题集合。
    """
    excluded_set = set()
    if not os.path.exists(json_path):
        print(f"警告: 找不到排除文件 {json_path}，将不进行过滤。")
        return excluded_set

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    excluded_set.add(item.strip())
                elif isinstance(item, dict) and 'question' in item:
                    excluded_set.add(item['question'].strip())
                elif isinstance(item, dict) and 'Objective' in item:
                    excluded_set.add(item['Objective'].strip())
    
    print(f"共加载了 {len(excluded_set)} 个需要排除的问题。")
    return excluded_set

def safe_eval(val):
    """安全解析可能被存为字符串的字典/列表"""
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except:
            try:
                return json.loads(val)
            except:
                return val
    return val

def extract_question_from_row(extra_info):
    """
    从 extra_info 字段中提取 question 内容
    """
    try:
        extra_info = safe_eval(extra_info)
        if isinstance(extra_info, dict):
            return str(extra_info.get('question', '')).strip()
    except Exception:
        pass
    return ""

def adapt_to_legal_format(row):
    """
    将当前数据行转换为合法的训练格式。
    补全 ground_truth, gt, url, id，并修复 prompt 的占位符格式。
    """
    extra_info = safe_eval(row['extra_info'])
    reward_model = safe_eval(row['reward_model'])
    prompt = safe_eval(row['prompt'])

    # --- 1. 修复 extra_info ---
    question = ""
    golden_answers = []
    if isinstance(extra_info, dict):
        question = extra_info.get('question', '')
        raw_golden_answers = extra_info.get('golden_answers', [])
        
        # 【关键修复】如果读取出来是 numpy 数组，将其转换为普通 python 列表
        if isinstance(raw_golden_answers, np.ndarray):
            golden_answers = raw_golden_answers.tolist()
        else:
            golden_answers = list(raw_golden_answers)
            
        # 【关键修复】使用 len() 判断是否为空，避免触发 Numpy 的布尔歧义报错
        fallback_gt = golden_answers[0] if len(golden_answers) > 0 else ""
        extra_info['gt'] = extra_info.get('selected_answer', fallback_gt)
        extra_info['url'] = 'https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing'
        extra_info['id'] = extra_info.get('index', 0)

    # --- 2. 修复 reward_model ---
    if isinstance(reward_model, dict):
        # 将 golden_answers 注入为 ground_truth
        reward_model['ground_truth'] = golden_answers

    # --- 3. 修复 prompt ---
    # 兼容 prompt 被解析为 numpy array 的情况
    if isinstance(prompt, (list, np.ndarray)):
        for p in prompt:
            if isinstance(p, dict):
                # 如果是 system prompt，替换为新的 system prompt
                if p.get('role') == 'system':
                    p['content'] = NEW_SYSTEM_PROMPT
                # 保留原有功能：如果是 user prompt，构造合法格式
                elif p.get('role') == 'user':
                    legal_content = (
                        f"Objective: {question}\n"
                        f"            URL: https://www.wikipedia.org/\n"
                        f"            Observation:None\n"
                        f"            Parsed Previous Action:None\n            "
                    )
                    p['content'] = legal_content
                
        # 统一转回 list 方便后续 JSON 序列化
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()

    # 更新行数据
    row['extra_info'] = extra_info
    row['reward_model'] = reward_model
    row['prompt'] = prompt
    
    return row

def extract_and_export_data(parquet_path, output_dir, num_samples, output_prefix="extracted_data", use_exclude=False, exclude_json_path=None, seed=None):
    """
    读取Parquet，(可选)过滤问题，(可选)随机提取指定数量并保存为 JSONL 和 Parquet。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 读取源 Parquet 文件
    print(f"正在读取 Parquet 文件: {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    original_len = len(df)
    print(f"原始数据总量: {original_len} 条")
    
    # 2. 根据开关决定是否进行数据过滤
    if use_exclude and exclude_json_path:
        print("已开启排除过滤功能，正在根据 JSON 文件过滤数据...")
        excluded_questions = load_excluded_questions(exclude_json_path)
        
        df['temp_question'] = df['extra_info'].apply(extract_question_from_row)
        filtered_df = df[~df['temp_question'].isin(excluded_questions)]
        filtered_len = len(filtered_df)
        print(f"过滤后剩余数据量: {filtered_len} 条 (过滤掉了 {original_len - filtered_len} 条)")
        
        filtered_df = filtered_df.drop(columns=['temp_question'])
    else:
        print("未开启排除过滤功能，直接在全量数据上进行提取。")
        filtered_df = df
        filtered_len = original_len
    
    # 3. 提取指定数量的数据 (加入随机抽样逻辑)
    if filtered_len == 0:
        print("错误: 当前没有可用数据！")
        return
        
    if filtered_len <= num_samples:
        print(f"警告: 可用数据量 ({filtered_len}) 小于等于请求的提取量 ({num_samples})。将提取所有可用数据。")
        sampled_df = filtered_df
    else:
        if seed is not None:
            print(f"开启随机抽样 (随机种子: {seed})...")
            sampled_df = filtered_df.sample(n=num_samples, random_state=seed)
        else:
            print("未指定随机种子，将按原始顺序截取头部数据...")
            sampled_df = filtered_df.head(num_samples)
    
    print(f"最终准备导出的数据量: {len(sampled_df)} 条")
    
    # ---------------- 核心修改点 ----------------
    print("正在格式化数据以适配训练要求 (补充 ground_truth、修改 prompt 格式等)...")
    sampled_df = sampled_df.apply(adapt_to_legal_format, axis=1)
    # ---------------------------------------------

    # 4. 保存为 Parquet 文件
    out_parquet_path = os.path.join(output_dir, f"{output_prefix}.parquet")
    sampled_df.to_parquet(out_parquet_path, index=False)
    print(f"已成功保存 Parquet 文件至: {out_parquet_path}")
    
    # 5. 保存为 JSONL 文件
    out_jsonl_path = os.path.join(output_dir, f"{output_prefix}.jsonl")
    # 因为内部包含了字典/列表结构，用 json 导出时可能会因为 numpy 格式报错，直接复用 df.to_json
    sampled_df.to_json(out_jsonl_path, orient='records', lines=True, force_ascii=False)
    print(f"已成功保存 JSONL 文件至: {out_jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从Parquet文件中提取指定数量的数据，支持随机抽样与数据去重过滤。")
    
    # 必填基础参数
    parser.add_argument("--parquet_path", type=str, required=True, help="原始parquet文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="提取后的文件保存目录")
    parser.add_argument("--num_samples", type=int, required=True, help="想要提取的数据数量")
    
    # 选填文件命名参数
    parser.add_argument("--output_prefix", type=str, default="train_data", help="输出文件的前缀名 (默认: train_data)")
    
    # 过滤相关参数
    parser.add_argument("--use_exclude", action="store_true", help="加上此参数则开启JSON文件过滤排除功能")
    parser.add_argument("--exclude_json", type=str, help="包含需要排除问题的json文件路径 (当开启 --use_exclude 时必填)")
    
    # 随机抽样相关参数
    parser.add_argument("--seed", type=int, default=None, help="随机种子。提供此参数(例如42)将开启随机抽样，否则按顺序截取。")
    
    args = parser.parse_args()
    
    if args.use_exclude and not args.exclude_json:
        parser.error("使用了 --use_exclude 参数，必须同时通过 --exclude_json 指定文件路径！")

    extract_and_export_data(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        output_prefix=args.output_prefix,
        use_exclude=args.use_exclude,
        exclude_json_path=args.exclude_json,
        seed=args.seed
    )


"""
python extract_rl.py \
  --parquet_path "/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/RL/dataset/BrowserAgent-SeedData/nq/train-00000-of-00001.parquet" \
  --output_dir "/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/RL/dataset/nq/" \
  --num_samples 1000 \
  --output_prefix "train_1000" \
  --use_exclude \
  --exclude_json "/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/step-opsrc-5000/obj.json" \
  --seed 42

python extract_rl.py \
  --parquet_path "/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/RL/dataset/BrowserAgent-SeedData/hotpot/train-00000-of-00001.parquet" \
  --output_dir "/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/RL/dataset/hotpot/" \
  --num_samples 1000 \
  --output_prefix "train_1000" \
  --use_exclude \
  --exclude_json "/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/step-opsrc-5000/obj.json" \
  --seed 42

python extract_rl.py \
  --parquet_path "/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/RL/dataset/BrowserAgent-SeedData/hotpot/validation-00000-of-00001.parquet" \
  --output_dir "/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/RL/dataset/hotpot" \
  --num_samples 1000 \
  --output_prefix "test_1000"
"""