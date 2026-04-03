import json

def view_jsonl(file_path, lines_per_page=10):
    with open(file_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    total = len(all_lines)
    start = 0
    
    while start < total:
        end = min(start + lines_per_page, total)
        for i in range(start, 6):
            data = json.loads(all_lines[i])
            # print(f"\n--- 行 {i+1}/{total} ---")
            print(json.dumps(data, indent=2, ensure_ascii=False)[:20000])
        
        if end < total:
            # input(f"\n显示 {end}/{total}，按Enter继续...")
            input()
        start = end

# 使用
view_jsonl('/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v1/sft.jsonl', lines_per_page=1)