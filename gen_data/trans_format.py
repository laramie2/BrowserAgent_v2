import json
import re
import argparse
from typing import Callable, Dict, List, Optional


# =========================
# 1. 模式一：v1 专用
#    </think> / </conclusion> 后的 ```...``` 转成 <action>...</action>
# =========================

ACTION_BLOCK_PATTERN = re.compile(
    r'</(think|conclusion)>\s*```\s*(.*?)\s*```',
    re.DOTALL
)

def convert_code_fence_to_action_tag(text: str, role: str = None) -> str:
    if not isinstance(text, str):
        return text

    def replacer(match: re.Match) -> str:
        tag = match.group(1)
        content = match.group(2)

        if role == "system":
            return f"</{tag}>\n   <action>{content}</action>"
        else:
            return f"</{tag}>\n<action>{content}</action>"

    text = ACTION_BLOCK_PATTERN.sub(replacer, text)

    if role == "system":
        text = text.replace(
            "enclosed in code fences",
            "enclosed in <action></action> tags"
        )
        text = text.replace(
            "```command [parameters]```",
            "<action>command [parameters]</action>"
        )

    return text


# =========================
# 2. 模式二：仅删除 click 的 content
#    click [42] [搜索按钮] -> click [42]
# =========================

CLICK_REMOVE_CONTENT_PATTERN = re.compile(
    r'click\s+(\[\d+\])\s+\[[^\n\]]*\]'
)

def remove_click_content(text: str, role: str = None) -> str:
    if not isinstance(text, str):
        return text
    return CLICK_REMOVE_CONTENT_PATTERN.sub(r'click \1', text)


# =========================
# 3. 模式三：统一动作参数 [] -> <>
#    仅处理指定动作，不误伤普通文本中的 [xxx]
# =========================

ACTIONS = r'(click|type|hover|press|scroll|tab_focus|close_tab|goto|stop|new_tab|go_back|go_forward)'

ACTION_PATTERN = re.compile(
    rf'(?P<prefix>(```|`|<action>)?\s*(?:HISTORY_ACTION:\s*)?)'
    rf'(?P<action>{ACTIONS})'
    rf'(?P<suffix>(?:\s+\[[^\[\]\n]+\])*)',
    re.MULTILINE
)

def replace_brackets_in_action(match: re.Match) -> str:
    prefix = match.group('prefix') or ''
    action = match.group('action')
    suffix = match.group('suffix') or ''

    suffix = re.sub(r'\[([^\[\]\n]+)\]', r'<\1>', suffix)
    return f"{prefix}{action}{suffix}"

def convert_action_brackets(text: str, role: str = None) -> str:
    if not isinstance(text, str):
        return text
    return ACTION_PATTERN.sub(replace_brackets_in_action, text)


# =========================
# 通用处理：对单条文本依次应用若干转换
# =========================

MODE_FUNC_MAP: Dict[str, Callable[[str, str], str]] = {
    "fence_to_action": convert_code_fence_to_action_tag,
    "remove_click_content": remove_click_content,
    "bracket_to_angle": convert_action_brackets,
}

VALID_MODES = list(MODE_FUNC_MAP.keys())


def apply_transforms(text: str, role: str, modes: List[str]) -> str:
    for mode in modes:
        text = MODE_FUNC_MAP[mode](text, role)
    return text


# =========================
# 读取 prompt 文件
# =========================

def load_prompt_file(prompt_file: Optional[str]) -> Optional[str]:
    if not prompt_file:
        return None
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


# =========================
# 主逻辑：处理 jsonl
# =========================

def convert_jsonl(
    input_file: str,
    output_file: str,
    modes: List[str],
    prompt_content: Optional[str] = None,
):
    processed_count = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            if not line.strip():
                continue

            data = json.loads(line)

            # 处理 messages[*]["content"]
            if "messages" in data and isinstance(data["messages"], list):
                for msg in data["messages"]:
                    if (
                        isinstance(msg, dict)
                        and "content" in msg
                        and isinstance(msg["content"], str)
                    ):
                        role = msg.get("role")

                        # 如果指定了 prompt_file，则直接覆盖 system prompt
                        if prompt_content is not None and role == "system":
                            msg["content"] = prompt_content

                        # 其余转换逻辑保持不变
                        msg["content"] = apply_transforms(
                            msg["content"],
                            role,
                            modes
                        )

            # 处理顶层 text
            if "text" in data and isinstance(data["text"], str):
                data["text"] = apply_transforms(
                    data["text"],
                    role=None,
                    modes=modes
                )

            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
            processed_count += 1

    print(f"✅ 转换完成！共处理了 {processed_count} 条数据。")
    print(f"📥 输入文件: {input_file}")
    print(f"📤 输出文件: {output_file}")
    print(f"🛠️ 执行模式: {', '.join(modes)}")
    if prompt_content is not None:
        print("📝 已使用外部 prompt 文件覆盖 system prompt")


# =========================
# 命令行入口
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一的 BrowserAgent 数据格式转换脚本")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 jsonl 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 jsonl 文件路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        required=True,
        choices=VALID_MODES + ["all"],
        help=(
            "转换模式，可多选：\n"
            "fence_to_action        : 将 </think>/<conclusion> 后的 ```...``` 转为 <action>...</action>\n"
            "remove_click_content   : 将 click [id] [content] 转为 click [id]\n"
            "bracket_to_angle       : 将动作参数 [x] 转为 <x>\n"
            "all                    : 依次执行以上全部模式"
        )
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="若指定，则用该文件内容覆盖每条数据中 role=system 的 prompt；否则不修改 prompt"
    )

    args = parser.parse_args()

    modes = args.mode
    if "all" in modes:
        modes = [
            "fence_to_action",
            "remove_click_content",
            "bracket_to_angle",
        ]

    prompt_content = load_prompt_file(args.prompt_file)

    convert_jsonl(
        input_file=args.input,
        output_file=args.output,
        modes=modes,
        prompt_content=prompt_content,
    )


"""
# 只做 v1 的 ```...``` -> <action>...</action>
python trans_format.py \
  --input /path/to/data.jsonl \
  --output /path/to/output.jsonl \
  --mode fence_to_action \
  --prompt_file /path/to/prompt.txt

# 只删除 click 的 content
python trans_format.py \
  --input /path/to/data.jsonl \
  --output /path/to/output.jsonl \
  --mode remove_click_content \
  --prompt_file /path/to/prompt.txt

# 只把动作参数 [] 变成 <>
python trans_format.py \
  --input /path/to/data.jsonl \
  --output /path/to/output.jsonl \
  --mode bracket_to_angle \
  --prompt_file /path/to/prompt.txt

# 串联执行多个模式
python trans_format.py \
  --input /path/to/data.jsonl \
  --output /path/to/output.jsonl \
  --mode fence_to_action remove_click_content bracket_to_angle \
  --prompt_file /path/to/prompt.txt

# 全部执行
python trans_format.py \
  --input /path/to/data.jsonl \
  --output /path/to/output.jsonl \
  --mode all \
  --prompt_file /path/to/prompt.txt
"""

"""
python trans_format.py \
  --input /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/task-opsrc-enhanced_format_yt_and_action/data.jsonl \
  --output /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/task-opsrc-enhanced_format_yt_and_action/data_1.jsonl \
  --mode fence_to_action \
  --prompt_file /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/prompt/system_prompt_with_history_info_enhance_yt_and_action.txt
"""