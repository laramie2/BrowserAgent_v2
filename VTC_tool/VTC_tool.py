import math
import re
from typing import Tuple, List, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class VTCTool:
    """
    Visual Text Compression (VTC) Tool
    用于将 Agent 的文本观测、轨迹等转换为高度优化的图像，支持 SFT 数据生成与 RL Rollout。
    """
    def __init__(
        self, 
        font_path: Optional[str] = None,
        font_size: int = 16,
        bg_color: Tuple[int, int, int] = (30, 30, 30),      # 深灰底色
        text_color: Tuple[int, int, int] = (220, 220, 220), # 浅灰文字
        title_color: Tuple[int, int, int] = (100, 200, 100), # 绿色标题
        highlight_configs: Optional[List[Dict[str, Any]]] = None
    ):
        self.font_size = font_size
        self.bg_color = bg_color
        self.text_color = text_color
        self.title_color = title_color
        self.font = self._load_font(font_path, font_size)
        
        # 预计算行高
        # getbbox 返回 (left, top, right, bottom)
        bbox = self.font.getbbox("A")
        self.line_height = (bbox[3] - bbox[1]) + 8  # 字体高度 + 8px 行距

        self.highlight_configs = highlight_configs or []
        self.highlight_dict = {
            cfg["context"]: cfg.get("color", (255, 255, 255)) 
            for cfg in self.highlight_configs
        }

        # --- 高亮配置解析与正则预编译 ---
        if self.highlight_dict:
            # 使用正则对关键词进行转义并分组，保留分隔符以便后续精准渲染
            escaped_keys = [re.escape(k) for k in self.highlight_dict.keys()]
            self.highlight_pattern = re.compile(f'({"|".join(escaped_keys)})')
        else:
            self.highlight_pattern = None

    def _load_font(self, font_path: Optional[str], font_size: int) -> ImageFont.FreeTypeFont:
        """加载字体，带有多级回退机制"""
        if font_path:
            try:
                return ImageFont.truetype(font_path, size=font_size)
            except IOError:
                print(f"[Warning] 无法加载指定字体 {font_path}，将尝试默认字体。")

        fallback_fonts = [
            "DejaVuSansMono.ttf", # Linux
            "Consolas.ttf",       # Windows
            "Menlo.ttc",          # macOS
            "simhei.ttf"          # 中文回退
        ]
        
        for ff in fallback_fonts:
            try:
                return ImageFont.truetype(ff, size=font_size)
            except IOError:
                continue
                
        return ImageFont.load_default()

    def _wrap_text_precise(self, text: str, max_pixel_width: int) -> List[str]:
        """像素级精准换行，支持中英混排"""
        lines = []
        for paragraph in text.split('\n'):
            if not paragraph:
                lines.append("")
                continue
                
            current_line = ""
            # 按字符遍历，对于中英文混排最安全
            for char in paragraph:
                test_line = current_line + char
                # 获取当前测试行的像素宽度
                pixel_width = self.font.getlength(test_line)
                
                if pixel_width <= max_pixel_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = char
            
            if current_line:
                lines.append(current_line)
        return lines

    def _compact_text(self, text: str) -> str:
        """
        将文本中的特殊字符（如换行、制表符）替换为可见的转义字符序列。
        这能将多行碎片化文本压缩为单行连续文本，极大缩减图像的空白空间。
        """
        text = text.replace('\n', '\\n ')
        text = text.replace('\t', '\\t ')
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _draw_line_with_highlights(self, draw: ImageDraw.Draw, x: int, y: int, line: str):
        """
        逐段绘制带有语法高亮的文本
        它会计算每个词的像素宽度，以此为基础向右推进画笔。
        """
        # 如果没有配置高亮，直接整行绘制以节省性能
        if not self.highlight_pattern:
            draw.text((x, y), line, fill=self.text_color, font=self.font)
            return

        # re.split 结合捕获组，会把匹配到的关键词和普通文本交替分割出来
        parts = self.highlight_pattern.split(line)
        current_x = x
        
        for part in parts:
            if not part:
                continue
            # 从字典中获取该片段的颜色，如果是普通文本，回退到默认颜色
            color = self.highlight_dict.get(part, self.text_color)
            draw.text((current_x, y), part, fill=color, font=self.font)
            
            # 精确累加当前片段的像素宽度，作为下一个片段的起始点
            current_x += self.font.getlength(part)

    def render_text_to_image(
        self, 
        obs_text: str, 
        max_width: int = 1024, 
        max_height: int = 2048,
        title: str = "--- SYSTEM OBSERVATION ---",
        use_compact_mode: bool = False  # 控制是否启用紧凑模式
    ) -> Tuple[Image.Image, int]:
        """
        将文本渲染为自适应高度的图像。
        
        Returns:
            (生成好的 PIL 图像, 实际渲染的字符数)
        """
        # --- 紧凑模式预处理 ---
        if use_compact_mode:
            obs_text = self._compact_text(obs_text)
            
        padding_x = 20
        padding_y = 15
        max_text_width = max_width - (padding_x * 2)
        
        # 1. 精准换行
        # 如果开启了紧凑模式，这里的 obs_text 内部将不再包含真实的换行符，
        # _wrap_text_precise 会将其视为一个完整的长段落进行自动折行铺满全图。
        lines = self._wrap_text_precise(obs_text, max_text_width)
        
        # 2. 计算动态高度
        title_area_height = 35
        estimated_text_height = len(lines) * self.line_height
        total_needed_height = padding_y + title_area_height + estimated_text_height + padding_y
        
        # 截断检测与自适应画布高度
        is_truncated = total_needed_height > max_height
        canvas_height = min(total_needed_height, max_height)
        
        # 3. 创建画布
        img = Image.new("RGB", (max_width, int(canvas_height)), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # 4. 绘制标题栏
        draw.text((padding_x, padding_y), title, fill=self.title_color, font=self.font)
        draw.line([(padding_x, padding_y + 20), (max_width - padding_x, padding_y + 20)], fill=(100, 100, 100), width=1)
        
        # 5. 逐行绘制文本
        y_cursor = padding_y + title_area_height
        char_count = 0
        
        for line in lines:
            # 检查剩余空间是否足够绘制下一行和可能的截断警告
            if is_truncated and y_cursor + (self.line_height * 2) > canvas_height:
                warning_text = "... [OBSERVATION TRUNCATED DUE TO LENGTH LIMIT]"
                draw.text((padding_x, y_cursor), warning_text, fill=(255, 100, 100), font=self.font)
                break
            
            if not self.highlight_pattern:
                draw.text((padding_x, y_cursor), line, fill=self.text_color, font=self.font)
            else:
                self._draw_line_with_highlights(draw, padding_x, y_cursor, line)
            char_count += len(line)
            y_cursor += self.line_height

        return img, char_count

    def compress_image_arrays(
        self, 
        images: List[Image.Image], 
        compression_factor: float = 1.0, 
        resample_method=Image.LANCZOS
    ) -> List[Image.Image]:
        """
        批量图像压缩与后处理。
        使用高质量重采样算法（默认 Lanczos）进行降维，保持文字抗锯齿特征。
        
        Args:
            images: 需要压缩的 PIL 图像列表
            compression_factor: 压缩因子（>=1.0）。例如 2.0 表示总体积/面积压缩 2 倍
        """
        if compression_factor <= 1.0:
            return images
            
        compressed_arrays = []
        # 按面积压缩系数，计算长宽单边的缩放比例
        scale = 1.0 / math.sqrt(compression_factor)
        
        for img in images:
            if img is None:
                compressed_arrays.append(None)
                continue
                
            new_width = max(28, int(img.width * scale))
            new_height = max(28, int(img.height * scale))
            
            # LANCZOS 能在向下采样时最大限度保留高频细节（如文字边缘）
            compressed_img = img.resize((new_width, new_height), resample=resample_method)
            compressed_arrays.append(compressed_img)
            
        return compressed_arrays
    
    def _get_image_tokens(self, image: Image.Image, patch_size: int = 14) -> int:
        """
        精确计算视觉 Token。
        基于 ViT 机制，长宽不能整除 Patch Size 时会进行 Padding。
        """
        if image is None:
            return 0
        w_patches = math.ceil(image.width / patch_size)
        h_patches = math.ceil(image.height / patch_size)
        return w_patches * h_patches

    def calculate_compression_ratio_from_ids(
        self, 
        text_token_count: int, 
        original_image: Image.Image, 
        final_image: Optional[Image.Image] = None,
        patch_size: int = 14
    ) -> Dict[str, Any]:
        """
        方法一：传入直接 encode 后的文本 Token 数量。
        适用于外部调用了 tokenizer.encode(text) 后直接将 len 传入的场景，性能最高。
        """
        original_img_tokens = self._get_image_tokens(original_image, patch_size)
        final_img_tokens = self._get_image_tokens(final_image, patch_size) if final_image else original_img_tokens
        
        # 避免除以 0
        ratio = final_img_tokens / text_token_count if text_token_count > 0 else float('inf')
        # 压缩倍率：原始文本 Token 是最终图像 Token 的多少倍（越大说明压缩效果越好）
        compression_factor = text_token_count / final_img_tokens if final_img_tokens > 0 else 0
        
        return {
            "text_tokens_exact": text_token_count,
            "original_image_tokens": original_img_tokens,
            "final_image_tokens": final_img_tokens,
            "token_cost_ratio": f"{ratio:.2%}",       # 图像开销占原文本开销的百分比
            "compression_factor": f"{compression_factor:.2f}x" # 压缩了多少倍
        }

    def calculate_compression_ratio_with_tokenizer(
        self, 
        text: str, 
        original_image: Image.Image, 
        final_image: Optional[Image.Image] = None,
        tokenizer: Any = None,
        patch_size: int = 14
    ) -> Dict[str, Any]:
        """
        方法二：传入文本和分词器实例进行实时计算。
        支持 HuggingFace PreTrainedTokenizer 或 tiktoken 实例。
        如果未传入 tokenizer，则默认加载 tiktoken 的 cl100k_base (GPT-4标准)。
        """
        text_token_count = 0
        
        if tokenizer is not None:
            # 兼容 HuggingFace 的 tokenizer (如 Qwen, LLaMA)
            if hasattr(tokenizer, "encode"):
                # transformers 通常 encode 返回 list，有的也可能是 tensor
                encoded = tokenizer.encode(text)
                text_token_count = len(encoded.tolist()) if hasattr(encoded, "tolist") else len(encoded)
            else:
                raise ValueError("提供的 tokenizer 必须具有 .encode() 方法")
        else:
            # 回退机制：动态加载 tiktoken 作为基准评估
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                text_token_count = len(enc.encode(text))
            except ImportError:
                raise ImportError("未提供 tokenizer，且未安装 tiktoken。请运行 `pip install tiktoken` 或传入自定义分词器。")

        return self.calculate_compression_ratio_from_ids(
            text_token_count=text_token_count,
            original_image=original_image,
            final_image=final_image,
            patch_size=patch_size
        )






# ==========================================
# 测试代码块
# ==========================================

def test_vtc_tool():
    print("--- 开始测试 VTCTool ---")
    
    # 定义高亮关键字与颜色映射
    highlights = [
        {"context": "Action:", "color": (255, 150, 150)},      # 动作标红
        {"context": "Observation:", "color": (150, 255, 150)}, # 观测标绿
        {"context": "Error:", "color": (255, 50, 50)},         # 错误标深红
        {"context": "\\n", "color": (100, 100, 255)}           # 紧凑模式下的转义换行符标蓝
    ]

    # 创建工具实例
    vtc_tool = VTCTool(
        font_size=18, 
        bg_color=(20, 20, 20), 
        text_color=(200, 200, 200), 
        title_color=(80, 180, 80), 
        highlight_configs=highlights
    )

    # 模拟一个冗长的观测文本 (包含 Observation 和 Action 关键字以测试高亮)
    # 生成 100 行以确保能够观察到排版和压缩效果
    print("正在生成模拟数据...")
    long_obs = "\n".join([f"Line {i}: Observation: File not found.\nAction: Retry the operation." for i in range(1, 400)])

    # 渲染文本为图像 (启用紧凑模式)
    print("正在渲染高分辨率原图...")
    img_compact, char_count = vtc_tool.render_text_to_image(
        long_obs, 
        use_compact_mode=True, 
        max_width=2048, # 为了测试截断和换行，宽度设为 2048 
        max_height=4096
    ) 
    
    # 批量压缩 (压缩因子 3.0)
    print("正在执行高质量降采样压缩...")
    img_compressed = vtc_tool.compress_image_arrays([img_compact], compression_factor=4.0)[0]

    # 保存图像到文件
    img_compact.save("output_compact.png")
    img_compressed.save("output_compressed.png")
    
    print(f"\n✅ 渲染完成！")
    print(f"- 原始紧凑模式图像已保存为 output_compact.png (分辨率: {img_compact.width}x{img_compact.height})")
    print(f"- 压缩后图像已保存为 output_compressed.png (分辨率: {img_compressed.width}x{img_compressed.height})")
    
    # 计算精准压缩率 (由于未传入 tokenizer 参数，内部默认调用 tiktoken cl100k_base)
    print("\n--- Token 消耗与压缩率分析 ---")
    try:
        compress_stats = vtc_tool.calculate_compression_ratio_with_tokenizer(
            text=long_obs, 
            original_image=img_compact, 
            final_image=img_compressed,
            patch_size=14 # 假设目标 VLM (如 Qwen-VL) 使用 14x14 Patch
        )
        for k, v in compress_stats.items():
            print(f"{k}: {v}")
    except Exception as e:
        print(f"计算压缩率时出错: {e}")

if __name__ == "__main__":
    test_vtc_tool()