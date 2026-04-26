import math
import re
import textwrap
from typing import Tuple, List, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class VTCTool:
    """
    Visual Text Compression (VTC) Tool (Merged Version)
    整合了 V1（快速折行+字典高亮）与 V2（像素级折行+ID高亮脱水）的功能。
    """
    def __init__(
        self, 
        font_path: Optional[str] = None,
        font_size: int = 16,
        bg_color: Tuple[int, int, int] = (30, 30, 30),      # 深灰底色
        text_color: Tuple[int, int, int] = (220, 220, 220), # 浅灰文字
        title_color: Tuple[int, int, int] = (100, 200, 100),# 绿色标题
        id_color: Tuple[int, int, int] = (255, 215, 0)      # V2 ID加粗高亮颜色 (金黄)
    ):
        self.font_size = font_size
        self.bg_color = bg_color
        self.text_color = text_color
        self.title_color = title_color
        self.id_color = id_color
        
        # 加载常规字体与加粗字体
        self.font_regular, self.font_bold = self._load_fonts(font_path, font_size)
        self.font = self.font_regular # 兼容 V1 的变量名调用
        
        # 预计算行高 (使用常规字体)
        bbox = self.font_regular.getbbox("A")
        self.line_height = (bbox[3] - bbox[1]) + 8  # 字体高度 + 8px 行距

        # ==========================================
        #  ID 高亮 (正则)
        # ==========================================
        self.id_pattern = re.compile(r'(\[\d+\])')

    def _load_fonts(self, font_path: Optional[str], font_size: int) -> Tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
        """加载常规与加粗字体，带有多级回退机制"""
        font_reg, font_bld = None, None
        
        # 如果指定了外部字体路径，优先加载
        if font_path:
            try:
                font_reg = ImageFont.truetype(font_path, size=font_size)
                font_bld = font_reg # 自定义字体默认加粗用同一个（除非额外指定加粗包）
            except IOError:
                print(f"[Warning] 无法加载指定字体 {font_path}，将尝试系统默认字体。")

        if not font_reg:
            font_pairs = [
                ("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf"), # Linux
                ("Consolas.ttf", "Consolab.ttf"),                  # Windows
                ("Menlo.ttc", "Menlo.ttc"),                        # macOS
                ("simhei.ttf", "simhei.ttf")                       # 中文回退
            ]
            for reg_path, bld_path in font_pairs:
                try:
                    font_reg = ImageFont.truetype(reg_path, size=font_size)
                    try:
                        font_bld = ImageFont.truetype(bld_path, size=font_size)
                    except IOError:
                        font_bld = font_reg
                    break
                except IOError:
                    continue
                    
        if not font_reg:
            font_reg = ImageFont.load_default()
            font_bld = font_reg
            
        return font_reg, font_bld

    # ==========================================
    # 文本预处理与折行模块 (区分 V1 和 V2)
    # ==========================================
    def _compact_text(self, text: str) -> str:
        """【V1 专用】将换行替换为可见转义字符"""
        text = re.sub(r'\\n|\\t|\n|\t', ' ', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def _preprocess_text(self, text: str) -> str:
        """【V2 专用】彻底清理文本，移除换行和 URL"""
        text = re.sub(r'\\n|\\t|\n|\t', ' ', text)
        text = re.sub(r"url:\s*https?://\S+", "", text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def _wrap_text_fast(self, text: str, max_text_width: int) -> list:
        """【V1 专用】使用 textwrap 进行极速估算折行"""
        if not text: return []
        avg_char_width = self.font.size * 0.6 
        chars_per_line = max(10, int(max_text_width / avg_char_width))
        result_lines = []
        paragraphs = text.split('\n')
        for paragraph in paragraphs:
            if not paragraph.strip():
                result_lines.append("")
                continue
            wrapped = textwrap.wrap(
                paragraph, width=chars_per_line, expand_tabs=False, 
                replace_whitespace=False, break_long_words=True
            )
            result_lines.extend(wrapped)
        return result_lines

    def _wrap_text_pixel_precise(self, text: str, max_pixel_width: int) -> List[str]:
        """【V2 专用】使用字体 getlength 进行像素级精确折行"""
        lines = []
        current_line = ""
        for char in text:
            test_line = current_line + char
            pixel_width = self.font_regular.getlength(test_line)
            if pixel_width <= max_pixel_width:
                current_line = test_line
            else:
                if current_line: lines.append(current_line)
                current_line = char
        if current_line: lines.append(current_line)
        return lines

    # ==========================================
    # 文本高亮绘制模块
    # ==========================================
    def _draw_line_with_id_highlights(self, draw: ImageDraw.Draw, x: int, y: int, line: str):
        """【V2 专用】基于 [ID] 格式进行加粗+金黄高亮"""
        parts = self.id_pattern.split(line)
        current_x = x
        for part in parts:
            if not part: continue
            if self.id_pattern.match(part):
                font = self.font_bold
                color = self.id_color
            else:
                font = self.font_regular
                color = self.text_color
            draw.text((current_x, y), part, fill=color, font=font)
            current_x += font.getlength(part)

    # ==========================================
    # 核心渲染入口 1 (对应原始 V1 逻辑)
    # ==========================================
    def render_text_to_image(
        self, 
        obs_text: str, 
        max_width: int = 1024, 
        max_height: int = 2048,
        title: str = "--- SYSTEM OBSERVATION ---",
        use_compact_mode: bool = False
    ) -> Tuple[Image.Image, int]:
        
        # 🔴 终极防卡顿：入口级暴力截断
        MAX_SAFE_CHARS = 25000 
        if len(obs_text) > MAX_SAFE_CHARS:
            obs_text = obs_text[:MAX_SAFE_CHARS]

        # --- 紧凑模式预处理 ---
        if use_compact_mode:
            obs_text = self._compact_text(obs_text)

        padding_x = 20
        padding_y = 15
        title_area_height = 35
        max_text_width = max_width - (padding_x * 2)

        # 1. 精准换行 (使用 V1 的快速换行)
        lines = self._wrap_text_fast(obs_text, max_text_width)
        
        # 2. 计算动态高度
        estimated_text_height = len(lines) * self.line_height
        total_needed_height = padding_y + title_area_height + estimated_text_height + padding_y
        
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
            if is_truncated and y_cursor + (self.line_height * 2) > canvas_height:
                warning_text = "... [OBSERVATION TRUNCATED DUE TO LENGTH LIMIT]"
                draw.text((padding_x, y_cursor), warning_text, fill=(255, 100, 100), font=self.font)
                break
            
            self._draw_line_with_id_highlights(draw, padding_x, y_cursor, line)
                
            char_count += len(line)
            y_cursor += self.line_height

        return img, char_count

    # ==========================================
    # 核心渲染入口 2 (对应简化版 V2 逻辑)
    # ==========================================
    def render_text_to_image_simple(
        self, 
        obs_text: str, 
        width: int = 1024, 
        aspect_ratio: str = "4:3",
        title: str = "--- SYSTEM OBSERVATION ---"
    ) -> Tuple[Image.Image, int]:
        """按照指定比例 (1:1 或 4:3) 渲染高度优化的图像"""
        
        # --- 1. 执行数据脱水 (使用 V2 的预处理) ---
        cleaned_text = self._preprocess_text(obs_text)
        
        # --- 2. 确定画布尺寸 ---
        if aspect_ratio == "1:1":
            height = width
        elif aspect_ratio == "4:3":
            height = int(width * 3 / 4)
        else:
            raise ValueError("aspect_ratio 仅支持 '1:1' 或 '4:3'")
            
        padding_x = 20
        padding_y = 15
        max_text_width = width - (padding_x * 2)
        
        # --- 3. 文本自动折行铺满 (使用 V2 的像素级精确换行) ---
        lines = self._wrap_text_pixel_precise(cleaned_text, max_text_width)
        
        # --- 4. 创建画布并渲染 ---
        img = Image.new("RGB", (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        title_area_height = 35
        draw.text((padding_x, padding_y), title, fill=self.title_color, font=self.font_bold)
        draw.line([(padding_x, padding_y + 25), (width - padding_x, padding_y + 25)], fill=(100, 100, 100), width=1)
        
        y_cursor = padding_y + title_area_height
        char_count = 0
        
        for line in lines:
            if y_cursor + (self.line_height * 2) > height:
                warning_text = "... [OBSERVATION TRUNCATED]"
                draw.text((padding_x, y_cursor), warning_text, fill=(255, 100, 100), font=self.font_bold)
                break
                
            # 使用 V2 的加粗 ID 渲染
            self._draw_line_with_id_highlights(draw, padding_x, y_cursor, line)
            
            char_count += len(line)
            y_cursor += self.line_height

        return img, char_count

    # ==========================================
    # 以下为共享通用工具函数 (未改变)
    # ==========================================
    def compress_image_arrays(self, images: List[Image.Image], compression_factor: float = 1.0, resample_method=Image.LANCZOS) -> List[Image.Image]:
        if compression_factor <= 1.0: return images
        compressed_arrays = []
        scale = 1.0 / math.sqrt(compression_factor)
        for img in images:
            if img is None:
                compressed_arrays.append(None)
                continue
            new_width = max(28, int(img.width * scale))
            new_height = max(28, int(img.height * scale))
            compressed_img = img.resize((new_width, new_height), resample=resample_method)
            compressed_arrays.append(compressed_img)
        return compressed_arrays
    
    def _get_image_tokens(self, image: Image.Image, patch_size: int = 14) -> int:
        if image is None: return 0
        w_patches = math.ceil(image.width / patch_size)
        h_patches = math.ceil(image.height / patch_size)
        return w_patches * h_patches

    def calculate_compression_ratio_from_ids(self, text_token_count: int, original_image: Image.Image, final_image: Optional[Image.Image] = None, patch_size: int = 14) -> Dict[str, Any]:
        original_img_tokens = self._get_image_tokens(original_image, patch_size)
        final_img_tokens = self._get_image_tokens(final_image, patch_size) if final_image else original_img_tokens
        ratio = final_img_tokens / text_token_count if text_token_count > 0 else float('inf')
        compression_factor = text_token_count / final_img_tokens if final_img_tokens > 0 else 0
        return {
            "text_tokens_exact": text_token_count,
            "original_image_tokens": original_img_tokens,
            "final_image_tokens": final_img_tokens,
            "token_cost_ratio": f"{ratio:.2%}",
            "compression_factor": f"{compression_factor:.2f}x"
        }

    def calculate_compression_ratio_with_tokenizer(self, text: str, original_image: Image.Image, final_image: Optional[Image.Image] = None, tokenizer: Any = None, patch_size: int = 14) -> Dict[str, Any]:
        text_token_count = 0
        if tokenizer is not None:
            if hasattr(tokenizer, "encode"):
                encoded = tokenizer.encode(text)
                text_token_count = len(encoded.tolist()) if hasattr(encoded, "tolist") else len(encoded)
            else:
                raise ValueError("提供的 tokenizer 必须具有 .encode() 方法")
        else:
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                text_token_count = len(enc.encode(text))
            except ImportError:
                raise ImportError("未提供 tokenizer，且未安装 tiktoken。请运行 `pip install tiktoken` 或传入自定义分词器。")

        return self.calculate_compression_ratio_from_ids(text_token_count, original_image, final_image, patch_size)