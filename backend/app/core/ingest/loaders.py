"""文档解析：把各种格式统一成带页码的文本块。

输出粒度刻意选在「页 / 段落块」而不是「整篇字符串」，因为：
  * 切分时要保留来源页码，才能在界面上做「引用溯源到第几页」；
  * PDF 一页往往就是语义完整的单元，按页切比按全文切更干净。

已知边界（论文「局限与展望」里要写）：
  * 扫描版 PDF 没有文本层，本模块产不出内容，需要 OCR 前置 —— 当前不支持，
    但会明确报错提示，而不是返回空文档让人以为索引成功了。
  * 复杂排版（多栏、公式）的抽取顺序依赖 pypdf，可能串行。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 支持的扩展名 -> 内部类型
SUPPORTED_TYPES: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".log": "text",
}


class ParseError(RuntimeError):
    """解析失败，message 直接展示给用户。"""


@dataclass(slots=True)
class TextBlock:
    text: str
    page: int | None = None


@dataclass(slots=True)
class ParsedDocument:
    blocks: list[TextBlock] = field(default_factory=list)
    page_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)


# ----------------------------------------------------------------------
# 文本归一化
# ----------------------------------------------------------------------
# 句末标点：出现在行尾说明这是真正的换行，不该合并
_SENTENCE_END = "。！？!?；;：:”\"')）】》…—"
_CJK = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


def normalize_text(text: str) -> str:
    """清洗 PDF / Word 抽取出来的文本。

    最要命的问题是「硬换行」：PDF 里每个视觉行都是一个 \\n，
    于是「梯度下降法是一种」和「常用的优化算法」被拆成两行。
    如果不处理，切出来的片会到处断句，检索质量明显变差。
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    # 连字符断词（英文 PDF 常见）：opti-\nmization -> optimization
    text = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)

    lines = [ln.rstrip() for ln in text.split("\n")]

    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            merged.append("")
            continue
        if not merged or not merged[-1]:
            merged.append(stripped)
            continue

        prev = merged[-1]
        prev_last = prev[-1]
        # 上一行以句末标点收尾 -> 是真换行，保留
        if prev_last in _SENTENCE_END:
            merged.append(stripped)
            continue

        # 中文之间直接拼接；中英/英文之间补空格，避免粘连成新词
        if _CJK.search(prev_last) or _CJK.search(stripped[0]):
            merged[-1] = prev + stripped
        else:
            merged[-1] = prev + " " + stripped

    text = "\n".join(merged)
    # 压掉三个以上连续空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ----------------------------------------------------------------------
# 各格式解析
# ----------------------------------------------------------------------
def load_pdf(path: Path) -> ParsedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ParseError("缺少 pypdf 依赖，无法解析 PDF") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ParseError(f"PDF 打开失败：{type(exc).__name__}: {exc}") from exc

    if reader.is_encrypted:
        # 很多「加密」PDF 只是空密码权限位，试一次再放弃
        try:
            if reader.decrypt("") == 0:
                raise ParseError("PDF 已加密且需要密码，无法解析")
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(f"PDF 加密处理失败：{exc}") from exc

    blocks: list[TextBlock] = []
    empty_pages = 0
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            raw = ""
            page_note = f"（第 {i} 页抽取失败：{type(exc).__name__}）"
        else:
            page_note = ""
        cleaned = normalize_text(raw)
        if not cleaned:
            empty_pages += 1
            continue
        blocks.append(TextBlock(text=cleaned + ("" if not page_note else page_note), page=i))

    if not blocks:
        raise ParseError(
            "PDF 中没有抽取到任何文本。若这是扫描件（图片型 PDF），"
            "需要先做 OCR 才能建立索引——当前版本不支持。"
        )

    meta: dict[str, Any] = {
        "page_count": len(reader.pages),
        "empty_pages": empty_pages,
    }
    try:
        info = reader.metadata or {}
        if info.get("/Title"):
            meta["title"] = str(info["/Title"])[:300]
        if info.get("/Author"):
            meta["author"] = str(info["/Author"])[:200]
    except Exception:
        pass

    return ParsedDocument(blocks=blocks, page_count=len(reader.pages), meta=meta)


def load_docx(path: Path) -> ParsedDocument:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover
        raise ParseError("缺少 python-docx 依赖，无法解析 Word 文档") from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ParseError(f"Word 文档打开失败：{type(exc).__name__}: {exc}") from exc

    pieces: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if text:
            pieces.append(text)

    # 表格单独抽：转成「单元格 | 单元格」的文本行，比丢弃更利于检索
    table_rows = 0
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                pieces.append(" | ".join(cells))
                table_rows += 1

    cleaned = normalize_text("\n\n".join(pieces))
    if not cleaned:
        raise ParseError("Word 文档中没有可提取的文本内容。")

    blocks = [TextBlock(text=cleaned, page=None)]
    return ParsedDocument(
        blocks=blocks,
        page_count=0,
        meta={"paragraphs": len(document.paragraphs), "table_rows": table_rows},
    )


def load_text_file(path: Path) -> ParsedDocument:
    """纯文本 / Markdown。

    编码探测顺序：BOM -> UTF-8 -> GBK。中文 Windows 上 GBK 的 txt 很常见，
    直接用 UTF-8 读会炸出一堆替换字符，检索必然失效。
    """
    raw_bytes = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw_bytes.decode("utf-8", errors="replace")

    cleaned = normalize_text(text)
    if not cleaned:
        raise ParseError("文件内容为空。")
    return ParsedDocument(blocks=[TextBlock(text=cleaned, page=None)], page_count=0)


_LOADERS = {
    "pdf": load_pdf,
    "docx": load_docx,
    "markdown": load_text_file,
    "text": load_text_file,
}


def detect_type(filename: str) -> str | None:
    return SUPPORTED_TYPES.get(Path(filename).suffix.lower())


def load_document(path: Path, file_type: str | None = None) -> ParsedDocument:
    ftype = file_type or detect_type(path.name)
    if ftype is None:
        raise ParseError(
            f"不支持的文件类型：{path.suffix or '(无扩展名)'}。"
            f"当前支持：{'、'.join(sorted(SUPPORTED_TYPES))}"
        )
    loader = _LOADERS.get(ftype)
    if loader is None:  # pragma: no cover
        raise ParseError(f"没有为类型 `{ftype}` 注册解析器")
    return loader(path)


def supported_extensions() -> list[str]:
    return sorted(SUPPORTED_TYPES.keys())


__all__ = [
    "SUPPORTED_TYPES",
    "ParseError",
    "ParsedDocument",
    "TextBlock",
    "detect_type",
    "load_document",
    "load_docx",
    "load_pdf",
    "load_text_file",
    "normalize_text",
    "supported_extensions",
]
