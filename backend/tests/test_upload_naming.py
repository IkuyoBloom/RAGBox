"""上传文件名的规范化测试。

这个函数看起来只是「清理一下字符串」，但它有两个不能出错的地方：

1. **路径穿越**。返回值会进数据库、日志、接口响应。若 `../../etc/passwd`
   被原样保留，虽然本项目落盘用的是 `<doc_id><后缀>`（不受影响），
   但任何一处把它当路径用的下游代码都会出事 —— 安全处理要在入口做完。

2. **文件夹导入的同名区分**。选择文件夹时浏览器会带上相对路径，
   把它剥掉会让 `a/README.md` 与 `b/README.md` 变成两个同名文档，
   用户在界面上完全分不清哪个是哪个。所以路径要**保留**，
   这与第 1 点的「清理」并不矛盾：保留的是相对结构，去掉的是危险成分。
"""

from __future__ import annotations

import pytest

from app.api.documents import MAX_NAME_CHARS, _safe_display_name


class TestNormalNames:
    def test_普通文件名原样保留(self):
        assert _safe_display_name("论文.pdf") == "论文.pdf"

    def test_带空格与中文(self):
        assert _safe_display_name("2024 年度 总结 final.docx") == "2024 年度 总结 final.docx"


class TestRelativePaths:
    def test_保留相对路径(self):
        """文件夹导入的核心诉求：不同子目录的同名文件必须能区分。"""
        assert _safe_display_name("论文/2024/注意力机制.pdf") == "论文/2024/注意力机制.pdf"

    def test_同名文件在不同子目录下不会混淆(self):
        a = _safe_display_name("项目A/README.md")
        b = _safe_display_name("项目B/README.md")
        assert a != b
        assert a.endswith("README.md") and b.endswith("README.md")

    def test_反斜杠分隔符被统一(self):
        """Windows 客户端可能传反斜杠。"""
        assert _safe_display_name("论文\\2024\\注意力.pdf") == "论文/2024/注意力.pdf"

    def test_混合分隔符(self):
        assert _safe_display_name("a\\b/c\\d.txt") == "a/b/c/d.txt"


class TestTraversalSafety:
    def test_去掉上级目录引用(self):
        assert _safe_display_name("../../etc/passwd") == "etc/passwd"

    def test_去掉中间的上级目录引用(self):
        """`..` 要按真实路径语义弹出上一级，而不是简单丢弃 ——
        简单丢弃会得到 `docs/secret.txt` 这种误导性名字，
        看名字以为文件在 docs 里，实际它来自别处。"""
        assert _safe_display_name("docs/../../secret.txt") == "secret.txt"

    def test_弹出后仍有剩余目录(self):
        assert _safe_display_name("a/b/../c.pdf") == "a/c.pdf"

    def test_去掉开头的斜杠(self):
        assert _safe_display_name("/absolute/path.pdf") == "absolute/path.pdf"

    def test_去掉盘符(self):
        assert _safe_display_name("C:/Users/alice/论文.pdf") == "Users/alice/论文.pdf"

    def test_去掉盘符且反斜杠(self):
        assert _safe_display_name("D:\\Downloads\\报告.docx") == "Downloads/报告.docx"

    def test_UNC_路径(self):
        # \\server\share\a.pdf -> server/share/a.pdf
        assert _safe_display_name("\\\\server\\share\\a.pdf") == "server/share/a.pdf"

    def test_单个点段被丢弃(self):
        assert _safe_display_name("./a/./b.txt") == "a/b.txt"

    def test_纯上级引用退化为_unnamed(self):
        assert _safe_display_name("..") == "unnamed"
        assert _safe_display_name("../") == "unnamed"
        # a/../.. : 先弹出 a 得空，再在根上遇到 .. 直接丢弃 -> 什么都不剩
        assert _safe_display_name("a/../..") == "unnamed"
        # 但 a/.. 只是回到根，剩不下东西也算 unnamed
        assert _safe_display_name("a/..") == "unnamed"

    def test_空字符串(self):
        assert _safe_display_name("") == "unnamed"

    def test_全是分隔符(self):
        assert _safe_display_name("///") == "unnamed"
        assert _safe_display_name("\\\\\\") == "unnamed"


class TestLengthLimit:
    def test_超长名被截断(self):
        long_name = "目录/" + "x" * 2000 + ".pdf"
        out = _safe_display_name(long_name)
        assert len(out) <= MAX_NAME_CHARS

    def test_截断后仍保留扩展名(self):
        """扩展名决定用哪个解析器，截断不能把它切掉。"""
        out = _safe_display_name("目录/" + "x" * 2000 + ".pdf")
        assert out.endswith(".pdf")

    def test_截断用省略号标记(self):
        out = _safe_display_name("目录/" + "x" * 2000 + ".pdf")
        assert out.startswith("…")

    def test_恰好等于上限时不截断(self):
        name = "x" * MAX_NAME_CHARS
        out = _safe_display_name(name)
        assert out == name
        assert not out.startswith("…")


class TestExtensionStillDetected:
    """规范化之后，类型识别必须仍然正确 —— 否则整个上传会被拒。"""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("论文.pdf", "pdf"),
            ("论文/2024/注意力机制.pdf", "pdf"),
            ("D:\\Downloads\\报告.docx", "docx"),
            ("a/b/c.md", "markdown"),
            ("a/b/c.txt", "text"),
        ],
    )
    def test_各种路径形态都能识别类型(self, raw, expected):
        from app.core.ingest import detect_type

        assert detect_type(_safe_display_name(raw)) == expected
