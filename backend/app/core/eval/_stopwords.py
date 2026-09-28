"""评估模块共用的中文停用词表。

## 为什么单独放一个内部模块

停用词会被两处用到，而且**必须一致**：

  * `metrics._normalize_tokens` —— 算 token F1 时过滤虚词；
  * `dataset.extract_keywords` —— 从语料里抽关键词时过滤虚词。

如果各写一份，两边的判断标准会慢慢漂移：某天给指标侧补了一个词，
关键词侧没补，于是同一个词一边被当噪声、一边被当术语，
评估结论就变得无法解释。共享一份常量能杜绝这类"静默不一致"。

## 收录原则

只收**高频虚词与结构性词汇**（介词、连词、代词、助词、常见的动词性框架词）。
刻意**不激进裁剪**：

  * 「不」「没有」「非」这类否定词一旦被删，「有监督学习」与「没有监督学习」
    会得到完全相同的 token 集合，相似度指标直接失去意义；
  * 「可以」「需要」虽然信息量低，但它们常出现在技术定义里，
    删掉对指标影响很小，留着也不会造成误导。

一句话：宁可留一点噪声，也不能删掉改变语义的词。
"""

from __future__ import annotations

# 中文虚词 / 结构词
STOPWORDS: frozenset[str] = frozenset(
    """的 了 和 是 在 我 你 他 她 它 我们 你们 他们 有 就 不 也 都 很 到 说 要 去 会 着
没有 看 好 自己 这 那 之一 与 及 或 而 但 被 把 让 从 对 为 以 于 中 上 下 后 前 里 时 个 吗 呢 吧 啊
地 得 之 其 此 该 等 一个 一些 一种 可以 能够 需要 进行 通过 以及 因为 所以 但是 如果
那么 就是 还是 只是 已经 正在 将要 应该 可能 表示 用于 属于 由于 关于 根据 按照 除了 例如 比如
这个 那个 什么 怎么 为什么 哪些 哪个 这样 那样 时候 方面 情况 问题 内容 部分 方式 方法 过程
一种 一定 一起 一直 一般 一样 主要 重要 相关 不同 相同 比如 后面 前面 上面 下面""".split()
)

# 英文功能词（技术文档里中英混排很常见）
STOPWORDS_EN: frozenset[str] = frozenset(
    """the a an of to in on at for and or is are was were be been being this that these those with by as it
its from not no do does did can could will would should may might must have has had""".split()
)

ALL_STOPWORDS: frozenset[str] = STOPWORDS | STOPWORDS_EN

__all__ = ["ALL_STOPWORDS", "STOPWORDS", "STOPWORDS_EN"]
