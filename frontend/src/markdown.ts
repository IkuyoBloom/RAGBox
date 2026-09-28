/**
 * 极简 Markdown 解析器（纯函数，不含 React）。
 *
 * ## 为什么手写而不是引入 react-markdown
 *
 * 1. **安全性**：本解析器只产出数据结构，渲染层用 React 元素表达，
 *    全程不用 `dangerouslySetInnerHTML` —— 于是 XSS 在结构上就不可能发生。
 *    引入现成库通常要走 HTML 字符串再塞进 DOM，反而多一个风险面。
 * 2. **引用角标要能点**：回答里的 `[n]` 必须渲染成可点击的溯源按钮。
 *    在行内解析这一层同时处理 `**加粗**` 和 `[n]`，是最自然的做法；
 *    用库的话得事后去 HTML 里做替换，既脆弱又容易破坏标签。
 * 3. **可单测**：纯函数，直接在 Node 里跑，不需要 jsdom。
 *
 * ## 刻意不支持的东西
 *
 * * **下划线强调 `_x_`**：语料里全是 `零电荷_EP15_3/-70-a-10.txt` 这类文件名，
 *   把 `_` 当强调符会把文件名切得面目全非。这是为真实数据做的取舍。
 * * **原始 HTML**：一律当纯文本显示。回答来自模型，不该被当成可信标记。
 * * 图片、脚注、任务列表等本场景用不到的语法。
 */

// ======================================================================
// 行内
// ======================================================================

export type Inline =
  | { type: 'text'; text: string }
  | { type: 'strong'; children: Inline[] }
  | { type: 'em'; children: Inline[] }
  | { type: 'code'; text: string }
  | { type: 'link'; href: string; children: Inline[] }
  | { type: 'citation'; index: number; raw: string }
  | { type: 'br' }

/** 只放行安全的协议，挡掉 `javascript:` 这类。 */
export function safeHref(href: string): string | null {
  const trimmed = (href || '').trim()
  if (!trimmed) return null
  if (/^(https?:\/\/|mailto:)/i.test(trimmed)) return trimmed
  if (trimmed.startsWith('/') || trimmed.startsWith('#')) return trimmed
  return null
}

const RE_CITATION = /^\[(\d{1,2})\]/
const RE_LINK = /^\[([^\]]*)\]\(([^)\s]+)\)/
const RE_CODE = /^`([^`]+)`/
const RE_STRONG = /^(\*\*)([\s\S]+?)\1/
const RE_EM = /^\*([^*\n]+?)\*/

/**
 * 解析一段行内文本。
 *
 * 顺序有讲究：**引用要排在链接前面判断**是错的（`[1](url)` 这种不存在），
 * 但链接必须排在引用前面 —— 否则 `[文字](url)` 会被当成 `[` + 文字 + `]`
 * 之后把 `(url)` 当普通文本。实测模型会输出 `[《文件名》](...)` 形式，所以顺序要紧。
 */
export function parseInline(src: string): Inline[] {
  const out: Inline[] = []
  let buf = ''
  let i = 0

  const flush = () => {
    if (buf) {
      out.push({ type: 'text', text: buf })
      buf = ''
    }
  }

  while (i < src.length) {
    const rest = src.slice(i)

    // 换行：聊天场景下单换行应当保留（模型常把每句话写成一行）
    if (src[i] === '\n') {
      flush()
      out.push({ type: 'br' })
      i += 1
      continue
    }

    // 链接 `[文字](url)` —— 必须早于引用判断
    let m = RE_LINK.exec(rest)
    if (m) {
      const href = safeHref(m[2])
      flush()
      if (href) {
        out.push({ type: 'link', href, children: parseInline(m[1]) })
      } else {
        // 协议不安全就退化成纯文本，不给出可点的链接
        out.push({ type: 'text', text: m[0] })
      }
      i += m[0].length
      continue
    }

    // 引用角标 `[3]`
    m = RE_CITATION.exec(rest)
    if (m) {
      flush()
      out.push({ type: 'citation', index: Number(m[1]), raw: m[0] })
      i += m[0].length
      continue
    }

    // 行内代码 `x`
    m = RE_CODE.exec(rest)
    if (m) {
      flush()
      out.push({ type: 'code', text: m[1] })
      i += m[0].length
      continue
    }

    // 加粗 **x**
    m = RE_STRONG.exec(rest)
    if (m) {
      flush()
      out.push({ type: 'strong', children: parseInline(m[2]) })
      i += m[0].length
      continue
    }

    // 斜体 *x*。注意不处理 `_x_`：文件名里全是下划线，会把名字切碎。
    m = RE_EM.exec(rest)
    if (m) {
      flush()
      out.push({ type: 'em', children: parseInline(m[1]) })
      i += m[0].length
      continue
    }

    buf += src[i]
    i += 1
  }

  flush()
  return out
}

// ======================================================================
// 块级
// ======================================================================

export interface ListItem {
  /** 该项自身的行内内容（可能含 `**加粗**` 与 `[n]`） */
  inlines: Inline[]
  /** 缩进更深的后续行，可能含嵌套列表 */
  children: Block[]
}

export type Block =
  | { type: 'paragraph'; inlines: Inline[] }
  | { type: 'heading'; level: number; inlines: Inline[] }
  | { type: 'list'; ordered: boolean; start: number; items: ListItem[] }
  | { type: 'code'; lang: string; text: string }
  | { type: 'quote'; inlines: Inline[] }
  | { type: 'hr' }
  | { type: 'table'; header: string[]; rows: string[][] }

const RE_FENCE = /^\s*(`{3,}|~{3,})\s*([\w+-]*)\s*$/
const RE_HEADING = /^(#{1,6})\s+(.*)$/
const RE_HR = /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/
const RE_UL = /^(\s*)([-*+])\s+(.*)$/
const RE_OL = /^(\s*)(\d{1,3})[.)]\s+(.*)$/
const RE_QUOTE = /^\s*>\s?(.*)$/
const RE_TABLE_ROW = /^\s*\|(.+)\|\s*$/
const RE_TABLE_SEP = /^\s*\|?[\s:|-]+\|[\s:|-]*$/

/** 把连续的列表项收集起来，处理一层嵌套。 */
function parseList(
  lines: string[],
  startIdx: number,
  baseIndent: number,
  ordered: boolean,
): { block: Block; next: number } {
  const items: ListItem[] = []
  let i = startIdx
  let listStart = 1

  while (i < lines.length) {
    const line = lines[i]
    if (!line.trim()) {
      // 空行后如果还是同级列表项，说明列表继续；否则结束
      const nxt = lines[i + 1] ?? ''
      const m = ordered ? RE_OL.exec(nxt) : RE_UL.exec(nxt)
      if (m && m[1].length >= baseIndent) {
        i += 1
        continue
      }
      break
    }

    const m = ordered ? RE_OL.exec(line) : RE_UL.exec(line)
    if (!m) break
    const indent = m[1].length
    if (indent < baseIndent) break
    if (indent > baseIndent) {
      // 更深的缩进交给上一层的 children 处理（这里不该出现）
      break
    }

    const content = m[3]
    if (items.length === 0) listStart = ordered ? Number(m[2]) : 1
    i += 1

    // 收集该项的续行（缩进更深、且不是新列表项起始）
    const childLines: string[] = []
    while (i < lines.length) {
      const nxt = lines[i]
      if (!nxt.trim()) {
        // 空行：后面若仍是本项的子内容才继续
        const after = lines[i + 1] ?? ''
        if (!after.trim()) break
        const afterIndent = after.length - after.trimStart().length
        if (afterIndent <= baseIndent) break
        childLines.push('')
        i += 1
        continue
      }
      const nxtIndent = nxt.length - nxt.trimStart().length
      if (nxtIndent <= baseIndent) break
      // 子列表项：递归留给下面统一处理
      if (RE_UL.exec(nxt) || RE_OL.exec(nxt)) break
      childLines.push(nxt.trim())
      i += 1
    }

    const children: Block[] = []
    // 子列表（缩进更深的列表项）
    while (i < lines.length) {
      const nxt = lines[i]
      const sub = RE_UL.exec(nxt) ?? RE_OL.exec(nxt)
      if (!sub) break
      const subIndent = sub[1].length
      if (subIndent <= baseIndent) break
      const subOrdered = RE_OL.test(nxt)
      const parsed = parseList(lines, i, subIndent, subOrdered)
      children.push(parsed.block)
      i = parsed.next
    }
    if (childLines.length) {
      children.unshift({ type: 'paragraph', inlines: parseInline(childLines.join('\n')) })
    }

    items.push({ inlines: parseInline(content), children })
  }

  return {
    block: { type: 'list', ordered, start: listStart, items },
    next: i,
  }
}

/** 把 Markdown 源码切成块级结构。 */
export function parseMarkdown(src: string): Block[] {
  const lines = (src ?? '').replace(/\r\n?/g, '\n').split('\n')
  const blocks: Block[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    // 空行
    if (!line.trim()) {
      i += 1
      continue
    }

    // 代码块
    const fence = RE_FENCE.exec(line)
    if (fence) {
      const marker = fence[1]
      const lang = fence[2] || ''
      const body: string[] = []
      i += 1
      while (i < lines.length && !new RegExp(`^\\s*${marker[0]}{3,}\\s*$`).test(lines[i])) {
        body.push(lines[i])
        i += 1
      }
      i += 1 // 跳过收尾的 ```
      blocks.push({ type: 'code', lang, text: body.join('\n') })
      continue
    }

    // 分隔线（要早于列表判断，否则 `---` 会被当成列表项）
    if (RE_HR.test(line)) {
      blocks.push({ type: 'hr' })
      i += 1
      continue
    }

    // 标题
    const heading = RE_HEADING.exec(line)
    if (heading) {
      blocks.push({
        type: 'heading',
        level: heading[1].length,
        inlines: parseInline(heading[2].trim()),
      })
      i += 1
      continue
    }

    // 引用
    const quote = RE_QUOTE.exec(line)
    if (quote) {
      const body: string[] = []
      while (i < lines.length) {
        const m = RE_QUOTE.exec(lines[i])
        if (!m) break
        body.push(m[1])
        i += 1
      }
      blocks.push({ type: 'quote', inlines: parseInline(body.join('\n')) })
      continue
    }

    // 表格：本行是 |...|，下一行是分隔行
    if (RE_TABLE_ROW.test(line) && i + 1 < lines.length && RE_TABLE_SEP.test(lines[i + 1])) {
      const cells = (row: string) =>
        row
          .replace(/^\s*\|/, '')
          .replace(/\|\s*$/, '')
          .split('|')
          .map((c) => c.trim())
      const header = cells(line)
      i += 2
      const rows: string[][] = []
      while (i < lines.length && RE_TABLE_ROW.test(lines[i])) {
        rows.push(cells(lines[i]))
        i += 1
      }
      blocks.push({ type: 'table', header, rows })
      continue
    }

    // 列表
    const ul = RE_UL.exec(line)
    const ol = RE_OL.exec(line)
    if (ul || ol) {
      const indent = (ul ?? ol)![1].length
      const parsed = parseList(lines, i, indent, Boolean(ol))
      blocks.push(parsed.block)
      i = parsed.next
      continue
    }

    // 普通段落：连续非空行合成一段（空行分段）
    const para: string[] = []
    while (i < lines.length && lines[i].trim()) {
      const l = lines[i]
      if (
        RE_FENCE.test(l) ||
        RE_HEADING.test(l) ||
        RE_HR.test(l) ||
        RE_QUOTE.test(l) ||
        RE_UL.test(l) ||
        RE_OL.test(l)
      ) {
        break
      }
      para.push(l.trim())
      i += 1
    }
    if (para.length) {
      blocks.push({ type: 'paragraph', inlines: parseInline(para.join('\n')) })
    } else {
      // 兜底：本行谁都不匹配，按普通文本吃掉，避免死循环
      blocks.push({ type: 'paragraph', inlines: parseInline(line.trim()) })
      i += 1
    }
  }

  return blocks
}

/** 收集一段文本里所有引用编号（测试与统计用）。 */
export function collectCitations(src: string): number[] {
  const found: number[] = []
  const walk = (inlines: Inline[]) => {
    for (const node of inlines) {
      if (node.type === 'citation') found.push(node.index)
      else if (node.type === 'strong' || node.type === 'em' || node.type === 'link') walk(node.children)
    }
  }
  for (const block of parseMarkdown(src)) {
    if (block.type === 'paragraph' || block.type === 'heading' || block.type === 'quote') {
      walk(block.inlines)
    } else if (block.type === 'list') {
      for (const item of block.items) {
        walk(item.inlines)
        for (const child of item.children) {
          if (child.type === 'paragraph') walk(child.inlines)
        }
      }
    }
  }
  return found
}
