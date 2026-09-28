/**
 * Markdown 解析器验证。
 *
 * 样例尽量取自**模型真实输出**（从实际问答里摘下来的），因为
 * 手写 Markdown 解析器最容易在真实文本上翻车：
 *   * 语料里的文件名带下划线（`零电荷_EP15_3/-70-a-10.txt`），
 *     一旦把 `_` 当强调符就会把文件名切碎；
 *   * 引用编号 `[n]` 会出现在加粗里、列表项里、表格单元格里；
 *   * 流式输出中途 `**` 常常是没闭合的。
 *
 * 用法（在 frontend 目录下）：
 *     node scripts/check-markdown.mjs
 */

import {
  collectCitations,
  parseInline,
  parseMarkdown,
  safeHref,
} from '../src/markdown.ts'

const PASS = '  [通过]  '
const FAIL = '  [失败]  '
const failures = []

function check(cond, label, detail = '') {
  if (cond) {
    console.log(`${PASS}${label}`)
  } else {
    console.log(`${FAIL}${label}${detail ? `\n           ${detail}` : ''}`)
    failures.push(label)
  }
  return cond
}

/** 把行内节点压成便于断言的字符串，例如 `**粗**` -> "<strong>粗</strong>" */
function flat(nodes) {
  return nodes
    .map((n) => {
      switch (n.type) {
        case 'text':
          return n.text
        case 'br':
          return '<br>'
        case 'code':
          return `<code>${n.text}</code>`
        case 'strong':
          return `<strong>${flat(n.children)}</strong>`
        case 'em':
          return `<em>${flat(n.children)}</em>`
        case 'link':
          return `<a:${n.href}>${flat(n.children)}</a>`
        case 'citation':
          return `<cite:${n.index}>`
        default:
          return ''
      }
    })
    .join('')
}

console.log('='.repeat(72))
console.log('  Markdown 解析验证')
console.log('='.repeat(72))

// ---------------------------------------------------------------- 文件名保护
console.log('\n[1] 文件名里的下划线绝不能被当成强调符')
const FILENAMES = [
  '零电荷_EP15_3/-70-a-10.txt',
  '03_其他/周年庆剧情/123.txt',
  '双联乱数_EP13_25/-41-3-1first.txt',
]
for (const f of FILENAMES) {
  check(
    flat(parseInline(f)) === f,
    `原样保留：${f}`,
    flat(parseInline(f)),
  )
}
check(
  flat(parseInline('在《零电荷_EP15_3/-70-a-10.txt》中提到')) === '在《零电荷_EP15_3/-70-a-10.txt》中提到',
  '夹在句子中间的文件名也不被破坏',
)

// ---------------------------------------------------------------- 行内
console.log('\n[2] 行内语法')
check(flat(parseInline('**加粗**')) === '<strong>加粗</strong>', '加粗')
check(flat(parseInline('*斜体*')) === '<em>斜体</em>', '斜体')
check(flat(parseInline('`code`')) === '<code>code</code>', '行内代码')
check(flat(parseInline('普通文本')) === '普通文本', '纯文本原样')
check(flat(parseInline('甲**乙**丙')) === '甲<strong>乙</strong>丙', '句中加粗')
check(flat(parseInline('多行\n文本')) === '多行<br>文本', '单换行转 br（聊天场景保留断行）')

console.log('\n[3] 引用角标')
check(flat(parseInline('结论[1]')) === '结论<cite:1>', '基本引用')
check(flat(parseInline('依据[1][3]')) === '依据<cite:1><cite:3>', '连续多个引用')
check(flat(parseInline('**要点**[2]')) === '<strong>要点</strong><cite:2>', '引用紧跟在加粗后面')
check(flat(parseInline('**要点[2]**')) === '<strong>要点<cite:2></strong>', '引用嵌在加粗里面')
check(flat(parseInline('[12]')) === '<cite:12>', '两位数编号')
// [2024] 是年份不是引用，{1,2} 限制让它落回纯文本
check(flat(parseInline('[2024]')) === '[2024]', '三位数不当作引用（避免把年份吃掉）')

console.log('\n[4] 链接与安全')
check(flat(parseInline('[文档](https://example.com)')) === '<a:https://example.com>文档</a>', 'https 链接')
check(flat(parseInline('[文档](/api/kb)')) === '<a:/api/kb>文档</a>', '站内相对链接')
check(
  flat(parseInline('[点我](javascript:alert(1))')) === '[点我](javascript:alert(1))',
  'javascript: 协议退化成纯文本（不给可点链接）',
)
check(safeHref('javascript:alert(1)') === null, 'safeHref 拒绝 javascript:')
check(safeHref('data:text/html,x') === null, 'safeHref 拒绝 data:')
check(safeHref('https://a.com') === 'https://a.com', 'safeHref 放行 https')
check(safeHref('') === null, 'safeHref 拒绝空串')
// 链接必须早于引用判断，否则 [文字](url) 会被当成引用+普通文本
check(
  flat(parseInline('见[附录](https://a.com)')).includes('<a:https://a.com>'),
  '链接优先于引用判断（否则会被拆错）',
)

// ---------------------------------------------------------------- 块级
console.log('\n[5] 块级语法')
const blocks1 = parseMarkdown('# 标题一\n\n## 标题二\n\n正文段落。')
check(blocks1.length === 3, `标题+标题+段落 = 3 块（实际 ${blocks1.length}）`)
check(blocks1[0].type === 'heading' && blocks1[0].level === 1, '一级标题')
check(blocks1[1].type === 'heading' && blocks1[1].level === 2, '二级标题')
check(blocks1[2].type === 'paragraph', '段落')

const blocks2 = parseMarkdown('第一行\n第二行\n\n新段落')
check(blocks2.length === 2, '空行分段')
check(flat(blocks2[0].inlines) === '第一行<br>第二行', '段内换行保留')

const hr = parseMarkdown('上文\n\n---\n\n下文')
check(hr.some((b) => b.type === 'hr'), '--- 识别为分隔线而不是列表')
check(hr.filter((b) => b.type === 'list').length === 0, '--- 没有被误判成无序列表')

const quote = parseMarkdown('> 引用的内容\n> 第二行')
check(quote[0].type === 'quote', '引用块')
check(flat(quote[0].inlines) === '引用的内容<br>第二行', '引用块内多行')

const codeBlock = parseMarkdown('```python\nprint(1)\nprint(2)\n```')
check(codeBlock[0].type === 'code', '代码块')
check(codeBlock[0].text === 'print(1)\nprint(2)', '代码块内容原样保留')
check(codeBlock[0].lang === 'python', '代码块语言')

// ---------------------------------------------------------------- 列表
console.log('\n[6] 列表（模型最常用的结构）')
const ul = parseMarkdown('- 甲\n- 乙\n- 丙')
check(ul[0].type === 'list' && ul[0].ordered === false, '无序列表')
check(ul[0].items.length === 3, `三个列表项（实际 ${ul[0]?.items.length}）`)

const ol = parseMarkdown('1. 甲\n2. 乙\n3. 丙')
check(ol[0].type === 'list' && ol[0].ordered === true, '有序列表')
check(ol[0].start === 1, '有序列表起始编号')

// 这段取自真实回答
const REAL_LIST = `这个资料主要涉及几个不同的场景：

1. **《零电荷_EP15_3/-70-a-10.txt》**：描述了Q与阿玛瑞丝之间的对话。[1]
2. **《迷笼猜想/-59-b-1-start.txt》**：记录了格琳娜与指挥官的对话。[3]

综上所述，这些资料主要涉及任务分配。[1][2]`
const real = parseMarkdown(REAL_LIST)
const listBlock = real.find((b) => b.type === 'list')
check(Boolean(listBlock), '真实回答里的列表被识别出来')
check(listBlock.items.length === 2, `两个列表项（实际 ${listBlock?.items.length}）`)
check(
  flat(listBlock.items[0].inlines) ===
    '<strong>《零电荷_EP15_3/-70-a-10.txt》</strong>：描述了Q与阿玛瑞丝之间的对话。<cite:1>',
  '列表项里的加粗、文件名、引用都正确',
  flat(listBlock.items[0]?.inlines ?? []),
)
check(collectCitations(REAL_LIST).length === 4, '整段收集到 4 个引用编号')

console.log('\n[7] 嵌套列表')
const nested = parseMarkdown('- 外层一\n  - 内层一\n  - 内层二\n- 外层二')
const nList = nested.find((b) => b.type === 'list')
check(nList?.items.length === 2, `外层两项（实际 ${nList?.items.length}）`)
check(
  nList?.items[0].children.some((c) => c.type === 'list'),
  '第一项下面挂了嵌套列表',
)
check(
  nList?.items[0].children.find((c) => c.type === 'list')?.items.length === 2,
  '嵌套列表有两项',
)

console.log('\n[8] 表格')
const TABLE = `| 模型 | 维度 |
|---|---|
| bge-m3 | 1024 |
| qwen3 | 2560 |`
const tbl = parseMarkdown(TABLE)
check(tbl[0].type === 'table', '表格被识别')
check(tbl[0].header.join(',') === '模型,维度', '表头单元格')
check(tbl[0].rows.length === 2, `两行数据（实际 ${tbl[0].rows.length}）`)
check(tbl[0].rows[1].join(',') === 'qwen3,2560', '第二行内容')

// ---------------------------------------------------------------- 流式容错
console.log('\n[9] 流式半成品不能崩')
const PARTIALS = [
  '**未闭合的加粗',
  '1. 列表项一\n2. 列表项',
  '```\n未闭合的代码块',
  '| 表头 | 还没写完',
  '> 引用',
  '',
  '\n\n',
  '***',
  '- ',
  '# ',
  '[](http://a.com)',
  '结论[',
]
for (const p of PARTIALS) {
  let ok = true
  let err = ''
  try {
    const out = parseMarkdown(p)
    ok = Array.isArray(out)
  } catch (e) {
    ok = false
    err = e.message
  }
  check(ok, `不抛异常：${JSON.stringify(p.slice(0, 20))}`, err)
}

console.log('\n[10] 兜底：任何输入都能解析出确定的块')
// 关键性质：绝不能死循环或丢字符
const FUZZ = ['abc', '   ', '\t', '|', '|||', '---', '___', '***', '>>>>', '`', '****']
for (const f of FUZZ) {
  let ok = true
  try {
    parseMarkdown(f)
  } catch {
    ok = false
  }
  check(ok, `不抛异常：${JSON.stringify(f)}`)
}
// 普通文本要被完整保留，不能因为匹配失败而吞掉
const kept = parseMarkdown('一段没有任何语法的普通说明文字。')
check(
  flat(kept[0].inlines) === '一段没有任何语法的普通说明文字。',
  '普通文本完整保留',
)

console.log('\n[11] 真实回答整体跑一遍')
const LONG_REAL = `根据参考资料中的内容，资料里涉及的主要人物包括：

1. **P99** - 在《03_其他/周年庆剧情/123.txt》中提到，是与指挥官交流的角色。
2. **AK15** - 在《03_其他/回忆录/293_2.txt》中提到，是狼群小队中具备实战经验的人形。
3. **AK12** - 在《03_其他/回忆录/293_2.txt》《镜像论_EP13_5/-44-a2-1.txt》中多次出现。

**需要注意的是**：这些人物在不同段落中有所提及，但具体背景信息并未详细说明。[2][3]

- 资料中未提及的部分已明确标注。`
const longBlocks = parseMarkdown(LONG_REAL)
check(longBlocks.length >= 3, `至少 3 个块（实际 ${longBlocks.length}）`)
check(
  longBlocks.some((b) => b.type === 'list' && b.ordered),
  '含有序列表',
)
check(
  longBlocks.some((b) => b.type === 'paragraph' && flat(b.inlines).includes('<strong>需要注意的是</strong>')),
  '含加粗的段落',
)
check(
  longBlocks.some((b) => b.type === 'list' && !b.ordered),
  '含无序列表',
)
check(collectCitations(LONG_REAL).length === 2, '收集到 2 个引用')

console.log('\n' + '='.repeat(72))
if (failures.length) {
  console.log(`未通过 ${failures.length} 项：`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
console.log('  全部通过 ✅')
