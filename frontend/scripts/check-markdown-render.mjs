/**
 * Markdown 渲染检查 + 预览导出。
 *
 * 解析器对了不代表渲染对了 —— 这里真的用 SSR 把组件渲染成 HTML，
 * 断言元素结构（<ol>/<li>/<strong>/引用按钮），并可选导出成能独立打开的预览文件。
 *
 * 为什么要在 HTML 里断言：解析结果正确但渲染时 key、嵌套、class 写错，
 * 页面看起来照样是坏的，而这只有渲染层能发现。
 *
 * 用法（在 frontend 目录下）：
 *     node scripts/check-markdown-render.mjs
 *     node scripts/check-markdown-render.mjs --out preview/markdown.html
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const HERE = dirname(fileURLToPath(import.meta.url))
const FRONTEND = join(HERE, '..')

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

/** 一段真实的回答（从实际问答里摘下来的，含列表/加粗/引用/文件名）。 */
const ANSWER = `这个资料主要涉及几个不同的场景和对话片段，内容较为分散。以下是概括：

1. **《零电荷_EP15_3/-70-a-10.txt》**：描述了Q与阿玛瑞丝之间的对话，提到了“NGAD”计划和《人类理智研究（第十六版）》的相关信息。[1]
2. **《迷笼猜想/-59-b-1-start.txt》**：记录了格琳娜与指挥官的对话，讨论了贝莱格莱德会议的安全问题。[3]
3. **《异构体_EP11_5/-31-1a11-1.txt》**：格琳娜提到安洁被借调到格里芬作为指挥官的战斗记录。[3]

**需要注意的是**：资料中提到的部分内容存在疑点。

- 有些资料被人为破坏，无法读取。[2]
- 终端 \`-44-t2-1-ex\` 已经报废。[6]

---

> 以上仅为资料中明确写到的内容。

| 来源 | 状态 |
|---|---|
| 安洁行动档案 | 已损坏 |
| 药企旧址档案 | 未提取 |

综上所述，这些资料主要涉及任务分配与情报收集。`

const CITATIONS = [1, 2, 3, 6].map((index) => ({
  index,
  chunk_id: `c${index}`,
  doc_id: `d${index}`,
  filename: `示例文档${index}.txt`,
  page: null,
  ordinal: 0,
  preview: `第 ${index} 条检索片段的预览内容`,
}))

const vite = await createServer({
  root: FRONTEND,
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'error',
})

let html = ''
let MarkdownComponent = null
try {
  const mod = await vite.ssrLoadModule('/src/components/Markdown.tsx')
  MarkdownComponent = mod.default
  const byIndex = new Map(CITATIONS.map((c) => [c.index, c]))
  html = renderToStaticMarkup(
    createElement(MarkdownComponent, {
      content: ANSWER,
      renderCitation: (index, key) => {
        const c = byIndex.get(index)
        if (!c) {
          return createElement('span', { key, className: 'cite-ref invalid' }, `[${index}]`)
        }
        return createElement(
          'button',
          { key, className: 'cite-ref', title: c.filename },
          `[${index}]`,
        )
      },
    }),
  )
} finally {
  await vite.close()
}

console.log('='.repeat(72))
console.log('  Markdown 渲染检查（SSR）')
console.log('='.repeat(72))
console.log(`  HTML 长度：${html.length} 字符\n`)

const count = (re) => (html.match(re) ?? []).length
const has = (s) => html.includes(s)

console.log('[1] 结构')
check(has('class="md-body"'), '外层容器')
check(count(/<ol class="md-list">/g) === 1, `有序列表 1 个（实际 ${count(/<ol class="md-list">/g)}）`)
check(count(/<ul class="md-list">/g) === 1, `无序列表 1 个（实际 ${count(/<ul class="md-list">/g)}）`)
check(count(/<li>/g) === 5, `列表项 5 个（实际 ${count(/<li>/g)}）`)
check(count(/<p class="md-p">/g) >= 2, '段落')
check(has('<blockquote class="md-quote">'), '引用块')
check(has('class="md-hr"'), '分隔线')
check(has('<table class="md-table">'), '表格')
check(has('<th>来源</th>'), '表头单元格')
check(has('<td>已损坏</td>'), '表格数据单元格')

console.log('\n[2] 行内')
check(count(/<strong>/g) >= 3, `加粗 ${count(/<strong>/g)} 处`)
check(has('<code class="md-inline-code">-44-t2-1-ex</code>'), '行内代码')

console.log('\n[3] 引用角标可点')
// 样例里实际有 5 处引用：[1] 一次、[3] 两次、[2] 一次、[6] 一次
check(
  count(/class="cite-ref"/g) === 5,
  `引用角标 5 个（实际 ${count(/class="cite-ref"/g)}）`,
)
check(count(/<button class="cite-ref"/g) >= 4, '有对应的编号渲染成按钮')
check(has('>[1]<'), '角标文字是 [1] 形式')
check(has('title="示例文档1.txt"'), '角标带悬停提示')

console.log('\n[4] 安全')
// 模型输出的尖括号必须被转义，不能被当成标签
const evil = renderToStaticMarkup(
  createElement(MarkdownComponent, {
    content: '<script>alert(1)</script>\n\n<img src=x onerror=alert(1)>',
    renderCitation: () => null,
  }),
)
check(!evil.includes('<script>'), '回答里的 <script> 被转义')
check(evil.includes('&lt;script&gt;'), '尖括号以实体形式出现')
check(!evil.includes('<img'), '回答里的 <img> 不被当成标签')

console.log('\n[5] 解析器一致性')
const { parseMarkdown } = await import('../src/markdown.ts')
const blocks = parseMarkdown(ANSWER)
check(blocks.length > 0, `真实回答解析出 ${blocks.length} 个块`)

const outIdx = process.argv.indexOf('--out')
if (outIdx >= 0) {
  const out = process.argv[outIdx + 1]
  const css = readFileSync(join(FRONTEND, 'src', 'styles.css'), 'utf8')
  mkdirSync(dirname(out), { recursive: true })
  writeFileSync(
    out,
    `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Markdown 渲染预览</title>
<style>
${css}
/* 预览专用：模拟一个回答气泡 */
body { margin: 0; padding: 28px; background: #f2f4f7; font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
.preview-card { max-width: 760px; margin: 0 auto; background: #fff; border: 1px solid #e3e7ec; border-radius: 10px; padding: 18px 20px; }
.preview-note { max-width: 760px; margin: 0 auto 12px; font-size: 12px; color: #6b7683; }
</style>
</head>
<body>
<div class="preview-note">下面是模型真实回答经 Markdown 渲染后的效果（含可点击的 [n] 引用角标）：</div>
<div class="preview-card">${html}</div>
</body>
</html>`,
    'utf8',
  )
  console.log(`\n  预览已导出：${out}`)
}

console.log('\n' + '='.repeat(72))
if (failures.length) {
  console.log(`未通过 ${failures.length} 项：`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
console.log('  全部通过 ✅')
