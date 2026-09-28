/**
 * 对话区渲染检查（SSR）。
 *
 * 这一轮 UI 优化改了 ChatPanel 的三处，都是「看着不对但不会报错」的类型，
 * 所以用断言钉住：
 *   1. 空状态要有品牌标记与示例问题卡片（原先只有两行字，四周全是空白）；
 *   2. 示例问题必须**与语料无关** —— 上一版是照着论文写的
 *      （「核心贡献」「实验数据集」），换成剧情文本就一条都问不通；
 *   3. 输入区要显示快捷键提示（原先只能靠猜）。
 *
 * 用法（在 frontend 目录下）：
 *     node scripts/check-chat-render.mjs
 */

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

const vite = await createServer({
  root: FRONTEND,
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'error',
})

const noop = () => {}

/**
 * 造一条完整的 Citation。
 * CitationList 会读 c.score.toFixed(4) —— 少给字段不是「测试数据不完整」，
 * 而是直接把整个对话区渲染崩掉（React 抛错 → 白屏），所以这里必须给全。
 */
const cite = (index, filename, preview) => ({
  index,
  chunk_id: `c${index}`,
  doc_id: `d${index}`,
  filename,
  page: null,
  ordinal: 0,
  score: 0.9 - index * 0.1,
  dense_score: 0.8,
  sparse_score: null,
  dense_rank: index,
  sparse_rank: null,
  source: 'dense',
  preview,
})

function renderChat(props) {
  return renderToStaticMarkup(
    createElement(ChatPanel, {
      kbName: '演示知识库',
      messages: [],
      streaming: false,
      onSend: noop,
      onStop: noop,
      onOpenChunk: noop,
      ...props,
    }),
  )
}

let ChatPanel = null
let empty = ''
let withMsgs = ''
let noKb = ''
try {
  ChatPanel = (await vite.ssrLoadModule('/src/components/ChatPanel.tsx')).default

  empty = renderChat({})

  noKb = renderChat({ kbName: null })

  withMsgs = renderChat({
    messages: [
      { id: 'm1', role: 'user', content: '这些资料主要讲了什么？', citations: [] },
      {
        id: 'm2',
        role: 'assistant',
        content: '资料主要讲了 **三件事**：\n\n1. 甲[1]\n2. 乙[2]',
        citations: [cite(1, 'a.txt', '甲'), cite(2, 'b.txt', '乙')],
      },
      // 第二条助手消息是刻意加的：头像用 useId 生成渐变 id，
      // 只有一个头像时测不出「多个实例的 id 是否唯一」
      { id: 'm3', role: 'user', content: '那第二点呢？', citations: [] },
      {
        id: 'm4',
        role: 'assistant',
        content: '第二点讲的是**乙**。[2]',
        citations: [cite(2, 'b.txt', '乙')],
      },
    ],
  })
} finally {
  await vite.close()
}

console.log('='.repeat(72))
console.log('  对话区渲染检查（SSR）')
console.log('='.repeat(72))
console.log()

const count = (h, re) => (h.match(re) ?? []).length
const has = (h, s) => h.includes(s)

// ---------------------------------------------------------------- 空状态
console.log('[1] 空状态（选中知识库）')
check(has(empty, 'chat-empty'), '有空状态容器')
check(has(empty, 'class="empty-mark"'), '有品牌标记（原先只有两行字，四周空荡）')
check(has(empty, '开始向「演示知识库」提问'), '标题带知识库名')
check(has(empty, 'suggestions-wrap'), '有示例问题区块')
check(has(empty, 'suggestions-label'), '有「试试这样问」引导语')
check(
  count(empty, /class="suggestion"/g) === 3,
  `3 张示例问题卡片（实际 ${count(empty, /class="suggestion"/g)}）`,
)
check(
  count(empty, /class="suggestion-arrow"/g) === 3,
  '每张卡片都有箭头（暗示可点击）',
)

console.log('\n[2] 示例问题必须与语料无关')
// 回归：上一版照着论文写，换成剧情文本 / 说明文档就一条都问不通
const FORBIDDEN = ['核心贡献', '实验', '数据集', '局限性', '论文']
for (const word of FORBIDDEN) {
  check(!has(empty, word), `不含论文专用词「${word}」`)
}
check(has(empty, '这些资料主要讲了什么？'), '含通用问题（资料主题）')
check(
  count(empty, /class="suggestion"/g) === 3 && has(empty, '关键信息'),
  '含通用问题（关键信息梳理）',
)

// ---------------------------------------------------------------- 无知识库
console.log('\n[3] 没有知识库时的引导')
check(has(noKb, 'empty-hint'), '给出「先建库上传」的引导而不是干巴巴一句话')
check(!has(noKb, 'suggestions-wrap'), '没库时不显示示例问题（点了也是报错）')
check(has(noKb, '欢迎使用本地 RAG'), '标题回退到欢迎语')

// ---------------------------------------------------------------- 输入区
console.log('\n[4] 输入区')
check(has(empty, 'composer-row'), '输入区内容列有独立容器')
check(has(empty, 'composer-hint'), '有快捷键提示行')
check(has(empty, '<kbd>Enter</kbd>'), '提示里画出了 Enter 键帽')
check(has(empty, 'Shift'), '提示里写了 Shift+Enter 换行')
check(has(empty, 'composer-hint-kb'), '右下角标出当前知识库')
check(has(noKb, '先在左侧选择或创建知识库'), '无库时提示去选择')
// 内容列与消息列同宽，否则上下两块对不齐
check(has(empty, 'composer-row') && has(empty, 'composer-hint'), '输入框与提示同属一个内容列')
check(has(empty, 'disabled'), '无草稿时发送按钮禁用')

// ---------------------------------------------------------------- 消息
console.log('\n[5] 已有消息时')
check(!has(withMsgs, 'chat-empty'), '有消息时不再渲染空状态')
check(has(withMsgs, 'message-list'), '渲染消息列表')
check(
  count(withMsgs, /class="message /g) === 4,
  `四条消息（实际 ${count(withMsgs, /class="message /g)}）`,
)
check(has(withMsgs, 'message-user'), '用户消息有独立样式')
check(has(withMsgs, 'message-assistant'), '助手消息有独立样式')
check(has(withMsgs, '<strong>三件事</strong>'), '助手消息走 Markdown 渲染')
check(has(withMsgs, 'class="cite-ref"'), '引用角标渲染成可点元素')

console.log('\n[6] 头像')
// 用户头像仍是文字「我」；AI 头像是内联 SVG（四角星）
check(has(withMsgs, 'class="avatar">我<'), '用户头像仍是文字「我」')
check(has(withMsgs, 'class="avatar-ai"'), 'AI 头像有独立样式类')
check(has(withMsgs, 'aria-label="AI 助手"'), 'AI 头像带无障碍标签')
check(has(withMsgs, 'viewBox="0 0 32 32"'), 'AI 头像是内联 SVG（不依赖外部图片）')
check(!has(withMsgs, 'class="avatar">AI<'), '不再用「AI」两个字的文字头像')
check(count(withMsgs, /class="avatar"/g) === 2, '文字头像只出现在两条用户消息里')
check(
  count(withMsgs, /class="avatar-ai"/g) === 2,
  `两条助手消息各有一个头像（实际 ${count(withMsgs, /class="avatar-ai"/g)}）`,
)

// 只从 id="..." 里取：同一个 id 还会出现在 fill="url(#...)" 里，
// 用宽松的正则会把每个 id 数两遍，误报「重复」。
const avatarIds = [...withMsgs.matchAll(/id="(rag-avatar-[A-Za-z0-9]+)"/g)].map((m) => m[1])
check(
  avatarIds.length === 2,
  `两个 AI 头像各生成一个渐变 id（实际 ${avatarIds.length}：${avatarIds.join(', ')}）`,
)
check(new Set(avatarIds).size === avatarIds.length, '两个头像的渐变 id 互不相同')
const avatarUrls = [...withMsgs.matchAll(/fill="url\(#(rag-avatar-[A-Za-z0-9]+)\)"/g)].map((m) => m[1])
check(avatarUrls.length === 2, '两处渐变填充都存在')
check(
  avatarUrls.every((u, i) => u === avatarIds[i]),
  '每个头像的填充指向自己的 id',
  `${avatarUrls.join(',')} vs ${avatarIds.join(',')}`,
)

const outIdx = process.argv.indexOf('--out')
if (outIdx >= 0) {
  const { mkdirSync, readFileSync, writeFileSync } = await import('node:fs')
  const out = process.argv[outIdx + 1]
  // --theme dark 让预览带上 data-theme，用于核对深色下的消息列表
  const themeIdx = process.argv.indexOf('--theme')
  const theme = themeIdx >= 0 && process.argv[themeIdx + 1] === 'dark' ? 'dark' : 'light'
  const css = readFileSync(join(FRONTEND, 'src', 'styles.css'), 'utf8')
  mkdirSync(dirname(out), { recursive: true })
  writeFileSync(
    out,
    `<!doctype html>
<html lang="zh-CN" data-theme="${theme}">
<head>
<meta charset="utf-8">
<title>对话区预览（${theme}）</title>
<style>
${css}
/* 预览专用：把对话区撑成独立卡片 */
body { margin: 0; padding: 24px; background: var(--bg);
       font-family: 'Segoe UI', 'Microsoft YaHei UI', system-ui, sans-serif; }
.preview-card { max-width: 1000px; margin: 0 auto; background: var(--bg);
                border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
                display: flex; flex-direction: column; height: calc(100vh - 48px); }
</style>
</head>
<body>
  <div class="preview-card">${withMsgs}</div>
</body>
</html>`,
    'utf8',
  )
  console.log(`\n  预览已导出：${out}（${theme}）`)
}

console.log('\n' + '='.repeat(72))
if (failures.length) {
  console.log(`未通过 ${failures.length} 项：`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
console.log('  全部通过 ✅')
