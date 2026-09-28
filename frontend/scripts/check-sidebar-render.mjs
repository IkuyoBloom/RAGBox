/**
 * 侧边栏渲染检查（SSR）：验证「当前向量模型」「知识库模型徽标」「文档可折叠」
 * 这三样真的出现在 DOM 里，而不只是逻辑算对了。
 *
 * 为什么用合成数据而不是真实知识库：真实数据只能覆盖碰巧存在的那一种状态，
 * 而徽标有 5 种状态。用构造出来的数据才能一次把「一致 / 不一致 / 未安装 / 待索引」
 * 全部渲染出来，并断言它们各自的 CSS 类正确。
 *
 * 用法（在 frontend 目录下）：
 *     node scripts/check-sidebar-render.mjs
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

// ---------------------------------------------------------------- 合成数据
const kb = (id, name, embed_model, chunk_count, over = {}) => ({
  id,
  name,
  description: '',
  embed_model,
  embed_dim: chunk_count > 0 ? 1024 : null,
  chunk_size: 500,
  chunk_overlap: 80,
  collection_name: `c_${id}`,
  created_at: '2026-09-25T00:00:00Z',
  updated_at: '2026-09-25T00:00:00Z',
  document_count: 1,
  ready_document_count: chunk_count > 0 ? 1 : 0,
  chunk_count,
  ...over,
})

// 注意：当前设置是 qwen3-embedding:4b，「一致库」必须用这个模型才会判为 ok。
// 第一版把 k1 写成 bge-m3，结果四个库全是 differs/missing/pending，
// 压根没有 ok 用例 —— 测试数据本身写错了。
const KBS = [
  kb('k1', '一致库', 'qwen3-embedding:4b', 10), // ok（与当前设置相同）
  kb('k2', '换过模型的库', 'bge-m3', 20), // differs
  kb('k3', '模型被删的库', 'ghost-model', 8), // missing
  kb('k4', '空库', 'qwen3-embedding:4b', 0), // pending
]

const EMBED_INFO = {
  current: 'qwen3-embedding:4b',
  installed: ['bge-m3:latest', 'qwen3-embedding:4b'],
  loaded: true,
}

const DETAIL = {
  ...KBS[2],
  documents: [
    {
      id: 'd1',
      kb_id: 'k3',
      filename: '论文.pdf',
      file_type: 'pdf',
      size_bytes: 204800,
      status: 'ready',
      chunk_count: 8,
      char_count: 3000,
      page_count: 12,
      error: '',
      meta: {},
      created_at: '2026-09-25T00:00:00Z',
      updated_at: '2026-09-25T00:00:00Z',
    },
  ],
}

// ---------------------------------------------------------------- 渲染
const vite = await createServer({
  root: FRONTEND,
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'error',
})

let html = ''
try {
  const { default: Sidebar } = await vite.ssrLoadModule('/src/components/Sidebar.tsx')
  html = renderToStaticMarkup(
    createElement(Sidebar, {
      kbs: KBS,
      selectedKbId: 'k3',
      kbDetail: DETAIL,
      embedInfo: EMBED_INFO,
      onSelect: () => {},
      onRefreshKbs: () => {},
      onRefreshDetail: () => {},
      notify: () => {},
    }),
  )
} finally {
  await vite.close()
}

console.log('='.repeat(72))
console.log('  侧边栏渲染检查（SSR）')
console.log('='.repeat(72))
console.log(`  HTML 长度：${html.length} 字符`)

// --out <文件> 导出可独立打开的预览。合成数据正好把四种徽标状态一次展示出来，
// 比跑起整个应用去看真实数据更直观。
const outIdx = process.argv.indexOf('--out')
if (outIdx >= 0) {
  const { readFileSync, writeFileSync, mkdirSync } = await import('node:fs')
  const out = process.argv[outIdx + 1]
  const css = readFileSync(join(FRONTEND, 'src', 'styles.css'), 'utf8')
  mkdirSync(dirname(out), { recursive: true })
  writeFileSync(
    out,
    `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>侧边栏布局预览</title>
<style>
${css}
/* 预览专用：把侧边栏撑开，方便整体查看 */
.app-body { display: block; }
.sidebar { width: 340px; height: auto; max-height: none; }
</style>
</head>
<body>
<div class="app"><div class="app-body">${html}</div></div>
</body>
</html>`,
    'utf8',
  )
  console.log(`  预览已导出：${out}`)
}
console.log('')

const count = (re) => (html.match(re) ?? []).length
const has = (s) => html.includes(s)

// ---------------------------------------------------------------- 当前模型
console.log('[1] 实时显示当前向量模型')
check(has('current-embed'), '有「当前向量模型」区域')
check(has('当前向量模型'), '标签文案存在')
check(
  /current-embed-value[^>]*>qwen3-embedding:4b</.test(html),
  '显示的是设置里的当前模型（qwen3-embedding:4b）',
)
check(has('新建知识库将使用这个向量模型'), '带了说明性 title（区分「库的模型」与「当前设置」）')

// ---------------------------------------------------------------- 徽标
console.log('\n[2] 知识库模型徽标（四种状态一次覆盖）')
// 正则必须带上尾部空格，否则 class="model-badge-note" 也会被算进来
// （它就长这样：class="model-badge-note"），数量会翻倍。
check(
  count(/class="model-badge /g) === 4,
  `4 个知识库各有一个徽标（实际 ${count(/class="model-badge /g)}）`,
)
check(has('model-badge model-ok'), '「一致」渲染为 model-ok')
check(has('model-badge model-differs'), '「与设置不同」渲染为 model-differs')
check(has('model-badge model-missing'), '「模型未安装」渲染为 model-missing')
check(has('model-badge model-pending'), '「待索引」渲染为 model-pending')
check(has('≠当前设置'), '不一致的徽标带文字说明而不是只有颜色')
check(has('未安装'), '未安装的徽标带文字说明')
check(has('待索引'), '待索引的徽标带文字说明')
check(has('ghost-model'), '库的模型名显示出来了')
check(
  /title="⚠ 该库使用的向量模型[^"]*ollama pull ghost-model/.test(html),
  '未安装的徽标给出了可执行的修复提示',
)
check(
  /title="该库使用 bge-m3，与当前设置里的「qwen3-embedding:4b」不同[\s\S]{0,120}新知识库将使用/.test(html),
  '不一致的提示说明了「这是正常的」并指出新库会用哪个模型',
)

// ---------------------------------------------------------------- 文档折叠
console.log('\n[3] 文档面板可折叠')
check(has('panel-head clickable'), '文档面板标题可点击（折叠开关）')
check(has('chev'), '带折叠箭头')
check(has('收起文档列表'), '默认展开时提示「收起」')
check(count(/class="doc-item"/g) === 1, '展开状态下文档列表已渲染')
check(has('向量 <code>ghost-model</code>'), '面板里也显示了该库的向量模型')
check(has('kb-config'), '切片/重叠/维度信息仍在')

// ---------------------------------------------------------------- 面板身份
console.log('\n[4] 两块面板的视觉身份')
// 用户反馈「左边栏区分视觉不明显」：原先两块都是同样的白卡片加一行
// 同样大小的灰色小标题，扫一眼分不出哪块管什么。
// 现在靠三个互相印证的线索区分：左侧色条、图标、计数徽标。
check(has('panel panel-kb'), '知识库面板带自己的身份类 panel-kb')
check(has('panel grow panel-docs'), '文档面板带自己的身份类 panel-docs')
check(
  count(/class="panel-icon"/g) === 2,
  `两块面板各有一个图标（实际 ${count(/class="panel-icon"/g)}）`,
)
check(has('<svg'), '图标是内联 SVG（跟着主题变色，不额外请求）')
check(
  count(/class="panel-count"/g) === 2,
  `两块面板各有一个计数徽标（实际 ${count(/class="panel-count"/g)}）`,
)
// 计数从标题里拆出来了：原先写死成「文档（5）」，标题字号下数字容易被忽略
check(has('>知识库</h2>'), '知识库标题不带括号计数')
check(has('>文档</h2>'), '文档标题不带括号计数')
check(!has('文档（'), '计数已改为独立徽标而不是塞在标题括号里')
check(count(/aria-hidden="true"/g) >= 2, '图标标记为 aria-hidden（装饰性，读屏跳过）')

// ---------------------------------------------------------------- 其它
console.log('\n[5] 不应出现的东西')
check(!has('create-form'), '未点「新建」时不渲染创建表单')
check(!has('<datalist'), '不使用原生 datalist')

console.log('\n' + '='.repeat(72))
if (failures.length) {
  console.log(`未通过 ${failures.length} 项：`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
console.log('  全部通过 ✅')
