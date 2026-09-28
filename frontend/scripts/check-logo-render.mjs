/**
 * Logo 渲染检查 + 预览导出。
 *
 * Logo 是纯内联 SVG，很容易犯两类「肉眼不一定看得出」的错：
 *   1. **渐变 id 撞车** —— 同一页面会渲染多个 Logo（顶栏一个、空状态一个），
 *      写死 id 的话浏览器只认第一个，第二个可能拿不到填充而变成透明；
 *   2. **小尺寸糊成一团** —— 图形要素太多，缩到 24px 就分不清了。
 *
 * 第 2 点没法断言，靠导出的预览图用肉眼判断；第 1 点可以断言。
 *
 * 用法（在 frontend 目录下）：
 *     node scripts/check-logo-render.mjs
 *     node scripts/check-logo-render.mjs --out preview/logo.html
 */

import { mkdirSync, writeFileSync } from 'node:fs'
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

let Logo = null
let Avatar = null
let one = ''
let two = ''
let big = ''
let avatars = ''
try {
  const viteMod = await vite.ssrLoadModule('/src/components/Logo.tsx')
  Logo = viteMod.default
  Avatar = (await vite.ssrLoadModule('/src/components/AssistantAvatar.tsx')).default
  one = renderToStaticMarkup(createElement(Logo, {}))
  two = renderToStaticMarkup(createElement(Logo, { size: 58, className: 'empty-mark' }))
  big = renderToStaticMarkup(createElement(Logo, { size: 96 }))
  // 两个头像放在同一棵树里，才能验证渐变 id 是否唯一
  avatars = renderToStaticMarkup(
    createElement(
      'div',
      null,
      createElement(Avatar, { className: 'avatar-ai' }),
      createElement(Avatar, { size: 30, className: 'avatar-ai' }),
    ),
  )
} finally {
  await vite.close()
}

console.log('='.repeat(72))
console.log('  品牌图形检查（Logo + AI 头像）')
console.log('='.repeat(72))
console.log()

const has = (s) => one.includes(s)
const count = (s, re) => (s.match(re) ?? []).length
/**
 * 数图形要素个数。
 *
 * 正则末尾必须加 `[\s/>]`：否则 `<(line)` 会连 `<linearGradient` 一起匹配，
 * 每个渐变定义都被多算成一个图形 —— 这正是本项目反复踩到的「子串匹配」
 * 陷阱（见 AGENTS.md 的自查纪律）。
 */
const SHAPE_RE = /<(rect|circle|line|path|polygon|ellipse)[\s/>]/g

console.log('[1] 结构')
check(has('<svg'), '是 SVG')
check(has('viewBox="0 0 32 32"'), '用了 viewBox（尺寸可变而图形比例不变）')
check(has('role="img"'), '带 role="img"（屏幕阅读器按图片处理）')
check(has('aria-label='), '带 aria-label（图形没有可读文本，必须给替代文字）')
check(has('<linearGradient'), '有渐变定义')
check(has('<rect'), '有底板')
check(has('<circle'), '有镜片圆环')
check(count(one, /<rect/g) === 3, `矩形 3 个：底板 + 2 条知识块（实际 ${count(one, /<rect/g)}）`)
check(has('<line'), '有镜片手柄')
check(has('stroke-linecap="round"'), '手柄端点是圆的（否则小尺寸下看着像毛刺）')

console.log('\n[2] 尺寸与样式')
check(has('width="34"'), '默认 34px（未传 size 时的兜底）')
check(two.includes('width="58"'), 'size 属性生效')
check(two.includes('class="empty-mark"'), 'className 透传（空状态靠它挂阴影）')

console.log('\n[3] 渐变 id 必须唯一')
// 关键：两个 Logo 要在**同一棵 React 树**里渲染。
// 分成两次 renderToStaticMarkup 各自是独立的树，useId 都会从 R0 开始 ——
// 那样测出来必然「重复」，但真实页面并不存在这个问题（它们同处一棵树）。
const both = renderToStaticMarkup(
  createElement(
    'div',
    null,
    createElement(Logo, {}),
    createElement(Logo, { size: 58, className: 'empty-mark' }),
  ),
)
const ids = [...both.matchAll(/id="([^"]+)"/g)].map((m) => m[1])
const urls = [...both.matchAll(/fill="url\(#([^)]+)\)"/g)].map((m) => m[1])
check(ids.length === 2, `同一棵树里两个实例各有一个渐变 id（实际 ${ids.length}）`)
check(ids[0] !== ids[1], `两个 id 不同（${ids[0]} vs ${ids[1]}）`)
check(urls.length === 2, '两处填充都引用了渐变')
check(urls[0] === ids[0] && urls[1] === ids[1], '每个实例的填充指向自己的 id')
// useId 原值形如 ":r0:"，冒号放进 url(#...) 不稳妥
check(
  ids.every((i) => !i.includes(':')),
  'id 里不含冒号',
  ids.join(', '),
)

console.log('\n[4] 小尺寸可用性（结构层面）')
// 要素数量直接决定小尺寸下会不会糊。超过 8 个就该考虑简化。
// Logo 实为 5 个：底板 + 2 条知识块 + 镜片圆环 + 手柄。
const shapes = count(one, SHAPE_RE)
check(shapes === 5, `Logo 图形要素 5 个（实际 ${shapes}）`)
check(shapes <= 8, `要素数量不超过 8（缩到 24px 仍可分辨）`)

console.log('\n[5] favicon 与组件同源')
// favicon 必须是独立文件（浏览器要真实 URL，不能给 React 组件），
// 所以存在「改了组件忘了改 favicon」的风险，这里校一遍。
const { readFileSync, existsSync } = await import('node:fs')
const faviconPath = join(FRONTEND, 'public', 'logo.svg')
check(existsSync(faviconPath), 'public/logo.svg 存在')
if (existsSync(faviconPath)) {
  const fav = readFileSync(faviconPath, 'utf8')
  // 逐个比对关键几何参数，避免只比「有没有这个文件」而漏掉图形漂移
  for (const token of [
    'viewBox="0 0 32 32"',
    'rx="9"',
    'cx="14"',
    'cy="13.6"',
    'r="6.2"',
    'stroke-width="2.4"',
    'x="10.5"',
    'y="10.7"',
    'y="14.5"',
    'x1="18.5"',
    'y2="23"',
  ]) {
    check(fav.includes(token), `favicon 与组件同源：${token}`)
  }
  check(fav.includes('stop-color="#5b93f8"'), 'favicon 渐变起点色一致')
  check(fav.includes('stop-color="#1f56c4"'), 'favicon 渐变终点色一致')
  check(fav.includes('gradientUnits="userSpaceOnUse"'), 'favicon 也用 userSpaceOnUse')
}

const indexHtml = readFileSync(join(FRONTEND, 'index.html'), 'utf8')
check(indexHtml.includes('rel="icon"'), 'index.html 引用了 favicon')
check(indexHtml.includes('/logo.svg'), 'favicon 指向 /logo.svg')
check(indexHtml.includes('theme-color'), '声明了 theme-color（移动端地址栏配色）')

console.log('\n[6] AI 头像（AssistantAvatar）')
// 头像是内联 SVG（四角星），不引用外部图片资源 ——
// 任何尺寸都清晰、不增加请求，也不会因为图片缺失显示成裂图。
check(avatars.includes('<svg'), '是 SVG')
check(avatars.includes('viewBox="0 0 32 32"'), '用了 viewBox')
check(avatars.includes('role="img"'), '带 role="img"')
check(avatars.includes('aria-label="AI 助手"'), '带无障碍标签')
check(avatars.includes('<path'), '图形是路径（四角星）')
// 每个头像 2 个要素：底板 + 四角星路径。两个头像共 4 个。
const avatarShapes = count(avatars, SHAPE_RE)
check(avatarShapes === 4, `两个头像各「底板 + 星」共 4 个要素（实际 ${avatarShapes}）`)
// 与 Logo 同源的视觉语言：同一套渐变、同样用 userSpaceOnUse
check(avatars.includes('gradientUnits="userSpaceOnUse"'), '渐变用 userSpaceOnUse（与 Logo 一致）')
check(avatars.includes('stop-color="#5b93f8"'), '渐变起点色与 Logo 一致')
check(avatars.includes('stop-color="#1f56c4"'), '渐变终点色与 Logo 一致')
check(avatars.includes('rx="9"'), '底板圆角与 Logo 一致')

// 一条长对话里有几十个头像，id 撞车会让除第一个以外全部拿不到填充
const avIds = [...avatars.matchAll(/id="([^"]+)"/g)].map((m) => m[1])
check(avIds.length === 2, `两个头像各生成一个渐变 id（实际 ${avIds.length}）`)
check(avIds[0] !== avIds[1], `两个 id 不同（${avIds.join(' vs ')}）`)
check(
  avIds.every((i) => i.startsWith('rag-avatar-')),
  'id 带 rag-avatar- 前缀，与 Logo 的 rag-logo- 区分开',
)
check(one !== '' && !one.includes('rag-avatar-'), 'Logo 与头像的 id 命名空间不混用')

const outIdx = process.argv.indexOf('--out')
if (outIdx >= 0) {
  const out = process.argv[outIdx + 1]
  mkdirSync(dirname(out), { recursive: true })
  const sizes = [16, 20, 24, 32, 48, 64, 96]
  writeFileSync(
    out,
    `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>项目 Logo 预览</title>
<style>
  body { margin: 0; padding: 32px 36px; background: #f5f6f8;
         font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; color: #1c2430; }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 4px; }
  p.note { font-size: 12px; color: #5b6675; margin: 0 0 26px; }
  h2 { font-size: 12px; font-weight: 600; color: #5b6675; margin: 30px 0 12px;
       letter-spacing: .5px; text-transform: uppercase; }
  .row { display: flex; align-items: flex-end; gap: 26px; flex-wrap: wrap; }
  .item { text-align: center; }
  .item span { display: block; margin-top: 8px; font-size: 10.5px; color: #8b95a3; }
  .dark { background: #16202e; padding: 22px 26px; border-radius: 12px; }
  .light { background: #fff; padding: 22px 26px; border-radius: 12px; border: 1px solid #e3e6ec; }
  .brandrow { display: flex; align-items: center; gap: 10px; font-size: 15px; font-weight: 600; }
</style>
</head>
<body>
  <h1>项目 Logo</h1>
  <p class="note">放大镜 + 知识块：一眼看出「在知识库里检索」。要素少，小尺寸也不糊。</p>

  <h2>各尺寸</h2>
  <div class="row">
    ${sizes
      .map(
        (s) =>
          `<div class="item">${renderToStaticMarkup(createElement(Logo, { size: s }))}<span>${s}px</span></div>`,
      )
      .join('\n    ')}
  </div>

  <h2>顶栏实际用法</h2>
  <div class="light">
    <div class="brandrow">${renderToStaticMarkup(createElement(Logo, { size: 34 }))}本地知识库问答系统</div>
  </div>

  <h2>深色背景</h2>
  <div class="dark">
    <div class="row">
      ${[32, 48, 64]
        .map(
          (s) =>
            `<div class="item">${renderToStaticMarkup(createElement(Logo, { size: s }))}<span style="color:#8b95a3">${s}px</span></div>`,
        )
        .join('\n      ')}
    </div>
  </div>

  <h2>大尺寸细节</h2>
  <div class="light">${big}</div>

  <h2>AI 头像 · 各尺寸</h2>
  <div class="row">
    ${[16, 20, 24, 30, 40, 56, 96]
      .map(
        (s) =>
          `<div class="item">${renderToStaticMarkup(createElement(Avatar, { size: s }))}<span>${s}px</span></div>`,
      )
      .join('\n    ')}
  </div>

  <h2>两者并列（是否像一套）</h2>
  <div class="light">
    <div class="row" style="align-items:center; gap:48px">
      <div class="item">${renderToStaticMarkup(createElement(Logo, { size: 56 }))}<span>Logo · 检索</span></div>
      <div class="item">${renderToStaticMarkup(createElement(Avatar, { size: 56 }))}<span>头像 · 生成</span></div>
      <div class="item" style="display:flex;align-items:center;gap:10px">
        ${renderToStaticMarkup(createElement(Avatar, { size: 30 }))}
        <span style="margin:0">对话里的实际大小 30px</span>
      </div>
    </div>
  </div>
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
