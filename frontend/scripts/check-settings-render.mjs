/**
 * 设置面板「真实渲染」检查（SSR）。
 *
 * 上一层的 check-settings-tree.mjs 只验证了纯逻辑（树构建、依赖判定）。
 * 「能编译 + 逻辑对」仍然不等于「DOM 里真的有这些层级」—— 忘记渲染某个小节、
 * 主控卡片没接上、标签条件写反，这些问题编译期都是发现不了的。
 *
 * 这里用 react-dom/server 真渲染一遍，断言产出的 HTML 里层级结构齐全。
 *
 * 为什么走 vite 的 ssrLoadModule 而不是自己用 esbuild 转：
 * esbuild 只是 vite 的**间接依赖**，pnpm 的严格 node_modules 布局下顶层 `esbuild`
 * 根本 import 不到。vite 是本项目的直接依赖，它的 ssrLoadModule 顺带把
 * TSX 编译、依赖解析、CSS 导入都处理好了。
 *
 * 注意：SSR 不执行 useEffect，草稿状态是空的，所以**所有依赖项的置灰状态都不准**。
 * 因此本脚本只断言结构（有哪些分组/小节/字段/标签），不断言置灰结果 ——
 * 置灰由 check-settings-tree.mjs 负责。
 *
 * 用法（在 frontend 目录下，需后端已在 8000 端口运行）：
 *     node scripts/check-settings-render.mjs
 */

import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const HERE = dirname(fileURLToPath(import.meta.url))
const FRONTEND = join(HERE, '..')

const args = process.argv.slice(2)
const urlIdx = args.indexOf('--url')
const BASE = urlIdx >= 0 ? args[urlIdx + 1] : 'http://127.0.0.1:8000'
// --out <文件> 会把渲染结果连同样式导出成可独立打开的 HTML 预览，
// 不用启动前端服务就能看布局效果。
const outIdx = args.indexOf('--out')
const OUT = outIdx >= 0 ? args[outIdx + 1] : null

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

let html = ''
// 声明在 try 外层：后面的断言要用它做动态基数（如总项数），
// 放在 try 里会被块级作用域挡住。
let settings = null
try {
  const { default: SettingsPanel } = await vite.ssrLoadModule(
    '/src/components/SettingsPanel.tsx',
  )

  const resp = await fetch(`${BASE}/api/settings`)
  if (!resp.ok) {
    console.error(`无法读取设置：HTTP ${resp.status}`)
    process.exit(1)
  }
  settings = await resp.json()

  // 必须包成 React 元素再渲染，不能直接调用组件函数 ——
  // 直接调用会让 hooks 失去所属的渲染上下文，报 "Invalid hook call"。
  html = renderToStaticMarkup(
    createElement(SettingsPanel, {
      settings,
      onClose: () => {},
      onSaved: () => {},
      notify: () => {},
    }),
  )
} finally {
  await vite.close()
}

console.log('='.repeat(70))
console.log('  设置面板渲染检查（SSR）')
console.log('='.repeat(70))
console.log(`  渲染出的 HTML 长度：${html.length} 字符`)

if (OUT) {
  const { readFileSync, writeFileSync, mkdirSync } = await import('node:fs')
  const css = readFileSync(join(FRONTEND, 'src', 'styles.css'), 'utf8')
  mkdirSync(dirname(OUT), { recursive: true })
  writeFileSync(
    OUT,
    `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>设置面板布局预览</title>
<style>
${css}
/* 预览专用：静态页面里把遮罩撑开，方便整体查看 */
.modal-mask { position: static; padding: 0; background: #eef1f5; }
.drawer { position: static; width: min(620px, 100%); margin: 0 auto; max-height: none; }
</style>
</head>
<body>
<div class="app"><div class="modal-mask">${html}</div></div>
</body>
</html>`,
    'utf8',
  )
  console.log(`  预览已导出：${OUT}`)
}
console.log('')

const count = (re) => (html.match(re) ?? []).length
const has = (s) => html.includes(s)

// ---------------------------------------------------------------- 抽屉骨架
console.log('[1] 抽屉骨架')
check(has('drawer'), '有抽屉容器')
check(has('系统设置'), '有标题')
check(has('settings-overview'), '有顶部概览条')
// 总项数从数据里取，不写死 —— 否则每加一个配置项都要回来改测试，
// 而这种「测试自己过期」的失败最容易让人误以为功能坏了。
const total = settings.items.length
check(
  new RegExp(`共\\s*<b>${total}</b>\\s*项可调`).test(html),
  `概览条显示总项数（${total}）`,
  html.slice(html.indexOf('settings-overview'), html.indexOf('settings-overview') + 240),
)

// ---------------------------------------------------------------- 一级分组
console.log('\n[2] 一级分组')
const groupHeads = count(/class="group-head"/g)
check(groupHeads === 5, `分组标题栏 5 个（4 个可调 + 1 个只读，实际 ${groupHeads}）`)
for (const name of ['生成通道', '生成参数', '检索', '文档入库', '只读环境信息']) {
  check(has(name), `包含分组「${name}」`)
}
check(has('决定用哪个模型来组织回答'), '分组带一句话定位')
check(count(/class="group-index/g) === 5, '每个分组都有序号圆盘')
check(count(/class="chev"/g) === 5, '每个分组都有折叠箭头')

// ---------------------------------------------------------------- 主控卡片
console.log('\n[3] 主控卡片')
const masterCards = count(/class="master-card"/g)
check(masterCards === 2, `主控卡片 2 个（实际 ${masterCards}）`)
check(count(/class="master-input"/g) === 2, '主控卡片都有输入区')
check(count(/class="master-label"/g) === 2, '主控卡片都有标签区')

// ---------------------------------------------------------------- 二级小节
console.log('\n[4] 二级小节')
const sections = count(/class="settings-section/g)
check(sections === 10, `小节 10 个（实际 ${sections}）`)
for (const name of [
  '本地 Ollama',
  '云端 / OpenAI 兼容接口',
  '采样控制',
  '上下文与显存',
  '回答准则',
  '双路召回',
  '融合与筛选',
  '多轮对话',
  '向量模型',
  '切片尺寸',
]) {
  check(has(name), `包含小节「${name}」`)
}
check(count(/class="section-head"/g) === 10, '每个小节都有标题行')

// ---------------------------------------------------------------- 字段
console.log('\n[5] 字段')
const fields = count(/class="field[" ]/g)
check(fields === 16, `字段行 16 个（18 项 - 2 个主控，实际 ${fields}）`)
check(count(/class="field-input"/g) === 16, '每个字段都有输入区')
check(count(/class="field-label"/g) === 16, '每个字段都有标签区')

// ---------------------------------------------------------------- 标签
console.log('\n[6] 标签与徽标')
check(
  count(/class="tag-scope"/g) === 3,
  `范围标签 3 个（切片长度/重叠/向量模型，实际 ${count(/class="tag-scope"/g)}）`,
)
check(has('仅对新建知识库生效'), '范围标签文案正确')
check(
  count(/class="badge-inactive"/g) >= 1,
  'SSR 空草稿下应出现「当前未启用」徽标（依赖项全部判为未生效）',
)
check(has('当前未启用'), '未启用徽标文案正确')

// ---------------------------------------------------------------- 控件类型
console.log('\n[7] 控件类型')
check(
  count(/class="switch"/g) === 3,
  `开关 3 个（严格作答/混合检索/查询改写，实际 ${count(/class="switch"/g)}）`,
)
check(has('switch-text'), '开关带状态文字')
// int/float 共 9 项：温度、最大生成长度、上下文窗口、向量召回数、BM25 召回数、
// RRF 平滑常数、最终送入模型片段数、切片长度、切片重叠
check(count(/type="number"/g) === 9, `数字输入 9 个（实际 ${count(/type="number"/g)}）`)
check(has('type="password"'), 'API Key 用密码框')
check(count(/<select/g) === 1, '下拉框 1 个（生成通道）')
check(has('纯本地 · Ollama'), '下拉框用的是显示名而不是裸值')
check(has('云端 · OpenAI 兼容接口'), '下拉框选项显示名完整')

// 模型字段必须是「可自由输入的输入框 + 可展开的候选下拉」，不能是原生 datalist ——
// datalist 会按输入框当前值过滤候选，填着 bge-m3 时点开只看得到 bge-m3。
// 也不能是严格 select：Ollama 没启动时列表为空，用户就连模型名都改不了。
//
// 注意：下拉菜单要点击后才渲染（open 是个 state），SSR 触发不了交互，
// 所以这里只验证选择器的外壳结构；候选值本身由 check-settings-tree.mjs 断言。
check(count(/class="picker"/g) === 2, `模型选择器 2 个（实际 ${count(/class="picker"/g)}）`)
check(count(/class="picker-control"/g) === 2, '每个选择器都有「输入框 + 展开按钮」')
check(count(/class="picker-toggle"/g) === 2, '每个选择器都有展开按钮')
check(count(/<datalist/g) === 0, '不再使用原生 datalist（它会按当前值过滤候选）')
// 注意：**不要断言输入框的 value**。SSR 不执行 useEffect，而这个面板的草稿状态
// 正是在 useEffect 里从 settings 填充的，所以 SSR 渲染出来的 value 一律是空串。
// 这条限制已经坑过好几次断言，凡是依赖「当前值」的检查都放
// check-settings-tree.mjs（那边直接读 API 数据，不受此限）。
check(
  /向量模型[\s\S]{0,800}?class="picker"/.test(html),
  '「向量模型」字段渲染成了选择器',
)
check(
  count(/placeholder="可直接输入模型名"/g) === 2,
  '模型字段带「可直接输入」的占位提示',
)

// ---------------------------------------------------------------- 底部操作栏
console.log('\n[8] 底部操作栏')
check(has('还原默认值'), '有还原按钮')
check(has('没有未保存的改动'), '初始状态下提示无改动')
check(count(/disabled=""/g) >= 1, '无改动时保存按钮禁用')

// ---------------------------------------------------------------- 只读区
console.log('\n[9] 只读环境信息')
check(has('由 .env 与运行环境决定'), '只读分组有说明')
check(has('group-readonly'), '只读分组有独立样式类')
check(!has('ollama_base_url'), '只读分组默认折叠，内容不渲染')

console.log('\n' + '='.repeat(70))
if (failures.length) {
  console.log(`未通过 ${failures.length} 项：`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
console.log('  全部通过 ✅')
