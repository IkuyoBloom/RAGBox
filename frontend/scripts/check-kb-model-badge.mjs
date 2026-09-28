/**
 * 知识库「向量模型徽标」状态判定验证。
 *
 * 判定逻辑本身是纯函数（src/embedModel.ts），但它的输入既来自 API
 * 又涉及「Ollama 装没装」这种外部状态，所以既要用真实数据跑一遍，
 * 也要覆盖几种 API 数据造不出来的边界（比如 Ollama 未启动、模型被删）。
 *
 * 用法（在 frontend 目录下，需后端已运行）：
 *     node scripts/check-kb-model-badge.mjs
 */

import { KB_MODEL_STATE_TEXT, describeKbModel, normalizeModel, sameModel } from '../src/embedModel.ts'

const args = process.argv.slice(2)
const urlIdx = args.indexOf('--url')
const BASE = urlIdx >= 0 ? args[urlIdx + 1] : 'http://127.0.0.1:8000'

const PASS = '  [通过]  '
const FAIL = '  [失败]  '
const INFO = '  [信息]  '
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

// ---------------------------------------------------------------- 纯函数
console.log('='.repeat(72))
console.log('  知识库向量模型徽标验证')
console.log('='.repeat(72))

console.log('\n[1] 模型名归一化')
check(normalizeModel('bge-m3') === 'bge-m3:latest', '省略 tag 补成 :latest')
check(normalizeModel('bge-m3:latest') === 'bge-m3:latest', '已有 tag 保持不变')
check(normalizeModel('qwen3-embedding:4b') === 'qwen3-embedding:4b', '带版本号的 tag 不被改写')
check(sameModel('bge-m3', 'bge-m3:latest'), 'bge-m3 与 bge-m3:latest 视为同一模型')
check(!sameModel('bge-m3', 'qwen3-embedding:4b'), '不同模型判为不同')
// 回归：曾经因为不归一化，把同一个模型误判成两个，进而误报「模型不一致」
check(sameModel('bge-m3:latest', 'bge-m3'), '两种写法互通（方向相反也成立）')

// ---------------------------------------------------------------- 合成场景
console.log('\n[2] 状态判定（合成场景，覆盖 API 造不出来的情况）')

const kb = (over = {}) => ({
  id: 'x',
  name: 'k',
  embed_model: 'bge-m3',
  embed_dim: 1024,
  chunk_count: 10,
  document_count: 1,
  ready_document_count: 1,
  chunk_size: 500,
  chunk_overlap: 80,
  description: '',
  collection_name: 'c',
  created_at: '',
  updated_at: '',
  ...over,
})

const upToDate = { current: 'bge-m3', installed: ['bge-m3', 'qwen3-embedding:4b'], loaded: true }
const switched = { current: 'qwen3-embedding:4b', installed: ['bge-m3', 'qwen3-embedding:4b'], loaded: true }
const removed = { current: 'bge-m3', installed: ['qwen3-embedding:4b'], loaded: true }
const offline = { current: 'bge-m3', installed: ['bge-m3'], loaded: false }

check(
  describeKbModel(kb(), upToDate).state === 'ok',
  '模型与设置一致 -> ok',
  describeKbModel(kb(), upToDate).state,
)
check(
  describeKbModel(kb(), switched).state === 'differs',
  '库的模型与当前设置不同 -> differs（正常情况，不是错误）',
  describeKbModel(kb(), switched).state,
)
check(
  describeKbModel(kb(), removed).state === 'missing',
  '库的模型不在 Ollama 中 -> missing',
  describeKbModel(kb(), removed).state,
)
check(
  describeKbModel(kb(), removed).hint.includes('ollama pull bge-m3'),
  'missing 状态的提示给出了可执行命令',
  describeKbModel(kb(), removed).hint,
)
check(
  describeKbModel(kb({ chunk_count: 0, embed_dim: null }), switched).state === 'pending',
  '尚未索引内容 -> pending',
  describeKbModel(kb({ chunk_count: 0 }), switched).state,
)
// 关键：Ollama 没启动时不能把每个库都标成红色
check(
  describeKbModel(kb(), offline).state === 'unknown',
  'Ollama 未启动（choices_loaded=false）-> unknown，不误报未安装',
  describeKbModel(kb(), offline).state,
)
// 优先级：模型没装 > 未索引
check(
  describeKbModel(kb({ chunk_count: 0 }), removed).state === 'missing',
  '「模型未安装」优先于「尚未索引」（前者更严重，会导致该库彻底不可用）',
  describeKbModel(kb({ chunk_count: 0 }), removed).state,
)
// 省略 tag 的库模型 vs 带 tag 的已装列表
check(
  describeKbModel(kb({ embed_model: 'bge-m3' }), upToDate).state === 'ok',
  '库写 bge-m3、Ollama 报 bge-m3:latest 时不误报不一致',
)
check(
  Object.keys(KB_MODEL_STATE_TEXT).length === 5,
  '五种状态都有对应的短标签文案',
)

// ---------------------------------------------------------------- 真实数据
console.log('\n[3] 真实数据')
let kbs = []
let settings = null
try {
  kbs = await (await fetch(`${BASE}/api/kb`)).json()
  settings = await (await fetch(`${BASE}/api/settings`)).json()
} catch (e) {
  console.log(`${FAIL}读取 API 失败：${e.message}`)
  process.exit(1)
}

const item = settings.items.find((i) => i.key === 'ollama_embed_model')
const info = {
  current: typeof item?.value === 'string' ? item.value : '',
  installed: item?.choices ?? [],
  loaded: item?.choices_loaded ?? false,
}

console.log(`${INFO}当前向量模型 : ${info.current}`)
console.log(`${INFO}Ollama 已装  : ${info.installed.join('、') || '(无)'}`)
console.log(`${INFO}候选已取到   : ${info.loaded}`)
console.log()
console.log(`  ${'知识库'.padEnd(16, '　')} ${'库的模型'.padEnd(22, '　')} ${'状态'}`)
console.log('  ' + '-'.repeat(60))
for (const k of kbs) {
  const b = describeKbModel(k, info)
  const note = KB_MODEL_STATE_TEXT[b.state]
  console.log(
    `  ${k.name.padEnd(16, '　')} ${k.embed_model.padEnd(22, '　')} ${b.state}${note ? ` (${note})` : ''}`,
  )
}

check(kbs.length > 0, `至少有一个知识库（实际 ${kbs.length} 个）`)
check(
  kbs.every((k) => ['ok', 'differs', 'pending', 'missing', 'unknown'].includes(describeKbModel(k, info).state)),
  '所有知识库都能判出合法状态',
)
if (info.loaded && info.installed.length > 0) {
  check(
    kbs.every((k) => describeKbModel(k, info).state !== 'missing' || !info.installed.some((n) => sameModel(n, k.embed_model))),
    '被标为 missing 的库，其模型确实不在已装列表里',
  )
}

console.log('\n' + '='.repeat(72))
if (failures.length) {
  console.log(`未通过 ${failures.length} 项：`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
console.log('  全部通过 ✅')
