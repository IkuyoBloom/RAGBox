/**
 * 设置界面层级逻辑验证。
 *
 * 直接调用后端 /api/settings，把返回的数据喂给 SettingsPanel 使用的同一份纯逻辑
 * （`src/settingsTree.ts`），断言三级层级与依赖判定都正确，并把树打印成 ASCII
 * 方便肉眼核对。
 *
 * 为什么要这一步：前端「能编译通过」不等于「层级渲染对」。
 * 这层验证不需要浏览器、不需要 jsdom，靠 Node 自带的 TypeScript 类型剥离就能跑。
 *
 * 用法（在 frontend 目录下，需后端已在 8000 端口运行）：
 *     node scripts/check-settings-tree.mjs
 *     node scripts/check-settings-tree.mjs --url http://127.0.0.1:8000
 */

import {
  buildTree,
  computeChanged,
  dependencyHint,
  draftFrom,
  groupActivation,
  isActive,
} from '../src/settingsTree.ts'

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

const resp = await fetch(`${BASE}/api/settings`)
if (!resp.ok) {
  console.error(`无法读取设置：HTTP ${resp.status}`)
  process.exit(1)
}
const settings = await resp.json()

console.log('='.repeat(70))
console.log('  设置界面层级验证')
console.log('='.repeat(70))

// ---------------------------------------------------------------- 结构与层级
console.log('\n[1] 三级层级')
const tree = buildTree(settings)
const byKey = new Map(settings.items.map((i) => [i.key, i]))

console.log(`  ${settings.items.length} 项 / ${tree.length} 组\n`)
const baseDraft = draftFrom(settings.items)
for (const [gi, group] of tree.entries()) {
  const { active, total } = groupActivation(group, baseDraft)
  console.log(`  ■ ${gi + 1}. ${group.name}   (${active}/${total} 生效)`)
  if (group.summary) console.log(`  │    ${group.summary}`)
  for (const item of group.master) {
    console.log(`  │  ★ ${item.label}  [主控]`)
  }
  for (const section of group.sections) {
    console.log(`  │  ├─ ${section.name}`)
    for (const item of section.items) {
      // 只在「依赖不满足」时才显示提示 —— 与组件里的行为保持一致。
      // 曾经这里无条件打印，导致 provider=ollama 时「本地对话模型」
      // 也挂着一个 ⚠，看输出会误以为逻辑坏了。
      const hint = isActive(item, baseDraft) ? null : dependencyHint(item, byKey)
      const marks = [item.scope && `«${item.scope}»`, hint && `⚠ ${hint}`].filter(Boolean)
      console.log(`  │  │    ${item.label.padEnd(20, '　')} ${marks.join('  ')}`)
    }
  }
  console.log('  │')
}

check(tree.length === 4, `一级分组数量为 4（实际 ${tree.length}）`)
check(
  tree.map((g) => g.name).join(',') === '生成通道,生成参数,检索,文档入库',
  '一级分组按后端 order 排序',
  tree.map((g) => g.name).join(','),
)
check(
  tree.every((g) => g.summary.length > 0),
  '每个分组都带一句话定位',
  tree.filter((g) => !g.summary).map((g) => g.name).join(','),
)
const allSections = tree.flatMap((g) => g.sections)
// 实际分布：生成通道 2 + 生成参数 3 + 检索 3 + 文档入库 2 = 10
check(allSections.length === 10, `二级小节数量为 10（实际 ${allSections.length}）`)
check(
  allSections.every((s) => s.items.length > 0),
  '没有空小节',
)
check(
  settings.items.every((i) => tree.some((g) => g.master.includes(i) || g.sections.some((s) => s.items.includes(i)))),
  '所有字段都被归入某个分组（无孤儿字段）',
)
check(
  tree.reduce((n, g) => n + g.total, 0) === settings.items.length,
  '分组字段数之和等于总项数（无重复归属）',
)

// ---------------------------------------------------------------- 主控项
console.log('\n[2] 主控项')
const masterKeys = tree.flatMap((g) => g.master).map((i) => i.key)
check(
  masterKeys.length === 2 && masterKeys.includes('llm_provider') && masterKeys.includes('enable_hybrid'),
  `主控项为 llm_provider / enable_hybrid（实际 ${masterKeys.join(',')}）`,
)

// ---------------------------------------------------------------- 依赖判定
console.log('\n[3] 依赖判定（这是层级的核心价值：能看出哪一项当前不生效）')

const items = draftFrom(settings.items)

// 切到本地通道：云端三项应失效，本地模型应生效
const ollamaDraft = { ...items, llm_provider: 'ollama' }
check(
  isActive(byKey.get('ollama_chat_model'), ollamaDraft) === true,
  'llm_provider=ollama 时「本地对话模型」生效',
)
for (const key of ['openai_base_url', 'openai_api_key', 'openai_model']) {
  check(
    isActive(byKey.get(key), ollamaDraft) === false,
    `llm_provider=ollama 时「${byKey.get(key).label}」不生效`,
  )
}
check(
  isActive(byKey.get('num_ctx'), ollamaDraft) === true,
  'llm_provider=ollama 时「上下文窗口」生效',
)

// 切到云端：反过来
const openaiDraft = { ...items, llm_provider: 'openai' }
check(
  isActive(byKey.get('openai_model'), openaiDraft) === true,
  'llm_provider=openai 时「模型名」生效',
)
check(
  isActive(byKey.get('ollama_chat_model'), openaiDraft) === false,
  'llm_provider=openai 时「本地对话模型」不生效',
)
check(
  isActive(byKey.get('num_ctx'), openaiDraft) === false,
  'llm_provider=openai 时「上下文窗口」不生效',
)

// 布尔依赖：enable_hybrid
const hybridOff = { ...items, enable_hybrid: false }
for (const key of ['dense_top_k', 'sparse_top_k', 'rrf_k']) {
  check(
    isActive(byKey.get(key), hybridOff) === false,
    `混合检索关闭时「${byKey.get(key).label}」不生效`,
  )
}
check(isActive(byKey.get('final_top_k'), hybridOff) === true, '「最终送入模型片段数」始终生效')
check(
  isActive(byKey.get('enable_query_rewrite'), hybridOff) === true,
  '「多轮查询改写」不受混合检索开关影响',
)

// 布尔依赖必须能匹配上 —— 曾经写成字符串 'true' 而草稿里是布尔 true，永远不生效
const hybridOn = { ...items, enable_hybrid: true }
check(
  isActive(byKey.get('dense_top_k'), hybridOn) === true,
  '布尔依赖能正确匹配 true（草稿里是 boolean，spec 里是字符串 "true"）',
)

// ---------------------------------------------------------------- 提示文案
console.log('\n[4] 依赖提示文案')
check(
  dependencyHint(byKey.get('openai_model'), byKey)?.includes('生成通道'),
  '依赖提示指出了上级开关的名字',
  dependencyHint(byKey.get('openai_model'), byKey),
)
check(
  dependencyHint(byKey.get('dense_top_k'), byKey)?.includes('开启'),
  '布尔依赖的提示用「开启」而不是「设为 true」',
  dependencyHint(byKey.get('dense_top_k'), byKey),
)
const openaiHint = dependencyHint(byKey.get('openai_model'), byKey)
check(
  openaiHint?.includes('云端'),
  '选项依赖的提示用显示名而不是裸值（应出现「云端 · OpenAI 兼容接口」）',
  openaiHint,
)
check(dependencyHint(byKey.get('temperature'), byKey) === null, '无依赖项不产生提示')

// ---------------------------------------------------------------- 范围标签
console.log('\n[5] 生效范围标签')
const scoped = settings.items.filter((i) => i.scope)
check(
  scoped.length === 3 &&
    scoped.every((i) => i.group === '文档入库'),
  `「文档入库」组有 3 项带生效范围标签（切片长度/重叠/向量模型，实际 ${scoped.length} 项）`,
  scoped.map((i) => `${i.key}@${i.group}`).join(', '),
)
check(
  scoped.every((i) => i.scope.includes('新建知识库')),
  '范围标签文案说明了对新建库生效',
)

// ---------------------------------------------------------------- 向量模型可切换
console.log('\n[6] 向量模型可切换（本次新增）')
const embed = settings.items.find((i) => i.key === 'ollama_embed_model')
check(Boolean(embed), '向量模型出现在可调项里（不再只是只读展示）')
if (embed) {
  check(
    embed.group === '文档入库' && embed.section === '向量模型',
    '向量模型归入「文档入库 > 向量模型」',
    `${embed.group} > ${embed.section}`,
  )
  check(
    embed.choices_source === 'ollama_embedding_models',
    '声明了动态选项来源',
    String(embed.choices_source),
  )
  check(
    Array.isArray(embed.choices) && embed.choices.length > 0,
    `拿到了已装向量模型列表（实际 ${embed.choices?.length ?? 0} 个）`,
    JSON.stringify(embed.choices),
  )
  check(
    (embed.choices ?? []).includes(embed.value),
    '当前配置值出现在候选列表中（否则界面看起来像配置坏了）',
    `value=${embed.value} choices=${JSON.stringify(embed.choices)}`,
  )
  // 回归：曾经候选列表里同时出现 bge-m3 与 bge-m3:latest，
  // 它们是同一个模型（Ollama 省略 tag 时默认 latest），看起来像配错了。
  const normalized = (embed.choices ?? []).map((n) => (n.includes(':') ? n : `${n}:latest`))
  check(
    new Set(normalized).size === normalized.length,
    '候选列表中没有「同一模型的两种写法」重复',
    JSON.stringify(embed.choices),
  )
}
check(
  !('ollama_embed_model' in settings.readonly),
  '向量模型已从「只读环境信息」中移出',
  JSON.stringify(Object.keys(settings.readonly)),
)
const chatModel = settings.items.find((i) => i.key === 'ollama_chat_model')
check(
  chatModel?.choices_source === 'ollama_chat_models',
  '对话模型也接了动态选项来源',
  String(chatModel?.choices_source),
)

// ---------------------------------------------------------------- 改动计算
console.log('\n[7] 改动收集')
const base = draftFrom(settings.items)
check(Object.keys(computeChanged(settings.items, base)).length === 0, '草稿等于当前值时无改动')

const mod = { ...base, temperature: 0.9 }
const c1 = computeChanged(settings.items, mod)
check(
  Object.keys(c1).length === 1 && c1.temperature === 0.9,
  '改动一个值只收集一项',
  JSON.stringify(c1),
)

// 密钥留空不能算改动，否则「保存」会把已配置的 Key 覆盖成空串
const withEmptyKey = { ...base, openai_api_key: '' }
check(
  computeChanged(settings.items, withEmptyKey).openai_api_key === undefined,
  '密钥留空不算改动（防止误清空已配置的 Key）',
)
const withKey = { ...base, openai_api_key: 'sk-test-123' }
check(
  computeChanged(settings.items, withKey).openai_api_key === 'sk-test-123',
  '填入密钥会被正确收集',
)

// ---------------------------------------------------------------- 汇总
console.log('\n' + '='.repeat(70))
if (failures.length) {
  console.log(`未通过 ${failures.length} 项：`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
console.log('  全部通过 ✅')
