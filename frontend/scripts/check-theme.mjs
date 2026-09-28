/**
 * 深色主题检查。
 *
 * 深色模式最容易坏的方式，是某天有人加了一条写死浅色的规则：
 *
 *     .some-new-card { background: #fff; }
 *
 * 浅色下完全正常，深色下就是一块刺眼的白 —— 而且**不会报错、不会崩**，
 * 只有切到深色才看得见。所以这里做静态检查，把这类回归挡在提交前。
 *
 * 检查四件事：
 *   1. 深色块覆盖了浅色块里的每一个颜色变量（漏一个就有地方不变色）；
 *   2. 代码里引用的 var(--x) 都有定义（打错字会静默失效）；
 *   3. 两个定义块之外没有硬编码颜色（`color: #fff` 除外 —— 彩色底上的白字
 *      在两种主题下都该是白的）；
 *   4. 两套主题的键集合一致，不多不少。
 *
 * 只读 CSS，不依赖 vite，跑得很快。
 *
 * 用法（在 frontend 目录下）：
 *     node scripts/check-theme.mjs
 */

import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const FRONTEND = join(HERE, '..')
const CSS_PATH = join(FRONTEND, 'src', 'styles.css')

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

const css = readFileSync(CSS_PATH, 'utf8')

/** 取出某个选择器块的 `--x: value` 声明。 */
function readBlock(selector) {
  const re = new RegExp(`(?:^|\\n)${selector}\\s*\\{([\\s\\S]*?)\\n\\}`)
  const m = css.match(re)
  if (!m) return null
  const vars = {}
  for (const line of m[1].split('\n')) {
    const d = line.match(/^\s*(--[\w-]+):\s*(.+?);\s*$/)
    if (d) vars[d[1]] = d[2].trim()
  }
  return { body: m[1], vars, full: m[0] }
}

console.log('='.repeat(72))
console.log('  深色主题检查')
console.log('='.repeat(72))
console.log()

const light = readBlock(':root')
const dark = readBlock("\\[data-theme='dark'\\]")

check(Boolean(light), '定义了 :root 浅色变量')
check(Boolean(dark), "定义了 [data-theme='dark'] 深色变量")
if (!light || !dark) process.exit(1)

console.log(`  浅色变量 ${Object.keys(light.vars).length} 个，深色 ${Object.keys(dark.vars).length} 个\n`)

// ---------------------------------------------------------------- 覆盖完整
console.log('[1] 深色必须覆盖每一个浅色变量')
// 尺寸类变量与主题无关，深色块不必重复声明
const THEME_NEUTRAL = new Set(['--radius', '--radius-sm', '--mono'])
const missing = Object.keys(light.vars).filter(
  (k) => !(k in dark.vars) && !THEME_NEUTRAL.has(k),
)
check(
  missing.length === 0,
  `颜色类变量全部被深色覆盖（缺 ${missing.length} 个）`,
  missing.join(', '),
)

const extra = Object.keys(dark.vars).filter((k) => !(k in light.vars))
check(extra.length === 0, `深色没有引入浅色里不存在的变量（多 ${extra.length} 个）`, extra.join(', '))

// 值必须真的不同 —— 复制粘贴忘了改是常见疏漏
const sameValue = Object.keys(dark.vars).filter(
  (k) => k in light.vars && light.vars[k] === dark.vars[k],
)
check(
  sameValue.length === 0,
  `深色的取值与浅色不同（相同 ${sameValue.length} 个）`,
  sameValue.join(', '),
)

// ---------------------------------------------------------------- 引用完整
console.log('\n[2] 代码里引用的变量必须都有定义')
const used = new Set([...css.matchAll(/var\((--[\w-]+)\)/g)].map((m) => m[1]))
const undefinedVars = [...used].filter((v) => !(v in light.vars))
check(
  undefinedVars.length === 0,
  `引用的 ${used.size} 个变量全部有定义（未定义 ${undefinedVars.length} 个）`,
  undefinedVars.join(', '),
)

// 自我引用会让整套变量失效（批量替换时踩过）
const selfRefs = Object.entries(light.vars)
  .filter(([k, v]) => v.includes(`var(${k})`))
  .map(([k]) => k)
check(selfRefs.length === 0, '浅色变量没有自我引用', selfRefs.join(', '))

// ---------------------------------------------------------------- 无硬编码
console.log('\n[3] 定义块之外不得出现硬编码颜色')
const stripped = css.replace(light.full, '').replace(dark.full, '')
const leftovers = [...stripped.matchAll(/#[0-9a-fA-F]{3,8}\b/g)].map((m) => m[0].toLowerCase())
// color: #fff 是彩色底上的白字，两种主题下都该是白的，允许保留
const allowed = leftovers.filter((v) => v === '#fff' || v === '#ffffff')
const offenders = leftovers.length - allowed.length
check(
  offenders === 0,
  `定义块之外没有硬编码颜色（${leftovers.length} 处，其中 ${allowed.length} 处是白字）`,
  leftovers.filter((v) => v !== '#fff' && v !== '#ffffff').join(', '),
)

// ---------------------------------------------------------------- 属性挂载
console.log('\n[4] 主题如何生效')
const appTsx = readFileSync(join(FRONTEND, 'src', 'App.tsx'), 'utf8')
check(appTsx.includes('document.documentElement.dataset.theme'), '主题写到 <html data-theme> 上')
check(appTsx.includes('prefers-color-scheme'), '默认跟随系统偏好')
check(appTsx.includes('localStorage.setItem'), '记住用户的选择')
check(appTsx.includes('FORCED_THEME'), '支持 URL 强制主题（供截图验证使用）')
check(css.includes('color-scheme: light'), '浅色声明 color-scheme（原生控件跟着变）')
check(css.includes('color-scheme: dark'), '深色声明 color-scheme')

console.log('\n' + '='.repeat(72))
if (failures.length) {
  console.log(`未通过 ${failures.length} 项：`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
console.log('  全部通过 ✅')
