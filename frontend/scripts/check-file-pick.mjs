/**
 * 文件夹导入的识别与过滤逻辑验证。
 *
 * 这里构造的是**真实会遇到的目录结构**：一个项目文件夹里混着依赖目录、
 * 编辑器配置、Office 锁文件、系统缩略图缓存，以及各种不支持的格式。
 * 用真实形态的样例才能发现「过滤规则写漏了一类」这种问题。
 *
 * 用法（在 frontend 目录下）：
 *     node scripts/check-file-pick.mjs
 */

import {
  SUPPORTED_EXTENSIONS,
  baseNameOf,
  classifyPaths,
  describeUnsupported,
  extensionOf,
  isJunkPath,
  isSupportedDocument,
} from '../src/filePick.ts'

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

console.log('='.repeat(72))
console.log('  文件夹导入识别逻辑验证')
console.log('='.repeat(72))

// ---------------------------------------------------------------- 扩展名
console.log('\n[1] 扩展名解析')
check(extensionOf('a/b/c.pdf') === '.pdf', '从路径里取出扩展名')
check(extensionOf('报告.DOCX') === '.docx', '扩展名统一转小写')
check(extensionOf('noext') === '', '无扩展名返回空串')
// .gitignore 的最后一个点在下标 0，不能被当成扩展名
check(extensionOf('.gitignore') === '', '隐藏文件不算有扩展名')
check(extensionOf('archive.tar.gz') === '.gz', '多点文件取最后一段')
check(baseNameOf('a/b/c.pdf') === 'c.pdf', '取基名')

// ---------------------------------------------------------------- 垃圾文件
console.log('\n[2] 垃圾文件识别（静默跳过）')
const junk = [
  ['node_modules/react/index.js', '依赖目录'],
  ['project/.git/config', '版本控制目录'],
  ['__pycache__/a.pyc', 'Python 缓存'],
  ['.venv/Lib/site-packages/x.py', '虚拟环境'],
  ['dist/bundle.js', '构建产物'],
  ['.vscode/settings.json', '编辑器配置'],
  ['__MACOSX/._报告.pdf', 'macOS 打包残留'],
  ['~$报告.docx', 'Office 锁文件'],
  ['.DS_Store', 'macOS 隐藏文件'],
  ['Thumbs.db', 'Windows 缩略图缓存'],
  ['desktop.ini', 'Windows 桌面配置'],
  ['a/b/.hidden.md', '隐藏文件'],
  ['CMakeFiles/obj/main.o', '构建中间产物'],
]
for (const [path, why] of junk) {
  check(isJunkPath(path), `${why}：${path}`)
}

// 反向样本：普通文档绝不能被误判成垃圾
console.log('\n[3] 反向样本（不能误杀）')
const real = [
  '论文.pdf',
  '论文/2024/注意力机制.pdf',
  'project/README.md',
  'data/环境配置.txt',
  'notes/2024-总结.markdown',
  'app.log',
  'a/b/c.docx',
]
for (const path of real) {
  check(!isJunkPath(path), `正常文档不被当作垃圾：${path}`)
}

// ---------------------------------------------------------------- 支持格式
console.log('\n[4] 支持格式判定')
for (const ext of SUPPORTED_EXTENSIONS) {
  check(isSupportedDocument(`dir/file${ext}`), `支持 ${ext}`)
}
for (const ext of ['.pptx', '.xlsx', '.png', '.zip', '.exe', '.py', '.json', '.html']) {
  check(!isSupportedDocument(`dir/file${ext}`), `不支持 ${ext}`)
}
check(!isSupportedDocument('noext'), '无扩展名不支持')
check(describeUnsupported('a/b.pptx') === '.pptx', '不支持格式的提示给出扩展名')
check(describeUnsupported('a/b') === '（无扩展名）', '无扩展名有专门提示')

// ---------------------------------------------------------------- 真实目录
console.log('\n[5] 真实目录结构分类')

const FOLDER = [
  // 应当导入的
  '毕业论文/开题报告.pdf',
  '毕业论文/参考文献/注意力机制综述.pdf',
  '毕业论文/参考文献/RAG技术.md',
  '毕业论文/笔记/第三章.txt',
  '毕业论文/README.md',
  '参考文献/README.md', // 与上面同名，但路径不同
  // 垃圾（静默跳过）
  '毕业论文/node_modules/pkg/index.js',
  '毕业论文/.git/config',
  '毕业论文/参考文献/~$注意力机制综述.docx',
  '毕业论文/Thumbs.db',
  '毕业论文/.DS_Store',
  '毕业论文/__pycache__/x.pyc',
  // 不支持（告知用户）
  '毕业论文/答辩.pptx',
  '毕业论文/数据统计.xlsx',
  '毕业论文/流程图.png',
  '毕业论文/附件.zip',
]

const cls = classifyPaths(FOLDER)
const usablePaths = cls.usable.map((i) => FOLDER[i])
const unsupportedPaths = cls.unsupported.map((i) => FOLDER[i])
const junkPaths = cls.junk.map((i) => FOLDER[i])

console.log(`  可导入 ${usablePaths.length} 个：`)
for (const p of usablePaths) console.log(`      ${p}`)
console.log(`  不支持 ${unsupportedPaths.length} 个：`)
for (const p of unsupportedPaths) console.log(`      ${p}`)
console.log(`  忽略   ${junkPaths.length} 个（临时/隐藏/依赖目录）`)

check(usablePaths.length === 6, `识别出 6 个可导入文档（实际 ${usablePaths.length}）`)
check(unsupportedPaths.length === 4, `识别出 4 个不支持的（实际 ${unsupportedPaths.length}）`)
check(junkPaths.length === 6, `识别出 6 个垃圾文件（实际 ${junkPaths.length}）`)
check(
  cls.usable.length + cls.unsupported.length + cls.junk.length === FOLDER.length,
  '分类结果互斥且完备（三个下标数组合计等于总数）',
)
check(
  new Set([...cls.usable, ...cls.unsupported, ...cls.junk]).size === FOLDER.length,
  '没有任何文件被重复分类',
)
check(
  usablePaths.includes('毕业论文/README.md') && usablePaths.includes('参考文献/README.md'),
  '不同子目录下的同名文件都被识别（这正是需要保留相对路径的原因）',
)
check(
  !junkPaths.includes('毕业论文/参考文献/RAG技术.md'),
  'md 文件不会因为目录里同时有 node_modules 就被误杀',
)

// ---------------------------------------------------------------- 边界
console.log('\n[6] 边界情况')
check(classifyPaths([]).usable.length === 0, '空列表不报错')
// 空路径现实中不会出现（每个 File 都有名字），所以没必要规定它归哪一类，
// 只断言唯一必须成立的性质：绝不能被当成可导入文档。
check(
  classifyPaths(['']).usable.length === 0,
  '空路径不会被当成可导入文档',
)
const allJunk = classifyPaths(['node_modules/a.js', '.git/b'])
check(
  allJunk.usable.length === 0 && allJunk.unsupported.length === 0,
  '全是垃圾文件时可用与不支持都为空（调用方据此提示「没有可导入的文档」）',
)
const allUnsupported = classifyPaths(['a.pptx', 'b.xlsx'])
check(
  allUnsupported.usable.length === 0 && allUnsupported.unsupported.length === 2,
  '全是不支持格式时能被识别出来（调用方据此提示支持哪些格式）',
)

console.log('\n' + '='.repeat(72))
if (failures.length) {
  console.log(`未通过 ${failures.length} 项：`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
console.log('  全部通过 ✅')
