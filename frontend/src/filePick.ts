/**
 * 文件选择与识别（纯逻辑，不含 React 依赖）。
 *
 * ## 为什么过滤必须在代码里做
 *
 * 「选择文件夹」时浏览器会把目录里**所有**文件都交给我们：Office 保存时留下的
 * 锁文件 `~$报告.docx`、系统的 `Thumbs.db`、以及 `node_modules` 里成千上万个文件。
 * `<input accept="...">` 属性**对文件夹选择器不生效**，指望它过滤是不可能的。
 *
 * 这里分成两类跳过，处理方式不同：
 *
 *   * **垃圾文件**（临时/隐藏/依赖目录里的东西）——静默跳过，不打扰用户。
 *     用户选了一个项目文件夹，不希望收到 300 条「node_modules/xxx 格式不支持」。
 *   * **格式不支持的文件**——列出来告诉用户。用户可能以为 `.pptx` 能传，
 *     不说清楚他会以为是自己选错了。
 *
 * 支持的扩展名与后端 `SUPPORTED_TYPES` 一致；后端才是权威，
 * 这里提前筛掉只是为了不白传一遍再被拒。
 */

export const SUPPORTED_EXTENSIONS = [
  '.pdf',
  '.docx',
  '.md',
  '.markdown',
  '.txt',
  '.text',
  '.log',
] as const

/** 这些目录名出现在路径里就直接跳过整个文件 */
const SKIP_DIR_SEGMENTS = new Set([
  'node_modules',
  '.git',
  '.svn',
  '.hg',
  '__pycache__',
  '__macosx',
  'venv',
  '.venv',
  'env',
  'site-packages',
  'dist',
  'build',
  '.idea',
  '.vscode',
  '.next',
  'target',
  'obj',
  'bin',
  'cmake-build-debug',
  'cmake-build-release',
])

const SKIP_BASENAMES = new Set(['thumbs.db', 'desktop.ini', '.ds_store', 'ehthumbs.db'])

/** 取扩展名（小写、含点）。没有扩展名或只有隐藏文件名时返回空串。 */
export function extensionOf(path: string): string {
  const base = path.split('/').pop() ?? path
  const dot = base.lastIndexOf('.')
  // dot > 0：排除 ".gitignore" 这种「点开头无扩展名」的情况
  return dot > 0 ? base.slice(dot).toLowerCase() : ''
}

export function baseNameOf(path: string): string {
  return path.split('/').pop() ?? path
}

/** 是否是应当静默跳过的垃圾文件。 */
export function isJunkPath(path: string): boolean {
  const segments = path.split('/').filter(Boolean)
  if (segments.some((s) => SKIP_DIR_SEGMENTS.has(s.toLowerCase()))) return true

  const base = (segments[segments.length - 1] ?? '').toLowerCase()
  if (SKIP_BASENAMES.has(base)) return true
  // Office 保存时产生的锁文件
  if (base.startsWith('~$')) return true
  // 隐藏文件
  if (base.startsWith('.')) return true
  return false
}

export function isSupportedDocument(path: string): boolean {
  return (SUPPORTED_EXTENSIONS as readonly string[]).includes(extensionOf(path))
}

export interface PickClassification {
  /** 可以上传的文件下标 */
  usable: number[]
  /** 格式不支持的下标 —— 会告知用户 */
  unsupported: number[]
  /** 垃圾文件下标 —— 静默跳过 */
  junk: number[]
}

/**
 * 按下标分类，而不是返回新的对象数组。
 *
 * 调用方手里是 `File[]`，返回对象数组会迫使它再做一次「按名字找回去」的
 * 映射 —— 而文件名可能重复，那一步是错的。返回下标就没这个问题。
 */
export function classifyPaths(paths: string[]): PickClassification {
  const out: PickClassification = { usable: [], unsupported: [], junk: [] }
  paths.forEach((path, index) => {
    if (isJunkPath(path)) out.junk.push(index)
    else if (isSupportedDocument(path)) out.usable.push(index)
    else out.unsupported.push(index)
  })
  return out
}

/** 上传失败时可能连不上，先给一句人话说明白支持什么。 */
export function supportedHint(): string {
  return SUPPORTED_EXTENSIONS.join('  ')
}

/** 取扩展名给用户看（用于「不支持 .pptx」这类提示）。 */
export function describeUnsupported(path: string): string {
  const ext = extensionOf(path)
  return ext || '（无扩展名）'
}
