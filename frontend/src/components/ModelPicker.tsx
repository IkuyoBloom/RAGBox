import { useEffect, useRef, useState } from 'react'

interface Props {
  value: string
  options: string[]
  placeholder?: string
  emptyHint?: string
  onChange: (value: string) => void
}

/**
 * 模型名选择器：可自由输入的输入框 + 可展开的候选下拉。
 *
 * ## 为什么不用原生 `<datalist>`
 *
 * datalist 看起来正合适，但它有一个致命行为：**浏览器会按输入框的当前值过滤候选**。
 * 输入框里已经填着 `bge-m3` 时，点开只看得到 `bge-m3`，
 * 另一个候选 `qwen3-embedding:4b` 仿佛不存在 —— 而用户恰恰是为了换成它才点开的。
 * 这个过滤行为无法通过属性关闭。
 *
 * 所以自己实现下拉，把「列全部」和「按输入过滤」分成两种状态：
 *   * 聚焦 / 点箭头 / 点输入框 → 列出**全部**候选；
 *   * 真正敲了字 → 才按输入过滤。
 *
 * 同时保留自由输入（不受候选限制）：Ollama 没启动时候选列表必然是空的，
 * 而「换个模型名试试」正是从这种状态里自救的主要手段。
 */
export default function ModelPicker({
  value,
  options,
  placeholder,
  emptyHint,
  onChange,
}: Props) {
  const [open, setOpen] = useState(false)
  // null 表示「还没开始输入」——此时列出全部候选。
  // 这是与 datalist 行为的分水岭，必须区分「聚焦」与「输入」。
  const [query, setQuery] = useState<string | null>(null)
  // 用 fixed 定位：抽屉主体是 overflow:auto 的，绝对定位的下拉会被裁掉，
  // 而「文档入库」这组正好在面板底部，被裁的概率很高。
  const [anchor, setAnchor] = useState<{ top: number; left: number; width: number } | null>(null)
  const boxRef = useRef<HTMLDivElement>(null)
  // 记录打开时刻：浏览器在聚焦输入框时可能为了把它滚入视口而触发一次 scroll，
  // 那不是用户主动滚动，不该把刚打开的下拉立刻收掉（表现为「闪一下就没」）。
  const openedAt = useRef(0)

  const measure = () => {
    const el = boxRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    // 若下方空间不足就向上弹，避免下拉超出视口
    const menuHeight = Math.min(240, options.length * 32 + 40)
    const below = window.innerHeight - rect.bottom
    const top = below >= menuHeight ? rect.bottom + 4 : Math.max(8, rect.top - menuHeight - 4)
    setAnchor({ top, left: rect.left, width: rect.width })
  }

  const openMenu = () => {
    measure()
    openedAt.current = Date.now()
    setOpen(true)
  }

  const close = () => {
    setOpen(false)
    setQuery(null)
  }

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) close()
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
    }
    // 滚动/缩放后固定定位的下拉会与输入框错位，直接收起最省事。
    // 但要放过「打开瞬间的那次自动滚动」，否则下拉会闪一下就没。
    const onShift = () => {
      if (Date.now() - openedAt.current < 350) return
      close()
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    window.addEventListener('resize', onShift)
    document.addEventListener('scroll', onShift, true)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
      window.removeEventListener('resize', onShift)
      document.removeEventListener('scroll', onShift, true)
    }
  }, [open])

  const shown =
    query === null
      ? options
      : options.filter((name) => name.toLowerCase().includes(query.trim().toLowerCase()))

  const pick = (name: string) => {
    onChange(name)
    close()
  }

  return (
    <div className="picker" ref={boxRef}>
      <div className="picker-control">
        <input
          type="text"
          value={value}
          placeholder={placeholder}
          onFocus={openMenu}
          onChange={(e) => {
            onChange(e.target.value)
            setQuery(e.target.value)
            if (!open) openMenu()
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && shown.length > 0 && query !== null) {
              e.preventDefault()
              pick(shown[0])
            }
          }}
        />
        <button
          type="button"
          className="picker-toggle"
          title={options.length > 0 ? `展开全部 ${options.length} 个候选` : '展开候选'}
          // 阻止默认动作，避免按下时输入框先失焦导致下拉闪一下
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => (open ? close() : openMenu())}
        >
          ▾
        </button>
      </div>

      {open && anchor && (
        <div
          className="picker-menu"
          style={{
            position: 'fixed',
            top: anchor.top,
            left: anchor.left,
            width: anchor.width,
          }}
        >
          {options.length === 0 ? (
            <div className="picker-empty">
              {emptyHint ?? '没有候选列表，可直接手动输入'}
            </div>
          ) : (
            <>
              <div className="picker-head">
                {query === null
                  ? `共 ${options.length} 个候选`
                  : `匹配 ${shown.length} / ${options.length}`}
              </div>
              {shown.length === 0 ? (
                <div className="picker-empty">
                  没有匹配「{query}」的候选，仍可直接使用这个名称
                </div>
              ) : (
                shown.map((name) => (
                  <button
                    type="button"
                    key={name}
                    className={`picker-item ${name === value ? 'current' : ''}`}
                    onClick={() => pick(name)}
                  >
                    <span className="picker-name">{name}</span>
                    {name === value && <span className="picker-badge">当前</span>}
                  </button>
                ))
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
