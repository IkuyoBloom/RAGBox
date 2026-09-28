import type { Health, SearchMode } from '../types'
import Logo from './Logo'

interface Props {
  health: Health | null
  provider: string
  providerOptions: string[]
  onProviderChange: (v: string) => void
  mode: SearchMode
  onModeChange: (v: SearchMode) => void
  onOpenSettings: () => void
  onRefresh: () => void
  onNewConversation: () => void
  hasConversation: boolean
  theme: 'light' | 'dark'
  onToggleTheme: () => void
}

/**
 * 主题切换图标。
 *
 * 用图形而不是「深色/浅色」两个字：顶栏已经有 3 个文字按钮，
 * 再加一个会让这一排越来越长；而且太阳/月亮是通用符号，不需要读文字。
 */
function ThemeIcon({ theme }: { theme: 'light' | 'dark' }) {
  // 当前是深色 -> 显示太阳（点它切到浅色）；反之显示月亮
  return theme === 'dark' ? (
    <svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="4.2" fill="currentColor" />
      <g stroke="currentColor" strokeWidth="2" strokeLinecap="round">
        <path d="M12 2.6v2.6M12 18.8v2.6M2.6 12h2.6M18.8 12h2.6" />
        <path d="M5.4 5.4l1.8 1.8M16.8 16.8l1.8 1.8M18.6 5.4l-1.8 1.8M7.2 16.8l-1.8 1.8" />
      </g>
    </svg>
  ) : (
    <svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M20.5 14.6A8.6 8.6 0 1 1 9.4 3.5a7 7 0 0 0 11.1 11.1z"
        fill="currentColor"
      />
    </svg>
  )
}

const MODE_LABELS: Record<SearchMode, string> = {
  hybrid: '混合检索',
  dense_only: '仅向量',
  sparse_only: '仅 BM25',
}

const PROVIDER_LABELS: Record<string, string> = {
  ollama: '本地 Ollama',
  openai: '云端 API',
}

export default function TopBar(props: Props) {
  const { health } = props
  const active = health?.providers.find((p) => p.name === props.provider)
  const ready = active?.ok ?? false

  return (
    <header className="topbar">
      <div className="topbar-brand">
        <Logo size={34} />
        <div className="brand-title">本地知识库问答系统</div>
      </div>

      <div className="topbar-controls">
        <label className="control">
          <span className="control-label">生成通道</span>
          <select
            value={props.provider}
            onChange={(e) => props.onProviderChange(e.target.value)}
          >
            {props.providerOptions.map((p) => (
              <option key={p} value={p}>
                {PROVIDER_LABELS[p] ?? p}
              </option>
            ))}
          </select>
        </label>

        <label className="control">
          <span className="control-label">检索模式</span>
          <select
            value={props.mode}
            onChange={(e) => props.onModeChange(e.target.value as SearchMode)}
          >
            {(Object.keys(MODE_LABELS) as SearchMode[]).map((m) => (
              <option key={m} value={m}>
                {MODE_LABELS[m]}
              </option>
            ))}
          </select>
        </label>

        <div
          className={`status-pill ${ready ? 'ok' : 'bad'}`}
          title={active?.detail ?? '未知状态'}
        >
          <span className="dot" />
          {ready ? '通道正常' : '通道不可用'}
        </div>

        <button
          className="btn ghost icon-only"
          onClick={props.onToggleTheme}
          title={props.theme === 'dark' ? '切换到浅色主题' : '切换到深色主题'}
          aria-label={props.theme === 'dark' ? '切换到浅色主题' : '切换到深色主题'}
        >
          <ThemeIcon theme={props.theme} />
        </button>
        <button className="btn ghost" onClick={props.onRefresh} title="重新检测状态">
          刷新
        </button>
        <button
          className="btn ghost"
          onClick={props.onNewConversation}
          disabled={!props.hasConversation}
        >
          新对话
        </button>
        <button className="btn" onClick={props.onOpenSettings}>
          设置
        </button>
      </div>
    </header>
  )
}
