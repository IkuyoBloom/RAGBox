import type { Health } from '../types'
import Logo from './Logo'

interface Props {
  health: Health | null
  provider: string
  onOpenSettings: () => void
  onRefresh: () => void
  onNewConversation: () => void
  hasConversation: boolean
  theme: 'light' | 'dark'
  onToggleTheme: () => void
}

function ThemeIcon({ theme }: { theme: 'light' | 'dark' }) {
  return theme === 'dark' ? (
    <svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true" fill="none">
      <circle cx="12" cy="12" r="4" fill="currentColor" />
      <g stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
        <path d="M12 3v2M12 19v2M3 12h2M19 12h2" />
        <path d="M5.6 5.6l1.4 1.4M16.9 16.9l1.4 1.4M18.4 5.6l-1.4 1.4M7.1 16.9l-1.4 1.4" />
      </g>
    </svg>
  ) : (
    <svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"
        fill="currentColor"
      />
    </svg>
  )
}

export default function TopBar(props: Props) {
  const { health } = props
  const active = health?.providers.find((p) => p.name === props.provider)
  const ready = active?.ok ?? false

  return (
    <header className="topbar">
      <div className="topbar-brand">
        <Logo size={26} />
        <span className="brand-title">知识库</span>
      </div>

      <div className="topbar-right">
        {/* 状态指示点 */}
        <div
          className={`status-pill ${ready ? 'ok' : 'bad'}`}
          title={active?.detail ?? '未知状态'}
          style={{ marginRight: 4 }}
        >
          <span className="dot" />
        </div>

        {props.hasConversation && (
          <button
            className="btn ghost"
            onClick={props.onNewConversation}
            title="清空当前对话，开始新对话"
          >
            新对话
          </button>
        )}

        <button
          className="btn icon-only"
          onClick={props.onToggleTheme}
          title={props.theme === 'dark' ? '切换到浅色' : '切换到深色'}
          aria-label={props.theme === 'dark' ? '切换到浅色' : '切换到深色'}
        >
          <ThemeIcon theme={props.theme} />
        </button>

        <button
          className="btn ghost"
          onClick={props.onRefresh}
          title="重新检测后端状态"
        >
          刷新
        </button>

        <button className="btn" onClick={props.onOpenSettings}>
          设置
        </button>
      </div>
    </header>
  )
}
