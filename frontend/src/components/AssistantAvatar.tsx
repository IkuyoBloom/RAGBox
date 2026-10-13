/**
 * 助手头像 — 简单文字缩写，无 AI 专属视觉。
 *
 * 用字母「K」代表「知识库」（Knowledge Base）。
 * 与用户头像「我」风格完全一致，只是背景色略有区分。
 */
interface Props {
  size?: number
  className?: string
}

export default function AssistantAvatar({ className }: Props) {
  return (
    <div className={`avatar-ai ${className ?? ''}`} aria-label="助手">
      K
    </div>
  )
}
