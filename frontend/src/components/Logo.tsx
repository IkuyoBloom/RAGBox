import { useId } from 'react'

/**
 * 项目 Logo — 简化版放大镜图标，单色，跟随主题色。
 */
interface Props {
  size?: number
  className?: string
}

export default function Logo({ size = 26, className }: Props) {
  const clipId = `logo-clip-${useId().replace(/:/g, '')}`

  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      role="img"
      aria-label="知识库"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="11" cy="11" r="7" />
      <line x1="16.5" y1="16.5" x2="21" y2="21" />
      {/* 镜片内的两条横线，表示文档/知识 */}
      <line x1="8.5" y1="10" x2="13.5" y2="10" strokeWidth="1.6" strokeOpacity="0.6" />
      <line x1="8.5" y1="13" x2="11.5" y2="13" strokeWidth="1.6" strokeOpacity="0.6" />
      <defs>
        <clipPath id={clipId}>
          <circle cx="11" cy="11" r="6.5" />
        </clipPath>
      </defs>
    </svg>
  )
}
