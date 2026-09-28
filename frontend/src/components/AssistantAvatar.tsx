import { useId } from 'react'

/**
 * AI 助手头像（纯内联 SVG）。
 *
 * ## 图形含义
 *
 * 一颗**四角星（闪光）**。
 *
 * 选它是为了和 Logo 形成对照：Logo 是放大镜 + 知识块，说的是「检索」；
 * 这颗星说的是「生成」。两者合起来正好是 RAG 的两半 ——
 * 从资料里找依据，再据它生成回答。
 *
 * 另一个好处是小尺寸可辨：整个图形只有「底板 + 一个路径」两个要素，
 * 30px 下仍然一眼认得出，而原来的「AI」两个字在这个尺寸只有 11.5px，
 * 又小又挤。
 *
 * ## 为什么不复用 Logo 的渐变 id
 *
 * useId 保证每个实例的渐变 id 独立 —— 一条消息列表里会有几十个头像，
 * 写死 id 会让除第一个以外的全部拿不到填充（浏览器只认第一个定义）。
 * 这和 Logo 那边是同一个坑，理由也一致。
 */
interface Props {
  size?: number
  className?: string
}

export default function AssistantAvatar({ size = 30, className }: Props) {
  const gradientId = `rag-avatar-${useId().replace(/:/g, '')}`

  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 32 32"
      role="img"
      aria-label="AI 助手"
    >
      <defs>
        {/* userSpaceOnUse：把渐变绑在 32×32 坐标系上。
            默认的 objectBoundingBox 会按每个元素的包围盒重映射，
            同一个渐变贴到不同图形上颜色就变了。 */}
        <linearGradient
          id={gradientId}
          gradientUnits="userSpaceOnUse"
          x1="0"
          y1="0"
          x2="32"
          y2="32"
        >
          <stop offset="0%" stopColor="#5b93f8" />
          <stop offset="100%" stopColor="#1f56c4" />
        </linearGradient>
      </defs>

      <rect width="32" height="32" rx="9" fill={`url(#${gradientId})`} />

      {/*
        四角星。四条边都用三次贝塞尔向中心方向「收腰」，
        才能得到闪光那种内凹的弧度 —— 直线连接会变成菱形，很呆。
        四个控制点两两对称，保证四个角一模一样。
      */}
      <path
        d="M16 6
           C16.6 11.4 20.6 15.4 26 16
           C20.6 16.6 16.6 20.6 16 26
           C15.4 20.6 11.4 16.6 6 16
           C11.4 15.4 15.4 11.4 16 6 Z"
        fill="#fff"
      />
    </svg>
  )
}
