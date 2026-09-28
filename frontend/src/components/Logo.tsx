import { useId } from 'react'

/**
 * 项目 Logo（纯内联 SVG）。
 *
 * ## 图形含义
 *
 * 一个**放大镜**，镜片里是几行**知识块** —— 合起来就是「在知识库里检索」。
 * 这比抽象图形好在能一眼读懂系统是干什么的，而且在小尺寸下依然成立：
 * 整个图形只有「圆环 + 两条横线 + 一根手柄」四笔，缩到 24px 也不会糊。
 *
 * ## 为什么内联 SVG 而不是图片文件
 *
 * * 矢量：任何尺寸都清晰，高分屏不糊；
 * * 跟着主题走：渐变颜色直接写在这里，换配色不用重新导出图片；
 * * 不增加网络请求，也不会有「图片没加载出来先闪一下」的问题。
 *
 * 渐变 id 用 useId 生成：同一页面会渲染多个 Logo（顶栏一个小、空状态一个大），
 * 写死 id 会撞车 —— 浏览器只认第一个，第二个就可能拿不到填充。
 */
interface Props {
  size?: number
  className?: string
}

export default function Logo({ size = 34, className }: Props) {
  // useId 的值形如 ":r0:"，冒号放进 url(#...) 里不稳妥，去掉
  const gradientId = `rag-logo-${useId().replace(/:/g, '')}`

  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 32 32"
      role="img"
      aria-label="本地 RAG 知识库问答系统"
    >
      <defs>
        {/* userSpaceOnUse：渐变绑定在 32×32 的坐标系上。
            默认的 objectBoundingBox 会把渐变重新映射到每个元素的包围盒，
            同一套颜色贴到不同大小的图形上就会偏色。 */}
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

      {/* 镜片。线宽刻意加粗：16~20px 时细圆环会收成一团，看不出是圆 */}
      <circle cx="14" cy="13.6" r="6.2" fill="none" stroke="#fff" strokeWidth="2.4" />

      {/* 镜片里的知识块：第二行略短、略淡，暗示「一堆长短不一的切片」 */}
      <rect x="10.5" y="10.7" width="7" height="2" rx="1" fill="#fff" />
      <rect x="10.5" y="14.5" width="4.6" height="2" rx="1" fill="#fff" opacity="0.72" />

      {/* 手柄 */}
      <line
        x1="18.5"
        y1="18.2"
        x2="23.3"
        y2="23"
        stroke="#fff"
        strokeWidth="2.7"
        strokeLinecap="round"
      />
    </svg>
  )
}
