import { Fragment, useMemo, type ReactNode } from 'react'
import { parseMarkdown, type Block, type Inline, type ListItem } from '../markdown'

/**
 * 把回答按 Markdown 渲染。
 *
 * ## 两个设计上的选择
 *
 * 1. **引用角标不在本组件里实现**，而是通过 `renderCitation` 回调交给调用方。
 *    这样「Markdown 怎么渲染」与「引用长什么样、点了做什么」互不牵扯 ——
 *    否则要么把 ChatPanel 的逻辑塞进来，要么渲染完再去 HTML 里做替换（脆弱）。
 *
 * 2. **全程构造 React 元素，不用 `dangerouslySetInnerHTML`**。
 *    回答里可能出现任何字符，交给 React 转义即可；一旦走 HTML 字符串路线，
 *    就得自己保证转义正确，XSS 风险凭空出现。
 */

export type RenderCitation = (index: number, key: string) => ReactNode

function renderInlines(
  nodes: Inline[],
  renderCitation: RenderCitation,
  prefix: string,
): ReactNode[] {
  return nodes.map((node, i) => {
    const key = `${prefix}${i}`
    switch (node.type) {
      case 'text':
        // Fragment 而不是 span：避免在 <p>/<li> 里多套一层元素影响行内排版
        return <Fragment key={key}>{node.text}</Fragment>
      case 'br':
        return <br key={key} />
      case 'strong':
        return <strong key={key}>{renderInlines(node.children, renderCitation, `${key}-`)}</strong>
      case 'em':
        return <em key={key}>{renderInlines(node.children, renderCitation, `${key}-`)}</em>
      case 'code':
        return (
          <code className="md-inline-code" key={key}>
            {node.text}
          </code>
        )
      case 'link':
        return (
          <a key={key} href={node.href} target="_blank" rel="noopener noreferrer">
            {renderInlines(node.children, renderCitation, `${key}-`)}
          </a>
        )
      case 'citation':
        return <Fragment key={key}>{renderCitation(node.index, key)}</Fragment>
      default:
        return null
    }
  })
}

function renderListItem(item: ListItem, renderCitation: RenderCitation, key: string): ReactNode {
  return (
    <li key={key}>
      {renderInlines(item.inlines, renderCitation, `${key}-`)}
      {item.children.map((child, ci) => renderBlock(child, renderCitation, `${key}-c${ci}`))}
    </li>
  )
}

function renderBlock(block: Block, renderCitation: RenderCitation, key: string): ReactNode {
  switch (block.type) {
    case 'heading': {
      const Tag = (`h${Math.min(block.level + 2, 6)}`) as 'h3'
      return (
        <Tag className="md-heading" key={key}>
          {renderInlines(block.inlines, renderCitation, `${key}-`)}
        </Tag>
      )
    }
    case 'paragraph':
      return (
        <p className="md-p" key={key}>
          {renderInlines(block.inlines, renderCitation, `${key}-`)}
        </p>
      )
    case 'list': {
      const items = block.items.map((item, i) => renderListItem(item, renderCitation, `${key}-i${i}`))
      return block.ordered ? (
        <ol className="md-list" key={key} start={block.start !== 1 ? block.start : undefined}>
          {items}
        </ol>
      ) : (
        <ul className="md-list" key={key}>
          {items}
        </ul>
      )
    }
    case 'code':
      return (
        <pre className="md-pre" key={key}>
          <code>{block.text}</code>
        </pre>
      )
    case 'quote':
      return (
        <blockquote className="md-quote" key={key}>
          {renderInlines(block.inlines, renderCitation, `${key}-`)}
        </blockquote>
      )
    case 'hr':
      return <hr className="md-hr" key={key} />
    case 'table':
      return (
        <div className="md-table-wrap" key={key}>
          <table className="md-table">
            <thead>
              <tr>
                {block.header.map((cell, i) => (
                  <th key={i}>{renderInlines(parseInlineSafe(cell), renderCitation, `${key}-h${i}-`)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, ri) => (
                <tr key={ri}>
                  {row.map((cell, ci) => (
                    <td key={ci}>
                      {renderInlines(parseInlineSafe(cell), renderCitation, `${key}-r${ri}c${ci}-`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
    default:
      return null
  }
}

/**
 * 表格单元格在解析阶段只存了原始字符串，这里补一次行内解析。
 *
 * 为什么不当场解析：块级解析器产出的是可序列化的纯数据（便于单测断言），
 * 而 Inline 里嵌 Inline 会让断言写起来很啰嗦。表格单元格是最次要的语法，
 * 在渲染时补解析代价可以忽略。
 */
function parseInlineSafe(cell: string): Inline[] {
  // 复用块解析器的行内能力：套一层再取回来
  const blocks = parseMarkdown(cell)
  const first = blocks[0]
  if (first && first.type === 'paragraph') return first.inlines
  return [{ type: 'text', text: cell }]
}

interface Props {
  content: string
  renderCitation: RenderCitation
  className?: string
}

export default function Markdown({ content, renderCitation, className }: Props) {
  // 流式输出时每个 delta 都会重渲染；按内容缓存解析结果，
  // 避免无关的状态变化（比如滚动）也重新解析一遍。
  const blocks = useMemo(() => parseMarkdown(content), [content])

  return (
    <div className={className ? `md-body ${className}` : 'md-body'}>
      {blocks.map((block, i) => renderBlock(block, renderCitation, `b${i}`))}
    </div>
  )
}
