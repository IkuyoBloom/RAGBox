import { useEffect, useMemo, useRef, useState } from 'react'
import type { UIMessage } from '../App'
import AssistantAvatar from './AssistantAvatar'
import CitationList from './CitationList'
import Logo from './Logo'
import Markdown, { type RenderCitation } from './Markdown'

interface Props {
  kbName: string | null
  messages: UIMessage[]
  streaming: boolean
  onSend: (text: string) => void
  onStop: () => void
  onOpenChunk: (chunkId: string) => void
}

// 示例问题刻意写得**与语料无关**。
// 原先那三条（「核心贡献」「实验数据集」「方法局限性」）是照着论文写的，
// 换成剧情文本、说明文档、聊天记录就全都问不通 —— 而示例问题的意义
// 恰恰是让用户一眼看懂「这个系统能拿我的资料做什么」。
const SUGGESTIONS = [
  '这些资料主要讲了什么？',
  '帮我梳理一下其中的关键信息',
  '有哪些值得注意的细节？',
]

/**
 * 构造引用角标的渲染函数，交给 Markdown 组件回调。
 *
 * 为什么做成回调而不是「渲染完再去替换」：回答的 Markdown 结构里，
 * `[n]` 可能出现在加粗里、列表项里、表格单元格里。等 HTML 出来再去做
 * 字符串替换，既要处理标签边界又容易破坏结构。在行内解析那一层
 * 顺带把引用识别出来，是最稳的做法。
 */
function makeCitationRenderer(
  citations: UIMessage['citations'],
  onOpenChunk: (id: string) => void,
): RenderCitation {
  const byIndex = new Map(citations.map((c) => [c.index, c]))

  return (index, key) => {
    const citation = byIndex.get(index)
    if (!citation) {
      // 模型写了一个不存在的编号 —— 直接标红，这是「编造来源」的可见信号
      return (
        <span
          key={key}
          className="cite-ref invalid"
          title="该编号没有对应的检索片段，可能是模型编造的来源"
        >
          [{index}]
        </span>
      )
    }
    return (
      <button
        key={key}
        className="cite-ref"
        title={`${citation.filename}${citation.page ? ` 第 ${citation.page} 页` : ''}\n\n${citation.preview}`}
        onClick={() => onOpenChunk(citation.chunk_id)}
      >
        [{index}]
      </button>
    )
  }
}

export default function ChatPanel(props: Props) {
  const { messages, streaming } = props
  const [draft, setDraft] = useState('')
  const listRef = useRef<HTMLDivElement>(null)
  const stickToBottom = useRef(true)

  // 用户往上翻看历史时不要强行拽回底部 —— 流式输出时这一步很影响体验
  useEffect(() => {
    const el = listRef.current
    if (!el || !stickToBottom.current) return
    el.scrollTop = el.scrollHeight
  }, [messages])

  const onScroll = () => {
    const el = listRef.current
    if (!el) return
    stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }

  const submit = () => {
    const text = draft.trim()
    if (!text || streaming) return
    stickToBottom.current = true
    props.onSend(text)
    setDraft('')
  }

  const empty = messages.length === 0
  const placeholder = useMemo(() => {
    if (!props.kbName) return '请先在左侧选择或创建一个知识库'
    return `向「${props.kbName}」提问，Enter 发送，Shift+Enter 换行`
  }, [props.kbName])

  return (
    <main className="chat">
      <div className="chat-scroll" ref={listRef} onScroll={onScroll}>
        {empty ? (
          <div className="chat-empty">
            {/* 品牌标记：空状态原本只有两行字，四周是巨大空白，看着像没加载出来 */}
            <Logo size={58} className="empty-mark" />
            <div className="empty-title">
              {props.kbName ? `开始向「${props.kbName}」提问` : '欢迎使用本地 RAG 知识库问答系统'}
            </div>
            <p className="empty-desc">
              系统会先在你的文档里做混合检索（BM25 + 向量 + RRF 融合），
              再让大模型<b>仅依据检索到的片段</b>作答，并标注 [n] 来源。
            </p>
            {props.kbName ? (
              <div className="suggestions-wrap">
                <div className="suggestions-label">试试这样问</div>
                <div className="suggestions">
                  {SUGGESTIONS.map((s) => (
                    <button key={s} className="suggestion" onClick={() => props.onSend(s)}>
                      <span className="suggestion-text">{s}</span>
                      <span className="suggestion-arrow" aria-hidden="true">
                        →
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="empty-hint">
                先在左侧新建知识库、上传文档并等索引完成，就可以开始提问了。
              </div>
            )}
          </div>
        ) : (
          <div className="message-list">
            {messages.map((m) => (
              <div key={m.id} className={`message message-${m.role}`}>
                {m.role === 'user' ? (
                  <div className="avatar">我</div>
                ) : (
                  <AssistantAvatar size={30} className="avatar-ai" />
                )}
                <div className="bubble-wrap">
                  {m.role === 'assistant' && m.rewritten && (
                    <div className="rewrite-note" title="系统自动把追问改写成了可以独立检索的问题">
                      查询改写：<code>{m.rewritten}</code>
                    </div>
                  )}

                  <div className="bubble">
                    {m.content ? (
                      m.role === 'assistant' ? (
                        <Markdown
                          content={m.content}
                          renderCitation={makeCitationRenderer(m.citations, props.onOpenChunk)}
                        />
                      ) : (
                        // 用户自己输入的按纯文本显示：没人希望自己打的字被当成 Markdown 解释
                        <div className="answer-text user-text">{m.content}</div>
                      )
                    ) : m.streaming ? (
                      <div className="thinking">
                        <span className="spinner" />
                        正在检索并生成…
                      </div>
                    ) : null}
                  </div>

                  {m.error && <div className="bubble-error">⚠ {m.error}</div>}

                  {m.role === 'assistant' && m.citations.length > 0 && (
                    <CitationList citations={m.citations} onOpenChunk={props.onOpenChunk} />
                  )}

                  {m.role === 'assistant' && m.trace && (
                    <div className="trace">
                      <span>检索 {m.trace.elapsed_ms} ms</span>
                      <span>向量 {m.trace.dense_hits} 条</span>
                      <span>BM25 {m.trace.sparse_hits} 条</span>
                      <span>融合 {m.trace.fused_hits} 条</span>
                      {m.latency_ms != null && <span>生成 {(m.latency_ms / 1000).toFixed(1)} s</span>}
                      {m.provider && (
                        <span>
                          {m.provider}
                          {m.model ? ` · ${m.model}` : ''}
                        </span>
                      )}
                      {m.completion_tokens != null && <span>{m.completion_tokens} tokens</span>}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="composer">
        {/* 内容列与消息列同宽：输入框铺满整个窗口宽度、消息却只有 920px
            的话，两者对不齐，看着像两个不相干的区域 */}
        <div className="composer-row">
          <textarea
            value={draft}
            placeholder={placeholder}
            disabled={!props.kbName}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
          />
          <div className="composer-actions">
            {streaming ? (
              <button className="btn danger" onClick={props.onStop}>
                停止生成
              </button>
            ) : (
              <button
                className="btn primary"
                disabled={!draft.trim() || !props.kbName}
                onClick={submit}
              >
                发送
              </button>
            )}
          </div>
        </div>
        {/* 快捷键原先没有任何提示，只能靠猜 —— 写出来 */}
        <div className="composer-hint">
          <span>
            {props.kbName ? (
              <>
                <kbd>Enter</kbd> 发送 · <kbd>Shift</kbd> + <kbd>Enter</kbd> 换行
              </>
            ) : (
              '先在左侧选择或创建知识库'
            )}
          </span>
          {props.kbName && <span className="composer-hint-kb">{props.kbName}</span>}
        </div>
      </div>
    </main>
  )
}
