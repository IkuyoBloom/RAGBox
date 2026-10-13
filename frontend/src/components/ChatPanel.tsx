import { useEffect, useMemo, useRef, useState } from 'react'
import type { UIMessage } from '../App'
import type { SearchMode } from '../types'
import AssistantAvatar from './AssistantAvatar'
import CitationList from './CitationList'
import Markdown, { type RenderCitation } from './Markdown'

interface Props {
  kbName: string | null
  docCount: number
  chunkCount: number
  messages: UIMessage[]
  streaming: boolean
  provider: string
  providerOptions: string[]
  onProviderChange: (v: string) => void
  mode: SearchMode
  onModeChange: (v: SearchMode) => void
  docsOpen: boolean
  onToggleDocs: () => void
  onNewConversation: () => void
  onSend: (text: string) => void
  onStop: () => void
  onOpenChunk: (chunkId: string) => void
}

const SUGGESTIONS = [
  '这些资料主要讲了什么？',
  '帮我梳理一下其中的关键信息',
  '有哪些值得注意的细节？',
]

const PROVIDER_LABELS: Record<string, string> = {
  ollama: '本地 Ollama',
  openai: '云端 API',
}

const MODE_LABELS: Record<SearchMode, string> = {
  hybrid: '混合检索',
  dense_only: '仅向量',
  sparse_only: '仅 BM25',
}

function makeCitationRenderer(
  citations: UIMessage['citations'],
  onOpenChunk: (id: string) => void,
): RenderCitation {
  const byIndex = new Map(citations.map((c) => [c.index, c]))

  return (index, key) => {
    const citation = byIndex.get(index)
    if (!citation) {
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
    if (!props.kbName) return '请先在左侧选择或创建一个知识库…'
    return `向「${props.kbName}」提问…`
  }, [props.kbName])

  return (
    <main className="chat">
      {/* 工作区顶栏：当前知识库上下文 + 新对话 + 右侧文档栏开关 */}
      <header className="chat-header">
        <div className="chat-header-left">
          <h1 className="chat-kb-title">{props.kbName ?? '未选择知识库'}</h1>
          {props.kbName && (
            <span className="chat-kb-meta">
              {props.docCount} 份文档 · {props.chunkCount} 个切片
            </span>
          )}
        </div>

        <div className="chat-header-right">
          {messages.length > 0 && (
            <button className="btn ghost tiny" onClick={props.onNewConversation}>
              新对话
            </button>
          )}
          {props.kbName && (
            <button
              className={`btn tiny ${props.docsOpen ? 'active-toggle' : 'ghost'}`}
              onClick={props.onToggleDocs}
              title={props.docsOpen ? '收起右侧文档栏' : '展开右侧文档栏'}
            >
              文档目录 ({props.docCount})
            </button>
          )}
        </div>
      </header>

      {/* 消息滚动区 */}
      <div className="chat-scroll" ref={listRef} onScroll={onScroll}>
        {empty ? (
          <div className="chat-empty">
            <div className="empty-title">
              {props.kbName ? `有什么关于「${props.kbName}」的问题？` : '选择或新建一个知识库'}
            </div>
            <p className="empty-desc">
              {props.kbName
                ? '所有回答均基于右侧库内文档检索生成，并附带原文引用角标。'
                : '在左侧创建知识库并在右侧上传文档，索引完成后即可开始提问。'}
            </p>

            {props.kbName && (
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
            )}
          </div>
        ) : (
          <div className="message-list">
            {messages.map((m) => (
              <div key={m.id} className={`message message-${m.role}`}>
                {m.role === 'assistant' && <AssistantAvatar className="avatar-ai" />}
                <div className="bubble-wrap">
                  {m.role === 'assistant' && m.rewritten && (
                    <div className="rewrite-note" title="系统自动把追问改写成了可独立检索的问题">
                      检索词改写：<code>{m.rewritten}</code>
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
                        <div className="answer-text user-text">{m.content}</div>
                      )
                    ) : m.streaming ? (
                      <div className="thinking">
                        <span className="spinner" />
                        正在检索文档并生成回答…
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
                      <span>向量 {m.trace.dense_hits}</span>
                      <span>BM25 {m.trace.sparse_hits}</span>
                      <span>融合 {m.trace.fused_hits}</span>
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

      {/* 一体化输入卡片：输入框与底部工具栏合一，减少割裂线条 */}
      <div className="composer">
        <div className="composer-card">
          <textarea
            value={draft}
            placeholder={placeholder}
            disabled={!props.kbName}
            rows={2}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
          />
          <div className="composer-toolbar">
            <div className="composer-Selectors">
              <label className="composer-pill-select" title="切换生成通道">
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

              <label className="composer-pill-select" title="切换检索模式">
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
            </div>

            <div className="composer-right">
              <span className="composer-shortcut">
                <kbd>Enter</kbd> 发送 · <kbd>Shift+Enter</kbd> 换行
              </span>
              {streaming ? (
                <button className="btn danger tiny" onClick={props.onStop}>
                  停止
                </button>
              ) : (
                <button
                  className="btn primary tiny"
                  disabled={!draft.trim() || !props.kbName}
                  onClick={submit}
                >
                  发送
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}
