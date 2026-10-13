import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as api from './api'
import ChatPanel from './components/ChatPanel'
import ChunkModal from './components/ChunkModal'
import DocumentPanel from './components/DocumentPanel'
import SettingsPanel from './components/SettingsPanel'
import Sidebar from './components/Sidebar'
import type { EmbedInfo } from './embedModel'
import type {
  ChatEvent,
  Citation,
  Health,
  KnowledgeBase,
  KnowledgeBaseDetail,
  RetrievalTrace,
  SearchMode,
  SettingsOut,
} from './types'

/** 界面上的一条消息。 */
export interface UIMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  streaming?: boolean
  error?: string
  rewritten?: string
  trace?: RetrievalTrace
  provider?: string | null
  model?: string | null
  latency_ms?: number | null
  retrieval_ms?: number | null
  prompt_tokens?: number | null
  completion_tokens?: number | null
}

export interface Toast {
  kind: 'ok' | 'err'
  text: string
}

type Theme = 'light' | 'dark'

const THEME_STORAGE_KEY = 'rag-theme'

function readForcedTheme(): Theme | null {
  if (typeof window === 'undefined') return null
  const fromSearch = new URLSearchParams(window.location.search).get('theme')
  const fromHash = new URLSearchParams(window.location.hash.replace(/^#/, '')).get('theme')
  const value = fromSearch ?? fromHash
  return value === 'light' || value === 'dark' ? value : null
}

const FORCED_THEME = readForcedTheme()

function readInitialTheme(): Theme {
  if (FORCED_THEME) return FORCED_THEME
  if (typeof window === 'undefined') return 'light'
  try {
    const saved = window.localStorage.getItem(THEME_STORAGE_KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    // 忽略隐私模式异常
  }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [kbs, setKbs] = useState<KnowledgeBase[]>([])
  const [kbDetail, setKbDetail] = useState<KnowledgeBaseDetail | null>(null)
  const [selectedKbId, setSelectedKbId] = useState<string | null>(null)
  const [settings, setSettings] = useState<SettingsOut | null>(null)
  const [showSettings, setShowSettings] = useState(false)
  const [docsOpen, setDocsOpen] = useState(true)
  const [theme, setTheme] = useState<Theme>(readInitialTheme)

  const [provider, setProvider] = useState<string>('')
  const [mode, setMode] = useState<SearchMode>('hybrid')

  const [messages, setMessages] = useState<UIMessage[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [streaming, setStreaming] = useState(false)

  const [toast, setToast] = useState<Toast | null>(null)
  const [openChunkId, setOpenChunkId] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const notify = useCallback((kind: Toast['kind'], text: string) => {
    setToast({ kind, text })
    window.setTimeout(() => setToast(null), kind === 'err' ? 8000 : 3500)
  }, [])

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api.getHealth())
    } catch (e) {
      notify('err', `无法连接后端：${(e as Error).message}`)
    }
  }, [notify])

  const refreshKbs = useCallback(async () => {
    try {
      setKbs(await api.listKbs())
    } catch (e) {
      notify('err', `加载知识库失败：${(e as Error).message}`)
    }
  }, [notify])

  const refreshSettings = useCallback(async () => {
    try {
      const s = await api.getSettings()
      setSettings(s)
      const active = s.items.find((i) => i.key === 'llm_provider')?.value
      setProvider((prev) => prev || (typeof active === 'string' && active ? active : 'ollama'))
    } catch (e) {
      notify('err', `加载设置失败：${(e as Error).message}`)
    }
  }, [notify])

  useEffect(() => {
    void refreshHealth()
    void refreshKbs()
    void refreshSettings()
  }, [refreshHealth, refreshKbs, refreshSettings])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    if (FORCED_THEME) return
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme)
    } catch {
      // 忽略
    }
  }, [theme])

  useEffect(() => {
    if (kbs.length === 0) return
    if (selectedKbId && kbs.some((k) => k.id === selectedKbId)) return
    setSelectedKbId(kbs[0].id)
  }, [kbs, selectedKbId])

  const refreshKbDetail = useCallback(
    async (kbId: string) => {
      try {
        setKbDetail(await api.getKb(kbId))
      } catch (e) {
        notify('err', `加载知识库详情失败：${(e as Error).message}`)
      }
    },
    [notify],
  )

  useEffect(() => {
    if (!selectedKbId) {
      setKbDetail(null)
      return
    }
    void refreshKbDetail(selectedKbId)
  }, [selectedKbId, refreshKbDetail])

  const pendingCount = useMemo(
    () =>
      kbDetail?.documents.filter((d) =>
        ['pending', 'parsing', 'embedding'].includes(d.status),
      ).length ?? 0,
    [kbDetail],
  )

  const pollBucket = pendingCount === 0 ? 0 : pendingCount > 50 ? 3 : pendingCount > 10 ? 2 : 1
  const pollInterval = pollBucket === 3 ? 8000 : pollBucket === 2 ? 4000 : 2000

  useEffect(() => {
    if (!selectedKbId || pollBucket === 0) return

    let cancelled = false
    let timer: number | undefined

    const tick = async () => {
      try {
        await Promise.all([refreshKbDetail(selectedKbId), refreshKbs()])
      } catch {
        // 单次轮询失败不中断循环
      }
      if (!cancelled) timer = window.setTimeout(() => void tick(), pollInterval)
    }
    timer = window.setTimeout(() => void tick(), pollInterval)

    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
    }
  }, [selectedKbId, pollBucket, pollInterval, refreshKbDetail, refreshKbs])

  const send = useCallback(
    async (text: string) => {
      const question = text.trim()
      if (!question || streaming) return
      if (!selectedKbId) {
        notify('err', '请先在左侧选择一个知识库')
        return
      }

      const userMsg: UIMessage = {
        id: `local-user-${Date.now()}`,
        role: 'user',
        content: question,
        citations: [],
      }
      const assistantId = `local-assistant-${Date.now()}`
      const assistantMsg: UIMessage = {
        id: assistantId,
        role: 'assistant',
        content: '',
        citations: [],
        streaming: true,
      }
      setMessages((prev) => [...prev, userMsg, assistantMsg])
      setStreaming(true)

      const controller = new AbortController()
      abortRef.current = controller

      const patch = (fn: (m: UIMessage) => UIMessage) => {
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)))
      }

      try {
        const stream = api.streamChat(
          {
            message: question,
            kb_id: selectedKbId,
            conversation_id: conversationId,
            provider: provider || null,
            mode,
          },
          controller.signal,
        )

        for await (const ev of stream as AsyncGenerator<ChatEvent>) {
          switch (ev.type) {
            case 'start':
              setConversationId(ev.conversation_id)
              break
            case 'rewrite':
              patch((m) => ({ ...m, rewritten: ev.query }))
              break
            case 'citations':
              patch((m) => ({ ...m, citations: ev.citations }))
              break
            case 'trace':
              patch((m) => ({ ...m, trace: ev }))
              break
            case 'delta':
              patch((m) => ({ ...m, content: m.content + ev.text }))
              break
            case 'done':
              patch((m) => ({
                ...m,
                streaming: false,
                provider: ev.provider,
                model: ev.model,
                latency_ms: ev.latency_ms,
                retrieval_ms: ev.retrieval_ms,
                prompt_tokens: ev.prompt_tokens,
                completion_tokens: ev.completion_tokens,
              }))
              break
            case 'error':
              patch((m) => ({
                ...m,
                streaming: false,
                error: ev.message,
                content: m.content || ev.partial || '',
              }))
              notify('err', ev.message)
              break
            case 'eos':
              patch((m) => ({ ...m, streaming: false }))
              break
          }
        }
      } catch (e) {
        const err = e as Error
        if (err.name === 'AbortError') {
          patch((m) => ({ ...m, streaming: false, content: m.content + '\n\n（已中止）' }))
        } else {
          patch((m) => ({ ...m, streaming: false, error: err.message }))
          notify('err', err.message)
        }
      } finally {
        setStreaming(false)
        abortRef.current = null
      }
    },
    [conversationId, mode, notify, provider, selectedKbId, streaming],
  )

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const newConversation = useCallback(() => {
    setMessages([])
    setConversationId(null)
  }, [])

  const providerOptions = useMemo(() => {
    const items = settings?.items.find((i) => i.key === 'llm_provider')
    return items?.choices ?? ['ollama', 'openai']
  }, [settings])

  const embedInfo = useMemo<EmbedInfo>(() => {
    const item = settings?.items.find((i) => i.key === 'ollama_embed_model')
    return {
      current: typeof item?.value === 'string' ? item.value : '',
      installed: item?.choices ?? [],
      loaded: item?.choices_loaded ?? false,
    }
  }, [settings])

  return (
    <div className="app">
      <div className={`app-workspace ${docsOpen && selectedKbId ? 'with-docs' : ''}`}>
        <Sidebar
          kbs={kbs}
          selectedKbId={selectedKbId}
          embedInfo={embedInfo}
          health={health}
          provider={provider}
          theme={theme}
          onSelect={(id) => {
            setSelectedKbId(id)
            newConversation()
          }}
          onRefreshKbs={refreshKbs}
          onRefreshAll={() => {
            void refreshHealth()
            void refreshKbs()
            if (selectedKbId) void refreshKbDetail(selectedKbId)
          }}
          onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
          onOpenSettings={() => setShowSettings(true)}
          notify={notify}
        />

        <ChatPanel
          kbName={kbDetail?.name ?? null}
          docCount={kbDetail?.documents.length ?? 0}
          chunkCount={kbDetail?.chunk_count ?? 0}
          messages={messages}
          streaming={streaming}
          provider={provider}
          providerOptions={providerOptions}
          onProviderChange={setProvider}
          mode={mode}
          onModeChange={setMode}
          docsOpen={docsOpen}
          onToggleDocs={() => setDocsOpen((v) => !v)}
          onNewConversation={newConversation}
          onSend={send}
          onStop={stop}
          onOpenChunk={setOpenChunkId}
        />

        {docsOpen && selectedKbId && (
          <DocumentPanel
            kbDetail={kbDetail}
            selectedKbId={selectedKbId}
            onRefreshKbs={refreshKbs}
            onRefreshDetail={() => selectedKbId && refreshKbDetail(selectedKbId)}
            notify={notify}
            onClose={() => setDocsOpen(false)}
          />
        )}
      </div>

      {showSettings && settings && (
        <SettingsPanel
          settings={settings}
          onClose={() => setShowSettings(false)}
          onSaved={(s) => {
            setSettings(s)
            const active = s.items.find((i) => i.key === 'llm_provider')?.value
            if (typeof active === 'string' && active) setProvider(active)
            notify('ok', '设置已保存（立即生效）')
          }}
          notify={notify}
        />
      )}

      {openChunkId && <ChunkModal chunkId={openChunkId} onClose={() => setOpenChunkId(null)} />}

      {toast && <div className={`toast toast-${toast.kind}`}>{toast.text}</div>}
    </div>
  )
}
