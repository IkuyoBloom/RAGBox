import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as api from './api'
import ChatPanel from './components/ChatPanel'
import ChunkModal from './components/ChunkModal'
import SettingsPanel from './components/SettingsPanel'
import Sidebar from './components/Sidebar'
import TopBar from './components/TopBar'
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

/** 界面上的一条消息。比后端模型多了「正在流式输出」这类瞬态字段。 */
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

/** 界面主题。深色只切换 <html data-theme>，具体配色全在 CSS 变量里。 */
type Theme = 'light' | 'dark'

const THEME_STORAGE_KEY = 'rag-theme'

/**
 * URL 上强制指定的主题（`?theme=dark` 或 `#theme=dark`）。
 *
 * 这个入口主要是为了**可验证**：无头浏览器截图时没法预置 localStorage，
 * 而应用的默认主题跟随系统偏好，无头模式又总报浅色 ——
 * 没有这个入口就没法用截图确认深色模式到底长什么样。
 * 顺带也方便把「某个主题下的样子」作为链接分享出去。
 */
function readForcedTheme(): Theme | null {
  if (typeof window === 'undefined') return null
  const fromSearch = new URLSearchParams(window.location.search).get('theme')
  const fromHash = new URLSearchParams(window.location.hash.replace(/^#/, '')).get('theme')
  const value = fromSearch ?? fromHash
  return value === 'light' || value === 'dark' ? value : null
}

const FORCED_THEME = readForcedTheme()

/**
 * 决定初始主题。
 *
 * 优先级：URL 强制 > 用户显式选过 > 系统偏好 > 浅色。
 * 跟随系统是重要的默认值 —— 用户在系统里开了深色模式，
 * 打开应用却是一片白，那是最刺眼的一种不体贴。
 */
function readInitialTheme(): Theme {
  if (FORCED_THEME) return FORCED_THEME
  if (typeof window === 'undefined') return 'light'
  try {
    const saved = window.localStorage.getItem(THEME_STORAGE_KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    // 隐私模式下 localStorage 可能抛异常，忽略即可
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

  // ---------------- 初始加载 ----------------
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
      // 首次拿到设置时，把配置里的当前通道同步到界面选择器；
      // 之后不再覆盖，否则用户手动切换会被刷新冲掉。
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

  // 主题落到 <html data-theme> 上：CSS 里所有深色规则都挂在这个属性下，
  // 组件内部一条深色判断都不用写。同时记住用户的选择。
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    // URL 强制指定的主题不写回本地 —— 那只是一次性的覆盖，
    // 不该污染用户自己的偏好设置
    if (FORCED_THEME) return
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme)
    } catch {
      // 存不了就存不了，不影响使用
    }
  }, [theme])

  // 自动选中知识库。
  //
  // 不这么做的话，打开页面看到的是「请先在左侧选择或创建一个知识库」，
  // 而左侧明明就摆着一个库 —— 用户得先点一下才能提问，纯属多余的一步，
  // 而且空状态里连示例问题都不显示（它们只在选中库之后才渲染）。
  //
  // 第二个条件同样重要：当前选中的库被删掉时要兜回来，
  // 否则界面会停在一个已经不存在的库上，提问必然报错。
  useEffect(() => {
    if (kbs.length === 0) return
    if (selectedKbId && kbs.some((k) => k.id === selectedKbId)) return
    setSelectedKbId(kbs[0].id)
  }, [kbs, selectedKbId])

  // ---------------- 选中知识库 -> 拉详情 ----------------
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

  // 有文档还在索引时轮询，索引完成自动停 —— 比让用户手动刷新体验好得多
  const pendingCount = useMemo(
    () =>
      kbDetail?.documents.filter((d) =>
        ['pending', 'parsing', 'embedding'].includes(d.status),
      ).length ?? 0,
    [kbDetail],
  )

  /**
   * 轮询节奏分成几档，而不是固定 2 秒。
   *
   * 一次导入 118 份文档要跑好几分钟，固定 2 秒意味着上百次轮询，
   * 而每次轮询要拉**全部**文档（200+ 份、含完整 meta）。这个量足以
   * 把后端连接池占满 —— 实测就是这样卡死的。用户其实只需要一个进度感，
   * 待处理越多越可以慢一点。
   *
   * 用「档位」而不是直接用数量：档位只在跨过阈值时才变，
   * 否则每完成一份文档就会重置计时器，节奏反而乱掉。
   */
  const pollBucket = pendingCount === 0 ? 0 : pendingCount > 50 ? 3 : pendingCount > 10 ? 2 : 1
  const pollInterval = pollBucket === 3 ? 8000 : pollBucket === 2 ? 4000 : 2000

  useEffect(() => {
    if (!selectedKbId || pollBucket === 0) return

    let cancelled = false
    let timer: number | undefined

    // 关键：用「跑完再等」代替 setInterval。
    // setInterval 不看上一次是否完成，请求一慢就会重叠堆积 ——
    // 后端连接池正是被这些堆积的请求打满的（QueuePool limit reached）。
    const tick = async () => {
      try {
        await Promise.all([refreshKbDetail(selectedKbId), refreshKbs()])
      } catch {
        // 单次轮询失败不该中断循环：后端可能在忙，下一轮会自己恢复
      }
      if (!cancelled) timer = window.setTimeout(() => void tick(), pollInterval)
    }
    timer = window.setTimeout(() => void tick(), pollInterval)

    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
    }
  }, [selectedKbId, pollBucket, pollInterval, refreshKbDetail, refreshKbs])

  // ---------------- 发送提问 ----------------
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

  /**
   * 向量模型现状，供侧边栏显示「当前设置用什么模型」与「每个库实际用什么模型」。
   *
   * `loaded` 用来区分「Ollama 里确实没装这个模型」和「压根没连上 Ollama」——
   * 少了这个标志，Ollama 一停，每个知识库都会被误标成「模型未安装」。
   */
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
      <TopBar
        health={health}
        provider={provider}
        providerOptions={providerOptions}
        onProviderChange={setProvider}
        mode={mode}
        onModeChange={setMode}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
        onOpenSettings={() => setShowSettings(true)}
        onRefresh={() => {
          void refreshHealth()
          void refreshKbs()
          if (selectedKbId) void refreshKbDetail(selectedKbId)
        }}
        onNewConversation={newConversation}
        hasConversation={messages.length > 0}
      />

      <div className="app-body">
        <Sidebar
          kbs={kbs}
          selectedKbId={selectedKbId}
          kbDetail={kbDetail}
          embedInfo={embedInfo}
          onSelect={(id) => {
            setSelectedKbId(id)
            newConversation()
          }}
          onRefreshKbs={refreshKbs}
          onRefreshDetail={() => selectedKbId && refreshKbDetail(selectedKbId)}
          notify={notify}
        />

        <ChatPanel
          kbName={kbDetail?.name ?? null}
          messages={messages}
          streaming={streaming}
          onSend={send}
          onStop={stop}
          onOpenChunk={setOpenChunkId}
        />
      </div>

      {showSettings && settings && (
        <SettingsPanel
          settings={settings}
          onClose={() => setShowSettings(false)}
          onSaved={(s) => {
            setSettings(s)
            notify('ok', '设置已保存（立即生效，无需重启）')
          }}
          notify={notify}
        />
      )}

      {openChunkId && <ChunkModal chunkId={openChunkId} onClose={() => setOpenChunkId(null)} />}

      {toast && <div className={`toast toast-${toast.kind}`}>{toast.text}</div>}
    </div>
  )
}
