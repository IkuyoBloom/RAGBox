/**
 * 后端 API 客户端。
 *
 * 流式问答用 fetch + ReadableStream 手写 SSE 解析，而不是 EventSource ——
 * EventSource 只支持 GET，没法带 JSON 请求体，而我们的提问、会话 id、
 * 通道覆盖参数都必须在 body 里。
 */

import type {
  ChatEvent,
  Chunk,
  Conversation,
  ConversationDetail,
  Health,
  KnowledgeBase,
  KnowledgeBaseDetail,
  RagDocument,
  SearchMode,
  SettingsOut,
  UploadResult,
} from './types'

const BASE = '/api'

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: init?.body instanceof FormData ? undefined : { 'Content-Type': 'application/json' },
    ...init,
  })

  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`
    try {
      const body = await resp.json()
      if (body?.detail) {
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      }
    } catch {
      /* 响应体不是 JSON，用状态码兜底 */
    }
    throw new ApiError(detail, resp.status)
  }

  if (resp.status === 204) return undefined as T
  return (await resp.json()) as T
}

// ---------------- 系统 ----------------
export const getHealth = () => request<Health>('/health')

// ---------------- 知识库 ----------------
export const listKbs = () => request<KnowledgeBase[]>('/kb')

export const getKb = (id: string) => request<KnowledgeBaseDetail>(`/kb/${id}`)

export const createKb = (payload: {
  name: string
  description?: string
  embed_model?: string | null
  chunk_size?: number | null
  chunk_overlap?: number | null
}) => request<KnowledgeBase>('/kb', { method: 'POST', body: JSON.stringify(payload) })

export const deleteKb = (id: string) => request<void>(`/kb/${id}`, { method: 'DELETE' })

export const renameKb = (id: string, name: string) =>
  request<KnowledgeBase>(`/kb/${id}`, { method: 'PATCH', body: JSON.stringify({ name }) })

// ---------------- 文档 ----------------
/**
 * 上传文件。
 *
 * 文件夹导入时用 `webkitRelativePath` 作为上报的文件名 —— 否则一个文件夹树里
 * 的多个同名文件（比如 `项目A/README.md` 与 `项目B/README.md`）在界面上
 * 会变成两个一模一样的条目，用户根本分不清哪个是哪个。
 * 后端会做路径穿越防护，这里只管把相对结构带上。
 */
export const uploadDocuments = (kbId: string, files: File[]) => {
  const form = new FormData()
  for (const f of files) {
    const relative = (f as File & { webkitRelativePath?: string }).webkitRelativePath
    if (relative) form.append('files', f, relative)
    else form.append('files', f)
  }
  return request<UploadResult[]>(`/kb/${kbId}/documents`, { method: 'POST', body: form })
}

export const getDocument = (id: string) => request<RagDocument>(`/documents/${id}`)

export const reindexDocument = (id: string) =>
  request<RagDocument>(`/documents/${id}/reindex`, { method: 'POST' })

export const deleteDocument = (id: string) =>
  request<void>(`/documents/${id}`, { method: 'DELETE' })

export const listChunks = (docId: string, offset = 0, limit = 50) =>
  request<Chunk[]>(`/documents/${docId}/chunks?offset=${offset}&limit=${limit}`)

export const getChunk = (chunkId: string) => request<Chunk>(`/chunks/${chunkId}`)

// ---------------- 会话 ----------------
export const listConversations = (kbId?: string) =>
  request<Conversation[]>(`/conversations${kbId ? `?kb_id=${kbId}` : ''}`)

export const getConversation = (id: string) => request<ConversationDetail>(`/conversations/${id}`)

export const deleteConversation = (id: string) =>
  request<void>(`/conversations/${id}`, { method: 'DELETE' })

// ---------------- 设置 ----------------
export const getSettings = () => request<SettingsOut>('/settings')

export const updateSettings = (values: Record<string, unknown>) =>
  request<SettingsOut>('/settings', { method: 'PUT', body: JSON.stringify({ values }) })

export const resetSettings = (keys?: string[]) =>
  request<SettingsOut>('/settings/reset', {
    method: 'POST',
    body: JSON.stringify({ keys: keys ?? null }),
  })

// ---------------- 流式问答 ----------------
export interface StreamChatRequest {
  message: string
  kb_id?: string | null
  conversation_id?: string | null
  provider?: string | null
  mode?: SearchMode | null
}

/**
 * 以异步生成器形式消费 SSE，调用方用 `for await` 逐事件处理。
 * 这样 UI 能在收到第一个 delta 时立刻渲染，而不是等整个回答。
 */
export async function* streamChat(
  req: StreamChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const resp = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  })

  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`
    try {
      const body = await resp.json()
      if (body?.detail) detail = body.detail
    } catch {
      /* ignore */
    }
    throw new ApiError(detail, resp.status)
  }
  if (!resp.body) throw new ApiError('响应没有可读的数据流', 500)

  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // SSE 以空行分帧
      let boundary = buffer.indexOf('\n\n')
      while (boundary >= 0) {
        const frame = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + 2)

        for (const line of frame.split('\n')) {
          const trimmed = line.trimStart()
          if (!trimmed.startsWith('data:')) continue
          const payload = trimmed.slice(5).trim()
          if (!payload) continue
          try {
            yield JSON.parse(payload) as ChatEvent
          } catch {
            // 半包或非 JSON 的心跳帧，忽略即可
          }
        }
        boundary = buffer.indexOf('\n\n')
      }
    }
  } finally {
    reader.cancel().catch(() => {})
  }
}

// ---------------- 工具 ----------------
export function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)))
  return `${(bytes / 1024 ** i).toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

export function formatTime(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}

export function formatDateTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return ''
  }
}
