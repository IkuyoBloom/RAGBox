/** 与后端 `app/schemas.py` 一一对应的类型定义。 */

export interface Citation {
  index: number
  chunk_id: string
  doc_id: string
  filename: string
  page: number | null
  ordinal: number
  score: number
  dense_score: number | null
  sparse_score: number | null
  dense_rank: number | null
  sparse_rank: number | null
  source: 'dense' | 'sparse' | 'both' | 'none'
  preview: string
}

export interface KnowledgeBase {
  id: string
  name: string
  description: string
  embed_model: string
  embed_dim: number | null
  chunk_size: number
  chunk_overlap: number
  collection_name: string
  created_at: string
  updated_at: string
  document_count: number
  ready_document_count: number
  chunk_count: number
}

export type DocumentStatus = 'pending' | 'parsing' | 'embedding' | 'ready' | 'failed'

export interface RagDocument {
  id: string
  kb_id: string
  filename: string
  file_type: string
  size_bytes: number
  status: DocumentStatus
  chunk_count: number
  char_count: number
  page_count: number
  error: string
  meta: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface KnowledgeBaseDetail extends KnowledgeBase {
  documents: RagDocument[]
}

export interface Chunk {
  id: string
  doc_id: string
  kb_id: string
  ordinal: number
  content: string
  char_count: number
  page: number | null
  filename: string
}

export interface UploadResult {
  filename: string
  ok: boolean
  document: RagDocument | null
  error: string | null
  duplicated: boolean
}

export interface ChatMessage {
  id: string
  conversation_id: string
  role: 'user' | 'assistant'
  content: string
  rewritten_query: string | null
  citations: Citation[]
  provider: string | null
  model: string | null
  latency_ms: number | null
  retrieval_ms: number | null
  prompt_tokens: number | null
  completion_tokens: number | null
  meta: Record<string, unknown>
  created_at: string
}

export interface Conversation {
  id: string
  kb_id: string | null
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface ConversationDetail extends Conversation {
  messages: ChatMessage[]
}

/** 依赖关系：只有当 `key` 取 `values` 中的值时，本项才真正生效。 */
export interface SettingDependency {
  key: string
  values: string[]
}

/** 一级分组：顺序与一句话定位由后端给出。 */
export interface SettingGroup {
  name: string
  summary: string
  order: number
}

export interface SettingItem {
  key: string
  value: unknown
  configured: boolean
  overridden: boolean
  label: string
  group: string
  /** 二级分组。空字符串表示「本组主控项」，界面会置顶并突出展示。 */
  section: string
  description: string
  type: 'bool' | 'int' | 'float' | 'str'
  secret: boolean
  choices: string[] | null
  /** 选项显示名 {值: 文本}，让下拉框显示「纯本地 · Ollama」而不是裸的 ollama */
  choice_labels: Record<string, string> | null
  /**
   * 动态选项来源（如 `ollama_embedding_models`）。
   * 非空表示 `choices` 是**运行时建议**而非强约束枚举，界面应渲染成
   * 「可自由输入 + 下拉建议」—— Ollama 没启动时建议列表会是空的。
   */
  choices_source: string | null
  /**
   * 动态候选是否真的取到了。false 表示查询失败（如 Ollama 未启动），
   * 此时 `choices` 只含当前值，应视为「未知」而非「没装这个模型」。
   */
  choices_loaded: boolean
  minimum: number | null
  maximum: number | null
  /** 生效范围提示，例如「仅对新建知识库生效」 */
  scope: string
  depends_on: SettingDependency | null
}

export interface SettingsOut {
  items: SettingItem[]
  groups: SettingGroup[]
  readonly: Record<string, unknown>
  overridden_keys: string[]
}

export interface ProviderHealth {
  name: string
  ok: boolean
  detail: string
  model: string
  models: string[]
  base_url: string
}

export interface Health {
  status: string
  version: string
  database: boolean
  vector_store: boolean
  active_provider: string
  active_model: string
  embed_model: string
  providers: ProviderHealth[]
}

// ---------------- SSE 事件 ----------------
export interface RetrievalTrace {
  type: 'trace'
  mode: string
  dense_hits: number
  sparse_hits: number
  fused_hits: number
  elapsed_ms: number
  stages: Array<Record<string, unknown>>
}

export type ChatEvent =
  | { type: 'start'; conversation_id: string; kb_id: string; question: string }
  | { type: 'rewrite'; query: string; original: string }
  | { type: 'citations'; citations: Citation[] }
  | RetrievalTrace
  | { type: 'delta'; text: string }
  | {
      type: 'done'
      message_id: string
      provider: string
      model: string
      latency_ms: number
      retrieval_ms: number
      prompt_tokens: number | null
      completion_tokens: number | null
      finish_reason: string
    }
  | { type: 'error'; message: string; provider: string; partial?: string }
  | { type: 'eos' }

export type SearchMode = 'hybrid' | 'dense_only' | 'sparse_only'
