import { useRef, useState } from 'react'
import * as api from '../api'
import { KB_MODEL_STATE_TEXT, describeKbModel, type EmbedInfo } from '../embedModel'
import { classifyPaths, describeUnsupported, supportedHint } from '../filePick'
import type { Toast } from '../App'
import type { KnowledgeBase, KnowledgeBaseDetail } from '../types'
import ModelPicker from './ModelPicker'

interface Props {
  kbs: KnowledgeBase[]
  selectedKbId: string | null
  kbDetail: KnowledgeBaseDetail | null
  /** 当前设置里的向量模型 + Ollama 已装模型，用于判断每个库的模型状态 */
  embedInfo: EmbedInfo
  onSelect: (id: string) => void
  onRefreshKbs: () => Promise<void> | void
  onRefreshDetail: () => void
  notify: (kind: Toast['kind'], text: string) => void
}

const STATUS_TEXT: Record<string, string> = {
  pending: '排队中',
  parsing: '解析中',
  embedding: '向量化中',
  ready: '就绪',
  failed: '失败',
}

/**
 * 一次最多渲染多少条文档。
 *
 * 为什么要设上限：本项目实测单个知识库有 **3500+ 份文档**。全量渲染会产生
 * 两万多个 DOM 节点，而且轮询每 8 秒刷新一次 kbDetail —— 每次都要重新协调
 * 这一整棵树，界面会明细发顿。
 *
 * 顺带也是可用性问题：几千条的列表本来就不可能靠滚动查找，
 * 需要定位某一份文档时用搜索更实际。分批展开作为折中。
 */
const DOC_RENDER_LIMIT = 200
const DOC_RENDER_STEP = 200

/**
 * 面板标题栏的小图标。
 *
 * 加图标是为了让两块面板**一眼能分开** —— 原先两块都是同样的白卡片
 * 加一行同样大小的灰色小标题，扫一眼分不出哪块管什么。
 * 配合各自的左侧色条与计数徽标，形成三个互相印证的视觉线索。
 */
function LibraryIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 2.6 1.9 7.8 12 13l10.1-5.2L12 2.6z" fill="currentColor" />
      <path
        d="M3 12.2 12 16.8l9-4.6M3 16.4 12 21l9-4.6"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function DocumentIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M6.5 2.6h7L18 7.1V20a1.4 1.4 0 0 1-1.4 1.4H6.5A1.4 1.4 0 0 1 5.1 20V4a1.4 1.4 0 0 1 1.4-1.4z"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <path
        d="M8.8 12.4h6.4M8.8 16.2h4.2"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  )
}

export default function Sidebar(props: Props) {
  const { kbDetail, embedInfo } = props
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [newModel, setNewModel] = useState('')
  const [busy, setBusy] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null)
  // 面板列表长了会占满整个侧边栏，收起来才能腾出空间看知识库列表
  const [docsOpen, setDocsOpen] = useState(true)
  // 文档多的时候只渲染前若干条，见 DOC_RENDER_LIMIT 的说明
  const [docLimit, setDocLimit] = useState(DOC_RENDER_LIMIT)
  const fileRef = useRef<HTMLInputElement>(null)
  const dirRef = useRef<HTMLInputElement>(null)

  // 只渲染前 docLimit 条文档，避免几千条把界面拖顿（见 DOC_RENDER_LIMIT）
  const allDocs = kbDetail?.documents ?? []
  const visibleDocs = allDocs.slice(0, docLimit)
  const hiddenDocCount = Math.max(0, allDocs.length - visibleDocs.length)

  const toggleCreate = () => {
    if (creating) {
      setCreating(false)
      return
    }
    // 打开时以「当前设置」为默认模型，用户可改
    setNewModel(embedInfo.current)
    setCreating(true)
  }

  const createKb = async () => {
    const name = newName.trim()
    if (!name) return
    setBusy(true)
    try {
      const kb = await api.createKb({
        name,
        // 留空表示跟随后端的全局默认，不要传空串
        embed_model: newModel.trim() || null,
      })
      setNewName('')
      setNewModel('')
      setCreating(false)
      await props.onRefreshKbs()
      props.onSelect(kb.id)
      props.notify('ok', `知识库「${kb.name}」已创建（向量模型 ${kb.embed_model}）`)
    } catch (e) {
      props.notify('err', (e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const removeKb = async (kb: KnowledgeBase) => {
    if (
      !window.confirm(
        `确定删除知识库「${kb.name}」？其下 ${kb.document_count} 份文档与全部向量都会被清除。`,
      )
    ) {
      return
    }
    try {
      await api.deleteKb(kb.id)
      await props.onRefreshKbs()
      props.notify('ok', `已删除知识库「${kb.name}」`)
    } catch (e) {
      props.notify('err', (e as Error).message)
    }
  }

  /**
   * 分批上传。
   *
   * 逐个传太慢（几百个文件几百次往返），一次全传又容易超时、且中途失败
   * 整个请求白费。分批兼顾两者，而且每批结束后刷新一次列表 ——
   * 用户能看到文档一个个出现，而不是等到全部传完才动。
   */
  const UPLOAD_BATCH = 8

  const runUpload = async (files: File[]) => {
    const kbId = props.selectedKbId
    if (!kbId || files.length === 0) return

    setUploading(true)
    setProgress({ done: 0, total: files.length })
    let okCount = 0
    let dupCount = 0
    const failures: string[] = []

    try {
      for (let start = 0; start < files.length; start += UPLOAD_BATCH) {
        const batch = files.slice(start, start + UPLOAD_BATCH)
        const results = await api.uploadDocuments(kbId, batch)
        for (const r of results) {
          if (r.ok) {
            okCount += 1
            if (r.duplicated) dupCount += 1
          } else {
            failures.push(`${r.filename}：${r.error}`)
          }
        }
        setProgress({ done: Math.min(start + batch.length, files.length), total: files.length })
        props.onRefreshDetail()
      }

      if (okCount > 0) {
        props.notify(
          'ok',
          `已接收 ${okCount} 个文件${dupCount ? `（其中 ${dupCount} 个内容重复，已秒传跳过）` : ''}，正在后台建立索引…`,
        )
      }
      // 失败最多逐条提示 3 个，其余汇总 —— 一次失败 40 个会刷屏
      for (const msg of failures.slice(0, 3)) props.notify('err', msg)
      if (failures.length > 3) {
        props.notify('err', `另有 ${failures.length - 3} 个文件失败，可在文档列表查看原因`)
      }
      await props.onRefreshKbs()
    } catch (e) {
      props.notify('err', (e as Error).message)
    } finally {
      setUploading(false)
      setProgress(null)
      if (fileRef.current) fileRef.current.value = ''
      if (dirRef.current) dirRef.current.value = ''
    }
  }

  /**
   * 处理一次选择（文件或文件夹）。
   *
   * 文件夹选择会把目录里所有文件都给过来 —— Office 锁文件、Thumbs.db、
   * node_modules 里的成千上万个文件。`accept` 属性对文件夹选择器无效，
   * 过滤只能在这里做，规则见 filePick.ts。
   */
  const handlePick = (list: FileList | null, fromFolder: boolean) => {
    if (!list || list.length === 0) return
    const all = Array.from(list)
    const paths = all.map((f) =>
      fromFolder
        ? (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name
        : f.name,
    )
    const cls = classifyPaths(paths)
    const usable = cls.usable.map((index) => all[index])

    if (usable.length === 0) {
      const kinds = Array.from(new Set(cls.unsupported.map((i) => describeUnsupported(paths[i]))))
      props.notify(
        'err',
        `选中的 ${all.length} 个文件里没有可导入的文档` +
          (kinds.length ? `（遇到不支持的格式：${kinds.slice(0, 6).join('  ')}）` : '') +
          `。支持：${supportedHint()}`,
      )
      return
    }

    // 汇总成一条提示。拆成「成功一条 + 跳过一条」会互相覆盖 ——
    // 本项目同一时刻只显示一个 toast。
    const parts = [`识别到 ${usable.length} 个可导入文档`]
    if (cls.unsupported.length > 0) {
      const kinds = Array.from(new Set(cls.unsupported.map((i) => describeUnsupported(paths[i]))))
      parts.push(`跳过 ${cls.unsupported.length} 个不支持的（${kinds.slice(0, 5).join('  ')}）`)
    }
    if (cls.junk.length > 0) parts.push(`忽略 ${cls.junk.length} 个临时/隐藏文件`)
    props.notify('ok', `${parts.join('，')}，开始导入…`)

    void runUpload(usable)
  }

  const removeDoc = async (docId: string, filename: string) => {
    if (!window.confirm(`删除文档「${filename}」及其全部向量？`)) return
    try {
      await api.deleteDocument(docId)
      props.onRefreshDetail()
      await props.onRefreshKbs()
      props.notify('ok', '文档已删除')
    } catch (e) {
      props.notify('err', (e as Error).message)
    }
  }

  const reindex = async (docId: string) => {
    try {
      await api.reindexDocument(docId)
      props.onRefreshDetail()
      props.notify('ok', '已重新排队建立索引')
    } catch (e) {
      props.notify('err', (e as Error).message)
    }
  }

  return (
    <aside className="sidebar">
      <section className="panel panel-kb">
        <div className="panel-head">
          <span className="panel-icon" aria-hidden="true">
            <LibraryIcon />
          </span>
          <h2>知识库</h2>
          <span className="panel-count" title={`共 ${props.kbs.length} 个知识库`}>
            {props.kbs.length}
          </span>
          <button className="btn tiny" onClick={toggleCreate}>
            {creating ? '取消' : '新建'}
          </button>
        </div>

        {/* 当前设置用的向量模型：新建知识库会用它。
            「库自己的模型」与「当前设置」是两件事，这里放在一起才能对照着看。 */}
        <div className="current-embed" title="新建知识库将使用这个向量模型；已有知识库的模型在创建时就固定了">
          <span className="current-embed-label">当前向量模型</span>
          {embedInfo.current ? (
            <code className="current-embed-value">{embedInfo.current}</code>
          ) : (
            <span className="current-embed-unknown">读取中…</span>
          )}
          {!embedInfo.loaded && embedInfo.current && (
            <span className="badge-inactive" title="未能获取 Ollama 的模型列表，无法确认它是否已安装">
              未确认
            </span>
          )}
        </div>

        {creating && (
          <div className="create-form">
            <input
              autoFocus
              placeholder="知识库名称，例如：项目资料"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && void createKb()}
            />
            <div className="create-model">
              <span className="create-model-label">向量模型</span>
              <ModelPicker
                value={newModel}
                options={embedInfo.installed}
                placeholder="默认跟随当前设置"
                emptyHint="未获取到候选列表（Ollama 未启动？），可手动输入模型名，留空则用当前设置"
                onChange={setNewModel}
              />
            </div>
            <div className="create-actions">
              <span className="create-hint">建库后模型不可更改</span>
              <button
                className="btn tiny"
                disabled={busy || !newName.trim()}
                onClick={() => void createKb()}
              >
                创建
              </button>
            </div>
          </div>
        )}

        <ul className="kb-list">
          {props.kbs.length === 0 && <li className="empty">还没有知识库，先新建一个</li>}
          {props.kbs.map((kb) => {
            const badge = describeKbModel(kb, embedInfo)
            const note = KB_MODEL_STATE_TEXT[badge.state]
            return (
              <li
                key={kb.id}
                className={`kb-item ${props.selectedKbId === kb.id ? 'active' : ''}`}
                onClick={() => props.onSelect(kb.id)}
              >
                <div className="kb-main">
                  <div className="kb-name">{kb.name}</div>
                  <div className="kb-meta">
                    {kb.ready_document_count}/{kb.document_count} 份 · {kb.chunk_count} 片
                  </div>
                  <span className={`model-badge model-${badge.state}`} title={badge.hint}>
                    <code>{badge.label}</code>
                    {note && <span className="model-badge-note">{note}</span>}
                  </span>
                </div>
                <button
                  className="icon-btn"
                  title="删除知识库"
                  onClick={(e) => {
                    e.stopPropagation()
                    void removeKb(kb)
                  }}
                >
                  ×
                </button>
              </li>
            )
          })}
        </ul>
      </section>

      {kbDetail && (
        <section className="panel grow panel-docs">
          <div
            className="panel-head clickable"
            onClick={() => setDocsOpen((v) => !v)}
            title={docsOpen ? '收起文档列表' : '展开文档列表'}
          >
            <span className="chev">{docsOpen ? '▾' : '▸'}</span>
            <span className="panel-icon" aria-hidden="true">
              <DocumentIcon />
            </span>
            <h2>文档</h2>
            <span
              className="panel-count"
              title={`共 ${kbDetail.documents.length} 份文档`}
            >
              {kbDetail.documents.length}
            </span>
            <button
              className="btn tiny"
              disabled={uploading}
              title="选择一个或多个文件"
              onClick={(e) => {
                e.stopPropagation()
                fileRef.current?.click()
              }}
            >
              文件
            </button>
            <button
              className="btn tiny"
              disabled={uploading}
              title="选择整个文件夹，自动识别其中的可用文档并批量导入（含子目录）"
              onClick={(e) => {
                e.stopPropagation()
                dirRef.current?.click()
              }}
            >
              文件夹
            </button>
            <input
              ref={fileRef}
              type="file"
              multiple
              hidden
              accept=".pdf,.docx,.md,.markdown,.txt,.text,.log"
              onChange={(e) => handlePick(e.target.files, false)}
            />
            {/* webkitdirectory 不在 React 的 JSX 类型定义里，用展开传原生属性。
                注意 accept 对目录选择器**无效**，真正的过滤在 handlePick 里做。 */}
            <input
              ref={dirRef}
              type="file"
              multiple
              hidden
              {...({ webkitdirectory: '', directory: '' } as Record<string, string>)}
              onChange={(e) => handlePick(e.target.files, true)}
            />
          </div>

          {progress && (
            <div className="upload-progress">
              <div className="upload-progress-text">
                正在导入 {progress.done} / {progress.total}
              </div>
              <div className="upload-progress-track">
                <span
                  className="upload-progress-bar"
                  style={{ width: `${Math.round((progress.done / progress.total) * 100)}%` }}
                />
              </div>
            </div>
          )}

          {docsOpen && (
            <>
              <div className="kb-config">
                <span title="该库建库时确定的向量模型，之后不会随设置变化">
                  向量 <code>{kbDetail.embed_model}</code>
                </span>
                <span title="向量维度，首个文档索引完成后确定">维度 {kbDetail.embed_dim ?? '待索引'}</span>
              </div>
              <div className="kb-config">
                <span>切片 {kbDetail.chunk_size}</span>
                <span>重叠 {kbDetail.chunk_overlap}</span>
              </div>

              <ul className="doc-list">
                {kbDetail.documents.length === 0 && (
                  <li className="empty">还没有文档，点「上传」添加 PDF / Word / Markdown / TXT</li>
                )}
                {visibleDocs.map((doc) => (
                  <li key={doc.id} className="doc-item">
                    <div className="doc-top">
                      <span className="doc-name" title={doc.filename}>
                        {doc.filename}
                      </span>
                      <span className={`badge badge-${doc.status}`}>
                        {STATUS_TEXT[doc.status] ?? doc.status}
                      </span>
                    </div>
                    <div className="doc-meta">
                      {api.formatBytes(doc.size_bytes)}
                      {doc.status === 'ready' && <> · {doc.chunk_count} 片</>}
                      {doc.page_count > 0 && <> · {doc.page_count} 页</>}
                    </div>
                    {doc.status === 'failed' && doc.error && (
                      <div className="doc-error" title={doc.error}>
                        {doc.error}
                      </div>
                    )}
                    <div className="doc-actions">
                      <button className="link-btn" onClick={() => void reindex(doc.id)}>
                        重建索引
                      </button>
                      <button
                        className="link-btn danger"
                        onClick={() => void removeDoc(doc.id, doc.filename)}
                      >
                        删除
                      </button>
                    </div>
                  </li>
                ))}
              </ul>

              {hiddenDocCount > 0 && (
                <button
                  className="doc-more"
                  onClick={() => setDocLimit((n) => n + DOC_RENDER_STEP)}
                >
                  显示更多（还有 {hiddenDocCount} 份未显示）
                </button>
              )}
              {docLimit > DOC_RENDER_LIMIT && hiddenDocCount === 0 && (
                <button className="doc-more" onClick={() => setDocLimit(DOC_RENDER_LIMIT)}>
                  收起，只显示前 {DOC_RENDER_LIMIT} 份
                </button>
              )}
            </>
          )}
        </section>
      )}
    </aside>
  )
}
