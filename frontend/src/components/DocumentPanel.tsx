import { useMemo, useRef, useState } from 'react'
import * as api from '../api'
import { classifyPaths, describeUnsupported, supportedHint } from '../filePick'
import type { Toast } from '../App'
import type { KnowledgeBaseDetail } from '../types'

interface Props {
  kbDetail: KnowledgeBaseDetail | null
  selectedKbId: string | null
  onRefreshKbs: () => Promise<void> | void
  onRefreshDetail: () => void
  notify: (kind: Toast['kind'], text: string) => void
  onClose: () => void
}

const STATUS_TEXT: Record<string, string> = {
  pending: '排队中',
  parsing: '解析中',
  embedding: '向量化',
  ready: '就绪',
  failed: '失败',
}

const DOC_RENDER_LIMIT = 200
const DOC_RENDER_STEP = 200
const UPLOAD_BATCH = 8

function getExtLabel(filename: string): string {
  const dot = filename.lastIndexOf('.')
  if (dot < 0) return 'FILE'
  const ext = filename.slice(dot + 1).toLowerCase()
  if (ext === 'markdown') return 'MD'
  if (ext === 'docx') return 'DOC'
  if (ext === 'text') return 'TXT'
  return ext.slice(0, 4).toUpperCase()
}

export default function DocumentPanel(props: Props) {
  const { kbDetail } = props
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null)
  const [docLimit, setDocLimit] = useState(DOC_RENDER_LIMIT)
  const [query, setQuery] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  const dirRef = useRef<HTMLInputElement>(null)

  const allDocs = kbDetail?.documents ?? []
  const filteredDocs = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return allDocs
    return allDocs.filter((d) => d.filename.toLowerCase().includes(q))
  }, [allDocs, query])

  const visibleDocs = filteredDocs.slice(0, docLimit)
  const hiddenDocCount = Math.max(0, filteredDocs.length - visibleDocs.length)

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
          `已接收 ${okCount} 个文件${dupCount ? `（其中 ${dupCount} 个重复已跳过）` : ''}，正在后台建立索引…`,
        )
      }
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
          (kinds.length ? `（跳过：${kinds.slice(0, 6).join('  ')}）` : '') +
          `。支持：${supportedHint()}`,
      )
      return
    }

    const parts = [`识别到 ${usable.length} 个可导入文档`]
    if (cls.unsupported.length > 0) {
      const kinds = Array.from(new Set(cls.unsupported.map((i) => describeUnsupported(paths[i]))))
      parts.push(`跳过 ${cls.unsupported.length} 个不支持格式（${kinds.slice(0, 5).join('  ')}）`)
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

  if (!kbDetail) {
    return (
      <aside className="doc-panel">
        <div className="doc-panel-head">
          <div className="doc-panel-title-wrap">
            <span className="doc-panel-title">文档目录</span>
          </div>
          <button className="icon-btn" onClick={props.onClose} title="收起文档栏">
            ×
          </button>
        </div>
        <div className="doc-empty-state">先在左侧选择一个知识库</div>
      </aside>
    )
  }

  return (
    <aside className="doc-panel">
      <div className="doc-panel-head">
        <div className="doc-panel-title-wrap">
          <span className="doc-panel-title">库内文档</span>
          <span className="doc-panel-count">{kbDetail.documents.length}</span>
        </div>

        <div className="doc-panel-actions">
          <button
            className="btn tiny"
            disabled={uploading}
            title="上传一个或多个文件"
            onClick={() => fileRef.current?.click()}
          >
            + 文件
          </button>
          <button
            className="btn tiny ghost"
            disabled={uploading}
            title="选择整个文件夹批量导入（含子目录）"
            onClick={() => dirRef.current?.click()}
          >
            文件夹
          </button>
          <button className="icon-btn" onClick={props.onClose} title="收起文档栏">
            ×
          </button>
        </div>

        <input
          ref={fileRef}
          type="file"
          multiple
          hidden
          accept=".pdf,.docx,.md,.markdown,.txt,.text,.log"
          onChange={(e) => handlePick(e.target.files, false)}
        />
        <input
          ref={dirRef}
          type="file"
          multiple
          hidden
          {...({ webkitdirectory: '', directory: '' } as Record<string, string>)}
          onChange={(e) => handlePick(e.target.files, true)}
        />
      </div>

      {/* 知识库索引参数摘要（无边框柔和胶囊） */}
      <div className="doc-kb-specs">
        <span className="spec-pill" title="该知识库绑定的向量模型（建库时冻结）">
          {kbDetail.embed_model}
        </span>
        <span className="spec-pill" title="向量维度">
          {kbDetail.embed_dim ? `${kbDetail.embed_dim} 维` : '待索引'}
        </span>
        <span className="spec-pill" title="切片长度 / 重叠长度">
          切片 {kbDetail.chunk_size}/{kbDetail.chunk_overlap}
        </span>
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

      {allDocs.length > 5 && (
        <div className="doc-search-wrap">
          <input
            type="text"
            className="doc-search-input"
            placeholder="搜索文件名…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
      )}

      <ul className="doc-list">
        {allDocs.length === 0 && (
          <li className="doc-empty-state">
            <div className="doc-empty-title">暂无文档</div>
            <div className="doc-empty-sub">
              支持 PDF、Word、Markdown、TXT，可点击上方按钮上传文件或整个文件夹
            </div>
          </li>
        )}
        {allDocs.length > 0 && visibleDocs.length === 0 && (
          <li className="doc-empty-state">没有匹配的文档</li>
        )}
        {visibleDocs.map((doc) => {
          const ext = getExtLabel(doc.filename)
          return (
            <li key={doc.id} className="doc-card">
              <div className="doc-ext-badge" data-ext={ext}>
                {ext}
              </div>
              <div className="doc-card-body">
                <div className="doc-top">
                  <span className="doc-name" title={doc.filename}>
                    {doc.filename}
                  </span>
                  <span className={`badge badge-${doc.status}`}>
                    {STATUS_TEXT[doc.status] ?? doc.status}
                  </span>
                </div>
                <div className="doc-meta-row">
                  <span className="doc-meta">
                    {api.formatBytes(doc.size_bytes)}
                    {doc.status === 'ready' && <> · {doc.chunk_count} 片</>}
                    {doc.page_count > 0 && <> · {doc.page_count} 页</>}
                  </span>
                  <span className="doc-actions">
                    <button className="link-btn" onClick={() => void reindex(doc.id)}>
                      重建
                    </button>
                    <button
                      className="link-btn danger"
                      onClick={() => void removeDoc(doc.id, doc.filename)}
                    >
                      删除
                    </button>
                  </span>
                </div>
                {doc.status === 'failed' && doc.error && (
                  <div className="doc-error" title={doc.error}>
                    {doc.error}
                  </div>
                )}
              </div>
            </li>
          )
        })}
      </ul>

      {hiddenDocCount > 0 && (
        <button className="doc-more" onClick={() => setDocLimit((n) => n + DOC_RENDER_STEP)}>
          显示更多（剩余 {hiddenDocCount} 份）
        </button>
      )}
      {docLimit > DOC_RENDER_LIMIT && hiddenDocCount === 0 && (
        <button className="doc-more" onClick={() => setDocLimit(DOC_RENDER_LIMIT)}>
          收起列表
        </button>
      )}
    </aside>
  )
}
