import { useState } from 'react'
import * as api from '../api'
import { KB_MODEL_STATE_TEXT, describeKbModel, type EmbedInfo } from '../embedModel'
import type { Toast } from '../App'
import type { Health, KnowledgeBase } from '../types'
import Logo from './Logo'
import ModelPicker from './ModelPicker'

interface Props {
  kbs: KnowledgeBase[]
  selectedKbId: string | null
  embedInfo: EmbedInfo
  health: Health | null
  provider: string
  theme: 'light' | 'dark'
  onSelect: (id: string) => void
  onRefreshKbs: () => Promise<void> | void
  onRefreshAll: () => void
  onToggleTheme: () => void
  onOpenSettings: () => void
  notify: (kind: Toast['kind'], text: string) => void
}

function ThemeIcon({ theme }: { theme: 'light' | 'dark' }) {
  return theme === 'dark' ? (
    <svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true" fill="none">
      <circle cx="12" cy="12" r="4" fill="currentColor" />
      <g stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
        <path d="M12 3v2M12 19v2M3 12h2M19 12h2" />
        <path d="M5.6 5.6l1.4 1.4M16.9 16.9l1.4 1.4M18.4 5.6l-1.4 1.4M7.1 16.9l-1.4 1.4" />
      </g>
    </svg>
  ) : (
    <svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" fill="currentColor" />
    </svg>
  )
}

export default function Sidebar(props: Props) {
  const { embedInfo, health } = props
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [newModel, setNewModel] = useState('')
  const [busy, setBusy] = useState(false)

  const activeProvider = health?.providers.find((p) => p.name === props.provider)
  const ready = activeProvider?.ok ?? false

  const toggleCreate = () => {
    if (creating) {
      setCreating(false)
      return
    }
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
        embed_model: newModel.trim() || null,
      })
      setNewName('')
      setNewModel('')
      setCreating(false)
      await props.onRefreshKbs()
      props.onSelect(kb.id)
      props.notify('ok', `知识库「${kb.name}」已创建`)
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

  return (
    <aside className="sidebar">
      {/* 顶栏品牌区 */}
      <div className="sidebar-brand">
        <div className="brand-left">
          <Logo size={22} />
          <span className="brand-title">知识库</span>
        </div>
        <button className="btn tiny" onClick={toggleCreate}>
          {creating ? '取消' : '+ 新建'}
        </button>
      </div>

      {/* 新建知识库内联卡片 */}
      {creating && (
        <div className="create-form">
          <input
            autoFocus
            type="text"
            placeholder="输入知识库名称…"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && void createKb()}
          />
          <div className="create-model">
            <span className="create-model-label">向量模型</span>
            <ModelPicker
              value={newModel}
              options={embedInfo.installed}
              placeholder="默认跟随全局设置"
              emptyHint="未获取到候选列表，可手动输入模型名，留空则用默认设置"
              onChange={setNewModel}
            />
          </div>
          <div className="create-actions">
            <span className="create-hint">建库后模型固定</span>
            <button
              className="btn tiny primary"
              disabled={busy || !newName.trim()}
              onClick={() => void createKb()}
            >
              创建
            </button>
          </div>
        </div>
      )}

      {/* 知识库导航列表 */}
      <div className="sidebar-section-label">
        <span>全部知识库</span>
        <span className="sidebar-section-count">{props.kbs.length}</span>
      </div>

      <ul className="kb-list">
        {props.kbs.length === 0 && (
          <li className="empty">还没有知识库，点击右上角「+ 新建」创建</li>
        )}
        {props.kbs.map((kb) => {
          const badge = describeKbModel(kb, embedInfo)
          const note = KB_MODEL_STATE_TEXT[badge.state]
          const isSelected = props.selectedKbId === kb.id
          return (
            <li
              key={kb.id}
              className={`kb-item ${isSelected ? 'active' : ''}`}
              onClick={() => props.onSelect(kb.id)}
            >
              <div className="kb-main">
                <div className="kb-row-top">
                  <span className="kb-name">{kb.name}</span>
                  <span className="kb-doc-pill">{kb.document_count} 份</span>
                </div>
                <div className="kb-row-sub">
                  <span>{kb.chunk_count} 个切片</span>
                  <span className={`model-badge model-${badge.state}`} title={badge.hint}>
                    <code>{badge.label}</code>
                    {note && <span className="model-badge-note">{note}</span>}
                  </span>
                </div>
              </div>
              <button
                className="icon-btn kb-del-btn"
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

      {/* 底部全局状态与设置栏 */}
      <div className="sidebar-footer">
        <div
          className="current-embed"
          title="新建知识库将默认使用该向量模型；已有知识库的模型在建库时已冻结"
        >
          <span className="current-embed-label">默认向量</span>
          {embedInfo.current ? (
            <code className="current-embed-value">{embedInfo.current}</code>
          ) : (
            <span className="current-embed-unknown">读取中…</span>
          )}
        </div>

        <div className="sidebar-footer-bar">
          <button
            className="status-btn"
            onClick={props.onRefreshAll}
            title={`${activeProvider?.detail ?? '检测中'}（点击刷新状态）`}
          >
            <span className={`status-dot-indicator ${ready ? 'ok' : 'bad'}`} />
            <span>{ready ? '服务正常' : '通道离线'}</span>
          </button>

          <div className="sidebar-footer-actions">
            <button
              className="btn icon-only"
              onClick={props.onToggleTheme}
              title={props.theme === 'dark' ? '切换浅色模式' : '切换深色模式'}
              aria-label="切换主题"
            >
              <ThemeIcon theme={props.theme} />
            </button>
            <button className="btn ghost tiny" onClick={props.onOpenSettings}>
              设置
            </button>
          </div>
        </div>
      </div>
    </aside>
  )
}
