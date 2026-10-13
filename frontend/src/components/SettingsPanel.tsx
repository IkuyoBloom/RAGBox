import { useEffect, useMemo, useState } from 'react'
import * as api from '../api'
import {
  buildTree,
  computeChanged,
  dependencyHint,
  draftFrom,
  isActive,
  type GroupNode,
} from '../settingsTree'
import type { Toast } from '../App'
import type { SettingItem, SettingsOut } from '../types'
import ModelPicker from './ModelPicker'

interface Props {
  settings: SettingsOut
  onClose: () => void
  onSaved: (s: SettingsOut) => void
  notify: (kind: Toast['kind'], text: string) => void
}

const READONLY_GROUP = '环境信息'
const READONLY_HINT = '由 .env 与运行环境决定，界面只读'

export default function SettingsPanel({ settings, onClose, onSaved, notify }: Props) {
  const [draft, setDraft] = useState<Record<string, unknown>>({})
  const [saving, setSaving] = useState(false)

  const tree = useMemo<GroupNode[]>(() => buildTree(settings), [settings])
  const [activeTab, setActiveTab] = useState<string>(() => tree[0]?.name ?? READONLY_GROUP)

  useEffect(() => {
    setDraft(draftFrom(settings.items))
  }, [settings])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const byKey = useMemo(
    () => new Map(settings.items.map((i) => [i.key, i])),
    [settings.items],
  )

  const changed = useMemo(
    () => computeChanged(settings.items, draft),
    [draft, settings.items],
  )

  const changedCount = Object.keys(changed).length

  const save = async () => {
    if (!changedCount) {
      onClose()
      return
    }
    setSaving(true)
    try {
      onSaved(await api.updateSettings(changed))
    } catch (e) {
      notify('err', (e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  const resetAll = async () => {
    if (!window.confirm('把所有配置还原为 .env 里的默认值？')) return
    setSaving(true)
    try {
      onSaved(await api.resetSettings())
      notify('ok', '已还原为默认配置')
    } catch (e) {
      notify('err', (e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  const renderControl = (item: SettingItem) => {
    const value = draft[item.key]

    if (item.type === 'bool') {
      return (
        <label className="switch">
          <input
            type="checkbox"
            checked={Boolean(value)}
            onChange={(e) => setDraft((d) => ({ ...d, [item.key]: e.target.checked }))}
          />
          <span className="switch-track" />
          <span className="switch-text">{value ? '开启' : '关闭'}</span>
        </label>
      )
    }

    if (item.choices_source) {
      return (
        <ModelPicker
          value={String(value ?? '')}
          options={item.choices ?? []}
          placeholder="可直接输入模型名"
          emptyHint="未获取到候选列表（Ollama 未启动？），可直接手动输入模型名"
          onChange={(v) => setDraft((d) => ({ ...d, [item.key]: v }))}
        />
      )
    }

    if (item.choices) {
      return (
        <select
          value={String(value ?? '')}
          onChange={(e) => setDraft((d) => ({ ...d, [item.key]: e.target.value }))}
        >
          {item.choices.map((c) => (
            <option key={c} value={c}>
              {item.choice_labels?.[c] ?? c}
            </option>
          ))}
        </select>
      )
    }

    if (item.type === 'int' || item.type === 'float') {
      return (
        <input
          type="number"
          value={value === null || value === undefined ? '' : String(value)}
          min={item.minimum ?? undefined}
          max={item.maximum ?? undefined}
          step={item.type === 'float' ? 0.05 : 1}
          onChange={(e) =>
            setDraft((d) => ({
              ...d,
              [item.key]: e.target.value === '' ? '' : Number(e.target.value),
            }))
          }
        />
      )
    }

    return (
      <input
        type={item.secret ? 'password' : 'text'}
        value={String(value ?? '')}
        placeholder={item.secret && item.configured ? '已配置，留空表示不修改' : ''}
        onChange={(e) => setDraft((d) => ({ ...d, [item.key]: e.target.value }))}
      />
    )
  }

  const renderField = (item: SettingItem) => {
    const active = isActive(item, draft)
    const hint = active ? null : dependencyHint(item, byKey)
    const isChanged = changed[item.key] !== undefined
    const noSuggestions = Boolean(item.choices_source) && !item.choices?.length

    return (
      <div className={`field ${active ? '' : 'field-inactive'}`} key={item.key}>
        <div className="field-label">
          <span className="field-name">
            {item.label}
            {isChanged && <span className="tag-dirty">待保存</span>}
            {!isChanged && item.overridden && <span className="tag-override">已修改</span>}
            {item.scope && <span className="tag-scope">{item.scope}</span>}
          </span>
          {item.description && <span className="field-desc">{item.description}</span>}
          {hint && <span className="field-hint">{hint}</span>}
          {noSuggestions && (
            <span className="field-note">未获取到候选列表，可直接手动输入模型名</span>
          )}
        </div>
        <div className="field-input">{renderControl(item)}</div>
      </div>
    )
  }

  const activeGroup = tree.find((g) => g.name === activeTab)

  return (
    <div className="modal-mask" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        {/* 左侧分类导航 */}
        <nav className="settings-nav">
          <div className="settings-nav-title">系统设置</div>

          <div className="settings-nav-list">
            {tree.map((group) => {
              const groupChanged = group.master
                .concat(group.sections.flatMap((s) => s.items))
                .filter((i) => changed[i.key] !== undefined).length
              const isCurrent = activeTab === group.name
              return (
                <button
                  key={group.name}
                  className={`settings-nav-item ${isCurrent ? 'active' : ''}`}
                  onClick={() => setActiveTab(group.name)}
                >
                  <span>{group.name}</span>
                  {groupChanged > 0 && <span className="nav-dirty-dot" title={`${groupChanged} 项待保存`} />}
                </button>
              )
            })}

            <button
              className={`settings-nav-item ${activeTab === READONLY_GROUP ? 'active' : ''}`}
              onClick={() => setActiveTab(READONLY_GROUP)}
            >
              <span>{READONLY_GROUP}</span>
            </button>
          </div>

          <div className="settings-nav-foot">
            <button className="btn ghost tiny" onClick={() => void resetAll()} disabled={saving}>
              还原全部默认
            </button>
          </div>
        </nav>

        {/* 右侧设置内容区 */}
        <div className="settings-main">
          <div className="settings-main-head">
            <div>
              <h2 className="settings-group-heading">
                {activeGroup ? activeGroup.name : READONLY_GROUP}
              </h2>
              <p className="settings-group-sub">
                {activeGroup ? activeGroup.summary : READONLY_HINT}
              </p>
            </div>
            <button className="icon-btn" onClick={onClose} title="关闭 (Esc)">
              ×
            </button>
          </div>

          <div className="settings-main-body">
            {activeGroup ? (
              <>
                {activeGroup.master.map((item) => {
                  const hint = dependencyHint(item, byKey)
                  return (
                    <div className="settings-surface-block master-block" key={item.key}>
                      <div className="field">
                        <div className="field-label">
                          <span className="field-name">
                            {item.label}
                            {changed[item.key] !== undefined && (
                              <span className="tag-dirty">待保存</span>
                            )}
                          </span>
                          {item.description && (
                            <span className="field-desc">{item.description}</span>
                          )}
                          {hint && <span className="field-hint">{hint}</span>}
                        </div>
                        <div className="field-input">{renderControl(item)}</div>
                      </div>
                    </div>
                  )
                })}

                {activeGroup.sections.map((section) => {
                  const activeCount = section.items.filter((i) => isActive(i, draft)).length
                  const sectionInactive = activeCount === 0

                  return (
                    <div
                      className={`settings-section-group ${sectionInactive ? 'section-inactive' : ''}`}
                      key={section.name}
                    >
                      <div className="section-head">
                        <span className="section-name">{section.name}</span>
                        {sectionInactive && <span className="badge-inactive">当前未启用</span>}
                      </div>
                      <div className="settings-surface-block">
                        {section.items.map(renderField)}
                      </div>
                    </div>
                  )
                })}
              </>
            ) : (
              <div className="settings-surface-block">
                <div className="readonly-grid">
                  {Object.entries(settings.readonly).map(([k, v]) => (
                    <div className="readonly-row" key={k}>
                      <span>{k}</span>
                      <code>{String(v)}</code>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="settings-main-foot">
            <span className="muted">
              {changedCount > 0 ? `已修改 ${changedCount} 项配置` : '修改后保存即刻生效，无需重启'}
            </span>
            <div className="foot-actions">
              <button className="btn ghost" onClick={onClose} disabled={saving}>
                取消
              </button>
              <button
                className="btn primary"
                onClick={() => void save()}
                disabled={saving || changedCount === 0}
              >
                {saving ? '保存中…' : '保存更改'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
