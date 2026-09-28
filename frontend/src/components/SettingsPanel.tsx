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

const READONLY_GROUP = '只读环境信息'
const READONLY_HINT = '由 .env 与运行环境决定，界面不可修改'

export default function SettingsPanel({ settings, onClose, onSaved, notify }: Props) {
  const [draft, setDraft] = useState<Record<string, unknown>>({})
  const [saving, setSaving] = useState(false)
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({ [READONLY_GROUP]: true })

  // 用 effect 而不是 useState 初始化：保存/还原之后 settings 会换新对象，
  // 草稿必须跟着重置。只在「外部刷新」时才需要担心覆盖用户编辑，
  // 而本项目的设置只会在保存与还原这两个动作后重新拉取。
  useEffect(() => {
    setDraft(draftFrom(settings.items))
  }, [settings])

  const tree = useMemo<GroupNode[]>(() => buildTree(settings), [settings])
  const byKey = useMemo(
    () => new Map(settings.items.map((i) => [i.key, i])),
    [settings.items],
  )

  /** 只包含真正需要提交的改动。密钥留空表示「不修改」，不参与提交。 */
  const changed = useMemo(
    () => computeChanged(settings.items, draft),
    [draft, settings.items],
  )

  const changedCount = Object.keys(changed).length
  const overriddenCount = settings.items.filter((i) => i.overridden).length
  const allCollapsed =
    tree.every((g) => collapsed[g.name]) && collapsed[READONLY_GROUP] === true

  const toggleGroup = (name: string) =>
    setCollapsed((c) => ({ ...c, [name]: !c[name] }))

  const toggleAll = () => {
    const next: Record<string, boolean> = { [READONLY_GROUP]: true }
    for (const g of tree) next[g.name] = !allCollapsed
    setCollapsed(next)
  }

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

  // ------------------------------------------------------------------
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
          <span className="switch-text">{value ? '已开启' : '已关闭'}</span>
        </label>
      )
    }

    // 动态来源（如 Ollama 已装模型列表）用 ModelPicker 而不是原生 datalist。
    // datalist 会按输入框当前值过滤候选 —— 填着 bge-m3 时点开只看得到 bge-m3，
    // 另一个候选仿佛不存在。详见该组件内的说明。
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
    // 建议列表为空通常意味着 Ollama 没启动 —— 明确告诉用户「可以手填」，
    // 否则一个空的下拉/无提示的输入框会让人以为功能坏了。
    const noSuggestions = Boolean(item.choices_source) && !item.choices?.length

    return (
      <div className={`field ${active ? '' : 'field-inactive'}`} key={item.key}>
        <div className="field-label">
          <span className="field-name">
            {item.label}
            {isChanged && <span className="tag-dirty">未保存</span>}
            {!isChanged && item.overridden && <span className="tag-override">已覆盖默认</span>}
            {item.scope && <span className="tag-scope">{item.scope}</span>}
          </span>
          {item.description && <span className="field-desc">{item.description}</span>}
          {hint && <span className="field-hint">⚠ {hint}</span>}
          {noSuggestions && (
            <span className="field-note">
              未获取到候选列表（Ollama 未启动？），可直接手动输入模型名
            </span>
          )}
        </div>
        <div className="field-input">{renderControl(item)}</div>
      </div>
    )
  }

  return (
    <div className="modal-mask" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="modal-title">系统设置</div>
            <div className="modal-sub">
              修改后立即生效，写入数据库覆盖层，可一键还原为 .env 默认值
            </div>
          </div>
          <button className="icon-btn" onClick={onClose} title="关闭">
            ×
          </button>
        </div>

        {/* 概览条：进面板第一眼就该知道「有多少项、我改了几项」 */}
        <div className="settings-overview">
          <span>
            共 <b>{settings.items.length}</b> 项可调
          </span>
          <span className={changedCount ? 'ov-dirty' : ''}>
            本次改动 <b>{changedCount}</b> 项
          </span>
          <span className={overriddenCount ? 'ov-override' : ''}>
            已覆盖默认 <b>{overriddenCount}</b> 项
          </span>
          <button className="link-btn ov-toggle" onClick={toggleAll}>
            {allCollapsed ? '展开全部' : '收起全部'}
          </button>
        </div>

        <div className="drawer-body">
          {tree.map((group, gi) => {
            const isCollapsed = Boolean(collapsed[group.name])
            const groupChanged = group.master
              .concat(group.sections.flatMap((s) => s.items))
              .filter((i) => changed[i.key] !== undefined).length

            return (
              <section className="settings-group" key={group.name}>
                <button className="group-head" onClick={() => toggleGroup(group.name)}>
                  <span className={`group-index ${groupChanged ? 'dirty' : ''}`}>{gi + 1}</span>
                  <span className="group-title">
                    <span className="group-name">{group.name}</span>
                    {group.summary && <span className="group-summary">{group.summary}</span>}
                  </span>
                  {groupChanged > 0 && <span className="badge-dirty">{groupChanged} 项待保存</span>}
                  <span className="group-count">{group.total} 项</span>
                  <span className="chev">{isCollapsed ? '▸' : '▾'}</span>
                </button>

                {!isCollapsed && (
                  <div className="group-body">
                    {/* 主控项：本组的「总开关」，其余字段的生效与否由它决定 */}
                    {group.master.map((item) => {
                      const hint = dependencyHint(item, byKey)
                      return (
                        <div className="master-card" key={item.key}>
                          <div className="master-label">
                            <span className="field-name">
                              {item.label}
                              {changed[item.key] !== undefined && (
                                <span className="tag-dirty">未保存</span>
                              )}
                            </span>
                            {item.description && (
                              <span className="field-desc">{item.description}</span>
                            )}
                            {hint && <span className="field-hint">⚠ {hint}</span>}
                          </div>
                          <div className="master-input">{renderControl(item)}</div>
                        </div>
                      )
                    })}

                    {group.sections.map((section) => {
                      const activeCount = section.items.filter((i) => isActive(i, draft)).length
                      const sectionInactive = activeCount === 0
                      const secChanged = section.items.filter(
                        (i) => changed[i.key] !== undefined,
                      ).length

                      return (
                        <div
                          className={`settings-section ${sectionInactive ? 'section-inactive' : ''}`}
                          key={section.name}
                        >
                          <div className="section-head">
                            <span className="section-name">{section.name}</span>
                            {sectionInactive && (
                              <span className="badge-inactive">当前未启用</span>
                            )}
                            {secChanged > 0 && <span className="badge-dirty">{secChanged}</span>}
                          </div>
                          <div className="section-body">
                            {section.items.map(renderField)}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </section>
            )
          })}

          {/* 只读信息：与可调项用同一套视觉语言，但默认折叠、样式更弱 */}
          <section className="settings-group group-readonly">
            <button className="group-head" onClick={() => toggleGroup(READONLY_GROUP)}>
              <span className="group-index readonly">i</span>
              <span className="group-title">
                <span className="group-name">{READONLY_GROUP}</span>
                <span className="group-summary">{READONLY_HINT}</span>
              </span>
              <span className="group-count">{Object.keys(settings.readonly).length} 项</span>
              <span className="chev">{collapsed[READONLY_GROUP] ? '▸' : '▾'}</span>
            </button>
            {!collapsed[READONLY_GROUP] && (
              <div className="group-body">
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
          </section>
        </div>

        <div className="modal-foot spread">
          <button className="btn ghost" onClick={() => void resetAll()} disabled={saving}>
            还原默认值
          </button>
          <div className="foot-actions">
            <span className="muted">
              {changedCount > 0 ? `${changedCount} 项待保存` : '没有未保存的改动'}
            </span>
            <button className="btn" onClick={onClose} disabled={saving}>
              取消
            </button>
            <button
              className="btn primary"
              onClick={() => void save()}
              disabled={saving || changedCount === 0}
            >
              {saving ? '保存中…' : '保存'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
