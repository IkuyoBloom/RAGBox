/**
 * 设置面板的层级树构建与依赖判定（纯逻辑，不含任何 React 依赖）。
 *
 * 单独抽出来的理由有两个：
 *  1. 这部分逻辑决定「哪一项当前真的会生效」，是设置界面的核心，值得独立验证；
 *  2. 纯函数可以直接用 Node 跑断言，不需要引入 jsdom / vitest 之类的测试栈 ——
 *     本项目多一层构建依赖就多一处会坏的地方。
 *
 * 顺带一个约束：本文件**只能**用 `import type` 引类型，不能引运行时值，
 * 否则 `node --experimental-strip-types` 会因为解析不了无扩展名的 './types' 而失败。
 */

import type { SettingDependency, SettingItem, SettingsOut } from './types'

/** 三级树的一个小节（二级分组）。 */
export interface SectionNode {
  name: string
  items: SettingItem[]
}

/** 三级树的一级分组。 */
export interface GroupNode {
  name: string
  summary: string
  /** 主控项（`section` 为空的项）。界面把它们置顶并渲染成醒目的卡片。 */
  master: SettingItem[]
  sections: SectionNode[]
  total: number
}

/**
 * 把后端返回的扁平 items 组装成「分组 > 小节 > 字段」三级树。
 *
 * 顺序完全听后端的：`groups[].order` 决定一级分组次序，小节按首次出现次序排列。
 * 这样后端调整 SPECS 的顺序或归类，界面自动跟着变 ——
 * 前端不维护任何「哪个键属于哪一组」的知识，从根上杜绝前后端清单不一致。
 */
export function buildTree(settings: SettingsOut): GroupNode[] {
  const byGroup = new Map<string, SettingItem[]>()
  for (const item of settings.items) {
    const list = byGroup.get(item.group)
    if (list) list.push(item)
    else byGroup.set(item.group, [item])
  }

  const declared = [...settings.groups]
    .sort((a, b) => a.order - b.order)
    .map((g) => g.name)
  // 兜底：items 里出现了 groups 未声明的分组，也要显示出来而不是静默丢弃。
  // 后端漏配 GROUPS 时，界面表现为「多出一组」而不是「少了几项」——
  // 前者一眼能发现，后者会让人以为配置项丢了。
  for (const name of byGroup.keys()) {
    if (!declared.includes(name)) declared.push(name)
  }

  return declared
    .filter((name) => byGroup.has(name))
    .map((name) => {
      const items = byGroup.get(name) ?? []
      const master = items.filter((i) => !i.section)

      const sections: SectionNode[] = []
      for (const item of items) {
        if (!item.section) continue
        let section = sections.find((s) => s.name === item.section)
        if (!section) {
          section = { name: item.section, items: [] }
          sections.push(section)
        }
        section.items.push(item)
      }

      return {
        name,
        summary: settings.groups.find((g) => g.name === name)?.summary ?? '',
        master,
        sections,
        total: items.length,
      }
    })
}

/**
 * 依赖是否满足 —— 决定这一项当前是否真的会生效。
 *
 * 比较统一转成字符串：布尔依赖在 spec 里写成 `("true",)`，
 * 而草稿里的值是真正的 `true`，不转字符串就永远匹配不上。
 */
export function isActive(item: SettingItem, draft: Record<string, unknown>): boolean {
  const dep = item.depends_on
  if (!dep) return true
  return dep.values.includes(String(draft[dep.key]))
}

/** 把依赖关系翻译成人话，例如「需先开启『启用混合检索』」。 */
export function dependencyHint(
  item: SettingItem,
  byKey: Map<string, SettingItem>,
): string | null {
  const dep: SettingDependency | null = item.depends_on
  if (!dep) return null

  const depItem = byKey.get(dep.key)
  const label = depItem?.label ?? dep.key

  if (depItem?.type === 'bool') {
    return `需先${dep.values.includes('true') ? '开启' : '关闭'}「${label}」`
  }
  const names = dep.values.map((v) => depItem?.choice_labels?.[v] ?? v)
  return `需先将「${label}」设为 ${names.join(' 或 ')}`
}

/** 某个分组下当前「实际生效」的字段数 / 总字段数。用于分组标题上的状态提示。 */
export function groupActivation(
  group: GroupNode,
  draft: Record<string, unknown>,
): { active: number; total: number } {
  const all = group.master.concat(group.sections.flatMap((s) => s.items))
  return {
    active: all.filter((i) => isActive(i, draft)).length,
    total: all.length,
  }
}

/** 用当前有效配置构造一份 draft（密钥留空表示「不修改」）。 */
export function draftFrom(items: SettingItem[]): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const item of items) {
    out[item.key] = item.secret ? '' : item.value
  }
  return out
}

/** 只包含真正需要提交的改动。密钥留空不算改动。 */
export function computeChanged(
  items: SettingItem[],
  draft: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const item of items) {
    const value = draft[item.key]
    if (item.secret) {
      if (typeof value === 'string' && value.trim()) out[item.key] = value
      continue
    }
    if (value !== undefined && value !== item.value) out[item.key] = value
  }
  return out
}
