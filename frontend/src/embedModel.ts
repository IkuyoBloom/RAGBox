/**
 * 向量模型的比较与状态判定（纯逻辑，不含 React 依赖）。
 *
 * 背景：向量模型在**建库时冻结**在知识库记录里，之后改全局设置不影响已有库。
 * 这个设计是对的（换模型等于换语义坐标系，旧向量会失效），但对用户来说
 * 「哪个库用的是哪个模型」就成了必须**看得见**的信息 ——
 * 否则改了设置以后，用户根本不知道自己新传的文档会落到哪个语义空间里。
 *
 * 本模块负责把「知识库的模型」与「当前设置 + Ollama 实际安装情况」比对成
 * 一个人能读懂的状态，供侧边栏渲染徽标。
 */

import type { KnowledgeBase } from './types'

/** 当前向量模型设置 + Ollama 中实际可用的向量模型。 */
export interface EmbedInfo {
  /** 设置里当前的默认向量模型 —— 新建知识库会用它 */
  current: string
  /** Ollama 中已安装的向量模型名 */
  installed: string[]
  /**
   * 候选列表是否真的取到了。
   * Ollama 未启动时为 false，此时**不能**断言某个模型「未安装」——
   * 那会把每个知识库都误标成红色。
   */
  loaded: boolean
}

export type KbModelState = 'ok' | 'differs' | 'pending' | 'missing' | 'unknown'

export interface KbModelBadge {
  state: KbModelState
  /** 徽标上显示的文字 */
  label: string
  /** 悬停提示：解释这个状态意味着什么、该怎么办 */
  hint: string
}

/**
 * 把模型名补全成带 tag 的形式。
 *
 * Ollama 把 `bge-m3` 与 `bge-m3:latest` 视为同一个模型（省略 tag 时默认 latest），
 * 而 .env 里通常写的是省略形式、`/api/tags` 返回的是完整形式。
 * 不归一化就会把一个模型误判成两个，进而误报「模型不一致」。
 */
export function normalizeModel(name: string): string {
  const trimmed = (name || '').trim()
  return trimmed.includes(':') ? trimmed : `${trimmed}:latest`
}

export function sameModel(a: string, b: string): boolean {
  return normalizeModel(a) === normalizeModel(b)
}

/**
 * 判定某个知识库的模型状态。
 *
 * 判定顺序有讲究：**先看装没装，再看有没有索引，最后才比是否与当前设置一致**。
 * 因为「模型没装」会导致这个库彻底不可用（上传、重建索引、检索全失败），
 * 是最该被看见的一条；而「与当前设置不同」其实是完全正常的。
 */
export function describeKbModel(kb: KnowledgeBase, info: EmbedInfo): KbModelBadge {
  const model = kb.embed_model

  if (!info.loaded) {
    return {
      state: 'unknown',
      label: model,
      hint: `该库使用 ${model}\n（未能获取 Ollama 的模型列表，无法确认它是否已安装）`,
    }
  }

  const installed = info.installed.some((name) => sameModel(name, model))
  if (!installed) {
    return {
      state: 'missing',
      label: model,
      hint:
        `⚠ 该库使用的向量模型「${model}」不在 Ollama 中。\n` +
        `上传文档、重建索引、检索都会失败。\n` +
        `修复：执行 ollama pull ${model}，或另建一个用其他模型的知识库并重新上传。`,
    }
  }

  if (kb.chunk_count === 0) {
    return {
      state: 'pending',
      label: model,
      hint: `该库将使用 ${model}（建库时确定，目前还没有已索引的内容）`,
    }
  }

  if (info.current && !sameModel(info.current, model)) {
    return {
      state: 'differs',
      label: model,
      hint:
        `该库使用 ${model}，与当前设置里的「${info.current}」不同。\n` +
        `这是正常的：向量模型在建库时就冻结了，已有库不会随设置改变。\n` +
        `新知识库将使用 ${info.current}。`,
    }
  }

  return { state: 'ok', label: model, hint: `该库使用 ${model}，与当前设置一致` }
}

/** 状态对应的中文短标签，用于徽标旁的小字。 */
export const KB_MODEL_STATE_TEXT: Record<KbModelState, string> = {
  ok: '',
  differs: '≠当前设置',
  pending: '待索引',
  missing: '未安装',
  unknown: '',
}
