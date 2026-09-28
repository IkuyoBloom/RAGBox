import { useEffect, useState } from 'react'
import * as api from '../api'
import type { Chunk } from '../types'

interface Props {
  chunkId: string
  onClose: () => void
}

/**
 * 引用原文弹窗 —— 「引用溯源」的最后一公里。
 * 前端拿到的 citation 只有 400 字预览，点开才去后端取完整切片，
 * 这样既保证了引用的可核查性，又不会把大段正文塞进每条消息的响应里。
 */
export default function ChunkModal({ chunkId, onClose }: Props) {
  const [chunk, setChunk] = useState<Chunk | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    api
      .getChunk(chunkId)
      .then((c) => {
        if (!cancelled) setChunk(c)
      })
      .catch((e) => {
        if (!cancelled) setError((e as Error).message)
      })
    return () => {
      cancelled = true
    }
  }, [chunkId])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="modal-mask" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="modal-title">{chunk?.filename ?? '加载中…'}</div>
            {chunk && (
              <div className="modal-sub">
                第 {chunk.ordinal + 1} 片
                {chunk.page != null && <> · 第 {chunk.page} 页</>} · {chunk.char_count} 字
              </div>
            )}
          </div>
          <button className="icon-btn" onClick={onClose} title="关闭（Esc）">
            ×
          </button>
        </div>

        <div className="modal-body">
          {error && <div className="bubble-error">⚠ {error}</div>}
          {!chunk && !error && <div className="thinking"><span className="spinner" />加载中…</div>}
          {chunk && <pre className="chunk-text">{chunk.content}</pre>}
        </div>

        <div className="modal-foot">
          <code className="chunk-id">chunk_id: {chunkId}</code>
        </div>
      </div>
    </div>
  )
}
