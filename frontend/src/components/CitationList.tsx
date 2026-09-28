import type { Citation } from '../types'

interface Props {
  citations: Citation[]
  onOpenChunk: (chunkId: string) => void
}

const SOURCE_LABEL: Record<string, string> = {
  both: '双路命中',
  dense: '向量命中',
  sparse: 'BM25 命中',
  none: '未知',
}

const SOURCE_CLASS: Record<string, string> = {
  both: 'src-both',
  dense: 'src-dense',
  sparse: 'src-sparse',
  none: '',
}

/**
 * 引用卡片列表。
 *
 * 特意把「哪条检索路径命中的」显示出来 —— 这既方便调试，
 * 也是答辩时展示混合检索价值最直观的证据：只看向量会漏、只看 BM25 也会漏，
 * 标着「双路命中」的那些片段才是 RRF 真正起作用的地方。
 */
export default function CitationList({ citations, onOpenChunk }: Props) {
  if (!citations.length) return null

  return (
    <div className="citations">
      <div className="citations-head">
        参考来源 <span className="muted">（点击查看原文片段）</span>
      </div>
      <div className="citation-grid">
        {citations.map((c) => (
          <button
            key={c.chunk_id}
            className="citation-card"
            onClick={() => onOpenChunk(c.chunk_id)}
            title={c.preview}
          >
            <div className="citation-top">
              <span className="citation-index">[{c.index}]</span>
              <span className="citation-file">{c.filename}</span>
              {c.page != null && <span className="citation-page">第 {c.page} 页</span>}
            </div>
            <div className="citation-preview">{c.preview}</div>
            <div className="citation-foot">
              <span className={`src-tag ${SOURCE_CLASS[c.source] ?? ''}`}>
                {SOURCE_LABEL[c.source] ?? c.source}
              </span>
              <span className="muted">
                {c.dense_rank != null && <>向量 #{c.dense_rank} </>}
                {c.sparse_rank != null && <>· BM25 #{c.sparse_rank} </>}
                · RRF {c.score.toFixed(4)}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
