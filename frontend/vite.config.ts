import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 前端只认 /api，开发期由 Vite 代理到后端。
// 这样生产环境把 dist 交给 FastAPI 托管时，路径完全一致，不用改代码。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // SSE 长连接：关掉代理侧的缓冲，否则 token 会攒着一起吐出来，
        // 「流式输出」就退化成了「等 20 秒后整段出现」。
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            if (String(proxyRes.headers['content-type'] || '').includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache, no-transform'
              proxyRes.headers['x-accel-buffering'] = 'no'
            }
          })
        },
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
