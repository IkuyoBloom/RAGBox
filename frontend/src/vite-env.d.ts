/// <reference types="vite/client" />

// 让 TypeScript 认识 `import './styles.css'` 这类副作用导入。
// 没有这个引用文件，tsc 会报 TS2307 "Cannot find module './styles.css'"。
