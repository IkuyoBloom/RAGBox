# 本地 RAG 知识库问答系统

一个**完全本地部署**的检索增强生成（Retrieval-Augmented Generation）系统。
上传文档即可提问，答案严格依据文档内容生成，并标注 `[n]` 来源角标可回溯到原文。

> 本地知识库问答系统 · Python + FastAPI + React + Ollama + ChromaDB

## ⚡ 快速开始（三条命令）

```powershell
# 1. 装依赖（只需一次）
cd backend;  python -m venv .venv;  .venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple;  Copy-Item .env.example .env;  cd ..
cd frontend; pnpm install; cd ..

# 2. 拉模型（只需一次，约 6 GB）
ollama pull bge-m3
ollama pull qwen2.5:7b-instruct

# 3. 启动：双击「启动服务.lnk」即可（或执行下面这行）
#    会开一个独立控制台窗口，关掉它 = 停止全部服务
python scripts\start.py
```

> **关于启动方式**：启动器本身完全是 Python（`scripts/start.py`）——
> 负责编码、日志、控制台标题、以及用 Windows Job Object 保证「关窗即全停」。
> 仓库根目录的 `启动服务.lnk` 快捷方式直接指向 venv 里的 python，
> **没有任何批处理参与**，双击就能用。`启动服务.cmd` 只是快捷方式丢失时的兜底。

然后浏览器打开 **http://127.0.0.1:5173** 即可。

详细步骤与故障排查见 [第六节 快速开始](#六快速开始)。

### 用 PyCharm 打开

仓库里已经配好 PyCharm 工程（`.idea/`），打开即可用：

* 解释器 `Python 3.12 (RAG)` → `backend\.venv\Scripts\python.exe`
* `backend/` 已标为 **Sources Root**，`from app.xxx import ...` 不会报红
* 运行配置已备好：**启动服务** / 后端热重载 / 全部单测 / 端到端验收 / 环境体检

解释器的**全局注册**需要跑一次脚本（PyCharm 要先关掉，否则改动会被它覆盖）：

```powershell
backend\.venv\Scripts\python.exe scripts\setup_pycharm.py --check   # 看状态
backend\.venv\Scripts\python.exe scripts\setup_pycharm.py           # 注册
```

没跑这一步也能用 —— 打开项目后 PyCharm 会提示解释器无效，
点一下选中 `backend\.venv` 即可。

---

## 一、这个系统解决什么问题

大语言模型存在三个固有缺陷：**知识截止**（训练后无法感知新信息）、
**幻觉**（缺失知识时仍自信编造）、**无法溯源**（答案没有出处）。

本系统把「记忆」从模型参数中外置到可检索的本地知识库：
先在你的文档里检索相关内容，再让模型**仅依据检索到的片段**作答，
并要求标注来源编号 —— 从而让答案可核查、可追溯、可离线运行。

---

## 二、核心特性

| 特性 | 说明 |
|---|---|
| **混合检索** | BM25 稀疏检索 + 向量稠密检索，用 RRF（倒数排名融合）合并，规避两种方法的互补缺陷 |
| **引用溯源** | 答案中的 `[n]` 角标可点击，查看原文切片、来源文件与页码；编号越界会被标红（识别模型编造来源） |
| **Markdown 渲染** | 回答按 Markdown 渲染（标题 / 列表含嵌套 / 加粗 / 行内代码与代码块 / 引用块 / 分隔线 / 表格）。解析器是**手写的纯函数**，渲染层全程构造 React 元素、不用 `dangerouslySetInnerHTML`，因此 XSS 在结构上不可能发生；引用角标通过回调注入，与 Markdown 渲染解耦 |
| **双通道生成** | 本地 Ollama（纯离线）与任意 OpenAI 兼容接口（云端 / 本地 vLLM）可随时切换，便于对比实验 |
| **多轮查询改写** | 「那它的缺点呢？」这类依赖上下文的追问，自动改写为可独立检索的完整问题 |
| **中文友好切分** | 递归字符切分，分隔符按中文标点优先级排列；段落作硬边界后再打包，避免碎块 |
| **多格式解析** | PDF / DOCX / Markdown / TXT，PDF 保留页码，Word 表格转文本行 |
| **文件夹批量导入** | 可以直接选一个文件夹（含子目录），自动识别其中的可用文档并分批导入。临时文件（`~$xx.docx`）、系统文件（`Thumbs.db`）、依赖目录（`node_modules`）自动忽略；不支持格式会明确告知。**保留相对路径**，因此不同子目录下的同名文件不会混淆 |
| **批量导入不卡死** | 入库有**并发上限**（2），解析/切分/向量库读写的同步操作全部走线程池，慢查询不再拖死整个服务。进程意外退出时**启动自动恢复**未完成的文档并重新排队 |
| **内容去重** | 基于 SHA-256 的同库秒传，避免同一份资料被检索到两次 |
| **运行时配置** | 界面可调的配置写入数据库覆盖层，改完立即生效，无需重启，可一键还原。设置页采用**三级层级**（分组 → 小节 → 字段）并做**依赖感知**：当前通道用不到的项自动降权并提示原因 |
| **模型可切换** | 对话模型与**向量模型**都能在界面上换。选项从 Ollama **已装的模型**动态拉取（按 `capabilities` 区分对话/向量），同时允许自由输入任意模型名 |
| **向量模型可见性** | 侧边栏实时显示「当前设置用什么向量模型」，每个知识库各自标注自己实际用的模型。四种状态一眼可辨：一致 / 与当前设置不同 / 待索引 / **模型未安装**（后者会导致该库彻底不可用，红色告警并给出修复命令） |
| **模型隔离** | 向量模型在**建库时冻结**在知识库记录里，索引与检索都按该库自己的模型走 —— 不同库用不同模型完全安全，不会混用。侧边栏的徽标让这件事对用户可见 |
| **检索消融** | `hybrid` / `dense_only` / `sparse_only` 三种模式均可运行，实验与线上行为出自同一份代码 |
| **评估模块** | 检索指标（Hit Rate / Recall / MRR / NDCG）+ 生成指标（忠实度 / 引用覆盖率） |

---

## 三、系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        前端 (React + Vite)                        │
│   知识库管理 · 文档上传 · 流式对话 · 引用卡片 · 设置面板            │
└───────────────────────────────┬──────────────────────────────────┘
                                │ HTTP / SSE
┌───────────────────────────────▼──────────────────────────────────┐
│                      后端 (FastAPI, 15+ 接口)                      │
│                                                                   │
│  ┌───────────── 在线推理链路 ─────────────┐                        │
│  │  查询改写 → 检索 → Prompt 组装 → 流式生成 → 引用落库  │                        │
│  └────────────────────────────────────────┘                        │
│                                                                   │
│  ┌───────────── 离线索引链路 ─────────────┐                        │
│  │  解析 → 切分 → 向量化 → 落库 + 建稀疏索引  │                        │
│  └────────────────────────────────────────┘                        │
│                                                                   │
│  ┌──────────── 检索层 ────────────┐   ┌────── 生成层 ──────┐        │
│  │  BM25 (jieba)  ┐               │   │  Ollama Provider   │        │
│  │                ├─ RRF 融合     │   │  OpenAI 兼容 Provider│      │
│  │  Chroma 向量   ┘               │   └────────────────────┘        │
│  └────────────────────────────────┘                                 │
└───────┬───────────────────────┬───────────────────────┬───────────┘
        │                       │                       │
   ┌────▼────┐            ┌─────▼─────┐          ┌──────▼──────┐
   │ SQLite  │            │  ChromaDB │          │   Ollama    │
   │ 唯一事实 │            │  仅存向量  │          │ bge-m3 向量  │
   │  来源    │            │ + 定位元数据│          │ qwen2.5 生成 │
   └─────────┘            └───────────┘          └─────────────┘
```

### 数据职责划分（关键设计）

**SQLite 是唯一事实来源，ChromaDB 只存向量与最小定位元数据，不存正文。**

这样做的三个好处：

1. 不存在「向量库里的文本」与「数据库里的文本」不一致的问题；
2. 向量检索与 BM25 拿到的是同一批 `chunk_id`，融合时天然对齐，不需要按文本去重；
3. 将来替换向量库（Milvus / Qdrant / pgvector），重放 embedding 即可，数据不丢。

---

## 四、目录结构

```
RAG/
├── backend/                          # 后端
│   ├── app/
│   │   ├── main.py                   # FastAPI 入口 + 生命周期
│   │   ├── config.py                 # 环境配置（出厂默认值）
│   │   ├── schemas.py                # Pydantic 数据契约
│   │   ├── db/
│   │   │   ├── base.py               # 引擎 / 会话 / WAL pragma
│   │   │   └── models.py             # ORM 模型
│   │   ├── api/                      # 路由层
│   │   │   ├── deps.py               # 依赖注入 + 统计聚合
│   │   │   ├── kb.py                 # 知识库 CRUD
│   │   │   ├── documents.py          # 上传 / 索引 / 切片
│   │   │   ├── chat.py               # SSE 流式问答 + 会话
│   │   │   ├── settings.py           # 运行时配置
│   │   │   ├── eval.py               # 评估接口
│   │   │   └── system.py             # 健康检查 / 模型列表
│   │   └── core/
│   │       ├── settings_service.py   # 配置分层（env 默认 + DB 覆盖）
│   │       ├── llm/                  # 生成通道抽象
│   │       │   ├── base.py           # LLMProvider 接口
│   │       │   ├── ollama_provider.py
│   │       │   ├── openai_provider.py
│   │       │   └── factory.py
│   │       ├── embedding/            # 向量化抽象
│   │       ├── vectorstore/          # Chroma 封装
│   │       ├── ingest/               # 解析 / 切分 / 入库流水线
│   │       ├── retrieval/            # BM25 + 混合检索 + RRF
│   │       ├── rag/                  # Prompt / 改写 / 引擎
│   │       └── eval/                 # 评估指标与数据集
│   ├── scripts/
│   │   ├── check_env.py              # 环境自检（依赖/路径/分词/切分/Ollama）
│   │   ├── check_providers.py        # 双通道验证（流式/非流式/错误处理）
│   │   ├── check_embed_switch.py     # 向量模型切换语义（新建库用新模型/旧库不变）
│   │   ├── e2e_test.py               # 端到端验收（35 项断言）
│   │   ├── prepare_demo_kb.py        # 一键铺演示数据
│   │   ├── bench_retrieval.py        # 纯检索耗时隔离实验
│   │   ├── run_eval.py               # 评估入口（支持 LLM 合成评估集）
│   │   └── make_experiment_report.py # 生成实验报告
│   ├── tests/                        # pytest 单元测试（199 个）
│   ├── requirements.txt
│   └── .env.example
├── frontend/                         # 前端 (React + TypeScript)
│   └── src/
│       ├── api.ts                    # API 客户端（含手写 SSE 解析）
│       ├── types.ts                  # 与后端 schemas 对应的类型
│       ├── settingsTree.ts           # 设置页的层级树与依赖判定（纯逻辑）
│       ├── embedModel.ts             # 向量模型比对与知识库模型状态（纯逻辑）
│       ├── App.tsx                   # 状态中枢
│       └── components/               # TopBar / Sidebar / ChatPanel / ModelPicker / ...
├── docs/
│   └── 实验结果.md                    # 自动生成的实验数据表
├── scripts/
│   ├── start.py                      # 一键启动（Python，Ctrl+C 停止全部）
│   ├── check_stream.py               # 验证 SSE 真流式（首字节 vs 总耗时）
│   └── sample_docs/                  # 演示用样例文档（5 份）
└── data/                             # 运行期数据（已 gitignore）
    ├── rag.db                        # SQLite
    ├── chroma/                       # 向量库持久化目录
    ├── uploads/                      # 原始上传文件
    └── eval/                         # 评估报告与合成数据集
```

---

## 五、环境要求

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.12+ | 后端运行时 |
| Node.js | 20+ | 前端构建 |
| pnpm | 9+ | 包管理器（也可用 npm） |
| Ollama | 0.3+ | 本地推理引擎 |
| 显卡 | 显存 ≥ 8 GB | 运行 7B 量化模型（无独显也可用 CPU，但速度很慢） |

### 模型准备

```bash
ollama pull bge-m3                # 向量模型，566M 参数，输出 1024 维
ollama pull qwen2.5:7b-instruct   # 生成模型，Q4_K_M 约 4.7 GB
```

> 若磁盘空间紧张，可用 `OLLAMA_MODELS` 环境变量把模型目录挪到别的盘：
> `setx OLLAMA_MODELS "E:\AI_Models"`（重启终端生效）。

---

## 六、快速开始

### 1. 后端

```powershell
cd backend

# 创建虚拟环境
python -m venv .venv

# 安装依赖（国内建议走清华源）
.venv\Scripts\python.exe -m pip install -r requirements.txt `
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制配置文件
Copy-Item .env.example .env

# 环境自检（检查依赖、路径、分词、切分、Ollama 连通性）
.venv\Scripts\python.exe scripts\check_env.py
```

### 2. 前端

```powershell
cd frontend
pnpm install
```

### 3. 启动

**在项目根目录**执行：

```powershell
python scripts\start.py
```

Ollama、后端、前端会在**同一个终端**里起来，日志带 `[后端]` / `[前端]` 前缀区分：

```
  Ollama 端点: 127.0.0.1:11434  （来自 backend/.env）

  [1/3] Ollama
  ✓ Ollama 已在 127.0.0.1:11434 运行

  [2/3] 启动后端 …
  [3/3] 启动前端 …

[后端] 12:20:37 INFO    app.main | 数据库已就绪：D:\Code\RAG\data\rag.db
[前端]   VITE v6.4.3  ready in 415 ms
  ✓ 后端 就绪
  ✓ 前端 就绪

   前端界面 : http://127.0.0.1:5173
   接口文档 : http://127.0.0.1:8000/docs
   健康检查 : http://127.0.0.1:8000/api/health
   按 Ctrl+C 停止全部服务
```

> **关闭服务**：在这个终端按 `Ctrl+C`，后端与前端会被**整棵进程树**一起收掉
> （`pnpm` → `node` → 子进程都会被终止，不会留下占端口的孤儿进程）。
> Ollama 是常驻服务，不需要每次关。

常用参数：

```powershell
python scripts\start.py --check          # 只做环境体检，不启动
python scripts\start.py --reload         # 后端热重载（改代码自动生效）
python scripts\start.py --backend-only   # 只起后端
python scripts\start.py --frontend-only  # 只起前端
python scripts\start.py --backend-port 8080   # 换端口
```

| 地址 | 用途 |
|---|---|
| **http://127.0.0.1:5173** | **前端界面（从这里开始用）** |
| http://127.0.0.1:8000/docs | Swagger 接口文档 |
| http://127.0.0.1:8000/api/health | 健康检查 |
| http://127.0.0.1:11434 | Ollama（供后端调用，不用手动访问） |

> 启动器会自己从 `backend\.env` 的 `OLLAMA_BASE_URL` 读出 Ollama 端口，
> 所以端口改了不用动脚本。Ollama 没起来时会打印排查步骤。

### 4. 使用流程

1. 左侧「知识库」→ 新建，起个名字（可以顺便指定向量模型，默认跟随当前设置）；
2. 选中该知识库 → 点「**文件**」选单个/多个文件，或点「**文件夹**」选整个目录
   （含子目录，自动识别可用文档并分批导入，带进度条）；
3. 等状态徽标从「向量化中」变成「就绪」（界面会自动轮询，无需手动刷新）；
4. 在右侧输入框提问，答案会带 `[n]` 角标，点击可查看原文切片。

**关于文件夹导入**：点「文件夹」后浏览器会把目录里**所有**文件交给我们，
所以过滤规则值得知道：

| 文件 | 处理 |
|---|---|
| `.pdf` `.docx` `.md` `.markdown` `.txt` `.text` `.log` | 导入 |
| `~$报告.docx`（Office 锁文件）、`.DS_Store`、`Thumbs.db`、隐藏文件 | **静默忽略**（不打扰你） |
| `node_modules/`、`.git/`、`__pycache__/`、`.venv/`、`dist/` 等目录里的文件 | **静默忽略** |
| `.pptx` `.xlsx` `.png` `.zip` 等 | 跳过并**列出扩展名**告诉你 |

一次选几百个文件也没问题：前端会**分批**（每批 8 个）上传并显示进度，
每批完成就刷新一次列表，所以你能看到文档一个个出现而不是干等。
子目录结构会被保留在文档名里（如 `论文/2024/综述.md`），
这样不同目录下的同名文件不会混淆。

### 常见启动故障

| 现象 | 原因与处理 |
|---|---|
| 端口 8000/5173 被占用 | 上次没关干净。`python scripts\start.py --check` 会直接报出占用进程的 PID |
| `ollama list` 报 `timed out waiting for server to start` | 当前终端的 `OLLAMA_HOST` 是旧值。**重开一个终端**即可（`setx` 只对新启动的进程生效） |
| Ollama 启动失败但端口没人占 | Windows 端口保留问题，见第十二节 FAQ |
| 前端能开但接口 502 | 后端没起来。直接访问 http://127.0.0.1:8000/api/health 确认 |

**为什么启动脚本是 Python 而不是 PowerShell**（如果你好奇原来那套为什么没了）：

原先有四个 `.ps1`。它们含中文，而 PowerShell 5.1 读取**无 BOM** 的 `.ps1` 时
按系统 ANSI（中文环境是 GBK）解码，中文被读乱后会吞掉字符串结束引号，
脚本直接语法报错 —— 而用编辑器改一次就可能把 BOM 剥掉，属于会反复复发的坑。
换成 Python 后编码行为在各平台一致，这类问题从根上消失；
顺带还解决了「各开一个窗口、关不干净会留下占端口孤儿进程」的问题。

---

## 七、验证与测试

所有验证脚本都在项目根目录或 `backend` 下，直接跑即可（需先启动服务）：

```powershell
# ---- 后端 ----
cd backend

# 环境自检：依赖 / 路径 / 数据库 / 向量库 / 分词 / 切分 / Ollama
.venv\Scripts\python.exe scripts\check_env.py

# 单元测试（199 个）
.venv\Scripts\python.exe -m pytest tests\ -v

# 双通道验证：本地 Ollama + OpenAI 兼容接口各自的流式/非流式/错误处理
# （默认把 Ollama 自带的 /v1 兼容端点当「本地假云端」，不需要任何 API Key）
.venv\Scripts\python.exe scripts\check_providers.py

# 端到端验收：建库 → 上传 → 索引 → 三种检索模式 → 流式问答
#             → 引用溯源 → 多轮改写 → 配置校验 → 去重 → 清理
.venv\Scripts\python.exe scripts\e2e_test.py

# 向量模型切换语义验证：新建库用新模型 / 已有库保持不变 / 新模型真能索引
.venv\Scripts\python.exe scripts\check_embed_switch.py

# 文件夹导入验证：相对路径保留 / 路径穿越防护 / 同名文件独立成条
.venv\Scripts\python.exe scripts\check_folder_upload.py

# 一键铺演示数据（建库 + 上传样例文档 + 等索引），并打印 KB_ID
.venv\Scripts\python.exe scripts\prepare_demo_kb.py --reset

# 纯检索耗时隔离实验（排除生成干扰，判断延迟到底花在哪一段）
.venv\Scripts\python.exe scripts\bench_retrieval.py --kb-id <知识库ID> --rounds 6

# 评估（消融实验）
.venv\Scripts\python.exe scripts\run_eval.py --kb-id <知识库ID> --dataset sample_retrieval
.venv\Scripts\python.exe scripts\run_eval.py --kb-id <知识库ID> --build-dataset 12 --dataset-name synth_rag

# ---- 项目根目录 ----

# SSE 真流式验证：首字节 vs 总耗时，证明 token 是边生成边推送
# （TestClient 会缓冲整个响应，只有打真服务器才看得出来）
python scripts\check_stream.py --list-kb
python scripts\check_stream.py <知识库ID> --question "BM25 检索有什么缺点？"
```

> 指向真实云端通道时：
> `scripts\check_providers.py --base-url https://api.deepseek.com/v1 --api-key sk-xxx --model deepseek-chat`

前端侧（在 `frontend` 目录下，需后端已运行）：

```powershell
# 设置界面层级验证：三级层级、依赖判定、提示文案、改动收集
pnpm check:settings

# 导出可独立打开的布局预览（不用起服务就能看设置页长什么样）
pnpm check:settings:preview   # -> frontend/preview/settings-panel.html

# 侧边栏：向量模型徽标的状态判定 + 渲染
pnpm check:sidebar
pnpm check:sidebar:preview    # -> frontend/preview/sidebar.html

# 文件夹导入的识别与过滤规则（临时文件/依赖目录/不支持格式）
pnpm check:filepick

# Markdown 解析与渲染（含 XSS 转义验证）
pnpm check:markdown
pnpm check:markdown:preview    # -> frontend/preview/markdown.html

# 上面七项一起跑
pnpm check:all
```

> 这两个脚本不需要浏览器或 jsdom：前者用 Node 自带的 TypeScript 类型剥离
> 直接跑纯逻辑，后者用 vite 的 `ssrLoadModule` 真渲染一遍并断言 DOM 结构。

### 测试覆盖分布

后端 pytest 共 **199 个用例**：

| 测试文件 | 用例数 | 覆盖内容 |
|---|---|---|
| `tests/test_eval_metrics.py` | 77 | 检索/生成指标纯函数、边界值、LLM 裁判降级 |
| `tests/test_core_retrieval.py` | 48 | **中文分词、BM25、RRF 融合、文本归一化、切分** |
| `tests/test_eval_pipeline.py` | 38 | 数据集读写、gold 三档降级、报告序列化、HTTP 契约 |
| `tests/test_upload_naming.py` | 26 | **上传文件名规范化：路径穿越防护、相对路径保留、长度截断** |
| `tests/test_concurrency.py` | 10 | **入库并发闸门、collection 句柄缓存、BM25 构建不重复、跨事件循环不炸** |

`test_concurrency.py` 的每一条都对应「9.0 一次真实的服务卡死」里的一个成因 ——
这类性质用「跑一遍看看快不快」是测不稳的，必须断言具体行为
（比如「8 个线程取同一个 collection 必须拿到同一个句柄」）。

前端另有 5 个脚本（`pnpm check:all`），都不依赖浏览器或 jsdom ——
纯逻辑的用 Node 自带的 TypeScript 类型剥离直接跑，涉及渲染的用 vite 的
`ssrLoadModule` 真渲染一遍再断言 DOM 结构。

`test_core_retrieval.py` 与 `test_upload_naming.py` 里每条用例都对应一个
**真实出现过的 bug 或安全边界**（见「九、关键设计决策」），不是为凑覆盖率而写。

### 实测结果

测试环境：RTX 4060 Laptop **8 GB** / Windows / Python 3.12 /
`bge-m3`（向量，566M）+ `qwen2.5:7b-instruct`（生成，Q4_K_M 约 4.7 GB）。

**端到端验收**：35 项断言全部通过，退出码 0。

| 阶段 | 冷启动（首次调用） | 预热后 |
|---|---|---|
| 文档向量化（5 片） | 30 502 ms | **370 ms** |
| 检索（hybrid，5 片语料） | 8 618 ms | **1 103 ms** |
| 生成（约 90 token 回答） | 59 977 ms | **2 413 ms** |

> 冷启动耗时几乎全部来自**模型加载进显存**。Ollama 默认在最后一次调用后
> 保留模型 5 分钟，因此连续对话时后续请求都是预热后的速度。

### 检索耗时：算法不慢，慢的是显存

用 `scripts/bench_retrieval.py` 做隔离实验（只跑检索、不调生成），
发现跨模式的耗时差异**不是算法造成的**：

| 模式 | 纯检索均值 | 中位数 | 最小 | 最大 | 分阶段（均值） |
|---|---|---|---|---|---|
| `hybrid` | 1478 ms | 1064 ms | 363 ms | 3222 ms | dense 1375 ms + sparse 102 ms |
| `dense_only` | 386 ms | 376 ms | 364 ms | 422 ms | dense 386 ms |
| `sparse_only` | **0 ms** | 0 ms | 0 ms | 1 ms | sparse 0 ms |

三条关键证据：

1. **`sparse` 阶段恒定 0~1 ms**：BM25 是纯内存计算，索引缓存命中后几乎零成本。
   唯一一次 610 ms 出现在首次调用 —— 那是 jieba 词典初始化，一次性开销。
2. **全部波动都在 `dense` 阶段**：它本质是一次 `POST /api/embed` 调用，
   实测在 **360 ms（模型常驻）到 3200 ms（需重新载入显存）** 之间跳变，与检索算法无关。
3. **稳定态下 `hybrid` ≈ `dense_only`**（363 ms vs 386 ms），
   正好符合「hybrid 只多跑一路 0 ms 的 BM25」的预期。

**结论**：8 GB 显存要同时容纳生成模型（4.7 GB）、向量模型（1.1 GB）与 KV cache，
两者会互相换入换出。**在显存受限的设备上，检索延迟由向量模型的驻留状态决定，
而不是由检索算法决定。** 这也解释了为什么评估报告里 `hybrid` 的检索耗时一度虚高到 2136 ms
—— 它排在第一个跑，替后面所有模式付了冷启动的账。

针对这个陷阱，评估模块已在计时前加入**预热**（BM25 建索引 → 生成模型 → 向量模型，
顺序有意为之：让向量模型成为显存里最新的那个），保证各模式站在同一起跑线上。

### 消融实验（真实链路，gold_chunk_ids 严格判定）

数据集 `synth_rag`：用 `--build-dataset 12` 从知识库**反向合成** 12 条问答，
gold 判定使用 `gold_chunk_ids`（最严格的一档）。语料 5 份文档 / 23 个切片，
共 12 × 3 = **36 条样本，零失败**。

| 模式 | 命中率 | 召回率 | 精确率 | **MRR** | **NDCG** | **引用覆盖率** | 引用有效率 | 字面重合度 |
|---|---|---|---|---|---|---|---|---|
| `hybrid` | 1.000 | 1.000 | 0.167 | **0.917** | **0.938** | **0.450** | 1.000 | 0.098 |
| `dense_only` | 1.000 | 1.000 | 0.167 | 0.840 | 0.880 | 0.419 | 1.000 | 0.093 |
| `sparse_only` | 1.000 | 1.000 | 0.167 | **0.917** | **0.938** | 0.382 | 1.000 | 0.103 |

**结论一：融合确实改善了排序质量。**
`hybrid` 的 MRR（0.917）与 NDCG（0.938）均高于 `dense_only`（0.840 / 0.880）。
MRR 衡量首个相关结果的排名，NDCG 衡量整条排序的质量 ——
两者同时提升说明 RRF 不是把两路结果简单拼起来，而是真的把「被两路共同认可」
的片段提到了前面。这正是 RRF 的设计目的。

**结论二：检索质量会传导到引用质量。**
`hybrid` 的引用覆盖率（0.450）也最高。引用覆盖率衡量回答中有多少比例的实义句子
带了 `[n]` 角标 —— 检索到的上下文越切题，模型越有依据可标。

**结论三：三种模式的引用有效率都是 1.000。**
所有 `[n]` 角标都落在实际检索到的片段范围内，**没有出现模型编造来源**。
这验证了「强制标注来源编号」这一提示策略的有效性。

**关于精确率 0.167 的说明（不是缺陷）**：
每个样本只有 1 条 `gold_chunk_id`，而 `final_top_k = 6`，
所以精确率上限就是 1/6 ≈ 0.167，三种模式自然相同。
这是**指标定义与实验参数共同决定的**，不构成模式间差异。
若要让精确率具备区分度，应增大数据集使其包含多 gold 片段，或减小 `final_top_k`。

### 耗时：均值会骗人，必须看分布

评估器最初只输出均值，而均值在存在冷启动离群值时会严重误导：

| 模式 | 阶段 | 均值 ms | **中位数 ms** | P90 ms | 最大 ms | 离群次数 |
|---|---|---|---|---|---|---|
| `hybrid` | 检索 | 2442 | **427** | 6916 | 14847 | **3** |
| `hybrid` | 生成 | 3062 | 2230 | 5480 | 5748 | 0 |
| `dense_only` | 检索 | 379 | **383** | 402 | 434 | 0 |
| `sparse_only` | 检索 | 1 | **1** | 4 | 4 | 2 |

`hybrid` 的检索均值（2442 ms）与中位数（427 ms）**相差 5.7 倍**，
差距全部来自 3 次模型重载的离群样本。

**中位数才反映真实性能**：`hybrid` 427 ms、`dense_only` 383 ms、`sparse_only` 1 ms。
前两者几乎相同，正好符合「hybrid 只比 dense_only 多跑一路耗时约 0 ms 的 BM25」
的理论预期。

> 这一段的方法论教训值得记下来：**报告性能数据时必须给出分布，
> 只报均值等于把冷启动开销平摊到每个样本上，结论会完全反过来**
> （均值会让人以为 hybrid 比 dense_only 慢 6 倍，而实际上两者等价）。

数据表格由 `scripts/make_experiment_report.py` 自动生成，落在 `docs/实验结果.md`。

**查询改写实测**：

```
第 1 轮  问：BM25 检索有什么优点和缺点？
第 2 轮  问：那它的缺点呢？          ← 直接检索必然失效，「它」没有指代对象
         改写为：BM25 检索的缺点有哪些？   ← 指代消解成功
```

---

## 八、配置说明

配置分两层：

- **`.env`（出厂默认值）**：进程启动时读取，定义默认行为；
- **数据库 `app_settings` 表（覆盖层）**：界面上「设置」页写入，立即生效。

优先级：**数据库覆盖 > 环境变量 > 代码默认值**。

### 设置页的结构

设置页是**三级层级**，层级关系完全由后端的 `SettingSpec` 声明
（`group` / `section` / `depends_on` / `scope` 四个字段），前端不维护任何归类知识 ——
后端加一项配置，界面自动归类，不会长出「孤儿字段」。

```
① 生成通道   决定用哪个模型来组织回答
   ★ 生成通道 [主控]  ← 置顶卡片，决定本组其余项是否生效
   ├─ 本地 Ollama
   └─ 云端 / OpenAI 兼容接口     ⓘ 当前未启用时整节降权
② 生成参数   控制回答的风格、长度与可靠性
   ├─ 采样控制 / 上下文与显存 / 回答准则
③ 检索       决定从知识库里捞出哪些片段送给模型
   ★ 启用混合检索 [主控]
   ├─ 双路召回 / 融合与筛选 / 多轮对话
④ 文档入库   决定文档如何被向量化与切分（影响之后新建的知识库）
   ├─ 向量模型        ← 本次新增，可切换
   └─ 切片尺寸
```

两个交互细节值得说明：

- **依赖不满足只降权、不禁用**。比如用着本地 Ollama 时，
  「云端 / OpenAI 兼容接口」整节会降到 58% 不透明度，但输入框**仍可编辑** ——
  允许「先把 Key 填好，再切通道」。鼠标移入或键盘聚焦即恢复满对比度。
- **模型名是可自由输入的**（输入框 + `datalist` 建议），不是严格下拉框。
  因为 Ollama 没启动时候选列表必然是空的，此时若用严格下拉框，
  用户连「换个模型名试试」这个自救动作都做不了。

### 侧边栏：向量模型怎么看

「当前设置用什么模型」和「某个库实际用什么模型」是**两件事**，侧边栏把它们并排显示：

```
┌─ 知识库 ─────────────────────┐
│ 当前向量模型  qwen3-embedding:4b  │ ← 实时读设置，新建库会用它
│                                  │
│ 项目资料                     │
│   3/3 份 · 128 片                │
│   ┌────────────────────────┐     │
│   │ bge-m3   ≠当前设置      │     │ ← 该库实际用的模型（建库时冻结）
│   └────────────────────────┘     │
└──────────────────────────────────┘
```

徽标有四种状态，**颜色加文字**双重区分（只靠颜色对色觉障碍用户不友好）：

| 徽标 | 含义 | 该怎么办 |
|---|---|---|
| 绿色 `bge-m3` | 与当前设置一致 | 无需处理 |
| 黄色 `bge-m3 ≠当前设置` | 该库建库用的模型与现在的设置不同 | **正常现象**。已有库不随设置变化；新库会用当前设置的模型 |
| 灰色 `bge-m3 待索引` | 建库时确定，但还没有已索引的内容 | 上传文档即可 |
| 红色 `bge-m3 未安装` | 该库用的模型不在 Ollama 里 | **必须处理**：上传/重建索引/检索都会失败。执行 `ollama pull <模型名>`，或另建一个用其他模型的库 |

> 判定顺序是**「装没装」优先于「是否与设置一致」** ——
> 前者会让该库彻底不可用，后者完全正常，严重程度不同，不该混在一个颜色里。

鼠标悬停在徽标上会给出完整解释（包括修复命令）。

**「文档」面板可折叠**：点标题栏即可收起，列表长了能腾出空间看知识库列表。

**新建知识库时可以直接选向量模型**（默认跟随当前设置）——
这样你可以在不动全局设置的前提下建一个用其他模型做对照实验的库，用于模型对比实验。
注意：**建库后模型不可更改**，想换只能新建库并重新上传。

### 生效范围：哪些改动不影响已有知识库

带 **«仅对新建知识库生效»** 标签的三项 —— `向量模型`、`切片长度`、`切片重叠` ——
都只在**创建知识库时**写入该库自己的记录，之后改全局设置不会动已有库。

向量模型尤其如此：**建库时就把模型名与维度冻结在知识库记录里**。
换模型等于换一整套语义坐标系，旧向量与新查询不在同一空间，
强行混用会得到毫无意义的相似度。所以系统按**知识库自己的模型**编码查询，
不匹配时报明确错误，绝不静默返回垃圾结果。

> 想换模型的正确做法：改设置 → **新建**一个知识库 → 重新上传文档。
> `scripts/check_embed_switch.py` 就是验证这套语义的脚本
> （实测：新建库用新模型、已有库保持原样、新模型真能索引出 2560 维向量）。

### 主要配置项

| 键 | 默认 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | 生成通道：`ollama` 本地 / `openai` 兼容接口 |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:7b-instruct` | 本地生成模型 |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` | — | 云端通道参数（界面上填，不写进 .env 也行） |
| `OLLAMA_EMBED_MODEL` | `bge-m3` | **默认**向量模型。可在界面「文档入库 → 向量模型」修改；**知识库创建时冻结**，改这里只影响新库 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 80 | 切片长度与重叠 |
| `ENABLE_HYBRID` | `true` | 开 = 混合检索；关 = 纯向量 |
| `DENSE_TOP_K` / `SPARSE_TOP_K` | 20 / 20 | 两路各自召回数量 |
| `FINAL_TOP_K` | 6 | 最终送入模型的片段数 |
| `RRF_K` | 60 | RRF 平滑常数，越大越平滑 |
| `ENABLE_QUERY_REWRITE` | `true` | 多轮追问改写 |
| `STRICT_GROUNDING` | `true` | 严格模式：资料不足时拒答而非编造 |

---

## 九、关键设计决策

以下取舍属于系统设计的核心，值得展开说明。

### 9.0 同步代码不能在协程里跑：一次真实的服务卡死

这是整个项目里**最有价值的一次排障**，因为它推翻了一个几乎所有人都会先入为主的判断。

**现象**：批量上传 111 份文档后，整个服务失去响应 —— 连 `/api/health` 都要
**60 秒**才返回，前端看起来完全卡死。

**第一个直觉（错的）**：Ollama 被 111 个文档的向量化请求压垮了。

**推翻它的证据**：`nvidia-smi` 显示 **GPU 利用率只有 6%**。推理根本没在跑。
后端进程 CPU 176 秒、55 个线程 —— 在烧 CPU，不是在等 GPU。

**真正的原因**（四层，每一层都足以单独造成卡死）：

1. **同步代码直接写在协程里**。`pypdf` 解析、docx 解析、`split_blocks` 切分、
   Chroma 读写，全都是**同步**调用，却出现在 `async def` 里。
   它们不 await，于是把整个事件循环占住 —— 所有 HTTP 请求排队等待。
   修法：`await asyncio.to_thread(...)` 挪到线程池。

2. **每次读写都重建 collection 句柄**。`get_or_create_collection` 会读一遍
   Chroma 内部的 SQLite 元数据表；原先每做一次 add/query/delete 都调一次。
   几十个入库任务反复打同一张表，锁争用被放大到整个服务。
   修法：缓存句柄，并用一把写锁把向量库写入串行化（嵌入式 SQLite 本就不适合并发写）。

3. **没有并发上限**。111 个文档 = 111 个后台任务同时开跑，一起抢 Ollama、
   抢向量库、抢事件循环。修法：`asyncio.Semaphore(2)`。
   取 2 是刻意的 —— 本机只有一个 Ollama 实例，再高的并发也只是互相排队，
   反而让状态更难推进、日志更难读。

4. **进程退出后状态永远停在中间态**。后台任务随进程消失，但数据库里的
   `status` 停在 `PARSING` / `EMBEDDING`。界面上表现为文档永远「向量化中」，
   用户既不能重试也不知道为什么。修法：启动时 `requeue_unfinished()`
   把这些文档捞回来重新排队。

**修复后的实测**：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| `/api/kb` | 60 092 ms | **18 ms** |
| `/api/health` | 60 016 ms | 1 837 ms |
| 根路径 | 不响应 | 666 ms |
| 92 份卡住的文档 | 永远卡住 | **75 秒全部跑完，0 失败**，期间服务全程可响应 |

**教训**：性能问题的第一直觉往往是「算得太慢」，但真正致命的是
**「把异步当同步用」**。判断依据不能靠感觉，要看资源利用率 ——
**GPU 6% 这个数字，比任何日志都更快地指向了真凶**。

### 9.1 为什么融合用 RRF 而不是加权求和
稠密检索的分数是余弦相似度（0~1 有界），BM25 是无上界的对数加权和。
二者**量纲不同、分布不同**，直接 `α·dense + β·sparse` 需要反复调参，
换一个 embedding 模型就得重调。

RRF 只看**名次**不看分数：`score(d) = Σ_r weight_r / (k + rank_r(d))`。
它天然免疫量纲差异，换模型、换语料都不用重新调参，是工业界的默认基线。

### 9.2 为什么 SQLite 是唯一事实来源

详见「三、系统架构」中的数据职责划分。核心是避免双写不一致，
并让两路检索的结果在 `chunk_id` 层面对齐。

### 9.3 为什么向量模型在知识库级别冻结

知识库一旦建好，向量空间就被模型固定了 —— 换模型等于换语义坐标系，
旧向量全部作废。因此建库时把 `embed_model` 与 `embed_dim` 写入知识库记录，
查询时按**知识库自己的模型**编码，而不是用全局当前设置。
不匹配时报明确错误，绝不静默返回垃圾结果。

### 9.4 为什么入库流水线要拆事务

朴素写法是把「解析 → 切分 → 向量化 → 落库」包在一个事务里，
但向量化几百个切片要几秒到几十秒，SQLite 会在这期间一直持有写锁。

因此拆成：**短读 → 无事务长计算 → 短写 → 事务外写向量库**。
向量库写入放在数据库提交之后，保证「有文本没向量」是可恢复状态
（重建索引即可），而不会出现「有向量没文本」这种检索时取不到正文的脏状态。

### 9.5 为什么引擎自己管理数据库会话

对话接口返回 SSE 流式响应，FastAPI 的依赖注入在**响应结束后**才关闭会话。
而一次生成要几十秒，期间若还持有 SQLite 锁，其他请求会被拖住。

引擎因此采用「短会话」模式，每步用完即关。
副作用是引擎完全不依赖请求上下文，**评估模块和离线脚本可以直接复用同一份代码** ——
这保证了实验数据与系统在线行为出自同一条实现。

### 9.6 中文切分的两个坑

1. **BM25 分词**：中文无空格，必须先分词。用 jieba 搜索引擎模式
   （`lcut_for_search`），在长词外额外吐出子词（神经网络 → 神经 / 网络），
   提升召回。索引与查询两侧用同一套切法保证词表一致。
   注意：过滤正则必须是「整段保留中文片段」，不能写成 `[\u4e00-\u9fff]` 再用
   `finditer` —— 那会把切好的「梯度下降法」逐字拆开，等于白分词。
2. **PDF 硬换行**：PDF 里每个视觉行都是一个换行符，直接切会把句子切碎。
   需要按「上一行是否以句末标点结尾」判断真换行，中文直接拼接、中英之间补空格。

### 9.7 引用编号的越界检测

系统要求模型标注 `[n]`，但小模型有时会编造不存在的编号。
前端在渲染时把 `[n]` 与本次检索到的引用列表比对，**越界编号标红提示**
—— 这是「模型编造来源」最直观的可见信号，也是评估「引用有效性」的依据。

---

## 十、接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 健康检查，含两条生成通道的探测结果 |
| `GET` | `/api/models` | 列出各通道可用模型 |
| `GET/POST` | `/api/kb` | 知识库列表 / 新建 |
| `GET/PATCH/DELETE` | `/api/kb/{kb_id}` | 详情（含文档）/ 改名 / 删除 |
| `POST` | `/api/kb/{kb_id}/documents` | 上传文档（多文件，后台异步建索引） |
| `GET/DELETE` | `/api/documents/{doc_id}` | 文档详情（含索引进度）/ 删除 |
| `POST` | `/api/documents/{doc_id}/reindex` | 重建索引 |
| `GET` | `/api/documents/{doc_id}/chunks` | 查看切片 |
| `GET` | `/api/chunks/{chunk_id}` | 读取单个切片正文（引用溯源） |
| `POST` | `/api/chat/stream` | **流式问答（SSE）** |
| `GET` | `/api/chat/preflight` | 提问前检查知识库是否可检索 |
| `GET/POST` | `/api/conversations` | 会话列表 / 新建 |
| `GET/PATCH/DELETE` | `/api/conversations/{id}` | 会话详情（含消息）/ 改名 / 删除 |
| `GET/PUT` | `/api/settings` | 读取 / 更新运行时配置 |
| `POST` | `/api/settings/reset` | 还原默认值 |
| `GET/POST` | `/api/eval/...` | 评估数据集与执行 |

### SSE 事件协议

`POST /api/chat/stream` 返回 `text/event-stream`，每帧为 `data: {json}`：

| `type` | 载荷 | 含义 |
|---|---|---|
| `start` | `conversation_id`, `kb_id`, `question` | 请求已受理，会话已建立 |
| `rewrite` | `query`, `original` | 追问被改写为独立检索式 |
| `citations` | `citations[]` | 检索完成，返回引用列表 |
| `trace` | `mode`, `dense_hits`, `sparse_hits`, `fused_hits`, `elapsed_ms` | 检索过程指标 |
| `delta` | `text` | 增量文本 |
| `done` | `message_id`, `provider`, `model`, `latency_ms`, `prompt_tokens`, ... | 生成结束 |
| `error` | `message`, `provider`, `partial?` | 出错（`partial` 为已生成的部分内容） |
| `eos` | — | 流结束 |

---

## 十一、常见问题

**Q：提问时报「无法连接 Ollama」**
A：确认 Ollama 在运行。`ollama list` 能列出模型即正常；
Windows 上可运行 `ollama app.exe`，或直接 `ollama serve`。

**Q：报「Ollama 中找不到模型 xxx」**
A：`ollama pull <模型名>`。模型名要和控制台里 `ollama list` 显示的一致。

**Q：上传 PDF 后索引进度一直失败**
A：多半是**扫描件 PDF**（没有文本层）。当前版本不支持 OCR，
系统会明确报错而不是静默建立空索引。可先用 OCR 工具转成文本再上传。

**Q：回答质量不好 / 老是说「根据现有资料无法回答」**
A：依次排查：① 知识库里是否真的有相关内容；
② 检索模式切到「仅 BM25」试试，若明显变好说明向量模型不适配该语料；
③ 在设置里调大 `final_top_k`（默认 6）给模型更多上下文；
④ 调小 `chunk_size` 让切片更聚焦。

**Q：回答里出现了红色的 `[n]` 角标**
A：说明模型标注了一个本次检索中不存在的编号，即**编造来源**。
这是有意保留的可见信号，可用于评估模型的可靠性。

**Q：第一次提问等了将近一分钟，之后就快了**
A：正常现象。首次调用要把向量模型（约 1.1 GB）和生成模型（约 4.7 GB）
加载进显存，实测约 60 秒；加载完成后单轮问答在 2~3 秒级。
Ollama 默认保留模型 5 分钟，连续使用不会反复加载。
可在设置里调大保活时间（`OLLAMA_KEEP_ALIVE`）避免间隔久了重新加载。

**Q：8G 显存跑得动吗**
A：`qwen2.5:7b-instruct` 的 Q4_K_M 量化约 4.7 GB，
配合 `bge-m3`（约 1.1 GB）可以共存。若显存吃紧，
可在设置里调小 `num_ctx`（默认 8192）或减少 `final_top_k`。

**Q：本地端口 11434 被 Docker 里的 Ollama 占用**
A：两者不能同时监听同一端口。若之前用 Docker 跑过 Ollama，
需要 `docker stop ollama`（或 `docker rm -f ollama` 彻底删掉），
或给其中一个改端口。

> 本项目已经不用 Docker 版 Ollama 了：模型迁移到 Windows 侧后，
> 容器、`ollama_data` 卷与 9.28 GB 镜像都已清理。

**Q：Ollama 启动报 `bind: An attempt was made to access a socket in a way forbidden by its access permissions`（WinError 10013），
但 `netstat` 显示没人占用这个端口**
A：这是 **Windows 的端口保留**问题，不是端口冲突，**跟 Docker 装没装没关系**。

判据很重要，别搞混：端口占用是「有别的进程占着」，杀掉它就好；
端口保留是**系统把这段端口从动态分配池里划走了**，此时没有任何进程占着它，
但**任何进程都绑不上，包括管理员**。

排查：

```powershell
# 看保留区间，确认目标端口是否落在某段内
netsh int ipv4 show excludedportrange protocol=tcp
```

本项目实测（`*` 号是「管理员保留」，没有 `*` 的是 Hyper-V / WinNAT 动态预留）：

```
     11109       11208
     11209       11308
     11309       11408
     11409       11508     ← 11434 落在这里
```

当时的排查结论（可复现）：Docker Desktop **完全没运行**、WSL 也**没有运行**，
保留区间依然存在且 `wsl --shutdown` 后毫无变化 ——
持有者是 `hns`（主机网络服务）/ `winnat`（Windows NAT 驱动）这层 Hyper-V 网络栈，
**不是 Docker 容器**。所以「卸载 Docker 的 Ollama」解决不了这个问题。

解决二选一：

```powershell
# 方案 a（推荐，需管理员，保留默认端口 11434）
# 注意 PowerShell 5.1 不支持 && 分隔语句，用换行或分号
net stop winnat /y
net start winnat

# 方案 b（无需管理员，换到区间外的端口）
setx OLLAMA_HOST "0.0.0.0:21434"
# 并把 backend\.env 的 OLLAMA_BASE_URL 改成 http://127.0.0.1:21434
# 然后重开终端（setx 只对新启动的进程生效）
```

**本项目最终采用方案 a，已还原到 11434。** 实测效果：

```
修复前：9 条保留区间，含 11409-11508
       （11434 落在其中，Python 原始 socket bind 报 WinError 10013）

net stop winnat /y ; net start winnat

修复后：仅剩 5357 与 50000-50059(*)
       11434 不在任何区间内，bind 成功
```

> **两个容易踩的点**：
> 1. `&&` 在 PowerShell 5.1 里不是合法的语句分隔符（那是 PS 7 的语法），
>    会报「标记"&&"不是此版本中的有效语句分隔符」。用换行或 `;`。
> 2. `net stop` 遇到依赖服务会弹交互确认，加 `/y` 免打扰。

> **这个保留区间可能复发**：它是 Hyper-V / WinNAT 按需创建的，
> 机器重启或 Docker Desktop 拉起 WSL2 虚拟机后可能重新划走一段端口。
> 若哪天 `start.py` 又报 Ollama 起不来，先跑
> `netsh int ipv4 show excludedportrange protocol=tcp` 看 11434 是否又被圈进去，
> 是的话重跑一次上面的方案 a 即可。
>
> 真出问题时也不必改代码：启动脚本从 `backend\.env` 的 `OLLAMA_BASE_URL`
> 读端口，改一行配置就能换到任意可用端口。

**Q：模型下载太慢 / 想换模型存放位置**
A：用 `OLLAMA_MODELS` 指定目录（本项目实测放在 `E:\AI_Models`）：
`setx OLLAMA_MODELS "E:\AI_Models"`，重启终端后 `ollama pull` 即生效。
国内从 `registry.ollama.ai` 拉取通常可达 30 MB/s 以上；
若下载 Ollama 安装包本身很慢，`ollama.com` 直连可能只有 1 MB/s，
可换用 GitHub 加速镜像（实测约 11 MB/s）。

---

## 十二、已知限制与后续工作

1. **扫描件 PDF 不支持**：没有文本层的 PDF 需要前置 OCR，当前会明确报错。
2. **重排序未启用**：`bge-reranker` 交叉编码器精排是明确的下一步优化，
   预计能明显提升 Top-K 精度。
3. **多跳推理薄弱**：需要串联多个片段才能回答的问题，单轮检索往往不足，
   可引入查询分解与迭代检索。
4. **评估依赖模型裁判**：以模型评判模型存在偏置，
   忠实度指标目前是字面重合度代理，需要更可靠的人工校验。
5. **单机单用户**：没有鉴权与并发控制，定位是本地研究与演示系统。

---

## 十三、技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| Web 框架 | FastAPI | 原生异步、自动生成 OpenAPI 文档、SSE 支持好 |
| ORM | SQLAlchemy 2.0 | 类型化 `Mapped` 声明，成熟稳定 |
| 关系库 | SQLite (WAL) | 零配置、单文件，WAL 模式让读写不互相阻塞 |
| 向量库 | ChromaDB | 纯本地持久化、Python 原生、无需独立服务 |
| 向量模型 | BAAI/bge-m3 | 中文检索表现优异，1024 维，Ollama 原生支持 |
| 生成模型 | Qwen2.5-7B-Instruct | 中文能力强，Q4 量化后适配 8G 显存 |
| 稀疏检索 | rank_bm25 + jieba | 纯 Python，无需编译，中文分词成熟 |
| 前端 | React 18 + TypeScript + Vite | 类型安全、构建快、生态成熟 |
| 包管理 | pip（清华源）+ pnpm | 国内网络环境下速度可接受 |

---

## 十四、许可

[MIT License](LICENSE)，Copyright (c) 2026 IkuyoBloom。

本项目为个人学习项目，仓库公开是为了方便查阅与交流。
欢迎参考，转载或二次开发保留版权声明即可。

> 注意：部分示例文档与界面素材来自第三方，不在本许可证覆盖范围内。
