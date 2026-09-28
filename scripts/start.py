"""一键启动：Ollama + 后端 + 前端。

用法（在项目根目录，或任意位置指定脚本路径）：

    python scripts/start.py                 # 启动全部
    python scripts/start.py --reload        # 后端热重载（改代码自动生效）
    python scripts/start.py --backend-only  # 只起后端
    python scripts/start.py --check         # 只做环境体检，不起服务

Ctrl+C 会对整棵进程树发终止信号，不留残留。
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"

DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 5173

# 终端配色。非 TTY（被重定向到文件/管道）时自动降级为纯文本，
# 免得日志里出现一堆看不懂的转义序列。
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def ok(text: str) -> str:
    return _c("32", text)


def warn(text: str) -> str:
    return _c("33", text)


def err(text: str) -> str:
    return _c("31", text)


def dim(text: str) -> str:
    return _c("90", text)


def info(text: str) -> str:
    return _c("36", text)


# ----------------------------------------------------------------------
# 配置读取
# ----------------------------------------------------------------------
def read_env_value(key: str, default: str = "") -> str:
    """从 backend/.env 读一个键。

    只支持 `KEY=VALUE` 这种简单形式（够用且不引入依赖）。
    注释行与不含 `=` 的行跳过；同名的取最后一个，与 shell 的覆盖语义一致。
    """
    env_file = BACKEND / ".env"
    if not env_file.exists():
        return default
    try:
        text = env_file.read_text(encoding="utf-8")
    except OSError:
        return default

    found = default
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            found = value.strip()
    return found


def resolve_ollama_endpoint() -> tuple[str, int, str]:
    """解析 Ollama 地址，返回 (host, port, 来源说明)。

    端口从 .env 的 OLLAMA_BASE_URL 读，不在脚本里硬编码 ——
    之前 .ps1 写死 11434，后来端口一改脚本就开始误报「未在 11434 监听」。
    """
    raw = read_env_value("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        from urllib.parse import urlparse

        parsed = urlparse(raw if "//" in raw else f"http://{raw}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 11434
        return host, port, "backend/.env"
    except ValueError:
        print(warn(f"  [警告] OLLAMA_BASE_URL 解析失败（{raw}），回落到 127.0.0.1:11434"))
        return "127.0.0.1", 11434, "默认值"


# ----------------------------------------------------------------------
# 端口探测
# ----------------------------------------------------------------------
def port_open(host: str, port: int, timeout: float = 0.7) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_port(host: str, port: int, timeout: float = 30.0, interval: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_open(host, port):
            return True
        time.sleep(interval)
    return False


def describe_port_holder(port: int) -> str:
    """给出占用端口的进程信息，方便用户自己决定要不要杀。"""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=10
            ).stdout
            pids = {
                line.split()[-1]
                for line in out.splitlines()
                if f":{port} " in line and "LISTENING" in line
            }
            details = []
            for pid in pids:
                if pid.isdigit() and pid != "0":
                    details.append(f"PID {pid}")
            return "、".join(details) if details else "未知进程"
    except Exception:
        pass
    return "未知进程"


# ----------------------------------------------------------------------
# Ollama
# ----------------------------------------------------------------------
def ollama_app_path() -> Path | None:
    """定位 ollama 可执行文件。

    优先用 `ollama app.exe`（Windows 上的托盘程序，会自动拉起 serve），
    找不到就退回 `ollama serve`。
    """
    candidates: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Programs" / "Ollama" / "ollama app.exe")
        candidates.append(Path(local) / "Programs" / "Ollama" / "ollama.exe")
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidates.append(Path(program_files) / "Ollama" / "ollama.exe")
    if sys.platform != "win32":
        which = shutil.which("ollama")
        if which:
            candidates.append(Path(which))

    for path in candidates:
        if path.exists():
            return path
    return None


def ensure_ollama(host: str, port: int) -> bool:
    if port_open(host, port):
        print(f"  {ok('✓')} Ollama 已在 {host}:{port} 运行")
        return True

    exe = ollama_app_path()
    if exe is None:
        print(f"  {warn('!')} 未找到 Ollama，本地生成通道不可用")
        print(dim("      安装：https://ollama.com/download  或 winget install Ollama.Ollama"))
        return False

    print(f"  {warn('…')} 启动 Ollama（{exe.name}）")
    creationflags = 0
    if os.name == "nt":
        # 让 Ollama 独立于本脚本存活：用户关掉启动器后 Ollama 仍可用
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [str(exe)],
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if wait_port(host, port, timeout=45):
        print(f"  {ok('✓')} Ollama 就绪（{host}:{port}）")
        return True

    print(f"  {err('✗')} Ollama 启动超时")
    print_ollama_troubleshoot(host, port)
    return False


def print_ollama_troubleshoot(host: str, port: int) -> None:
    """Ollama 起不来时给出可执行的排查步骤。

    最常见的原因不是「端口被别的程序占了」，而是 Windows 的 Hyper-V / WinNAT
    把该端口**保留**了：此时 netstat 显示无人占用，但任何进程都绑不上，
    报 WinError 10013。判据是保留区间包含该端口。
    """
    print()
    print(warn("  Ollama 没有响应，按顺序排查："))
    print(dim(f"    1) 看日志：Get-Content \"$env:LOCALAPPDATA\\Ollama\\server.log\" -Tail 20"))
    print(dim(f"    2) 查端口保留：netsh int ipv4 show excludedportrange protocol=tcp"))
    print(dim(f"       若 {port} 落在某个区间内，就是被 Hyper-V/WinNAT 保留了"))
    print(dim("       （表现：netstat 无人占用，但绑定报 WinError 10013）"))
    print(dim("       修复需管理员：net stop winnat && net start winnat"))
    print(dim(f"       或换端口：setx OLLAMA_HOST \"0.0.0.0:21434\" 并同步改 backend\\.env"))
    print(dim(f"    3) 本脚本探测的是 {host}:{port}（来自 backend/.env 的 OLLAMA_BASE_URL）"))
    print()


# ----------------------------------------------------------------------
# 子进程管理
# ----------------------------------------------------------------------
class Service:
    """带输出前缀转发的子进程。

    两个服务的日志交织在同一个终端里，加前缀才分得清谁是谁。
    不做前缀的话，uvicorn 的一行 ERROR 和 vite 的一行报错会看起来像同一件事。
    """

    def __init__(self, name: str, cmd: list[str], cwd: Path, color: str = "36") -> None:
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.color = color
        self.proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        env = os.environ.copy()
        # Python 默认按系统区域编码写 stdout（中文 Windows 是 GBK），
        # 被重定向时中文日志会变乱码。钉成 UTF-8 后两种场景都正常。
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        self.proc = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        prefix = _c(self.color, f"[{self.name}]")
        for line in self.proc.stdout:
            sys.stdout.write(f"{prefix} {line.rstrip()}\n")
            sys.stdout.flush()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout: float = 8.0) -> None:
        """终止整棵进程树。

        pnpm/vite 是「cmd → node → 子进程」的多层结构，只杀直接子进程会留下
        占着端口的孙子进程。Windows 上用 taskkill /T 递归终止。
        """
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                    capture_output=True,
                    timeout=timeout,
                )
            else:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        except Exception:
            pass


# ----------------------------------------------------------------------
# 前置检查
# ----------------------------------------------------------------------
def check_prerequisites(need_backend: bool, need_frontend: bool) -> list[str]:
    problems: list[str] = []

    if need_backend:
        python = BACKEND / ".venv" / "Scripts" / "python.exe"
        if sys.platform != "win32":
            python = BACKEND / ".venv" / "bin" / "python"
        if not python.exists():
            problems.append(
                f"找不到后端虚拟环境：{python}\n"
                f"      请先执行：python -m venv backend/.venv\n"
                f"                backend/.venv/Scripts/python.exe -m pip install "
                f"-r backend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
        if not (BACKEND / ".env").exists():
            problems.append(
                "缺少 backend/.env\n"
                "      请先执行：Copy-Item backend/.env.example backend/.env"
            )

    if need_frontend:
        if shutil.which("pnpm") is None:
            problems.append("PATH 中找不到 pnpm。安装：npm install -g pnpm")
        if not (FRONTEND / "node_modules").exists():
            problems.append("缺少前端依赖。请先执行：cd frontend && pnpm install")

    return problems


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def set_console_title(text: str) -> None:
    """给控制台窗口起个显眼的标题。

    为什么放在 Python 里而不是写在 .cmd 里：**批处理文件对中文极不友好**。
    cmd.exe 按控制台代码页（中文 Windows 是 936/GBK）读取 .cmd 的内容，
    而编辑器存的是 UTF-8 —— 中文字节会被错位配对，GBK 的第二字节范围又包含
    空格与 & | 等符号，于是「吃掉」后面的空格，让整行文本被当成命令执行：

        'E' is not recognized as an internal or external command

    所以 .cmd 保持纯 ASCII，中文一律由 Python 输出（Python 自己管编码）。
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # 必须声明 argtypes。ctypes 在不声明时会把 Python str 转成 **char\***
        # （ANSI 字节串），而这是个 ...W 函数，它期待 wchar_t* ——
        # 于是收到的是一串按 UTF-16 解读的 ASCII 字节，标题变成乱码或直接为空。
        # 和前面 Job Object 那个句柄被截断的 bug 是同一类：
        # **ctypes 调 Win32 API，签名一个都不能省。**
        kernel32.SetConsoleTitleW.argtypes = [wintypes.LPCWSTR]
        kernel32.SetConsoleTitleW.restype = wintypes.BOOL
        kernel32.SetConsoleTitleW(text)
    except Exception:
        pass


def install_kill_on_close_job() -> tuple[bool, str]:
    """把本进程放进 Job Object，并设置「最后一个句柄关闭时杀掉整组进程」。

    ## 为什么需要

    用户是**双击 `启动服务.cmd` 开一个控制台窗口**来启动的，关闭窗口就是停止服务。
    正常情况下 Windows 会给同一控制台上的进程发 CTRL_CLOSE_EVENT，
    uvicorn 与 vite 会跟着退出。但只要有子进程脱离了控制台，它就会变成孤儿 ——
    继续占着端口，下次启动就报「端口被占用」。本项目就遇到过前端残留的情况。

    Job Object 是内核级的保证：**父进程无论怎么死**（点 X、Ctrl+C、被强杀），
    整组进程一起被回收，不留孤儿。

    ## 失败是允许的，但必须说清原因

    如果当前进程已经在别人的 Job 里且不允许嵌套，AssignProcessToJobObject 会失败。
    这时服务照样能跑，只是少一层保障 —— 所以返回原因而不是抛异常，
    让用户知道「关窗可能留残留」而不是收到一句含糊的「不支持」。
    """
    if os.name != "nt":
        return False, "非 Windows 平台，不需要 Job Object"
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # ---- 必须显式声明签名 ----
        # ctypes 默认把返回值当 32 位 int，而 Windows 句柄是 64 位。
        # GetCurrentProcess() 返回的伪句柄是 0xFFFFFFFFFFFFFFFF，
        # 被截断成 0x00000000FFFFFFFF 就成了无效句柄，
        # AssignProcessToJobObject 会报 ERROR_INVALID_HANDLE(6) ——
        # 看起来像「系统不支持嵌套 Job」，其实是自己把参数传坏了。
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []

        def why(step: str) -> str:
            code = ctypes.get_last_error()
            return f"{step}失败（Win32 错误码 {code}）"

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class _BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(wintypes.ULONG)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC_LIMIT),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return False, why("创建 Job Object")

        limits = _EXTENDED_LIMIT()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            return False, why("设置 Job 限制")

        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            return False, why("把本进程加入 Job")

        # 句柄必须留着：一旦关闭，「最后一个句柄关闭」会立刻触发，把自己也杀掉。
        global _JOB_HANDLE
        _JOB_HANDLE = job
        return True, "已启用关窗即停"
    except Exception as exc:
        return False, f"Job Object 初始化异常：{type(exc).__name__}: {exc}"


_JOB_HANDLE: int | None = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="本地 RAG 知识库问答系统 - 一键启动",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/start.py                # 全部启动\n"
            "  python scripts/start.py --reload       # 后端热重载\n"
            "  python scripts/start.py --check        # 只体检\n"
            "  python scripts/start.py --backend-only # 只起后端\n"
        ),
    )
    parser.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT)
    parser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT)
    parser.add_argument("--reload", action="store_true", help="后端热重载（开发用）")
    parser.add_argument("--backend-only", action="store_true")
    parser.add_argument("--frontend-only", action="store_true")
    parser.add_argument("--no-ollama", action="store_true", help="跳过 Ollama 检查")
    parser.add_argument("--check", action="store_true", help="只做环境体检，不启动服务")
    args = parser.parse_args()

    need_backend = not args.frontend_only
    need_frontend = not args.backend_only

    print()
    print(info("  ══════════════════════════════════════════════"))
    print(info("    本地 RAG 知识库问答系统 - 启动器"))
    print(info("  ══════════════════════════════════════════════"))
    print()

    # 窗口标题由 Python 设置：.cmd 里写中文会被 cmd.exe 按 GBK 解错（见函数注释）
    set_console_title("本地 RAG 知识库 - 关闭此窗口即停止全部服务")

    # 装上「关窗即全停」的内核级保证。失败不影响启动，只是少一层保障。
    job_ok, job_note = install_kill_on_close_job()
    print(dim(f"  {job_note}"))
    if not job_ok:
        print(dim("    关窗后若仍有残留占着端口，重启电脑或手动结束对应进程即可"))
    print()

    # ---------------- 前置检查 ----------------
    problems = check_prerequisites(need_backend, need_frontend)
    if problems:
        print(err("  启动前检查未通过："))
        for p in problems:
            print(f"    - {p}")
        print()
        return 1

    host, ollama_port, source = resolve_ollama_endpoint()
    print(f"  {dim('Ollama 端点')}: {host}:{ollama_port}  {dim(f'（来自 {source}）')}")

    if need_backend:
        if port_open("127.0.0.1", args.backend_port):
            print(warn(f"  ! 端口 {args.backend_port} 已被占用（{describe_port_holder(args.backend_port)}）"))
            print(dim("    后端会启动失败；先关掉占用方，或用 --backend-port 换端口"))
        if args.reload:
            print(dim("  后端：热重载已开启"))

    if need_frontend and port_open("127.0.0.1", args.frontend_port):
        print(warn(f"  ! 端口 {args.frontend_port} 已被占用（{describe_port_holder(args.frontend_port)}）"))

    if args.check:
        print()
        print(ok("  环境体检通过，可以启动。"))
        return 0

    print()

    # ---------------- Ollama ----------------
    if need_backend and not args.no_ollama:
        print(f"  {dim('[1/3]')} Ollama")
        ensure_ollama(host, ollama_port)
        print()

    # ---------------- 后端 ----------------
    services: list[Service] = []

    if need_backend:
        python = BACKEND / ".venv" / "Scripts" / "python.exe"
        if sys.platform != "win32":
            python = BACKEND / ".venv" / "bin" / "python"

        cmd = [
            str(python),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.backend_port),
        ]
        if args.reload:
            cmd.append("--reload")

        print(f"  {dim('[2/3]')} 启动后端 …")
        services.append(Service("后端", cmd, BACKEND, color="36"))
        services[-1].start()

    # ---------------- 前端 ----------------
    if need_frontend:
        pnpm = shutil.which("pnpm")
        assert pnpm is not None
        if os.name == "nt":
            # pnpm 在 Windows 上是 .cmd 批处理，CreateProcess 不能直接执行，
            # 需要经 cmd /c 转发。这样进程树仍然是 cmd → node，taskkill /T 能收干净。
            cmd = ["cmd", "/c", pnpm, "dev", "--port", str(args.frontend_port)]
        else:
            cmd = [pnpm, "dev", "--port", str(args.frontend_port)]

        print(f"  {dim('[3/3]')} 启动前端 …")
        services.append(Service("前端", cmd, FRONTEND, color="35"))
        services[-1].start()

    if not services:
        print(warn("  没有需要启动的服务。"))
        return 0

    # ---------------- 等待就绪 ----------------
    print()
    checks: list[tuple[str, str, int, float]] = []
    if need_backend:
        checks.append(("后端", "127.0.0.1", args.backend_port, 90.0))
    if need_frontend:
        checks.append(("前端", "127.0.0.1", args.frontend_port, 90.0))

    ready: dict[str, bool] = {}
    for name, h, p, timeout in checks:
        sys.stdout.write(dim(f"  等待 {name} 就绪 …"))
        sys.stdout.flush()
        got = wait_port(h, p, timeout=timeout)
        ready[name] = got
        print(f"\r  {ok('✓') if got else err('✗')} {name} {'就绪' if got else '启动超时'}        ")

    # ---------------- 汇总 ----------------
    print()
    print(info("  ══════════════════════════════════════════════"))
    if need_frontend and ready.get("前端"):
        print(f"   前端界面 : {_c('1', f'http://127.0.0.1:{args.frontend_port}')}")
    if need_backend and ready.get("后端"):
        print(f"   接口文档 : http://127.0.0.1:{args.backend_port}/docs")
        print(f"   健康检查 : http://127.0.0.1:{args.backend_port}/api/health")
    print(dim("   ══════════════════════════════════════════════"))
    print(dim("   按 Ctrl+C 停止全部服务"))
    print()

    # 有服务没起来时，把最近的日志留在这里给用户看，而不是直接退出
    if not all(ready.values()):
        failed = [n for n, v in ready.items() if not v]
        print(err(f"  {('、'.join(failed))} 未在超时内就绪，日志见上方。"))
        print()

    # ---------------- 守护循环 ----------------
    stop_flag = threading.Event()

    def _on_signal(signum, _frame):  # noqa: ARG001
        stop_flag.set()

    signal.signal(signal.SIGINT, _on_signal)
    try:
        signal.signal(signal.SIGTERM, _on_signal)
    except (AttributeError, ValueError):
        pass

    try:
        while not stop_flag.is_set():
            for svc in services:
                if not svc.alive():
                    code = svc.proc.returncode if svc.proc else -1
                    print()
                    print(err(f"  {svc.name} 已退出（退出码 {code}），正在收掉其余服务…"))
                    stop_flag.set()
                    break
            stop_flag.wait(1.0)
    except KeyboardInterrupt:
        stop_flag.set()

    print()
    print(dim("  正在停止服务 …"))
    for svc in services:
        svc.stop()
    print(ok("  已全部停止。"))
    return 0


def pause_before_exit() -> None:
    """退出前等一次回车，让用户看清日志。

    双击快捷方式启动时，Python 进程一结束控制台窗口就关掉了 ——
    启动失败的话用户根本来不及看报错，只会看到窗口一闪而过。
    所以在这里停一下。

    只在真正的交互式控制台里等（`isatty`）。被脚本调用、输出被重定向时
    直接返回，否则会卡住调用方。
    """
    try:
        if not sys.stdin.isatty():
            return
        print()
        input("  按回车关闭此窗口…")
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    # --pause-on-exit 在交给 argparse 之前摘掉：它是给「双击启动」用的行为开关，
    # 不属于启动参数，也不需要出现在 --help 里。
    _pause = "--pause-on-exit" in sys.argv
    if _pause:
        sys.argv.remove("--pause-on-exit")

    _code = main()
    if _pause:
        pause_before_exit()
    raise SystemExit(_code)
