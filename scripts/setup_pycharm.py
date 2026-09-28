"""把项目的虚拟环境注册进 PyCharm，让它开箱可用。

## 为什么需要这个脚本

PyCharm 把「解释器列表」存在**用户级全局配置**里
（`%APPDATA%\\JetBrains\\PyCharm<版本>\\options\\jdk.table.xml`），
而 `.idea/misc.xml` 只用**名字**引用它：

    project-jdk-name="Python 3.12 (RAG)"

所以光写项目里的 `.idea/` 是不够的 —— 名字对不上，PyCharm 会显示
「无效的解释器」，用户还得手动去设置里点一遍。

这个脚本把虚拟环境按 PyCharm 自己的格式写进那张全局表，
打开项目时解释器就已经就位。

## 为什么必须等 PyCharm 关闭

PyCharm 运行时会把配置缓存在内存里，退出时整份写回 ——
在它运行期间改 `jdk.table.xml`，改动会被覆盖掉。
所以脚本检测到 PyCharm 在跑就直接拒绝，并提示先关掉。

用法：

    python scripts/setup_pycharm.py            # 注册（PyCharm 需已关闭）
    python scripts/setup_pycharm.py --check    # 只看当前状态，不写入
    python scripts/setup_pycharm.py --remove   # 撤销注册
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = ROOT / "backend" / ".venv" / "Scripts" / "python.exe"
SDK_NAME = "Python 3.12 (RAG)"

OK = "  [OK]   "
WARN = "  [注意] "
ERR = "  [错误] "


# ----------------------------------------------------------------------
def jetbrains_options_dirs() -> list[Path]:
    """列出所有 PyCharm 版本的 options 目录（可能装了多个版本）。"""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return []
    base = Path(appdata) / "JetBrains"
    if not base.is_dir():
        return []
    dirs: list[Path] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or not entry.name.lower().startswith("pycharm"):
            continue
        options = entry / "options"
        if (options / "jdk.table.xml").is_file():
            dirs.append(options)
    return dirs


def pycharm_running() -> list[str]:
    """返回正在运行的 PyCharm 进程名（空列表表示没在跑）。"""
    try:
        # 用 tasklist 而不是 psutil：项目里没装 psutil，也不想为一个检查加依赖
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq pycharm64.exe", "/NH"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
    except Exception:
        return []
    return ["pycharm64.exe"] if "pycharm64.exe" in out.lower() else []


def interpreter_info() -> dict[str, str]:
    """向虚拟环境问出注册时需要的信息（版本、site-packages、基础解释器路径）。"""
    # 注意 get_paths() 的第一个位置参数是 **scheme 名**（如 'nt'、'posix_prefix'），
    # 不是 'purelib' 这种路径键 —— 传错了会直接抛异常。
    # 基础解释器的路径要用 vars 覆盖 'base'，否则拿到的是 venv 自己的。
    code = (
        "import json, sys, sysconfig;"
        "paths = sysconfig.get_paths();"
        "base = sysconfig.get_paths(vars={'base': sys.base_prefix});"
        "print(json.dumps({"
        "'version': '%d.%d.%d' % sys.version_info[:3],"
        "'prefix': sys.prefix,"
        "'base_prefix': sys.base_prefix,"
        "'site_packages': paths['purelib'],"
        "'base_site_packages': base['purelib'],"
        "'dlls': base['platstdlib'],"
        "}))"
    )
    proc = subprocess.run(
        [str(PYTHON_EXE), "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        # 把 stderr 带出来。否则失败只剩一句「returned non-zero exit status」，
        # 还得手工再跑一遍才能知道原因 —— 这个脚本刚才就这么坑了自己一次。
        raise RuntimeError(
            f"向虚拟环境查询信息失败（退出码 {proc.returncode}）：\n{proc.stderr.strip()}"
        )

    import json

    return json.loads(proc.stdout.strip())


# ----------------------------------------------------------------------
def build_entry(info: dict[str, str], *, sdk_uuid: str) -> str:
    """按 PyCharm 自己的格式拼一条 <jdk> 记录。

    路径统一用正斜杠、并以 `file://` 前缀 —— 这是 PyCharm 内部 VirtualFile
    的表示法，写成 Windows 反斜杠路径它认不出来。
    """

    def url(p: str) -> str:
        return "file://" + str(Path(p)).replace("\\", "/")

    venv_root = str(PYTHON_EXE.parents[1])
    roots = [
        url(venv_root),
        url(info["site_packages"]),
        url(info["dlls"]),
        url(str(Path(info["base_prefix"]) / "Lib")),
        url(info["base_site_packages"]),
        # typeshed 存根由 PyCharm 自己维护，这里给个占位目录即可，
        # 它会在首次加载该解释器时重建真正的骨架。
        "file://$APPLICATION_HOME_DIR$/plugins/python-ce/helpers/typeshed/stdlib",
    ]
    # 去重：Windows 上 platstdlib 与基础解释器的 Lib 常常是同一个目录，
    # 写两遍没有意义，还会让 PyCharm 的解释器路径列表看着很脏。
    seen: set[str] = set()
    roots = [r for r in roots if not (r in seen or seen.add(r))]
    root_lines = "\n".join(f'              <root url="{r}" type="simple" />' for r in roots)

    return f"""      <jdk version="2">
        <name value="{SDK_NAME}" />
        <type value="Python SDK" />
        <version value="Python {info['version']}" />
        <homePath value="{str(PYTHON_EXE).replace(chr(92), '/')}" />
        <roots>
          <classPath>
            <root type="composite">
{root_lines}
            </root>
          </classPath>
          <sourcePath>
            <root type="composite" />
          </sourcePath>
        </roots>
        <additional SDK_UUID="{sdk_uuid}">
          <setting name="FLAVOR_ID" value="VirtualEnvSdkFlavor" />
          <setting name="FLAVOR_DATA" value="{{}}" />
        </additional>
      </jdk>
"""


def sdk_uuid_for(text: str) -> str | None:
    """读取已注册的 SDK_UUID —— 重复执行时要沿用，否则每次都会换一个。"""
    m = re.search(
        rf'<name value="{re.escape(SDK_NAME)}" />[\s\S]*?SDK_UUID="([^"]+)"',
        text,
    )
    return m.group(1) if m else None


def register(options_dir: Path, info: dict[str, str], *, remove: bool = False) -> str:
    table = options_dir / "jdk.table.xml"
    text = table.read_text(encoding="utf-8")

    # 先摘掉已有的同名记录，保证可重复执行
    text = re.sub(
        rf'\s*<jdk version="2">\s*<name value="{re.escape(SDK_NAME)}" />[\s\S]*?</jdk>',
        "",
        text,
    )

    if remove:
        shutil.copy2(table, table.with_suffix(".xml.bak"))
        table.write_text(text, encoding="utf-8")
        return "已撤销"

    entry = build_entry(info, sdk_uuid=sdk_uuid_for(text) or str(uuid.uuid4()))

    # 插到最后一行 </component> 之前。
    # 不能写死缩进（早先写死 4 个空格，2024.3 的文件格式不同就直接失败），
    # 也不能依赖「只有一个 component」以外的假设 —— 这里按行首插入，
    # 让新块自成一行，与既有格式解耦。
    idx = text.rfind("</component>")
    if idx < 0:
        raise RuntimeError(f"没找到 </component>，文件格式可能变了：{table}")
    line_start = text.rfind("\n", 0, idx) + 1
    text = text[:line_start] + entry + text[line_start:]

    shutil.copy2(table, table.with_suffix(".xml.bak"))
    table.write_text(text, encoding="utf-8")
    return "已注册"


# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="把虚拟环境注册进 PyCharm")
    parser.add_argument("--check", action="store_true", help="只看状态，不写入")
    parser.add_argument("--remove", action="store_true", help="撤销注册")
    parser.add_argument(
        "--force",
        action="store_true",
        help="PyCharm 正在运行时也写入（改动可能被它在退出时覆盖，但不会损坏文件）",
    )
    args = parser.parse_args()

    print("=" * 66)
    print("  PyCharm 虚拟环境配置")
    print("=" * 66)
    print()

    if not PYTHON_EXE.is_file():
        print(f"{ERR}找不到虚拟环境：{PYTHON_EXE}")
        print("       先建环境：cd backend; python -m venv .venv; .venv\\Scripts\\python.exe -m pip install -r requirements.txt")
        return 1

    info = interpreter_info()
    print(f"{OK}虚拟环境：{PYTHON_EXE}")
    print(f"        Python {info['version']}")
    print(f"        site-packages: {info['site_packages']}")
    print()

    dirs = jetbrains_options_dirs()
    if not dirs:
        print(f"{WARN}没有找到任何 PyCharm 配置目录（PyCharm 可能还没运行过）")
        print("       先启动一次 PyCharm，再回来跑这个脚本即可。")
        return 0

    print(f"  发现 {len(dirs)} 个 PyCharm 配置目录：")
    for d in dirs:
        has = SDK_NAME in (d / "jdk.table.xml").read_text(encoding="utf-8")
        print(f"    {'✔' if has else '·'} {d.parent.name}  ({'已注册' if has else '未注册'})")
    print()

    if args.check:
        print("  --check：只看状态，未做修改。")
        return 0

    running = pycharm_running()
    if running and not args.force:
        print(f"{ERR}PyCharm 正在运行（{', '.join(running)}）—— 现在改可能被它退出时覆盖。")
        print("       请先完全关闭 PyCharm 再运行本脚本；")
        print("       或加 --force 继续（最坏情况只是改动丢失，文件不会损坏）。")
        return 1
    if running:
        print(f"{WARN}PyCharm 正在运行，按 --force 继续写入。")
        print("       写完后请重开 PyCharm 验证；若解释器没出现，说明被覆盖了，")
        print("       关掉 PyCharm 再跑一次即可。")
        print()

    for d in dirs:
        result = register(d, info, remove=args.remove)
        print(f"{OK}{d.parent.name}: {result}（原文件已备份为 jdk.table.xml.bak）")

    print()
    if args.remove:
        print("  已撤销。重新打开 PyCharm 后需要手动选择解释器。")
    else:
        print(f"  完成。用 PyCharm 打开 {ROOT} 即可，解释器「{SDK_NAME}」已经就位。")
        print("  （.idea/misc.xml 与 .idea/RAG.iml 也已配好源码根，import 不会再报红）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
