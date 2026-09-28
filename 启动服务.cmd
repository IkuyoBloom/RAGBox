@echo off
REM ===================================================================
REM  Fallback launcher only.
REM
REM  The PRIMARY way to start is the shortcut  "启动服务.lnk" , which runs
REM  the venv python directly - no batch file, no encoding issues.
REM  Use this .cmd only if the shortcut is missing (e.g. after copying the
REM  repo without it).
REM
REM  Keep this file PURE ASCII, comments included: cmd.exe reads .cmd files
REM  using the console code page (936/GBK on Chinese Windows) while editors
REM  write UTF-8, so Chinese bytes mis-pair and swallow the following
REM  character - the text then gets executed as a command.
REM  All Chinese output comes from scripts/start.py. See AGENTS.md.
REM ===================================================================

chcp 65001 >nul
cd /d "%~dp0"
"%~dp0backend\.venv\Scripts\python.exe" "%~dp0scripts\start.py" --pause-on-exit
