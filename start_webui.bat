@echo off
setlocal

cd /d "%~dp0"
set "PYTHONPATH=%CD%\src"
set "H3_CPU_AFFINITY=0xFF"
set "H3_CPU_PRIORITY=AboveNormal"
set "H3_VALIDATE_FINITE=1"
set "WEBUI_EXE=%CD%\.venv\Scripts\h3-workbench.exe"

if not exist "%WEBUI_EXE%" (
    echo [ERROR] Virtual environment executable not found:
    echo         %WEBUI_EXE%
    echo Run the project setup first, then try again.
    pause
    exit /b 1
)

echo Starting MiniMax H3 WebUI at http://127.0.0.1:7860
echo Keep this window open while using the WebUI.
echo.

"%WEBUI_EXE%" --host 127.0.0.1 --port 7860 --workspace "%CD%"

echo.
echo WebUI stopped. Press any key to close this window.
pause >nul
