@echo off
setlocal EnableExtensions

cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo [ERROR] uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/
  pause
  exit /b 1
)

echo Creating or updating the Python 3.11 environment...
uv python install 3.11
if errorlevel 1 goto :failed
uv sync --extra dev --no-editable
if errorlevel 1 goto :failed

echo.
echo Installation complete.
echo Models are optional and can be selected in the WebUI model manager.
echo Start with start_webui.bat
pause
exit /b 0

:failed
echo.
echo [ERROR] Installation failed. Check the command output above.
pause
exit /b 1
