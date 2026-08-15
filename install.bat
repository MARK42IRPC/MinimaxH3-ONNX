@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo [ERROR] uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/
  pause
  exit /b 1
)

set "GPU_DETECTED=0"
set "GPU_INSTALL=0"
set "GPU_OVERRIDE=%H3_INSTALL_GPU%"

echo Checking for an NVIDIA CUDA device...
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  nvidia-smi -L >nul 2>nul
  if not errorlevel 1 set "GPU_DETECTED=1"
)
if "!GPU_DETECTED!"=="1" (
  echo NVIDIA device detected:
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
) else (
  echo No usable NVIDIA device detected. Core dependencies will be installed.
)

if /I "!GPU_OVERRIDE!"=="0" (
  set "GPU_INSTALL=0"
  echo H3_INSTALL_GPU=0: skipping optional GPU development tools.
) else if /I "!GPU_OVERRIDE!"=="1" (
  set "GPU_INSTALL=1"
  echo H3_INSTALL_GPU=1: installing optional GPU development tools.
) else if "!GPU_DETECTED!"=="1" (
  set "GPU_INSTALL=1"
  echo Automatic GPU mode: installing optional GPU development tools.
) else (
  echo Automatic CPU mode: use H3_INSTALL_GPU=1 to force the GPU extra.
)

echo Creating or updating the Python 3.11 environment...
uv python install 3.11
if errorlevel 1 goto :failed

if "!GPU_INSTALL!"=="1" (
  uv sync --locked --extra dev --extra gpu --no-editable
) else (
  uv sync --locked --extra dev --no-editable
)
if errorlevel 1 goto :failed

echo Creating runtime directories...
for %%D in (".h3-workbench" "onnx_models" "exported" "qwen_tokenizer") do (
  if not exist "%%~D" mkdir "%%~D"
  if not exist "%%~D\" goto :failed
)

echo Running installation diagnostics...
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] The virtual environment was not created.
  goto :failed
)
".venv\Scripts\python.exe" -c "import sys, torch, onnxruntime as ort; cuda=bool(torch.cuda.is_available()); providers=ort.get_available_providers(); print('Torch:', torch.__version__); print('Torch CUDA:', torch.version.cuda); print('Torch CUDA available:', cuda); print('ONNX Runtime:', ort.__version__); print('ONNX providers:', ', '.join(providers)); sys.exit(2 if !GPU_DETECTED! == 1 and (not cuda or 'CUDAExecutionProvider' not in providers) else 0)"
if errorlevel 2 goto :gpu_check_failed
if errorlevel 1 goto :failed

if "!GPU_INSTALL!"=="1" (
  ".venv\Scripts\python.exe" -c "import cupy, cutlass_library; print('CuPy:', cupy.__version__); print('CUTLASS Python package: available')"
  if errorlevel 1 goto :gpu_toolchain_failed
)

echo.
echo Installation complete.
echo Models are optional and can be selected in the WebUI model manager.
echo Start with start_webui.bat, then open http://127.0.0.1:7860
pause
exit /b 0

:gpu_check_failed
echo.
echo [ERROR] An NVIDIA device was detected, but CUDA acceleration is not ready.
echo Check the NVIDIA driver, CUDA 12 runtime, cuDNN 9, and the ONNX Runtime provider output above.
pause
exit /b 2

:gpu_toolchain_failed
echo.
echo [ERROR] The optional GPU toolchain was selected but CuPy or CUTLASS could not be imported.
echo Set H3_INSTALL_GPU=0 to install only the core runtime, or inspect the package output above.
pause
exit /b 1

:failed
echo.
echo [ERROR] Installation failed. Check the command output above.
pause
exit /b 1
