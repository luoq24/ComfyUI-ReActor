@echo off
rem ============================================================
rem  ReActor ONNX GPU repair script (ASCII only, CRLF required)
rem  Fix: conflicting/broken onnxruntime installs in the webui venv
rem  Install: onnxruntime-gpu 1.20.2 (CUDA 12 + cuDNN 9, matches torch 2.8.0+cu128)
rem
rem  Usage:
rem    fix_onnx_gpu.bat          normal run
rem    fix_onnx_gpu.bat /dry     dry run (prints commands, changes nothing)
rem
rem  NOTE: close ComfyUI and SD-WebUI before the normal run.
rem ============================================================
setlocal
set "VENV=D:\Program_Green\stable-diffusion-webui\venv"
set "SP=%VENV%\Lib\site-packages"
set PIP=%VENV%\Scripts\python.exe -m pip
set "RMDIR_CMD=rmdir"
set VERIFY=%VENV%\Scripts\python.exe -c "import onnxruntime as ort; print('onnxruntime', ort.__version__); print('providers:', ort.get_available_providers())"

if /I "%~1"=="/dry" set "PIP=echo [dry] pip"
if /I "%~1"=="/dry" set "RMDIR_CMD=echo [dry] rmdir"
if /I "%~1"=="/dry" set "VERIFY=echo [dry] verification skipped"

echo.
echo ============================================
echo  ReActor ONNX GPU repair
echo  venv: %VENV%
echo ============================================
echo.
echo [WARN] Close ComfyUI and SD-WebUI before continuing if they are running.
echo.

echo [1/4] Uninstall broken/conflicting onnxruntime packages (errors here are OK)...
%PIP% uninstall -y onnxruntime
%PIP% uninstall -y onnxruntime-gpu

echo.
echo [2/4] Clean leftover directories from half-finished uninstalls...
if exist "%SP%\onnxruntime" %RMDIR_CMD% /S /Q "%SP%\onnxruntime"
if exist "%SP%\onnxruntime-1.20.1.dist-info" %RMDIR_CMD% /S /Q "%SP%\onnxruntime-1.20.1.dist-info"
if exist "%SP%\onnxruntime_gpu-1.13.1.dist-info" %RMDIR_CMD% /S /Q "%SP%\onnxruntime_gpu-1.13.1.dist-info"
for /d %%D in ("%SP%\~nnxruntime*") do %RMDIR_CMD% /S /Q "%%D"
for /d %%D in ("%SP%\~onnxruntime*") do %RMDIR_CMD% /S /Q "%%D"
for /d %%D in ("%SP%\~rotobuf*") do %RMDIR_CMD% /S /Q "%%D"

echo.
echo [3/4] Install onnxruntime-gpu 1.20.2 (CUDA 12 build)...
%PIP% install onnxruntime-gpu==1.20.2

echo.
echo [4/4] Verify...
%VERIFY%

echo.
echo ============================================
echo  DONE. "providers" above must contain CUDAExecutionProvider.
echo  Do NOT install CPU "onnxruntime" into this venv again.
echo ============================================
pause
