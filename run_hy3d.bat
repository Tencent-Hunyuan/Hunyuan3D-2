@echo off
REM Hunyuan3D launcher - strips PYTHONPATH and uses correct Python venv
setlocal

REM Hardcode our CUDA PyTorch venv python (bypasses uv/system python)
set "PY=%~dp0.venv\Scripts\python.exe"

REM Clear PYTHONPATH so Hermes packages don't shadow our venv
set PYTHONPATH=
set PYTHONHOME=

REM Set model cache location
set HF_HOME=G:\Hunyuan3D-2-Standalone\models

echo Running Hunyuan3D with: %PY%
"%PY%" "%~dp0run_hy3d.py" %*

endlocal
