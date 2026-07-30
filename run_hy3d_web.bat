@echo off
REM Hunyuan3D-2.1 Gradio Web UI launcher (local-only, no HF downloads)
setlocal

set "PY=%~dp0.venv\Scripts\python.exe"
set PYTHONPATH=
set PYTHONHOME=

:: Point hy3dgen to our local model cache root
set HY3DGEN_MODELS=G:\hf_cache
set HF_HOME=G:\hf_cache
set TRANSFORMERS_CACHE=G:\hf_cache

:: Force offline mode to prevent any HuggingFace downloads
set HF_HUB_OFFLINE=1

echo Starting Hunyuan3D-2.1 Web UI at http://localhost:7860 ...
"%PY%" "%~dp0gradio_app.py" ^
    --port 7860 ^
    --low_vram_mode ^
    --model_path "G:\hf_cache\models--tencent--Hunyuan3D-2\snapshots\9cd649ba6913f7a852e3286bad86bfa9a2d83dcf" ^
    --subfolder "hunyuan3d-dit-v2-0" ^
    --texgen_model_path "G:\hf_cache\models--tencent--Hunyuan3D-2\snapshots\9cd649ba6913f7a852e3286bad86bfa9a2d83dcf"

endlocal
