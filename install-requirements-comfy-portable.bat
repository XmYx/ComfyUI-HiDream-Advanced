@echo off
echo ========================================================================
echo HiDream Sampler - ComfyUI-portable Dependency Installer
echo ========================================================================
echo Created by SanDiegoDude (https://github.com/SanDiegoDude/ComfyUI-HiDream-Sampler)
echo.
echo This script will install dependencies for the HiDream Sampler node.
echo This is specifically for ComfyUI-portable users.
echo.

rem Move up to the ComfyUI root directory from custom_nodes/ComfyUI-HiDream-Sampler
cd ..\..\

set COMFY_ROOT=%CD%
echo ComfyUI root directory: %COMFY_ROOT%

rem Check if python_embedded exists
if not exist "python_embedded\python.exe" (
    echo ERROR: Could not find python_embedded\python.exe
    echo This script is intended for ComfyUI-portable installations only.
    echo For standard installations, please use your virtual environment.
    goto :end
)

echo.
echo Installing basic dependencies...
python_embedded\python.exe -s -m pip install -r custom_nodes\ComfyUI-HiDream-Sampler\requirements.txt

echo.
echo Installing NF4 model dependencies (optimum, accelerate)...
python_embedded\python.exe -s -m pip install optimum accelerate

echo.
echo Would you like to install BitsAndBytes for 4-bit quantization models? (y/n)
echo (Required for non-NF4 models)
set /p CHOICE_BNB="> "
if /i "%CHOICE_BNB%"=="y" (
    echo.
    echo Installing BitsAndBytes...
    python_embedded\python.exe -s -m pip install bitsandbytes
)

echo.
echo Would you like to install Flash Attention for faster processing? (y/n)
echo (Optional but recommended)
set /p CHOICE_FLASH="> "
if /i "%CHOICE_FLASH%"=="y" (
    echo.
    echo Installing Flash Attention...
    python_embedded\python.exe -s -m pip install flash-attn
)

echo.
echo ========================================================================
echo IMPORTANT NOTE ABOUT AUTO-GPTQ:
echo ========================================================================
echo The HiDream sampler uses transformers.GPTQModel by default for NF4 models.
echo auto-gptq is only used as a fallback and has known issues:
echo  - It is NOT compatible with Python 3.12
echo  - It often has installation problems on Windows
echo.
echo Would you like to attempt installing auto-gptq anyway? (y/n)
echo (Only recommended if transformers.GPTQModel isn't working)
set /p CHOICE_AUTOGPTQ="> "
if /i "%CHOICE_AUTOGPTQ%"=="y" (
    echo.
    echo Installing auto-gptq (this may fail on some systems)...
    python_embedded\python.exe -s -m pip install auto-gptq
)

echo.
echo ========================================================================
echo Dependencies installed! You may now start ComfyUI.
echo If you encounter any issues, please report them on GitHub:
echo https://github.com/SanDiegoDude/ComfyUI-HiDream-Sampler
echo ========================================================================

:end
rem Navigate back to the original directory
cd custom_nodes\ComfyUI-HiDream-Sampler
pause
