@echo off
echo ========================================
echo CL-RAM with GPU Acceleration (RTX 5090)
echo ========================================
set "PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
echo Starting with Python 3.10 + CUDA...
.\venv_gpu\Scripts\python.exe main_gui.py
pause
