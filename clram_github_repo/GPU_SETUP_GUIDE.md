# CL-RAM GPU Setup Guide (RTX 5090 / Blackwell)
## Ръководство за настройка на GPU ускорение

---

## 📋 Изисквания

| Компонент | Изискване |
|-----------|-----------|
| GPU | NVIDIA RTX 5090 (или друга Blackwell/Ada GPU) |
| NVIDIA Driver | 591.xx или по-нов |
| CUDA Toolkit | 13.0 или по-нов |
| Python | 3.10.x (НЕ 3.14 - няма CUDA wheels!) |
| OS | Windows 10/11 |

---

## 🚀 Бързи стъпки (TL;DR)

```powershell
# 1. Създай venv с Python 3.10
py -3.10 -m venv venv_gpu

# 2. Инсталирай Blackwell CUDA wheel
.\venv_gpu\Scripts\python.exe -m pip install --upgrade pip
.\venv_gpu\Scripts\python.exe -m pip install huggingface_hub
.\venv_gpu\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='marcorez8/llama-cpp-python-windows-blackwell-cuda', filename='llama_cpp_python-0.3.9-cp310-cp310-win_amd64.whl', local_dir='.')"
.\venv_gpu\Scripts\python.exe -m pip install llama_cpp_python-0.3.9-cp310-cp310-win_amd64.whl

# 3. Инсталирай зависимости
.\venv_gpu\Scripts\python.exe -m pip install pandas matplotlib seaborn scipy scikit-learn Pillow psutil langdetect PyQt6

# 4. Стартирай
start.bat
```

---

## 📖 Подробни стъпки

### Стъпка 1: Провери системата

Отвори PowerShell и изпълни:

```powershell
# Провери NVIDIA драйвер
nvidia-smi

# Провери налични Python версии
py -0p

# Провери CUDA Toolkit (ако е инсталиран)
nvcc --version
```

**Очакван резултат:**
- nvidia-smi трябва да покаже RTX 5090 с CUDA 13.x
- py -0p трябва да покаже Python 3.10.x

---

### Стъпка 2: Създай виртуална среда с Python 3.10

```powershell
cd c:\Users\ADMIN\Desktop\clram_v2\clram_publish

# Създай venv с Python 3.10 (НЕ 3.14!)
py -3.10 -m venv venv_gpu
```

**ВАЖНО:** Python 3.14 е твърде нов и няма предкомпилирани CUDA wheels!

---

### Стъпка 3: Ъпгрейдни pip

```powershell
.\venv_gpu\Scripts\python.exe -m pip install --upgrade pip
```

---

### Стъпка 4: Изтегли Blackwell CUDA wheel от HuggingFace

```powershell
# Инсталирай huggingface_hub
.\venv_gpu\Scripts\python.exe -m pip install huggingface_hub

# Изтегли wheel файла
.\venv_gpu\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='marcorez8/llama-cpp-python-windows-blackwell-cuda', filename='llama_cpp_python-0.3.9-cp310-cp310-win_amd64.whl', local_dir='.')"
```

---

### Стъпка 5: Инсталирай llama-cpp-python с CUDA

```powershell
.\venv_gpu\Scripts\python.exe -m pip install llama_cpp_python-0.3.9-cp310-cp310-win_amd64.whl
```

---

### Стъпка 6: Провери дали CUDA DLL е налична

```powershell
Get-ChildItem -Path ".\venv_gpu\Lib\site-packages\llama_cpp\lib" -Include *.dll -Recurse | Select-Object Name
```

**Очакван резултат:**
```
ggml-base.dll
ggml-cpu.dll
ggml-cuda.dll    ← ТОВА Е ВАЖНО!
ggml.dll
llama.dll
llava.dll
```

Ако виждаш `ggml-cuda.dll` - GPU ускорението ще работи!

---

### Стъпка 7: Инсталирай всички зависимости

```powershell
.\venv_gpu\Scripts\python.exe -m pip install pandas matplotlib seaborn scipy scikit-learn Pillow psutil langdetect PyQt6
```

---

### Стъпка 8: Тествай GPU

```powershell
.\venv_gpu\Scripts\python.exe test_gpu.py
```

**Очакван резултат:**
```
[OK] SUCCESS: CUDA/GPU acceleration is AVAILABLE!
```

---

### Стъпка 9: Стартирай приложението

```batch
start.bat
```

Или директно:
```powershell
.\venv_gpu\Scripts\python.exe main_gui.py
```

---

## ⚠️ Често срещани проблеми

### Проблем: "Could not find cudart64_12.dll"
**Решение:** Това е само предупреждение за PyTorch, не влияе на GPU ускорението с llama-cpp-python.

### Проблем: "No module named 'X'"
**Решение:** Инсталирай липсващия модул:
```powershell
.\venv_gpu\Scripts\python.exe -m pip install X
```

### Проблем: "ModuleNotFoundError: No module named 'llama_cpp'"
**Решение:** Преинсталирай llama-cpp-python wheel:
```powershell
.\venv_gpu\Scripts\python.exe -m pip install llama_cpp_python-0.3.9-cp310-cp310-win_amd64.whl --force-reinstall
```

### Проблем: GPU не се използва
**Решение:** Провери дали `ggml-cuda.dll` съществува. Ако не:
1. Изтрий venv_gpu
2. Започни от Стъпка 2 отначало

---

## 📁 Структура на файловете

```
clram_publish/
├── venv_gpu/                  ← Виртуална среда с Python 3.10
├── start.bat                  ← Стартов скрипт
├── test_gpu.py               ← GPU тест скрипт  
├── main_gui.py               ← Главно приложение
├── model_runner.py           ← Model runner с GPU настройки
├── GPU_SETUP_GUIDE.md        ← Това ръководство
└── llama_cpp_python-*.whl    ← Blackwell CUDA wheel файл
```

---

## 🔗 Полезни линкове

- Blackwell CUDA wheels: https://huggingface.co/marcorez8/llama-cpp-python-windows-blackwell-cuda
- llama-cpp-python: https://github.com/abetlen/llama-cpp-python
- NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit

---

*Последна актуализация: 2026-02-05*
*Тествано на: Windows 11 Pro, RTX 5090, Python 3.10.10, CUDA 13.0*
