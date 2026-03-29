# CL-RAM v2: Cross-Lingual Risk Analysis & Mitigation

**A Systematic Empirical Framework for Evaluating Multilingual LLM Safety Under Adversarial Conditions**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Research_Ready-brightgreen.svg)]()

---

## Overview

CL-RAM (Comparative Language Research for AI Model Security Assessment) is a research framework for evaluating the safety robustness of locally-deployed large language models (LLMs) under adversarial prompts derived from the **MITRE ATT&CK** cybersecurity taxonomy.

The framework conducts controlled experiments across multiple open-weight models, natural languages, and inference temperature settings to quantify **cross-lingual safety gaps** — the phenomenon where English-centric safety alignment fails to generalize to other languages.

### Key Findings (1,680 Controlled Experiments)

- **Language-Safety Gap**: Bulgarian-language prompts yield statistically significant higher jailbreak success rates compared to English across all 14 attack categories (mean gap: +10.5%, p < 0.001)
- **Temperature x Language Interaction**: Up to 30x amplification of vulnerability when combining high temperature (T=1.0) with non-English prompts
- **Multilingual Fine-tuning Paradox**: EuroLLM-22B, specifically fine-tuned for European languages, exhibits paradoxically *greater* Bulgarian vulnerability (+20.0% over English)
- **Structural Defense Degradation**: 76.8% of safe refusals are EMPTY (suppressed) responses, and this defense mechanism degrades by 38.9% at high temperature
- **Response Length as Safety Signal**: 8.0x mean amplification ratio, suggesting feasible real-time safety monitoring

---
---

## Architecture

CL-RAM implements a three-layer **Detection Funnel** for scientific-grade classification accuracy:

| Layer | Method | Description |
|-------|--------|-------------|
| **Layer 1** | Pattern-Based | High-speed heuristic filtering for known adversarial patterns |
| **Layer 2** | AI Ensemble | 10-model LLM ensemble for deep semantic validation (92.3% AI-to-human agreement) |
| **Layer 3** | Manual Expert | Human-in-the-loop ground truth calibration |

### Models Tested

| Model | Parameters | Quantization | Type |
|-------|-----------|-------------|------|
| Mistral-7B | 7B | Q8_0 | General |
| EuroLLM-22B | 22B | Q8_0 | Multilingual (EU) |
| Phi-4 | 14B | Q8_0 | Instruction-tuned |

---

## Project Structure

| File / Directory | Description |
|---|---|
| `main_gui.py` | Main GUI application (research suite) |
| `config.py` | Configuration management |
| `model_runner.py` | llama.cpp model inference engine |
| `memory_manager.py` | GPU/CPU memory management |
| `session_persistence.py` | Session state management |
| `visualization_engine.py` | Scientific visualization generator |
| `advanced_classifier.py` | Advanced response classification |
| `cross_validation_classifier.py` | Cross-validation pipeline |
| `enhanced_validator.py` | Enhanced multi-layer validator |
| `improved_validator.py` | Improved validation with LLM judge |
| `start.bat` | Windows launch script |
| `clram/` | Core framework package |
| `clram/validators/` | Validation pipeline (pattern, LLM, registry) |
| `clram/gui/` | GUI components (manual validation, details) |
| `clram/logging/` | Structured logging system |
| `jailbreak/` | Adversarial testing module (MITRE ATT&CK) |
| `visual_engine/` | Scientific chart generation (25+ types) |
| `Wiki/` | Technical documentation |
| `data/` | Runtime data (inputs, outputs, models, checkpoints) |
| `prompts/` | Prompt templates |

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended, 8GB+ VRAM)
- GGUF-format model files (Mistral-7B, EuroLLM-22B, or Phi-4)

### Setup

```bash
git clone https://github.com/Antondimitrov9605/CL-RAM-PUBLISH.git
cd CL-RAM-PUBLISH

python -m venv venv_gpu
venv_gpu\Scripts\activate  # Windows
# or: source venv_gpu/bin/activate  # Linux/Mac

pip install llama-cpp-python  # with CUDA: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
pip install matplotlib numpy pandas scipy psutil

# Place GGUF model files in data/models/
python main_gui.py
```

For detailed GPU setup, see [GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md).

---

## Usage

```bash
python main_gui.py
```

The GUI provides:

- **Model Selection**: Load and configure GGUF models
- **Experiment Configuration**: Set languages, temperatures, attack categories
- **Batch Execution**: Run controlled experiments across parameter combinations
- **Real-time Visualization**: Monitor results with 25+ scientific chart types
- **Three-Layer Validation**: Pattern → AI Ensemble → Manual annotation pipeline
- **Export**: Generate publication-ready figures and statistical reports

### Attack Categories (MITRE ATT&CK)

14 categories: Reconnaissance, Resource Development, Initial Access, Execution, Persistence, Privilege Escalation, Defense Evasion, Credential Access, Discovery, Lateral Movement, Collection, Command & Control, Exfiltration, and Impact.

---

## Citation

If you use CL-RAM in your research, please cite:

```bibtex
@misc{dimitrov2026clram,
  title={CL-RAM: A Tiered Framework for Quantifying Multilingual Safety Gaps in Locally-Deployed Large Language Models},
  author={Dimitrov, Anton Zdravkov},
  year={2026},
  note={Technical Report}
}
```
<img width="960" height="1032" alt="Screenshot 2026-01-02 140555" src="https://github.com/user-attachments/assets/fa7dd258-249a-4b8b-b8e9-0802bb0c24fe" />
<img width="1915" height="1032" alt="Screenshot 2026-01-02 140238" src="https://github.com/user-attachments/assets/e0cae2ed-224c-42ab-bb87-e86d5c6a34e5" />
<img width="1916" height="1029" alt="Screenshot 2026-01-02 140211" src="https://github.com/user-attachments/assets/4589fb9c-9572-4d21-a1be-4f05775e9dc0" />
<img width="959" height="1026" alt="Screenshot 2026-01-02 140141" src="https://github.com/user-attachments/assets/f6b53bd0-8eee-4832-8e25-d79a01f857a7" />
<img width="960" height="1031" alt="Screenshot 2026-01-02 140104" src="https://github.com/user-attachments/assets/62723579-1408-403d-b502-589c641f76d3" />
<img width="1228" height="1029" alt="Screenshot 2026-01-02 140027" src="https://github.com/user-attachments/assets/89a8d868-10c5-484c-9f96-db1e782be38a" />
<img width="960" height="1033" alt="Screenshot 2026-01-02 135850" src="https://github.com/user-attachments/assets/8168b143-5e4e-4d88-9a5f-42146d5699f4" />
<img width="960" height="1032" alt="Screenshot 2026-01-02 135838" src="https://github.com/user-attachments/assets/23adb343-ad6c-46e0-b0cf-44b2f30aec04" />
<img width="960" height="1031" alt="Screenshot 2026-01-02 135650" src="https://github.com/user-attachments/assets/c5902113-9064-4707-975f-a701919b3fb1" />
<img width="957" height="1032" alt="Screenshot 2026-01-02 135401" src="https://github.com/user-attachments/assets/c844eb55-3964-4556-8d65-b739568486c5" />

---<img width="1473" height="872" alt="amplification_ratio_chart" src="https://github.com/user-attachments/assets/85ff5e88-7239-4a88-a21d-bff0ef122aca" />
<img width="5350" height="2947" alt="cat_bar_ALL_MODELS_OVERALL" src="https://github.com/user-attachments/assets/6bf49a27-e5cb-4707-951f-e4e48300ec9b" />
<img width="4150" height="2342" alt="language_comparison_chart" src="https://github.com/user-attachments/assets/c2830668-b93c-499b-9070-bc97e8955955" />
<img width="3553" height="2355" alt="model_progression_OVERALL" src="https://github.com/user-attachments/assets/3ec0a3be-6c10-48b9-a470-e90c95676647" />
<img width="4110" height="2365" alt="validator_asr_comparison" src="https://github.com/user-attachments/assets/561295a8-2d58-400e-8ac5-c62ce329a51e" />
<img width="3600" height="2100" alt="pipeline_model_comparison" src="https://github.com/user-attachments/assets/74392403-50ce-42ef-86ba-bd995ea0fd5a" />
<img width="2946" height="2050" alt="phase_transition_safety_delta" src="https://github.com/user-attachments/assets/f8044a3b-1692-46f0-82f1-aed022db8c10" />
<img width="3564" height="1760" alt="entropy_threshold_classifier" src="https://github.com/user-attachments/assets/263680a4-cb9c-4d11-a151-c2c81c8a414e" />
<img width="2738" height="2008" alt="pipeline_maturity_model" src="https://github.com/user-attachments/assets/c0d87148-b51e-4972-b650-3c7dee273b48" />


## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Author

**Anton Dimitrov** — MSc Cybersecurity

## Disclaimer

This tool is designed exclusively for **academic security research** and **authorized safety auditing**. The adversarial testing capabilities are intended to identify and help mitigate safety vulnerabilities in LLMs. Use responsibly and in compliance with applicable laws and institutional review policies.
