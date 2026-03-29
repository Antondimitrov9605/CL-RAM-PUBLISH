# CL-RAM-PUBLISH# CL-RAM v2: Cross-Lingual Risk Analysis & Mitigation

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

## Architecture

CL-RAM implements a three-layer **Detection Funnel** for scientific-grade classification accuracy:
Layer 1: Pattern-Based Classification
High-speed heuristic filtering for known adversarial patterns
Layer 2: AI Ensemble Validator
10-model LLM ensemble for deep semantic validation (92.3% AI-to-human agreement)
Layer 3: Manual Expert Annotation
Human-in-the-loop ground truth calibration

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

## Installation
## Key Results

### Cross-Lingual Vulnerability Gap
![Cross-Lingual Vulnerability](screenshots/crosslingual_vulnerability.png)
*Bulgarian prompts yield higher jailbreak success rates across all 14 MITRE ATT&CK categories (mean gap: +10.5%)*

### The Cross-Lingual Safety Mirror
![Safety Mirror](screenshots/safety_mirror.png)
*Only 27.9% of prompts are consistently safe across both languages — 17.1% leak from English-safe to Bulgarian-vulnerable*

### Temperature × Language Interaction (30x Amplification)
![Temperature Language Interaction](screenshots/temperature_language_interaction.png)
*Phi-4: Bulgarian vulnerability is amplified up to 30x more than English by temperature increase*

### Three-Layer Detection Pipeline
![Pipeline Maturity Model](screenshots/pipeline_maturity_model.png)
*The tiered validation approach reduces residual risk from 100% to near-zero while balancing computational cost*
### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended, 8GB+ VRAM)
- GGUF-format model files

### Setup
```bash
git clone https://github.com/Antondimitrov9605/CL-RAM-PUBLISH.git
cd CL-RAM-PUBLISH

python -m venv venv_gpu
venv_gpu\Scripts\activate  # Windows

pip install llama-cpp-python
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

The GUI provides model selection, experiment configuration across languages and temperatures, batch execution, real-time visualization with 25+ scientific chart types, three-layer validation pipeline, and publication-ready export.

### Attack Categories (MITRE ATT&CK)

14 categories: Reconnaissance, Resource Development, Initial Access, Execution, Persistence, Privilege Escalation, Defense Evasion, Credential Access, Discovery, Lateral Movement, Collection, Command & Control, Exfiltration, and Impact.

---

## Citation
```bibtex
@misc{dimitrov2026clram,
  title={CL-RAM: A Tiered Framework for Quantifying Multilingual Safety Gaps in Locally-Deployed Large Language Models},
  author={Dimitrov, Anton Zdravkov},
  year={2026},
  note={Technical Report}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Author

**Anton Dimitrov** — MSc Cybersecurity

## Disclaimer

This tool is designed exclusively for **academic security research** and **authorized safety auditing**. Use responsibly and in compliance with applicable laws and institutional review policies.
