# 🛡️ CL-RAM: Cross-Lingual Risk Analysis & Mitigation

**The Scientific Evaluation Framework for Multilingual LLM Safety**

CL-RAM is a specialized research tool designed for the automated auditing, validation, and visualization of safety vulnerabilities in Large Language Models (LLMs), with a specific focus on cross-lingual "Safety Leakage" and adversarial behaviors in low-resource languages.

---

## 🔬 Core Scientific Innovations (The 5 PhD Gems)

The current version of CL-RAM integrates five major research findings discovered during the 2026 audit phase:

1.  **Safety Leakage (38.1%)**: Statistical proof that English-centric safety guardrails fail to generalize, leading to 38.1% of "safe" prompts being exploitable in Bulgarian.
2.  **The Defense Paradox**: Discovery of a behavioral shift where English models refuse via "Silence" (EMPTY), while Bulgarian models decay into "Unclear Rambling" (UNCLEAR).
3.  **Stochastic Safety Collapse**: Quantified a **30x increase** in jailbreak sensitivity at high temperatures ($T=1.0$) for Bulgarian compared to English.
4.  **Complexity Signature**: Identification of a ~2x density delta between adversarial compliance (Long) and refusals (Short).
5.  **The Tiered Maturity Model**: A recursive defense strategy (Pattern -> AI -> Manual) that reduces residual risk by **75%**.

---

## 🏗️ System Architecture (3-Tier Pipeline)

CL-RAM implements a unique "Detection Funnel" to ensure state-of-the-art accuracy:

*   **Layer 1 (Pattern)**: High-speed heuristic filtering for known adversarial patterns.
*   **Layer 2 (AI-on-AI)**: Deep semantic validation using an LLM-based "Judge" to detect subtle jailbreaks.
*   **Layer 3 (Manual)**: Expert-in-the-loop ground truth calibration for a scientific-grade benchmark.

---

## 🛠️ Installation & Setup

1.  **Venev Setup**: `venv_gpu` for CUDA-accelerated inference.
2.  **Model Configuration**: Use `model_runner.py` for `llama.cpp` compatible GGUF models.
3.  **Inference**: Run `main_gui.py` for the full research suite.

---

## 📊 Deep Research Outputs
All scientific visualizations and statistics are automatically exported to:
- `data/outputs/visualizations/`
- `statistics_output/statistics_report.md`

---

## 📜 Publication Reference
This system is the basis for the ArXiv preprint: *"CL-RAM: A Tiered Framework for Quantifying Multilingual Safety Gaps."* (Draft included in `.gemini/antigravity/brain/`)

**Author:** Anton Dimitrov (2026)
