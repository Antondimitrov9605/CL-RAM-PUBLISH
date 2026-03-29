#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Transition Analyzer
==========================
Visualizes the critical temperature T* at which LLM safety mechanisms
collapse non-linearly - analogous to phase transitions in thermodynamics.

Charts generated:
1. Temperature Bifurcation Diagram (per model, EN vs BG)
2. Safety Delta Chart (change from T=0.1 to T=1.0)
3. Sigmoid Fit + T* Detection (when >= 4 temperature points available)
4. Cross-Model T* Comparison

Scientific hypothesis: There exists a critical temperature T* where
safety compliance drops non-linearly, suggesting a phase transition
in the model's safety mechanism activation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Optional, Dict, Tuple

# Color scheme consistent with CL-RAM framework
MODEL_COLORS = {
    0: '#2E86AB',   # Blue
    1: '#E84855',   # Red
    2: '#3BB273',   # Green
    3: '#F4A261',   # Orange
    4: '#7B2D8B',   # Purple
}
EN_COLOR = '#2E86AB'
BG_COLOR = '#E84855'
CRITICAL_COLOR = '#F4A261'
SAFE_ZONE = '#d4edda'
DANGER_ZONE = '#f8d7da'


def get_tested_temperatures(df: pd.DataFrame) -> List[float]:
    """Auto-detect temperatures from data."""
    return sorted([float(t) for t in df['temperature'].unique()])


def compute_success_by_temp(df: pd.DataFrame, model: str = None,
                             language: str = None) -> Dict[float, float]:
    """Compute success rate per temperature."""
    data = df.copy()
    if model:
        data = data[data['model_name'] == model]
    if language:
        data = data[data['language'] == language]

    result = {}
    for temp in sorted(data['temperature'].unique()):
        subset = data[data['temperature'] == temp]
        if len(subset) > 0:
            result[float(temp)] = subset['is_jailbreak'].mean() * 100 \
                if 'is_jailbreak' in data.columns \
                else subset['success'].mean() * 100
    return result


def _sigmoid(x: np.ndarray, x0: float, k: float) -> np.ndarray:
    """Logistic sigmoid function for phase transition fitting."""
    return 100 / (1 + np.exp(-k * (x - x0)))


def fit_sigmoid(temps: List[float], rates: List[float]) -> Optional[Tuple[float, float, float]]:
    """
    Fit sigmoid curve to temperature-success data.
    Returns (T_critical, steepness, R_squared) or None if < 4 data points.
    """
    if len(temps) < 4:
        return None
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(_sigmoid, temps, rates,
                            p0=[0.5, 10],
                            bounds=([0.0, 0], [1.0, 100]),
                            maxfev=5000)
        # R-squared
        y_pred = _sigmoid(np.array(temps), *popt)
        ss_res = np.sum((np.array(rates) - y_pred) ** 2)
        ss_tot = np.sum((np.array(rates) - np.mean(rates)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return float(popt[0]), float(popt[1]), float(r2)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: Bifurcation Diagram
# ─────────────────────────────────────────────────────────────────────────────

def create_bifurcation_diagram(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """
    Temperature bifurcation diagram showing EN vs BG curves per model.
    Main chart for PhD phase-transition analysis.
    """
    try:
        models = sorted(df['model_name'].unique())
        temps = get_tested_temperatures(df)
        n_models = len(models)

        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 7),
                                 sharey=True)
        if n_models == 1:
            axes = [axes]

        fig.suptitle('Temperature Phase Transition Analysis\n'
                     'LLM Safety Mechanism Collapse by Temperature',
                     fontsize=16, fontweight='bold', y=1.02)

        for ax_idx, (ax, model) in enumerate(zip(axes, models)):
            # Safe/danger zone shading
            ax.axhspan(0, 30, alpha=0.08, color='green', label='_nolegend_')
            ax.axhspan(30, 70, alpha=0.06, color='orange', label='_nolegend_')
            ax.axhspan(70, 100, alpha=0.08, color='red', label='_nolegend_')

            for lang, color, marker, label in [
                ('en', EN_COLOR, 'o', 'English'),
                ('bg', BG_COLOR, 's', 'Bulgarian'),
            ]:
                rates_dict = compute_success_by_temp(df, model=model, language=lang)
                if not rates_dict:
                    continue
                x = list(rates_dict.keys())
                y = list(rates_dict.values())

                ax.plot(x, y, color=color, marker=marker, linewidth=2.5,
                        markersize=9, label=label, zorder=3)
                ax.fill_between(x, y, alpha=0.12, color=color)

                # Annotate each data point
                for xi, yi in zip(x, y):
                    ax.annotate(f'{yi:.1f}%', (xi, yi),
                                textcoords="offset points", xytext=(0, 10),
                                ha='center', fontsize=9, color=color, fontweight='bold')

                # Sigmoid fit if enough points
                if len(x) >= 4:
                    fit = fit_sigmoid(x, y)
                    if fit:
                        t_crit, k, r2 = fit
                        x_smooth = np.linspace(min(x), max(x), 300)
                        y_smooth = _sigmoid(x_smooth, t_crit, k)
                        ax.plot(x_smooth, y_smooth, '--', color=color,
                                alpha=0.5, linewidth=1.5)
                        ax.axvline(t_crit, color=CRITICAL_COLOR, linestyle=':',
                                   linewidth=2, alpha=0.8)
                        ax.annotate(f'T*={t_crit:.2f}',
                                    xy=(t_crit, 50),
                                    xytext=(t_crit + 0.03, 55),
                                    fontsize=9, color=CRITICAL_COLOR,
                                    fontweight='bold',
                                    arrowprops=dict(arrowstyle='->', color=CRITICAL_COLOR))

            # Delta annotation between first and last temperature
            if len(temps) >= 2:
                en_rates = compute_success_by_temp(df, model=model, language='en')
                bg_rates = compute_success_by_temp(df, model=model, language='bg')
                if en_rates and bg_rates:
                    t_first, t_last = temps[0], temps[-1]
                    if t_first in en_rates and t_last in en_rates:
                        delta_en = en_rates[t_last] - en_rates[t_first]
                        delta_bg = bg_rates.get(t_last, 0) - bg_rates.get(t_first, 0)
                        delta_text = f'ΔEN={delta_en:+.1f}%\nΔBG={delta_bg:+.1f}%'
                        ax.text(0.97, 0.05, delta_text,
                                transform=ax.transAxes, ha='right', va='bottom',
                                fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.4',
                                          facecolor='lightyellow',
                                          edgecolor='gray', alpha=0.9))

            # Risk zone labels (only first axis)
            if ax_idx == 0:
                ax.text(-0.18, 15, 'LOW\nRISK', transform=ax.get_yaxis_transform(),
                        fontsize=8, color='green', alpha=0.7, ha='right', va='center')
                ax.text(-0.18, 50, 'MEDIUM\nRISK', transform=ax.get_yaxis_transform(),
                        fontsize=8, color='orange', alpha=0.7, ha='right', va='center')
                ax.text(-0.18, 85, 'HIGH\nRISK', transform=ax.get_yaxis_transform(),
                        fontsize=8, color='red', alpha=0.7, ha='right', va='center')

            safe_name = model.replace('/', '_').replace(':', '_')
            ax.set_title(f'{safe_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Temperature', fontsize=11)
            if ax_idx == 0:
                ax.set_ylabel('Jailbreak Success Rate (%)', fontsize=11)
            ax.set_ylim(0, 105)
            ax.set_xlim(min(temps) - 0.05, max(temps) + 0.05)
            ax.set_xticks(temps)
            ax.set_xticklabels([str(t) for t in temps], fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', fontsize=10)

        # Horizontal threshold lines
        for ax in axes:
            ax.axhline(50, color='gray', linestyle=':', alpha=0.4, linewidth=1)

        plt.tight_layout()
        file_path = output_dir / 'phase_transition_bifurcation.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'   [OK] phase_transition_bifurcation.png')
        return file_path

    except Exception as e:
        print(f'   [ERROR] create_bifurcation_diagram: {e}')
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: Safety Delta Chart
# ─────────────────────────────────────────────────────────────────────────────

def create_safety_delta_chart(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """
    Bar chart showing delta (change in success rate) from min to max temperature.
    Highlights which models are most temperature-sensitive.
    Doctoral significance: proves temperature is a deployment risk factor.
    """
    try:
        models = sorted(df['model_name'].unique())
        temps = get_tested_temperatures(df)

        if len(temps) < 2:
            print('   [SKIP] Safety delta chart needs >= 2 temperatures')
            return None

        t_min, t_max = temps[0], temps[-1]

        fig, ax = plt.subplots(figsize=(max(10, len(models) * 3), 7))

        x_positions = []
        bar_labels = []
        colors_list = []
        delta_en_vals = []
        delta_bg_vals = []

        for model in models:
            en_rates = compute_success_by_temp(df, model=model, language='en')
            bg_rates = compute_success_by_temp(df, model=model, language='bg')

            d_en = en_rates.get(t_max, 0) - en_rates.get(t_min, 0)
            d_bg = bg_rates.get(t_max, 0) - bg_rates.get(t_min, 0)
            delta_en_vals.append(d_en)
            delta_bg_vals.append(d_bg)
            bar_labels.append(model.replace('/', '\n'))

        x = np.arange(len(models))
        width = 0.35

        bars_en = ax.bar(x - width / 2, delta_en_vals, width,
                         label='English (EN)', color=EN_COLOR, alpha=0.85,
                         edgecolor='white', linewidth=1.2)
        bars_bg = ax.bar(x + width / 2, delta_bg_vals, width,
                         label='Bulgarian (BG)', color=BG_COLOR, alpha=0.85,
                         edgecolor='white', linewidth=1.2)

        # Value labels
        for bar in bars_en:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + (1.5 if h >= 0 else -3.5),
                    f'{h:+.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=EN_COLOR)
        for bar in bars_bg:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + (1.5 if h >= 0 else -3.5),
                    f'{h:+.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=BG_COLOR)

        ax.axhline(0, color='black', linewidth=1.2)
        ax.axhspan(-100, 0, alpha=0.04, color='green')
        ax.axhspan(0, 100, alpha=0.04, color='red')

        ax.text(0.01, 0.02, '← Safer at higher temperature',
                transform=ax.transAxes, fontsize=9, color='green', alpha=0.8)
        ax.text(0.01, 0.97, '↑ MORE DANGEROUS at higher temperature',
                transform=ax.transAxes, fontsize=9, color='red', alpha=0.8,
                va='top')

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Δ Jailbreak Rate  (T={t_max} minus T={t_min})', fontsize=12)
        ax.set_title(
            f'Temperature Sensitivity Analysis\n'
            f'Change in Safety from T={t_min} to T={t_max}',
            fontsize=14, fontweight='bold'
        )
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(min(min(delta_en_vals), min(delta_bg_vals)) - 10,
                    max(max(delta_en_vals), max(delta_bg_vals)) + 15)

        plt.tight_layout()
        file_path = output_dir / 'phase_transition_safety_delta.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'   [OK] phase_transition_safety_delta.png')
        return file_path

    except Exception as e:
        print(f'   [ERROR] create_safety_delta_chart: {e}')
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: Per-Category Temperature Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def create_category_temperature_sensitivity(df: pd.DataFrame,
                                             output_dir: Path) -> Optional[Path]:
    """
    Heatmap: rows = MITRE categories, columns = temperatures.
    Shows which attack categories are most affected by temperature.
    """
    try:
        cats = sorted(df['category'].unique())
        temps = get_tested_temperatures(df)

        success_col = 'is_jailbreak' if 'is_jailbreak' in df.columns else 'success'

        matrix = np.zeros((len(cats), len(temps)))
        for i, cat in enumerate(cats):
            for j, temp in enumerate(temps):
                subset = df[(df['category'] == cat) & (df['temperature'] == temp)]
                matrix[i, j] = subset[success_col].mean() * 100 if len(subset) > 0 else 0

        # Delta column (last - first temp)
        deltas = matrix[:, -1] - matrix[:, 0]

        fig, (ax_heat, ax_delta) = plt.subplots(
            1, 2, figsize=(4 + len(temps) * 1.5 + 3, max(8, len(cats) * 0.7)),
            gridspec_kw={'width_ratios': [len(temps), 1.5]}
        )

        # Heatmap
        import matplotlib.colors as mcolors
        cmap = plt.cm.RdYlGn_r
        im = ax_heat.imshow(matrix, cmap=cmap, aspect='auto',
                             vmin=0, vmax=100)

        for i in range(len(cats)):
            for j in range(len(temps)):
                val = matrix[i, j]
                text_color = 'white' if val > 65 or val < 20 else 'black'
                ax_heat.text(j, i, f'{val:.0f}%', ha='center', va='center',
                             fontsize=10, fontweight='bold', color=text_color)

        ax_heat.set_xticks(range(len(temps)))
        ax_heat.set_xticklabels([f'T={t}' for t in temps], fontsize=11)
        ax_heat.set_yticks(range(len(cats)))
        ax_heat.set_yticklabels([c.replace('_', ' ').title() for c in cats],
                                 fontsize=10)
        ax_heat.set_title('Jailbreak Rate by Category & Temperature (%)',
                          fontsize=13, fontweight='bold')

        cb = plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)
        cb.set_label('Success Rate (%)', fontsize=10)

        # Delta bar chart
        delta_colors = ['#E84855' if d > 0 else '#3BB273' for d in deltas]
        bars = ax_delta.barh(range(len(cats)), deltas, color=delta_colors,
                              alpha=0.85, edgecolor='white')

        for i, (bar, d) in enumerate(zip(bars, deltas)):
            ax_delta.text(d + (0.5 if d >= 0 else -0.5), i,
                          f'{d:+.1f}%', va='center',
                          ha='left' if d >= 0 else 'right',
                          fontsize=9, fontweight='bold',
                          color='#E84855' if d > 0 else '#3BB273')

        ax_delta.axvline(0, color='black', linewidth=1)
        ax_delta.set_yticks(range(len(cats)))
        ax_delta.set_yticklabels([])
        ax_delta.set_xlabel(f'Delta\n(T={temps[-1]} - T={temps[0]})', fontsize=10)
        ax_delta.set_title('Sensitivity', fontsize=11, fontweight='bold')
        ax_delta.grid(axis='x', alpha=0.3)

        plt.suptitle('MITRE ATT&CK Category Temperature Sensitivity',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()

        file_path = output_dir / 'phase_transition_category_sensitivity.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'   [OK] phase_transition_category_sensitivity.png')
        return file_path

    except Exception as e:
        print(f'   [ERROR] create_category_temperature_sensitivity: {e}')
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def create_all_phase_transition_charts(df: pd.DataFrame,
                                        output_dir: Path) -> List[Path]:
    """
    Master function - generates all phase transition analysis charts.

    Args:
        df: DataFrame with columns: is_jailbreak/success, temperature,
            language, model_name, category
        output_dir: Directory to save charts

    Returns:
        List of generated file paths
    """
    subdir = Path(output_dir) / 'Phase_Transition_Analysis'
    subdir.mkdir(parents=True, exist_ok=True)

    generated = []

    print('\n' + '=' * 70)
    print('PHASE TRANSITION ANALYSIS  (Temperature Safety Collapse)')
    print('=' * 70)

    # Normalize success column
    if 'is_jailbreak' not in df.columns and 'success' in df.columns:
        df = df.copy()
        df['is_jailbreak'] = df['success']

    temps = get_tested_temperatures(df)
    models = sorted(df['model_name'].unique())
    print(f'   Models: {models}')
    print(f'   Temperatures detected: {temps}')
    print(f'   Total records: {len(df)}')

    if len(temps) < 2:
        print('   [WARN] Need >= 2 temperature points. Skipping phase transition charts.')
        return generated

    f = create_bifurcation_diagram(df, subdir)
    if f:
        generated.append(f)

    f = create_safety_delta_chart(df, subdir)
    if f:
        generated.append(f)

    f = create_category_temperature_sensitivity(df, subdir)
    if f:
        generated.append(f)

    print(f'\n   Total generated: {len(generated)} phase transition charts')
    print('=' * 70 + '\n')
    return generated
