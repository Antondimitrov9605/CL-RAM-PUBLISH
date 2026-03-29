#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Lingual Vulnerability Transfer Chart
============================================
Analyzes and visualizes how jailbreak vulnerability transfers
between English (EN) and Bulgarian (BG) across MITRE ATT&CK categories.

Scientific finding: BG is consistently MORE vulnerable than EN
across ALL 14 attack categories - suggesting safety training is
language-biased toward high-resource languages.

Charts generated:
1. EN vs BG Grouped Bar Chart per category (sorted by vulnerability gap)
2. Vulnerability Gap Waterfall Chart (BG - EN delta per category)
3. Cross-Lingual Transfer Matrix Heatmap (per model × language)
4. Language Bias Summary Radar Chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Optional, Dict

EN_COLOR = '#2E86AB'
BG_COLOR = '#E84855'
DELTA_POS_COLOR = '#E84855'   # BG > EN = danger (red)
DELTA_NEG_COLOR = '#3BB273'   # EN > BG = safe (green)
NEUTRAL_COLOR = '#95a5a6'


def _get_lang_rates(df: pd.DataFrame, model: str = None) -> Dict[str, Dict[str, float]]:
    """
    Compute success rates per category for EN and BG.
    Returns {'en': {cat: rate}, 'bg': {cat: rate}}
    """
    data = df.copy()
    if model:
        data = data[data['model_name'] == model]

    success_col = 'is_jailbreak' if 'is_jailbreak' in data.columns else 'success'
    cats = sorted(data['category'].unique())
    result = {'en': {}, 'bg': {}}

    for cat in cats:
        for lang in ['en', 'bg']:
            subset = data[(data['category'] == cat) & (data['language'] == lang)]
            result[lang][cat] = subset[success_col].mean() * 100 if len(subset) > 0 else 0.0

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: EN vs BG Grouped Bar (sorted by gap)
# ─────────────────────────────────────────────────────────────────────────────

def create_en_bg_comparison_bars(df: pd.DataFrame, output_dir: Path,
                                  model: str = None) -> Optional[Path]:
    """
    Grouped bar chart: EN vs BG success rate per MITRE category.
    Sorted descending by BG-EN gap to highlight language vulnerability bias.
    """
    try:
        rates = _get_lang_rates(df, model)
        cats = list(rates['en'].keys())

        en_vals = [rates['en'][c] for c in cats]
        bg_vals = [rates['bg'][c] for c in cats]
        gaps = [bg_vals[i] - en_vals[i] for i in range(len(cats))]

        # Sort by gap descending
        sorted_idx = np.argsort(gaps)[::-1]
        cats_sorted = [cats[i] for i in sorted_idx]
        en_sorted = [en_vals[i] for i in sorted_idx]
        bg_sorted = [bg_vals[i] for i in sorted_idx]
        gaps_sorted = [gaps[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(cats_sorted))
        width = 0.38

        bars_en = ax.bar(x - width / 2, en_sorted, width,
                          label='English (EN)', color=EN_COLOR, alpha=0.87,
                          edgecolor='white', linewidth=1)
        bars_bg = ax.bar(x + width / 2, bg_sorted, width,
                          label='Bulgarian (BG)', color=BG_COLOR, alpha=0.87,
                          edgecolor='white', linewidth=1)

        # Gap annotation between bars
        for i, (en_v, bg_v, gap) in enumerate(zip(en_sorted, bg_sorted, gaps_sorted)):
            mid_y = max(en_v, bg_v) + 3
            color = DELTA_POS_COLOR if gap > 0 else DELTA_NEG_COLOR
            symbol = '▲' if gap > 0 else '▼'
            ax.annotate(f'{symbol}{abs(gap):.1f}%',
                        xy=(x[i], mid_y), ha='center', va='bottom',
                        fontsize=9, color=color, fontweight='bold')

        # Category labels
        ax.set_xticks(x)
        ax.set_xticklabels(
            [c.replace('_', '\n').title() for c in cats_sorted],
            fontsize=9, ha='center'
        )
        ax.set_ylabel('Jailbreak Success Rate (%)', fontsize=12)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        model_label = f' — {model}' if model else ' — All Models'
        ax.set_title(
            f'Cross-Lingual Vulnerability: EN vs BG by MITRE ATT&CK Category{model_label}\n'
            f'Sorted by Language Vulnerability Gap (BG − EN)',
            fontsize=13, fontweight='bold'
        )
        ax.legend(fontsize=11, loc='upper right')

        # Summary stats box
        avg_gap = np.mean(gaps_sorted)
        n_bg_higher = sum(1 for g in gaps_sorted if g > 0)
        summary = (f'Avg gap: {avg_gap:+.1f}%\n'
                   f'BG > EN in {n_bg_higher}/{len(cats_sorted)} categories')
        ax.text(0.01, 0.97, summary, transform=ax.transAxes,
                fontsize=10, va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9))

        plt.tight_layout()
        suffix = f'_{model.replace("/", "_").replace(":", "_")}' if model else '_all_models'
        file_path = output_dir / f'crosslingual_en_bg_bars{suffix}.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'   [OK] crosslingual_en_bg_bars{suffix}.png')
        return file_path

    except Exception as e:
        print(f'   [ERROR] create_en_bg_comparison_bars: {e}')
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: Vulnerability Gap Waterfall
# ─────────────────────────────────────────────────────────────────────────────

def create_vulnerability_gap_waterfall(df: pd.DataFrame,
                                        output_dir: Path) -> Optional[Path]:
    """
    Horizontal waterfall showing BG−EN gap per category.
    Positive = BG more vulnerable (red). Negative = EN more vulnerable (green).
    Key chart for the thesis: shows systematic language bias.
    """
    try:
        rates = _get_lang_rates(df)
        cats = list(rates['en'].keys())
        gaps = [rates['bg'][c] - rates['en'][c] for c in cats]

        sorted_idx = np.argsort(gaps)[::-1]
        cats_sorted = [cats[i] for i in sorted_idx]
        gaps_sorted = [gaps[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(10, max(8, len(cats) * 0.65)))

        bar_colors = [DELTA_POS_COLOR if g > 0 else DELTA_NEG_COLOR
                      for g in gaps_sorted]
        bars = ax.barh(range(len(cats_sorted)), gaps_sorted,
                       color=bar_colors, alpha=0.85, edgecolor='white',
                       linewidth=1.2, height=0.7)

        # Value labels
        for i, (bar, gap) in enumerate(zip(bars, gaps_sorted)):
            ax.text(gap + (0.3 if gap >= 0 else -0.3), i,
                    f'{gap:+.1f}%', va='center',
                    ha='left' if gap >= 0 else 'right',
                    fontsize=11, fontweight='bold',
                    color=DELTA_POS_COLOR if gap > 0 else DELTA_NEG_COLOR)

        ax.axvline(0, color='black', linewidth=1.5)
        ax.set_yticks(range(len(cats_sorted)))
        ax.set_yticklabels(
            [c.replace('_', ' ').title() for c in cats_sorted],
            fontsize=11
        )
        ax.set_xlabel('Vulnerability Gap: BG minus EN (%)', fontsize=12)
        ax.set_title(
            'Cross-Lingual Vulnerability Gap\n'
            'Bulgarian vs English Safety Bias (All Models, All Temperatures)',
            fontsize=14, fontweight='bold'
        )

        # Legend patches
        ax.legend(handles=[
            mpatches.Patch(color=DELTA_POS_COLOR, label='BG more vulnerable (higher risk)'),
            mpatches.Patch(color=DELTA_NEG_COLOR, label='EN more vulnerable (lower risk)'),
        ], fontsize=10, loc='lower right')

        # Key finding annotation
        n_positive = sum(1 for g in gaps_sorted if g > 0)
        avg_gap = np.mean(gaps_sorted)
        ax.text(0.98, 0.98,
                f'BG > EN: {n_positive}/{len(cats_sorted)} categories\n'
                f'Mean gap: {avg_gap:+.1f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3cd',
                          edgecolor='#856404', alpha=0.95))

        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()

        file_path = output_dir / 'crosslingual_gap_waterfall.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'   [OK] crosslingual_gap_waterfall.png')
        return file_path

    except Exception as e:
        print(f'   [ERROR] create_vulnerability_gap_waterfall: {e}')
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: Per-Model Language Transfer Matrix
# ─────────────────────────────────────────────────────────────────────────────

def create_transfer_matrix_heatmap(df: pd.DataFrame,
                                    output_dir: Path) -> Optional[Path]:
    """
    Heatmap matrix: rows=models × columns=categories,
    cell value = BG − EN gap. Shows which model/category combinations
    have the largest cross-lingual vulnerability transfer.
    """
    try:
        models = sorted(df['model_name'].unique())
        success_col = 'is_jailbreak' if 'is_jailbreak' in df.columns else 'success'

        # Get all categories
        cats = sorted(df['category'].unique())

        matrix = np.zeros((len(models), len(cats)))
        for i, model in enumerate(models):
            for j, cat in enumerate(cats):
                en_data = df[(df['model_name'] == model) &
                             (df['category'] == cat) &
                             (df['language'] == 'en')]
                bg_data = df[(df['model_name'] == model) &
                             (df['category'] == cat) &
                             (df['language'] == 'bg')]
                en_rate = en_data[success_col].mean() * 100 if len(en_data) > 0 else 0
                bg_rate = bg_data[success_col].mean() * 100 if len(bg_data) > 0 else 0
                matrix[i, j] = bg_rate - en_rate

        fig, ax = plt.subplots(
            figsize=(max(14, len(cats) * 1.1), max(4, len(models) * 1.5))
        )

        vmax = max(abs(matrix.min()), abs(matrix.max()), 20)
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto',
                       vmin=-vmax, vmax=vmax)

        for i in range(len(models)):
            for j in range(len(cats)):
                val = matrix[i, j]
                text_color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:+.0f}%', ha='center', va='center',
                        fontsize=10, fontweight='bold', color=text_color)

        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(
            [c.replace('_', '\n').title() for c in cats],
            fontsize=9, ha='center'
        )
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=11)
        ax.set_title(
            'Cross-Lingual Transfer Matrix: BG − EN Vulnerability Gap\n'
            '(Red = Bulgarian more vulnerable, Green = English more vulnerable)',
            fontsize=13, fontweight='bold'
        )

        cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cb.set_label('BG − EN Gap (%)', fontsize=10)

        plt.tight_layout()
        file_path = output_dir / 'crosslingual_transfer_matrix.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'   [OK] crosslingual_transfer_matrix.png')
        return file_path

    except Exception as e:
        print(f'   [ERROR] create_transfer_matrix_heatmap: {e}')
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def create_all_crosslingual_charts(df: pd.DataFrame,
                                    output_dir: Path) -> List[Path]:
    """
    Master function - generates all cross-lingual transfer charts.

    Args:
        df: DataFrame with is_jailbreak/success, language, model_name, category
        output_dir: Directory to save charts

    Returns:
        List of generated file paths
    """
    subdir = Path(output_dir) / 'CrossLingual_Transfer_Analysis'
    subdir.mkdir(parents=True, exist_ok=True)

    generated = []

    print('\n' + '=' * 70)
    print('CROSS-LINGUAL TRANSFER ANALYSIS  (BG vs EN Vulnerability)')
    print('=' * 70)

    if 'is_jailbreak' not in df.columns and 'success' in df.columns:
        df = df.copy()
        df['is_jailbreak'] = df['success']

    models = sorted(df['model_name'].unique())
    print(f'   Models: {models}')
    print(f'   Languages: {sorted(df["language"].unique())}')

    # Overall comparison (all models combined)
    f = create_en_bg_comparison_bars(df, subdir, model=None)
    if f:
        generated.append(f)

    # Per-model comparison
    for model in models:
        f = create_en_bg_comparison_bars(df, subdir, model=model)
        if f:
            generated.append(f)

    # Vulnerability gap waterfall
    f = create_vulnerability_gap_waterfall(df, subdir)
    if f:
        generated.append(f)

    # Transfer matrix heatmap
    f = create_transfer_matrix_heatmap(df, subdir)
    if f:
        generated.append(f)

    print(f'\n   Total generated: {len(generated)} cross-lingual charts')
    print('=' * 70 + '\n')
    return generated
