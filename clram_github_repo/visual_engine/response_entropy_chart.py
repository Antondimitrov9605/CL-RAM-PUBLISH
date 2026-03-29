#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Response Entropy / Length Analysis
=====================================
Uses response length as a proxy for response entropy to analyze
the relationship between response complexity and jailbreak success.

Key finding: Jailbreak responses are ~8x longer than safe refusals.
This suggests response length can serve as a lightweight, fast
pre-classifier for safety assessment.

Charts generated:
1. Length Distribution: Jailbreak vs Safe (overlapping histograms)
2. Box Plot: Response length by model × outcome
3. Length as Classifier ROC-style threshold analysis
4. Scatter: Response length vs temperature colored by outcome
5. Per-Category Mean Length Comparison (success vs refusal)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Optional

JB_COLOR = '#E84855'       # Red = jailbreak
SAFE_COLOR = '#3BB273'     # Green = safe
EN_COLOR = '#2E86AB'
BG_COLOR = '#E84855'


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and add computed fields."""
    data = df.copy()
    if 'is_jailbreak' not in data.columns:
        data['is_jailbreak'] = data.get('success', False)
    if 'response_length' not in data.columns and 'response' in data.columns:
        data['response_length'] = data['response'].fillna('').str.len()
    return data


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: Length Distribution Histogram
# ─────────────────────────────────────────────────────────────────────────────

def create_length_distribution(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """
    Overlapping histogram of response lengths for jailbreak vs safe outcomes.
    Shows clear separation that makes length a viable lightweight classifier.
    """
    try:
        data = _prep(df)
        jb_lengths = data[data['is_jailbreak'] == True]['response_length'].dropna()
        safe_lengths = data[data['is_jailbreak'] == False]['response_length'].dropna()

        if len(jb_lengths) == 0 or len(safe_lengths) == 0:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: overlapping histogram
        ax = axes[0]
        max_len = min(int(data['response_length'].quantile(0.98)), 6000)
        bins = np.linspace(0, max_len, 50)

        ax.hist(jb_lengths.clip(upper=max_len), bins=bins, alpha=0.55,
                color=JB_COLOR, label=f'Jailbreak (n={len(jb_lengths):,})',
                density=True)
        ax.hist(safe_lengths.clip(upper=max_len), bins=bins, alpha=0.55,
                color=SAFE_COLOR, label=f'Safe Refusal (n={len(safe_lengths):,})',
                density=True)

        # Mean lines
        ax.axvline(jb_lengths.mean(), color=JB_COLOR, linestyle='--',
                   linewidth=2, label=f'JB mean: {jb_lengths.mean():.0f}')
        ax.axvline(safe_lengths.mean(), color=SAFE_COLOR, linestyle='--',
                   linewidth=2, label=f'Safe mean: {safe_lengths.mean():.0f}')

        ax.set_xlabel('Response Length (characters)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Response Length Distribution\nJailbreak vs Safe Outcomes',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        # Right: box plots per model
        ax2 = axes[1]
        models = sorted(data['model_name'].unique())
        positions = []
        box_data = []
        box_colors = []
        tick_labels = []

        for i, model in enumerate(models):
            base = i * 3
            jb = data[(data['model_name'] == model) &
                      (data['is_jailbreak'] == True)]['response_length'].dropna()
            safe = data[(data['model_name'] == model) &
                        (data['is_jailbreak'] == False)]['response_length'].dropna()

            if len(jb) > 0:
                positions.append(base)
                box_data.append(jb.clip(upper=max_len).values)
                box_colors.append(JB_COLOR)
                tick_labels.append(f'{model[:12]}\nJB')
            if len(safe) > 0:
                positions.append(base + 1.2)
                box_data.append(safe.clip(upper=max_len).values)
                box_colors.append(SAFE_COLOR)
                tick_labels.append(f'{model[:12]}\nSafe')

        if box_data:
            bp = ax2.boxplot(box_data, positions=positions, patch_artist=True,
                             widths=0.9, showfliers=False,
                             medianprops=dict(color='black', linewidth=2))
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax2.set_xticks(positions)
        ax2.set_xticklabels(tick_labels, fontsize=8)
        ax2.set_ylabel('Response Length (characters)', fontsize=12)
        ax2.set_title('Response Length by Model & Outcome', fontsize=13,
                      fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Ratio annotation
        ratio = jb_lengths.mean() / max(safe_lengths.mean(), 1)
        fig.text(0.5, -0.02,
                 f'Key finding: Jailbreak responses are {ratio:.1f}x longer than safe refusals  '
                 f'(JB mean={jb_lengths.mean():.0f}  |  Safe mean={safe_lengths.mean():.0f} chars)',
                 ha='center', fontsize=11, fontweight='bold',
                 color='#2c3e50',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3cd',
                           edgecolor='#856404', alpha=0.9))

        plt.tight_layout()
        file_path = output_dir / 'entropy_length_distribution.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'   [OK] entropy_length_distribution.png')
        return file_path

    except Exception as e:
        print(f'   [ERROR] create_length_distribution: {e}')
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: Threshold Classifier Analysis
# ─────────────────────────────────────────────────────────────────────────────

def create_threshold_classifier(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """
    Shows classification accuracy if we use response_length as a binary
    classifier (length > threshold → jailbreak).
    Finds optimal threshold that maximizes accuracy.
    """
    try:
        data = _prep(df)
        lengths = data['response_length'].fillna(0).values
        labels = data['is_jailbreak'].fillna(False).astype(int).values

        if len(lengths) == 0:
            return None

        # Test thresholds from 0 to 95th percentile
        thresholds = np.linspace(0, np.percentile(lengths, 95), 200)
        accuracies = []
        precisions = []
        recalls = []

        for t in thresholds:
            predicted = (lengths > t).astype(int)
            acc = np.mean(predicted == labels)
            tp = np.sum((predicted == 1) & (labels == 1))
            fp = np.sum((predicted == 1) & (labels == 0))
            fn = np.sum((predicted == 0) & (labels == 1))
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            accuracies.append(acc * 100)
            precisions.append(prec * 100)
            recalls.append(rec * 100)

        best_idx = np.argmax(accuracies)
        best_threshold = thresholds[best_idx]
        best_accuracy = accuracies[best_idx]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(thresholds, accuracies, color='#2E86AB', linewidth=2.5,
                label='Accuracy')
        ax.plot(thresholds, precisions, color=JB_COLOR, linewidth=2,
                linestyle='--', label='Precision', alpha=0.8)
        ax.plot(thresholds, recalls, color=SAFE_COLOR, linewidth=2,
                linestyle=':', label='Recall', alpha=0.8)

        ax.axvline(best_threshold, color='#F4A261', linewidth=2.5,
                   linestyle='-.',
                   label=f'Optimal threshold = {best_threshold:.0f} chars')
        ax.annotate(f'Best accuracy:\n{best_accuracy:.1f}%',
                    xy=(best_threshold, best_accuracy),
                    xytext=(best_threshold + max(thresholds) * 0.05, best_accuracy - 8),
                    fontsize=11, fontweight='bold', color='#F4A261',
                    arrowprops=dict(arrowstyle='->', color='#F4A261'))

        ax.fill_between(thresholds, accuracies, alpha=0.08, color='#2E86AB')
        ax.set_xlabel('Length Threshold (characters)', fontsize=12)
        ax.set_ylabel('Classification Metric (%)', fontsize=12)
        ax.set_title(
            'Response Length as Jailbreak Classifier\n'
            f'Optimal threshold: {best_threshold:.0f} chars → {best_accuracy:.1f}% accuracy',
            fontsize=13, fontweight='bold'
        )
        ax.set_ylim(0, 105)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        file_path = output_dir / 'entropy_threshold_classifier.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'   [OK] entropy_threshold_classifier.png')
        return file_path

    except Exception as e:
        print(f'   [ERROR] create_threshold_classifier: {e}')
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: Length vs Temperature Scatter
# ─────────────────────────────────────────────────────────────────────────────

def create_length_vs_temperature_scatter(df: pd.DataFrame,
                                          output_dir: Path) -> Optional[Path]:
    """
    Scatter plot: temperature (x) vs response_length (y),
    colored by jailbreak outcome.
    Shows whether higher temperature produces longer (more helpful/dangerous) responses.
    """
    try:
        data = _prep(df)
        max_len = int(data['response_length'].quantile(0.97))

        models = sorted(data['model_name'].unique())
        n = len(models)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=True)
        if n == 1:
            axes = [axes]

        fig.suptitle('Response Length vs Temperature by Jailbreak Outcome',
                     fontsize=14, fontweight='bold', y=1.01)

        for ax, model in zip(axes, models):
            mdata = data[data['model_name'] == model].copy()
            mdata['response_length'] = mdata['response_length'].clip(upper=max_len)

            for outcome, color, label, alpha in [
                (True, JB_COLOR, 'Jailbreak', 0.5),
                (False, SAFE_COLOR, 'Safe', 0.5),
            ]:
                subset = mdata[mdata['is_jailbreak'] == outcome]
                if len(subset) == 0:
                    continue
                ax.scatter(
                    subset['temperature'] + np.random.uniform(-0.015, 0.015, len(subset)),
                    subset['response_length'],
                    c=color, alpha=alpha, s=25, label=label, zorder=3
                )
                # Mean line per temperature
                for temp in sorted(subset['temperature'].unique()):
                    t_sub = subset[subset['temperature'] == temp]
                    mean_len = t_sub['response_length'].mean()
                    ax.plot(temp, mean_len, 'D', color=color,
                            markersize=12, zorder=5,
                            markeredgecolor='white', markeredgewidth=1.5)

            ax.set_xlabel('Temperature', fontsize=11)
            if ax == axes[0]:
                ax.set_ylabel('Response Length (chars)', fontsize=11)
            ax.set_title(model.replace('/', '\n'), fontsize=11, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_xticks(sorted(data['temperature'].unique()))

        plt.tight_layout()
        file_path = output_dir / 'entropy_length_vs_temperature.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'   [OK] entropy_length_vs_temperature.png')
        return file_path

    except Exception as e:
        print(f'   [ERROR] create_length_vs_temperature_scatter: {e}')
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def create_all_response_entropy_charts(df: pd.DataFrame,
                                        output_dir: Path) -> List[Path]:
    """
    Master function - generates all response entropy / length analysis charts.

    Args:
        df: DataFrame with response_length (or response), is_jailbreak/success,
            temperature, model_name, category, language
        output_dir: Directory to save charts

    Returns:
        List of generated file paths
    """
    subdir = Path(output_dir) / 'Response_Entropy_Analysis'
    subdir.mkdir(parents=True, exist_ok=True)

    generated = []

    print('\n' + '=' * 70)
    print('RESPONSE ENTROPY / LENGTH ANALYSIS')
    print('=' * 70)

    data = _prep(df)
    print(f'   Total records: {len(data)}')
    jb_count = data['is_jailbreak'].sum()
    safe_count = len(data) - jb_count
    print(f'   Jailbreak: {jb_count}  |  Safe: {safe_count}')

    if 'response_length' not in data.columns:
        print('   [WARN] No response_length column - skipping entropy charts')
        return generated

    mean_jb = data[data['is_jailbreak'] == True]['response_length'].mean()
    mean_safe = data[data['is_jailbreak'] == False]['response_length'].mean()
    print(f'   Mean JB length: {mean_jb:.0f}  |  Mean Safe length: {mean_safe:.0f}  '
          f'|  Ratio: {mean_jb/max(mean_safe,1):.1f}x')

    f = create_length_distribution(data, subdir)
    if f:
        generated.append(f)

    f = create_threshold_classifier(data, subdir)
    if f:
        generated.append(f)

    f = create_length_vs_temperature_scatter(data, subdir)
    if f:
        generated.append(f)

    print(f'\n   Total generated: {len(generated)} entropy charts')
    print('=' * 70 + '\n')
    return generated
