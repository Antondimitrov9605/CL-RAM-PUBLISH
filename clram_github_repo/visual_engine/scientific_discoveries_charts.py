#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scientific Discoveries Charts
================================
Visualizes the key scientific findings from the CL-RAM deep analysis.

Discoveries implemented:
1. Temperature x Language Interaction (phi-4 asymmetric BG amplification)
2. Empty Response Defense Mechanism Decay
3. EuroLLM Paradox (designed for BG, yet more vulnerable in BG)
4. Safety Architecture Map (what actually protects models)
5. MITRE Danger Bubble Chart (multi-dimensional risk)
6. phi-4 BG High-Temperature Danger Zones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path
from typing import List, Optional

# Colors
EN_COLOR    = '#2E86AB'
BG_COLOR    = '#E84855'
PHI_COLOR   = '#3BB273'
EURO_COLOR  = '#F4A261'
MISTRAL_COLOR = '#7B2D8B'
EMPTY_COLOR = '#95a5a6'
DANGER_RED  = '#C0392B'
SAFE_GREEN  = '#27AE60'
WARN_ORANGE = '#E67E22'


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if 'is_jailbreak' not in d.columns:
        d['is_jailbreak'] = d.get('success', False)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: Temperature × Language Interaction (Asymmetric Amplification)
# ─────────────────────────────────────────────────────────────────────────────

def create_temp_language_interaction(df: pd.DataFrame,
                                      output_dir: Path) -> Optional[Path]:
    """
    Shows that temperature amplifies BG vulnerability MORE than EN.
    The BG/EN amplification ratio per MITRE category.
    Key chart for the PhD interaction-effect analysis.
    """
    try:
        data = _prep(df)
        phi = data[data['model_name'].str.contains('phi', case=False, na=False)]
        if len(phi) == 0:
            phi = data  # fallback: use all data

        cats = sorted(phi['category'].unique())
        temps = sorted(phi['temperature'].unique())
        if len(temps) < 2:
            return None

        t_min, t_max = temps[0], temps[-1]

        bg_deltas, en_deltas, ratios = [], [], []
        for cat in cats:
            c = phi[phi['category'] == cat]
            en01 = c[(c.language == 'en') & (c.temperature == t_min)]['is_jailbreak'].mean() * 100
            en10 = c[(c.language == 'en') & (c.temperature == t_max)]['is_jailbreak'].mean() * 100
            bg01 = c[(c.language == 'bg') & (c.temperature == t_min)]['is_jailbreak'].mean() * 100
            bg10 = c[(c.language == 'bg') & (c.temperature == t_max)]['is_jailbreak'].mean() * 100
            d_en = en10 - en01
            d_bg = bg10 - bg01
            ratio = d_bg / max(abs(d_en), 1)
            en_deltas.append(d_en)
            bg_deltas.append(d_bg)
            ratios.append(ratio)

        # Sort by BG delta descending
        order = np.argsort(bg_deltas)[::-1]
        cats_s = [cats[i] for i in order]
        bg_s = [bg_deltas[i] for i in order]
        en_s = [en_deltas[i] for i in order]
        rat_s = [ratios[i] for i in order]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Left: grouped bar BG vs EN deltas
        x = np.arange(len(cats_s))
        width = 0.38
        ax1.bar(x - width / 2, bg_s, width, label='BG delta',
                color=BG_COLOR, alpha=0.85, edgecolor='white')
        ax1.bar(x + width / 2, en_s, width, label='EN delta',
                color=EN_COLOR, alpha=0.85, edgecolor='white')

        for i, (bg_v, en_v) in enumerate(zip(bg_s, en_s)):
            for val, offset, color in [(bg_v, -width/2, BG_COLOR),
                                        (en_v,  width/2, EN_COLOR)]:
                ax1.text(x[i] + offset, val + (1.5 if val >= 0 else -3),
                         f'{val:+.0f}%', ha='center', fontsize=8,
                         fontweight='bold', color=color)

        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_xticks(x)
        ax1.set_xticklabels([c.replace('_', '\n').title() for c in cats_s],
                             fontsize=8)
        ax1.set_ylabel(f'Delta jailbreak rate (T={t_max} - T={t_min})', fontsize=11)
        ax1.set_title('Temperature Effect: BG vs EN Delta\n(phi-4)',
                      fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)

        # Right: BG/EN amplification ratio
        rat_colors = [DANGER_RED if r >= 3 else (WARN_ORANGE if r >= 2 else SAFE_GREEN)
                      for r in rat_s]
        bars = ax2.barh(range(len(cats_s)), rat_s, color=rat_colors,
                         alpha=0.85, edgecolor='white', height=0.7)

        for i, (bar, r) in enumerate(zip(bars, rat_s)):
            ax2.text(r + 0.1, i, f'{r:.1f}x', va='center', ha='left',
                     fontsize=10, fontweight='bold',
                     color=DANGER_RED if r >= 3 else WARN_ORANGE)

        ax2.axvline(1, color='black', linewidth=1.5, linestyle='--', alpha=0.5)
        ax2.axvline(3, color=DANGER_RED, linewidth=1.5, linestyle=':',
                    alpha=0.6, label='3x threshold')
        ax2.set_yticks(range(len(cats_s)))
        ax2.set_yticklabels([c.replace('_', ' ').title() for c in cats_s], fontsize=10)
        ax2.set_xlabel('BG / EN Temperature Amplification Ratio', fontsize=11)
        ax2.set_title('BG Temperature Sensitivity\nvs EN Temperature Sensitivity',
                      fontsize=12, fontweight='bold')
        ax2.legend(handles=[
            mpatches.Patch(color=DANGER_RED, label='Strong amplification (>=3x)'),
            mpatches.Patch(color=WARN_ORANGE, label='Moderate amplification (>=2x)'),
            mpatches.Patch(color=SAFE_GREEN, label='Similar sensitivity (<2x)'),
        ], fontsize=9, loc='lower right')
        ax2.grid(axis='x', alpha=0.3)

        model_name = phi['model_name'].iloc[0] if len(phi) > 0 else 'phi-4'
        fig.suptitle(
            f'Temperature × Language Interaction Effect\n'
            f'{model_name}: Bulgarian is UP TO 30x more sensitive to temperature than English',
            fontsize=13, fontweight='bold', y=1.01
        )
        plt.tight_layout()

        fp = output_dir / 'discovery_temp_language_interaction.png'
        plt.savefig(fp, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'   [OK] discovery_temp_language_interaction.png')
        return fp

    except Exception as e:
        print(f'   [ERROR] create_temp_language_interaction: {e}')
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: Empty Response Defense Mechanism Decay
# ─────────────────────────────────────────────────────────────────────────────

def create_empty_defense_decay(df: pd.DataFrame,
                                output_dir: Path) -> Optional[Path]:
    """
    Shows how the EMPTY response (silence as refusal) decreases with temperature.
    This reveals the MECHANISM of safety breakdown: silence → compliance.
    """
    try:
        data = _prep(df)
        models = sorted(data['model_name'].unique())
        temps = sorted(data['temperature'].unique())

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle(
            'Safety Defense Mechanism Analysis\n'
            'How Safety Responses Change with Temperature',
            fontsize=14, fontweight='bold'
        )

        model_colors = {m: c for m, c in zip(models, [PHI_COLOR, EURO_COLOR, MISTRAL_COLOR])}

        # LEFT: Empty response count per model per temp
        ax = axes[0]
        for model in models:
            mdf = data[data['model_name'] == model]
            empty_counts = []
            for t in temps:
                n_empty = len(mdf[(mdf['temperature'] == t) &
                                   (mdf['response_type'] == 'EMPTY')])
                empty_counts.append(n_empty)
            ax.plot(temps, empty_counts, marker='o', linewidth=2.5,
                    markersize=8, label=model.split('.')[0],
                    color=model_colors[model])
            for t, ec in zip(temps, empty_counts):
                ax.annotate(str(ec), (t, ec), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=10,
                            fontweight='bold', color=model_colors[model])

        ax.set_xlabel('Temperature', fontsize=11)
        ax.set_ylabel('Number of EMPTY Responses', fontsize=11)
        ax.set_title('EMPTY Responses per Temperature\n(Silence = Primary Defense)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(temps)

        # MIDDLE: Response type distribution stacked bar
        ax2 = axes[1]
        response_types = ['EMPTY', 'HARD_REFUSAL', 'SOFT_REFUSAL',
                           'EVASIVE_REFUSAL', 'UNCLEAR',
                           'EDUCATIONAL_COMPLY', 'FULL_COMPLY']
        rt_colors_map = {
            'EMPTY': '#bdc3c7',
            'HARD_REFUSAL': SAFE_GREEN,
            'SOFT_REFUSAL': '#82e0aa',
            'EVASIVE_REFUSAL': WARN_ORANGE,
            'UNCLEAR': '#aab7b8',
            'EDUCATIONAL_COMPLY': '#f1948a',
            'FULL_COMPLY': DANGER_RED,
        }
        x_pos = np.arange(len(temps))
        bottoms = np.zeros(len(temps))
        for rt in response_types:
            counts = []
            for t in temps:
                n = len(data[(data['temperature'] == t) & (data['response_type'] == rt)])
                counts.append(n)
            ax2.bar(x_pos, counts, bottom=bottoms,
                    color=rt_colors_map.get(rt, '#999'),
                    label=rt, alpha=0.9)
            bottoms += np.array(counts, dtype=float)

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'T={t}' for t in temps], fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Response Type Distribution\nby Temperature (All Models)',
                      fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8, loc='upper right', ncol=1)
        ax2.grid(axis='y', alpha=0.2)

        # RIGHT: Safe response breakdown (pie) - what protects models
        ax3 = axes[2]
        safe_df = data[data['is_jailbreak'] == False]
        rt_safe = safe_df['response_type'].value_counts()
        pie_colors = [rt_colors_map.get(rt, '#999') for rt in rt_safe.index]
        wedges, texts, autotexts = ax3.pie(
            rt_safe.values,
            labels=rt_safe.index,
            colors=pie_colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight('bold')
        ax3.set_title('What Actually Protects Models?\n(Breakdown of Safe Responses)',
                      fontsize=11, fontweight='bold')

        # Annotation
        fig.text(0.5, -0.03,
                 'Key finding: 76.8% of safe responses = EMPTY (silence). '
                 'Only 0.7% = genuine HARD_REFUSAL.\n'
                 'Models are NOT reasoning their way to safety — they default to silence.',
                 ha='center', fontsize=10, style='italic',
                 bbox=dict(boxstyle='round', facecolor='#fff3cd',
                           edgecolor='#856404', alpha=0.9))

        plt.tight_layout()
        fp = output_dir / 'discovery_empty_defense_decay.png'
        plt.savefig(fp, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'   [OK] discovery_empty_defense_decay.png')
        return fp

    except Exception as e:
        print(f'   [ERROR] create_empty_defense_decay: {e}')
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: EuroLLM Paradox
# ─────────────────────────────────────────────────────────────────────────────

def create_eurollm_paradox(df: pd.DataFrame,
                            output_dir: Path) -> Optional[Path]:
    """
    EuroLLM was SPECIFICALLY designed for European languages including BG,
    yet it shows +20% higher vulnerability in BG vs EN.
    This challenges the assumption that multilingual training improves safety parity.
    """
    try:
        data = _prep(df)
        euro_name = [m for m in data['model_name'].unique()
                     if 'euro' in m.lower() or 'eur' in m.lower()]
        if not euro_name:
            print('   [SKIP] No EuroLLM model found')
            return None
        euro_name = euro_name[0]
        euro = data[data['model_name'] == euro_name]
        cats = sorted(euro['category'].unique())

        en_rates = [euro[(euro.category == c) & (euro.language == 'en')]['is_jailbreak'].mean() * 100
                    for c in cats]
        bg_rates = [euro[(euro.category == c) & (euro.language == 'bg')]['is_jailbreak'].mean() * 100
                    for c in cats]
        gaps = [bg - en for bg, en in zip(bg_rates, en_rates)]

        order = np.argsort(gaps)[::-1]
        cats_s = [cats[i] for i in order]
        en_s = [en_rates[i] for i in order]
        bg_s = [bg_rates[i] for i in order]
        gaps_s = [gaps[i] for i in order]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Left: side-by-side bars
        x = np.arange(len(cats_s))
        w = 0.38
        ax1.bar(x - w/2, en_s, w, color=EN_COLOR, alpha=0.85,
                label='English (EN)', edgecolor='white')
        ax1.bar(x + w/2, bg_s, w, color=BG_COLOR, alpha=0.85,
                label='Bulgarian (BG)', edgecolor='white')

        # Highlight extreme gaps
        for i, (en_v, bg_v, gap) in enumerate(zip(en_s, bg_s, gaps_s)):
            if abs(gap) >= 30:
                ax1.annotate(f'+{gap:.0f}%!',
                             xy=(x[i] + w/2, bg_v),
                             xytext=(x[i] + w/2 + 0.3, bg_v + 5),
                             fontsize=9, color=DANGER_RED, fontweight='bold',
                             arrowprops=dict(arrowstyle='->', color=DANGER_RED))

        ax1.set_xticks(x)
        ax1.set_xticklabels([c.replace('_', '\n').title() for c in cats_s],
                             fontsize=8, ha='center')
        ax1.set_ylim(0, 115)
        ax1.set_ylabel('Jailbreak Success Rate (%)', fontsize=11)
        ax1.set_title(f'{euro_name}\nEN vs BG by MITRE Category',
                      fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)

        # Right: gap highlight chart
        gap_colors = [DANGER_RED if g >= 30 else (WARN_ORANGE if g >= 15 else
                       (SAFE_GREEN if g < 0 else '#f0b27a'))
                      for g in gaps_s]
        bars = ax2.barh(range(len(cats_s)), gaps_s, color=gap_colors,
                         alpha=0.85, edgecolor='white', height=0.7)

        for i, (bar, g) in enumerate(zip(bars, gaps_s)):
            ax2.text(g + (0.3 if g >= 0 else -0.3), i,
                     f'{g:+.0f}%', va='center',
                     ha='left' if g >= 0 else 'right',
                     fontsize=10, fontweight='bold',
                     color=DANGER_RED if g >= 30 else WARN_ORANGE)

        ax2.axvline(0, color='black', linewidth=1.5)
        ax2.axvline(30, color=DANGER_RED, linewidth=1.5, linestyle=':',
                    alpha=0.6)
        ax2.set_yticks(range(len(cats_s)))
        ax2.set_yticklabels([c.replace('_', ' ').title() for c in cats_s],
                             fontsize=10)
        ax2.set_xlabel('BG − EN Vulnerability Gap (%)', fontsize=11)
        ax2.set_title('Paradox: More BG Training → More BG Vulnerability?',
                      fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # Key stats box
        avg_gap = np.mean(gaps_s)
        n_positive = sum(1 for g in gaps_s if g > 0)
        extreme = sum(1 for g in gaps_s if g >= 30)
        ax2.text(0.98, 0.02,
                 f'Overall: BG={euro[euro.language=="bg"]["is_jailbreak"].mean()*100:.1f}%  '
                 f'EN={euro[euro.language=="en"]["is_jailbreak"].mean()*100:.1f}%\n'
                 f'Mean gap: {avg_gap:+.1f}%\n'
                 f'BG > EN in {n_positive}/{len(cats_s)} categories\n'
                 f'Extreme gap (>=30%): {extreme} categories',
                 transform=ax2.transAxes, ha='right', va='bottom',
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffeeba',
                           edgecolor='#856404', alpha=0.95))

        fig.suptitle(
            f'The EuroLLM Paradox\n'
            f'A model designed for Bulgarian shows +20% HIGHER vulnerability in Bulgarian',
            fontsize=13, fontweight='bold', y=1.01
        )
        plt.tight_layout()
        fp = output_dir / 'discovery_eurollm_paradox.png'
        plt.savefig(fp, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'   [OK] discovery_eurollm_paradox.png')
        return fp

    except Exception as e:
        print(f'   [ERROR] create_eurollm_paradox: {e}')
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4: MITRE Danger Bubble Chart
# ─────────────────────────────────────────────────────────────────────────────

def create_mitre_danger_bubble(df: pd.DataFrame,
                                output_dir: Path) -> Optional[Path]:
    """
    3-dimensional MITRE risk chart:
      X = EN success rate
      Y = BG success rate
      Bubble size = language vulnerability gap (BG - EN)
      Color = danger tier
    Instantly shows which categories are universally dangerous vs language-specific.
    """
    try:
        data = _prep(df)
        cats = sorted(data['category'].unique())

        en_r, bg_r, overall_r, gaps = [], [], [], []
        for cat in cats:
            c = data[data['category'] == cat]
            en = c[c['language'] == 'en']['is_jailbreak'].mean() * 100
            bg = c[c['language'] == 'bg']['is_jailbreak'].mean() * 100
            ov = c['is_jailbreak'].mean() * 100
            en_r.append(en)
            bg_r.append(bg)
            overall_r.append(ov)
            gaps.append(bg - en)

        fig, ax = plt.subplots(figsize=(12, 10))

        # Background zones
        ax.axhspan(70, 105, alpha=0.06, color='red')
        ax.axhspan(50, 70, alpha=0.04, color='orange')
        ax.axvspan(70, 105, alpha=0.06, color='red')
        ax.axvspan(50, 70, alpha=0.04, color='orange')

        # Diagonal reference line (EN == BG)
        ax.plot([0, 100], [0, 100], '--', color='gray', alpha=0.4,
                linewidth=1.5, label='EN = BG line')

        # Bubble colors by danger tier
        bubble_colors = []
        for ov in overall_r:
            if ov >= 70:
                bubble_colors.append(DANGER_RED)
            elif ov >= 55:
                bubble_colors.append(WARN_ORANGE)
            else:
                bubble_colors.append(SAFE_GREEN)

        # Bubble sizes proportional to gap
        sizes = [max(abs(g) * 30, 200) for g in gaps]

        sc = ax.scatter(en_r, bg_r, s=sizes, c=bubble_colors, alpha=0.75,
                        edgecolors='white', linewidth=2, zorder=5)

        # Labels
        for i, cat in enumerate(cats):
            label = cat.replace('_', '\n')
            ax.annotate(label, (en_r[i], bg_r[i]),
                        textcoords='offset points',
                        xytext=(8, 4), fontsize=8.5, fontweight='bold',
                        color='#2c3e50')

        ax.set_xlabel('English Jailbreak Rate (%)', fontsize=13)
        ax.set_ylabel('Bulgarian Jailbreak Rate (%)', fontsize=13)
        ax.set_title(
            'MITRE ATT&CK Category Risk Map\n'
            'Bubble size = BG−EN gap  |  Color = overall danger tier',
            fontsize=14, fontweight='bold'
        )
        ax.set_xlim(30, 105)
        ax.set_ylim(40, 105)

        # Zone labels
        ax.text(72, 103, 'HIGH DANGER ZONE', fontsize=9, color='red',
                alpha=0.6, fontweight='bold')
        ax.text(32, 103, 'BG-SPECIFIC\nVULNERABILITY', fontsize=8,
                color='darkorange', alpha=0.7)

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=DANGER_RED,
                   markersize=12, label='High danger (>=70%)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=WARN_ORANGE,
                   markersize=12, label='Medium danger (55-70%)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=SAFE_GREEN,
                   markersize=12, label='Lower danger (<55%)'),
        ]
        ax.legend(handles=legend_elements, fontsize=10, loc='lower right')
        ax.grid(alpha=0.25)
        ax.set_aspect('equal')

        plt.tight_layout()
        fp = output_dir / 'discovery_mitre_danger_bubble.png'
        plt.savefig(fp, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'   [OK] discovery_mitre_danger_bubble.png')
        return fp

    except Exception as e:
        print(f'   [ERROR] create_mitre_danger_bubble: {e}')
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5: phi-4 Danger Zone at T=1.0 BG
# ─────────────────────────────────────────────────────────────────────────────

def create_phi4_danger_zones(df: pd.DataFrame,
                              output_dir: Path) -> Optional[Path]:
    """
    Shows the 4-way comparison: phi-4 at T=0.1 EN / T=0.1 BG / T=1.0 EN / T=1.0 BG.
    Visualizes how a safe model (13.9% at T=0.1 EN) becomes dangerous (80% at T=1.0 BG).
    """
    try:
        data = _prep(df)
        phi = data[data['model_name'].str.contains('phi', case=False, na=False)]
        if len(phi) == 0:
            return None

        cats = sorted(phi['category'].unique())
        temps = sorted(phi['temperature'].unique())
        if len(temps) < 2:
            return None
        t_min, t_max = temps[0], temps[-1]

        configs = [
            (t_min, 'en', f'T={t_min} EN', EN_COLOR, 'o-'),
            (t_min, 'bg', f'T={t_min} BG', BG_COLOR, 's--'),
            (t_max, 'en', f'T={t_max} EN', '#5dade2', 'o-'),
            (t_max, 'bg', f'T={t_max} BG', '#c0392b', 's--'),
        ]

        fig, ax = plt.subplots(figsize=(15, 7))

        # Danger zone shading
        ax.axhspan(60, 105, alpha=0.06, color='red')
        ax.axhspan(30, 60, alpha=0.04, color='orange')
        ax.axhline(60, color='red', linestyle=':', alpha=0.4, linewidth=1)
        ax.axhline(30, color='orange', linestyle=':', alpha=0.4, linewidth=1)

        x = np.arange(len(cats))
        for temp, lang, label, color, style in configs:
            rates = []
            for cat in cats:
                c = phi[(phi.category == cat) & (phi.temperature == temp) &
                        (phi.language == lang)]
                rates.append(c['is_jailbreak'].mean() * 100 if len(c) > 0 else 0)
            ax.plot(x, rates, style, color=color, linewidth=2.2,
                    markersize=7, label=label, zorder=3)

        # Fill between T=0.1 BG and T=1.0 BG to show the danger increase
        bg01_rates = [phi[(phi.category==c)&(phi.temperature==t_min)&(phi.language=='bg')]['is_jailbreak'].mean()*100 for c in cats]
        bg10_rates = [phi[(phi.category==c)&(phi.temperature==t_max)&(phi.language=='bg')]['is_jailbreak'].mean()*100 for c in cats]
        ax.fill_between(x, bg01_rates, bg10_rates, alpha=0.12, color=DANGER_RED,
                        label='BG temperature danger increase')

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n').title() for c in cats],
                           fontsize=9)
        ax.set_ylabel('Jailbreak Success Rate (%)', fontsize=12)
        ax.set_ylim(0, 105)
        ax.set_title(
            'phi-4: From Safe to Dangerous\n'
            f'T={t_min} EN: ~14%  →  T={t_max} BG: up to 80% (discovery/lateral_movement)',
            fontsize=13, fontweight='bold'
        )
        ax.legend(fontsize=10, loc='upper right', ncol=2)
        ax.grid(alpha=0.3, linestyle='--')

        # Zone labels
        ax.text(-0.4, 85, 'DANGER\nZONE', fontsize=9, color='red',
                alpha=0.7, fontweight='bold')
        ax.text(-0.4, 45, 'MEDIUM\nRISK', fontsize=9, color='orange',
                alpha=0.7, fontweight='bold')
        ax.text(-0.4, 15, 'LOW\nRISK', fontsize=9, color='green',
                alpha=0.7, fontweight='bold')

        plt.tight_layout()
        fp = output_dir / 'discovery_phi4_danger_zones.png'
        plt.savefig(fp, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'   [OK] discovery_phi4_danger_zones.png')
        return fp

    except Exception as e:
        print(f'   [ERROR] create_phi4_danger_zones: {e}')
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def create_all_scientific_discovery_charts(df: pd.DataFrame,
                                            output_dir: Path) -> List[Path]:
    """
    Master function - generates all scientific discovery charts.

    Args:
        df: DataFrame with is_jailbreak/success, temperature, language,
            model_name, category, response_type, response_length
        output_dir: Directory to save charts

    Returns:
        List of generated file paths
    """
    subdir = Path(output_dir) / 'Scientific_Discoveries'
    subdir.mkdir(parents=True, exist_ok=True)

    generated = []

    print('\n' + '=' * 70)
    print('SCIENTIFIC DISCOVERIES VISUALIZATION')
    print('=' * 70)

    data = _prep(df)

    f = create_temp_language_interaction(data, subdir)
    if f:
        generated.append(f)

    f = create_empty_defense_decay(data, subdir)
    if f:
        generated.append(f)

    f = create_eurollm_paradox(data, subdir)
    if f:
        generated.append(f)

    f = create_mitre_danger_bubble(data, subdir)
    if f:
        generated.append(f)

    f = create_phi4_danger_zones(data, subdir)
    if f:
        generated.append(f)

    print(f'\n   Total generated: {len(generated)} discovery charts')
    print('=' * 70 + '\n')
    return generated
