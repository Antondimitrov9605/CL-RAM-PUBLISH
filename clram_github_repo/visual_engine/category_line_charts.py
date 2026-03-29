#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Category Line Charts - Full Module
===================================
Generates:
- Per-model, per-category charts (BG vs EN with data table)
- Per-model overall chart
- Global overall chart (all models)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def create_category_chart(model_name, category, bg_rates, en_rates, temperatures, output_dir):
    """Create a single category chart with BG vs EN comparison and data table"""
    
    # Create figure with space for table on the right
    fig = plt.figure(figsize=(18, 8))
    
    # Main chart takes left 70% of space
    ax_chart = plt.subplot2grid((1, 10), (0, 0), colspan=7)
    
    # BG line (blue)
    ax_chart.plot(temperatures, bg_rates, 'o-', 
            linewidth=3, markersize=10,
            color='#2E86AB', label='Bulgarian (BG)',
            markerfacecolor='white',
            markeredgewidth=2.5,
            markeredgecolor='#2E86AB')
    
    # EN line (orange/red)
    ax_chart.plot(temperatures, en_rates, 'o-', 
            linewidth=3, markersize=10,
            color='#E74C3C', label='English (EN)',
            markerfacecolor='white',
            markeredgewidth=2.5,
            markeredgecolor='#E74C3C')
    
    # Add EXACT value labels (1 decimal place)
    for temp, rate in zip(temperatures, bg_rates):
        ax_chart.text(temp, rate + 2, f'{rate:.1f}%', 
                ha='center', va='bottom', fontsize=8,
                color='#2E86AB', fontweight='bold')
    
    for temp, rate in zip(temperatures, en_rates):
        ax_chart.text(temp, rate - 2, f'{rate:.1f}%', 
                ha='center', va='top', fontsize=8,
                color='#E74C3C', fontweight='bold')
    
    # Styling
    category_title = category.replace('_', ' ').title()
    ax_chart.set_title(f'{model_name}\n{category_title} - Bulgarian vs English Success Rate',
                 fontsize=16, fontweight='bold', pad=15)
    ax_chart.set_xlabel('Temperature', fontsize=13, fontweight='bold')
    ax_chart.set_ylabel('Attack Success Rate (%)', fontsize=13, fontweight='bold')
    ax_chart.grid(True, alpha=0.3, linestyle='--')
    ax_chart.set_ylim(-5, 105)  # Extended padding to keep lines inside chart
    ax_chart.set_xlim(0.05, 1.05)
    ax_chart.set_xticks(temperatures)
    ax_chart.legend(fontsize=11, loc='best', framealpha=0.95)
    
    # Create data table on the right
    ax_table = plt.subplot2grid((1, 10), (0, 7), colspan=3)
    ax_table.axis('off')
    
    # Prepare table data (ASCENDING order - 0.1 at top, 1.0 at bottom)
    table_data = []
    table_data.append(['Temp', 'BG %', 'EN %', 'Diff'])
    for temp, bg, en in zip(temperatures, bg_rates, en_rates):
        diff = bg - en
        table_data.append([
            f'{temp:.1f}',
            f'{bg:.1f}',
            f'{en:.1f}',
            f'{diff:+.1f}'
        ])
    
    # Create table
    table = ax_table.table(cellText=table_data,
                           loc='center',
                           cellLoc='center',
                           colWidths=[0.2, 0.27, 0.27, 0.26])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Header styling
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#E8E8E8')
        cell.set_text_props(weight='bold', fontsize=11)
    
    # Color code difference column
    for i in range(1, len(table_data)):
        diff_val = float(table_data[i][3])
        cell = table[(i, 3)]
        if diff_val > 0:
            cell.set_facecolor('#D4EDDA')  # Light green
            cell.set_text_props(color='#155724', weight='bold')
        elif diff_val < 0:
            cell.set_facecolor('#F8D7DA')  # Light red
            cell.set_text_props(color='#721C24', weight='bold')
    
    # Add table title
    ax_table.text(0.5, 0.95, 'Success Rate Data (%)',
                  ha='center', va='top', fontsize=12, fontweight='bold',
                  transform=ax_table.transAxes)
    
    plt.tight_layout()
    
    # Save
    safe_category = category.replace(' ', '_').replace('/', '_')
    safe_model = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
    chart_file = output_dir / f"{safe_model}_cat_{safe_category}.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_file


def create_model_overall_chart(model_name, bg_rates, en_rates, temperatures, output_dir):
    """Create overall model chart (average across all categories)"""
    
    # Same layout as category charts
    fig = plt.figure(figsize=(18, 8))
    ax_chart = plt.subplot2grid((1, 10), (0, 0), colspan=7)
    
    # BG line
    ax_chart.plot(temperatures, bg_rates, 'o-', 
            linewidth=3.5, markersize=12,
            color='#2E86AB', label='Bulgarian (BG)',
            markerfacecolor='white',
            markeredgewidth=3,
            markeredgecolor='#2E86AB')
    
    # EN line
    ax_chart.plot(temperatures, en_rates, 'o-', 
            linewidth=3.5, markersize=12,
            color='#E74C3C', label='English (EN)',
            markerfacecolor='white',
            markeredgewidth=3,
            markeredgecolor='#E74C3C')
    
    # Value labels
    for temp, bg, en in zip(temperatures, bg_rates, en_rates):
        ax_chart.text(temp, bg + 2, f'{bg:.1f}%', 
                ha='center', va='bottom', fontsize=9,
                color='#2E86AB', fontweight='bold')
        ax_chart.text(temp, en - 2, f'{en:.1f}%', 
                ha='center', va='top', fontsize=9,
                color='#E74C3C', fontweight='bold')
    
    ax_chart.set_title(f'{model_name}\nOverall Attack Success Rate - Bulgarian vs English',
                 fontsize=17, fontweight='bold', pad=15)
    ax_chart.set_xlabel('Temperature', fontsize=14, fontweight='bold')
    ax_chart.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
    ax_chart.grid(True, alpha=0.3)
    ax_chart.set_ylim(-5, 105)  # Extended padding
    ax_chart.set_xlim(0.05, 1.05)
    ax_chart.set_xticks(temperatures)
    ax_chart.legend(fontsize=12, loc='best', framealpha=0.95)
    
    # Add data table
    ax_table = plt.subplot2grid((1, 10), (0, 7), colspan=3)
    ax_table.axis('off')
    
    table_data = [['Temp', 'BG %', 'EN %', 'Diff']]
    for temp, bg, en in zip(temperatures, bg_rates, en_rates):
        diff = bg - en
        table_data.append([f'{temp:.1f}', f'{bg:.1f}', f'{en:.1f}', f'{diff:+.1f}'])
    
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center',
                           colWidths=[0.2, 0.27, 0.27, 0.26])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold', fontsize=11)
    
    for i in range(1, len(table_data)):
        diff_val = float(table_data[i][3])
        cell = table[(i, 3)]
        if diff_val > 0:
            cell.set_facecolor('#D4EDDA')
            cell.set_text_props(color='#155724', weight='bold')
        elif diff_val < 0:
            cell.set_facecolor('#F8D7DA')
            cell.set_text_props(color='#721C24', weight='bold')
    
    ax_table.text(0.5, 0.95, 'Overall Success Rate (%)',
                  ha='center', va='top', fontsize=12, fontweight='bold',
                  transform=ax_table.transAxes)
    
    plt.tight_layout()
    
    safe_model = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
    chart_file = output_dir / f"{safe_model}_OVERALL.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_file


def main():
    print("="*80)
    print("CATEGORY LINE CHARTS - FULL GENERATION")
    print("="*80)
    
    # Load data
    data_file = Path("data/outputs/session_20251107_031023/results_20251107_031023.json")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    print(f"\nLoaded {len(df)} tests")
    
    # Get models and categories
    models = sorted(df['model_name'].unique())
    temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print(f"Models: {len(models)}")
    print(f"Temperatures: {len(temperatures)}")
    
    # Create output directory
    output_dir = Path("line_charts_full")
    output_dir.mkdir(exist_ok=True)
    
    all_files = []
    
    # Process each model
    for model_idx, model in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"[{model_idx}/{len(models)}] Processing: {model}")
        print(f"{'='*80}")
        
        model_data = df[df['model_name'] == model]
        categories = sorted(model_data['category'].unique())
        
        print(f"Categories: {len(categories)}")
        
        # 1. Per-category charts
        print(f"\n  -> Generating category charts...")
        for cat_idx, category in enumerate(categories, 1):
            cat_data = model_data[model_data['category'] == category]
            
            # BG data
            bg_data = cat_data[cat_data['language'] == 'bg']
            bg_rates = [bg_data[bg_data['temperature'] == t]['success'].mean() * 100 
                       if len(bg_data[bg_data['temperature'] == t]) > 0 else 0 
                       for t in temperatures]
            
            # EN data
            en_data = cat_data[cat_data['language'] == 'en']
            en_rates = [en_data[en_data['temperature'] == t]['success'].mean() * 100 
                       if len(en_data[en_data['temperature'] == t]) > 0 else 0 
                       for t in temperatures]
            
            # Create chart
            chart_file = create_category_chart(model, category, bg_rates, en_rates, 
                                               temperatures, output_dir)
            all_files.append(chart_file)
            print(f"     [{cat_idx}/{len(categories)}] {category}")
        
        # 2. Model overall chart
        print(f"\n  -> Generating model overall chart...")
        bg_overall = [model_data[(model_data['language'] == 'bg') & 
                                 (model_data['temperature'] == t)]['success'].mean() * 100 
                     for t in temperatures]
        en_overall = [model_data[(model_data['language'] == 'en') & 
                                 (model_data['temperature'] == t)]['success'].mean() * 100 
                     for t in temperatures]
        
        overall_file = create_model_overall_chart(model, bg_overall, en_overall, 
                                                  temperatures, output_dir)
        all_files.append(overall_file)
        print(f"     Model overall chart created")
    
    # 3. Global chart (all models combined)
    print(f"\n{'='*80}")
    print("CREATING GLOBAL CHART (All Models)")
    print(f"{'='*80}")
    
    bg_global = [df[(df['language'] == 'bg') & (df['temperature'] == t)]['success'].mean() * 100 
                for t in temperatures]
    en_global = [df[(df['language'] == 'en') & (df['temperature'] == t)]['success'].mean() * 100 
                for t in temperatures]
    
    global_file = create_model_overall_chart("ALL MODELS", bg_global, en_global, 
                                             temperatures, output_dir)
    all_files.append(global_file)
    
    print("\n" + "="*80)
    print(f"SUCCESS! Generated {len(all_files)} charts")
    print("="*80)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"\nBreakdown:")
    print(f"  - Category charts: {len(all_files) - len(models) - 1}")
    print(f"  - Model overall charts: {len(models)}")
    print(f"  - Global chart: 1")
    print(f"\nTotal: {len(all_files)} files")


if __name__ == "__main__":
    main()
