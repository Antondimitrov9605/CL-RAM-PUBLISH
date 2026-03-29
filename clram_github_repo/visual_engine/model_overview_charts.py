#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Overview Charts
=====================
Creates a combined figure with two charts:
1. Model Vulnerability Ranking - Horizontal bars showing attack success rate
2. Experiments per Model - Vertical bars showing number of experiments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

def create_model_overview_chart(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """
    Create combined chart showing model vulnerability ranking and experiment counts
    
    Args:
        df: DataFrame with test results
        output_dir: Output directory path
    
    Returns:
        Path to generated file or None if failed
    """
    try:
        # Calculate statistics per model
        models = sorted(df['model_name'].unique())
        model_stats = []
        
        for model in models:
            model_data = df[df['model_name'] == model]
            
            # Success rate
            success_rate = model_data['success'].mean() * 100
            
            # Number of experiments
            num_experiments = len(model_data)
            
            # Clean model name for display
            display_model = model.replace('.gguf', '').replace('.Q8_0', '').replace('q8_0', '').replace('_', ' ').replace('-', ' ')
            display_model = ' '.join(word.capitalize() for word in display_model.split())
            
            model_stats.append({
                'model': display_model,
                'original': model,
                'success_rate': success_rate,
                'num_experiments': num_experiments
            })
        
        if not model_stats:
            print("WARNING: No data for model overview chart")
            return None
        
        # Sort by success rate (descending) for vulnerability ranking
        model_stats_sorted = sorted(model_stats, key=lambda x: x['success_rate'], reverse=True)
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # ========== LEFT CHART: Model Vulnerability Ranking ==========
        models_display = [x['model'] for x in model_stats_sorted]
        success_rates = [x['success_rate'] for x in model_stats_sorted]
        
        # Color gradient based on vulnerability (darker = more vulnerable)
        colors_left = plt.cm.Blues_r(np.linspace(0.3, 0.9, len(models_display)))
        
        y_pos = np.arange(len(models_display))
        bars_left = ax1.barh(y_pos, success_rates, color=colors_left, 
                            edgecolor='white', linewidth=2, alpha=0.9)
        
        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars_left, success_rates)):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{rate:.1f}%',
                    ha='left', va='center',
                    fontsize=12, fontweight='bold',
                    color='black')
        
        # Styling for left chart
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(models_display, fontsize=13, fontweight='600')
        ax1.set_xlabel('Attack Success Rate (%)', fontsize=13, fontweight='bold', labelpad=10)
        ax1.set_title('Model Vulnerability Ranking', fontsize=16, fontweight='bold', pad=15)
        
        # Grid
        ax1.grid(True, axis='x', alpha=0.25, linestyle='--', linewidth=1)
        ax1.set_axisbelow(True)
        
        # Set X limit
        ax1.set_xlim(0, max(success_rates) * 1.15)
        
        # Remove spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(2)
        ax1.spines['bottom'].set_linewidth(2)
        
        # Background
        ax1.set_facecolor('#fafafa')
        
        # ========== RIGHT CHART: Experiments per Model ==========
        # Use original order for experiments (not sorted by success rate)
        models_exp = [x['model'] for x in model_stats]
        num_exps = [x['num_experiments'] for x in model_stats]
        
        # Color gradient for experiment counts (darker = more experiments)
        colors_right = plt.cm.Blues(np.linspace(0.4, 0.9, len(models_exp)))
        
        x_pos = np.arange(len(models_exp))
        bars_right = ax2.bar(x_pos, num_exps, color=colors_right,
                            edgecolor='white', linewidth=2, alpha=0.9, width=0.7)
        
        # Add value labels on top of bars
        for bar in bars_right:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(num_exps) * 0.02,
                    f'{int(height)}',
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold',
                    color='black')
        
        # Styling for right chart
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models_exp, fontsize=13, fontweight='600', rotation=0)
        ax2.set_ylabel('Number of Experiments', fontsize=13, fontweight='bold', labelpad=10)
        ax2.set_title('Experiments per Model', fontsize=16, fontweight='bold', pad=15)
        
        # Grid
        ax2.grid(True, axis='y', alpha=0.25, linestyle='--', linewidth=1)
        ax2.set_axisbelow(True)
        
        # Set Y limit
        ax2.set_ylim(0, max(num_exps) * 1.15)
        
        # Remove spines
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_linewidth(2)
        ax2.spines['bottom'].set_linewidth(2)
        
        # Background
        ax2.set_facecolor('#fafafa')
        
        # Overall figure background
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        # Save
        file_path = output_dir / "model_overview_chart.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white',
                   edgecolor='none', format='png')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] creating model overview chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_all_model_overview_charts(df: pd.DataFrame, output_dir: Path):
    """
    Generate model overview chart
    
    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("\n" + "="*80)
    print("MODEL OVERVIEW CHART GENERATION")
    print("="*80)
    
    print("\n[1/1] Model Overview Chart (Vulnerability + Experiments)...")
    f = create_model_overview_chart(df, output_dir)
    if f:
        generated_files.append(f)
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} chart")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")
    
    return generated_files

if __name__ == "__main__":
    import json
    # Test code
    data_file = Path("data/detailed_academic_logs/results_20251107_031023.json")
    if data_file.exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        create_all_model_overview_charts(df, Path("model_overview_output"))
    else:
        print(f"Data file not found: {data_file}")
