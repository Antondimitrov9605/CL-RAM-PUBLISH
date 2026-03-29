#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Language Comparison Charts
==========================
Creates two types of language comparison charts:
1. Language Advantage Chart - Diverging horizontal bars (BG vs EN difference)
2. Language Comparison Chart - Grouped vertical bars (absolute values)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List

def create_language_advantage_chart(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """
    Create diverging horizontal bar chart showing language advantage (BG vs EN)
    
    Args:
        df: DataFrame with test results
        output_dir: Output directory path
    
    Returns:
        Path to generated file or None if failed
    """
    try:
        # Calculate success rates per model and language
        models = sorted(df['model_name'].unique())
        model_stats = []
        
        for model in models:
            model_data = df[df['model_name'] == model]
            
            # BG success rate
            bg_data = model_data[model_data['language'] == 'bg']
            bg_rate = bg_data['success'].mean() * 100 if len(bg_data) > 0 else 0
            
            # EN success rate
            en_data = model_data[model_data['language'] == 'en']
            en_rate = en_data['success'].mean() * 100 if len(en_data) > 0 else 0
            
            # Calculate advantage (positive = BG better, negative = EN better)
            advantage = bg_rate - en_rate
            
            # Clean model name for display - remove technical suffixes
            display_model = model.replace('.gguf', '').replace('.Q8_0', '').replace('q8_0', '').replace('_', ' ').replace('-', ' ')
            # Capitalize properly
            display_model = ' '.join(word.capitalize() for word in display_model.split())
            
            model_stats.append({
                'model': display_model,
                'advantage': advantage,
                'bg_rate': bg_rate,
                'en_rate': en_rate
            })
        
        if not model_stats:
            print("WARNING: No data for language advantage chart")
            return None
        
        # Sort by advantage (most negative first)
        model_stats = sorted(model_stats, key=lambda x: x['advantage'])
        
        # Prepare data
        models_display = [x['model'] for x in model_stats]
        advantages = [x['advantage'] for x in model_stats]
        
        # Create figure with more height for better spacing
        fig, ax = plt.subplots(figsize=(16, 7))
        
        # Create horizontal bars with colors based on sign
        colors = ['#e74c3c' if adv < 0 else '#27ae60' for adv in advantages]
        y_pos = np.arange(len(models_display))
        bar_height = 0.6  # Thinner bars for better spacing
        
        bars = ax.barh(y_pos, advantages, height=bar_height, color=colors, alpha=0.9, 
                      edgecolor='white', linewidth=2.5)
        
        # Add value labels with better positioning
        for i, (bar, adv) in enumerate(zip(bars, advantages)):
            width = bar.get_width()
            # Position label further from bar
            label_x = width + (0.5 if width > 0 else -0.5)
            ha = 'left' if width > 0 else 'right'
            
            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                   f'{adv:+.1f}%',
                   ha=ha, va='center',
                   fontsize=13, fontweight='bold',
                   color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='none', alpha=0.8))
        
        # Add vertical line at x=0 (thicker and more visible)
        ax.axvline(x=0, color='#2c3e50', linewidth=3, linestyle='-', alpha=0.9, zorder=0)
        
        # Add labels at top with better styling
        max_abs = max(abs(min(advantages)), abs(max(advantages))) if advantages else 10
        
        # EN Better label (left side)
        ax.text(-max_abs * 0.85, len(models_display) + 0.5, 
               'EN Better', 
               ha='center', va='center',
               fontsize=14, fontweight='bold',
               color='white',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#e74c3c', 
                        edgecolor='white', linewidth=2))
        
        # BG Better label (right side)
        ax.text(max_abs * 0.85, len(models_display) + 0.5, 
               'BG Better', 
               ha='center', va='center',
               fontsize=14, fontweight='bold',
               color='white',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#27ae60', 
                        edgecolor='white', linewidth=2))
        
        # Styling - Y axis (model names)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models_display, fontsize=14, fontweight='600')
        
        # X axis
        ax.set_xlabel('Bulgarian Advantage (%)', fontsize=15, fontweight='bold', labelpad=10)
        
        # Title
        ax.set_title('Language Advantage by Model (BG vs EN)', 
                    fontsize=18, fontweight='bold', pad=25)
        
        # Grid - vertical only
        ax.grid(True, axis='x', alpha=0.25, linestyle='--', linewidth=1)
        ax.set_axisbelow(True)
        
        # Set Y limits for better spacing
        ax.set_ylim(-0.8, len(models_display) + 1.2)
        
        # Set X limits with padding
        x_padding = max_abs * 0.15
        ax.set_xlim(-max_abs - x_padding, max_abs + x_padding)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # Background
        ax.set_facecolor('#fafafa')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        # Save with high DPI for crisp text
        file_path = output_dir / "language_advantage_chart.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] creating language advantage chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_language_comparison_chart(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """
    Create grouped bar chart showing absolute success rates for BG and EN
    
    Args:
        df: DataFrame with test results
        output_dir: Output directory path
    
    Returns:
        Path to generated file or None if failed
    """
    try:
        # Calculate success rates per model and language
        models = sorted(df['model_name'].unique())
        bg_rates = []
        en_rates = []
        models_display = []
        
        for model in models:
            model_data = df[df['model_name'] == model]
            
            # BG success rate
            bg_data = model_data[model_data['language'] == 'bg']
            bg_rate = bg_data['success'].mean() * 100 if len(bg_data) > 0 else 0
            bg_rates.append(bg_rate)
            
            # EN success rate
            en_data = model_data[model_data['language'] == 'en']
            en_rate = en_data['success'].mean() * 100 if len(en_data) > 0 else 0
            en_rates.append(en_rate)
            
            # Clean model name for display
            display_model = model.replace('.gguf', '').replace('.Q8_0', '').replace('_', '-')
            models_display.append(display_model)
        
        if not models_display:
            print("WARNING: No data for language comparison chart")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set up bar positions
        x = np.arange(len(models_display))
        width = 0.35
        
        # Create grouped bars
        bars1 = ax.bar(x - width/2, bg_rates, width, 
                      label='Bulgarian', 
                      color='#3498db', 
                      alpha=0.85,
                      edgecolor='white',
                      linewidth=2)
        
        bars2 = ax.bar(x + width/2, en_rates, width, 
                      label='English', 
                      color='#e74c3c', 
                      alpha=0.85,
                      edgecolor='white',
                      linewidth=2)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%',
                       ha='center', va='bottom',
                       fontweight='bold', fontsize=11,
                       color='black')
        
        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels(models_display, fontsize=12, fontweight='500')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=13, fontweight='bold')
        ax.set_title('Success Rates by Language Across Models', fontsize=17, fontweight='bold', pad=20)
        
        # Legend
        ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
        
        # Grid
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Y-axis limit
        max_rate = max(max(bg_rates), max(en_rates))
        ax.set_ylim(0, max_rate + 15)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Background
        ax.set_facecolor('#f9f9f9')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        # Save
        file_path = output_dir / "language_comparison_chart.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] creating language comparison chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_all_language_charts(df: pd.DataFrame, output_dir: Path):
    """
    Generate both language comparison charts
    
    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("\n" + "="*80)
    print("LANGUAGE COMPARISON CHARTS GENERATION")
    print("="*80)
    
    # 1. Language Advantage Chart (diverging horizontal bars)
    print("\n[1/2] Language Advantage Chart (BG vs EN difference)...")
    f1 = create_language_advantage_chart(df, output_dir)
    if f1:
        generated_files.append(f1)
    
    # 2. Language Comparison Chart (grouped vertical bars)
    print("\n[2/2] Language Comparison Chart (absolute values)...")
    f2 = create_language_comparison_chart(df, output_dir)
    if f2:
        generated_files.append(f2)
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} charts")
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
        create_all_language_charts(df, Path("language_charts_output"))
    else:
        print(f"Data file not found: {data_file}")
