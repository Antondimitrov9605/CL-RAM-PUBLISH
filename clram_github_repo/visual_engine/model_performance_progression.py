#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Progression Charts
=====================================
Creates linear progression charts showing model performance ranking.
Generates: Overall, Bulgarian, English versions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def create_linear_progression_chart(df: pd.DataFrame, output_dir: Path, 
                                    language: Optional[str] = None) -> Optional[Path]:
    """
    Create linear progression chart showing models ranked by performance
    
    Args:
        df: DataFrame with test results
        output_dir: Output directory path
        language: 'bg', 'en', or None for overall
    """
    try:
        # Filter by language if specified
        if language:
            data = df[df['language'] == language]
            lang_label = "Bulgarian (BG)" if language == 'bg' else "English (EN)"
            file_suffix = f"_{language.upper()}"
        else:
            data = df
            lang_label = "Overall"
            file_suffix = "_OVERALL"
        
        # Calculate success rate per model
        models = sorted(data['model_name'].unique())
        model_stats = []
        
        for model in models:
            model_data = data[data['model_name'] == model]
            success_rate = model_data['success'].mean() * 100
            model_stats.append({
                'model': model,
                'rate': success_rate
            })
        
        # Sort by success rate (ascending - best at top)
        model_stats = sorted(model_stats, key=lambda x: x['rate'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot progression line
        y_positions = list(range(len(model_stats)))
        x_values = [stat['rate'] for stat in model_stats]
        model_names = [stat['model'] for stat in model_stats]
        
        # Draw connecting line
        ax.plot(x_values, y_positions, 'o-', 
                linewidth=3, markersize=14,
                color='#3498DB',
                markerfacecolor='#3498DB',
                markeredgewidth=0,
                alpha=0.8,
                zorder=2)
        
        # Add value labels
        for x, y, model_name in zip(x_values, y_positions, model_names):
            # Label on the right side
            ax.text(x + 1.5, y, f'{x:.1f}%', 
                   va='center', ha='left',
                   fontsize=13, fontweight='bold',
                   color='#2C3E50')
            
            # Add subtle grid line
            ax.axhline(y=y, color='#BDC3C7', linestyle='--', 
                      linewidth=0.5, alpha=0.3, zorder=1)
        
        # Styling
        ax.set_yticks(y_positions)
        ax.set_yticklabels(model_names, fontsize=11)
        ax.set_xlabel('Attack Success Rate (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'Linear Model Performance Progression\n{lang_label}',
                    fontsize=16, fontweight='bold', pad=20)
        
        # X-axis range
        ax.set_xlim(0, max(x_values) + 10)
        ax.grid(True, axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        
        # Save
        file_path = output_dir / f'model_progression{file_suffix}.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR creating progression chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_comparison_table(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Create a table comparing model performance across languages"""
    try:
        models = sorted(df['model_name'].unique())
        
        table_data = []
        for model in models:
            model_df = df[df['model_name'] == model]
            
            overall = model_df['success'].mean() * 100
            bg = model_df[model_df['language'] == 'bg']['success'].mean() * 100
            en = model_df[model_df['language'] == 'en']['success'].mean() * 100
            
            table_data.append({
                'model': model,
                'overall': overall,
                'bg': bg,
                'en': en
            })
        
        # Sort by overall (descending - worst first for vulnerability emphasis)
        table_data = sorted(table_data, key=lambda x: x['overall'], reverse=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, len(models) + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table rows
        rows = []
        for stat in table_data:
            rows.append([
                stat['model'],
                f"{stat['overall']:.1f}%",
                f"{stat['bg']:.1f}%",
                f"{stat['en']:.1f}%"
            ])
        
        # Headers
        headers = ['Model', 'Overall\nSuccess', 'Bulgarian\nSuccess', 'English\nSuccess']
        
        # Create table
        table = ax.table(cellText=rows, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.4, 0.2, 0.2, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.8)
        
        # Style header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(weight='bold', color='white', fontsize=12)
        
        # Style cells
        for i in range(len(rows)):
            # Model name
            table[(i+1, 0)].set_facecolor('#ECF0F1')
            table[(i+1, 0)].set_text_props(weight='bold', ha='left')
            
            # Overall - color coded
            overall_val = float(rows[i][1].strip('%'))
            if overall_val >= 70:
                color = '#E74C3C'  # Red - high vulnerability
            elif overall_val >= 40:
                color = '#F39C12'  # Orange
            else:
                color = '#2ECC71'  # Green - low vulnerability
            table[(i+1, 1)].set_facecolor(color)
            table[(i+1, 1)].set_text_props(color='white', weight='bold', fontsize=12)
            
            # BG/EN - neutral styling
            table[(i+1, 2)].set_facecolor('#BDC3C7')
            table[(i+1, 2)].set_text_props(fontsize=11)
            table[(i+1, 3)].set_facecolor('#BDC3C7')
            table[(i+1, 3)].set_text_props(fontsize=11)
        
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
        
        file_path = output_dir / 'model_comparison_table.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR creating comparison table: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_all_model_progression_charts(df: pd.DataFrame, output_dir: Path):
    """Generate all model progression visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE PROGRESSION CHARTS")
    print("="*80)
    
    # 1. Overall progression
    print("\n[1/4] Overall progression chart...")
    file1 = create_linear_progression_chart(df, output_dir, language=None)
    if file1:
        generated_files.append(file1)
        print(f"   SUCCESS: {file1.name}")
    
    # 2. Bulgarian progression
    print("\n[2/4] Bulgarian progression chart...")
    file2 = create_linear_progression_chart(df, output_dir, language='bg')
    if file2:
        generated_files.append(file2)
        print(f"   SUCCESS: {file2.name}")
    
    # 3. English progression
    print("\n[3/4] English progression chart...")
    file3 = create_linear_progression_chart(df, output_dir, language='en')
    if file3:
        generated_files.append(file3)
        print(f"   SUCCESS: {file3.name}")
    
    # 4. Comparison table
    print("\n[4/4] Comparison table...")
    file4 = create_comparison_table(df, output_dir)
    if file4:
        generated_files.append(file4)
        print(f"   SUCCESS: {file4.name}")
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} visualizations")
    print("="*80 + "\n")
    
    return generated_files


# Test/Demo
if __name__ == "__main__":
    import json
    
    print("="*80)
    print("MODEL PROGRESSION CHARTS - TEST")
    print("="*80)
    
    # Load real data
    data_file = Path("data/outputs/session_20251107_031023/results_20251107_031023.json")
    
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        exit(1)
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    print(f"\nLoaded {len(df)} tests")
    print(f"Models: {len(df['model_name'].unique())}")
    
    # Create output directory
    output_dir = Path("model_progression_charts")
    
    # Generate charts
    files = create_all_model_progression_charts(df, output_dir)
    
    print(f"\nGenerated files:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    print(f"\nOutput directory: {output_dir.absolute()}")
