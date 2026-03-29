#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack Success Distribution Across Temperature Range
=====================================================
Creates donut charts showing Success vs Failed for each temperature
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

# Temperature range
TEMPERATURE_RANGE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Colors
SUCCESS_COLOR = '#F39C12'  # Orange
FAILED_COLOR = '#95A5A6'   # Gray


def create_temperature_distribution_chart(df: pd.DataFrame, 
                                          model: Optional[str],
                                          language: Optional[str],
                                          output_dir: Path) -> Optional[Path]:
    """
    Create temperature distribution donut chart
    
    Args:
        df: DataFrame with temperature, success columns
        model: Model name (None for all models)
        language: 'bg', 'en', or None for both
        output_dir: Output directory
        
    Returns:
        Path to generated chart file
    """
    try:
        # Filter data
        data = df.copy()
        
        if model:
            data = data[data['model_name'] == model]
        
        if language:
            data = data[data['language'] == language]
        
        if len(data) == 0:
            print(f"WARNING: No data for model={model}, language={language}")
            return None
        
        # Create figure with subplots (2 rows × 5 cols)
        fig, axes = plt.subplots(2, 5, figsize=(22, 10))
        axes = axes.flatten()
        
        # Calculate success rate for each temperature
        for idx, temp in enumerate(TEMPERATURE_RANGE):
            ax = axes[idx]
            
            temp_data = data[data['temperature'] == temp]
            
            if len(temp_data) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.axis('off')
                continue
            
            # Calculate success/failed counts
            success_rate = temp_data['success'].mean() * 100
            failed_rate = 100 - success_rate
            
            # Data for pie chart
            sizes = [success_rate, failed_rate]
            colors = [SUCCESS_COLOR, FAILED_COLOR]
            labels = ['Success', 'Failed']
            
            # Create donut chart
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=None,
                colors=colors,
                autopct='%1.0f%%',
                startangle=90,
                pctdistance=0.85,
                wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
            )
            
            # Style percentages
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(11)
                autotext.set_fontweight('bold')
            
            # Add temperature label in center
            ax.text(0, 0, f'T={temp:.1f}\n{success_rate:.1f}%',
                   ha='center', va='center',
                   fontsize=12, fontweight='bold')
            
            # Add temperature title above
            ax.set_title(f'Temperature {temp:.1f}', fontsize=11, pad=10)
        
        # Create title based on filters
        if model and language:
            lang_label = 'Bulgarian' if language == 'bg' else 'English'
            title = f'Attack Success Distribution Across Temperature Range\n{model} - {lang_label}'
        elif model:
            title = f'Attack Success Distribution Across Temperature Range\n{model} - Overall (BG+EN)'
        elif language:
            lang_label = 'Bulgarian' if language == 'bg' else 'English'
            title = f'Attack Success Distribution Across Temperature Range\nAll Models - {lang_label}'
        else:
            title = 'Attack Success Distribution Across Temperature Range\nAll Models - Overall (BG+EN)'
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Add legend at bottom
        fig.legend(labels=['Success', 'Failed'],
                  loc='lower center',
                  ncol=2,
                  fontsize=12,
                  frameon=True,
                  bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        # Generate filename
        if model and language:
            filename = f"temp_dist_{model.replace(':', '_')}_{language.upper()}.png"
        elif model:
            filename = f"temp_dist_{model.replace(':', '_')}_OVERALL.png"
        elif language:
            filename = f"temp_dist_AllModels_{language.upper()}.png"
        else:
            filename = "temp_dist_AllModels_OVERALL.png"
        
        file_path = output_dir / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR creating distribution chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_all_temperature_distribution_charts(df: pd.DataFrame,
                                               output_dir: Path) -> List[Path]:
    """
    Create all temperature distribution charts
    
    Args:
        df: DataFrame with columns: model_name, temperature, language, success
        output_dir: Directory to save charts
        
    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("\n" + "="*80)
    print("GENERATING TEMPERATURE DISTRIBUTION CHARTS")
    print("="*80)
    
    # Validate required columns
    required_cols = ['model_name', 'temperature', 'language', 'success']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Missing required columns. Need: {required_cols}")
        return generated_files
    
    models = sorted(df['model_name'].unique())
    
    # 1. Overall (all models, BG+EN)
    print("\n[1] Overall (All Models, BG+EN)...")
    overall_file = create_temperature_distribution_chart(df, None, None, output_dir)
    if overall_file:
        generated_files.append(overall_file)
        print(f"   SUCCESS: {overall_file.name}")
    
    # 2. Per-Model Overall (each model, BG+EN)
    for idx, model in enumerate(models, 2):
        print(f"\n[{idx}] {model} Overall (BG+EN)...")
        model_file = create_temperature_distribution_chart(df, model, None, output_dir)
        if model_file:
            generated_files.append(model_file)
            print(f"   SUCCESS: {model_file.name}")
    
    # 3. Per-Model-Per-Language (each model × each language)
    counter = len(models) + 2
    for model in models:
        for lang in ['bg', 'en']:
            lang_label = 'Bulgarian' if lang == 'bg' else 'English'
            print(f"\n[{counter}] {model} - {lang_label}...")
            file = create_temperature_distribution_chart(df, model, lang, output_dir)
            if file:
                generated_files.append(file)
                print(f"   SUCCESS: {file.name}")
            counter += 1
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} distribution charts")
    print("="*80 + "\n")
    
    return generated_files


# Test/Demo
if __name__ == "__main__":
    import json
    
    print("="*80)
    print("TEMPERATURE DISTRIBUTION CHARTS - TEST")
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
    print(f"Models: {df['model_name'].nunique()}")
    
    # Create output directory
    output_dir = Path("temp_distribution_charts")
    
    # Generate charts
    files = create_all_temperature_distribution_charts(df, output_dir)
    
    print(f"\nGenerated files:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    print(f"\nOutput directory: {output_dir.absolute()}")
