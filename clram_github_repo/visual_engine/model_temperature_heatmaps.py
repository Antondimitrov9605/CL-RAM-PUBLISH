#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-Temperature Vulnerability Heatmaps
========================================
Creates 3 heatmaps:
1. Bulgarian only (BG)
2. English only (EN)
3. Overall (BG+EN combined)

Each shows: Model (Y-axis) × Temperature (X-axis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional

# Temperature range
TEMPERATURE_RANGE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def create_model_temperature_heatmap(df: pd.DataFrame, language: str, 
                                      output_dir: Path) -> Optional[Path]:
    """
    Create Model × Temperature heatmap for a specific language
    
    Args:
        df: DataFrame with model_name, temperature, language, success columns
        language: 'bg', 'en', or 'all'
        output_dir: Output directory
        
    Returns:
        Path to generated heatmap file
    """
    try:
        # Filter by language if specified
        if language in ['bg', 'en']:
            data = df[df['language'] == language].copy()
            lang_title = 'Bulgarian' if language == 'bg' else 'English'
        else:
            data = df.copy()
            lang_title = 'Overall (BG+EN)'
        
        if len(data) == 0:
            print(f"WARNING: No data for language={language}")
            return None
        
        # Calculate success rate for each model × temperature combination
        pivot_data = []
        models = sorted(data['model_name'].unique())
        
        for model in models:
            model_data = data[data['model_name'] == model]
            rates = []
            
            for temp in TEMPERATURE_RANGE:
                temp_data = model_data[model_data['temperature'] == temp]
                if len(temp_data) > 0:
                    rate = temp_data['success'].mean() * 100
                else:
                    rate = 0
                rates.append(rate)
            
            # Calculate overall success rate for sorting
            overall_rate = model_data['success'].mean() * 100
            pivot_data.append({
                'model': model,
                'overall': overall_rate,
                'rates': rates
            })
        
        # Sort models by overall success rate (descending)
        pivot_data.sort(key=lambda x: x['overall'], reverse=True)
        
        # Create pivot table
        pivot_matrix = []
        model_labels = []
        
        for item in pivot_data:
            pivot_matrix.append(item['rates'])
            # Add overall rate to model label
            model_labels.append(f"{item['model']} ({item['overall']:.1f}%)")
        
        pivot_matrix = np.array(pivot_matrix)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(6, len(models) * 1.5)))
        
        # Create heatmap with RdYlGn_r colormap (same as category heatmaps)
        sns.heatmap(pivot_matrix,
                    annot=True,
                    fmt='.0f',
                    cmap='RdYlGn_r',  # Green (low) → Yellow → Red (high)
                    cbar_kws={'label': 'Attack Success Rate (%)'},
                    ax=ax,
                    linewidths=1.5,
                    linecolor='white',
                    vmin=0,
                    vmax=100,
                    annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                    yticklabels=model_labels,
                    xticklabels=[f'{t:.1f}' for t in TEMPERATURE_RANGE])
        
        # Styling
        ax.set_title(f'Temperature-Model Vulnerability Heatmap\n{lang_title}',
                     fontsize=17, fontweight='bold', pad=20)
        ax.set_xlabel('Temperature', fontsize=14, fontweight='bold')
        ax.set_ylabel('Model (sorted by vulnerability)', fontsize=14, fontweight='bold')
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save
        if language == 'bg':
            filename = "heatmap_model_temp_BG.png"
        elif language == 'en':
            filename = "heatmap_model_temp_EN.png"
        else:
            filename = "heatmap_model_temp_OVERALL.png"
        
        file_path = output_dir / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR creating {language} heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_all_model_temperature_heatmaps(df: pd.DataFrame, 
                                          output_dir: Path) -> List[Path]:
    """
    Create all 3 Model × Temperature heatmaps
    
    Args:
        df: DataFrame with columns: model_name, temperature, language, success
        output_dir: Directory to save heatmaps
        
    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("\n" + "="*80)
    print("GENERATING MODEL-TEMPERATURE HEATMAPS")
    print("="*80)
    
    # Validate required columns
    required_cols = ['model_name', 'temperature', 'language', 'success']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Missing required columns. Need: {required_cols}")
        return generated_files
    
    # 1. Bulgarian heatmap
    print("\n[1/3] Bulgarian heatmap...")
    bg_file = create_model_temperature_heatmap(df, 'bg', output_dir)
    if bg_file:
        generated_files.append(bg_file)
        print(f"   SUCCESS: {bg_file.name}")
    
    # 2. English heatmap
    print("\n[2/3] English heatmap...")
    en_file = create_model_temperature_heatmap(df, 'en', output_dir)
    if en_file:
        generated_files.append(en_file)
        print(f"   SUCCESS: {en_file.name}")
    
    # 3. Overall heatmap (BG+EN combined)
    print("\n[3/3] Overall heatmap (BG+EN)...")
    overall_file = create_model_temperature_heatmap(df, 'all', output_dir)
    if overall_file:
        generated_files.append(overall_file)
        print(f"   SUCCESS: {overall_file.name}")
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} heatmaps")
    print("="*80 + "\n")
    
    return generated_files


# Test/Demo
if __name__ == "__main__":
    import json
    
    print("="*80)
    print("MODEL-TEMPERATURE HEATMAPS - TEST")
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
    print(f"Languages: {df['language'].nunique()}")
    
    # Create output directory
    output_dir = Path("model_temp_heatmaps")
    
    # Generate heatmaps
    files = create_all_model_temperature_heatmaps(df, output_dir)
    
    print(f"\nGenerated files:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    print(f"\nOutput directory: {output_dir.absolute()}")
