#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Category-Language Heatmap
=========================
Shows attack success rate by MITRE category and language (BG vs EN).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


def create_category_language_heatmap(df: pd.DataFrame, output_dir: Path, 
                                     model_name: Optional[str] = None) -> Optional[Path]:
    """
    Create heatmap showing Category × Language success rates
    
    Args:
        df: DataFrame with test results
        output_dir: Output directory
        model_name: Specific model name, or None for all models
    """
    try:
        # Filter by model if specified
        if model_name:
            data = df[df['model_name'] == model_name]
            title_suffix = f"\n({model_name})"
            file_suffix = f"_{model_name.replace('-', '_').replace('.', '_')}"
        else:
            data = df
            title_suffix = "\n(CL-RAM Framework)"
            file_suffix = "_ALL_MODELS"
        
        # Get all categories
        categories = sorted(data['category'].unique())
        languages = ['bg', 'en']
        
        # Build matrix: rows = categories, cols = languages
        matrix = []
        for category in categories:
            row = []
            for lang in languages:
                cat_lang_data = data[(data['category'] == category) & 
                                     (data['language'] == lang)]
                if len(cat_lang_data) > 0:
                    success_rate = cat_lang_data['success'].mean() * 100
                    row.append(success_rate)
                else:
                    row.append(0)
            matrix.append(row)
        
        # Convert to numpy array
        matrix = np.array(matrix)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, len(categories) * 0.6 + 2))
        
        # Create heatmap
        sns.heatmap(matrix, 
                   annot=True,           # Show values
                   fmt='.1f',            # 1 decimal place
                   cmap='YlOrRd',        # Yellow-Orange-Red colormap
                   cbar_kws={'label': 'Attack Success Rate (%)'},
                   linewidths=2,         # White lines between cells
                   linecolor='white',
                   vmin=0,
                   vmax=100,
                   ax=ax)
        
        # Set labels
        ax.set_xticklabels(['bg', 'en'], rotation=0, ha='center', fontsize=12, fontweight='bold')
        ax.set_yticklabels(categories, rotation=0, ha='right', fontsize=10)
        
        # Title
        ax.set_title(f'Attack Success Rate by Category and Language{title_suffix}',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Language', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Category', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        file_path = output_dir / f'category_language_heatmap{file_suffix}.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR creating category-language heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_all_category_language_heatmaps(df: pd.DataFrame, output_dir: Path):
    """Generate category-language heatmaps for all models + overall"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("\n" + "="*80)
    print("CATEGORY-LANGUAGE HEATMAPS")
    print("="*80)
    
    # 1. Overall (all models combined)
    print("\n[1/4] Overall heatmap (all models)...")
    file1 = create_category_language_heatmap(df, output_dir, model_name=None)
    if file1:
        generated_files.append(file1)
        print(f"   SUCCESS: {file1.name}")
    
    # 2-4. Per-model heatmaps
    models = sorted(df['model_name'].unique())
    for idx, model in enumerate(models, 2):
        print(f"\n[{idx}/4] Heatmap for {model}...")
        file = create_category_language_heatmap(df, output_dir, model_name=model)
        if file:
            generated_files.append(file)
            print(f"   SUCCESS: {file.name}")
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} heatmaps")
    print("="*80 + "\n")
    
    return generated_files


# Test/Demo
if __name__ == "__main__":
    import json
    
    print("="*80)
    print("CATEGORY-LANGUAGE HEATMAP - TEST")
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
    print(f"Categories: {len(df['category'].unique())}")
    print(f"Languages: {df['language'].unique()}")
    print(f"Models: {len(df['model_name'].unique())}")
    
    # Create output directory
    output_dir = Path("category_language_heatmaps")
    
    # Generate heatmaps
    files = create_all_category_language_heatmaps(df, output_dir)
    
    print(f"\nGenerated files:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    print(f"\nOutput directory: {output_dir.absolute()}")
