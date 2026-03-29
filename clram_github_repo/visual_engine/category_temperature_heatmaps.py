#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Category-Temperature Heatmap Generator
================================================
Creates 4 levels of heatmaps:
1. Overall (all models, all languages)
2. Per Model (each model, BG+EN combined)
3. Per Model BG (each model, Bulgarian only)
4. Per Model EN (each model, English only)

Author: Anton Dimitrov
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
import warnings

warnings.filterwarnings('ignore')


# Configuration
TEMPERATURE_RANGE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def create_all_category_temperature_heatmaps(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """
    Master function to create all 4 levels of heatmaps
    
    Args:
        df: DataFrame with columns: category, temperature, success, model_name, language
        output_dir: Directory to save heatmaps
        
    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("\n" + "="*80)
    print("GENERATING CATEGORY-TEMPERATURE HEATMAPS")
    print("="*80)
    
    # Validate required columns
    required_cols = ['category', 'temperature', 'success']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Missing required columns. Need: {required_cols}")
        return generated_files
    
    # 1. Overall Heatmap (all models, all languages)
    print("\n[1/4] Generating Overall Heatmap (all data)...")
    overall_file = create_overall_heatmap(df, output_dir)
    if overall_file:
        generated_files.append(overall_file)
        print(f"   SUCCESS: Created {overall_file.name}")
    
    # 2-4. Per-model heatmaps
    if 'model_name' in df.columns:
        models = sorted(df['model_name'].unique())
        print(f"\n[2/4] Generating Per-Model Heatmaps ({len(models)} models)...")
        
        for idx, model in enumerate(models, 1):
            model_data = df[df['model_name'] == model]
            
            # 2. Per model (BG+EN combined)
            model_file = create_per_model_heatmap(model_data, model, output_dir)
            if model_file:
                generated_files.append(model_file)
                print(f"   [{idx}/{len(models)}] {model}: Combined")
            
            # 3. Per model BG only
            if 'language' in df.columns:
                bg_data = model_data[model_data['language'] == 'bg']
                if len(bg_data) > 0:
                    bg_file = create_per_model_language_heatmap(
                        bg_data, model, 'BG', output_dir
                    )
                    if bg_file:
                        generated_files.append(bg_file)
                        print(f"   [{idx}/{len(models)}] {model}: Bulgarian")
                
                # 4. Per model EN only
                en_data = model_data[model_data['language'] == 'en']
                if len(en_data) > 0:
                    en_file = create_per_model_language_heatmap(
                        en_data, model, 'EN', output_dir
                    )
                    if en_file:
                        generated_files.append(en_file)
                        print(f"   [{idx}/{len(models)}] {model}: English")
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} heatmaps")
    print("="*80 + "\n")
    
    return generated_files


def create_overall_heatmap(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """
    Create overall heatmap for all models and languages combined
    
    Category (Y-axis) × Temperature (X-axis)
    Shows success rate %
    """
    try:
        # Create pivot table
        pivot = df.groupby(['category', 'temperature'])['success'].mean() * 100
        pivot = pivot.unstack(fill_value=0)
        
        # Ensure all temperatures are present
        for temp in TEMPERATURE_RANGE:
            if temp not in pivot.columns:
                pivot[temp] = 0
        
        # Sort columns by temperature
        pivot = pivot[sorted(pivot.columns)]
        
        # Sort categories by average success rate
        pivot['avg'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('avg', ascending=False)
        pivot = pivot.drop('avg', axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.5)))
        
        # Create heatmap with GREEN to RED colormap
        sns.heatmap(pivot, 
                    annot=True, 
                    fmt='.0f',
                    cmap='RdYlGn_r',  # Reversed: Green (low) -> Yellow -> Red (high)
                    cbar_kws={'label': 'Attack Success Rate (%)'},
                    ax=ax,
                    linewidths=1.0,
                    linecolor='white',
                    vmin=0,
                    vmax=100,
                    annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        
        # Styling
        ax.set_title('Attack Category Success Rates Across Temperature Range\n(All Models, All Languages)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Temperature', fontsize=13, fontweight='bold')
        ax.set_ylabel('Attack Category', fontsize=13, fontweight='bold')
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save
        file_path = output_dir / "heatmap_overall.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR: Error creating overall heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_per_model_heatmap(df: pd.DataFrame, model_name: str, 
                             output_dir: Path) -> Optional[Path]:
    """
    Create heatmap for a specific model (BG+EN combined)
    
    Category (Y-axis) × Temperature (X-axis)
    Shows success rate %
    """
    try:
        if len(df) == 0:
            return None
        
        # Create pivot table
        pivot = df.groupby(['category', 'temperature'])['success'].mean() * 100
        pivot = pivot.unstack(fill_value=0)
        
        # Ensure all temperatures are present
        for temp in TEMPERATURE_RANGE:
            if temp not in pivot.columns:
                pivot[temp] = 0
        
        # Sort columns by temperature
        pivot = pivot[sorted(pivot.columns)]
        
        # Sort categories by average success rate
        pivot['avg'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('avg', ascending=False)
        pivot = pivot.drop('avg', axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.5)))
        
        # Create heatmap with GREEN to RED colormap
        sns.heatmap(pivot, 
                    annot=True, 
                    fmt='.0f',
                    cmap='RdYlGn_r',  # Green -> Yellow -> Red
                    cbar_kws={'label': 'Attack Success Rate (%)'},
                    ax=ax,
                    linewidths=1.0,
                    linecolor='white',
                    vmin=0,
                    vmax=100,
                    annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        
        # Styling
        model_short = model_name[:50] + '...' if len(model_name) > 50 else model_name
        ax.set_title(f'Attack Category Success Rates Across Temperature Range\n{model_short} (Bulgarian + English)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Temperature', fontsize=13, fontweight='bold')
        ax.set_ylabel('Attack Category', fontsize=13, fontweight='bold')
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save with safe filename
        safe_name = make_safe_filename(model_name)
        file_path = output_dir / f"heatmap_model_{safe_name}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR: Error creating model heatmap for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_per_model_language_heatmap(df: pd.DataFrame, model_name: str, 
                                      language: str, output_dir: Path) -> Optional[Path]:
    """
    Create heatmap for a specific model and language
    
    Category (Y-axis) × Temperature (X-axis)
    Shows success rate %
    
    Args:
        df: DataFrame filtered for specific model and language
        model_name: Name of the model
        language: 'BG' or 'EN'
        output_dir: Output directory
    """
    try:
        if len(df) == 0:
            return None
        
        # Create pivot table
        pivot = df.groupby(['category', 'temperature'])['success'].mean() * 100
        pivot = pivot.unstack(fill_value=0)
        
        # Ensure all temperatures are present
        for temp in TEMPERATURE_RANGE:
            if temp not in pivot.columns:
                pivot[temp] = 0
        
        # Sort columns by temperature
        pivot = pivot[sorted(pivot.columns)]
        
        # Sort categories by average success rate
        pivot['avg'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('avg', ascending=False)
        pivot = pivot.drop('avg', axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.5)))
        
        # Language-specific color adjustments
        if language == 'BG':
            title_color = '#2E86AB'  # Blue for Bulgarian
        else:
            title_color = '#E74C3C'  # Red for English
        
        # Create heatmap with GREEN to RED colormap
        sns.heatmap(pivot, 
                    annot=True, 
                    fmt='.0f',
                    cmap='RdYlGn_r',  # Green -> Yellow -> Red
                    cbar_kws={'label': 'Attack Success Rate (%)'},
                    ax=ax,
                    linewidths=1.0,
                    linecolor='white',
                    vmin=0,
                    vmax=100,
                    annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        
        # Styling
        model_short = model_name[:50] + '...' if len(model_name) > 50 else model_name
        language_full = 'Bulgarian' if language == 'BG' else 'English'
        
        ax.set_title(f'Attack Category Success Rates Across Temperature Range\n{model_short} ({language_full} Only)',
                     fontsize=16, fontweight='bold', pad=20, color=title_color)
        ax.set_xlabel('Temperature', fontsize=13, fontweight='bold')
        ax.set_ylabel('Attack Category', fontsize=13, fontweight='bold')
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save with safe filename
        safe_name = make_safe_filename(model_name)
        file_path = output_dir / f"heatmap_model_{safe_name}_{language}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR: Error creating {language} heatmap for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def make_safe_filename(name: str) -> str:
    """
    Convert model name to safe filename
    
    Example: 
        'Llama-3.1-8B/instruct' -> 'Llama_3_1_8B_instruct'
    """
    # Replace problematic characters
    safe = name.replace('/', '_').replace('\\', '_').replace(':', '_')
    safe = safe.replace('.', '_').replace(' ', '_').replace('-', '_')
    
    # Keep only alphanumeric and underscore
    safe = ''.join(c for c in safe if c.isalnum() or c == '_')
    
    # Limit length
    if len(safe) > 50:
        safe = safe[:50]
    
    return safe


# Standalone execution example
if __name__ == "__main__":
    print("Category-Temperature Heatmap Generator")
    print("="*50)
    print("\nUsage:")
    print("  from visual_engine.category_temperature_heatmaps import create_all_category_temperature_heatmaps")
    print("  files = create_all_category_temperature_heatmaps(df, output_dir)")
    print("\nRequired DataFrame columns:")
    print("  - category: Attack category (MITRE)")
    print("  - temperature: Temperature value (0.1-1.0)")
    print("  - success: Boolean or 0/1 (attack success)")
    print("  - model_name: Model identifier (optional, for per-model)")
    print("  - language: 'bg' or 'en' (optional, for per-language)")
    print("\nOutputs:")
    print("  1. heatmap_overall.png")
    print("  2. heatmap_model_{model}.png (per model)")
    print("  3. heatmap_model_{model}_BG.png (per model, Bulgarian)")
    print("  4. heatmap_model_{model}_EN.png (per model, English)")
