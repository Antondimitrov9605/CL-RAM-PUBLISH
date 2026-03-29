#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temperature Sensitivity by Attack Category
===========================================
Compares Min Rate (T=0.1) vs Max Rate (T=1.0) for each category
Shows temperature sensitivity (delta) for all 14 categories
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


def create_temperature_sensitivity_chart(df: pd.DataFrame, language: str,
                                         output_dir: Path) -> Optional[Path]:
    """
    Create temperature sensitivity comparison chart
    
    Args:
        df: DataFrame with model_name, temperature, language, success, category
        language: 'bg', 'en', or 'all'
        output_dir: Output directory
        
    Returns:
        Path to generated chart file
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
        
        # Get all categories
        categories = sorted(data['category'].unique())
        
        # Calculate min (T=0.1) and max (T=1.0) rates for each category
        sensitivity_data = []
        
        for category in categories:
            cat_data = data[data['category'] == category]
            
            # Min rate at T=0.1
            min_data = cat_data[cat_data['temperature'] == 0.1]
            min_rate = min_data['success'].mean() * 100 if len(min_data) > 0 else 0
            
            # Max rate at T=1.0
            max_data = cat_data[cat_data['temperature'] == 1.0]
            max_rate = max_data['success'].mean() * 100 if len(max_data) > 0 else 0
            
            # Delta (sensitivity)
            delta = max_rate - min_rate
            
            sensitivity_data.append({
                'category': category,
                'min_rate': min_rate,
                'max_rate': max_rate,
                'delta': delta
            })
        
        # Don't sort by delta - keep categories in fixed alphabetical order
        # This ensures categories are always at the same position across BG/EN/Overall
        # sensitivity_data already has categories in alphabetical order from sorted()
        
        # Prepare data for plotting
        categories_sorted = [item['category'].replace('_', ' ').title() 
                            for item in sensitivity_data]
        min_rates = [item['min_rate'] for item in sensitivity_data]
        max_rates = [item['max_rate'] for item in sensitivity_data]
        deltas = [item['delta'] for item in sensitivity_data]
        
        # Create figure (wider to fit horizontal labels)
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Set up bar positions
        x = np.arange(len(categories_sorted))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, min_rates, width, 
                      label='Min Rate (T=0.1)',
                      color='#6BAED6', edgecolor='white', linewidth=1.5)
        
        bars2 = ax.bar(x + width/2, max_rates, width,
                      label='Max Rate (T=1.0)',
                      color='#E57373', edgecolor='white', linewidth=1.5)
        
        # Add delta labels above bars
        for i, (delta, max_val) in enumerate(zip(deltas, max_rates)):
            # Position label above the taller bar
            y_pos = max(min_rates[i], max_rates[i]) + 3
            ax.text(i, y_pos, f'Δ={delta:.1f}%',
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='yellow' if abs(delta) > 15 else 'lightgray',
                            alpha=0.7, edgecolor='none'))
        
        # Styling
        ax.set_title(f'Temperature Sensitivity by Attack Category (T=0.1 vs T=1.0)\n{lang_title}',
                    fontsize=17, fontweight='bold', pad=20)
        ax.set_xlabel('Attack Category', fontsize=14, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories_sorted, rotation=0, ha='center', fontsize=9)
        ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height - 3,
                       f'{height:.1f}',
                       ha='center', va='top', fontsize=9,
                       color='white', fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        if language == 'bg':
            filename = "temp_sensitivity_BG.png"
        elif language == 'en':
            filename = "temp_sensitivity_EN.png"
        else:
            filename = "temp_sensitivity_OVERALL.png"
        
        file_path = output_dir / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR creating sensitivity chart for {language}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_model_temperature_sensitivity_chart(df: pd.DataFrame, model_name: str,
                                                language: str, output_dir: Path) -> Optional[Path]:
    """
    Create temperature sensitivity chart for a SPECIFIC MODEL
    
    Args:
        df: DataFrame with model_name, temperature, language, success, category
        model_name: Name of the model to filter
        language: 'bg', 'en', or 'all'
        output_dir: Output directory
        
    Returns:
        Path to generated chart file
    """
    try:
        # Filter by model
        data = df[df['model_name'] == model_name].copy()
        
        if len(data) == 0:
            print(f"WARNING: No data for model={model_name}")
            return None
        
        # Filter by language if specified
        if language in ['bg', 'en']:
            data = data[data['language'] == language]
            lang_title = 'Bulgarian' if language == 'bg' else 'English'
        else:
            lang_title = 'Overall (BG+EN)'
        
        if len(data) == 0:
            print(f"WARNING: No data for model={model_name}, language={language}")
            return None
        
        # Get all categories
        categories = sorted(data['category'].unique())
        
        # Calculate min (T=0.1) and max (T=1.0) rates for each category
        sensitivity_data = []
        
        for category in categories:
            cat_data = data[data['category'] == category]
            
            # Min rate at T=0.1
            min_data = cat_data[cat_data['temperature'] == 0.1]
            min_rate = min_data['success'].mean() * 100 if len(min_data) > 0 else 0
            
            # Max rate at T=1.0
            max_data = cat_data[cat_data['temperature'] == 1.0]
            max_rate = max_data['success'].mean() * 100 if len(max_data) > 0 else 0
            
            # Delta (sensitivity)
            delta = max_rate - min_rate
            
            sensitivity_data.append({
                'category': category,
                'min_rate': min_rate,
                'max_rate': max_rate,
                'delta': delta
            })
        
        # Prepare data for plotting
        categories_sorted = [item['category'].replace('_', ' ').title() 
                            for item in sensitivity_data]
        min_rates = [item['min_rate'] for item in sensitivity_data]
        max_rates = [item['max_rate'] for item in sensitivity_data]
        deltas = [item['delta'] for item in sensitivity_data]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Set up bar positions
        x = np.arange(len(categories_sorted))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, min_rates, width, 
                      label='Min Rate (T=0.1)',
                      color='#6BAED6', edgecolor='white', linewidth=1.5)
        
        bars2 = ax.bar(x + width/2, max_rates, width,
                      label='Max Rate (T=1.0)',
                      color='#E57373', edgecolor='white', linewidth=1.5)
        
        # Add delta labels above bars
        for i, (delta, max_val) in enumerate(zip(deltas, max_rates)):
            y_pos = max(min_rates[i], max_rates[i]) + 3
            ax.text(i, y_pos, f'Δ={delta:.1f}%',
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='yellow' if abs(delta) > 15 else 'lightgray',
                            alpha=0.7, edgecolor='none'))
        
        # Styling - include model name in title
        short_model = model_name.split('.')[0][:30]  # Shorten long model names
        ax.set_title(f'Temperature Sensitivity (T=0.1 vs T=1.0)\nModel: {short_model} | {lang_title}',
                    fontsize=17, fontweight='bold', pad=20)
        ax.set_xlabel('Attack Category', fontsize=14, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories_sorted, rotation=0, ha='center', fontsize=9)
        ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height - 3,
                       f'{height:.1f}',
                       ha='center', va='top', fontsize=9,
                       color='white', fontweight='bold')
        
        plt.tight_layout()
        
        # Save with model name in filename
        safe_model = model_name.replace('.', '_').replace(' ', '_')[:30]
        if language == 'bg':
            filename = f"temp_sensitivity_{safe_model}_BG.png"
        elif language == 'en':
            filename = f"temp_sensitivity_{safe_model}_EN.png"
        else:
            filename = f"temp_sensitivity_{safe_model}_OVERALL.png"
        
        file_path = output_dir / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR creating model sensitivity chart for {model_name}/{language}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_all_temperature_sensitivity_charts(df: pd.DataFrame,
                                              output_dir: Path) -> List[Path]:
    """
    Create ALL temperature sensitivity charts:
    - 3 overall charts (BG, EN, Overall)
    - Per-model charts (Overall, BG, EN for each model)
    
    Args:
        df: DataFrame with columns: model_name, temperature, language, success, category
        output_dir: Directory to save charts
        
    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("\n" + "="*80)
    print("GENERATING TEMPERATURE SENSITIVITY CHARTS")
    print("="*80)
    
    # Validate required columns
    required_cols = ['model_name', 'temperature', 'language', 'success', 'category']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Missing required columns. Need: {required_cols}")
        return generated_files
    
    # Get list of models
    models = sorted(df['model_name'].unique())
    total_charts = 3 + (len(models) * 3)  # 3 overall + 3 per model
    chart_num = 0
    
    # ===== OVERALL CHARTS (all models combined) =====
    print("\n--- OVERALL CHARTS (All Models) ---")
    
    # 1. Bulgarian chart
    chart_num += 1
    print(f"\n[{chart_num}/{total_charts}] Overall Bulgarian sensitivity chart...")
    bg_file = create_temperature_sensitivity_chart(df, 'bg', output_dir)
    if bg_file:
        generated_files.append(bg_file)
        print(f"   SUCCESS: {bg_file.name}")
    
    # 2. English chart
    chart_num += 1
    print(f"\n[{chart_num}/{total_charts}] Overall English sensitivity chart...")
    en_file = create_temperature_sensitivity_chart(df, 'en', output_dir)
    if en_file:
        generated_files.append(en_file)
        print(f"   SUCCESS: {en_file.name}")
    
    # 3. Overall chart (BG+EN)
    chart_num += 1
    print(f"\n[{chart_num}/{total_charts}] Overall combined sensitivity chart (BG+EN)...")
    overall_file = create_temperature_sensitivity_chart(df, 'all', output_dir)
    if overall_file:
        generated_files.append(overall_file)
        print(f"   SUCCESS: {overall_file.name}")
    
    # ===== PER-MODEL CHARTS =====
    print(f"\n--- PER-MODEL CHARTS ({len(models)} models) ---")
    
    for model in models:
        short_model = model.split('.')[0][:25]
        print(f"\n  Model: {short_model}")
        
        # Model Overall (BG+EN combined)
        chart_num += 1
        print(f"    [{chart_num}/{total_charts}] Overall...")
        model_overall = create_model_temperature_sensitivity_chart(df, model, 'all', output_dir)
        if model_overall:
            generated_files.append(model_overall)
            print(f"       SUCCESS: {model_overall.name}")
        
        # Model Bulgarian
        chart_num += 1
        print(f"    [{chart_num}/{total_charts}] Bulgarian...")
        model_bg = create_model_temperature_sensitivity_chart(df, model, 'bg', output_dir)
        if model_bg:
            generated_files.append(model_bg)
            print(f"       SUCCESS: {model_bg.name}")
        
        # Model English
        chart_num += 1
        print(f"    [{chart_num}/{total_charts}] English...")
        model_en = create_model_temperature_sensitivity_chart(df, model, 'en', output_dir)
        if model_en:
            generated_files.append(model_en)
            print(f"       SUCCESS: {model_en.name}")
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} sensitivity charts")
    print(f"  - 3 overall charts")
    print(f"  - {len(models)} models × 3 languages = {len(models)*3} per-model charts")
    print("="*80 + "\n")
    
    return generated_files


# Test/Demo
if __name__ == "__main__":
    import json
    
    print("="*80)
    print("TEMPERATURE SENSITIVITY CHARTS - TEST")
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
    print(f"Categories: {df['category'].nunique()}")
    
    # Create output directory
    output_dir = Path("temp_sensitivity_charts")
    
    # Generate charts
    files = create_all_temperature_sensitivity_charts(df, output_dir)
    
    print(f"\nGenerated files:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    print(f"\nOutput directory: {output_dir.absolute()}")
