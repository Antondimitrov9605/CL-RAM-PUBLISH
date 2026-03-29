#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temperature Effect on Model Vulnerability
==========================================
Creates line chart showing all models + overall average
with risk zones (High/Medium/Low)
Shows ONLY the temperatures that were actually tested.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

# Model colors (distinct colors for each model)
MODEL_COLORS = {
    0: '#4C72B0',  # Blue
    1: '#DD8452',  # Orange
    2: '#55A868',  # Green
    3: '#C44E52',  # Red
    4: '#8172B3',  # Purple
    5: '#937860',  # Brown
    6: '#DA8BC3',  # Pink
    7: '#8C8C8C',  # Gray
    8: '#CCB974',  # Olive
    9: '#64B5CD',  # Cyan
}


def get_tested_temperatures(df: pd.DataFrame) -> List[float]:
    """
    Auto-detect which temperatures were actually tested in the data
    
    Returns:
        Sorted list of unique temperatures found in data
    """
    temps = sorted(df['temperature'].unique())
    return [float(t) for t in temps]


def create_temperature_vulnerability_chart(df: pd.DataFrame, language: str,
                                           output_dir: Path, model_filter: str = None) -> Optional[Path]:
    """
    Create temperature effect chart showing all models + overall average
    Uses ONLY the temperatures that were actually tested in the data.
    
    Args:
        df: DataFrame with model_name, temperature, language, success columns
        language: 'bg', 'en', or 'all'
        output_dir: Output directory
        model_filter: Optional - if specified, create chart for this model only
        
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
        
        # Filter by model if specified
        if model_filter:
            data = data[data['model_name'] == model_filter]
            if len(data) == 0:
                print(f"WARNING: No data for model={model_filter}, language={language}")
                return None
        
        # AUTO-DETECT tested temperatures from data
        tested_temps = get_tested_temperatures(data)
        print(f"   Detected {len(tested_temps)} temperatures: {tested_temps}")
        
        # Get models
        models = sorted(data['model_name'].unique())
        
        # Calculate success rates for each model across temperatures
        model_data = {}
        for model in models:
            model_df = data[data['model_name'] == model]
            rates = []
            
            for temp in tested_temps:  # Use auto-detected temps
                temp_data = model_df[model_df['temperature'] == temp]
                if len(temp_data) > 0:
                    rate = temp_data['success'].mean() * 100
                else:
                    rate = None  # Mark as no data
                rates.append(rate)
            
            # Calculate overall vulnerability
            overall = model_df['success'].mean() * 100
            model_data[model] = {
                'rates': rates,
                'overall': overall
            }
        
        # Calculate overall average across all models
        overall_avg = []
        for temp in tested_temps:  # Use auto-detected temps
            temp_data = data[data['temperature'] == temp]
            if len(temp_data) > 0:
                rate = temp_data['success'].mean() * 100
            else:
                rate = None
            overall_avg.append(rate)
        
        # Sort models by overall vulnerability (descending)
        sorted_models = sorted(model_data.items(), 
                              key=lambda x: x[1]['overall'], 
                              reverse=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Add risk zone backgrounds
        ax.axhspan(67, 100, facecolor='#FFE6E6', alpha=0.3, zorder=0)  # High risk (light red)
        ax.axhspan(33, 67, facecolor='#FFF9E6', alpha=0.3, zorder=0)   # Medium risk (light yellow)
        ax.axhspan(0, 33, facecolor='#E6F9E6', alpha=0.3, zorder=0)    # Low risk (light green)
        
        # Add risk zone labels on the right
        ax.text(1.02, 83.5, 'High Risk', transform=ax.get_yaxis_transform(),
                fontsize=11, fontweight='bold', color='#C44E52', va='center')
        ax.text(1.02, 50, 'Medium Risk', transform=ax.get_yaxis_transform(),
                fontsize=11, fontweight='bold', color='#DD8452', va='center')
        ax.text(1.02, 16.5, 'Low Risk', transform=ax.get_yaxis_transform(),
                fontsize=11, fontweight='bold', color='#55A868', va='center')
        
        # Plot each model
        for idx, (model, info) in enumerate(sorted_models):
            color = MODEL_COLORS.get(idx % 10, '#888888')  # Cycle through colors
            # Filter out None values for plotting
            valid_temps = [t for t, r in zip(tested_temps, info['rates']) if r is not None]
            valid_rates = [r for r in info['rates'] if r is not None]
            
            if valid_temps and valid_rates:
                short_model = model.split('.')[0][:20]  # Shorten name
                ax.plot(valid_temps, valid_rates, 'o-',
                       linewidth=2.5, markersize=8,
                       color=color,
                       label=f"{short_model} ({info['overall']:.1f}%)",
                       zorder=3)
        
        # Plot overall average (dashed line) - only if multiple models
        if len(models) > 1:
            valid_temps = [t for t, r in zip(tested_temps, overall_avg) if r is not None]
            valid_rates = [r for r in overall_avg if r is not None]
            if valid_temps and valid_rates:
                overall_avg_value = np.mean(valid_rates)
                ax.plot(valid_temps, valid_rates, 'k--',
                       linewidth=2.5, alpha=0.7,
                       label=f"Overall Average ({overall_avg_value:.1f}%)",
                       zorder=2)
        
        # Title - include model name if filtering
        if model_filter:
            short_model = model_filter.split('.')[0][:30]
            title = f'Temperature Effect on Vulnerability\nModel: {short_model} | {lang_title}'
        else:
            title = f'Temperature Effect on Model Vulnerability\n{lang_title} - Models Sorted by Overall Vulnerability'
        
        # Styling
        ax.set_title(title, fontsize=17, fontweight='bold', pad=20)
        ax.set_xlabel('Temperature', fontsize=14, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
        ax.set_ylim(0, 105)
        
        # Dynamic X-axis based on tested temperatures
        ax.set_xlim(min(tested_temps) - 0.05, max(tested_temps) + 0.05)
        ax.set_xticks(tested_temps)
        ax.set_xticklabels([f'{t:.1f}' for t in tested_temps])
        
        # Legend outside plot area
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left',
                 fontsize=10, framealpha=0.95)
        
        plt.tight_layout()
        
        # Save - include model in filename if filtering
        if model_filter:
            safe_model = model_filter.replace('.', '_').replace(' ', '_')[:25]
            if language == 'bg':
                filename = f"temp_effect_{safe_model}_BG.png"
            elif language == 'en':
                filename = f"temp_effect_{safe_model}_EN.png"
            else:
                filename = f"temp_effect_{safe_model}_OVERALL.png"
        else:
            if language == 'bg':
                filename = "temp_effect_BG.png"
            elif language == 'en':
                filename = "temp_effect_EN.png"
            else:
                filename = "temp_effect_OVERALL.png"
        
        file_path = output_dir / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR creating {language} chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_all_temperature_effect_charts(df: pd.DataFrame,
                                         output_dir: Path) -> List[Path]:
    """
    Create ALL temperature effect charts:
    - 3 overall charts (BG, EN, Overall - all models combined)
    - Per-model charts (Overall, BG, EN for each model)
    
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
    print("GENERATING TEMPERATURE EFFECT CHARTS")
    print("="*80)
    
    # Validate required columns
    required_cols = ['model_name', 'temperature', 'language', 'success']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Missing required columns. Need: {required_cols}")
        return generated_files
    
    # Show detected temperatures
    tested_temps = get_tested_temperatures(df)
    print(f"\nDetected {len(tested_temps)} temperatures in data: {tested_temps}")
    
    # Get list of models
    models = sorted(df['model_name'].unique())
    total_charts = 3 + (len(models) * 3)  # 3 overall + 3 per model
    chart_num = 0
    
    # ===== OVERALL CHARTS (all models combined) =====
    print("\n--- OVERALL CHARTS (All Models Combined) ---")
    
    # 1. Bulgarian chart
    chart_num += 1
    print(f"\n[{chart_num}/{total_charts}] Overall Bulgarian chart...")
    bg_file = create_temperature_vulnerability_chart(df, 'bg', output_dir)
    if bg_file:
        generated_files.append(bg_file)
        print(f"   SUCCESS: {bg_file.name}")
    
    # 2. English chart
    chart_num += 1
    print(f"\n[{chart_num}/{total_charts}] Overall English chart...")
    en_file = create_temperature_vulnerability_chart(df, 'en', output_dir)
    if en_file:
        generated_files.append(en_file)
        print(f"   SUCCESS: {en_file.name}")
    
    # 3. Overall chart (BG+EN combined)
    chart_num += 1
    print(f"\n[{chart_num}/{total_charts}] Overall combined chart (BG+EN)...")
    overall_file = create_temperature_vulnerability_chart(df, 'all', output_dir)
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
        model_overall = create_temperature_vulnerability_chart(df, 'all', output_dir, model_filter=model)
        if model_overall:
            generated_files.append(model_overall)
            print(f"       SUCCESS: {model_overall.name}")
        
        # Model Bulgarian
        chart_num += 1
        print(f"    [{chart_num}/{total_charts}] Bulgarian...")
        model_bg = create_temperature_vulnerability_chart(df, 'bg', output_dir, model_filter=model)
        if model_bg:
            generated_files.append(model_bg)
            print(f"       SUCCESS: {model_bg.name}")
        
        # Model English
        chart_num += 1
        print(f"    [{chart_num}/{total_charts}] English...")
        model_en = create_temperature_vulnerability_chart(df, 'en', output_dir, model_filter=model)
        if model_en:
            generated_files.append(model_en)
            print(f"       SUCCESS: {model_en.name}")
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} temperature effect charts")
    print(f"  - 3 overall charts (all models)")
    print(f"  - {len(models)} models × 3 languages = {len(models)*3} per-model charts")
    print(f"  - Temperatures shown: {tested_temps}")
    print("="*80 + "\n")
    
    return generated_files


# Test/Demo
if __name__ == "__main__":
    import json
    
    print("="*80)
    print("TEMPERATURE EFFECT CHARTS - TEST")
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
    output_dir = Path("temp_effect_charts")
    
    # Generate charts
    files = create_all_temperature_effect_charts(df, output_dir)
    
    print(f"\nGenerated files:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    print(f"\nOutput directory: {output_dir.absolute()}")
