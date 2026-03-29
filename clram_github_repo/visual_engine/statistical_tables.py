#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Tables Generator
=============================
Creates comprehensive statistical tables as PNG images
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

# Temperature range
TEMPERATURE_RANGE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def create_model_summary_table(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Create overall model summary table"""
    try:
        models = sorted(df['model_name'].unique())
        
        table_data = []
        for model in models:
            model_df = df[df['model_name'] == model]
            
            overall = model_df['success'].mean() * 100
            bg = model_df[model_df['language'] == 'bg']['success'].mean() * 100
            en = model_df[model_df['language'] == 'en']['success'].mean() * 100
            delta = bg - en
            total = len(model_df)
            
            table_data.append([
                model,
                f'{overall:.1f}%',
                f'{bg:.1f}%',
                f'{en:.1f}%',
                f'{delta:+.1f}%',
                total
            ])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, len(models) + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Headers
        headers = ['Model', 'Overall\nSuccess', 'Bulgarian\nSuccess', 'English\nSuccess', 
                  'BG-EN\nDelta', 'Total\nTests']
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.3, 0.15, 0.15, 0.15, 0.15, 0.1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#34495E')
            cell.set_text_props(weight='bold', color='white')
        
        # Style cells with colors
        for i in range(len(table_data)):
            # Model name
            table[(i+1, 0)].set_facecolor('#ECF0F1')
            table[(i+1, 0)].set_text_props(weight='bold')
            
            # Overall
            overall_val = float(table_data[i][1].strip('%'))
            if overall_val >= 70:
                color = '#E74C3C'  # Red
            elif overall_val >= 40:
                color = '#F39C12'  # Orange
            else:
                color = '#2ECC71'  # Green
            table[(i+1, 1)].set_facecolor(color)
            table[(i+1, 1)].set_text_props(color='white', weight='bold')
            
            # BG/EN - neutral
            table[(i+1, 2)].set_facecolor('#BDC3C7')
            table[(i+1, 3)].set_facecolor('#BDC3C7')
            
            # Delta
            delta_val = float(table_data[i][4].strip('%').strip('+'))
            if abs(delta_val) < 5:
                delta_color = '#95A5A6'  # Gray
            elif delta_val > 0:
                delta_color = '#3498DB'  # Blue (BG better)
            else:
                delta_color = '#E67E22'  # Orange (EN better)
            table[(i+1, 4)].set_facecolor(delta_color)
            table[(i+1, 4)].set_text_props(color='white', weight='bold')
            
            # Total
            table[(i+1, 5)].set_facecolor('#ECF0F1')
        
        plt.title('Model Summary Statistics', fontsize=16, fontweight='bold', pad=20)
        
        file_path = output_dir / 'table_model_summary.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR creating model summary table: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_category_vulnerability_table(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Create category vulnerability table"""
    try:
        categories = sorted(df['category'].unique())
        
        table_data = []
        for category in categories:
            cat_df = df[df['category'] == category]
            
            overall = cat_df['success'].mean() * 100
            bg = cat_df[cat_df['language'] == 'bg']['success'].mean() * 100
            en = cat_df[cat_df['language'] == 'en']['success'].mean() * 100
            delta = bg - en
            
            # Calculate per-model
            models = sorted(df['model_name'].unique())
            model_rates = []
            for model in models:
                model_cat_df = cat_df[cat_df['model_name'] == model]
                if len(model_cat_df) > 0:
                    rate = model_cat_df['success'].mean() * 100
                    model_rates.append(f'{rate:.0f}%')
                else:
                    model_rates.append('-')
            
            row = [
                category.replace('_', ' ').title(),
                f'{overall:.1f}%',
                f'{bg:.1f}%',
                f'{en:.1f}%',
                f'{delta:+.1f}%'
            ] + model_rates
            
            table_data.append(row)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, len(categories) + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Headers
        models = sorted(df['model_name'].unique())
        model_headers = [m.split('.')[0] if '.' in m else m[:10] for m in models]
        headers = ['Category', 'Overall', 'BG', 'EN', 'Δ'] + model_headers
        
        # Create table
        col_widths = [0.2, 0.1, 0.1, 0.1, 0.1] + [0.1] * len(models)
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=col_widths)
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(weight='bold', color='white', fontsize=8)
        
        # Style cells
        for i in range(len(table_data)):
            # Category name
            table[(i+1, 0)].set_facecolor('#ECF0F1')
            table[(i+1, 0)].set_text_props(ha='left')
            
            # Overall - color coded
            overall_val = float(table_data[i][1].strip('%'))
            if overall_val >= 60:
                color = '#E74C3C'
            elif overall_val >= 30:
                color = '#F39C12'
            else:
                color = '#2ECC71'
            table[(i+1, 1)].set_facecolor(color)
            table[(i+1, 1)].set_text_props(color='white', weight='bold')
            
            # BG/EN
            table[(i+1, 2)].set_facecolor('#D5DBDB')
            table[(i+1, 3)].set_facecolor('#D5DBDB')
            
            # Delta
            delta_val = float(table_data[i][4].strip('%').strip('+'))
            if abs(delta_val) < 5:
                delta_color = '#95A5A6'
            elif delta_val > 0:
                delta_color = '#3498DB'
            else:
                delta_color = '#E67E22'
            table[(i+1, 4)].set_facecolor(delta_color)
            table[(i+1, 4)].set_text_props(color='white', fontsize=8)
            
            # Per-model cells
            for j in range(len(models)):
                table[(i+1, 5+j)].set_facecolor('#F8F9F9')
                table[(i+1, 5+j)].set_text_props(fontsize=8)
        
        plt.title('Category Vulnerability Analysis', fontsize=16, fontweight='bold', pad=20)
        
        file_path = output_dir / 'table_category_vulnerability.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR creating category table: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_temperature_analysis_table(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Create temperature analysis table"""
    try:
        table_data = []
        
        for temp in TEMPERATURE_RANGE:
            temp_df = df[df['temperature'] == temp]
            
            overall = temp_df['success'].mean() * 100
            bg = temp_df[temp_df['language'] == 'bg']['success'].mean() * 100
            en = temp_df[temp_df['language'] == 'en']['success'].mean() * 100
            
            # Per-model
            models = sorted(df['model_name'].unique())
            model_rates = []
            for model in models:
                model_temp_df = temp_df[temp_df['model_name'] == model]
                if len(model_temp_df) > 0:
                    rate = model_temp_df['success'].mean() * 100
                    model_rates.append(f'{rate:.1f}%')
                else:
                    model_rates.append('-')
            
            row = [
                f'{temp:.1f}',
                f'{overall:.1f}%',
                f'{bg:.1f}%',
                f'{en:.1f}%'
            ] + model_rates
            
            table_data.append(row)
        
        # Add delta row (T=1.0 - T=0.1)
        delta_row = ['Δ (1.0-0.1)']
        for col_idx in range(1, 4 + len(sorted(df['model_name'].unique()))):
            val_high = float(table_data[-1][col_idx].strip('%'))
            val_low = float(table_data[0][col_idx].strip('%'))
            delta = val_high - val_low
            delta_row.append(f'{delta:+.1f}%')
        table_data.append(delta_row)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 13))
        ax.axis('tight')
        ax.axis('off')
        
        # Headers
        models = sorted(df['model_name'].unique())
        model_headers = [m.split('.')[0] if '.' in m else m[:12] for m in models]
        headers = ['Temp', 'Overall', 'BG', 'EN'] + model_headers
        
        # Create table
        col_widths = [0.12, 0.12, 0.12, 0.12] + [0.14] * len(models)
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=col_widths)
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)
        
        # Style header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#16A085')
            cell.set_text_props(weight='bold', color='white')
        
        # Style cells
        for i in range(len(table_data)):
            # Temperature column
            table[(i+1, 0)].set_facecolor('#ECF0F1')
            table[(i+1, 0)].set_text_props(weight='bold')
            
            # Last row (delta) - special styling
            if i == len(table_data) - 1:
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor('#34495E')
                    table[(i+1, j)].set_text_props(color='white', weight='bold')
            else:
                # Regular temperature rows
                for j in range(1, len(headers)):
                    table[(i+1, j)].set_facecolor('#F8F9F9')
        
        plt.title('Temperature Analysis Table', fontsize=16, fontweight='bold', pad=20)
        
        file_path = output_dir / 'table_temperature_analysis.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR creating temperature table: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_all_statistical_tables(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Create all statistical tables"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("\n" + "="*80)
    print("GENERATING STATISTICAL TABLES")
    print("="*80)
    
    # 1. Model Summary
    print("\n[1/3] Model Summary Table...")
    file1 = create_model_summary_table(df, output_dir)
    if file1:
        generated_files.append(file1)
        print(f"   SUCCESS: {file1.name}")
    
    # 2. Category Vulnerability
    print("\n[2/3] Category Vulnerability Table...")
    file2 = create_category_vulnerability_table(df, output_dir)
    if file2:
        generated_files.append(file2)
        print(f"   SUCCESS: {file2.name}")
    
    # 3. Temperature Analysis
    print("\n[3/3] Temperature Analysis Table...")
    file3 = create_temperature_analysis_table(df, output_dir)
    if file3:
        generated_files.append(file3)
        print(f"   SUCCESS: {file3.name}")
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} tables")
    print("="*80 + "\n")
    
    return generated_files


# Test/Demo
if __name__ == "__main__":
    import json
    
    print("="*80)
    print("STATISTICAL TABLES - TEST")
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
    
    # Create output directory
    output_dir = Path("statistical_tables")
    
    # Generate tables
    files = create_all_statistical_tables(df, output_dir)
    
    print(f"\nGenerated files:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    print(f"\nOutput directory: {output_dir.absolute()}")
