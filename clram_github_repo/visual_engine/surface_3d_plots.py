#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Temperature-Model Success Rate Surface
==========================================
Creates 3D surface plots showing Model × Temperature vulnerability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Optional

# Temperature range
TEMPERATURE_RANGE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def create_3d_surface_plot(df: pd.DataFrame, language: str,
                           output_dir: Path) -> Optional[Path]:
    """
    Create 3D surface plot for Temperature × Model vulnerability
    
    Args:
        df: DataFrame with model_name, temperature, language, success columns
        language: 'bg', 'en', or 'all'
        output_dir: Output directory
        
    Returns:
        Path to generated 3D plot file
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
        
        # Get models and calculate success rates
        models = sorted(data['model_name'].unique())
        
        # Build data matrix: rows=models, cols=temperatures
        Z_data = []
        model_labels = []
        
        for model in models:
            model_df = data[data['model_name'] == model]
            rates = []
            
            for temp in TEMPERATURE_RANGE:
                temp_data = model_df[model_df['temperature'] == temp]
                if len(temp_data) > 0:
                    rate = temp_data['success'].mean() * 100
                else:
                    rate = 0
                rates.append(rate)
            
            # Calculate overall for sorting
            overall = model_df['success'].mean() * 100
            Z_data.append({
                'model': model,
                'rates': rates,
                'overall': overall
            })
        
        # Sort by overall vulnerability (ascending for 3D view)
        Z_data.sort(key=lambda x: x['overall'])
        
        # Build matrices for 3D plot
        Z = np.array([item['rates'] for item in Z_data])
        model_labels = [f"{item['model']}\n({item['overall']:.1f}%)" 
                       for item in Z_data]
        
        # Create meshgrid
        X = np.arange(len(TEMPERATURE_RANGE))  # Temperature indices
        Y = np.arange(len(models))              # Model indices
        X, Y = np.meshgrid(X, Y)
        
        # Create 3D plot
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface with viridis-like colormap
        surf = ax.plot_surface(X, Y, Z, 
                               cmap='plasma',  # Dark purple to yellow/orange
                               alpha=0.9,
                               edgecolor='white',
                               linewidth=0.5,
                               antialiased=True,
                               vmin=0, vmax=100)
        
        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        
        # Labels and title
        ax.set_xlabel('Temperature', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylabel('Model (sorted by vulnerability: Low → High)', 
                     fontsize=13, fontweight='bold', labelpad=15)
        ax.set_zlabel('Success Rate (%)', fontsize=13, fontweight='bold', labelpad=10)
        
        title = f'3D Temperature-Model Success Rate Surface\n{lang_title}'
        if language != 'all':
            title += f'\n(Models sorted by vulnerability: Low → High)'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Set ticks
        ax.set_xticks(X[0])
        ax.set_xticklabels([f'{t:.1f}' for t in TEMPERATURE_RANGE], fontsize=9)
        
        ax.set_yticks(Y[:, 0])
        ax.set_yticklabels(model_labels, fontsize=9)
        
        # Set viewing angle (rotated to face viewer)
        ax.view_init(elev=25, azim=-45)  # Changed from 45 to -45
        
        # Set Z limits
        ax.set_zlim(0, 100)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        if language == 'bg':
            filename = "3d_surface_BG.png"
        elif language == 'en':
            filename = "3d_surface_EN.png"
        else:
            filename = "3d_surface_OVERALL.png"
        
        file_path = output_dir / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR creating 3D plot for {language}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_all_3d_surface_plots(df: pd.DataFrame,
                                output_dir: Path) -> List[Path]:
    """
    Create all 3 3D surface plots
    
    Args:
        df: DataFrame with columns: model_name, temperature, language, success
        output_dir: Directory to save plots
        
    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("\n" + "="*80)
    print("GENERATING 3D SURFACE PLOTS")
    print("="*80)
    
    # Validate required columns
    required_cols = ['model_name', 'temperature', 'language', 'success']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Missing required columns. Need: {required_cols}")
        return generated_files
    
    # 1. Bulgarian 3D
    print("\n[1/3] Bulgarian 3D surface...")
    bg_file = create_3d_surface_plot(df, 'bg', output_dir)
    if bg_file:
        generated_files.append(bg_file)
        print(f"   SUCCESS: {bg_file.name}")
    
    # 2. English 3D
    print("\n[2/3] English 3D surface...")
    en_file = create_3d_surface_plot(df, 'en', output_dir)
    if en_file:
        generated_files.append(en_file)
        print(f"   SUCCESS: {en_file.name}")
    
    # 3. Overall 3D (BG+EN)
    print("\n[3/3] Overall 3D surface (BG+EN)...")
    overall_file = create_3d_surface_plot(df, 'all', output_dir)
    if overall_file:
        generated_files.append(overall_file)
        print(f"   SUCCESS: {overall_file.name}")
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} 3D plots")
    print("="*80 + "\n")
    
    return generated_files


# Test/Demo
if __name__ == "__main__":
    import json
    
    print("="*80)
    print("3D SURFACE PLOTS - TEST")
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
    output_dir = Path("3d_surface_plots")
    
    # Generate 3D plots
    files = create_all_3d_surface_plots(df, output_dir)
    
    print(f"\nGenerated files:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    print(f"\nOutput directory: {output_dir.absolute()}")
