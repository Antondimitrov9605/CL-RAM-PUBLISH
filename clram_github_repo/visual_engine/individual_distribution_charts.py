#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Individual Distribution Charts
================================
Creates individual pie charts for different distribution views:
1. Overall Attack Success/Failure
2. Experiment Distribution by Language
3. Experiment Distribution by Model
4. Experiment Distribution by Category
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import numpy as np


def create_success_distribution_chart(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Overall Attack Success Distribution - Success vs Failure"""
    try:
        total = len(df)
        successful = df['success'].sum()
        failed = total - successful
        
        sizes = [successful, failed]
        labels = ['Successful Attacks', 'Failed Attacks']
        colors = ['#27ae60', '#e74c3c']  # Green for success, Red for failure
        explode = (0.1, 0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=explode,
            wedgeprops=dict(edgecolor='black', linewidth=0.3),
            textprops=dict(fontsize=12, fontweight='bold')
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
        
        ax.set_title('Overall Attack Success Distribution', fontsize=16, fontweight='bold', pad=20)
        ax.axis('equal')
        
        plt.tight_layout()
        
        file_path = output_dir / "distribution_success_failure.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] creating success distribution: {e}")
        return None


def create_language_distribution_chart(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Experiment Distribution by Language - BG vs EN"""
    try:
        lang_counts = df['language'].value_counts()
        
        sizes = lang_counts.values
        labels = [f"{lang.upper()}" for lang in lang_counts.index]
        colors = ['#3498db', '#2980b9']  # Blue shades
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(edgecolor='black', linewidth=0.3),
            textprops=dict(fontsize=12, fontweight='bold')
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
        
        ax.set_title('Experiment Distribution by Language', fontsize=16, fontweight='bold', pad=20)
        ax.axis('equal')
        
        plt.tight_layout()
        
        file_path = output_dir / "distribution_language.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] creating language distribution: {e}")
        return None


def create_model_distribution_chart(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Experiment Distribution by Model"""
    try:
        model_counts = df['model_name'].value_counts()
        
        sizes = model_counts.values
        labels = [model.replace('.gguf', '').replace('.Q8_0', '').replace('q8_0', '').replace('_', ' ').replace('-', ' ').title()
                  for model in model_counts.index]
        
        # Different colors for each model
        colors = ['#3498db', '#27ae60', '#f39c12']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors[:len(sizes)],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(edgecolor='black', linewidth=0.3),
            textprops=dict(fontsize=11, fontweight='bold')
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(13)
        
        ax.set_title('Experiment Distribution by Model', fontsize=16, fontweight='bold', pad=20)
        ax.axis('equal')
        
        plt.tight_layout()
        
        file_path = output_dir / "distribution_model.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] creating model distribution: {e}")
        return None


def create_category_experiments_distribution_chart(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Experiment Distribution by Category - shows how experiments are distributed across categories"""
    try:
        category_counts = df['category'].value_counts()
        
        # Create gradient colors from light to dark blue
        import matplotlib.colors as mcolors
        n_categories = len(category_counts)
        colormap = plt.get_cmap('Blues')
        colors = [colormap(0.4 + 0.6 * (i / n_categories)) for i in range(n_categories)]
        
        sizes = category_counts.values
        labels = [cat.replace('_', ' ').title() for cat in category_counts.index]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(edgecolor='black', linewidth=0.3),
            textprops=dict(fontsize=8),
            labeldistance=1.1
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        
        ax.set_title('Experiment Distribution by Category', fontsize=16, fontweight='bold', pad=20)
        ax.axis('equal')
        
        plt.tight_layout()
        
        file_path = output_dir / "distribution_category.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] creating category distribution: {e}")
        return None


def create_all_individual_distribution_charts(df: pd.DataFrame, output_dir: Path):
    """
    Generate all 4 individual distribution charts
    
    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("\n" + "="*80)
    print("INDIVIDUAL DISTRIBUTION CHARTS GENERATION")
    print("="*80)
    
    # 1. Success/Failure distribution
    print("\n[1/4] Success/Failure Distribution...")
    f = create_success_distribution_chart(df, output_dir)
    if f:
        generated_files.append(f)
    
    # 2. Language distribution
    print("\n[2/4] Language Distribution...")
    f = create_language_distribution_chart(df, output_dir)
    if f:
        generated_files.append(f)
    
    # 3. Model distribution
    print("\n[3/4] Model Distribution...")
    f = create_model_distribution_chart(df, output_dir)
    if f:
        generated_files.append(f)
    
    # 4. Category distribution
    print("\n[4/4] Category Distribution...")
    f = create_category_experiments_distribution_chart(df, output_dir)
    if f:
        generated_files.append(f)
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} individual charts")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")
    
    return generated_files


if __name__ == "__main__":
    # Test with real data
    data_file = Path("data/outputs/session_20251107_031023/complete_results_20251107_031023.csv")
    if data_file.exists():
        df = pd.read_csv(data_file, encoding='utf-8')
        create_all_individual_distribution_charts(df, Path("individual_distribution_output"))
    else:
        print(f"Data file not found: {data_file}")
