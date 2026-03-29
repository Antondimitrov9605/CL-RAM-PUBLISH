#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Category Distribution Pie Charts
=================================
Creates pie charts showing distribution of successful attacks across
14 MITRE ATT&CK categories (centered pie with top 3 legend box).
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List

# Color palette for 14 categories
CATEGORY_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896'
]


def create_category_distribution_pie(df: pd.DataFrame, 
                                     model: Optional[str] = None,
                                     language: Optional[str] = None,
                                     output_dir: Path = Path("output")) -> Optional[Path]:
    """
    Create pie chart showing successful attack distribution by category
    
    Args:
        df: DataFrame with columns: model_name, category, language, success
        model: Optional model name to filter by
        language: Optional language ('bg', 'en') to filter by
        output_dir: Output directory
        
    Returns:
        Path to generated file or None if failed
    """
    try:
        # Filter data
        data = df.copy()
        
        # Only successful attacks
        data = data[data['success'] == 1]
        
        title_parts = []
        filename_parts = ["category_dist"]
        
        if model:
            data = data[data['model_name'] == model]
            # Clean model name for display
            display_model = model.replace('.gguf', '').replace('.Q8_0', '').replace('q8_0', '').replace('_', ' ').replace('-', ' ')
            display_model = ' '.join(word.capitalize() for word in display_model.split())
            title_parts.append(display_model)
            safe_model = model.replace('.', '_').replace('-', '_')
            filename_parts.append(safe_model)
            
        if language:
            lang_map = {'bg': 'Bulgarian', 'en': 'English', 'all': 'All Languages'}
            data = data[data['language'] == language] if language != 'all' else data
            title_parts.append(f"({lang_map.get(language, language)})")
            filename_parts.append(language)
        
        if len(data) == 0:
            print(f"[WARNING] No successful attacks found for filters")
            return None
            
        # Count successful attacks per category
        category_counts = data['category'].value_counts().sort_values(ascending=False)
        
        total_attacks = len(data)
        total_experiments = len(df) if model is None else len(df[df['model_name'] == model])
        success_rate = (total_attacks / total_experiments) * 100
        
        # Create gradient colors from green (low) to yellow (medium) to red (high)
        import matplotlib.colors as mcolors
        colormap = plt.get_cmap('RdYlGn_r')  # Green → Yellow → Red (reversed)
        # Normalize counts for color mapping
        norm = mcolors.Normalize(vmin=category_counts.min(), vmax=category_counts.max())
        colors = [colormap(norm(val)) for val in category_counts.values]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Format category labels
        labels = [cat.replace('_', ' ').title() for cat in category_counts.index]
        
        # Create pie chart with very thin separators and connecting lines to labels
        wedges, texts, autotexts = ax.pie(
            category_counts.values,
            labels=labels,
            autopct='%1.1f%%',  # Show ALL percentages
            startangle=90,
            colors=colors,
            textprops=dict(fontsize=9),
            pctdistance=0.85,
            labeldistance=1.15,  # Position labels further out
            wedgeprops=dict(edgecolor='black', linewidth=0.3),  # Very thin black borders
            explode=[0.02] * len(category_counts)  # Slight separation for clarity
        )
        
        # Make percentage text BLACK with white outline for visibility on all colors
        import matplotlib.patheffects as path_effects
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
            # Add white outline for better readability on dark backgrounds
            autotext.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground='white'),
                path_effects.Normal()
            ])
        
        # Equal aspect ratio for circle
        ax.axis('equal')
        
        # Create legend box with ALL categories
        table_lines = ["Categories:"]
        for i, (cat, count) in enumerate(category_counts.items(), 1):
            cat_display = cat.replace('_', ' ').title()
            pct = (count / total_attacks) * 100
            table_lines.append(f"{i}. {cat_display}: {count} ({pct:.1f}%)")
        
        # Add legend box with all categories
        legend_text = '\n'.join(table_lines)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.45, 0.5, legend_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='center', bbox=props, family='monospace')
        
        # Title
        title = "Distribution of Successful Attacks by Category\n"
        if title_parts:
            title += ' - '.join(title_parts) + '\n'
        title += f"Total Successful Attacks: {total_attacks:,} out of {total_experiments:,} experiments ({success_rate:.1f}% overall success rate)"
        
        plt.title(title, fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save
        filename = '_'.join(filename_parts) + '.png'
        file_path = output_dir / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] creating category distribution pie: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_all_category_distribution_pies(df: pd.DataFrame, 
                                          output_dir: Path) -> List[Path]:
    """
    Generate all 10 category distribution pie charts
    
    Creates:
    - 1 overall (all models, all languages)
    - 3 per-model (one for each model, all languages)
    - 6 per-model-per-language (3 models × 2 languages)
    
    Args:
        df: DataFrame with columns: model_name, category, language, success
        output_dir: Output directory
        
    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("\n" + "="*80)
    print("CATEGORY DISTRIBUTION PIE CHARTS GENERATION")
    print("="*80)
    
    # 1. Overall (all models, all languages)
    print("\n[1/10] Overall Category Distribution...")
    f = create_category_distribution_pie(df, output_dir=output_dir)
    if f:
        generated_files.append(f)
    
    # Get unique models
    models = sorted(df['model_name'].unique())
    languages = ['bg', 'en']
    
    # 2-4. Per-model (all languages)
    for i, model in enumerate(models, 2):
        print(f"\n[{i}/10] Category Distribution for {model}...")
        f = create_category_distribution_pie(df, model=model, output_dir=output_dir)
        if f:
            generated_files.append(f)
    
    # 5-10. Per-model-per-language
    counter = 5
    for model in models:
        for lang in languages:
            print(f"\n[{counter}/10] Category Distribution for {model} ({lang.upper()})...")
            f = create_category_distribution_pie(df, model=model, language=lang, output_dir=output_dir)
            if f:
                generated_files.append(f)
            counter += 1
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} pie charts")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")
    
    return generated_files


if __name__ == "__main__":
    # Test with real data
    data_file = Path("data/outputs/session_20251107_031023/complete_results_20251107_031023.csv")
    if data_file.exists():
        df = pd.read_csv(data_file, encoding='utf-8')
        
        output_dir = Path("category_distribution_output")
        files = create_all_category_distribution_pies(df, output_dir)
        
        print(f"\nGenerated {len(files)} files:")
        for f in files:
            print(f"  - {f.name}")
    else:
        print(f"Data file not found: {data_file}")
