#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Category Performance Enhanced Analysis
======================================
Creates lollipop-style charts for category performance ranking.
Generates: 
1. Overall (all models, all languages)
2. Per-Model Overall (BG + EN combined)
3. Per-Model Bulgarian
4. Per-Model English
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

def create_enhanced_category_chart(df: pd.DataFrame, output_dir: Path, 
                                   model_name: Optional[str] = None,
                                   language: Optional[str] = None) -> Optional[Path]:
    """
    Create category performance linear analysis chart (lollipop style)
    
    Args:
        df: DataFrame with test results
        output_dir: Output directory path
        model_name: Specific model name or None for overall
        language: 'bg', 'en', or None for combined
    
    Returns:
        Path to generated file or None if failed
    """
    try:
        # Filter data
        data = df.copy()
        title_parts = ["Category Performance Linear Analysis"]
        filename_parts = ["cat_perf_enhanced"]
        
        if model_name:
            data = data[data['model_name'] == model_name]
            # Clean model name for display
            display_model = model_name.replace('.gguf', '').replace('.Q8_0', '').replace('-', ' ')
            title_parts.append(f"({display_model})")
            filename_parts.append(model_name.replace('.', '_').replace('-', '_'))
        else:
            title_parts.append("(All Models)")
            filename_parts.append("ALL_MODELS")
            
        if language:
            data = data[data['language'] == language]
            lang_label = "Bulgarian" if language == 'bg' else "English"
            title_parts.append(f"- {lang_label}")
            filename_parts.append(language.upper())
        else:
            if model_name:
                title_parts.append("- Combined Languages")
                filename_parts.append("COMBINED")
            else:
                filename_parts.append("OVERALL")
            
        # Calculate success rate per category
        categories = data['category'].unique()
        cat_stats = []
        
        for cat in categories:
            cat_data = data[data['category'] == cat]
            if len(cat_data) > 0:
                rate = cat_data['success'].mean() * 100
                count = len(cat_data)
                cat_stats.append({
                    'category': cat, 
                    'rate': rate,
                    'count': count
                })
        
        # Sort by rate descending (highest first)
        cat_stats = sorted(cat_stats, key=lambda x: x['rate'], reverse=True)
        
        if not cat_stats:
            print(f"WARNING: No data for {model_name} {language}")
            return None
            
        # Prepare plotting data
        categories = [x['category'].replace('_', ' ').title() for x in cat_stats]
        rates = [x['rate'] for x in cat_stats]
        counts = [x['count'] for x in cat_stats]
        x_pos = np.arange(len(categories))
        
        # Create figure with better proportions
        fig, ax = plt.subplots(figsize=(18, 10))
        
        # Color gradient: Blue gradient from light to dark based on position
        # Higher values = darker blue
        colors = plt.cm.Blues(np.linspace(0.5, 0.95, len(rates)))
        
        # 1. Draw vertical lines (stems) with gradient
        for i, (x, y, color) in enumerate(zip(x_pos, rates, colors)):
            ax.vlines(x=x, ymin=0, ymax=y, color=color, alpha=0.7, linewidth=5)
        
        # 2. Draw points (heads) with larger size
        ax.scatter(x_pos, rates, s=200, color=colors, zorder=3, edgecolors='white', linewidths=2)
        
        # 3. Draw connecting line (dashed gray) - trend line
        ax.plot(x_pos, rates, color='gray', linestyle='--', alpha=0.5, linewidth=2, zorder=1)
        
        # 4. Add value labels above points
        for i, (x, y) in enumerate(zip(x_pos, rates)):
            ax.text(x, y + 2, f'{y:.1f}%', 
                   ha='center', va='bottom', 
                   fontweight='bold', fontsize=12,
                   color='black')
        
        # 5. Add sample count below x-axis labels
        for i, (x, count) in enumerate(zip(x_pos, counts)):
            ax.text(x, -3, f'n={count}', 
                   ha='center', va='top', 
                   fontsize=9, style='italic',
                   color='gray')
        
        # Styling
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=11, fontweight='500')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=13, fontweight='bold')
        ax.set_title(" ".join(title_parts), fontsize=17, fontweight='bold', pad=20)
        
        # Y-axis limit with padding
        max_rate = max(rates) if rates else 100
        ax.set_ylim(-5, max_rate + 10)
        
        # Grid - only horizontal
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Add subtle background color
        ax.set_facecolor('#f9f9f9')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        # Save with high quality
        file_path = output_dir / f"{'_'.join(filename_parts)}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] creating chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_all_enhanced_category_charts(df: pd.DataFrame, output_dir: Path):
    """
    Generate all enhanced category performance charts:
    1. Overall (all models, all languages) - 1 chart
    2. Per-Model Overall (BG + EN) - N charts (one per model)
    3. Per-Model Bulgarian - N charts
    4. Per-Model English - N charts
    
    Total: 1 + 3*N charts (where N = number of models)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    print("\n" + "="*80)
    print("ENHANCED CATEGORY PERFORMANCE LINEAR ANALYSIS")
    print("="*80)
    
    # 1. Overall chart (All Models, All Languages)
    print("\n[1] Overall Chart (All Models, All Languages)...")
    f = create_enhanced_category_chart(df, output_dir)
    if f: 
        generated_files.append(f)
    
    # 2. Per Model Charts
    models = sorted(df['model_name'].unique())
    print(f"\n[2] Processing {len(models)} models...")
    
    for idx, model in enumerate(models, 1):
        print(f"\n  Model {idx}/{len(models)}: {model}")
        
        # 2.1 Model Overall (BG + EN combined)
        print(f"    |-- Combined (BG + EN)...")
        f = create_enhanced_category_chart(df, output_dir, model_name=model, language=None)
        if f: generated_files.append(f)
        
        # 2.2 Model Bulgarian
        print(f"    |-- Bulgarian only...")
        f = create_enhanced_category_chart(df, output_dir, model_name=model, language='bg')
        if f: generated_files.append(f)
        
        # 2.3 Model English
        print(f"    `-- English only...")
        f = create_enhanced_category_chart(df, output_dir, model_name=model, language='en')
        if f: generated_files.append(f)
            
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} charts")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")
    
    return generated_files

if __name__ == "__main__":
    import json
    # Test code
    data_file = Path("data/detailed_academic_logs/results_20251107_031023.json")
    if data_file.exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        create_all_enhanced_category_charts(df, Path("category_performance_enhanced"))
    else:
        print(f"Data file not found: {data_file}")
