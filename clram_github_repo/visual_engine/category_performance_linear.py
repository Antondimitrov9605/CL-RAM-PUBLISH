#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Category Performance Linear Analysis
====================================
Creates lollipop-style charts for category performance ranking.
Generates: Overall, Per-Model, and Per-Model-Language breakdowns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

def create_category_performance_chart(df: pd.DataFrame, output_dir: Path, 
                                      model_name: Optional[str] = None,
                                      language: Optional[str] = None) -> Optional[Path]:
    """
    Create category performance linear analysis chart (lollipop style)
    
    Args:
        df: DataFrame with test results
        output_dir: Output directory path
        model_name: Specific model name or None for overall
        language: 'bg', 'en', or None for combined
    """
    try:
        # Filter data
        data = df.copy()
        title_parts = ["Category Performance Linear Analysis"]
        filename_parts = ["cat_perf"]
        
        if model_name:
            data = data[data['model_name'] == model_name]
            title_parts.append(f"({model_name})")
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
            filename_parts.append("OVERALL")
            
        # Calculate success rate per category
        categories = data['category'].unique()
        cat_stats = []
        
        for cat in categories:
            cat_data = data[data['category'] == cat]
            if len(cat_data) > 0:
                rate = cat_data['success'].mean() * 100
                cat_stats.append({'category': cat, 'rate': rate})
        
        # Sort by rate descending (highest first)
        cat_stats = sorted(cat_stats, key=lambda x: x['rate'], reverse=True)
        
        if not cat_stats:
            print(f"No data for {model_name} {language}")
            return None
            
        # Prepare plotting data
        categories = [x['category'].replace('_', ' ').title() for x in cat_stats]
        rates = [x['rate'] for x in cat_stats]
        x_pos = np.arange(len(categories))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 1. Draw vertical lines (stems)
        # Color gradient based on value
        colors = plt.cm.Blues(np.linspace(0.4, 1, len(rates)))
        
        ax.vlines(x=x_pos, ymin=0, ymax=rates, color=colors, alpha=0.6, linewidth=4)
        
        # 2. Draw points (heads)
        ax.scatter(x_pos, rates, s=150, color=colors, zorder=3)
        
        # 3. Draw connecting line (dashed gray)
        ax.plot(x_pos, rates, color='gray', linestyle='--', alpha=0.4, zorder=1)
        
        # 4. Add value labels
        for i, (x, y) in enumerate(zip(x_pos, rates)):
            ax.text(x, y + 1.5, f'{y:.1f}%', 
                   ha='center', va='bottom', 
                   fontweight='bold', fontsize=11,
                   color='black')
        
        # Styling
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(" ".join(title_parts), fontsize=16, fontweight='bold', pad=20)
        
        # Y-axis limit with padding
        ax.set_ylim(0, max(rates) + 10)
        
        # Grid
        ax.grid(True, axis='y', alpha=0.3, linestyle='-')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save
        file_path = output_dir / f"{'_'.join(filename_parts)}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return file_path
        
    except Exception as e:
        print(f"ERROR creating chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_all_category_performance_charts(df: pd.DataFrame, output_dir: Path):
    """Generate all category performance charts"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    print("\n" + "="*80)
    print("CATEGORY PERFORMANCE LINEAR ANALYSIS")
    print("="*80)
    
    # 1. Overall (All Models, Combined Lang)
    print("\n[1] Overall chart (all models, all languages)...")
    f = create_category_performance_chart(df, output_dir)
    if f: generated_files.append(f)
    
    # 2. Per Model - ONLY BG and EN (NO overall per model)
    models = sorted(df['model_name'].unique())
    for model in models:
        print(f"\nProcessing {model}...")
        
        # BG for model
        print(f"  - Bulgarian chart...")
        f = create_category_performance_chart(df, output_dir, model_name=model, language='bg')
        if f: generated_files.append(f)
        
        # EN for model
        print(f"  - English chart...")
        f = create_category_performance_chart(df, output_dir, model_name=model, language='en')
        if f: generated_files.append(f)
            
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} charts")
    print("="*80 + "\n")
    
    return generated_files

if __name__ == "__main__":
    import json
    # Test code
    data_file = Path("data/outputs/session_20251107_031023/results_20251107_031023.json")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    create_all_category_performance_charts(df, Path("category_performance_charts"))
