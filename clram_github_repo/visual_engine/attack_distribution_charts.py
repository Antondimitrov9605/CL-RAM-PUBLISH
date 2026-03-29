#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack Success Distribution Chart
==================================
Creates donut charts showing attack success rate distributions:
1. Global
2. Per Model (by Language)
3. Per Temperature
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Union

def _create_donut_chart(df: pd.DataFrame, title: str, filename: str, output_dir: Path) -> Optional[Path]:
    """
    Helper to create a standard donut chart.
    """
    try:
        total_attacks = len(df)
        if total_attacks == 0:
            return None

        successful_attacks = df['success'].sum()
        failed_attacks = total_attacks - successful_attacks
        
        success_rate = (successful_attacks / total_attacks) * 100
        
        # Data for pie chart
        # Data for pie chart
        sizes = [successful_attacks, failed_attacks]
        labels = ['Successful', 'Failed']
        # Swap colors to be more intuitive: Red for Attack Success (Risk), Green for Failure (Safe)
        colors = ['#e74c3c', '#27ae60']  
        explode = (0.05, 0.05) if failed_attacks > 0 and successful_attacks > 0 else None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                # Ensure text is readable
                return '{p:.1f}%\n({v:d})'.format(p=pct,v=val) if pct > 0 else ''
            return my_autopct

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct=make_autopct(sizes),
            startangle=90,
            explode=explode,
            pctdistance=0.85, # Move percentages inside a bit more
            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
            textprops=dict(color='black', fontsize=12, fontweight='bold')
        )
        
        # Center text
        center_text = f'{success_rate:.1f}%\nASR'
        ax.text(0, 0, center_text, ha='center', va='center', fontsize=14, fontweight='bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('equal')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        file_path = output_dir / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] Chart failed for {filename}: {e}")
        return None

def create_attack_distribution_donut(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Create global donut chart."""
    return _create_donut_chart(
        df, 
        "Global Attack Success Rate", 
        "attack_dist_global.png", 
        output_dir
    )

def create_all_attack_distribution_charts(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """
    Generate hierarchy of charts:
    1. Global
    2. Per Model -> By Language (BG, EN)
    3. Per Model -> By Temperature
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("="*60)
    print("GENERATING ATTACK DISTRIBUTION CHARTS")
    print("="*60)

    # 1. Global
    f = create_attack_distribution_donut(df, output_dir)
    if f: generated_files.append(f)

    # Get models
    models = sorted(df['model_name'].unique())
    languages = sorted(df['language'].unique())
    temperatures = sorted(df['temperature'].unique())

    for model in models:
        model_data = df[df['model_name'] == model]
        if model_data.empty: continue
        
        safe_model = model.replace('.gguf','').replace('.','_').replace('-','_')
        display_model = model.split('.')[0]

        # 2. Per Model & Language (BG / EN)
        for lang in languages:
            lang_data = model_data[model_data['language'] == lang]
            if not lang_data.empty:
                f = _create_donut_chart(
                    lang_data,
                    f"ASR: {display_model} ({lang.upper()})",
                    f"attack_dist_{safe_model}_{lang}.png",
                    output_dir
                )
                if f: generated_files.append(f)
        
        # 3. Per Model & Temperature
        for temp in temperatures:
            temp_data = model_data[model_data['temperature'] == temp]
            if not temp_data.empty:
                # We can also split by language here if requested, but user said "then by temperatures"
                # implying Model -> Temp. We can do Model -> Temp (Global) or Model -> Temp -> Lang.
                # Request: "one common for model on bg and en then by temperatures"
                # Interpretation: 
                # Model Global (BG+EN) - User didn't ask for this specifically, asked for BG and EN separately.
                # "then by temperatures" -> likely Model+Temp (BG+EN combined) or Model+Temp+Lang is too many.
                # Let's do Model + Temp (BG+EN combined) to show temp effect on that model.
                
                f = _create_donut_chart(
                    temp_data,
                    f"ASR: {display_model} (T={temp})",
                    f"attack_dist_{safe_model}_t{str(temp).replace('.','')}.png",
                    output_dir
                )
                if f: generated_files.append(f)

    print(f"\nTotal Charts Generated: {len(generated_files)}")
    return generated_files

if __name__ == "__main__":
    # Test stub
    pass
