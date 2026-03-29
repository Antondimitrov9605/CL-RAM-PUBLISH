#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Visualization Generator
==============================
Orchestrates the generation of all visualizations in the visual_engine package.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import time

# Import all visualization modules
# ... existing imports ...
from .category_temperature_heatmaps import create_all_category_temperature_heatmaps
from .category_language_heatmaps import create_all_category_language_heatmaps
from .model_temperature_heatmaps import create_all_model_temperature_heatmaps
from .temperature_effect_charts import create_all_temperature_effect_charts
from .temperature_sensitivity_charts import create_all_temperature_sensitivity_charts
from .temperature_distribution_charts import create_all_temperature_distribution_charts
from .surface_3d_plots import create_all_3d_surface_plots
from .statistical_tables import create_all_statistical_tables
from .model_performance_progression import create_all_model_progression_charts
from .category_performance_linear import create_all_category_performance_charts
from .category_line_charts import main as generate_all_line_charts
# NEW IMPORTS
from .attack_distribution_charts import create_attack_distribution_donut, create_all_attack_distribution_charts
from .category_bar_charts import create_all_category_bar_charts
from .category_distribution_pies import create_all_category_distribution_pies
from .category_performance_enhanced import create_all_enhanced_category_charts
from .individual_distribution_charts import create_all_individual_distribution_charts
from .language_comparison_charts import create_all_language_charts
from .model_overview_charts import create_all_model_overview_charts
from .statistics_generator import generate_statistics_report
from .category_line_charts import create_category_chart, create_model_overall_chart

# SCIENTIFIC RESEARCH MODULES (PhD level)
from .phase_transition_analyzer import create_all_phase_transition_charts
from .crosslingual_transfer_chart import create_all_crosslingual_charts
from .response_entropy_chart import create_all_response_entropy_charts
from .scientific_discoveries_charts import create_all_scientific_discovery_charts
from .deep_research_charts import create_all_deep_research_charts

def generate_all_visualizations(data_path: str, output_base_dir: str = "visualizations_output"):
    """
    Generates all available visualizations from the provided data file.
    
    Args:
        data_path: Path to the JSON results file.
        output_base_dir: Base directory for outputs. A timestamped subdirectory will be created inside.
    """
    start_time = time.time()
    
    # 1. Setup Paths
    data_file = Path(data_path)
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(output_base_dir) / f"run_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"STARTING VISUALIZATION GENERATION")
    print(f"Data: {data_file}")
    print(f"Output: {output_root}")
    print(f"{'='*80}\n")
    
    # 2. Load Data
    print("Loading data...")
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} records.")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return

    # 3. Generate Visualizations
    
    # Helper to run a module safely
    def run_module(name, func, *args, **kwargs):
        print(f"\n--- Running {name} ---")
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            print(f"FAILED {name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # 3.1 Heatmaps
    run_module("Category-Temperature Heatmaps", create_all_category_temperature_heatmaps, df, output_root / "category_temp_heatmaps")
    run_module("Category-Language Heatmaps", create_all_category_language_heatmaps, df, output_root / "category_lang_heatmaps")
    run_module("Model-Temperature Heatmaps", create_all_model_temperature_heatmaps, df, output_root / "model_temp_heatmaps")
    
    # 3.2 Charts
    run_module("Attack Distribution Charts", create_all_attack_distribution_charts, df, output_root / "attack_distribution_charts")
    run_module("Temperature Effect Charts", create_all_temperature_effect_charts, df, output_root / "temp_effect_charts")
    run_module("Temperature Sensitivity Charts", create_all_temperature_sensitivity_charts, df, output_root / "temp_sensitivity_charts")
    run_module("Temperature Distribution Charts", create_all_temperature_distribution_charts, df, output_root / "temp_distribution_charts")
    run_module("Model Progression Charts", create_all_model_progression_charts, df, output_root / "model_progression_charts")
    run_module("Category Performance Linear", create_all_category_performance_charts, df, output_root / "category_performance_charts")
    
    # 3.3 3D Plots
    run_module("3D Surface Plots", create_all_3d_surface_plots, df, output_root / "3d_surface_plots")
    
    # 3.4 Tables
    run_module("Statistical Tables", create_all_statistical_tables, df, output_root / "statistical_tables")
    
    # 3.4b Scientific Research Modules (NEW)
    run_module("Phase Transition Analysis", create_all_phase_transition_charts, df, output_root)
    run_module("Cross-Lingual Transfer Analysis", create_all_crosslingual_charts, df, output_root)
    run_module("Response Entropy Analysis", create_all_response_entropy_charts, df, output_root)
    run_module("Scientific Discoveries", create_all_scientific_discovery_charts, df, output_root)
    run_module("Deep Research Findings", create_all_deep_research_charts, df, output_root)
    
    # 3.5 Line Charts (Custom implementation to use passed DF)
    print(f"\n--- Running Category Line Charts ---")
    try:
        from .category_line_charts import create_category_chart, create_model_overall_chart
        line_charts_dir = output_root / "line_charts"
        line_charts_dir.mkdir(parents=True, exist_ok=True)
        
        models = sorted(df['model_name'].unique())
        temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        count = 0
        for model in models:
            model_data = df[df['model_name'] == model]
            categories = sorted(model_data['category'].unique())
            
            # Per category
            for category in categories:
                cat_data = model_data[model_data['category'] == category]
                bg_data = cat_data[cat_data['language'] == 'bg']
                en_data = cat_data[cat_data['language'] == 'en']
                
                bg_rates = [bg_data[bg_data['temperature'] == t]['success'].mean() * 100 if len(bg_data[bg_data['temperature'] == t]) > 0 else 0 for t in temperatures]
                en_rates = [en_data[en_data['temperature'] == t]['success'].mean() * 100 if len(en_data[en_data['temperature'] == t]) > 0 else 0 for t in temperatures]
                
                create_category_chart(model, category, bg_rates, en_rates, temperatures, line_charts_dir)
                count += 1
            
            # Overall
            bg_overall = [model_data[(model_data['language'] == 'bg') & (model_data['temperature'] == t)]['success'].mean() * 100 for t in temperatures]
            en_overall = [model_data[(model_data['language'] == 'en') & (model_data['temperature'] == t)]['success'].mean() * 100 for t in temperatures]
            create_model_overall_chart(model, bg_overall, en_overall, temperatures, line_charts_dir)
            count += 1
            
        # Global
        bg_global = [df[(df['language'] == 'bg') & (df['temperature'] == t)]['success'].mean() * 100 for t in temperatures]
        en_global = [df[(df['language'] == 'en') & (df['temperature'] == t)]['success'].mean() * 100 for t in temperatures]
        create_model_overall_chart("ALL MODELS", bg_global, en_global, temperatures, line_charts_dir)
        count += 1
        print(f"Generated {count} line charts.")
        
    except Exception as e:
        print(f"FAILED Category Line Charts: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE in {elapsed:.1f} seconds")
    print(f"All visualizations saved to: {output_root}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    # Default test path
    default_data = "data/outputs/session_20251107_031023/results_20251107_031023.json"
    generate_all_visualizations(default_data)
