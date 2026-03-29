#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Visualization Examples
================================
Demonstrates 6 new chart types for LLM vulnerability analysis:
1. Radar/Spider Chart - Multi-metric model comparison
2. ROC Curves - Model performance evaluation
3. Waterfall Chart - Impact analysis
4. Treemap - Hierarchical vulnerability distribution
5. Stacked Area Chart - Category evolution across temperatures
6. Sankey Diagram - Attack flow visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.patches as mpatches


def create_radar_chart(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Radar/Spider Chart - Compare models across multiple metrics"""
    try:
        # Calculate metrics per model
        models = sorted(df['model_name'].unique())
        
        # Metrics to compare
        metrics = []
        values_per_model = []
        
        for model in models:
            model_data = df[df['model_name'] == model]
            
            # Calculate various metrics
            overall_asr = (model_data['success'].sum() / len(model_data)) * 100
            
            # ASR by category (average across top 5 categories)
            cat_asr = model_data.groupby('category')['success'].mean().sort_values(ascending=False).head(5).mean() * 100
            
            # Temperature sensitivity (difference between T=0.1 and T=1.0)
            temps = model_data.groupby('temperature')['success'].mean()
            if len(temps) >= 2:
                temp_sens = abs(temps.iloc[-1] - temps.iloc[0]) * 100
            else:
                temp_sens = 0
            
            # Language difference
            lang_diff = abs(model_data[model_data['language']=='bg']['success'].mean() - 
                          model_data[model_data['language']=='en']['success'].mean()) * 100
            
            values = [overall_asr, cat_asr, temp_sens, lang_diff]
            values_per_model.append(values)
        
        metrics = ['Overall ASR', 'Top 5\nCategories', 'Temperature\nSensitivity', 'Language\nDifference']
        
        # Number of variables
        num_vars = len(metrics)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Colors for each model
        colors = ['#3498db', '#27ae60', '#f39c12']
        
        # Plot each model
        for i, (model, values) in enumerate(zip(models, values_per_model)):
            values += values[:1]  # Complete the circle
            
            model_name = model.replace('.gguf', '').replace('.Q8_0', '').replace('q8_0', '').replace('_', ' ').replace('-', ' ').title()
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        # Fix axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=10)
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        
        # Add title and legend
        ax.set_title('Multi-Metric Model Comparison\n(Radar Chart)', size=16, weight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # Grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        file_path = output_dir / "example_radar_chart.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] creating radar chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_roc_curve(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """ROC-style Curve - Model performance across thresholds"""
    try:
        models = sorted(df['model_name'].unique())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#3498db', '#27ae60', '#f39c12']
        
        # For each model, calculate TPR/FPR-like metrics across temperature thresholds
        for i, model in enumerate(models):
            model_data = df[df['model_name'] == model]
            
            # Sort by temperature and calculate cumulative success rate
            temps = sorted(model_data['temperature'].unique())
            tpr_values = []
            fpr_values = []
            
            for temp in temps:
                # "Positive" = attacks at this temperature or higher
                high_temp_data = model_data[model_data['temperature'] >= temp]
                low_temp_data = model_data[model_data['temperature'] < temp]
                
                if len(high_temp_data) > 0:
                    tpr = high_temp_data['success'].mean()  # True positive rate
                else:
                    tpr = 0
                
                if len(low_temp_data) > 0:
                    fpr = low_temp_data['success'].mean()  # False positive rate (inverse)
                else:
                    fpr = 0
                
                tpr_values.append(tpr)
                fpr_values.append(1 - fpr)  # Invert for ROC-like appearance
            
            # Add origin and end points
            fpr_values = [0] + fpr_values + [1]
            tpr_values = [0] + tpr_values + [1]
            
            # Calculate simple AUC (trapezoidal)
            auc_value = np.trapz(tpr_values, fpr_values)
            
            model_name = model.replace('.gguf', '').replace('.Q8_0', '').replace('q8_0', '').replace('_', ' ').replace('-', ' ').title()
            
            ax.plot(fpr_values, tpr_values, color=colors[i], lw=2, marker='o', markersize=4,
                   label=f'{model_name} (AUC ≈ {abs(auc_value):.2f})')
        
        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Baseline')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Inverse Low-Temp Success Rate', fontsize=12)
        ax.set_ylabel('High-Temp Success Rate', fontsize=12)
        ax.set_title('Performance Curve - Temperature Threshold Analysis', fontsize=16, weight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        file_path = output_dir / "example_roc_curve.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] creating ROC curve: {e}")
        import traceback
        traceback.print_exc()
        return None
        
        models = sorted(df['model_name'].unique())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#3498db', '#27ae60', '#f39c12']
        
        for i, model in enumerate(models):
            model_data = df[df['model_name'] == model]
            
            # Get true labels and predictions
            y_true = model_data['success'].values
            # Use a simple probability based on temperature (higher temp = higher prob)
            y_score = model_data['temperature'].values / 10.0  # Normalize to 0-1
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            model_name = model.replace('.gguf', '').replace('.Q8_0', '').replace('q8_0', '').replace('_', ' ').replace('-', ' ').title()
            
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                   label=f'{model_name} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Performance Comparison', fontsize=16, weight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        file_path = output_dir / "example_roc_curve.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] creating ROC curve: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_waterfall_chart(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Waterfall Chart - Impact of different factors on attack success"""
    try:
        # Calculate baseline and impacts
        baseline = df['success'].mean() * 100
        
        # Impact of temperature (high vs low)
        low_temp = df[df['temperature'] <= 0.3]['success'].mean() * 100
        high_temp = df[df['temperature'] >= 0.8]['success'].mean() * 100
        temp_impact = high_temp - low_temp
        
        # Impact of language
        bg_asr = df[df['language'] == 'bg']['success'].mean() * 100
        en_asr = df[df['language'] == 'en']['success'].mean() * 100
        lang_impact = bg_asr - en_asr
        
        # Impact of model (best vs worst)
        model_asrs = df.groupby('model_name')['success'].mean() * 100
        model_impact = model_asrs.max() - model_asrs.min()
        
        # Impact of category (top vs bottom)
        cat_asrs = df.groupby('category')['success'].mean() * 100
        cat_impact = cat_asrs.max() - cat_asrs.min()
        
        # Build waterfall
        categories = ['Baseline\nASR', 'Temperature\nEffect', 'Language\nEffect', 
                     'Model\nVariation', 'Category\nVariation', 'Final\nRange']
        values = [baseline, temp_impact, lang_impact, model_impact, cat_impact, 0]
        
        # Calculate cumulative
        cumulative = [baseline]
        for val in values[1:-1]:
            cumulative.append(cumulative[-1] + val)
        cumulative.append(cumulative[-1])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Colors
        colors = ['#3498db' if val >= 0 else '#e74c3c' for val in values]
        colors[0] = '#95a5a6'  # Baseline
        colors[-1] = '#2ecc71'  # Final
        
        # Plot bars
        for i, (cat, val, cum) in enumerate(zip(categories, values, cumulative)):
            if i == 0:
                ax.bar(i, val, color=colors[i], edgecolor='black', linewidth=1.5)
            elif i == len(categories) - 1:
                ax.bar(i, cum, color=colors[i], edgecolor='black', linewidth=1.5)
            else:
                bottom = cum - val if val >= 0 else cum
                ax.bar(i, abs(val), bottom=bottom, color=colors[i], 
                      edgecolor='black', linewidth=1.5)
                # Add connector line
                if i < len(categories) - 1:
                    ax.plot([i + 0.5, i + 1.5], [cum, cum], 'k--', linewidth=1)
        
        # Add value labels
        for i, (cat, val, cum) in enumerate(zip(categories, values, cumulative)):
            if i == 0 or i == len(categories) - 1:
                y_pos = val / 2 if i == 0 else cum / 2
                ax.text(i, y_pos, f'{val:.1f}%', ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')
            else:
                ax.text(i, cum, f'{val:+.1f}%', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
        ax.set_title('Waterfall Analysis - Factors Impacting Attack Success', 
                    fontsize=16, weight='bold', pad=20)
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(y=baseline, color='gray', linestyle='--', alpha= 0.5, label='Baseline')
        
        plt.tight_layout()
        
        file_path = output_dir / "example_waterfall_chart.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] creating waterfall chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_treemap(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Treemap - Hierarchical vulnerability distribution"""
    try:
        import squarify
        
        # Get top 12 categories by attack count
        cat_counts = df.groupby('category')['success'].sum().sort_values(ascending=False).head(12)
        
        labels = [cat.replace('_', ' ').title() + f'\n({count})' 
                 for cat, count in cat_counts.items()]
        sizes = cat_counts.values
        
        # Create color gradient
        import matplotlib.colors as mcolors
        colormap = plt.get_cmap('RdYlGn_r')
        norm = mcolors.Normalize(vmin=sizes.min(), vmax=sizes.max())
        colors = [colormap(norm(val)) for val in sizes]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8,
                     text_kwargs={'fontsize': 9, 'weight': 'bold'},
                     edgecolor='white', linewidth=2, ax=ax)
        
        ax.set_title('Treemap - Attack Distribution by Category\n(Size = Number of Successful Attacks)',
                    fontsize=16, weight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        file_path = output_dir / "example_treemap.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except ImportError:
        print("[WARNING] squarify not installed. Install with: pip install squarify")
        return None
    except Exception as e:
        print(f"[ERROR] creating treemap: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_stacked_area_chart(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Stacked Area Chart - Category evolution across temperatures"""
    try:
        # Get success rate by temperature and category for ALL categories
        # Previously we limited to top 6: .head(6) 
        # Now we take all unique categories present in the data
        all_cats = df['category'].unique()
        
        # Pivot data
        pivot = df.groupby(['temperature', 'category'])['success'].mean().unstack()
        pivot = pivot * 100  # Convert to percentage
        pivot = pivot.fillna(0)
        
        temperatures = sorted(df['temperature'].unique())
        pivot = pivot.reindex(temperatures)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Use a larger colormap since we have many categories (up to 14 for MITRE)
        import matplotlib.cm as cm
        # Generate colors dynamically based on number of categories
        num_cats = len(pivot.columns)
        colors = cm.tab20(np.linspace(0, 1, num_cats))
        
        # Create stacked area
        ax.stackplot(pivot.index, 
                    [pivot[col].values for col in pivot.columns],
                    labels=[col.replace('_', ' ').title() for col in pivot.columns],
                    colors=colors, alpha=0.8)
        
        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
        ax.set_title('Stacked Area Chart - Category Success Across Temperatures\n(All Categories)',
                    fontsize=16, weight='bold', pad=20)
        # Adjust legend to fit all categories (maybe outside if too many)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(temperatures[0], temperatures[-1])
        
        plt.tight_layout()
        
        file_path = output_dir / "example_stacked_area.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[+] Generated: {file_path.name}")
        return file_path
        
    except Exception as e:
        print(f"[ERROR] creating stacked area chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_sankey_diagram(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Sankey Diagram - Attack flow visualization"""
    try:
        import plotly.graph_objects as go
        
        # Sample data for Sankey (Temperature → Model → Result)
        # Get counts for flows
        flows = []
        
        # Group temperatures into 3 bins
        df['temp_bin'] = pd.cut(df['temperature'], bins=[0, 0.4, 0.7, 1.0], 
                                labels=['Low T', 'Med T', 'High T'])
        
        for temp_bin in ['Low T', 'Med T', 'High T']:
            temp_data = df[df['temp_bin'] == temp_bin]
            
            for model in sorted(df['model_name'].unique()):
                model_data = temp_data[temp_data['model_name'] == model]
                
                if len(model_data) > 0:
                    success_count = model_data['success'].sum()
                    fail_count = len(model_data) - success_count
                    
                    model_short = model.split('.')[0].split('-')[0][:8]
                    
                    flows.append({
                        'source': temp_bin,
                        'target': model_short,
                        'value': len(model_data)
                    })
                    
                    flows.append({
                        'source': model_short,
                        'target': 'Success',
                        'value': success_count
                    })
                    
                    flows.append({
                        'source': model_short,
                        'target': 'Failure',
                        'value': fail_count
                    })
        
        # Create node labels
        all_nodes = list(set([f['source'] for f in flows] + [f['target'] for f in flows]))
        node_dict = {node: i for i, node in enumerate(all_nodes)}
        
        # Build Sankey data
        source_indices = [node_dict[f['source']] for f in flows]
        target_indices = [node_dict[f['target']] for f in flows]
        values = [f['value'] for f in flows]
        
        # Node colors
        node_colors = []
        for node in all_nodes:
            if 'Low' in node:
                node_colors.append('#3498db')
            elif 'Med' in node:
                node_colors.append('#f39c12')
            elif 'High' in node:
                node_colors.append('#e74c3c')
            elif node == 'Success':
                node_colors.append('#27ae60')
            elif node == 'Failure':
                node_colors.append('#95a5a6')
            else:
                node_colors.append('#9b59b6')
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=all_nodes,
                color=node_colors
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values
            )
        )])
        
        fig.update_layout(
            title_text="Sankey Diagram - Attack Flow (Temperature → Model → Result)",
            title_font_size=16,
            font_size=10,
            height=600,
            width=1200
        )
        
        file_path = output_dir / "example_sankey.html"
        fig.write_html(str(file_path))
        
        # Also save as static image
        try:
            file_path_png = output_dir / "example_sankey.png"
            fig.write_image(str(file_path_png), width=1200, height=600)
            print(f"[+] Generated: {file_path_png.name}")
        except:
            print(f"[+] Generated: {file_path.name} (HTML only - install kaleido for PNG)")
        
        return file_path
        
    except ImportError:
        print("[WARNING] plotly not installed. Install with: pip install plotly")
        return None
    except Exception as e:
        print(f"[ERROR] creating Sankey diagram: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_all_example_charts(df: pd.DataFrame, output_dir: Path):
    """Generate all 6 example advanced charts"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    print("\n" + "="*80)
    print("ADVANCED VISUALIZATION EXAMPLES - 6 NEW CHART TYPES")
    print("="*80)
    
    print("\n[1/6] Radar/Spider Chart...")
    f = create_radar_chart(df, output_dir)
    if f: generated_files.append(f)
    
    print("\n[2/6] ROC Curve...")
    f = create_roc_curve(df, output_dir)
    if f: generated_files.append(f)
    
    print("\n[3/6] Waterfall Chart...")
    f = create_waterfall_chart(df, output_dir)
    if f: generated_files.append(f)
    
    print("\n[4/6] Treemap...")
    f = create_treemap(df, output_dir)
    if f: generated_files.append(f)
    
    print("\n[5/6] Stacked Area Chart...")
    f = create_stacked_area_chart(df, output_dir)
    if f: generated_files.append(f)
    
    print("\n[6/6] Sankey Diagram...")
    f = create_sankey_diagram(df, output_dir)
    if f: generated_files.append(f)
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated {len(generated_files)} example charts")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")
    
    return generated_files


if __name__ == "__main__":
    # Test with real data
    data_file = Path("data/outputs/session_20251107_031023/complete_results_20251107_031023.csv")
    if data_file.exists():
        df = pd.read_csv(data_file, encoding='utf-8')
        create_all_example_charts(df, Path("advanced_examples_output"))
    else:
        print(f"Data file not found: {data_file}")
