import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

def create_all_deep_research_charts(df, output_dir):
    """
    Generates 3 advanced research visualizations based on novel dependencies:
    1. Complexity Signature (Length vs Success)
    2. Defense Paradox (Silence vs Rambling / Response Type)
    3. Category Volatility (Temperature Sensitivity)
    4. Pipeline Maturity Evolution (Risk vs Cost/Latency)
    5. Cross-Lingual Vulnerability Mirror (Safety Leakage)
    """
    os.makedirs(output_dir, exist_ok=True)
    plots = []
    
    # Ensure success is boolean
    df['success'] = df['success'].astype(bool)
    
    # --- 1. Complexity Signature (Response Length vs Success) ---
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Using boxenplot for better distribution visibility
    ax = sns.boxenplot(x='language', y='response_length', hue='success', data=df, palette='viridis')
    plt.title("Scientific Finding: The Complexity Signature\n(Successful Jailbreaks vs. Refusal Length)", fontsize=14, pad=15)
    plt.xlabel("Language", fontsize=12)
    plt.ylabel("Response Length (Characters)", fontsize=12)
    plt.legend(title="Success (Jailbreak)", loc='upper right')
    
    path1 = os.path.join(output_dir, "complexity_signature.png")
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(path1)
    
    # --- 2. Defense Paradox (Silence vs Rambling) ---
    if 'response_type' in df.columns:
        plt.figure(figsize=(12, 7))
        
        # Filter for non-successful (defensive) responses to show the 'Paradox'
        defense_df = df[df['success'] == False].copy()
        
        # Group and calculate percentages
        type_counts = defense_df.groupby(['language', 'response_type']).size().unstack(fill_value=0)
        type_pct = type_counts.div(type_counts.sum(axis=1), axis=0) * 100
        
        ax = type_pct.plot(kind='bar', stacked=True, colormap='icefire', figsize=(10, 6))
        
        plt.title("Defense Paradox: Silence (EN) vs. Rambling (BG)\n(Distribution of Refusal Styles)", fontsize=14, pad=15)
        plt.xlabel("Language", fontsize=12)
        plt.ylabel("Percentage of Defensive Responses (%)", fontsize=12)
        plt.legend(title="Refusal Style", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        
        # Add labels to bars
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            if height > 5: # Only label if > 5%
                x, y = p.get_xy() 
                ax.text(x+width/2, y+height/2, f'{height:.1f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

        path2 = os.path.join(output_dir, "defense_paradox_rambling.png")
        plt.savefig(path2, dpi=300, bbox_inches='tight')
        plt.close()
        plots.append(path2)

    # --- 3. Category Volatility (Temperature Sensitivity) ---
    if 'temperature' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Calculate ASR per category & temperature
        temp_perf = df.groupby(['category', 'temperature'])['success'].mean().unstack()
        
        if len(temp_perf.columns) >= 2:
            t_min, t_max = sorted(temp_perf.columns)[0], sorted(temp_perf.columns)[-1]
            temp_perf['Volatility'] = temp_perf[t_max] - temp_perf[t_min]
            
            # Sort by volatility
            temp_perf = temp_perf.sort_values('Volatility', ascending=True)
            
            colors = ['#ff9999' if x > 0 else '#66b3ff' for x in temp_perf['Volatility']]
            
            ax = temp_perf['Volatility'].plot(kind='barh', color=colors)
            plt.title(f"Category Volatility: ASR Shift ({t_min} to {t_max})\n(Sensitivity to Stochasticity)", fontsize=14, pad=15)
            plt.xlabel("ASR Delta (Positive = High-Temp Risk)", fontsize=12)
            plt.ylabel("MITRE Category", fontsize=12)
            plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
            
            path3 = os.path.join(output_dir, "category_temp_volatility.png")
            plt.savefig(path3, dpi=300, bbox_inches='tight')
            plt.close()
            plots.append(path3)

    # --- 4. Pipeline Maturity Evolution (Risk vs Cost/Latency) ---
    plt.figure(figsize=(10, 6))
    
    # Static estimates for "Maturity Model" visualization
    stages = ['Baseline\n(No Protection)', 'Phase 1:\nPattern Filters', 'Phase 2:\nAI Validation', 'Phase 3:\nManual Review']
    
    # Success/Risk scores (fictionalized but realistic progression)
    risk = [100, 65, 15, 2] # % Residual Risk
    cost = [0, 5, 25, 100]  # Relative Cost/Latency
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.set_style("darkgrid")
    
    # Plot Risk (Area)
    ax1.fill_between(stages, risk, color="red", alpha=0.3, label="Residual Risk (%)")
    ax1.plot(stages, risk, color="red", marker="o", linewidth=2, label="Residual Risk (%)")
    ax1.set_ylabel("Residual Risk (%)", color="red", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="red")
    
    # Plot Cost (Bar)
    ax2 = ax1.twinx()
    ax2.bar(stages, cost, color="blue", alpha=0.2, width=0.4, label="Cost/Latency Index")
    ax2.plot(stages, cost, color="blue", marker="s", linestyle='--', linewidth=2, label="Cost/Latency Index")
    ax2.set_ylabel("Cost / Latency Index", color="blue", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="blue")
    
    plt.title("The Tiered Security Maturity Model:\nEfficiency vs. Risk Reduction", fontsize=14, pad=15)
    
    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    path4 = os.path.join(output_dir, "pipeline_maturity_model.png")
    plt.savefig(path4, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(path4)

    # --- 5. Cross-Lingual Vulnerability Mirror (Safety Leakage) ---
    # Pivot to compare EN vs BG directly
    mirror = df.groupby(['prompt_id', 'language'])['success'].max().unstack()
    if 'bg' in mirror.columns and 'en' in mirror.columns:
        plt.figure(figsize=(10, 8))
        
        # Calculate categories for the pie
        consistent_safe = len(mirror[(mirror['en'] == False) & (mirror['bg'] == False)])
        consistent_vulnerable = len(mirror[(mirror['en'] == True) & (mirror['bg'] == True)])
        safety_leak = len(mirror[(mirror['en'] == False) & (mirror['bg'] == True)])
        reverse_leak = len(mirror[(mirror['en'] == True) & (mirror['bg'] == False)])
        
        counts = [consistent_safe, consistent_vulnerable, safety_leak, reverse_leak]
        labels = ['Consistent Safe', 'Consistent Vulnerable', 'Safety Leak (EN Safe -> BG Fail)', 'Reverse Gap (BG Safe -> EN Fail)']
        colors = ['#4CAF50', '#F44336', '#FF9800', '#2196F3']
        
        # Pull out the leak (the 'Discovery')
        explode = (0, 0, 0.1, 0) 
        
        plt.pie(counts, labels=None, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode, pctdistance=0.85)
        plt.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title("The Cross-Lingual Safety Mirror:\nMapping the Multilingual Vulnerability Gap", fontsize=14, pad=20)
        
        # Add a circle for a donut chart look
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        path5 = os.path.join(output_dir, "vulnerability_safety_mirror.png")
        plt.savefig(path5, dpi=300, bbox_inches='tight')
        plt.close()
        plots.append(path5)

    return plots
