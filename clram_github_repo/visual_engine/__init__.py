"""
Visual Engine Module
====================
Modular visualization system for CL-RAM framework.

Includes:
- Category-Temperature Heatmaps (Category × Temperature)
- Category-Language Heatmaps (Category × Language: BG vs EN)
- Category Performance Linear (Lollipop charts ranking categories by performance)
- Model-Temperature Heatmaps (Model × Temperature)
- Temperature Effect Charts (All models comparison with risk zones)
- Temperature Sensitivity Charts (Min vs Max temperature per category)
- Temperature Distribution Charts (Success/Failed donut charts per temperature)
- 3D Surface Plots (Temperature × Model × Success Rate)
- Category Line Charts (BG vs EN with data tables)
- Statistical Tables (Model summary, Category vulnerability, Temperature analysis)
- Model Performance Progression (Linear progression charts + comparison table)
"""

from .category_temperature_heatmaps import create_all_category_temperature_heatmaps
from .category_language_heatmaps import create_all_category_language_heatmaps
from .model_temperature_heatmaps import create_all_model_temperature_heatmaps
from .temperature_effect_charts import create_all_temperature_effect_charts
from .temperature_sensitivity_charts import create_all_temperature_sensitivity_charts
from .temperature_distribution_charts import create_all_temperature_distribution_charts
from .category_distribution_pies import create_all_category_distribution_pies
from .surface_3d_plots import create_all_3d_surface_plots
from .statistical_tables import create_all_statistical_tables
from .model_performance_progression import create_all_model_progression_charts
from .category_performance_linear import create_all_category_performance_charts
from .category_performance_enhanced import create_all_enhanced_category_charts
from .category_bar_charts import create_all_category_bar_charts
from .language_comparison_charts import create_all_language_charts
from .model_overview_charts import create_all_model_overview_charts
from .attack_distribution_charts import create_all_attack_distribution_charts
from .generate_all_charts import generate_all_visualizations
from .phase_transition_analyzer import create_all_phase_transition_charts
from .crosslingual_transfer_chart import create_all_crosslingual_charts
from .response_entropy_chart import create_all_response_entropy_charts
from .scientific_discoveries_charts import create_all_scientific_discovery_charts
from .category_line_charts import (
    create_category_chart,
    create_model_overall_chart,
    main as generate_all_line_charts
)

__all__ = [
    'create_all_model_overview_charts',
    'create_all_attack_distribution_charts',
    'generate_all_visualizations',
    'create_category_chart',
    'create_model_overall_chart',
    'generate_all_line_charts',
    'create_all_phase_transition_charts',
    'create_all_crosslingual_charts',
    'create_all_response_entropy_charts',
    'create_all_scientific_discovery_charts',
    'create_all_deep_research_charts'
]

__version__ = '3.4.0'
