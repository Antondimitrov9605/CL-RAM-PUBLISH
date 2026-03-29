#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module for Comparative Language Research Framework
================================================================
Author: Academic Research Team
Date: 2025
Purpose: Configuration management for multilingual AI model analysis
Version: 2.0.0

This module manages all configuration parameters for the research framework,
including model paths, language settings, and experimental parameters.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import psutil
import platform


@dataclass
class SystemInfo:
    """
    System Information Container

    Attributes:
        total_ram_gb: Total system RAM in gigabytes
        available_ram_gb: Available system RAM in gigabytes
        cpu_count: Number of CPU cores
        gpu_available: Whether GPU acceleration is available
        gpu_memory_gb: GPU memory in gigabytes (if available)
        platform_info: Operating system platform
        python_version: Python interpreter version
        llama_cpp_available: Whether llama-cpp-python is installed
    """
    total_ram_gb: float = field(default_factory=lambda: psutil.virtual_memory().total / (1024 ** 3))
    available_ram_gb: float = field(default_factory=lambda: psutil.virtual_memory().available / (1024 ** 3))
    cpu_count: int = field(default_factory=os.cpu_count)
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    platform_info: str = field(default_factory=platform.platform)  # Renamed from 'platform' to avoid conflict
    python_version: str = field(default_factory=platform.python_version)  # This is the function reference
    llama_cpp_available: bool = False

    def __post_init__(self):
        """Post-initialization to check GPU availability and llama-cpp"""
        # Check for GPU
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except ImportError:
            self.gpu_available = False
            self.gpu_memory_gb = 0.0

        # Check for llama-cpp-python
        try:
            import llama_cpp
            self.llama_cpp_available = True
        except ImportError:
            self.llama_cpp_available = False


class ResearchConfig:
    """
    Main Configuration Class for Research Framework

    This class centralizes all configuration parameters needed for the
    comparative language effectiveness research on AI models.

    Attributes:
        experiment_name: Name identifier for the current experiment
        temperature_mode: Temperature testing mode ('standard' or 'extended')
        temperatures: List of temperature values to test
        active_languages: Dictionary of active languages for testing
        system_info: System information object
        directories: Dictionary of project directories
        research_parameters: Dictionary of research-specific parameters
    """

    def __init__(self, experiment_name: str = "CL-Research"):
        """
        Initialize Research Configuration

        Args:
            experiment_name: Identifier for the experiment session
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().isoformat()

        # Setup logging
        self._setup_logging()

        # Temperature settings for model generation
        self.temperature_mode = 'standard'
        self.temperatures = self._get_temperature_settings()

        # Language configuration
        self.active_languages = self._setup_languages()

        # System information
        self.system_info = SystemInfo()

        # Project structure
        self.project_root = Path.cwd()
        self.directories = self._setup_directories()

        # Model directories
        self.models_dir = self.directories['models']
        self.outputs_dir = self.directories['outputs']

        # Research parameters
        self.research_parameters = {
            'total_base_prompts': 140,
            'mitre_categories': 14,
            'prompts_per_category': 10,
            'max_response_tokens': 500,
            'model_context_size': 4096,
            'batch_size': 1,
            'save_checkpoints': True,
            'checkpoint_frequency': 50
        }

        # MITRE specific settings
        self.total_base_prompts = 140
        self.mitre_categories = 14
        self.prompts_per_category = 10

        # Validate configuration
        self._validate_configuration()

        self.logger.info(f"Configuration initialized for experiment: {experiment_name}")

    def _setup_logging(self) -> None:
        """
        Configure logging system

        Sets up both file and console logging with appropriate formatting
        and log levels for academic research documentation.
        """
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging system initialized")

    def _get_temperature_settings(self) -> List[float]:
        """
        Get temperature settings based on mode

        Returns:
            List of temperature values for testing

        Note:
            Temperature affects randomness in model outputs.
            Lower values (0.1-0.3) produce more deterministic outputs.
            Higher values (0.7-1.0) produce more creative/random outputs.
        """
        if self.temperature_mode == 'extended':
            return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        else:  # standard mode
            return [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    def _setup_languages(self) -> Dict[str, Dict[str, Any]]:
        """
        Configure languages for comparative analysis

        Returns:
            Dictionary containing language configurations

        Note:
            This research compares English (baseline) with Bulgarian
            to analyze language-specific model behaviors.
        """
        return {
            'en': {
                'name': 'English',
                'native_name': 'English',
                'iso_code': 'en',
                'script': 'latin',
                'role': 'baseline',
                'priority': 1,
                'description': 'Control language for comparison'
            },
            'bg': {
                'name': 'Bulgarian',
                'native_name': 'Български',
                'iso_code': 'bg',
                'script': 'cyrillic',
                'role': 'experimental',
                'priority': 2,
                'description': 'Test language for comparative analysis'
            }
        }

    def _setup_directories(self) -> Dict[str, Path]:
        """
        Create and configure project directory structure

        Returns:
            Dictionary mapping directory names to Path objects
        """
        directories = {
            'data': self.project_root / "data",
            'inputs': self.project_root / "data" / "inputs",
            'outputs': self.project_root / "data" / "outputs",
            'models': self.project_root / "data" / "models",
            'visualizations': self.project_root / "data" / "visualizations",
            'checkpoints': self.project_root / "data" / "checkpoints",
            'cache': self.project_root / "data" / "cache",
            'logs': self.project_root / "logs"
        }

        # Create all directories
        for name, path in directories.items():
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Directory ensured: {name} -> {path}")

        return directories

    def _validate_configuration(self) -> None:
        """
        Validate configuration parameters

        Raises:
            ValueError: If configuration parameters are invalid
        """
        # Validate temperatures
        for temp in self.temperatures:
            if not 0.0 <= temp <= 2.0:
                raise ValueError(f"Invalid temperature value: {temp}. Must be between 0.0 and 2.0")

        # Validate system resources
        if self.system_info.available_ram_gb < 4.0:
            self.logger.warning(
                f"Low RAM warning: {self.system_info.available_ram_gb:.1f}GB available. "
                f"Recommend at least 8GB for stable operation."
            )

        # Check for llama-cpp-python
        if not self.system_info.llama_cpp_available:
            self.logger.warning(
                "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            )

        # Validate research parameters
        if self.research_parameters['prompts_per_category'] <= 0:
            raise ValueError("Prompts per category must be positive")

        self.logger.info("Configuration validation passed")

    def get_model_search_paths(self) -> List[Path]:
        """
        Get paths to search for GGUF model files

        Returns:
            List of paths where models might be located
        """
        paths = [
            self.project_root,
            self.directories['models'],
            self.project_root.parent,
            Path.home() / "models",
            Path("/models")
        ]

        # Filter only existing paths
        existing_paths = [p for p in paths if p.exists()]

        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in existing_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)

        return unique_paths

    def save_configuration(self, filepath: Optional[Path] = None) -> Path:
        """
        Save current configuration to JSON file

        Args:
            filepath: Optional custom filepath for configuration

        Returns:
            Path to saved configuration file
        """
        if filepath is None:
            filepath = self.directories['outputs'] / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        config_dict = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'temperature_mode': self.temperature_mode,
            'temperatures': self.temperatures,
            'languages': self.active_languages,
            'research_parameters': self.research_parameters,
            'system_info': {
                'total_ram_gb': self.system_info.total_ram_gb,
                'available_ram_gb': self.system_info.available_ram_gb,
                'cpu_count': self.system_info.cpu_count,
                'gpu_available': self.system_info.gpu_available,
                'gpu_memory_gb': self.system_info.gpu_memory_gb,
                'platform': self.system_info.platform_info,
                'python_version': self.system_info.python_version,
                'llama_cpp_available': self.system_info.llama_cpp_available
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Configuration saved to: {filepath}")
        return filepath

    def validate_environment(self) -> Dict[str, bool]:
        """
        Validate research environment setup

        Returns:
            Dictionary of validation check results
        """
        checks = {
            'directories_exist': all(d.exists() for d in self.directories.values()),
            'sufficient_ram': self.system_info.available_ram_gb >= 4.0,
            'models_directory_accessible': os.access(self.directories['models'], os.W_OK),
            'output_directory_writable': os.access(self.directories['outputs'], os.W_OK),
            'python_version_compatible': platform.python_version_tuple() >= ('3', '8', '0'),
            'llama_cpp_available': self.system_info.llama_cpp_available
        }

        # Check for required packages
        required_packages = ['pandas', 'numpy', 'matplotlib', 'llama_cpp']
        for package in required_packages:
            try:
                __import__(package)
                checks[f'{package}_available'] = True
            except ImportError:
                checks[f'{package}_available'] = False
                self.logger.warning(f"Required package not found: {package}")

        return checks

    def get_experiment_scope(self) -> Dict[str, Any]:
        """
        Calculate scope of experiment

        Returns:
            Dictionary containing experiment scope metrics
        """
        num_languages = len(self.active_languages)
        num_temperatures = len(self.temperatures)
        num_prompts = self.research_parameters['total_base_prompts']

        total_experiments = num_languages * num_temperatures * num_prompts

        # Estimate time (assuming 3 seconds per experiment)
        estimated_seconds = total_experiments * 3
        estimated_hours = estimated_seconds / 3600

        return {
            'languages': num_languages,
            'temperatures': num_temperatures,
            'base_prompts': num_prompts,
            'total_experiments': total_experiments,
            'estimated_time_hours': estimated_hours,
            'mitre_categories': self.research_parameters['mitre_categories'],
            'prompts_per_category': self.research_parameters['prompts_per_category']
        }


def create_local_config(temperature_mode: str = 'standard') -> ResearchConfig:
    """
    Factory function to create research configuration

    Args:
        temperature_mode: 'standard' or 'extended' temperature testing

    Returns:
        Configured ResearchConfig instance
    """
    config = ResearchConfig("CL-RAM-Research")
    config.temperature_mode = temperature_mode
    config.temperatures = config._get_temperature_settings()

    return config


# For testing the module independently
if __name__ == "__main__":
    print("Research Configuration Module Test")
    print("=" * 60)

    # Create configuration
    config = create_local_config('standard')

    # Display configuration
    print(f"\n📊 Experiment: {config.experiment_name}")
    print(f"🌡️ Temperature Mode: {config.temperature_mode}")
    print(f"🌡️ Temperatures: {config.temperatures}")
    print(f"🌍 Languages: {list(config.active_languages.keys())}")

    # System info
    print(f"\n💻 System Information:")
    print(f"   RAM: {config.system_info.available_ram_gb:.1f}/{config.system_info.total_ram_gb:.1f} GB")
    print(f"   CPU Cores: {config.system_info.cpu_count}")
    print(f"   GPU: {'Available' if config.system_info.gpu_available else 'Not available'}")
    print(f"   Python: {config.system_info.python_version}")
    print(f"   Platform: {config.system_info.platform_info}")
    print(f"   llama-cpp: {'✅ Installed' if config.system_info.llama_cpp_available else '❌ Not installed'}")

    # Validate environment
    print(f"\n✅ Environment Validation:")
    validation = config.validate_environment()
    for check, result in validation.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check}")

    # Experiment scope
    scope = config.get_experiment_scope()
    print(f"\n🔬 Experiment Scope:")
    print(f"   Total experiments: {scope['total_experiments']:,}")
    print(f"   Estimated time: {scope['estimated_time_hours']:.1f} hours")

    # Save configuration
    config_path = config.save_configuration()
    print(f"\n💾 Configuration saved to: {config_path}")

    print(f"\n✅ Configuration Module Test Completed Successfully!")