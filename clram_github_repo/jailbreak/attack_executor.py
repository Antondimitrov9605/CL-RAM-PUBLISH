#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack Executor Module for Research Framework
==============================================
Author: Academic Research Team
Date: 2026
Version: 2.0.0
Purpose: Coordinates attack execution across models and languages

This module manages the execution of prompts against language models,
handling model loading, response generation, and result collection.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

from .model_runner import create_model_runner


class AttackExecutor:
    """
    Attack Execution Coordinator

    Manages the execution of security research prompts against
    language models with proper logging and error handling.

    Attributes:
        config: Configuration object
        model_runner: Model runner instance
        results: List of execution results
        logger: Logger instance
    """

    def __init__(self, config):
        """
        Initialize Attack Executor

        Args:
            config: Research configuration object
        """
        self.config = config
        self.model_runner = create_model_runner(config)
        self.results = []
        self.logger = logging.getLogger(__name__)

    def execute_single_attack(self, prompt: str, model_name: str,
                            temperature: float = 0.7,
                            max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Execute a single attack/prompt against a model

        Args:
            prompt: Input prompt text
            model_name: Name of model to use
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary containing execution results
        """
        # Load model if needed
        if self.model_runner.current_model_name != model_name:
            self.logger.info(f"Loading model: {model_name}")
            if not self.model_runner.load_model(model_name):
                return {
                    'success': False,
                    'error': f'Failed to load model {model_name}',
                    'response': '',
                    'model': model_name,
                    'temperature': temperature
                }

        # Generate response WITH max_tokens
        self.logger.info(f"Generating response with T={temperature}, max_tokens={max_tokens}")
        result = self.model_runner.generate_response(prompt, temperature, max_tokens)

        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['prompt'] = prompt
        result['model'] = model_name

        # Log the response for debugging
        if result.get('success'):
            response_preview = result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
            self.logger.info(f"Response preview: {response_preview}")
            self.logger.info(f"Response length: {len(result['response'])} chars")

        return result

    def execute_batch_attacks(self, prompts: List[str], model_name: str,
                            temperature: float = 0.7,
                            max_tokens: int = 1000) -> List[Dict[str, Any]]:
        """
        Execute multiple attacks in batch

        Args:
            prompts: List of prompt texts
            model_name: Name of model to use
            temperature: Generation temperature
            max_tokens: Maximum tokens per response

        Returns:
            List of result dictionaries
        """
        results = []

        # Load model once for batch
        if not self.model_runner.load_model(model_name):
            self.logger.error(f"Failed to load model for batch: {model_name}")
            return results

        for i, prompt in enumerate(prompts):
            self.logger.info(f"Executing attack {i+1}/{len(prompts)}")
            result = self.execute_single_attack(prompt, model_name, temperature, max_tokens)
            results.append(result)

        return results

    def unload_all_models(self):
        """Unload any loaded models to free memory"""
        self.model_runner.unload_model()

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze attack results for summary statistics

        Args:
            results: List of result dictionaries

        Returns:
            Dictionary containing analysis summary
        """
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))

        analysis = {
            'total_attacks': total,
            'successful_attacks': successful,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'failed_attacks': total - successful,
            'timestamp': datetime.now().isoformat()
        }

        return analysis


def create_attack_executor(config):
    """
    Factory function to create attack executor

    Args:
        config: Research configuration object

    Returns:
        AttackExecutor instance
    """
    return AttackExecutor(config)


# Testing
if __name__ == "__main__":
    print("Attack Executor Test")
    print("=" * 60)

    # Mock config
    class MockConfig:
        pass

    config = MockConfig()

    # Create executor
    executor = create_attack_executor(config)
    print("✅ Attack executor created")

    # Test with available models
    models = executor.model_runner.discover_available_models()
    if models:
        test_model = models[0]['name']
        print(f"\n🧪 Testing with model: {test_model}")

        # Test single attack WITH max_tokens
        test_prompt = "Explain how to write a simple Python function."
        result = executor.execute_single_attack(test_prompt, test_model, 0.7, 1000)

        if result['success']:
            print(f"✅ Attack executed successfully")
            print(f"Response length: {len(result['response'])} chars")
        else:
            print(f"❌ Attack failed: {result.get('error')}")

    print("\n✅ Attack Executor test complete!")
