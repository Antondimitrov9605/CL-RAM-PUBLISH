#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Runner Module for GGUF Models
====================================
Author: Academic Research Team
Date: 2024
Version: 2.0.0
Purpose: Manages loading and running GGUF language models for research

This module handles the discovery, loading, and execution of GGUF models
using llama-cpp-python for academic security research.
"""

import os
import gc
import json
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import re

# Language detection for response analysis
try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not installed. Run: pip install langdetect")

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not installed")


class ModelRunner:
    """
    GGUF Model Runner for Research Framework

    Handles discovery, loading, and execution of GGUF models
    with proper memory management and error handling.

    Attributes:
        current_model: Currently loaded Llama model instance
        current_model_name: Name of currently loaded model
        model_cache: Cache of discovered models
        logger: Logger instance
    """

    def __init__(self, config=None):
        """
        Initialize Model Runner

        Args:
            config: Optional configuration object
        """
        self.current_model = None
        self.current_model_name = None
        self.model_cache = {}
        self.config = config

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Check llama-cpp availability
        if not LLAMA_CPP_AVAILABLE:
            self.logger.error("llama-cpp-python not available")

    def discover_available_models(self) -> List[Dict[str, Any]]:
        """
        Discover all available GGUF models in search paths

        Returns:
            List of dictionaries containing model information
        """
        models = []

        # Define search paths
        search_paths = [
            Path.cwd(),  # Current directory
            Path.cwd() / "models",  # models subdirectory
            Path.cwd() / "data" / "models",  # data/models subdirectory
            Path.cwd().parent,  # Parent directory
            Path.cwd().parent / "models",  # Parent models subdirectory
            Path.cwd().parent / "data" / "models",  # Parent data/models subdirectory
        ]
        
        self.logger.info(f"Searching for models in: {[str(p) for p in search_paths]}")

        # Search for GGUF files
        seen_models = set()  # Track unique models

        for search_path in search_paths:
            if not search_path.exists():
                continue

            # Search for .gguf files recursively (but not too deep)
            for pattern in ['*.gguf', '**/*.gguf']:
                for model_path in search_path.glob(pattern):
                    # Skip if already seen
                    if model_path.name in seen_models:
                        continue

                    seen_models.add(model_path.name)

                    # Get file info
                    try:
                        file_size = model_path.stat().st_size
                        file_size_gb = file_size / (1024**3)

                        # Check if model can run based on available RAM
                        available_ram = psutil.virtual_memory().available / (1024**3)
                        can_run = file_size_gb < (available_ram * 0.8)  # Use 80% of available RAM

                        model_info = {
                            'name': model_path.stem,  # Name without extension
                            'path': str(model_path.absolute()),
                            'file_size': file_size,
                            'file_size_gb': file_size_gb,
                            'can_run': can_run,
                            'directory': str(model_path.parent)
                        }

                        models.append(model_info)
                        self.logger.info(f"Found model: {model_path.name} ({file_size_gb:.1f}GB)")

                    except Exception as e:
                        self.logger.warning(f"Error processing {model_path}: {e}")

        # Sort by file size (smallest first)
        models.sort(key=lambda x: x['file_size'])

        self.logger.info(f"Total models discovered: {len(models)}")
        return models

    def get_runnable_models(self) -> List[Dict[str, Any]]:
        """
        Get only models that can run with available RAM

        Returns:
            List of runnable model dictionaries
        """
        all_models = self.discover_available_models()
        runnable = [m for m in all_models if m['can_run']]

        self.logger.info(f"Runnable models: {len(runnable)}/{len(all_models)}")
        return runnable

    def load_model(self, model_name: str) -> bool:
        """
        Load a GGUF model by name

        Args:
            model_name: Name of the model to load

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            # If same model already loaded, skip
            if self.current_model_name == model_name:
                self.logger.info(f"Model {model_name} already loaded")
                return True

            # Unload current model first
            self.unload_model()

            # Find model path
            models = self.discover_available_models()
            model_info = None

            for model in models:
                if model['name'] == model_name or model_name in model['path']:
                    model_info = model
                    break

            if not model_info:
                self.logger.error(f"Model not found: {model_name}")
                return False

            model_path = model_info['path']
            self.logger.info(f"Loading model: {model_name} from {model_path}")

            # Determine GPU layers based on available VRAM
            n_gpu_layers = self._calculate_gpu_layers(model_info['file_size_gb'])

            # Load model with llama-cpp
            # Load model with llama-cpp - OPTIMIZED FOR GPU
            self.logger.info(f"🔧 Loading with {n_gpu_layers} GPU layers...")

            self.current_model = Llama(
                model_path=model_path,
                n_ctx=1024,  # Context size kept safe for prompt + response
                n_batch=512,  # Batch size
                n_gpu_layers=-1,  # GPU acceleration
                verbose=True,  # Show loading progress
                n_threads=os.cpu_count() // 2,  # Use half of CPU cores
                seed=42,  # For reproducibility
                use_mlock=True,  # Lock model in RAM
                use_mmap=True,  # Memory mapping for faster loading
                low_vram=False  # Use full VRAM (not low VRAM mode)
            )

            self.current_model_name = model_name
            self.logger.info(f"✅ Model loaded successfully: {model_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            self.current_model = None
            self.current_model_name = None
            return False

    def _calculate_gpu_layers(self, model_size_gb: float) -> int:
        """
        Calculate optimal GPU layers - MAXIMIZED FOR GPU USAGE

        Args:
            model_size_gb: Size of model in GB

        Returns:
            Number of layers to offload to GPU
        """
        # CRITICAL: UNCONDITIONAL FORCE GPU FOR RTX 5090
        # User confirmed all models fit. We force -1 (all layers) always.
        try:
             self.logger.info(f"🚀 FORCING GPU OFFLOAD: -1 layers (ALL) for model size {model_size_gb:.2f}GB")
        except:
             pass
        return -1

    def unload_model(self) -> None:
        """
        Unload current model and free memory
        """
        if self.current_model:
            self.logger.info(f"Unloading model: {self.current_model_name}")

            try:
                del self.current_model
                self.current_model = None
                self.current_model_name = None

                # Force garbage collection
                gc.collect()

                # Clear GPU cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass

                self.logger.info("✅ Model unloaded and memory freed")

            except Exception as e:
                self.logger.error(f"Error unloading model: {e}")

    def generate_response(self, prompt: str, temperature: float = 0.7,
                          max_tokens: int = 512, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate response from loaded model

        Args:
            prompt: Input prompt text
            temperature: Generation temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary containing response and metadata
        """
        if not self.current_model:
            return {
                'success': False,
                'error': 'No model loaded',
                'response': ''
            }

        try:
            # Generate response
            start_time = datetime.now()

            output = self.current_model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
            output = self.current_model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
                stop=stop if stop else ["Human:", "User:", "Assistant:"], # Removed \n\n\n to prevent early cutoff
                echo=False  # Don't include prompt in response
            )

            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()

            # Extract response text
            response_text = output['choices'][0]['text'].strip()

            # --- DEEPSEEK CLEANING LOGIC ---
            # Remove <think>...</think> blocks if present
            if "<think>" in response_text:
                # Use non-greedy match with DOTALL to catch multi-line thoughts
                clean_pattern = r'<think>(.*?)</think>'
                thoughts = re.findall(clean_pattern, response_text, flags=re.DOTALL)
                
                # Strip the tags to get the "clean" response
                result_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
                
                # FALLBACK LOGIC: If cleaning leaves us with nothing, but we had thoughts,
                # implies the model put its entire refusal/response inside the thoughts.
                if not result_text and thoughts:
                    self.logger.warning(f"⚠️ Empty response after cleaning! Reverting to content inside <think> tags (first block).")
                    # Return the content of the thought (likely the refusal)
                    result_text = thoughts[0].strip()
                
                response_text = result_text

            # --- PROMPT SEPARATION LOGIC ---
            # Ensure the response doesn't contain the prompt (echo leak fix)
            clean_prompt = prompt.strip()
            if response_text.startswith(clean_prompt):
                self.logger.warning(f"Response contained prompt (echo leak). Stripping {len(clean_prompt)} chars...")
                response_text = response_text[len(clean_prompt):].strip()
            
            # FINAL FALLBACK: If we are left with nothing at all
            if not response_text:
                response_text = "[NO_RESPONSE_GENERATED]"

            # --- RESPONSE LANGUAGE DETECTION ---
            response_language = 'unknown'
            response_language_confidence = 0.0
            
            if LANGDETECT_AVAILABLE and response_text and response_text != "[NO_RESPONSE_GENERATED]":
                try:
                    # Get language with confidence scores
                    lang_probs = detect_langs(response_text[:500])  # Use first 500 chars for speed
                    if lang_probs:
                        response_language = lang_probs[0].lang  # e.g., 'en', 'bg', 'ru'
                        response_language_confidence = lang_probs[0].prob
                        self.logger.info(f"🌍 Response Language: {response_language.upper()} (confidence: {response_language_confidence:.2f})")
                except LangDetectException:
                    response_language = 'unknown'
                except Exception as e:
                    self.logger.warning(f"Language detection failed: {e}")

            return {
                'success': True,
                'response': response_text,
                'response_language': response_language,
                'response_language_confidence': response_language_confidence,
                'model': self.current_model_name,
                'temperature': temperature,
                'generation_time': generation_time,
                'token_count': output['usage']['completion_tokens'],
                'prompt_tokens': output['usage']['prompt_tokens'],
                'total_tokens': output['usage']['total_tokens']
            }

        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': ''
            }

    def batch_generate(self, prompts: List[str], temperature: float = 0.7,
                       max_tokens: int = 1000) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple prompts

        Args:
            prompts: List of input prompts
            temperature: Generation temperature
            max_tokens: Maximum tokens per response

        Returns:
            List of response dictionaries
        """
        results = []

        for i, prompt in enumerate(prompts):
            self.logger.info(f"Generating response {i+1}/{len(prompts)}")
            result = self.generate_response(prompt, temperature, max_tokens)
            results.append(result)

        return results

    def test_model(self, model_name: str) -> bool:
        """
        Test if a model can be loaded and generate text

        Args:
            model_name: Name of model to test

        Returns:
            True if model works, False otherwise
        """
        try:
            # Try to load model
            if not self.load_model(model_name):
                return False

            # Try simple generation
            test_prompt = "Hello, please respond with a short greeting."
            result = self.generate_response(test_prompt, temperature=0.5, max_tokens=50)

            # Check if generation successful
            if result['success'] and len(result['response']) > 0:
                self.logger.info(f"✅ Model test passed: {model_name}")
                return True
            else:
                self.logger.error(f"❌ Model test failed: {model_name}")
                return False

        except Exception as e:
            self.logger.error(f"Model test error: {e}")
            return False
        finally:
            self.unload_model()


def create_model_runner(config=None) -> ModelRunner:
    """
    Factory function to create model runner

    Args:
        config: Optional configuration object

    Returns:
        ModelRunner instance
    """
    return ModelRunner(config)


# Testing
if __name__ == "__main__":
    print("Model Runner Test")
    print("=" * 60)

    # Create runner
    runner = create_model_runner()

    # Discover models
    print("\n🔍 Discovering models...")
    models = runner.discover_available_models()

    print(f"\n📦 Found {len(models)} models:")
    for i, model in enumerate(models, 1):
        status = "✅" if model['can_run'] else "⚠️"
        print(f"  {i}. {status} {model['name']} ({model['file_size_gb']:.1f}GB)")

    # Get runnable models
    runnable = runner.get_runnable_models()
    print(f"\n🚀 Runnable models: {len(runnable)}")

    # Test first runnable model if available
    if runnable:
        test_model = runnable[0]['name']
        print(f"\n🧪 Testing model: {test_model}")

        if runner.test_model(test_model):
            print("✅ Model test successful!")
        else:
            print("❌ Model test failed!")

    print("\n✅ Model Runner test complete!")