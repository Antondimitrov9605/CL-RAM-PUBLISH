#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Memory Manager for GGUF Model Research
================================================
Author: Academic Research Team
Date: 2024
Version: 3.0.0
Purpose: Intelligent memory management for large language model testing

This module provides comprehensive memory management to prevent crashes
and optimize performance when testing multiple GGUF models.
"""

import gc
import os
import psutil
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import json
import threading
import time

# Try GPU memory management
try:
    import torch

    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False


class MemoryManager:
    """
    Advanced memory management for model testing

    Features:
    - Real-time memory monitoring
    - Automatic garbage collection
    - GPU memory tracking
    - Model size prediction
    - Memory leak detection
    - Automatic cleanup triggers
    """

    # Class-level tracking
    _instance = None
    _lock = threading.Lock()
    _loaded_models: Dict[str, Any] = {}
    _memory_history: List[Dict] = []
    _cleanup_threshold = 85  # Trigger cleanup at 85% RAM usage
    _critical_threshold = 95  # Emergency cleanup at 95%

    def __new__(cls):
        """Singleton pattern for global memory management"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize memory manager"""
        self.logger = logging.getLogger(__name__)
        self.monitoring_active = False
        self.monitor_thread = None

        # System capabilities
        self.total_ram = psutil.virtual_memory().total
        self.cpu_count = psutil.cpu_count()

        # GPU capabilities
        self.gpu_available = False
        self.gpu_memory_total = 0
        self._detect_gpu()

        # Memory tracking
        self.baseline_memory = self._get_current_memory()
        self.peak_memory = self.baseline_memory

        self.logger.info(f"Memory Manager initialized - RAM: {self.total_ram / (1024 ** 3):.1f}GB")

    def _detect_gpu(self):
        """Detect and configure GPU capabilities"""
        if TORCH_AVAILABLE:
            try:
                self.gpu_available = True
                self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                self.logger.info(f"GPU detected: {self.gpu_memory_total / (1024 ** 3):.1f}GB VRAM")
            except:
                self.gpu_available = False

        if NVML_AVAILABLE and not self.gpu_available:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.gpu_memory_total = info.total
                    self.gpu_available = True
                    self.logger.info(f"GPU detected via NVML: {self.gpu_memory_total / (1024 ** 3):.1f}GB")
            except:
                pass

    def _get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        mem = psutil.virtual_memory()
        stats = {
            'ram_used': mem.used,
            'ram_available': mem.available,
            'ram_percent': mem.percent,
            'ram_used_gb': mem.used / (1024 ** 3),
            'ram_available_gb': mem.available / (1024 ** 3)
        }

        # GPU memory if available
        if self.gpu_available:
            if TORCH_AVAILABLE:
                try:
                    stats['gpu_allocated'] = torch.cuda.memory_allocated()
                    stats['gpu_reserved'] = torch.cuda.memory_reserved()
                    stats['gpu_allocated_gb'] = stats['gpu_allocated'] / (1024 ** 3)
                except:
                    pass
            elif NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    stats['gpu_used'] = info.used
                    stats['gpu_free'] = info.free
                    stats['gpu_used_gb'] = info.used / (1024 ** 3)
                except:
                    pass

        return stats

    def register_model(self, model_name: str, model_object: Any,
                       model_size_gb: float = None) -> bool:
        """
        Register a loaded model for tracking

        Args:
            model_name: Identifier for the model
            model_object: The actual model object
            model_size_gb: Estimated size in GB

        Returns:
            True if successfully registered
        """
        try:
            # Check memory before registration
            current_mem = self._get_current_memory()

            if current_mem['ram_percent'] > self._critical_threshold:
                self.logger.error(f"Critical memory usage: {current_mem['ram_percent']:.1f}%")
                self.emergency_cleanup()
                return False

            # Unload existing model with same name
            if model_name in self._loaded_models:
                self.unload_model(model_name)

            # Register new model
            self._loaded_models[model_name] = {
                'object': model_object,
                'loaded_at': datetime.now(),
                'size_gb': model_size_gb or self._estimate_model_size(model_object),
                'memory_before': current_mem['ram_used_gb'],
                'memory_after': None
            }

            # Update memory after loading
            time.sleep(0.5)  # Allow memory to settle
            after_mem = self._get_current_memory()
            self._loaded_models[model_name]['memory_after'] = after_mem['ram_used_gb']

            actual_size = after_mem['ram_used_gb'] - current_mem['ram_used_gb']
            self.logger.info(f"Model {model_name} registered - Size: {actual_size:.2f}GB")

            # Update peak memory
            if after_mem['ram_used_gb'] > self.peak_memory['ram_used_gb']:
                self.peak_memory = after_mem

            # Add to history
            self._memory_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'load',
                'model': model_name,
                'memory_gb': after_mem['ram_used_gb'],
                'memory_percent': after_mem['ram_percent']
            })

            return True

        except Exception as e:
            self.logger.error(f"Failed to register model {model_name}: {e}")
            return False

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model and free memory

        Args:
            model_name: Model identifier to unload

        Returns:
            True if successfully unloaded
        """
        if model_name not in self._loaded_models:
            return False

        try:
            before_mem = self._get_current_memory()

            # Delete model object
            model_info = self._loaded_models[model_name]
            del model_info['object']
            del self._loaded_models[model_name]

            # Force garbage collection
            gc.collect()

            # Clear GPU cache if available
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()

            # Measure memory freed
            time.sleep(0.5)
            after_mem = self._get_current_memory()
            freed = before_mem['ram_used_gb'] - after_mem['ram_used_gb']

            self.logger.info(f"Model {model_name} unloaded - Freed: {freed:.2f}GB")

            # Add to history
            self._memory_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'unload',
                'model': model_name,
                'memory_gb': after_mem['ram_used_gb'],
                'memory_freed_gb': freed
            })

            return True

        except Exception as e:
            self.logger.error(f"Failed to unload model {model_name}: {e}")
            return False

    def unload_all_models(self):
        """Unload all registered models"""
        models = list(self._loaded_models.keys())
        for model_name in models:
            self.unload_model(model_name)

        # Extra cleanup
        gc.collect()
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()

    def can_load_model(self, size_gb: float) -> Tuple[bool, str]:
        """
        Check if a model of given size can be loaded

        Args:
            size_gb: Estimated model size in GB

        Returns:
            Tuple of (can_load, reason)
        """
        current = self._get_current_memory()
        
        # Check VRAM first (Primary for GPU offloading)
        vram_sufficient = False
        if self.gpu_available:
            # Get real free VRAM from NVML or Torch
            gpu_free_gb = 0
            if 'gpu_free' in current: # From NVML
                 gpu_free_gb = current['gpu_free'] / (1024**3)
            elif 'gpu_used_gb' in current: # Estimated from used
                 # This path is shaky if we don't know total free, but let's try:
                 gpu_free_gb = (self.gpu_memory_total / (1024**3)) - current['gpu_used_gb']
            elif TORCH_AVAILABLE:
                 try:
                     free, total = torch.cuda.mem_get_info()
                     gpu_free_gb = free / (1024**3)
                 except:
                     pass

            # If model fits in VRAM (with 10% buffer)
            if gpu_free_gb > size_gb * 1.1:
                vram_sufficient = True
        
        # Check RAM
        # If VRAM is sufficient, we only need minimal RAM for OS/buffering (e.g. 4GB or 20% of size if mmapped)
        required_ram_gb = 4.0 if vram_sufficient else (size_gb * 1.2)
        
        if current['ram_available_gb'] < required_ram_gb:
             return False, f"Insufficient RAM: {current['ram_available_gb']:.1f}GB available, need {required_ram_gb:.1f}GB {'(VRAM offload enabled)' if vram_sufficient else '(CPU fallback)'}"

        if current['ram_percent'] > self._critical_threshold:
            return False, f"High memory usage: {current['ram_percent']:.1f}%"

        return True, "OK"

    def optimize_for_model(self, model_size_gb: float) -> Dict[str, Any]:
        """
        Optimize system for loading a model

        Args:
            model_size_gb: Size of model to load

        Returns:
            Optimization recommendations
        """
        recommendations = {
            'should_cleanup': False,
            'unload_models': [],
            'gpu_layers': 0,
            'batch_size': 512,
            'context_size': 2048
        }

        current = self._get_current_memory()

        # Determine if cleanup needed
        if current['ram_percent'] > 70:
            recommendations['should_cleanup'] = True

        # Check which models to unload
        if current['ram_available_gb'] < model_size_gb * 1.5:
            # Need to free memory
            for name, info in self._loaded_models.items():
                recommendations['unload_models'].append(name)
                if sum(m['size_gb'] for m in self._loaded_models.values()
                       if m['object'] in recommendations['unload_models']) >= model_size_gb:
                    break

        # GPU optimization
        if self.gpu_available:
            gpu_memory_gb = self.gpu_memory_total / (1024 ** 3)
            if model_size_gb < gpu_memory_gb * 0.8:
                recommendations['gpu_layers'] = 35  # Full offload
            elif model_size_gb < gpu_memory_gb * 1.5:
                recommendations['gpu_layers'] = 20  # Partial offload
            else:
                recommendations['gpu_layers'] = 10  # Minimal offload

        # Batch size optimization based on available RAM
        if current['ram_available_gb'] > 16:
            recommendations['batch_size'] = 1024
            recommendations['context_size'] = 1600  # Optimized to ~40% (User req: 35% = 1433, Max observed: 1374)
        elif current['ram_available_gb'] > 8:
            recommendations['batch_size'] = 512
            recommendations['context_size'] = 2048
        else:
            recommendations['batch_size'] = 256
            recommendations['context_size'] = 1024

        return recommendations

    def cleanup_memory(self, aggressive: bool = False) -> float:
        """
        Perform memory cleanup

        Args:
            aggressive: If True, performs more aggressive cleanup

        Returns:
            Amount of memory freed in GB
        """
        before = self._get_current_memory()

        # Standard cleanup
        gc.collect()

        if aggressive:
            # Aggressive cleanup
            gc.collect(2)  # Full collection

            # Clear caches
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Trim memory
            if hasattr(gc, 'freeze'):
                gc.freeze()
                gc.collect()
                gc.unfreeze()

        after = self._get_current_memory()
        freed = before['ram_used_gb'] - after['ram_used_gb']

        self.logger.info(f"Cleanup freed {freed:.2f}GB ({'aggressive' if aggressive else 'standard'})")
        return freed

    def emergency_cleanup(self):
        """Emergency cleanup when memory is critical"""
        self.logger.warning("EMERGENCY CLEANUP INITIATED")

        # Unload all models
        self.unload_all_models()

        # Aggressive garbage collection
        self.cleanup_memory(aggressive=True)

        # Clear any remaining caches
        if TORCH_AVAILABLE:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def start_monitoring(self, interval: int = 10):
        """
        Start background memory monitoring

        Args:
            interval: Check interval in seconds
        """
        if self.monitoring_active:
            return

        self.monitoring_active = True

        def monitor():
            while self.monitoring_active:
                current = self._get_current_memory()

                # Check thresholds
                if current['ram_percent'] > self._critical_threshold:
                    self.logger.critical(f"CRITICAL MEMORY: {current['ram_percent']:.1f}%")
                    self.emergency_cleanup()
                elif current['ram_percent'] > self._cleanup_threshold:
                    self.logger.warning(f"High memory usage: {current['ram_percent']:.1f}%")
                    self.cleanup_memory()

                # Log status
                self.logger.debug(f"Memory: {current['ram_used_gb']:.1f}/{self.total_ram / (1024 ** 3):.1f}GB "
                                  f"({current['ram_percent']:.1f}%)")

                time.sleep(interval)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop background memory monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        self.logger.info("Memory monitoring stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive memory status"""
        current = self._get_current_memory()

        status = {
            'timestamp': datetime.now().isoformat(),
            'memory': current,
            'loaded_models': list(self._loaded_models.keys()),
            'model_count': len(self._loaded_models),
            'peak_memory_gb': self.peak_memory.get('ram_used_gb', 0),
            'baseline_memory_gb': self.baseline_memory.get('ram_used_gb', 0),
            'can_load_small': self.can_load_model(2)[0],  # 2GB model
            'can_load_medium': self.can_load_model(5)[0],  # 5GB model
            'can_load_large': self.can_load_model(10)[0],  # 10GB model
            'monitoring_active': self.monitoring_active
        }

        # Health check
        if current['ram_percent'] > self._critical_threshold:
            status['health'] = 'CRITICAL'
        elif current['ram_percent'] > self._cleanup_threshold:
            status['health'] = 'WARNING'
        else:
            status['health'] = 'GOOD'

        return status

    def save_memory_report(self, filepath: Optional[Path] = None) -> Path:
        """Save detailed memory report"""
        if filepath is None:
            filepath = Path(f"memory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        report = {
            'generated': datetime.now().isoformat(),
            'system': {
                'total_ram_gb': self.total_ram / (1024 ** 3),
                'cpu_count': self.cpu_count,
                'gpu_available': self.gpu_available,
                'gpu_memory_gb': self.gpu_memory_total / (1024 ** 3) if self.gpu_available else 0
            },
            'current_status': self.get_status(),
            'loaded_models': self._loaded_models,
            'memory_history': self._memory_history[-100:],  # Last 100 entries
            'peak_usage': {
                'ram_gb': self.peak_memory.get('ram_used_gb', 0),
                'ram_percent': self.peak_memory.get('ram_percent', 0)
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Memory report saved to {filepath}")
        return filepath

    def _estimate_model_size(self, model_object: Any) -> float:
        """Estimate model size in GB"""
        try:
            # Try to get size from model object
            if hasattr(model_object, 'model_size'):
                return model_object.model_size / (1024 ** 3)

            # Estimate from current memory usage
            import sys
            size_bytes = sys.getsizeof(model_object)
            return size_bytes / (1024 ** 3)
        except:
            return 2.0  # Default estimate

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.stop_monitoring()
            self.unload_all_models()
        except:
            pass


# Global memory manager instance
memory_manager = MemoryManager()


# Convenience functions
def check_memory() -> Dict[str, Any]:
    """Quick memory check"""
    return memory_manager.get_status()


def cleanup() -> float:
    """Quick cleanup"""
    return memory_manager.cleanup_memory()


def can_load(size_gb: float) -> bool:
    """Check if model can be loaded"""
    return memory_manager.can_load_model(size_gb)[0]


# Context manager for automatic cleanup
class ManagedModel:
    """Context manager for automatic model cleanup"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model_name in memory_manager._loaded_models:
            memory_manager.unload_model(self.model_name)


# Testing
if __name__ == "__main__":
    print("Memory Manager Test")
    print("=" * 60)

    # Get status
    status = check_memory()
    print(f"\nSystem Status: {status['health']}")
    print(f"RAM Usage: {status['memory']['ram_used_gb']:.1f}/{status['memory']['ram_available_gb']:.1f} GB")
    print(f"RAM Percent: {status['memory']['ram_percent']:.1f}%")

    if status['memory'].get('gpu_available'):
        print(f"GPU Available: Yes")
        if 'gpu_used_gb' in status['memory']:
            print(f"GPU Usage: {status['memory']['gpu_used_gb']:.1f} GB")

    # Test cleanup
    print("\nTesting cleanup...")
    freed = cleanup()
    print(f"Freed: {freed:.2f} GB")

    # Test monitoring
    print("\nStarting monitoring...")
    memory_manager.start_monitoring(interval=5)
    time.sleep(10)
    memory_manager.stop_monitoring()

    # Save report
    report_path = memory_manager.save_memory_report()
    print(f"\nReport saved to: {report_path}")

    print("\n✅ Memory Manager test complete!")