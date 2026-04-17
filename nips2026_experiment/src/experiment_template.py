#!/usr/bin/env python3
"""
===============================================================================
M049: Experiment Templating System Module
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module provides a templating system for creating and managing experiments
with standardized configurations, parameter sweeps, and reproducibility.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Features:
- Experiment templates for common scenarios
- Parameter sweep generation
- Configuration inheritance
- Template validation
- Experiment registration and discovery
- Reproducibility utilities
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M049"

import os
import sys
import json
import copy
import itertools
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    List, Dict, Optional, Any, Callable, TypeVar, Union,
    Tuple, Set, Iterator, Type
)
from datetime import datetime
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
import yaml
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# PART 1: EXPERIMENT TYPES AND REGISTRY
# =============================================================================

class ExperimentType(Enum):
    """Types of experiments supported."""
    ABLATION = auto()          # Parameter ablation study
    SCALING = auto()           # Scaling experiments
    COMPARISON = auto()        # Method comparison
    HYPERPARAMETER = auto()    # Hyperparameter search
    CONVERGENCE = auto()       # Convergence analysis
    COMMUNICATION = auto()     # Communication efficiency
    WALLCLOCK = auto()         # Wall-clock timing


class ExperimentRegistry:
    """Registry for experiment templates."""
    
    _templates: Dict[str, 'ExperimentTemplate'] = {}
    _instances: Dict[str, 'ExperimentConfig'] = {}
    
    @classmethod
    def register_template(cls, name: str, template: 'ExperimentTemplate'):
        """Register an experiment template."""
        cls._templates[name] = template
        logger.debug(f"Registered template: {name}")
    
    @classmethod
    def get_template(cls, name: str) -> Optional['ExperimentTemplate']:
        """Get a registered template."""
        return cls._templates.get(name)
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """List all registered templates."""
        return list(cls._templates.keys())
    
    @classmethod
    def register_instance(cls, config: 'ExperimentConfig'):
        """Register an experiment instance."""
        cls._instances[config.experiment_id] = config
    
    @classmethod
    def get_instance(cls, experiment_id: str) -> Optional['ExperimentConfig']:
        """Get a registered experiment instance."""
        return cls._instances.get(experiment_id)


# =============================================================================
# PART 2: CONFIGURATION STRUCTURES
# =============================================================================

@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "transformer"
    num_layers: int = 12
    hidden_size: int = 768
    num_heads: int = 12
    vocab_size: int = 50257
    max_seq_length: int = 512
    dropout: float = 0.1
    
    # Architecture variants
    use_rope: bool = True
    use_swiglu: bool = False
    use_flash_attention: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = "desloc"
    base_optimizer: str = "adam"
    
    # Learning rate
    lr: float = 3e-4
    weight_decay: float = 0.1
    
    # Adam parameters
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # DES-LOC parameters
    Kx: int = 32
    Ku: int = 64
    Kv: int = 128
    px: float = 1.0
    pu: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Batch settings
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Training duration
    max_steps: int = 100000
    max_epochs: Optional[int] = None
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1
    
    # Mixed precision
    precision: str = "bf16"
    
    # Distributed
    world_size: int = 1
    
    # Checkpointing
    checkpoint_every_n_steps: int = 1000
    eval_every_n_steps: int = 500
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Identification
    experiment_id: str = ""
    experiment_name: str = ""
    experiment_type: ExperimentType = ExperimentType.ABLATION
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parent_template: str = ""
    
    # Output settings
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    
    # Seed for reproducibility
    seed: int = 42
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique experiment ID."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{self.experiment_name}_{timestamp}_{hash_val}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'experiment_type': self.experiment_type.name,
            'model': self.model.to_dict(),
            'optimizer': self.optimizer.to_dict(),
            'training': self.training.to_dict(),
            'description': self.description,
            'tags': self.tags,
            'parent_template': self.parent_template,
            'output_dir': self.output_dir,
            'log_dir': self.log_dir,
            'seed': self.seed,
            'custom_params': self.custom_params,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        config = cls()
        
        config.experiment_id = data.get('experiment_id', '')
        config.experiment_name = data.get('experiment_name', '')
        config.experiment_type = ExperimentType[data.get('experiment_type', 'ABLATION')]
        config.description = data.get('description', '')
        config.tags = data.get('tags', [])
        config.parent_template = data.get('parent_template', '')
        config.output_dir = data.get('output_dir', './outputs')
        config.log_dir = data.get('log_dir', './logs')
        config.seed = data.get('seed', 42)
        config.custom_params = data.get('custom_params', {})
        
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'optimizer' in data:
            config.optimizer = OptimizerConfig(**data['optimizer'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        
        return config
    
    def save(self, path: Path):
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


# =============================================================================
# PART 3: EXPERIMENT TEMPLATES
# =============================================================================

class ExperimentTemplate(ABC):
    """Base class for experiment templates."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.base_config = ExperimentConfig()
    
    @abstractmethod
    def create_config(self, **kwargs) -> ExperimentConfig:
        """Create experiment configuration from template."""
        pass
    
    def validate(self, config: ExperimentConfig) -> List[str]:
        """Validate configuration. Returns list of errors."""
        errors = []
        
        # Basic validation
        if config.training.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if config.optimizer.lr <= 0:
            errors.append("learning rate must be positive")
        
        if config.model.num_layers <= 0:
            errors.append("num_layers must be positive")
        
        return errors
    
    def register(self):
        """Register this template."""
        ExperimentRegistry.register_template(self.name, self)


class DESLOCBaseTemplate(ExperimentTemplate):
    """Base template for DES-LOC experiments."""
    
    def __init__(self):
        super().__init__(
            "desloc_base",
            "Base template for DES-LOC experiments"
        )
        
        # Set base configuration
        self.base_config.optimizer.name = "desloc"
        self.base_config.optimizer.base_optimizer = "adam"
        self.base_config.optimizer.Kx = 32
        self.base_config.optimizer.Ku = 64
        self.base_config.optimizer.Kv = 128
    
    def create_config(self, **kwargs) -> ExperimentConfig:
        config = copy.deepcopy(self.base_config)
        config.experiment_name = kwargs.get('name', 'desloc_experiment')
        config.parent_template = self.name
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.optimizer, key):
                setattr(config.optimizer, key, value)
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
        
        return config


class MomentumAblationTemplate(ExperimentTemplate):
    """Template for momentum ablation study (Figure 2)."""
    
    def __init__(self):
        super().__init__(
            "momentum_ablation",
            "Momentum coefficient ablation study"
        )
    
    def create_config(self, **kwargs) -> ExperimentConfig:
        config = ExperimentConfig()
        config.experiment_name = "momentum_ablation"
        config.experiment_type = ExperimentType.ABLATION
        config.parent_template = self.name
        
        # Default beta1 values for sweep
        beta1 = kwargs.get('beta1', 0.9)
        config.optimizer.beta1 = beta1
        
        # Use smaller model for ablation
        config.model.num_layers = 6
        config.model.hidden_size = 384
        
        config.tags = ['ablation', 'momentum', f'beta1_{beta1}']
        
        return config
    
    def generate_sweep(self) -> List[ExperimentConfig]:
        """Generate sweep over beta1 values."""
        beta1_values = [0.8, 0.9, 0.95, 0.99]
        configs = []
        
        for beta1 in beta1_values:
            config = self.create_config(beta1=beta1)
            config.experiment_name = f"momentum_ablation_beta1_{beta1}"
            configs.append(config)
        
        return configs


class SyncPeriodAblationTemplate(ExperimentTemplate):
    """Template for sync period ablation study (Figure 3)."""
    
    def __init__(self):
        super().__init__(
            "sync_period_ablation",
            "Synchronization period ablation study"
        )
    
    def create_config(self, **kwargs) -> ExperimentConfig:
        config = ExperimentConfig()
        config.experiment_name = "sync_period_ablation"
        config.experiment_type = ExperimentType.ABLATION
        config.parent_template = self.name
        
        # Sync period parameters
        config.optimizer.Kx = kwargs.get('Kx', 32)
        config.optimizer.Ku = kwargs.get('Ku', 64)
        config.optimizer.Kv = kwargs.get('Kv', 128)
        
        config.tags = ['ablation', 'sync_period']
        
        return config
    
    def generate_sweep(self, param: str = 'Kx') -> List[ExperimentConfig]:
        """Generate sweep over specified sync period."""
        if param == 'Kx':
            values = [16, 32, 64, 128]
        elif param == 'Ku':
            values = [32, 64, 128, 256]
        elif param == 'Kv':
            values = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unknown parameter: {param}")
        
        configs = []
        for val in values:
            kwargs = {param: val}
            config = self.create_config(**kwargs)
            config.experiment_name = f"sync_ablation_{param}_{val}"
            configs.append(config)
        
        return configs


class ScalingTemplate(ExperimentTemplate):
    """Template for scaling experiments (Figure 5)."""
    
    SCALE_PRESETS = {
        '125M': {'num_layers': 12, 'hidden_size': 768, 'num_heads': 12},
        '360M': {'num_layers': 24, 'hidden_size': 1024, 'num_heads': 16},
        '1.7B': {'num_layers': 24, 'hidden_size': 2048, 'num_heads': 32},
        '7B': {'num_layers': 32, 'hidden_size': 4096, 'num_heads': 32},
    }
    
    def __init__(self):
        super().__init__(
            "scaling",
            "Model scaling experiments"
        )
    
    def create_config(self, scale: str = '125M', **kwargs) -> ExperimentConfig:
        config = ExperimentConfig()
        config.experiment_name = f"scaling_{scale}"
        config.experiment_type = ExperimentType.SCALING
        config.parent_template = self.name
        
        # Apply scale preset
        if scale in self.SCALE_PRESETS:
            preset = self.SCALE_PRESETS[scale]
            config.model.num_layers = preset['num_layers']
            config.model.hidden_size = preset['hidden_size']
            config.model.num_heads = preset['num_heads']
        
        # Use RoPE and SwiGLU for larger models
        config.model.use_rope = True
        config.model.use_swiglu = scale in ['1.7B', '7B']
        
        config.tags = ['scaling', scale]
        
        return config
    
    def generate_sweep(self) -> List[ExperimentConfig]:
        """Generate sweep over model scales."""
        return [self.create_config(scale=s) for s in self.SCALE_PRESETS.keys()]


# =============================================================================
# PART 4: PARAMETER SWEEP GENERATOR
# =============================================================================

class ParameterSweepGenerator:
    """Generates parameter sweep configurations."""
    
    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config
        self.sweep_params: Dict[str, List[Any]] = {}
    
    def add_sweep(self, param_path: str, values: List[Any]):
        """Add parameter to sweep."""
        self.sweep_params[param_path] = values
    
    def _set_nested(self, config: ExperimentConfig, path: str, value: Any):
        """Set a nested parameter value."""
        parts = path.split('.')
        obj = config
        
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise ValueError(f"Invalid path: {path}")
        
        setattr(obj, parts[-1], value)
    
    def generate(self) -> List[ExperimentConfig]:
        """Generate all sweep configurations."""
        if not self.sweep_params:
            return [copy.deepcopy(self.base_config)]
        
        # Generate all combinations
        keys = list(self.sweep_params.keys())
        values = [self.sweep_params[k] for k in keys]
        
        configs = []
        for combo in itertools.product(*values):
            config = copy.deepcopy(self.base_config)
            
            for key, val in zip(keys, combo):
                self._set_nested(config, key, val)
            
            # Update experiment name
            suffix = '_'.join(f"{k.split('.')[-1]}_{v}" for k, v in zip(keys, combo))
            config.experiment_name = f"{self.base_config.experiment_name}_{suffix}"
            config.experiment_id = ""  # Will regenerate
            
            configs.append(config)
        
        return configs
    
    def count(self) -> int:
        """Count total configurations."""
        if not self.sweep_params:
            return 1
        
        total = 1
        for values in self.sweep_params.values():
            total *= len(values)
        return total


# =============================================================================
# PART 5: TEMPLATE MANAGER
# =============================================================================

class TemplateManager:
    """Manages experiment templates."""
    
    def __init__(self, template_dir: Path = None):
        self.template_dir = Path(template_dir) if template_dir else Path("./templates")
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Register built-in templates
        self._register_builtins()
    
    def _register_builtins(self):
        """Register built-in templates."""
        templates = [
            DESLOCBaseTemplate(),
            MomentumAblationTemplate(),
            SyncPeriodAblationTemplate(),
            ScalingTemplate(),
        ]
        
        for template in templates:
            template.register()
    
    def list_templates(self) -> List[Dict[str, str]]:
        """List all available templates."""
        templates = []
        
        for name in ExperimentRegistry.list_templates():
            template = ExperimentRegistry.get_template(name)
            templates.append({
                'name': name,
                'description': template.description,
            })
        
        return templates
    
    def create_from_template(
        self,
        template_name: str,
        **kwargs
    ) -> ExperimentConfig:
        """Create experiment config from template."""
        template = ExperimentRegistry.get_template(template_name)
        if template is None:
            raise ValueError(f"Unknown template: {template_name}")
        
        config = template.create_config(**kwargs)
        
        # Validate
        errors = template.validate(config)
        if errors:
            raise ValueError(f"Validation errors: {errors}")
        
        # Register
        ExperimentRegistry.register_instance(config)
        
        return config
    
    def save_template(self, config: ExperimentConfig, name: str):
        """Save configuration as reusable template."""
        path = self.template_dir / f"{name}.yaml"
        config.save(path)
        logger.info(f"Saved template: {path}")
    
    def load_template(self, name: str) -> ExperimentConfig:
        """Load template from file."""
        path = self.template_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        
        return ExperimentConfig.load(path)


# =============================================================================
# PART 6: EXPERIMENT PLAN
# =============================================================================

@dataclass
class ExperimentPlan:
    """A collection of experiments to run."""
    name: str
    description: str = ""
    experiments: List[ExperimentConfig] = field(default_factory=list)
    
    # Execution settings
    parallel: bool = False
    max_parallel: int = 4
    
    # Dependencies
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    def add(self, config: ExperimentConfig):
        """Add experiment to plan."""
        self.experiments.append(config)
    
    def add_dependency(self, exp_id: str, depends_on: List[str]):
        """Add dependency between experiments."""
        self.dependencies[exp_id] = depends_on
    
    def get_execution_order(self) -> List[List[str]]:
        """Get execution order respecting dependencies."""
        if not self.dependencies:
            if self.parallel:
                return [self.experiments]
            return [[exp] for exp in self.experiments]
        
        # Topological sort
        remaining = {exp.experiment_id for exp in self.experiments}
        completed: Set[str] = set()
        order: List[List[str]] = []
        
        while remaining:
            # Find experiments with all dependencies satisfied
            ready = []
            for exp_id in remaining:
                deps = self.dependencies.get(exp_id, [])
                if all(d in completed for d in deps):
                    ready.append(exp_id)
            
            if not ready:
                raise ValueError("Circular dependency detected")
            
            order.append(ready)
            for exp_id in ready:
                remaining.remove(exp_id)
                completed.add(exp_id)
        
        return order
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'experiments': [e.to_dict() for e in self.experiments],
            'parallel': self.parallel,
            'max_parallel': self.max_parallel,
            'dependencies': self.dependencies,
        }
    
    def save(self, path: Path):
        """Save plan to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# =============================================================================
# PART 7: MAIN / DEMO
# =============================================================================

def demo():
    """Demonstrate templating system."""
    print("=" * 70)
    print("DES-LOC Experiment Templating Demo")
    print("=" * 70)
    
    # Create template manager
    manager = TemplateManager()
    
    # List templates
    print("\nAvailable Templates:")
    for t in manager.list_templates():
        print(f"  - {t['name']}: {t['description']}")
    
    # Create config from template
    print("\nCreating config from 'desloc_base' template...")
    config = manager.create_from_template(
        'desloc_base',
        name='my_experiment',
        lr=1e-4,
        batch_size=64,
    )
    print(f"Created: {config.experiment_name} (ID: {config.experiment_id})")
    
    # Generate parameter sweep
    print("\nGenerating parameter sweep...")
    sweep = ParameterSweepGenerator(config)
    sweep.add_sweep('optimizer.lr', [1e-4, 3e-4, 1e-3])
    sweep.add_sweep('optimizer.Kx', [16, 32, 64])
    
    print(f"Total configurations: {sweep.count()}")
    configs = sweep.generate()
    for c in configs[:3]:
        print(f"  - {c.experiment_name}")
    print(f"  ... and {len(configs) - 3} more")
    
    # Create experiment plan
    print("\nCreating experiment plan...")
    plan = ExperimentPlan(
        name="desloc_ablation_study",
        description="Complete ablation study for NeurIPS 2026",
        parallel=True,
    )
    
    for c in configs:
        plan.add(c)
    
    print(f"Plan contains {len(plan.experiments)} experiments")
    
    # Save plan
    plan.save(Path("./demo_plan.yaml"))
    print("Saved plan to ./demo_plan.yaml")
    
    # Cleanup
    import os
    os.remove("./demo_plan.yaml")
    
    print("\n[M049] Experiment Templating Demo - COMPLETED")


if __name__ == "__main__":
    demo()
