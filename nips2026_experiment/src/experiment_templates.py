#!/usr/bin/env python3
"""
===============================================================================
M049: Experiment Templating Module
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module provides experiment templating capabilities for reproducible
and configurable experiments with parameter sweeps and inheritance.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Features:
- YAML/JSON experiment templates
- Template inheritance and composition
- Parameter sweeps and grid search
- Conditional configuration
- Environment variable substitution
- Experiment versioning
- Template validation
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M049"

import os
import sys
import json
import re
import copy
import itertools
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    List, Dict, Optional, Any, Callable, TypeVar, Union,
    Tuple, Iterator, Set
)
from datetime import datetime
from enum import Enum, auto
import logging

# Optional imports
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# PART 1: TEMPLATE DATA STRUCTURES
# =============================================================================

class SweepType(Enum):
    """Types of parameter sweeps."""
    GRID = auto()       # All combinations
    RANDOM = auto()     # Random sampling
    BAYESIAN = auto()   # Bayesian optimization (future)
    SEQUENTIAL = auto()  # One at a time


@dataclass
class ParameterSweep:
    """Definition of a parameter sweep."""
    name: str
    values: List[Any] = field(default_factory=list)
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    num_samples: int = 10
    log_scale: bool = False
    sweep_type: SweepType = SweepType.GRID
    
    def get_values(self) -> List[Any]:
        """Get the sweep values."""
        if self.values:
            return self.values
        
        if self.min_val is not None and self.max_val is not None:
            import numpy as np
            if self.log_scale:
                return list(np.logspace(
                    np.log10(self.min_val),
                    np.log10(self.max_val),
                    self.num_samples
                ))
            else:
                return list(np.linspace(
                    self.min_val,
                    self.max_val,
                    self.num_samples
                ))
        
        return []


@dataclass
class ExperimentTemplate:
    """Template for an experiment configuration."""
    name: str
    description: str = ""
    version: str = "1.0.0"
    
    # Base configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Inheritance
    extends: Optional[str] = None
    
    # Parameter sweeps
    sweeps: List[ParameterSweep] = field(default_factory=list)
    
    # Conditional overrides
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    author: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['sweeps'] = [
            {**asdict(s), 'sweep_type': s.sweep_type.name}
            for s in self.sweeps
        ]
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentTemplate':
        sweeps = []
        for s in data.pop('sweeps', []):
            if isinstance(s.get('sweep_type'), str):
                s['sweep_type'] = SweepType[s['sweep_type']]
            sweeps.append(ParameterSweep(**s))
        
        return cls(sweeps=sweeps, **data)


# =============================================================================
# PART 2: TEMPLATE REGISTRY
# =============================================================================

class TemplateRegistry:
    """Registry for experiment templates."""
    
    def __init__(self, template_dirs: List[str] = None):
        self.template_dirs = [Path(d) for d in (template_dirs or [])]
        self._templates: Dict[str, ExperimentTemplate] = {}
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """Load built-in DES-LOC templates."""
        # DES-LOC base template
        self.register(ExperimentTemplate(
            name="desloc_base",
            description="Base DES-LOC configuration",
            config={
                "optimizer": {
                    "type": "desloc",
                    "lr": 1e-4,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "weight_decay": 0.01,
                    "Kx": 32,
                    "Ku": 64,
                    "Kv": 128,
                    "px": 1.0,
                    "pu": 1.0,
                },
                "training": {
                    "max_steps": 100000,
                    "batch_size": 32,
                    "gradient_accumulation": 1,
                    "mixed_precision": "bf16",
                },
                "model": {
                    "type": "transformer",
                    "hidden_size": 768,
                    "num_layers": 12,
                    "num_heads": 12,
                },
            },
            tags=["desloc", "base"],
        ))
        
        # Small-scale ablation template
        self.register(ExperimentTemplate(
            name="desloc_ablation_small",
            description="Small-scale ablation experiments",
            extends="desloc_base",
            config={
                "model": {
                    "hidden_size": 256,
                    "num_layers": 4,
                    "num_heads": 4,
                },
                "training": {
                    "max_steps": 10000,
                    "batch_size": 64,
                },
            },
            tags=["desloc", "ablation", "small"],
        ))
        
        # Billion-scale template
        self.register(ExperimentTemplate(
            name="desloc_billion",
            description="Billion-parameter scale experiments",
            extends="desloc_base",
            config={
                "model": {
                    "type": "transformer",
                    "hidden_size": 2048,
                    "num_layers": 24,
                    "num_heads": 16,
                    "intermediate_size": 8192,
                },
                "training": {
                    "max_steps": 500000,
                    "batch_size": 8,
                    "gradient_accumulation": 8,
                    "gradient_checkpointing": True,
                },
                "distributed": {
                    "world_size": 8,
                    "tensor_parallel": 2,
                    "pipeline_parallel": 1,
                },
            },
            tags=["desloc", "billion-scale"],
        ))
        
        # Momentum ablation sweep
        self.register(ExperimentTemplate(
            name="desloc_momentum_sweep",
            description="Momentum (beta1) ablation sweep",
            extends="desloc_ablation_small",
            sweeps=[
                ParameterSweep(
                    name="optimizer.beta1",
                    values=[0.8, 0.85, 0.9, 0.95, 0.99],
                ),
            ],
            tags=["desloc", "ablation", "momentum"],
        ))
        
        # Sync period sweep
        self.register(ExperimentTemplate(
            name="desloc_sync_sweep",
            description="Sync period (Kx, Ku, Kv) ablation sweep",
            extends="desloc_ablation_small",
            sweeps=[
                ParameterSweep(
                    name="optimizer.Kx",
                    values=[8, 16, 32, 64, 128],
                ),
                ParameterSweep(
                    name="optimizer.Ku",
                    values=[32, 64, 128, 256],
                ),
            ],
            tags=["desloc", "ablation", "sync"],
        ))
    
    def register(self, template: ExperimentTemplate):
        """Register a template."""
        self._templates[template.name] = template
        logger.debug(f"Registered template: {template.name}")
    
    def get(self, name: str) -> Optional[ExperimentTemplate]:
        """Get a template by name."""
        return self._templates.get(name)
    
    def load_from_file(self, path: str) -> ExperimentTemplate:
        """Load template from file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            if path.suffix in ('.yaml', '.yml') and YAML_AVAILABLE:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        template = ExperimentTemplate.from_dict(data)
        self.register(template)
        return template
    
    def save_to_file(self, name: str, path: str):
        """Save template to file."""
        template = self.get(name)
        if not template:
            raise ValueError(f"Template not found: {name}")
        
        path = Path(path)
        
        with open(path, 'w') as f:
            if path.suffix in ('.yaml', '.yml') and YAML_AVAILABLE:
                yaml.dump(template.to_dict(), f, default_flow_style=False)
            else:
                json.dump(template.to_dict(), f, indent=2)
    
    def list_templates(self, tags: List[str] = None) -> List[str]:
        """List available templates."""
        templates = list(self._templates.keys())
        
        if tags:
            templates = [
                name for name in templates
                if any(t in self._templates[name].tags for t in tags)
            ]
        
        return sorted(templates)


# =============================================================================
# PART 3: TEMPLATE RESOLVER
# =============================================================================

class TemplateResolver:
    """Resolves templates with inheritance and variable substitution."""
    
    def __init__(self, registry: TemplateRegistry):
        self.registry = registry
        self._env_pattern = re.compile(r'\$\{(\w+)(?::([^}]*))?\}')
        self._ref_pattern = re.compile(r'\$\{config\.([^}]+)\}')
    
    def resolve(
        self,
        template_name: str,
        overrides: Dict[str, Any] = None,
        env_vars: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Resolve a template to a final configuration."""
        template = self.registry.get(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Resolve inheritance chain
        config = self._resolve_inheritance(template)
        
        # Apply overrides
        if overrides:
            config = self._deep_merge(config, overrides)
        
        # Substitute environment variables
        env_vars = env_vars or dict(os.environ)
        config = self._substitute_env_vars(config, env_vars)
        
        # Resolve internal references
        config = self._resolve_references(config, config)
        
        # Apply conditions
        config = self._apply_conditions(template.conditions, config)
        
        return config
    
    def _resolve_inheritance(self, template: ExperimentTemplate) -> Dict[str, Any]:
        """Resolve template inheritance."""
        if not template.extends:
            return copy.deepcopy(template.config)
        
        parent = self.registry.get(template.extends)
        if not parent:
            raise ValueError(f"Parent template not found: {template.extends}")
        
        # Recursively resolve parent
        parent_config = self._resolve_inheritance(parent)
        
        # Merge with current template
        return self._deep_merge(parent_config, template.config)
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _substitute_env_vars(self, config: Any, env_vars: Dict[str, str]) -> Any:
        """Substitute environment variables in config."""
        if isinstance(config, str):
            def replace(match):
                var_name = match.group(1)
                default = match.group(2)
                return env_vars.get(var_name, default or '')
            return self._env_pattern.sub(replace, config)
        elif isinstance(config, dict):
            return {k: self._substitute_env_vars(v, env_vars) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item, env_vars) for item in config]
        return config
    
    def _resolve_references(self, config: Any, root: Dict) -> Any:
        """Resolve internal config references."""
        if isinstance(config, str):
            def replace(match):
                ref_path = match.group(1)
                return str(self._get_nested(root, ref_path))
            return self._ref_pattern.sub(replace, config)
        elif isinstance(config, dict):
            return {k: self._resolve_references(v, root) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_references(item, root) for item in config]
        return config
    
    def _get_nested(self, obj: Dict, path: str) -> Any:
        """Get nested value by dot-separated path."""
        keys = path.split('.')
        for key in keys:
            if isinstance(obj, dict):
                obj = obj.get(key)
            else:
                return None
        return obj
    
    def _apply_conditions(
        self,
        conditions: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply conditional overrides."""
        for condition in conditions:
            if self._evaluate_condition(condition.get('when', {}), config):
                config = self._deep_merge(config, condition.get('then', {}))
        return config
    
    def _evaluate_condition(self, condition: Dict[str, Any], config: Dict) -> bool:
        """Evaluate a condition against config."""
        for key, expected in condition.items():
            actual = self._get_nested(config, key)
            if actual != expected:
                return False
        return True


# =============================================================================
# PART 4: SWEEP GENERATOR
# =============================================================================

class SweepGenerator:
    """Generates experiment configurations from sweeps."""
    
    def __init__(self, registry: TemplateRegistry):
        self.registry = registry
        self.resolver = TemplateResolver(registry)
    
    def generate(
        self,
        template_name: str,
        max_configs: int = None,
        random_seed: int = None,
    ) -> Iterator[Dict[str, Any]]:
        """Generate configurations from template sweeps."""
        template = self.registry.get(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Get base config
        base_config = self.resolver.resolve(template_name)
        
        if not template.sweeps:
            yield base_config
            return
        
        # Generate sweep combinations
        sweep_values = []
        sweep_names = []
        
        for sweep in template.sweeps:
            values = sweep.get_values()
            sweep_values.append(values)
            sweep_names.append(sweep.name)
        
        # Grid or random sampling
        if template.sweeps[0].sweep_type == SweepType.GRID:
            combinations = itertools.product(*sweep_values)
        else:
            import random
            if random_seed:
                random.seed(random_seed)
            
            all_combinations = list(itertools.product(*sweep_values))
            if max_configs and max_configs < len(all_combinations):
                combinations = random.sample(all_combinations, max_configs)
            else:
                combinations = all_combinations
        
        # Generate configs
        count = 0
        for combo in combinations:
            if max_configs and count >= max_configs:
                break
            
            config = copy.deepcopy(base_config)
            
            for name, value in zip(sweep_names, combo):
                self._set_nested(config, name, value)
            
            # Add sweep metadata
            config['_sweep'] = {
                'template': template_name,
                'parameters': dict(zip(sweep_names, combo)),
                'config_id': self._generate_config_id(sweep_names, combo),
            }
            
            yield config
            count += 1
    
    def _set_nested(self, obj: Dict, path: str, value: Any):
        """Set nested value by dot-separated path."""
        keys = path.split('.')
        for key in keys[:-1]:
            if key not in obj:
                obj[key] = {}
            obj = obj[key]
        obj[keys[-1]] = value
    
    def _generate_config_id(self, names: List[str], values: Tuple) -> str:
        """Generate unique ID for a configuration."""
        param_str = "_".join(
            f"{n.split('.')[-1]}={v}" for n, v in zip(names, values)
        )
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    def count_configurations(self, template_name: str) -> int:
        """Count total configurations in a sweep."""
        template = self.registry.get(template_name)
        if not template or not template.sweeps:
            return 1
        
        count = 1
        for sweep in template.sweeps:
            count *= len(sweep.get_values())
        return count


# =============================================================================
# PART 5: EXPERIMENT FACTORY
# =============================================================================

class ExperimentFactory:
    """Factory for creating experiments from templates."""
    
    def __init__(self, template_dirs: List[str] = None):
        self.registry = TemplateRegistry(template_dirs)
        self.resolver = TemplateResolver(self.registry)
        self.sweep_generator = SweepGenerator(self.registry)
    
    def create_experiment(
        self,
        template_name: str,
        overrides: Dict[str, Any] = None,
        experiment_name: str = None,
    ) -> Dict[str, Any]:
        """Create a single experiment configuration."""
        config = self.resolver.resolve(template_name, overrides)
        
        # Add experiment metadata
        config['_experiment'] = {
            'name': experiment_name or f"{template_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'template': template_name,
            'created_at': datetime.now().isoformat(),
            'version': self.registry.get(template_name).version,
        }
        
        return config
    
    def create_sweep(
        self,
        template_name: str,
        sweep_name: str = None,
        max_configs: int = None,
    ) -> List[Dict[str, Any]]:
        """Create a sweep of experiment configurations."""
        configs = list(self.sweep_generator.generate(template_name, max_configs))
        
        sweep_name = sweep_name or f"{template_name}_sweep"
        
        for i, config in enumerate(configs):
            config['_experiment'] = {
                'name': f"{sweep_name}_{i:04d}",
                'sweep_name': sweep_name,
                'sweep_index': i,
                'total_configs': len(configs),
                'template': template_name,
                'created_at': datetime.now().isoformat(),
            }
        
        return configs
    
    def register_template(self, template: ExperimentTemplate):
        """Register a new template."""
        self.registry.register(template)
    
    def load_template(self, path: str) -> ExperimentTemplate:
        """Load template from file."""
        return self.registry.load_from_file(path)
    
    def list_templates(self, tags: List[str] = None) -> List[str]:
        """List available templates."""
        return self.registry.list_templates(tags)
    
    def get_template_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get template information."""
        template = self.registry.get(name)
        if template:
            return {
                'name': template.name,
                'description': template.description,
                'version': template.version,
                'extends': template.extends,
                'tags': template.tags,
                'sweeps': len(template.sweeps),
                'total_configs': self.sweep_generator.count_configurations(name),
            }
        return None


# =============================================================================
# PART 6: MAIN / DEMO
# =============================================================================

def demo():
    """Demonstrate experiment templating capabilities."""
    print("=" * 70)
    print("DES-LOC Experiment Templating Demo")
    print("=" * 70)
    
    factory = ExperimentFactory()
    
    # List available templates
    print("\nAvailable Templates:")
    for name in factory.list_templates():
        info = factory.get_template_info(name)
        print(f"  - {name}: {info['description'][:50]}... ({info['total_configs']} configs)")
    
    # Create single experiment
    print("\n--- Single Experiment ---")
    config = factory.create_experiment(
        "desloc_base",
        overrides={"training.max_steps": 50000},
    )
    print(f"Experiment: {config['_experiment']['name']}")
    print(f"Config keys: {list(config.keys())}")
    
    # Create sweep
    print("\n--- Momentum Sweep ---")
    configs = factory.create_sweep(
        "desloc_momentum_sweep",
        sweep_name="beta1_ablation",
    )
    print(f"Generated {len(configs)} configurations:")
    for cfg in configs:
        beta1 = cfg['optimizer']['beta1']
        print(f"  - {cfg['_experiment']['name']}: beta1={beta1}")
    
    # Create sync period sweep
    print("\n--- Sync Period Sweep ---")
    sync_configs = factory.create_sweep(
        "desloc_sync_sweep",
        max_configs=5,  # Limit for demo
    )
    print(f"Generated {len(sync_configs)} configurations (limited to 5):")
    for cfg in sync_configs:
        Kx = cfg['optimizer']['Kx']
        Ku = cfg['optimizer']['Ku']
        print(f"  - {cfg['_sweep']['config_id']}: Kx={Kx}, Ku={Ku}")
    
    print("\n[M049] Experiment Templating Demo - COMPLETED")


if __name__ == "__main__":
    demo()
