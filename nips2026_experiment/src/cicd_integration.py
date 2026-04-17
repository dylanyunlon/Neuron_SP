#!/usr/bin/env python3
"""
===============================================================================
M050: CI/CD Integration Module
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module provides CI/CD integration including test runners, GitHub Actions
workflows, Docker configurations, and automated deployment utilities.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Features:
- Test runner with coverage reporting
- GitHub Actions workflow generation
- Docker image building
- Automated experiment deployment
- Result validation and reporting
- Continuous benchmarking
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M050"

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    List, Dict, Optional, Any, Callable, TypeVar, Union,
    Tuple
)
from datetime import datetime
from enum import Enum, auto
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: TEST CONFIGURATION
# =============================================================================

class TestType(Enum):
    """Types of tests."""
    UNIT = auto()
    INTEGRATION = auto()
    BENCHMARK = auto()
    SMOKE = auto()
    REGRESSION = auto()


@dataclass
class TestConfig:
    """Configuration for test execution."""
    test_type: TestType = TestType.UNIT
    
    # Paths
    test_dir: str = "tests"
    source_dir: str = "src"
    
    # Execution
    parallel: bool = True
    num_workers: int = 4
    timeout_seconds: int = 300
    fail_fast: bool = False
    
    # Coverage
    coverage_enabled: bool = True
    coverage_threshold: float = 80.0
    coverage_report_format: str = "html"
    
    # Reporting
    output_dir: str = "test_results"
    junit_xml: bool = True
    
    # GPU tests
    require_gpu: bool = False
    gpu_memory_limit_gb: float = 8.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'test_type': self.test_type.name,
        }


# =============================================================================
# PART 2: TEST RUNNER
# =============================================================================

class TestRunner:
    """Runs tests with coverage and reporting."""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.results: List[Dict[str, Any]] = []
    
    def run(self, test_pattern: str = None) -> Dict[str, Any]:
        """Run tests and return results."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        # Add test directory
        cmd.append(self.config.test_dir)
        
        # Pattern filtering
        if test_pattern:
            cmd.extend(["-k", test_pattern])
        
        # Parallelization
        if self.config.parallel:
            cmd.extend(["-n", str(self.config.num_workers)])
        
        # Timeout
        cmd.extend(["--timeout", str(self.config.timeout_seconds)])
        
        # Fail fast
        if self.config.fail_fast:
            cmd.append("-x")
        
        # Coverage
        if self.config.coverage_enabled:
            cmd.extend([
                f"--cov={self.config.source_dir}",
                f"--cov-report={self.config.coverage_report_format}",
                f"--cov-fail-under={self.config.coverage_threshold}",
            ])
        
        # JUnit XML
        if self.config.junit_xml:
            cmd.extend([f"--junitxml={output_dir}/junit.xml"])
        
        # Verbose output
        cmd.append("-v")
        
        # Run tests
        logger.info(f"Running: {' '.join(cmd)}")
        
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds * 2,
            )
            
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr
            
        except subprocess.TimeoutExpired:
            success = False
            stdout = ""
            stderr = "Test execution timed out"
        except Exception as e:
            success = False
            stdout = ""
            stderr = str(e)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Parse results
        test_result = {
            'success': success,
            'duration_seconds': duration,
            'stdout': stdout,
            'stderr': stderr,
            'test_type': self.config.test_type.name,
            'timestamp': end_time.isoformat(),
        }
        
        # Parse coverage
        if self.config.coverage_enabled and success:
            test_result['coverage'] = self._parse_coverage(stdout)
        
        self.results.append(test_result)
        
        # Save results
        self._save_results(output_dir)
        
        return test_result
    
    def _parse_coverage(self, stdout: str) -> Dict[str, float]:
        """Parse coverage from pytest output."""
        # Simple parsing - in practice would use coverage.py API
        coverage = {
            'total': 0.0,
            'by_file': {},
        }
        
        # Look for coverage lines
        for line in stdout.split('\n'):
            if 'TOTAL' in line and '%' in line:
                try:
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            coverage['total'] = float(part.strip('%'))
                            break
                except:
                    pass
        
        return coverage
    
    def _save_results(self, output_dir: Path):
        """Save test results to file."""
        results_file = output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def run_benchmark_tests(self) -> Dict[str, Any]:
        """Run benchmark-specific tests."""
        self.config.test_type = TestType.BENCHMARK
        return self.run(test_pattern="benchmark")
    
    def run_smoke_tests(self) -> Dict[str, Any]:
        """Run smoke tests (quick sanity checks)."""
        self.config.test_type = TestType.SMOKE
        self.config.timeout_seconds = 60
        return self.run(test_pattern="smoke")


# =============================================================================
# PART 3: GITHUB ACTIONS WORKFLOW GENERATOR
# =============================================================================

class GitHubActionsGenerator:
    """Generates GitHub Actions workflow files."""
    
    def __init__(self, project_name: str = "desloc-benchmark"):
        self.project_name = project_name
    
    def generate_ci_workflow(self) -> str:
        """Generate CI workflow for testing."""
        workflow = {
            'name': 'CI',
            'on': {
                'push': {'branches': ['main', 'develop']},
                'pull_request': {'branches': ['main']},
            },
            'jobs': {
                'test': {
                    'runs-on': 'ubuntu-latest',
                    'strategy': {
                        'matrix': {
                            'python-version': ['3.9', '3.10', '3.11'],
                        },
                    },
                    'steps': [
                        {'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Set up Python ${{ matrix.python-version }}',
                            'uses': 'actions/setup-python@v5',
                            'with': {'python-version': '${{ matrix.python-version }}'},
                        },
                        {
                            'name': 'Install dependencies',
                            'run': '\n'.join([
                                'python -m pip install --upgrade pip',
                                'pip install -r requirements.txt',
                                'pip install -r requirements-dev.txt',
                            ]),
                        },
                        {
                            'name': 'Lint with flake8',
                            'run': 'flake8 src tests --max-line-length=100',
                        },
                        {
                            'name': 'Type check with mypy',
                            'run': 'mypy src --ignore-missing-imports',
                        },
                        {
                            'name': 'Run tests',
                            'run': 'pytest tests -v --cov=src --cov-report=xml',
                        },
                        {
                            'name': 'Upload coverage',
                            'uses': 'codecov/codecov-action@v3',
                            'with': {'files': 'coverage.xml'},
                        },
                    ],
                },
            },
        }
        
        return self._to_yaml(workflow)
    
    def generate_gpu_benchmark_workflow(self) -> str:
        """Generate workflow for GPU benchmarks."""
        workflow = {
            'name': 'GPU Benchmarks',
            'on': {
                'schedule': [{'cron': '0 0 * * 0'}],  # Weekly
                'workflow_dispatch': {},
            },
            'jobs': {
                'benchmark': {
                    'runs-on': 'self-hosted',
                    'strategy': {
                        'matrix': {
                            'model-size': ['125M', '360M', '1.7B'],
                        },
                    },
                    'steps': [
                        {'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Setup CUDA',
                            'run': '\n'.join([
                                'nvidia-smi',
                                'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"',
                            ]),
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'pip install -r requirements.txt',
                        },
                        {
                            'name': 'Run benchmark',
                            'run': '\n'.join([
                                'python -m src.experiments.run_billion_scale \\',
                                '  --model-size ${{ matrix.model-size }} \\',
                                '  --output-dir benchmark_results',
                            ]),
                        },
                        {
                            'name': 'Upload results',
                            'uses': 'actions/upload-artifact@v4',
                            'with': {
                                'name': 'benchmark-${{ matrix.model-size }}',
                                'path': 'benchmark_results/',
                            },
                        },
                    ],
                },
            },
        }
        
        return self._to_yaml(workflow)
    
    def generate_release_workflow(self) -> str:
        """Generate workflow for releases."""
        workflow = {
            'name': 'Release',
            'on': {
                'push': {'tags': ['v*']},
            },
            'jobs': {
                'release': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v5',
                            'with': {'python-version': '3.10'},
                        },
                        {
                            'name': 'Install build tools',
                            'run': 'pip install build twine',
                        },
                        {
                            'name': 'Build package',
                            'run': 'python -m build',
                        },
                        {
                            'name': 'Publish to PyPI',
                            'uses': 'pypa/gh-action-pypi-publish@release/v1',
                            'with': {'password': '${{ secrets.PYPI_API_TOKEN }}'},
                        },
                        {
                            'name': 'Create GitHub Release',
                            'uses': 'softprops/action-gh-release@v1',
                            'with': {
                                'files': 'dist/*',
                                'generate_release_notes': True,
                            },
                        },
                    ],
                },
            },
        }
        
        return self._to_yaml(workflow)
    
    def _to_yaml(self, data: Dict) -> str:
        """Convert dict to YAML string."""
        try:
            import yaml
            return yaml.dump(data, default_flow_style=False, sort_keys=False)
        except ImportError:
            # Fallback to manual formatting
            return self._manual_yaml(data, indent=0)
    
    def _manual_yaml(self, data: Any, indent: int = 0) -> str:
        """Manual YAML serialization."""
        lines = []
        prefix = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._manual_yaml(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {self._format_value(value)}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    first = True
                    for key, value in item.items():
                        marker = "- " if first else "  "
                        first = False
                        if isinstance(value, (dict, list)):
                            lines.append(f"{prefix}{marker}{key}:")
                            lines.append(self._manual_yaml(value, indent + 2))
                        else:
                            lines.append(f"{prefix}{marker}{key}: {self._format_value(value)}")
                else:
                    lines.append(f"{prefix}- {self._format_value(item)}")
        
        return '\n'.join(lines)
    
    def _format_value(self, value: Any) -> str:
        """Format a value for YAML."""
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, str) and ('\n' in value or ':' in value or '$' in value):
            return f"'{value}'"
        return str(value)
    
    def save_workflows(self, output_dir: str):
        """Save all workflows to directory."""
        output_path = Path(output_dir) / ".github" / "workflows"
        output_path.mkdir(parents=True, exist_ok=True)
        
        workflows = {
            'ci.yml': self.generate_ci_workflow(),
            'benchmark.yml': self.generate_gpu_benchmark_workflow(),
            'release.yml': self.generate_release_workflow(),
        }
        
        for filename, content in workflows.items():
            filepath = output_path / filename
            with open(filepath, 'w') as f:
                f.write(content)
            logger.info(f"Generated: {filepath}")


# =============================================================================
# PART 4: DOCKER CONFIGURATION
# =============================================================================

class DockerConfigGenerator:
    """Generates Docker configurations."""
    
    def __init__(
        self,
        base_image: str = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel",
        project_name: str = "desloc-benchmark",
    ):
        self.base_image = base_image
        self.project_name = project_name
    
    def generate_dockerfile(self) -> str:
        """Generate Dockerfile."""
        return f'''# DES-LOC Benchmark Framework
# NeurIPS 2026 Submission
FROM {self.base_image}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    git \\
    wget \\
    curl \\
    vim \\
    htop \\
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Copy requirements first for caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \\
    pip install -r requirements.txt && \\
    pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Set entrypoint
ENTRYPOINT ["python", "-m"]
CMD ["src.experiments.run_all"]
'''
    
    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml."""
        return f'''version: '3.8'

services:
  {self.project_name}:
    build:
      context: .
      dockerfile: Dockerfile
    image: {self.project_name}:latest
    container_name: {self.project_name}
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - WANDB_API_KEY=${{WANDB_API_KEY}}
    volumes:
      - ./data:/workspace/data
      - ./outputs:/workspace/outputs
      - ./checkpoints:/workspace/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: src.experiments.run_all

  tensorboard:
    image: tensorflow/tensorflow:latest
    container_name: {self.project_name}-tensorboard
    ports:
      - "6006:6006"
    volumes:
      - ./outputs/logs:/logs
    command: tensorboard --logdir=/logs --bind_all

  jupyter:
    build:
      context: .
      dockerfile: Dockerfile
    image: {self.project_name}:latest
    container_name: {self.project_name}-jupyter
    runtime: nvidia
    ports:
      - "8888:8888"
    volumes:
      - ./:/workspace
    command: jupyter lab --ip=0.0.0.0 --allow-root --no-browser
'''
    
    def generate_dockerignore(self) -> str:
        """Generate .dockerignore."""
        return '''# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.egg-info
.eggs
dist
build
*.egg

# Virtual environments
venv
.venv
env

# IDE
.idea
.vscode
*.swp
*.swo

# Data and outputs (mounted as volumes)
data/
outputs/
checkpoints/
*.pt
*.pth

# Logs
*.log
logs/

# Notebooks
.ipynb_checkpoints

# Testing
.pytest_cache
.coverage
htmlcov
'''
    
    def save_docker_configs(self, output_dir: str):
        """Save Docker configurations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        configs = {
            'Dockerfile': self.generate_dockerfile(),
            'docker-compose.yml': self.generate_docker_compose(),
            '.dockerignore': self.generate_dockerignore(),
        }
        
        for filename, content in configs.items():
            filepath = output_path / filename
            with open(filepath, 'w') as f:
                f.write(content)
            logger.info(f"Generated: {filepath}")


# =============================================================================
# PART 5: DEPLOYMENT AUTOMATION
# =============================================================================

class DeploymentAutomation:
    """Automates deployment of experiments."""
    
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate deployment environment."""
        checks = {
            'python_version': self._check_python(),
            'cuda_available': self._check_cuda(),
            'dependencies': self._check_dependencies(),
            'gpu_memory': self._check_gpu_memory(),
            'disk_space': self._check_disk_space(),
        }
        
        all_passed = all(c.get('passed', False) for c in checks.values())
        
        return {
            'passed': all_passed,
            'checks': checks,
            'timestamp': datetime.now().isoformat(),
        }
    
    def _check_python(self) -> Dict[str, Any]:
        """Check Python version."""
        version = sys.version_info
        required = (3, 9)
        passed = version >= required
        
        return {
            'passed': passed,
            'current': f"{version.major}.{version.minor}.{version.micro}",
            'required': f"{required[0]}.{required[1]}+",
        }
    
    def _check_cuda(self) -> Dict[str, Any]:
        """Check CUDA availability."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_count = torch.cuda.device_count()
                devices = [torch.cuda.get_device_name(i) for i in range(device_count)]
            else:
                device_count = 0
                devices = []
            
            return {
                'passed': cuda_available,
                'device_count': device_count,
                'devices': devices,
            }
        except ImportError:
            return {'passed': False, 'error': 'PyTorch not installed'}
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies."""
        required = ['torch', 'numpy', 'pandas', 'matplotlib']
        missing = []
        
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        
        return {
            'passed': len(missing) == 0,
            'missing': missing,
        }
    
    def _check_gpu_memory(self) -> Dict[str, Any]:
        """Check GPU memory."""
        try:
            import torch
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return {
                    'passed': total_memory >= 24,  # Minimum 24GB for benchmarks
                    'total_gb': total_memory,
                    'required_gb': 24,
                }
            return {'passed': False, 'error': 'No GPU available'}
        except:
            return {'passed': False, 'error': 'Could not check GPU memory'}
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        import shutil
        total, used, free = shutil.disk_usage(self.project_dir)
        free_gb = free / (1024**3)
        
        return {
            'passed': free_gb >= 50,  # Need at least 50GB
            'free_gb': free_gb,
            'required_gb': 50,
        }
    
    def deploy_experiment(
        self,
        config: Dict[str, Any],
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Deploy an experiment."""
        # Validate environment
        validation = self.validate_environment()
        if not validation['passed']:
            return {
                'success': False,
                'error': 'Environment validation failed',
                'validation': validation,
            }
        
        # Create experiment directory
        exp_name = config.get('_experiment', {}).get('name', 'unnamed')
        exp_dir = self.project_dir / 'experiments' / exp_name
        
        if not dry_run:
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save config
            config_path = exp_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        return {
            'success': True,
            'experiment_dir': str(exp_dir),
            'config_path': str(exp_dir / 'config.json') if not dry_run else None,
            'dry_run': dry_run,
        }


# =============================================================================
# PART 6: MAIN / DEMO
# =============================================================================

def demo():
    """Demonstrate CI/CD integration capabilities."""
    print("=" * 70)
    print("DES-LOC CI/CD Integration Demo")
    print("=" * 70)
    
    # Test runner demo
    print("\n--- Test Runner ---")
    runner = TestRunner()
    print(f"Test config: {runner.config.test_type.name}")
    print(f"Coverage threshold: {runner.config.coverage_threshold}%")
    
    # GitHub Actions demo
    print("\n--- GitHub Actions Generator ---")
    gh_gen = GitHubActionsGenerator()
    ci_workflow = gh_gen.generate_ci_workflow()
    print(f"CI workflow preview (first 500 chars):\n{ci_workflow[:500]}...")
    
    # Docker demo
    print("\n--- Docker Configuration ---")
    docker_gen = DockerConfigGenerator()
    dockerfile = docker_gen.generate_dockerfile()
    print(f"Dockerfile preview (first 500 chars):\n{dockerfile[:500]}...")
    
    # Deployment validation demo
    print("\n--- Deployment Validation ---")
    automation = DeploymentAutomation(".")
    validation = automation.validate_environment()
    print(f"Environment validation: {'PASSED' if validation['passed'] else 'FAILED'}")
    for name, check in validation['checks'].items():
        status = '✓' if check.get('passed', False) else '✗'
        print(f"  {status} {name}: {check}")
    
    print("\n[M050] CI/CD Integration Demo - COMPLETED")


if __name__ == "__main__":
    demo()
