#!/usr/bin/env python3
"""
===============================================================================
M050: CI/CD Integration Module
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module provides CI/CD integration for automated testing, benchmarking,
and deployment of DES-LOC experiments.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Features:
- GitHub Actions workflow generation
- Automated test suite
- Benchmark regression testing
- Docker container management
- Artifact publishing
- Slack/email notifications
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M050"

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    List, Dict, Optional, Any, Callable, TypeVar, Union,
    Tuple, Set
)
from datetime import datetime
from enum import Enum, auto
import logging
import yaml
import hashlib
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: CI/CD CONFIGURATION
# =============================================================================

class CIPlatform(Enum):
    """Supported CI platforms."""
    GITHUB_ACTIONS = auto()
    GITLAB_CI = auto()
    JENKINS = auto()
    CIRCLECI = auto()


class TestType(Enum):
    """Types of tests."""
    UNIT = auto()
    INTEGRATION = auto()
    BENCHMARK = auto()
    REGRESSION = auto()
    SMOKE = auto()


@dataclass
class CIConfig:
    """CI/CD configuration."""
    # Platform
    platform: CIPlatform = CIPlatform.GITHUB_ACTIONS
    
    # Test settings
    run_unit_tests: bool = True
    run_integration_tests: bool = True
    run_benchmarks: bool = True
    run_regression_tests: bool = True
    
    # Python versions
    python_versions: List[str] = field(default_factory=lambda: ['3.10', '3.11'])
    
    # GPU testing
    gpu_enabled: bool = True
    gpu_types: List[str] = field(default_factory=lambda: ['a100', 'h100'])
    
    # Docker
    use_docker: bool = True
    docker_registry: str = "ghcr.io"
    docker_image_name: str = "desloc-benchmark"
    
    # Notifications
    notify_slack: bool = False
    slack_webhook: str = ""
    notify_email: bool = False
    email_recipients: List[str] = field(default_factory=list)
    
    # Artifacts
    publish_artifacts: bool = True
    artifact_retention_days: int = 30
    
    # Caching
    cache_pip: bool = True
    cache_models: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'platform': self.platform.name,
        }


@dataclass
class BenchmarkThreshold:
    """Thresholds for benchmark regression testing."""
    metric_name: str
    warning_threshold: float  # Percent increase to warn
    failure_threshold: float  # Percent increase to fail
    
    def check(self, baseline: float, current: float) -> Tuple[bool, str]:
        """Check if current value exceeds thresholds."""
        if baseline == 0:
            return True, "OK"
        
        change_pct = ((current - baseline) / baseline) * 100
        
        if change_pct > self.failure_threshold:
            return False, f"FAIL: {change_pct:.1f}% increase (threshold: {self.failure_threshold}%)"
        elif change_pct > self.warning_threshold:
            return True, f"WARN: {change_pct:.1f}% increase"
        else:
            return True, "OK"


# =============================================================================
# PART 2: GITHUB ACTIONS WORKFLOW GENERATOR
# =============================================================================

class GitHubActionsGenerator:
    """Generates GitHub Actions workflow files."""
    
    def __init__(self, config: CIConfig):
        self.config = config
    
    def generate_main_workflow(self) -> Dict[str, Any]:
        """Generate main CI workflow."""
        workflow = {
            'name': 'DES-LOC CI',
            'on': {
                'push': {'branches': ['main', 'develop']},
                'pull_request': {'branches': ['main']},
            },
            'env': {
                'PYTHON_VERSION': self.config.python_versions[0],
            },
            'jobs': {},
        }
        
        # Lint job
        workflow['jobs']['lint'] = self._generate_lint_job()
        
        # Unit test job
        if self.config.run_unit_tests:
            workflow['jobs']['unit-tests'] = self._generate_test_job(TestType.UNIT)
        
        # Integration test job
        if self.config.run_integration_tests:
            workflow['jobs']['integration-tests'] = self._generate_test_job(
                TestType.INTEGRATION, needs=['unit-tests']
            )
        
        # Benchmark job
        if self.config.run_benchmarks:
            workflow['jobs']['benchmarks'] = self._generate_benchmark_job()
        
        # Docker build job
        if self.config.use_docker:
            workflow['jobs']['docker'] = self._generate_docker_job()
        
        return workflow
    
    def _generate_lint_job(self) -> Dict[str, Any]:
        """Generate lint job."""
        return {
            'runs-on': 'ubuntu-latest',
            'steps': [
                {'uses': 'actions/checkout@v4'},
                {
                    'name': 'Set up Python',
                    'uses': 'actions/setup-python@v5',
                    'with': {'python-version': '${{ env.PYTHON_VERSION }}'},
                },
                {
                    'name': 'Install linters',
                    'run': 'pip install ruff mypy',
                },
                {
                    'name': 'Run ruff',
                    'run': 'ruff check src/',
                },
                {
                    'name': 'Run mypy',
                    'run': 'mypy src/ --ignore-missing-imports',
                },
            ],
        }
    
    def _generate_test_job(
        self,
        test_type: TestType,
        needs: List[str] = None,
    ) -> Dict[str, Any]:
        """Generate test job."""
        job = {
            'runs-on': 'ubuntu-latest',
            'strategy': {
                'matrix': {
                    'python-version': self.config.python_versions,
                },
            },
            'steps': [
                {'uses': 'actions/checkout@v4'},
                {
                    'name': 'Set up Python',
                    'uses': 'actions/setup-python@v5',
                    'with': {'python-version': '${{ matrix.python-version }}'},
                },
            ],
        }
        
        if needs:
            job['needs'] = needs
        
        # Cache pip
        if self.config.cache_pip:
            job['steps'].append({
                'name': 'Cache pip',
                'uses': 'actions/cache@v4',
                'with': {
                    'path': '~/.cache/pip',
                    'key': "${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}",
                },
            })
        
        # Install dependencies
        job['steps'].append({
            'name': 'Install dependencies',
            'run': 'pip install -e ".[dev]"',
        })
        
        # Run tests
        if test_type == TestType.UNIT:
            test_cmd = 'pytest tests/unit -v --cov=src --cov-report=xml'
        elif test_type == TestType.INTEGRATION:
            test_cmd = 'pytest tests/integration -v --timeout=300'
        else:
            test_cmd = 'pytest tests/ -v'
        
        job['steps'].append({
            'name': f'Run {test_type.name.lower()} tests',
            'run': test_cmd,
        })
        
        # Upload coverage
        if test_type == TestType.UNIT:
            job['steps'].append({
                'name': 'Upload coverage',
                'uses': 'codecov/codecov-action@v4',
                'with': {'files': 'coverage.xml'},
            })
        
        return job
    
    def _generate_benchmark_job(self) -> Dict[str, Any]:
        """Generate benchmark job."""
        job = {
            'runs-on': 'self-hosted',
            'needs': ['unit-tests'],
            'steps': [
                {'uses': 'actions/checkout@v4'},
                {
                    'name': 'Set up Python',
                    'uses': 'actions/setup-python@v5',
                    'with': {'python-version': '${{ env.PYTHON_VERSION }}'},
                },
                {
                    'name': 'Install dependencies',
                    'run': 'pip install -e ".[benchmark]"',
                },
                {
                    'name': 'Run benchmarks',
                    'run': 'python -m pytest tests/benchmarks --benchmark-json=benchmark.json',
                },
                {
                    'name': 'Store benchmark result',
                    'uses': 'benchmark-action/github-action-benchmark@v1',
                    'with': {
                        'tool': 'pytest',
                        'output-file-path': 'benchmark.json',
                        'github-token': '${{ secrets.GITHUB_TOKEN }}',
                        'auto-push': 'true',
                    },
                },
            ],
        }
        
        if self.config.gpu_enabled:
            job['runs-on'] = [{'self-hosted': True, 'gpu': self.config.gpu_types[0]}]
        
        return job
    
    def _generate_docker_job(self) -> Dict[str, Any]:
        """Generate Docker build job."""
        return {
            'runs-on': 'ubuntu-latest',
            'needs': ['unit-tests'],
            'steps': [
                {'uses': 'actions/checkout@v4'},
                {
                    'name': 'Set up Docker Buildx',
                    'uses': 'docker/setup-buildx-action@v3',
                },
                {
                    'name': 'Login to Container Registry',
                    'uses': 'docker/login-action@v3',
                    'with': {
                        'registry': self.config.docker_registry,
                        'username': '${{ github.actor }}',
                        'password': '${{ secrets.GITHUB_TOKEN }}',
                    },
                },
                {
                    'name': 'Build and push',
                    'uses': 'docker/build-push-action@v5',
                    'with': {
                        'context': '.',
                        'push': '${{ github.event_name != \'pull_request\' }}',
                        'tags': f'{self.config.docker_registry}/${{{{ github.repository }}}}/{self.config.docker_image_name}:${{{{ github.sha }}}}',
                        'cache-from': 'type=gha',
                        'cache-to': 'type=gha,mode=max',
                    },
                },
            ],
        }
    
    def save(self, output_dir: Path):
        """Save workflow files."""
        output_dir = Path(output_dir)
        workflows_dir = output_dir / '.github' / 'workflows'
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        # Main CI workflow
        main_workflow = self.generate_main_workflow()
        with open(workflows_dir / 'ci.yml', 'w') as f:
            yaml.dump(main_workflow, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Generated GitHub Actions workflows in {workflows_dir}")


# =============================================================================
# PART 3: TEST RUNNER
# =============================================================================

class TestRunner:
    """Runs tests and collects results."""
    
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.results: Dict[TestType, Dict[str, Any]] = {}
    
    def run(
        self,
        test_type: TestType,
        extra_args: List[str] = None,
    ) -> Dict[str, Any]:
        """Run tests of specified type."""
        extra_args = extra_args or []
        
        # Determine test directory
        if test_type == TestType.UNIT:
            test_dir = self.project_dir / 'tests' / 'unit'
        elif test_type == TestType.INTEGRATION:
            test_dir = self.project_dir / 'tests' / 'integration'
        elif test_type == TestType.BENCHMARK:
            test_dir = self.project_dir / 'tests' / 'benchmarks'
        else:
            test_dir = self.project_dir / 'tests'
        
        # Build command
        cmd = [
            sys.executable, '-m', 'pytest',
            str(test_dir),
            '-v',
            '--tb=short',
            f'--junitxml=test-results-{test_type.name.lower()}.xml',
        ]
        cmd.extend(extra_args)
        
        logger.info(f"Running {test_type.name} tests: {' '.join(cmd)}")
        
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            test_result = {
                'test_type': test_type.name,
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'duration_seconds': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'timestamp': start_time.isoformat(),
            }
            
        except Exception as e:
            test_result = {
                'test_type': test_type.name,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }
        
        self.results[test_type] = test_result
        return test_result
    
    def run_all(self, test_types: List[TestType] = None) -> Dict[str, Any]:
        """Run all specified test types."""
        if test_types is None:
            test_types = [TestType.UNIT, TestType.INTEGRATION]
        
        all_results = {}
        overall_success = True
        
        for test_type in test_types:
            result = self.run(test_type)
            all_results[test_type.name] = result
            
            if not result.get('success', False):
                overall_success = False
        
        return {
            'overall_success': overall_success,
            'results': all_results,
            'timestamp': datetime.now().isoformat(),
        }


# =============================================================================
# PART 4: BENCHMARK REGRESSION TESTING
# =============================================================================

class BenchmarkRegressionTester:
    """Tests for benchmark regressions."""
    
    def __init__(self, baseline_path: Path, thresholds: List[BenchmarkThreshold] = None):
        self.baseline_path = Path(baseline_path)
        self.thresholds = thresholds or self._default_thresholds()
        self.baseline: Dict[str, float] = {}
        self.current: Dict[str, float] = {}
        
        self._load_baseline()
    
    def _default_thresholds(self) -> List[BenchmarkThreshold]:
        """Create default thresholds."""
        return [
            BenchmarkThreshold('training_step_time_ms', 10.0, 25.0),
            BenchmarkThreshold('memory_usage_gb', 5.0, 15.0),
            BenchmarkThreshold('communication_overhead_pct', 10.0, 20.0),
            BenchmarkThreshold('convergence_steps', 5.0, 15.0),
        ]
    
    def _load_baseline(self):
        """Load baseline benchmark results."""
        if self.baseline_path.exists():
            with open(self.baseline_path) as f:
                self.baseline = json.load(f)
    
    def record_current(self, metric_name: str, value: float):
        """Record current benchmark value."""
        self.current[metric_name] = value
    
    def check_regressions(self) -> Dict[str, Any]:
        """Check for regressions against baseline."""
        results = {
            'overall_pass': True,
            'checks': [],
            'timestamp': datetime.now().isoformat(),
        }
        
        for threshold in self.thresholds:
            metric = threshold.metric_name
            
            if metric not in self.baseline:
                results['checks'].append({
                    'metric': metric,
                    'status': 'SKIP',
                    'message': 'No baseline available',
                })
                continue
            
            if metric not in self.current:
                results['checks'].append({
                    'metric': metric,
                    'status': 'SKIP',
                    'message': 'No current value',
                })
                continue
            
            baseline_val = self.baseline[metric]
            current_val = self.current[metric]
            
            passed, message = threshold.check(baseline_val, current_val)
            
            results['checks'].append({
                'metric': metric,
                'baseline': baseline_val,
                'current': current_val,
                'status': 'PASS' if passed else 'FAIL',
                'message': message,
            })
            
            if not passed:
                results['overall_pass'] = False
        
        return results
    
    def update_baseline(self):
        """Update baseline with current values."""
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.baseline_path, 'w') as f:
            json.dump(self.current, f, indent=2)
        
        logger.info(f"Updated baseline: {self.baseline_path}")


# =============================================================================
# PART 5: DOCKER MANAGEMENT
# =============================================================================

class DockerManager:
    """Manages Docker containers for CI/CD."""
    
    def __init__(self, config: CIConfig):
        self.config = config
        self.image_tag = f"{config.docker_registry}/{config.docker_image_name}"
    
    def generate_dockerfile(self) -> str:
        """Generate Dockerfile content."""
        dockerfile = f"""# DES-LOC Benchmark Framework
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3.11 \\
    python3.11-dev \\
    python3-pip \\
    git \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \\
    torch==2.3.0 \\
    torchvision==0.18.0 \\
    --index-url https://download.pytorch.org/whl/cu121

# Copy project files
COPY . .

# Install project
RUN pip install -e .

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v"]
"""
        return dockerfile
    
    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml content."""
        compose = f"""version: '3.8'

services:
  desloc:
    build: .
    image: {self.image_tag}:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./checkpoints:/app/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  benchmark:
    extends:
      service: desloc
    command: python -m pytest tests/benchmarks -v --benchmark-json=/app/outputs/benchmark.json
"""
        return compose
    
    def save_files(self, output_dir: Path):
        """Save Docker files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dockerfile
        with open(output_dir / 'Dockerfile', 'w') as f:
            f.write(self.generate_dockerfile())
        
        # docker-compose.yml
        with open(output_dir / 'docker-compose.yml', 'w') as f:
            f.write(self.generate_docker_compose())
        
        logger.info(f"Generated Docker files in {output_dir}")


# =============================================================================
# PART 6: NOTIFICATION SERVICE
# =============================================================================

class NotificationService:
    """Sends notifications for CI/CD events."""
    
    def __init__(self, config: CIConfig):
        self.config = config
    
    def send_slack(self, message: str, success: bool = True):
        """Send Slack notification."""
        if not self.config.notify_slack or not self.config.slack_webhook:
            return
        
        color = "#36a64f" if success else "#ff0000"
        
        payload = {
            'attachments': [{
                'color': color,
                'title': 'DES-LOC CI/CD',
                'text': message,
                'ts': datetime.now().timestamp(),
            }]
        }
        
        try:
            import urllib.request
            req = urllib.request.Request(
                self.config.slack_webhook,
                data=json.dumps(payload).encode(),
                headers={'Content-Type': 'application/json'},
            )
            urllib.request.urlopen(req)
            logger.info("Sent Slack notification")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {e}")
    
    def notify_test_results(self, results: Dict[str, Any]):
        """Send notification about test results."""
        success = results.get('overall_success', False)
        
        if success:
            message = "✅ All tests passed!"
        else:
            failed = [
                name for name, r in results.get('results', {}).items()
                if not r.get('success', False)
            ]
            message = f"❌ Tests failed: {', '.join(failed)}"
        
        self.send_slack(message, success)


# =============================================================================
# PART 7: CI/CD MANAGER
# =============================================================================

class CICDManager:
    """Unified CI/CD management."""
    
    def __init__(self, project_dir: Path, config: CIConfig = None):
        self.project_dir = Path(project_dir)
        self.config = config or CIConfig()
        
        self.workflow_generator = GitHubActionsGenerator(self.config)
        self.docker_manager = DockerManager(self.config)
        self.notification_service = NotificationService(self.config)
    
    def setup(self):
        """Setup CI/CD for project."""
        # Generate GitHub Actions workflows
        self.workflow_generator.save(self.project_dir)
        
        # Generate Docker files
        self.docker_manager.save_files(self.project_dir)
        
        logger.info("CI/CD setup complete")
    
    def run_ci(self, test_types: List[TestType] = None) -> Dict[str, Any]:
        """Run CI pipeline locally."""
        runner = TestRunner(self.project_dir)
        results = runner.run_all(test_types)
        
        # Send notifications
        self.notification_service.notify_test_results(results)
        
        return results


# =============================================================================
# PART 8: MAIN / DEMO
# =============================================================================

def demo():
    """Demonstrate CI/CD functionality."""
    print("=" * 70)
    print("DES-LOC CI/CD Integration Demo")
    print("=" * 70)
    
    # Create config
    config = CIConfig(
        platform=CIPlatform.GITHUB_ACTIONS,
        python_versions=['3.10', '3.11'],
        gpu_enabled=True,
        use_docker=True,
    )
    
    print(f"\nCI/CD Config:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Create manager
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Setup CI/CD
        manager = CICDManager(tmpdir, config)
        manager.setup()
        
        # List generated files
        print("\nGenerated files:")
        for f in tmpdir.rglob('*'):
            if f.is_file():
                print(f"  - {f.relative_to(tmpdir)}")
        
        # Show workflow content
        workflow_path = tmpdir / '.github' / 'workflows' / 'ci.yml'
        if workflow_path.exists():
            print("\nGenerated CI workflow (excerpt):")
            with open(workflow_path) as f:
                content = f.read()
                # Show first 40 lines
                lines = content.split('\n')[:40]
                print('\n'.join(lines))
                if len(content.split('\n')) > 40:
                    print("  ...")
    
    # Demo benchmark regression
    print("\nBenchmark Regression Test Demo:")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        baseline = {
            'training_step_time_ms': 100.0,
            'memory_usage_gb': 20.0,
        }
        json.dump(baseline, f)
        baseline_path = Path(f.name)
    
    tester = BenchmarkRegressionTester(baseline_path)
    tester.record_current('training_step_time_ms', 105.0)  # 5% increase
    tester.record_current('memory_usage_gb', 25.0)  # 25% increase
    
    results = tester.check_regressions()
    print(json.dumps(results, indent=2))
    
    # Cleanup
    baseline_path.unlink()
    
    print("\n[M050] CI/CD Integration Demo - COMPLETED")


if __name__ == "__main__":
    demo()
