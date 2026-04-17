#!/usr/bin/env python3
"""
===============================================================================
M042: Trainium2 Migration Analysis Module
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module analyzes the migration path from NVIDIA GPUs to AWS Trainium2.
Maps CUDA/cuDNN operations to NeuronSDK equivalents and identifies
compatibility requirements.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Features:
- CUDA to NeuronCore operation mapping
- Megatron-LM pattern analysis
- Communication primitive translation (NCCL → Neuron CC)
- Performance estimation for Trainium2
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M042"

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Set
from datetime import datetime
from enum import Enum
import logging
import ast
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: OPERATION CATEGORIES
# =============================================================================

class OperationCategory(Enum):
    """Categories of operations for migration."""
    COMPUTE = "compute"
    MEMORY = "memory"
    COMMUNICATION = "communication"
    SYNCHRONIZATION = "synchronization"
    CUSTOM_KERNEL = "custom_kernel"


class CompatibilityLevel(Enum):
    """Compatibility levels for operation migration."""
    NATIVE = "native"           # Direct equivalent exists
    COMPATIBLE = "compatible"   # Requires minor changes
    WORKAROUND = "workaround"   # Requires significant changes
    UNSUPPORTED = "unsupported" # Not supported


@dataclass
class OperationMapping:
    """Mapping between CUDA and Neuron operations."""
    cuda_op: str
    neuron_op: str
    category: OperationCategory
    compatibility: CompatibilityLevel
    notes: str = ""
    performance_factor: float = 1.0  # Relative performance on Trainium2


# =============================================================================
# PART 2: CUDA TO NEURON MAPPING DATABASE
# =============================================================================

class OperationMappingDB:
    """Database of CUDA to Neuron SDK mappings."""
    
    # Core tensor operations
    TENSOR_OPS = [
        OperationMapping(
            "torch.matmul", "nki.isa.matmul",
            OperationCategory.COMPUTE, CompatibilityLevel.NATIVE,
            "Direct mapping via NKI", 1.2
        ),
        OperationMapping(
            "torch.bmm", "nki.isa.matmul",
            OperationCategory.COMPUTE, CompatibilityLevel.NATIVE,
            "Batch matmul supported", 1.2
        ),
        OperationMapping(
            "F.linear", "nki.isa.matmul",
            OperationCategory.COMPUTE, CompatibilityLevel.NATIVE,
            "Linear layer mapping", 1.2
        ),
        OperationMapping(
            "F.conv2d", "nki.isa.conv2d",
            OperationCategory.COMPUTE, CompatibilityLevel.NATIVE,
            "Convolution supported", 1.1
        ),
    ]
    
    # Attention operations
    ATTENTION_OPS = [
        OperationMapping(
            "F.scaled_dot_product_attention", "nki.kernels.flash_attention",
            OperationCategory.COMPUTE, CompatibilityLevel.NATIVE,
            "FlashAttention via NKI-FA", 1.3
        ),
        OperationMapping(
            "xformers.memory_efficient_attention", "nki.kernels.flash_attention",
            OperationCategory.COMPUTE, CompatibilityLevel.COMPATIBLE,
            "Use NKI FlashAttention instead", 1.3
        ),
        OperationMapping(
            "flash_attn.flash_attn_func", "nki.kernels.flash_attention",
            OperationCategory.COMPUTE, CompatibilityLevel.NATIVE,
            "Direct NKI-FA mapping (da964f3)", 1.3
        ),
    ]
    
    # Activation functions
    ACTIVATION_OPS = [
        OperationMapping(
            "F.gelu", "nki.isa.activation.gelu",
            OperationCategory.COMPUTE, CompatibilityLevel.NATIVE,
            "Native GELU support", 1.0
        ),
        OperationMapping(
            "F.silu", "nki.isa.activation.silu",
            OperationCategory.COMPUTE, CompatibilityLevel.NATIVE,
            "SwiGLU supported", 1.0
        ),
        OperationMapping(
            "F.relu", "nki.isa.activation.relu",
            OperationCategory.COMPUTE, CompatibilityLevel.NATIVE,
            "Native ReLU", 1.0
        ),
        OperationMapping(
            "F.softmax", "nki.isa.softmax",
            OperationCategory.COMPUTE, CompatibilityLevel.NATIVE,
            "Optimized softmax", 1.1
        ),
    ]
    
    # Normalization
    NORMALIZATION_OPS = [
        OperationMapping(
            "F.layer_norm", "nki.isa.layer_norm",
            OperationCategory.COMPUTE, CompatibilityLevel.NATIVE,
            "Fused LayerNorm", 1.1
        ),
        OperationMapping(
            "F.rms_norm", "nki.isa.rms_norm",
            OperationCategory.COMPUTE, CompatibilityLevel.NATIVE,
            "RMSNorm for LLMs", 1.2
        ),
        OperationMapping(
            "F.batch_norm", "nki.isa.batch_norm",
            OperationCategory.COMPUTE, CompatibilityLevel.NATIVE,
            "BatchNorm supported", 1.0
        ),
    ]
    
    # Communication operations
    COMMUNICATION_OPS = [
        OperationMapping(
            "dist.all_reduce", "neuron_cc.all_reduce",
            OperationCategory.COMMUNICATION, CompatibilityLevel.NATIVE,
            "Neuron Collective Communication", 1.0
        ),
        OperationMapping(
            "dist.broadcast", "neuron_cc.broadcast",
            OperationCategory.COMMUNICATION, CompatibilityLevel.NATIVE,
            "Direct mapping", 1.0
        ),
        OperationMapping(
            "dist.all_gather", "neuron_cc.all_gather",
            OperationCategory.COMMUNICATION, CompatibilityLevel.NATIVE,
            "Native collective", 1.0
        ),
        OperationMapping(
            "dist.reduce_scatter", "neuron_cc.reduce_scatter",
            OperationCategory.COMMUNICATION, CompatibilityLevel.NATIVE,
            "Native collective", 1.0
        ),
        OperationMapping(
            "nccl.all_reduce", "neuron_cc.all_reduce",
            OperationCategory.COMMUNICATION, CompatibilityLevel.COMPATIBLE,
            "Replace NCCL with Neuron CC", 1.0
        ),
    ]
    
    # Memory operations
    MEMORY_OPS = [
        OperationMapping(
            "torch.cuda.memory.allocate", "neuron.memory.allocate",
            OperationCategory.MEMORY, CompatibilityLevel.COMPATIBLE,
            "Different memory model", 0.9
        ),
        OperationMapping(
            "torch.cuda.synchronize", "neuron.synchronize",
            OperationCategory.SYNCHRONIZATION, CompatibilityLevel.NATIVE,
            "Device sync", 1.0
        ),
        OperationMapping(
            "torch.cuda.Stream", "neuron.Stream",
            OperationCategory.SYNCHRONIZATION, CompatibilityLevel.COMPATIBLE,
            "Stream abstraction available", 1.0
        ),
    ]
    
    # Custom CUDA kernels
    CUSTOM_OPS = [
        OperationMapping(
            "triton.jit", "nki.jit",
            OperationCategory.CUSTOM_KERNEL, CompatibilityLevel.WORKAROUND,
            "Rewrite Triton kernels in NKI", 0.8
        ),
        OperationMapping(
            "torch.cuda.amp", "neuron.amp",
            OperationCategory.COMPUTE, CompatibilityLevel.NATIVE,
            "Automatic mixed precision", 1.2
        ),
    ]
    
    @classmethod
    def get_all_mappings(cls) -> List[OperationMapping]:
        """Get all operation mappings."""
        return (cls.TENSOR_OPS + cls.ATTENTION_OPS + cls.ACTIVATION_OPS +
                cls.NORMALIZATION_OPS + cls.COMMUNICATION_OPS +
                cls.MEMORY_OPS + cls.CUSTOM_OPS)
    
    @classmethod
    def lookup(cls, cuda_op: str) -> Optional[OperationMapping]:
        """Look up mapping for a CUDA operation."""
        for mapping in cls.get_all_mappings():
            if mapping.cuda_op == cuda_op or cuda_op.endswith(mapping.cuda_op):
                return mapping
        return None


# =============================================================================
# PART 3: CODE ANALYZER
# =============================================================================

class CodeAnalyzer:
    """Analyzes PyTorch code for migration requirements."""
    
    # Patterns to detect
    CUDA_PATTERNS = [
        (r'torch\.cuda\.\w+', 'CUDA API'),
        (r'\.cuda\(\)', 'Device placement'),
        (r'torch\.device\([\'"]cuda', 'Device specification'),
        (r'F\.\w+', 'Functional ops'),
        (r'nn\.\w+', 'Module ops'),
        (r'dist\.\w+', 'Distributed ops'),
        (r'nccl', 'NCCL communication'),
        (r'triton\.jit', 'Triton kernel'),
        (r'@cuda\.jit', 'CUDA kernel'),
    ]
    
    def __init__(self):
        self.findings: List[Dict[str, Any]] = []
        self.operation_counts: Dict[str, int] = {}
    
    def analyze_file(self, filepath: Path) -> Dict[str, Any]:
        """Analyze a Python file for CUDA operations."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return {}
        
        file_findings = {
            'filepath': str(filepath),
            'operations': [],
            'compatibility_issues': [],
            'recommendations': [],
        }
        
        # Pattern matching
        for pattern, category in self.CUDA_PATTERNS:
            matches = re.findall(pattern, content)
            for match in matches:
                file_findings['operations'].append({
                    'operation': match,
                    'category': category,
                    'mapping': self._get_mapping_info(match),
                })
                
                self.operation_counts[match] = self.operation_counts.get(match, 0) + 1
        
        # AST analysis for more precise detection
        try:
            tree = ast.parse(content)
            self._analyze_ast(tree, file_findings)
        except SyntaxError:
            pass
        
        self.findings.append(file_findings)
        return file_findings
    
    def _get_mapping_info(self, operation: str) -> Dict[str, Any]:
        """Get mapping information for an operation."""
        mapping = OperationMappingDB.lookup(operation)
        
        if mapping:
            return {
                'neuron_equivalent': mapping.neuron_op,
                'compatibility': mapping.compatibility.value,
                'performance_factor': mapping.performance_factor,
                'notes': mapping.notes,
            }
        
        return {
            'neuron_equivalent': 'Unknown',
            'compatibility': 'needs_investigation',
            'performance_factor': 1.0,
            'notes': 'No direct mapping found',
        }
    
    def _analyze_ast(self, tree: ast.AST, findings: Dict):
        """Analyze AST for detailed operation detection."""
        for node in ast.walk(tree):
            # Detect function calls
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name:
                    # Check for known operations
                    if any(pattern in func_name for pattern in 
                          ['cuda', 'distributed', 'nccl', 'all_reduce']):
                        findings['operations'].append({
                            'operation': func_name,
                            'category': 'AST detection',
                            'line': getattr(node, 'lineno', 'unknown'),
                        })
    
    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract function name from Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        return None
    
    def analyze_directory(self, directory: Path) -> Dict[str, Any]:
        """Analyze all Python files in a directory."""
        directory = Path(directory)
        
        for py_file in directory.rglob('*.py'):
            self.analyze_file(py_file)
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate migration analysis report."""
        # Categorize by compatibility
        compatibility_summary = {
            'native': 0,
            'compatible': 0,
            'workaround': 0,
            'unsupported': 0,
            'unknown': 0,
        }
        
        for finding in self.findings:
            for op in finding.get('operations', []):
                mapping = op.get('mapping', {})
                compat = mapping.get('compatibility', 'unknown')
                compatibility_summary[compat] = compatibility_summary.get(compat, 0) + 1
        
        # Top operations
        top_ops = sorted(
            self.operation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        return {
            'summary': {
                'files_analyzed': len(self.findings),
                'total_operations': sum(self.operation_counts.values()),
                'unique_operations': len(self.operation_counts),
                'compatibility': compatibility_summary,
            },
            'top_operations': top_ops,
            'detailed_findings': self.findings,
            'recommendations': self._generate_recommendations(compatibility_summary),
        }
    
    def _generate_recommendations(self, compatibility: Dict[str, int]) -> List[str]:
        """Generate migration recommendations."""
        recommendations = []
        
        if compatibility.get('unsupported', 0) > 0:
            recommendations.append(
                "⚠️ Found unsupported operations. Review and implement alternatives."
            )
        
        if compatibility.get('workaround', 0) > 0:
            recommendations.append(
                "⚙️ Some operations require workarounds. Plan for kernel rewrites."
            )
        
        if compatibility.get('compatible', 0) > 0:
            recommendations.append(
                "✓ Many operations are compatible with minor changes."
            )
        
        if compatibility.get('native', 0) > 0:
            recommendations.append(
                "✓ Native support available for core operations."
            )
        
        # Specific recommendations
        recommendations.extend([
            "1. Replace NCCL with Neuron Collective Communication",
            "2. Rewrite Triton kernels using NKI",
            "3. Use torch_neuronx for model compilation",
            "4. Enable XLA optimizations for better fusion",
            "5. Profile with Neuron tools for performance tuning",
        ])
        
        return recommendations


# =============================================================================
# PART 4: MEGATRON-LM PATTERN ANALYZER
# =============================================================================

class MegatronPatternAnalyzer:
    """Analyzes Megatron-LM patterns for migration."""
    
    # Key Megatron-LM patterns
    PATTERNS = {
        'tensor_parallel': {
            'pattern': r'ColumnParallelLinear|RowParallelLinear',
            'neuron_approach': 'Use tensor_parallel_size in Neuron config',
            'complexity': 'medium',
        },
        'pipeline_parallel': {
            'pattern': r'PipelineParallel|pipe_parallel',
            'neuron_approach': 'Neuron pipeline parallelism via PP groups',
            'complexity': 'high',
        },
        'data_parallel': {
            'pattern': r'DistributedDataParallel|data_parallel',
            'neuron_approach': 'Native DDP support in torch_neuronx',
            'complexity': 'low',
        },
        'sequence_parallel': {
            'pattern': r'SequenceParallel|sequence_parallel',
            'neuron_approach': 'Implement via custom sharding',
            'complexity': 'high',
        },
        'gradient_checkpointing': {
            'pattern': r'checkpoint_activations|gradient_checkpointing',
            'neuron_approach': 'Use torch.utils.checkpoint',
            'complexity': 'low',
        },
        'mixed_precision': {
            'pattern': r'fp16|bf16|amp',
            'neuron_approach': 'Native BF16 on Trainium2',
            'complexity': 'low',
        },
    }
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """Analyze code for Megatron-LM patterns."""
        findings = {}
        
        for name, info in self.PATTERNS.items():
            matches = re.findall(info['pattern'], code)
            if matches:
                findings[name] = {
                    'found': True,
                    'count': len(matches),
                    'neuron_approach': info['neuron_approach'],
                    'complexity': info['complexity'],
                }
        
        return findings


# =============================================================================
# PART 5: PERFORMANCE ESTIMATOR
# =============================================================================

@dataclass
class PerformanceEstimate:
    """Performance estimate for Trainium2 migration."""
    operation: str
    cuda_baseline_tflops: float
    trainium2_estimate_tflops: float
    speedup: float
    notes: str


class Trainium2PerformanceEstimator:
    """Estimates performance on Trainium2."""
    
    # Trainium2 specifications
    TRAINIUM2_SPECS = {
        'bf16_tflops': 380,      # Per chip
        'fp32_tflops': 190,
        'hbm_bandwidth_tb': 1.6, # TB/s
        'interconnect_tb': 1.6,  # NeuronLink
    }
    
    # Reference NVIDIA specs (H100)
    H100_SPECS = {
        'bf16_tflops': 1979,     # Tensor Core
        'fp32_tflops': 67,       # FP32
        'hbm_bandwidth_tb': 3.35,
    }
    
    def estimate_operation(
        self,
        operation: str,
        flops: float,
        memory_bound: bool = False,
    ) -> PerformanceEstimate:
        """Estimate performance for an operation."""
        mapping = OperationMappingDB.lookup(operation)
        perf_factor = mapping.performance_factor if mapping else 1.0
        
        if memory_bound:
            # Memory bandwidth limited
            h100_time = flops / (self.H100_SPECS['hbm_bandwidth_tb'] * 1e12)
            trn2_time = flops / (self.TRAINIUM2_SPECS['hbm_bandwidth_tb'] * 1e12)
        else:
            # Compute limited
            h100_time = flops / (self.H100_SPECS['bf16_tflops'] * 1e12)
            trn2_time = flops / (self.TRAINIUM2_SPECS['bf16_tflops'] * 1e12)
        
        # Apply performance factor
        trn2_time = trn2_time / perf_factor
        
        return PerformanceEstimate(
            operation=operation,
            cuda_baseline_tflops=self.H100_SPECS['bf16_tflops'],
            trainium2_estimate_tflops=self.TRAINIUM2_SPECS['bf16_tflops'] * perf_factor,
            speedup=h100_time / trn2_time if trn2_time > 0 else 0,
            notes=f"Performance factor: {perf_factor:.2f}",
        )


# =============================================================================
# PART 6: MIGRATION REPORT GENERATOR
# =============================================================================

class MigrationReportGenerator:
    """Generates comprehensive migration report."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.code_analyzer = CodeAnalyzer()
        self.megatron_analyzer = MegatronPatternAnalyzer()
        self.performance_estimator = Trainium2PerformanceEstimator()
    
    def generate_report(
        self,
        source_dir: Path,
        output_name: str = "migration_report",
    ) -> Path:
        """Generate full migration report."""
        # Analyze code
        analysis = self.code_analyzer.analyze_directory(source_dir)
        
        # Generate report
        report = {
            'title': 'Trainium2 Migration Analysis',
            'timestamp': datetime.now().isoformat(),
            'source_directory': str(source_dir),
            'analysis': analysis,
            'operation_mappings': [
                asdict(m) if hasattr(m, '__dict__') else {
                    'cuda_op': m.cuda_op,
                    'neuron_op': m.neuron_op,
                    'category': m.category.value,
                    'compatibility': m.compatibility.value,
                    'notes': m.notes,
                }
                for m in OperationMappingDB.get_all_mappings()
            ],
            'recommendations': analysis.get('recommendations', []),
        }
        
        # Save JSON report
        json_path = self.output_dir / f"{output_name}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save text report
        txt_path = self.output_dir / f"{output_name}.txt"
        self._write_text_report(report, txt_path)
        
        logger.info(f"Migration report saved to {json_path}")
        return json_path
    
    def _write_text_report(self, report: Dict, path: Path):
        """Write human-readable text report."""
        with open(path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TRAINIUM2 MIGRATION ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {report['timestamp']}\n")
            f.write(f"Source: {report['source_directory']}\n\n")
            
            # Summary
            summary = report['analysis'].get('summary', {})
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Files analyzed: {summary.get('files_analyzed', 0)}\n")
            f.write(f"Total operations: {summary.get('total_operations', 0)}\n")
            f.write(f"Unique operations: {summary.get('unique_operations', 0)}\n\n")
            
            # Compatibility
            compat = summary.get('compatibility', {})
            f.write("COMPATIBILITY BREAKDOWN\n")
            f.write("-" * 40 + "\n")
            for level, count in compat.items():
                f.write(f"  {level}: {count}\n")
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            for rec in report.get('recommendations', []):
                f.write(f"• {rec}\n")


# =============================================================================
# PART 7: MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Trainium2 Migration Analysis")
    parser.add_argument("--source-dir", type=str, required=True,
                       help="Source code directory to analyze")
    parser.add_argument("--output-dir", type=str, default="./migration_reports")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 70)
    print("NeurIPS 2026 - DES-LOC Benchmark")
    print("Trainium2 Migration Analysis")
    print("=" * 70)
    
    generator = MigrationReportGenerator(Path(args.output_dir))
    report_path = generator.generate_report(Path(args.source_dir))
    
    print(f"\nReport saved to: {report_path}")
    print("\n[M042] Trainium2 Migration Analysis - COMPLETED")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
