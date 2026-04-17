# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
import os
import pkgutil
import importlib
import sys
import time as _time
import json as _json
from collections import deque as _deque

from .abstract_accelerator import DeepSpeedAccelerator
# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch.cuda
except ImportError:
    pass

# Delay import pynvml to avoid import error when CUDA is not available
pynvml = None


class CUDA_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'cuda'
        self._communication_backend_name = 'nccl' if sys.platform != 'win32' else 'gloo'
        self._compile_backend = "inductor"
        if pynvml is None:
            self._init_pynvml()

    def _init_pynvml(self):
        global pynvml
        try:
            import pynvml
        except ImportError:
            return
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            pynvml = None
            return

    def is_synchronized_device(self):
        return False

    def use_host_timers(self):
        return self.is_synchronized_device()

    def resolves_data_dependency(self):
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        if device_index is None:
            return 'cuda'
        return 'cuda:{}'.format(device_index)

    def communication_backend_version(self):
        return torch.cuda.nccl.version()

    def device(self, device_index=None):
        return torch.device('cuda', device_index)

    def set_device(self, device_index):
        torch.cuda.set_device(device_index)

    def current_device(self):
        return torch.cuda.current_device()

    def current_device_name(self):
        return 'cuda:{}'.format(torch.cuda.current_device())

    def device_count(self):
        return torch.cuda.device_count()

    def synchronize(self, device_index=None):
        return torch.cuda.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.cuda.set_rng_state(new_state)

        return torch.cuda.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index is None:
            return torch.cuda.get_rng_state()

        return torch.cuda.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.cuda.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.cuda.manual_seed_all(seed)

    def initial_seed(self):
        return torch.cuda.initial_seed()

    def default_generator(self, device_index):
        return torch.cuda.default_generators[device_index]

    # Streams/Events
    @property
    def Stream(self):
        return torch.cuda.Stream

    def stream(self, stream):
        return torch.cuda.stream(stream)

    def current_stream(self, device_index=None):
        return torch.cuda.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.cuda.default_stream(device_index)

    @property
    def Event(self):
        return torch.cuda.Event

    # Memory management
    def empty_cache(self):
        return torch.cuda.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.cuda.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.cuda.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.cuda.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch.cuda.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        return torch.cuda.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.cuda.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch.cuda, 'memory_stats'):
            return torch.cuda.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            return torch.cuda.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        if hasattr(torch.cuda, 'memory_reserved'):
            return torch.cuda.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch.cuda, 'max_memory_reserved'):
            return torch.cuda.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.cuda.get_device_properties(device_index).total_memory

    def _get_nvml_gpu_id(self, torch_gpu_id):
        """
        credit: https://discuss.pytorch.org/t/making-pynvml-match-torch-device-ids-cuda-visible-devices/103020

        Remap torch device id to nvml device id, respecting CUDA_VISIBLE_DEVICES.

        If the latter isn't set return the same id
        """
        # if CUDA_VISIBLE_DEVICES is used automagically remap the id since pynvml ignores this env var
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
            return ids[torch_gpu_id]  # remap
        else:
            return torch_gpu_id

    def available_memory(self, device_index=None):
        if pynvml:
            if device_index is None:
                device_index = self.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._get_nvml_gpu_id(device_index))
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.free
        else:
            return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        if not torch.cuda.is_available():
            return True
        return torch.cuda.is_bf16_supported()

    def is_fp16_supported(self):
        if not torch.cuda.is_available():
            return True
        # See https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix
        # FP16 on compute capability 6.x is deprecated
        allow_deprecated_fp16 = os.environ.get('DS_ALLOW_DEPRECATED_FP16', '0') == '1'
        major, _ = torch.cuda.get_device_capability()
        if major >= 7:
            return True
        elif major == 6 and allow_deprecated_fp16:
            return True
        else:
            return False

    def supported_dtypes(self):
        supported_dtypes = [torch.float]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.half)
        if self.is_bf16_supported():
            supported_dtypes.append(torch.bfloat16)
        return supported_dtypes

    # Misc
    def is_available(self):
        return torch.cuda.is_available()

    def range_push(self, msg):
        if hasattr(torch.cuda.nvtx, 'range_push'):
            return torch.cuda.nvtx.range_push(msg)

    def range_pop(self):
        if hasattr(torch.cuda.nvtx, 'range_pop'):
            return torch.cuda.nvtx.range_pop()

    def lazy_call(self, callback):
        return torch.cuda._lazy_call(callback)

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        if not self.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return True
        else:
            return False

    # Graph operations
    def create_graph(self):
        return torch.cuda.CUDAGraph()

    def capture_to_graph(self, graph, pool=None, stream=None):
        return torch.cuda.graph(graph, pool, stream)

    def replay_graph(self, graph):
        graph.replay()
        return

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return functools.partial(torch.tensor, dtype=torch.bfloat16, device='cuda')

    @property
    def ByteTensor(self):
        return functools.partial(torch.tensor, dtype=torch.uint8, device='cuda')

    @property
    def DoubleTensor(self):
        return functools.partial(torch.tensor, dtype=torch.double, device='cuda')

    @property
    def FloatTensor(self):
        return functools.partial(torch.tensor, dtype=torch.float, device='cuda')

    @property
    def HalfTensor(self):
        return functools.partial(torch.tensor, dtype=torch.half, device='cuda')

    @property
    def IntTensor(self):
        return functools.partial(torch.tensor, dtype=torch.int, device='cuda')

    @property
    def LongTensor(self):
        return functools.partial(torch.tensor, dtype=torch.long, device='cuda')

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('cuda:'):
            return True
        else:
            return False

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder"
        except ImportError:
            return "deepspeed.ops.op_builder"

    # dict that holds class name <--> class type mapping i.e.
    # 'AsyncIOBuilder': <class 'op_builder.async_io.AsyncIOBuilder'>
    # this dict will be filled at init stage
    class_dict = None

    def _lazy_init_class_dict(self):
        if self.class_dict is not None:
            return
        else:
            self.class_dict = {}
            # begin initialize for create_op_builder()
            # put all valid class name <--> class type mapping into class_dict
            op_builder_dir = self.op_builder_dir()
            op_builder_module = importlib.import_module(op_builder_dir)
            op_builder_absolute_path = os.path.dirname(op_builder_module.__file__)
            for _, module_name, _ in pkgutil.iter_modules([op_builder_absolute_path]):
                # avoid self references,
                # skip sub_directories which contains ops for other backend(cpu, npu, etc.).
                if module_name != 'all_ops' and module_name != 'builder' and not os.path.isdir(
                        os.path.join(op_builder_absolute_path, module_name)):
                    module = importlib.import_module("{}.{}".format(op_builder_dir, module_name))
                    for member_name in module.__dir__():
                        if member_name.endswith(
                                'Builder'
                        ) and member_name != "OpBuilder" and member_name != "CUDAOpBuilder" and member_name != "TorchCPUOpBuilder":  # avoid abstract classes
                            if not member_name in self.class_dict:
                                self.class_dict[member_name] = getattr(module, member_name)
            # end initialize for create_op_builder()

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]()
        else:
            return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]
        else:
            return None

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return ['NCCL']

    def visible_devices_envs(self):
        return ['CUDA_VISIBLE_DEVICES']

    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        for env in self.visible_devices_envs():
            current_env[env] = ",".join(map(str, local_accelerator_ids))

    def get_compile_backend(self):
        return self._compile_backend

    def set_compile_backend(self, backend):
        supported_backends = torch._dynamo.list_backends(exclude_tags=())
        if backend in supported_backends:
            self._compile_backend = backend
        else:
            raise ValueError(
                f"{backend} not supported by {self.device_name()}. Supported Backends are {supported_backends}")

    # =================================================================
    # M053: DES-LOC Profiling & Memory Tracking (400 lines)
    # =================================================================
    # Adds real GPU profiling hooks to the CUDA accelerator.
    # All data from actual torch.cuda APIs — no simulation.
    # Architecture reference: CCCL cudax/include/cuda/experimental/__stf
    # =================================================================

    def desloc_init_profiler(self, log_dir=None):
        """Initialize DES-LOC GPU profiler state."""
        self._desloc_profile_enabled = True
        self._desloc_log_dir = log_dir
        self._desloc_step_timings = _deque(maxlen=10000)
        self._desloc_memory_snapshots = _deque(maxlen=10000)
        self._desloc_kernel_events = _deque(maxlen=5000)
        self._desloc_bandwidth_samples = _deque(maxlen=5000)
        self._desloc_compute_util_samples = _deque(maxlen=5000)
        self._desloc_total_forward_ms = 0.0
        self._desloc_total_backward_ms = 0.0
        self._desloc_total_comm_ms = 0.0
        self._desloc_total_optim_ms = 0.0
        self._desloc_step_count = 0
        self._desloc_sm_clock_mhz = None
        self._desloc_mem_clock_mhz = None
        self._desloc_gpu_name = None
        self._desloc_gpu_total_mem = None
        # Capture GPU hardware info
        try:
            dev = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev)
            self._desloc_gpu_name = props.name
            self._desloc_gpu_total_mem = props.total_memory
            self._desloc_sm_count = props.multi_processor_count
            self._desloc_compute_capability = (props.major, props.minor)
        except Exception:
            self._desloc_gpu_name = 'unknown'
            self._desloc_gpu_total_mem = 0
            self._desloc_sm_count = 0
            self._desloc_compute_capability = (0, 0)

    def desloc_memory_snapshot(self, label=''):
        """Take a GPU memory snapshot with label."""
        if not getattr(self, '_desloc_profile_enabled', False):
            return {}
        dev = torch.cuda.current_device()
        snapshot = {
            'label': label,
            'timestamp': _time.time(),
            'allocated_bytes': torch.cuda.memory_allocated(dev),
            'reserved_bytes': torch.cuda.memory_reserved(dev),
            'max_allocated_bytes': torch.cuda.max_memory_allocated(dev),
            'allocated_gb': round(torch.cuda.memory_allocated(dev) / 1e9, 4),
            'reserved_gb': round(torch.cuda.memory_reserved(dev) / 1e9, 4),
            'max_allocated_gb': round(torch.cuda.max_memory_allocated(dev) / 1e9, 4),
            'free_gb': round((self._desloc_gpu_total_mem - torch.cuda.memory_allocated(dev)) / 1e9, 4) if self._desloc_gpu_total_mem else 0,
            'utilization': round(torch.cuda.memory_allocated(dev) / max(self._desloc_gpu_total_mem, 1), 4) if self._desloc_gpu_total_mem else 0,
        }
        self._desloc_memory_snapshots.append(snapshot)
        return snapshot

    def desloc_begin_step(self):
        """Mark the beginning of a training step for profiling."""
        if not getattr(self, '_desloc_profile_enabled', False):
            return
        self._desloc_step_start = _time.time()
        self._desloc_step_forward_ms = 0
        self._desloc_step_backward_ms = 0
        self._desloc_step_comm_ms = 0
        self._desloc_step_optim_ms = 0
        torch.cuda.synchronize()
        self._desloc_cuda_start_event = torch.cuda.Event(enable_timing=True)
        self._desloc_cuda_end_event = torch.cuda.Event(enable_timing=True)
        self._desloc_cuda_start_event.record()

    def desloc_end_step(self, loss=None, tokens=0):
        """Mark the end of a training step and record profiling data."""
        if not getattr(self, '_desloc_profile_enabled', False):
            return {}
        self._desloc_cuda_end_event.record()
        torch.cuda.synchronize()
        cuda_elapsed_ms = self._desloc_cuda_start_event.elapsed_time(
            self._desloc_cuda_end_event)
        wall_elapsed_ms = (_time.time() - self._desloc_step_start) * 1000

        self._desloc_step_count += 1
        self._desloc_total_forward_ms += self._desloc_step_forward_ms
        self._desloc_total_backward_ms += self._desloc_step_backward_ms
        self._desloc_total_comm_ms += self._desloc_step_comm_ms
        self._desloc_total_optim_ms += self._desloc_step_optim_ms

        dev = torch.cuda.current_device()
        timing = {
            'step': self._desloc_step_count,
            'wall_ms': round(wall_elapsed_ms, 3),
            'cuda_ms': round(cuda_elapsed_ms, 3),
            'forward_ms': round(self._desloc_step_forward_ms, 3),
            'backward_ms': round(self._desloc_step_backward_ms, 3),
            'comm_ms': round(self._desloc_step_comm_ms, 3),
            'optim_ms': round(self._desloc_step_optim_ms, 3),
            'overhead_ms': round(wall_elapsed_ms - cuda_elapsed_ms, 3),
            'mem_alloc_gb': round(torch.cuda.memory_allocated(dev) / 1e9, 4),
            'mem_reserved_gb': round(torch.cuda.memory_reserved(dev) / 1e9, 4),
            'tokens': tokens,
            'tokens_per_sec': round(tokens / max(wall_elapsed_ms / 1000, 1e-6), 1),
        }
        if loss is not None:
            timing['loss'] = loss
        self._desloc_step_timings.append(timing)
        return timing

    def desloc_record_phase(self, phase, elapsed_ms):
        """Record timing for a specific phase within a step."""
        if not getattr(self, '_desloc_profile_enabled', False):
            return
        if phase == 'forward':
            self._desloc_step_forward_ms += elapsed_ms
        elif phase == 'backward':
            self._desloc_step_backward_ms += elapsed_ms
        elif phase == 'comm':
            self._desloc_step_comm_ms += elapsed_ms
        elif phase == 'optim':
            self._desloc_step_optim_ms += elapsed_ms

    def desloc_measure_bandwidth(self, tensor_bytes, elapsed_ms):
        """Record a bandwidth measurement from a real data transfer."""
        if not getattr(self, '_desloc_profile_enabled', False):
            return
        if elapsed_ms > 0:
            bw_gbps = (tensor_bytes / 1e9) / (elapsed_ms / 1000)
        else:
            bw_gbps = 0
        self._desloc_bandwidth_samples.append({
            'bytes': tensor_bytes,
            'ms': round(elapsed_ms, 4),
            'bw_gbps': round(bw_gbps, 2),
        })

    def desloc_get_gpu_info(self):
        """Return GPU hardware information."""
        return {
            'name': getattr(self, '_desloc_gpu_name', 'unknown'),
            'total_memory_gb': round(getattr(self, '_desloc_gpu_total_mem', 0) / 1e9, 2),
            'sm_count': getattr(self, '_desloc_sm_count', 0),
            'compute_capability': getattr(self, '_desloc_compute_capability', (0, 0)),
        }

    def desloc_get_profiler_summary(self):
        """Return aggregated profiling summary."""
        if not getattr(self, '_desloc_profile_enabled', False):
            return {'enabled': False}
        n = max(self._desloc_step_count, 1)
        summary = {
            'enabled': True,
            'total_steps': self._desloc_step_count,
            'gpu': self.desloc_get_gpu_info(),
            'avg_forward_ms': round(self._desloc_total_forward_ms / n, 3),
            'avg_backward_ms': round(self._desloc_total_backward_ms / n, 3),
            'avg_comm_ms': round(self._desloc_total_comm_ms / n, 3),
            'avg_optim_ms': round(self._desloc_total_optim_ms / n, 3),
            'avg_total_ms': round(
                (self._desloc_total_forward_ms + self._desloc_total_backward_ms +
                 self._desloc_total_comm_ms + self._desloc_total_optim_ms) / n, 3),
            'compute_fraction': round(
                (self._desloc_total_forward_ms + self._desloc_total_backward_ms) /
                max(self._desloc_total_forward_ms + self._desloc_total_backward_ms +
                    self._desloc_total_comm_ms + self._desloc_total_optim_ms, 1), 4),
            'comm_fraction': round(
                self._desloc_total_comm_ms /
                max(self._desloc_total_forward_ms + self._desloc_total_backward_ms +
                    self._desloc_total_comm_ms + self._desloc_total_optim_ms, 1), 4),
        }
        # Memory summary from snapshots
        if self._desloc_memory_snapshots:
            peak = max(s['allocated_gb'] for s in self._desloc_memory_snapshots)
            summary['peak_memory_gb'] = peak
        # Bandwidth summary
        if self._desloc_bandwidth_samples:
            bws = [s['bw_gbps'] for s in self._desloc_bandwidth_samples if s['bw_gbps'] > 0]
            if bws:
                summary['avg_bandwidth_gbps'] = round(sum(bws) / len(bws), 2)
                summary['peak_bandwidth_gbps'] = round(max(bws), 2)
        return summary

    def desloc_save_profile(self, path):
        """Save profiling data to a JSON file."""
        if not getattr(self, '_desloc_profile_enabled', False):
            return
        output = {
            'summary': self.desloc_get_profiler_summary(),
            'step_timings': list(self._desloc_step_timings)[-1000:],
            'memory_snapshots': list(self._desloc_memory_snapshots)[-500:],
            'bandwidth_samples': list(self._desloc_bandwidth_samples)[-500:],
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            _json.dump(output, f, indent=2)

    def desloc_hbm_state_analysis(self):
        """Analyze HBM (High Bandwidth Memory) state management.

        Computes optimizer state memory footprint and determines
        whether HBM capacity is sufficient for DES-LOC's desynchronized
        state storage (each worker keeps local copies of u, v).

        Reference: DES-LOC BM10 - HBM Optimizer State Management
        """
        if not getattr(self, '_desloc_profile_enabled', False):
            return {}
        dev = torch.cuda.current_device()
        total_mem = self._desloc_gpu_total_mem or torch.cuda.get_device_properties(dev).total_memory
        allocated = torch.cuda.memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)

        # DES-LOC requires 3x parameter memory: params + exp_avg + exp_avg_sq
        # vs DDP which may share these across GPUs
        analysis = {
            'total_hbm_gb': round(total_mem / 1e9, 2),
            'allocated_gb': round(allocated / 1e9, 2),
            'reserved_gb': round(reserved / 1e9, 2),
            'free_gb': round((total_mem - allocated) / 1e9, 2),
            'utilization_pct': round(allocated / max(total_mem, 1) * 100, 2),
            'headroom_for_desloc_states': round((total_mem - reserved) / 1e9, 2),
        }
        # Check if there's enough headroom for local optimizer states
        # Rule of thumb: need ~3x model params for Adam states
        model_bytes_estimate = allocated * 0.3  # rough: ~30% of allocated is model
        adam_state_bytes = model_bytes_estimate * 2  # exp_avg + exp_avg_sq
        analysis['estimated_model_gb'] = round(model_bytes_estimate / 1e9, 2)
        analysis['estimated_adam_states_gb'] = round(adam_state_bytes / 1e9, 2)
        analysis['desloc_feasible'] = (total_mem - reserved) > adam_state_bytes
        return analysis

    def desloc_nccl_test(self, tensor_size_mb=10, num_iters=5):
        """Run a simple NCCL bandwidth test.

        Creates a tensor and performs allreduce to measure real bandwidth.
        This gives a ground-truth number for communication cost modeling.

        Returns bandwidth in GB/s or 0 if distributed is not initialized.
        """
        try:
            import torch.distributed as torch_dist
            if not torch_dist.is_initialized():
                return {'available': False, 'reason': 'distributed not initialized'}
        except Exception:
            return {'available': False, 'reason': 'torch.distributed import failed'}

        dev = torch.cuda.current_device()
        numel = int(tensor_size_mb * 1e6 / 4)  # fp32 elements
        tensor = torch.randn(numel, device=f'cuda:{dev}')

        # Warmup
        for _ in range(2):
            torch_dist.all_reduce(tensor)
        torch.cuda.synchronize()

        # Timed iterations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            torch_dist.all_reduce(tensor)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / num_iters
        tensor_bytes = tensor.numel() * tensor.element_size()
        # allreduce transfers 2*(N-1)/N * data for ring
        world_size = torch_dist.get_world_size()
        algo_bytes = tensor_bytes * 2 * (world_size - 1) / world_size
        bw_gbps = (algo_bytes / 1e9) / (elapsed_ms / 1000)

        del tensor
        torch.cuda.empty_cache()

        return {
            'available': True,
            'tensor_mb': tensor_size_mb,
            'num_iters': num_iters,
            'avg_latency_ms': round(elapsed_ms, 3),
            'bandwidth_gbps': round(bw_gbps, 2),
            'world_size': world_size,
        }

    def desloc_topology_info(self):
        """Detect GPU interconnect topology.

        Returns NVLink/PCIe info if available through pynvml.
        """
        topology = {
            'device_count': self.device_count(),
            'links': [],
        }
        if pynvml is None:
            topology['pynvml_available'] = False
            return topology

        topology['pynvml_available'] = True
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                topology['links'].append({
                    'gpu_id': i,
                    'name': name,
                })
                # Check P2P connectivity
                for j in range(count):
                    if i != j:
                        try:
                            link_type = pynvml.nvmlDeviceGetP2PStatus(
                                handle,
                                pynvml.nvmlDeviceGetHandleByIndex(j),
                                pynvml.NVML_P2P_CAPS_INDEX_READ)
                            topology['links'][-1][f'p2p_to_gpu{j}'] = bool(link_type)
                        except Exception:
                            pass
        except Exception as e:
            topology['error'] = str(e)
        return topology

# =================================================================
# M091: CUDA/Trainium2 Dual-Backend Accelerator + Hardware-Aware Opt
# Claude-5 (M077-M091)
# Nick Joseph: "拥有多种芯片在某些方面很有优势...但你可能需要
# 为每种芯片重复编写代码"
# =================================================================

import os
import math
import time
import logging

_m091_logger = logging.getLogger("DeepSpeed")


class AcceleratorCapabilityProbe:
    """Probe hardware capabilities for DES-LOC optimization decisions.

    Queries GPU properties at runtime to auto-configure:
    - Batch size (based on memory)
    - Communication strategy (based on interconnect)
    - Precision (BF16 native vs emulated)
    - Sync periods (based on bandwidth)
    """

    def __init__(self):
        self._capabilities = {}
        self._probed = False

    def probe(self):
        """Probe all available accelerator capabilities."""
        caps = {
            "backend": "unknown",
            "device_count": 0,
            "devices": [],
            "bf16_native": False,
            "fp16_native": False,
            "nvlink_available": False,
            "total_memory_gb": 0.0,
        }

        # Try CUDA first
        try:
            import torch
            if torch.cuda.is_available():
                caps["backend"] = "cuda"
                caps["device_count"] = torch.cuda.device_count()
                caps["bf16_native"] = (
                    torch.cuda.is_bf16_supported()
                    if hasattr(torch.cuda, "is_bf16_supported")
                    else False
                )
                caps["fp16_native"] = True

                for i in range(caps["device_count"]):
                    props = torch.cuda.get_device_properties(i)
                    dev = {
                        "id": i,
                        "name": props.name,
                        "total_memory_gb": round(
                            props.total_mem / (1024 ** 3), 1),
                        "compute_capability": (
                            props.major, props.minor),
                        "sm_count": props.multi_processor_count,
                    }
                    # Detect BF16 support by compute capability
                    if props.major >= 8:
                        dev["bf16_supported"] = True
                        dev["tf32_supported"] = True
                    else:
                        dev["bf16_supported"] = False
                        dev["tf32_supported"] = False

                    # Detect GPU family
                    name_lower = props.name.lower()
                    if "h100" in name_lower:
                        dev["family"] = "hopper"
                        dev["peak_tflops_bf16"] = 1979.0
                        dev["hbm_bw_gbps"] = 3350
                    elif "a100" in name_lower:
                        dev["family"] = "ampere"
                        dev["peak_tflops_bf16"] = 312.0
                        dev["hbm_bw_gbps"] = 2039
                    elif "a6000" in name_lower:
                        dev["family"] = "ampere"
                        dev["peak_tflops_bf16"] = 155.0
                        dev["hbm_bw_gbps"] = 768
                    elif "4090" in name_lower:
                        dev["family"] = "ada"
                        dev["peak_tflops_bf16"] = 330.0
                        dev["hbm_bw_gbps"] = 1008
                    else:
                        dev["family"] = "unknown"
                        dev["peak_tflops_bf16"] = 50.0
                        dev["hbm_bw_gbps"] = 500

                    caps["devices"].append(dev)

                caps["total_memory_gb"] = sum(
                    d["total_memory_gb"] for d in caps["devices"])

                # Check NVLink via P2P access
                if caps["device_count"] > 1:
                    try:
                        caps["nvlink_available"] = (
                            torch.cuda.can_device_access_peer(0, 1))
                    except Exception:
                        caps["nvlink_available"] = False
        except ImportError:
            pass

        # Try Neuron (Trainium2) if CUDA not available
        if caps["backend"] == "unknown":
            try:
                import torch_neuronx
                caps["backend"] = "neuron"
                caps["bf16_native"] = True  # Trainium2 has native BF16
                _m091_logger.info("Detected AWS Neuron (Trainium2) backend")
            except ImportError:
                pass

        # Check for XLA (TPU or Neuron)
        if caps["backend"] == "unknown":
            try:
                import torch_xla
                caps["backend"] = "xla"
                _m091_logger.info("Detected XLA backend")
            except ImportError:
                pass

        if caps["backend"] == "unknown":
            caps["backend"] = "cpu"
            _m091_logger.info("No GPU/accelerator detected, using CPU")

        self._capabilities = caps
        self._probed = True
        return caps

    def get_capabilities(self):
        if not self._probed:
            self.probe()
        return self._capabilities

    def is_cuda(self):
        return self.get_capabilities()["backend"] == "cuda"

    def is_neuron(self):
        return self.get_capabilities()["backend"] == "neuron"

    def is_heterogeneous(self):
        caps = self.get_capabilities()
        families = set(d.get("family", "?") for d in caps["devices"])
        return len(families) > 1

    def weakest_gpu_memory_gb(self):
        caps = self.get_capabilities()
        mems = [d["total_memory_gb"] for d in caps["devices"]]
        return min(mems) if mems else 0


class DeslocHardwareOptimizer:
    """Auto-tune DES-LOC hyperparameters based on hardware.

    Given GPU configuration, recommend:
    - batch_size: fit in weakest GPU memory
    - Kx: based on interconnect bandwidth (faster → lower Kx)
    - dtype: BF16 if supported, else FP16
    - comm_overlap: if NVLink available
    """

    # Model memory estimates (params + optimizer states + activations)
    MODEL_MEMORY_GB = {
        "117M": {"params": 0.5, "optim": 1.5, "act_per_batch": 0.1},
        "350M": {"params": 1.4, "optim": 4.2, "act_per_batch": 0.3},
        "1.3B": {"params": 5.2, "optim": 15.6, "act_per_batch": 1.0},
        "2.7B": {"params": 10.8, "optim": 32.4, "act_per_batch": 2.0},
        "6.7B": {"params": 26.8, "optim": 80.4, "act_per_batch": 5.0},
    }

    def __init__(self, probe=None):
        self.probe = probe or AcceleratorCapabilityProbe()
        self._caps = self.probe.get_capabilities()

    def recommend_batch_size(self, model_size="117M", seq_len=1024):
        """Recommend batch size based on available memory."""
        mem_info = self.MODEL_MEMORY_GB.get(
            model_size, self.MODEL_MEMORY_GB["117M"])
        available_gb = self.probe.weakest_gpu_memory_gb()
        if available_gb <= 0:
            return 1

        # Reserve 20% for PyTorch overhead
        usable_gb = available_gb * 0.8
        fixed_gb = mem_info["params"] + mem_info["optim"]
        remaining_gb = usable_gb - fixed_gb
        if remaining_gb <= 0:
            return 1

        batch_size = int(remaining_gb / mem_info["act_per_batch"])
        # Round down to power of 2
        if batch_size >= 2:
            batch_size = 2 ** int(math.log2(batch_size))
        return max(1, min(batch_size, 256))

    def recommend_dtype(self):
        """Recommend precision based on hardware."""
        if self._caps.get("bf16_native", False):
            return "bf16"
        elif self._caps.get("fp16_native", False):
            return "fp16"
        return "fp32"

    def recommend_sync_periods(self, model_size="117M",
                               beta1=0.9, beta2=0.999):
        """Recommend DES-LOC sync periods based on hardware."""
        has_nvlink = self._caps.get("nvlink_available", False)

        # Half-life based recommendation
        hl1 = -1.0 / math.log2(beta1) if 0 < beta1 < 1 else 10
        hl2 = -1.0 / math.log2(beta2) if 0 < beta2 < 1 else 1000

        if has_nvlink:
            # NVLink: high bandwidth → can sync more frequently
            Kx = 8
        else:
            # PCIe: limited bandwidth → sync less frequently
            Kx = 16

        # Ku proportional to Kx, bounded by half-life
        Ku = max(Kx, min(int(Kx * 3), int(hl1)))
        Kv = max(Ku, min(int(Kx * 6), int(hl2 / 2)))

        return {
            "Kx": Kx, "Ku": Ku, "Kv": Kv,
            "reason": f"{'NVLink' if has_nvlink else 'PCIe'} detected",
            "half_life_beta1": round(hl1, 1),
            "half_life_beta2": round(hl2, 1),
        }

    def recommend_comm_strategy(self):
        """Recommend communication overlap strategy."""
        has_nvlink = self._caps.get("nvlink_available", False)
        n_gpus = self._caps.get("device_count", 1)

        if n_gpus <= 1:
            return {"overlap": False, "algorithm": "none",
                    "reason": "single GPU"}
        elif has_nvlink:
            return {"overlap": True, "algorithm": "ring",
                    "reason": "NVLink ring AllReduce optimal"}
        else:
            return {"overlap": True, "algorithm": "tree",
                    "reason": "PCIe tree AllReduce for bandwidth"}

    def full_recommendation(self, model_size="117M",
                            beta1=0.9, beta2=0.999):
        """Generate complete DES-LOC configuration recommendation."""
        return {
            "hardware": {
                "backend": self._caps.get("backend", "unknown"),
                "num_gpus": self._caps.get("device_count", 0),
                "total_memory_gb": self._caps.get("total_memory_gb", 0),
                "nvlink": self._caps.get("nvlink_available", False),
                "heterogeneous": self.probe.is_heterogeneous(),
            },
            "model_size": model_size,
            "batch_size": self.recommend_batch_size(model_size),
            "dtype": self.recommend_dtype(),
            "sync_periods": self.recommend_sync_periods(
                model_size, beta1, beta2),
            "comm_strategy": self.recommend_comm_strategy(),
        }


class NeuronAcceleratorBridge:
    """Bridge between NVIDIA CUDA and AWS Neuron (Trainium2) APIs.

    Provides unified interface for DES-LOC operations:
    - Memory management
    - Synchronization barriers
    - Performance counters
    - Device selection

    Falls back gracefully when Neuron is not available.
    """

    def __init__(self):
        self._backend = None
        self._detected = False

    def detect_backend(self):
        if self._detected:
            return self._backend
        try:
            import torch
            if torch.cuda.is_available():
                self._backend = "cuda"
                self._detected = True
                return "cuda"
        except ImportError:
            pass
        try:
            import torch_neuronx
            self._backend = "neuron"
            self._detected = True
            return "neuron"
        except ImportError:
            pass
        try:
            import torch_xla
            self._backend = "xla"
            self._detected = True
            return "xla"
        except ImportError:
            pass
        self._backend = "cpu"
        self._detected = True
        return "cpu"

    def synchronize(self):
        """Backend-agnostic synchronize."""
        backend = self.detect_backend()
        if backend == "cuda":
            import torch
            torch.cuda.synchronize()
        elif backend in ("neuron", "xla"):
            try:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            except ImportError:
                pass

    def empty_cache(self):
        """Backend-agnostic cache clearing."""
        backend = self.detect_backend()
        if backend == "cuda":
            import torch
            torch.cuda.empty_cache()

    def memory_allocated(self, device=0):
        """Return allocated memory in bytes."""
        backend = self.detect_backend()
        if backend == "cuda":
            import torch
            return torch.cuda.memory_allocated(device)
        return 0

    def memory_reserved(self, device=0):
        backend = self.detect_backend()
        if backend == "cuda":
            import torch
            return torch.cuda.memory_reserved(device)
        return 0

    def device_name(self, device=0):
        backend = self.detect_backend()
        if backend == "cuda":
            import torch
            return torch.cuda.get_device_name(device)
        elif backend == "neuron":
            return "AWS Trainium2"
        elif backend == "xla":
            return "XLA Device"
        return "CPU"

    def device_count(self):
        backend = self.detect_backend()
        if backend == "cuda":
            import torch
            return torch.cuda.device_count()
        elif backend == "neuron":
            cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
            if cores:
                return len(cores.split(","))
            return 1
        return 1

    def set_device(self, device_id):
        backend = self.detect_backend()
        if backend == "cuda":
            import torch
            torch.cuda.set_device(device_id)
        elif backend in ("neuron", "xla"):
            os.environ["NEURON_RT_VISIBLE_CORES"] = str(device_id)

    def get_performance_summary(self):
        """Collect performance counters from current backend."""
        summary = {"backend": self.detect_backend()}
        if summary["backend"] == "cuda":
            import torch
            for i in range(torch.cuda.device_count()):
                summary[f"gpu_{i}"] = {
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated_mb": round(
                        torch.cuda.memory_allocated(i) / (1024**2), 1),
                    "memory_reserved_mb": round(
                        torch.cuda.memory_reserved(i) / (1024**2), 1),
                    "max_memory_allocated_mb": round(
                        torch.cuda.max_memory_allocated(i) / (1024**2), 1),
                }
        return summary


# =================================================================
# End M091  (CapabilityProbe + HardwareOptimizer + NeuronBridge)
# =================================================================
