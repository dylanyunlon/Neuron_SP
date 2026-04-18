# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
import os
import pkgutil
import importlib
import sys

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


# =========================================================================
# DES-LOC CUDA Accelerator Extensions
# Ref: Nick Joseph — 'understand physical layout of hardware'
# =========================================================================

class DeslocGPUCapabilityProbe:
    """Probe GPU capabilities for DES-LOC Kx selection.
    Key factors: compute capability, memory size, interconnect bandwidth."""

    @staticmethod
    def probe_all():
        try:
            import torch
            if not torch.cuda.is_available():
                return []
            return [{
                'idx': i,
                'name': torch.cuda.get_device_properties(i).name,
                'cc': f'{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}',
                'mem_gb': round(torch.cuda.get_device_properties(i).total_mem / 1e9, 2),
                'sms': torch.cuda.get_device_properties(i).multi_processor_count,
                'hopper': torch.cuda.get_device_properties(i).major >= 9,
                'bf16': torch.cuda.get_device_properties(i).major >= 8,
            } for i in range(torch.cuda.device_count())]
        except Exception:
            return []

    @staticmethod
    def estimate_peak_tflops(info):
        name = info.get('name', '').upper()
        if 'H100' in name or 'H200' in name:
            return 312.0
        elif 'A100' in name:
            return 312.0
        elif 'A6000' in name:
            return 77.4
        elif '4090' in name:
            return 165.2
        return info.get('sms', 0) * 0.5

    @staticmethod
    def recommend_Kx(info, model_params, batch_size=4, seq_len=512):
        import math
        peak = DeslocGPUCapabilityProbe.estimate_peak_tflops(info)
        if peak <= 0:
            return 32
        flops = 6 * model_params * seq_len * batch_size
        compute_s = flops / (peak * 1e12)
        param_bytes = model_params * 2
        bw = 32e9  # conservative PCIe estimate
        ar_s = 2 * param_bytes / bw
        if ar_s <= compute_s:
            return 1
        return min(256, 2 ** int(math.ceil(math.log2(ar_s / compute_s))))

    @staticmethod
    def detect_heterogeneous():
        devices = DeslocGPUCapabilityProbe.probe_all()
        if len(devices) <= 1:
            return {'hetero': False}
        names = set(d['name'] for d in devices)
        mems = set(d['mem_gb'] for d in devices)
        return {
            'hetero': len(names) > 1 or (max(mems)/min(mems) > 1.5 if mems else False),
            'gpus': list(names),
            'mem_range': [min(mems), max(mems)] if mems else [],
        }


class DeslocDeviceMapper:
    """Map DES-LOC tiers to GPU devices in heterogeneous clusters.
    Faster GPUs get more frequent sync (lower effective Kx).
    Slower GPUs get less frequent sync (higher effective Kx).
    Ref: Nick Joseph — 'some chips have lots of FLOPS but not much memory'"""

    def __init__(self, device_infos):
        self.devices = device_infos
        self.device_Kx = {}

    def compute_per_device_Kx(self, base_Kx, model_params):
        """Compute per-device Kx scaled by relative compute capability."""
        if not self.devices:
            return {}
        peak_tflops = [DeslocGPUCapabilityProbe.estimate_peak_tflops(d) for d in self.devices]
        max_peak = max(peak_tflops) if peak_tflops else 1
        for i, d in enumerate(self.devices):
            ratio = max_peak / max(0.1, peak_tflops[i])
            self.device_Kx[i] = max(1, int(round(base_Kx * ratio)))
        return dict(self.device_Kx)


class DeslocDeviceMapper:
    """Map DES-LOC tiers to devices in heterogeneous clusters.
    Faster GPUs get more frequent sync (lower effective Kx).
    Ref: Nick Joseph — 'some chips have lots of FLOPS but not much memory'."""

    def __init__(self, device_infos=None):
        self.devices = device_infos or []
        self.device_Kx = {}

    def compute_per_device_Kx(self, base_Kx, model_params=0):
        if not self.devices:
            return {}
        peaks = []
        for d in self.devices:
            name = d.get('name', '').upper()
            if 'H100' in name: peaks.append(312)
            elif 'A100' in name: peaks.append(312)
            elif 'A6000' in name: peaks.append(77)
            else: peaks.append(d.get('sms', 50) * 0.5)
        max_peak = max(peaks) if peaks else 1
        for i, p in enumerate(peaks):
            import math
            ratio = max_peak / max(0.1, p)
            self.device_Kx[i] = max(1, int(round(base_Kx * ratio)))
        return dict(self.device_Kx)


class DeslocHardwareProfiler:
    """Comprehensive hardware profiler for DES-LOC config selection.
    Measures: compute throughput, memory bandwidth, interconnect BW.
    Recommends optimal Kx, Ku, Kv based on hardware profile.

    Ref: Nick Joseph — 'six or seven bottleneck constraints'."""

    def __init__(self):
        self.compute_tflops = 0
        self.memory_bw_gbps = 0
        self.network_bw_gbps = 0
        self.device_count = 0

    def probe(self):
        """Run hardware profiling."""
        try:
            import torch
            if not torch.cuda.is_available():
                return self
            self.device_count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            name = props.name.upper()
            # Estimate compute
            if 'H100' in name: self.compute_tflops = 312
            elif 'A100' in name: self.compute_tflops = 312
            elif 'A6000' in name: self.compute_tflops = 77
            else: self.compute_tflops = props.multi_processor_count * 0.5
            # Estimate memory BW
            if 'H100' in name: self.memory_bw_gbps = 3350
            elif 'A100' in name: self.memory_bw_gbps = 2039
            else: self.memory_bw_gbps = 768
            # Estimate network BW
            has_nvlink = any(g in name for g in ('H100', 'A100', 'H200'))
            self.network_bw_gbps = 600 if has_nvlink else 32
        except Exception:
            pass
        return self

    def recommend_config(self, model_params, batch_size=4, seq_len=512):
        """Recommend DES-LOC config based on hardware profile."""
        import math
        if self.compute_tflops <= 0:
            return {'Kx': 32, 'Ku': 96, 'Kv': 192}
        flops = 6 * model_params * seq_len * batch_size
        compute_s = flops / (self.compute_tflops * 1e12)
        if self.network_bw_gbps <= 0 or self.device_count <= 1:
            return {'Kx': 1, 'Ku': 1, 'Kv': 1}
        ring = 2.0 * (self.device_count - 1) / self.device_count
        ar_s = ring * model_params * 2 / (self.network_bw_gbps * 1e9)
        if ar_s <= compute_s:
            kx = 1
        else:
            kx = 2 ** int(math.ceil(math.log2(ar_s / compute_s)))
            kx = min(kx, 256)
        return {'Kx': kx, 'Ku': max(1, kx * 3), 'Kv': max(1, kx * 6),
                'compute_ms': round(compute_s * 1000, 4),
                'comm_ms': round(ar_s * 1000, 4),
                'bottleneck': 'comm' if ar_s > compute_s else 'compute'}

    def summary(self):
        return {
            'compute_tflops': self.compute_tflops,
            'memory_bw_gbps': self.memory_bw_gbps,
            'network_bw_gbps': self.network_bw_gbps,
            'device_count': self.device_count,
        }


class DeslocNVLinkDetector:
    """Detect NVLink topology for intra-node DES-LOC optimization.
    NVLink provides ~600 GB/s — much higher than PCIe ~32 GB/s.
    With NVLink, intra-node allreduce is nearly free, so DES-LOC
    primarily benefits inter-node (EFA/InfiniBand) communication.

    Strategy: use Kx=1 for intra-node, Kx>1 for inter-node.
    Ref: Megatron — separate intra/inter-node parallelism groups."""

    @staticmethod
    def detect_nvlink():
        try:
            import torch
            if not torch.cuda.is_available():
                return {'has_nvlink': False}
            name = torch.cuda.get_device_name(0).upper()
            has = any(g in name for g in ('H100', 'A100', 'H200', 'GH200'))
            count = torch.cuda.device_count()
            return {
                'has_nvlink': has,
                'device_count': count,
                'device_name': name,
                'estimated_bw_gbps': 600 if has else 32,
                'intra_node_kx_recommendation': 1 if has else 4,
            }
        except Exception:
            return {'has_nvlink': False}

    @staticmethod
    def recommend_hierarchical_Kx(intra_bw_gbps, inter_bw_gbps, model_params,
                                   compute_time_s):
        """Recommend separate Kx for intra/inter-node communication.
        Returns dict with intra_Kx and inter_Kx."""
        import math
        param_bytes = model_params * 2
        intra_ar = 2 * param_bytes / (intra_bw_gbps * 1e9) if intra_bw_gbps > 0 else 0
        inter_ar = 2 * param_bytes / (inter_bw_gbps * 1e9) if inter_bw_gbps > 0 else 0
        intra_kx = 1 if intra_ar <= compute_time_s else min(16, 2 ** int(math.ceil(math.log2(intra_ar / compute_time_s))))
        inter_kx = 1 if inter_ar <= compute_time_s else min(256, 2 ** int(math.ceil(math.log2(inter_ar / compute_time_s))))
        return {'intra_Kx': intra_kx, 'inter_Kx': inter_kx,
                'intra_ar_ms': round(intra_ar * 1000, 4),
                'inter_ar_ms': round(inter_ar * 1000, 4)}
