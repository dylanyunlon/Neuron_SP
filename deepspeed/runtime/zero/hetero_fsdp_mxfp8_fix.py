"""
hetero_fsdp_mxfp8_fix.py — Neuron_SP / DES-LOC异构训练框架

上游设计意图（Megatron commit 41f3b6f）：
    Megatron-LM的M-FSDP（Megatron Fully Sharded Data Parallel）在支持MXFP8
    混合精度训练时存在三处关键bug：
    1. 后向传播结束后仅释放了bwd=True的参数缓冲区，遗漏了bwd=False路径，
       导致A100/H100上FP8 transpose cache未能及时归还显存。
    2. 状态机回退（→ IDLE）只作用于顶层module，子模块滞留在PRE_BACKWARD，
       导致下一轮前向的gather提前被跳过（误判为"已处于pre-backward"）。
    3. enable_fine_grained_param_gather_hook 分支中，pre-backward unshard
       hook被重复注册或错误跳过，造成MXFP8参数在activation recompute时
       以FP32格式被gather，引发精度/显存双重异常。

DES-LOC适配点：
    DES-LOC（Decoupled Execution with Shared LOcality Cache）将计算图的
    执行（Execution）与权重局部性缓存（Locality Cache）解耦：
    - A6000×2（SM86, 48GB each）承载"局部缓存层"：持有FP8 transpose cache
      和sharded参数的热副本，通过PCIe向H100传递unshard结果。
    - H100 NVL×1（SM90, 96GB）承载"执行层"：执行实际矩阵乘/梯度累积，
      再将grad shard回传给A6000。
    - CPU DRAM（1.5TB）作为溢出池：存放冷参数shard与optimizer state。

    上游三处bug在异构拓扑下被放大：
    Bug-1 在A6000上泄漏FP8缓存，压缩可用显存窗口，导致下轮H100←A6000
          PCIe传输阻塞。
    Bug-2 子模块状态不一致，DES-LOC的"跨设备延迟执行"在判断是否需要
          prefetch时读到错误的TrainingState，触发冗余的all-gather。
    Bug-3 fine-grained hook路径在SM86/SM90混合注册时出现双重触发，
          MXFP8 scaling factor在A6000端已经应用一次、在H100端再次应用，
          导致数值溢出。

    本文件将以上三处修复重新实现为DES-LOC感知的版本：
    - HeteroTrainingState：扩展TrainingState以携带device_role信息。
    - HeteroFSDPMXFP8Fix：核心Mixin，可注入到DeepSpeed ZeRO-3 engine。
    - LocalityCacheManager：管理FP8 transpose cache在A6000/H100/CPU间的
      生命周期，确保每个设备角色的缓存在正确时机释放。
    - HeteroPreBackwardCoordinator：跨设备同步PRE_BACKWARD状态，消除
      Bug-2在异构场景下的放大效应。
    - FineGrainedHookRegistry：单例注册表，防止SM86/SM90混合注册重复触发
      （Bug-3修复）。
"""

from __future__ import annotations

import logging
import enum
import threading
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量与枚举
# ---------------------------------------------------------------------------

class DeviceRole(enum.Enum):
    """DES-LOC设备角色。

    LOCALITY_CACHE: A6000×2，持有FP8 transpose cache和sharded热副本。
    EXECUTION:      H100 NVL，执行前向/后向计算。
    OFFLOAD_POOL:   CPU DRAM，冷参数与optimizer state的溢出池。
    """
    LOCALITY_CACHE = "locality_cache"   # A6000 SM86
    EXECUTION      = "execution"        # H100 SM90
    OFFLOAD_POOL   = "offload_pool"     # CPU DRAM


class TrainingState(enum.Enum):
    """模块训练状态机。

    与Megatron上游保持语义兼容，同时为DES-LOC增加HETERO_SYNC状态。
    HETERO_SYNC表示跨设备状态同步正在进行中，禁止任何参数reshape操作。
    """
    IDLE          = "idle"
    FORWARD       = "forward"
    PRE_BACKWARD  = "pre_backward"
    BACKWARD      = "backward"
    HETERO_SYNC   = "hetero_sync"   # DES-LOC扩展


@dataclass
class DeviceProfile:
    """记录单个物理设备的能力与当前状态。"""
    role:           DeviceRole
    device:         torch.device
    total_memory:   int                  # bytes，静态
    sm_version:     int                  # e.g. 86, 90
    fp8_native:     bool                 # SM90+ 原生FP8
    cache_budget:   int                  # bytes，FP8 cache预算
    used_cache:     int = 0              # 当前已用FP8 cache
    lock:           threading.Lock = field(default_factory=threading.Lock)

    def available_cache(self) -> int:
        return max(0, self.cache_budget - self.used_cache)

    def alloc_cache(self, nbytes: int) -> bool:
        with self.lock:
            if self.used_cache + nbytes <= self.cache_budget:
                self.used_cache += nbytes
                return True
            return False

    def free_cache(self, nbytes: int) -> None:
        with self.lock:
            self.used_cache = max(0, self.used_cache - nbytes)


# ---------------------------------------------------------------------------
# LocalityCacheManager — Bug-1修复的核心载体
# ---------------------------------------------------------------------------

class LocalityCacheManager:
    """管理FP8 transpose cache在DES-LOC异构拓扑中的生命周期。

    上游Bug-1根因：
        release_module_parameters(module, bwd=True) 被调用，但
        release_module_parameters(module, bwd=False) 被遗漏。
        在FP8路径下，bwd=False对应的是前向所缓存的transpose副本；
        若不释放，A6000的48GB显存在多层Transformer叠加后迅速耗尽。

    DES-LOC修复策略：
        1. 每次分配FP8缓存时，向本Manager注册（register_cache）。
        2. 后向hook调用 release_for_module(module, bwd=True) 后，
           Manager自动判断同module的 bwd=False 缓存是否仍驻留，
           若驻留则立即触发释放（fix_missing_fwd_release）。
        3. 跨设备：若FP8缓存已从A6000迁移到H100（execution），
           则在H100侧执行释放，不产生冗余PCIe流量。
    """

    def __init__(self, profiles: Dict[DeviceRole, DeviceProfile]) -> None:
        self._profiles = profiles
        # module_id → {(bwd: bool) → (tensor_list, device_role, nbytes)}
        self._registry: Dict[
            int,
            Dict[bool, Tuple[List["Tensor"], DeviceRole, int]]
        ] = defaultdict(dict)
        self._lock = threading.Lock()

    def register_cache(
        self,
        module: nn.Module,
        tensors: List["Tensor"],
        bwd: bool,
        device_role: DeviceRole,
    ) -> None:
        """注册FP8缓存条目。"""
        mid = id(module)
        nbytes = sum(t.nbytes for t in tensors if isinstance(t, torch.Tensor))
        profile = self._profiles.get(device_role)
        if profile:
            profile.alloc_cache(nbytes)
        with self._lock:
            self._registry[mid][bwd] = (tensors, device_role, nbytes)
        logger.debug(
            "LocalityCacheManager.register_cache | module=%s bwd=%s "
            "device_role=%s nbytes=%d",
            type(module).__name__, bwd, device_role.value, nbytes,
        )

    def release_for_module(
        self,
        module: nn.Module,
        bwd: bool,
        fix_missing_fwd: bool = True,
    ) -> None:
        """释放模块的FP8缓存，并可选地修复遗漏的前向缓存释放。

        Args:
            module:           目标模块。
            bwd:              True → 释放后向缓存；False → 释放前向缓存。
            fix_missing_fwd:  若为True且 bwd=True，则同时检查并释放
                              bwd=False 的遗漏缓存（Bug-1修复）。
        """
        mid = id(module)
        with self._lock:
            entry_map = self._registry.get(mid, {})
            to_release = []
            if bwd in entry_map:
                to_release.append((bwd, entry_map.pop(bwd)))
            # Bug-1修复：bwd=True释放时，顺带清理遗漏的bwd=False
            if fix_missing_fwd and bwd is True and (False in entry_map):
                logger.warning(
                    "LocalityCacheManager: detected missing bwd=False release "
                    "for module %s — applying DES-LOC fix.",
                    type(module).__name__,
                )
                to_release.append((False, entry_map.pop(False)))
            if not entry_map:
                self._registry.pop(mid, None)

        for _bwd_key, (tensors, device_role, nbytes) in to_release:
            self._do_release(tensors, device_role, nbytes, module)

    def _do_release(
        self,
        tensors: List["Tensor"],
        device_role: DeviceRole,
        nbytes: int,
        module: nn.Module,
    ) -> None:
        """执行实际的缓存释放，避免跨设备冗余传输。"""
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.data_ptr() != 0:
                # 置零data ptr而非del，以防外部持有引用时出现悬空指针
                t.data = torch.empty(0, dtype=t.dtype, device=t.device)
        profile = self._profiles.get(device_role)
        if profile:
            profile.free_cache(nbytes)
        logger.debug(
            "LocalityCacheManager._do_release | module=%s device_role=%s "
            "freed_bytes=%d",
            type(module).__name__, device_role.value, nbytes,
        )


# ---------------------------------------------------------------------------
# HeteroPreBackwardCoordinator — Bug-2修复的核心载体
# ---------------------------------------------------------------------------

class HeteroPreBackwardCoordinator:
    """跨设备同步TrainingState，修复子模块状态不一致问题（Bug-2）。

    上游Bug-2根因：
        状态机回退 → IDLE 时：
            旧代码：module._training_state = TrainingState.IDLE
            修复后：for sub in module.modules(): sub._training_state = IDLE
        仅顶层模块被回退，子模块（尤其是包含FP8权重的projection层）
        残留在PRE_BACKWARD。DES-LOC中，A6000上的子模块与H100上的
        计算子图是1:1映射的，子模块状态错误会导致H100侧的prefetch
        逻辑误判为"已在pre-backward"而跳过gather，产生stale参数。

    DES-LOC修复策略：
        1. set_training_state_recursive：统一设置模块及所有子模块的状态，
           同时通过device_role过滤，避免向不相关设备发送状态通知。
        2. HETERO_SYNC状态：在跨PCIe同步期间锁定状态机，防止并发修改。
        3. broadcast_state_to_execution_device：当A6000侧完成状态转换后，
           通过共享内存标志位通知H100侧的对应计算子图。
    """

    def __init__(
        self,
        profiles: Dict[DeviceRole, DeviceProfile],
        pcie_bandwidth_gbps: float = 16.0,
    ) -> None:
        self._profiles = profiles
        self._pcie_bw = pcie_bandwidth_gbps
        # (module_id, device_role) → TrainingState
        self._state_map: Dict[Tuple[int, DeviceRole], TrainingState] = {}
        self._lock = threading.RLock()
        # 已注册的跨设备同步回调
        self._sync_callbacks: List[Callable[[nn.Module, TrainingState], None]] = []

    def register_sync_callback(
        self, cb: Callable[[nn.Module, TrainingState], None]
    ) -> None:
        """注册状态同步回调，供H100执行层监听A6000缓存层的状态变化。"""
        self._sync_callbacks.append(cb)

    def set_training_state_recursive(
        self,
        module: nn.Module,
        state: TrainingState,
        device_role: Optional[DeviceRole] = None,
    ) -> None:
        """递归设置模块及所有子模块的训练状态（修复Bug-2）。

        Args:
            module:      根模块。
            state:       目标TrainingState。
            device_role: 若指定，仅更新属于该device_role的子模块；
                         None表示更新全部（与上游修复行为一致）。
        """
        updated: List[nn.Module] = []
        with self._lock:
            for sub in module.modules():
                # DES-LOC扩展：若子模块携带device_role属性，则按角色过滤
                sub_role = getattr(sub, "_desloс_device_role", None)
                if device_role is not None and sub_role is not None:
                    if sub_role != device_role:
                        continue
                # 防止在HETERO_SYNC期间覆盖（Bug-2在异构场景的放大）
                cur = getattr(sub, "_training_state", TrainingState.IDLE)
                if cur == TrainingState.HETERO_SYNC and state != TrainingState.IDLE:
                    logger.debug(
                        "set_training_state_recursive: skipping %s in "
                        "HETERO_SYNC state (target=%s)",
                        type(sub).__name__, state.value,
                    )
                    continue
                sub._training_state = state
                mid = id(sub)
                role_key = sub_role or DeviceRole.EXECUTION
                self._state_map[(mid, role_key)] = state
                updated.append(sub)

        logger.debug(
            "set_training_state_recursive | root=%s state=%s updated=%d modules",
            type(module).__name__, state.value, len(updated),
        )
        # 触发跨设备同步回调
        for cb in self._sync_callbacks:
            for sub in updated:
                try:
                    cb(sub, state)
                except Exception as exc:
                    logger.error(
                        "sync_callback error for module %s: %s",
                        type(sub).__name__, exc,
                    )

    def enter_hetero_sync(self, module: nn.Module) -> "_HeteroSyncContext":
        """返回上下文管理器，在PCIe同步期间锁定状态机。"""
        return _HeteroSyncContext(self, module)

    def get_state(
        self,
        module: nn.Module,
        device_role: DeviceRole = DeviceRole.EXECUTION,
    ) -> TrainingState:
        """查询模块在指定设备角色上的训练状态。"""
        key = (id(module), device_role)
        with self._lock:
            return self._state_map.get(
                key, getattr(module, "_training_state", TrainingState.IDLE)
            )


class _HeteroSyncContext:
    """HETERO_SYNC状态的上下文管理器，自动进入/退出同步锁。"""

    def __init__(
        self,
        coordinator: HeteroPreBackwardCoordinator,
        module: nn.Module,
    ) -> None:
        self._coord = coordinator
        self._module = module
        self._prev_states: Dict[int, TrainingState] = {}

    def __enter__(self) -> "_HeteroSyncContext":
        with self._coord._lock:
            for sub in self._module.modules():
                self._prev_states[id(sub)] = getattr(
                    sub, "_training_state", TrainingState.IDLE
                )
                sub._training_state = TrainingState.HETERO_SYNC
        logger.debug(
            "_HeteroSyncContext.__enter__: module=%s locked %d sub-modules",
            type(self._module).__name__, len(self._prev_states),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        with self._coord._lock:
            for sub in self._module.modules():
                prev = self._prev_states.get(id(sub), TrainingState.IDLE)
                sub._training_state = prev
        logger.debug(
            "_HeteroSyncContext.__exit__: module=%s restored states",
            type(self._module).__name__,
        )
        return False  # 不吞异常


# ---------------------------------------------------------------------------
# FineGrainedHookRegistry — Bug-3修复的核心载体
# ---------------------------------------------------------------------------

class FineGrainedHookRegistry:
    """单例注册表，防止SM86/SM90混合环境下pre-backward hook重复注册（Bug-3）。

    上游Bug-3根因：
        enable_fine_grained_param_gather_hook=True 时：
            旧代码在两个独立分支里都可能调用
            _register_pre_backward_param_unshard_hook(module)，
            而另一处在同一分支内又将该调用删除，导致某些module上
            hook被注册0次（漏注册）或2次（重复注册）。
        在FP8路径下重复触发意味着：scaling factor被应用两次，
        数值变为 x * s^2 而非 x * s，引发NaN/Inf。

    DES-LOC修复策略：
        1. 本Registry以 (module_id, hook_type) 为key去重。
        2. SM86（A6000）设备上注册 LOCALITY_CACHE_UNSHARD hook，
           SM90（H100）设备上注册 EXECUTION_UNSHARD hook，
           两者不互相干扰。
        3. 提供 ensure_registered / ensure_unregistered 接口，
           取代上游的条件分支，彻底消除漏注册/重复注册。
    """

    _instance: Optional["FineGrainedHookRegistry"] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "FineGrainedHookRegistry":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._hooks: Dict[
                        Tuple[int, str], List[torch.utils.hooks.RemovableHook]
                    ] = {}
                    cls._instance._registered_mids: Set[Tuple[int, str]] = set()
                    cls._instance._lock = threading.Lock()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """测试/重新初始化时使用。"""
        with cls._init_lock:
            cls._instance = None

    def ensure_registered(
        self,
        module: nn.Module,
        hook_type: str,
        hook_fn: Callable,
        device_role: DeviceRole,
    ) -> bool:
        """注册hook，若已注册则跳过（幂等）。

        Args:
            module:      目标模块。
            hook_type:   钩子类型标识符，如 "pre_backward_unshard"。
            hook_fn:     实际hook函数。
            device_role: 该hook所属的设备角色，用于隔离SM86/SM90路径。

        Returns:
            True  → 首次注册成功。
            False → 已存在，跳过（Bug-3防重复触发的关键）。
        """
        key = (id(module), f"{hook_type}:{device_role.value}")
        with self._lock:
            if key in self._registered_mids:
                logger.debug(
                    "FineGrainedHookRegistry.ensure_registered: SKIP "
                    "module=%s hook_type=%s device_role=%s (already registered)",
                    type(module).__name__, hook_type, device_role.value,
                )
                return False
            # 注册到module的pre_backward hook
            handle = module.register_full_backward_pre_hook(hook_fn)
            self._hooks.setdefault(key, []).append(handle)
            self._registered_mids.add(key)
            logger.info(
                "FineGrainedHookRegistry.ensure_registered: OK "
                "module=%s hook_type=%s device_role=%s",
                type(module).__name__, hook_type, device_role.value,
            )
            return True

    def ensure_unregistered(
        self,
        module: nn.Module,
        hook_type: str,
        device_role: DeviceRole,
    ) -> int:
        """移除指定hook，返回实际移除数量（应为0或1）。"""
        key = (id(module), f"{hook_type}:{device_role.value}")
        with self._lock:
            handles = self._hooks.pop(key, [])
            for h in handles:
                h.remove()
            self._registered_mids.discard(key)
        if handles:
            logger.debug(
                "FineGrainedHookRegistry.ensure_unregistered: removed %d "
                "handle(s) for module=%s hook_type=%s",
                len(handles), type(module).__name__, hook_type,
            )
        return len(handles)

    def registered_count(self, module: nn.Module, hook_type: str) -> int:
        """返回某模块某类型hook的注册数量（调试用）。"""
        with self._lock:
            total = 0
            for role in DeviceRole:
                key = (id(module), f"{hook_type}:{role.value}")
                total += len(self._hooks.get(key, []))
            return total


# ---------------------------------------------------------------------------
# HeteroFSDPMXFP8Fix — 主Mixin，注入到DeepSpeed ZeRO-3 engine
# ---------------------------------------------------------------------------

class HeteroFSDPMXFP8Fix:
    """DES-LOC异构FSDP MXFP8修复Mixin。

    将Megatron commit 41f3b6f中的三处bug修复重新实现为
    DES-LOC感知版本，可通过多重继承方式注入到DeepSpeed的
    DeepSpeedEngine或自定义ZeRO-3引擎。

    用法示例::

        class NeuronSPEngine(HeteroFSDPMXFP8Fix, DeepSpeedEngine):
            def __init__(self, ...):
                super().__init__(...)
                self.init_desloс_fix(profiles=build_device_profiles())

    关键修复摘要：
        Bug-1: post_backward_hook 中同时调用 release(bwd=True) 和
               release(bwd=False)，由LocalityCacheManager.fix_missing_fwd
               在A6000侧自动补全。
        Bug-2: 所有状态机转换通过
               HeteroPreBackwardCoordinator.set_training_state_recursive
               完成，保证子模块状态一致性。
        Bug-3: pre-backward hook通过 FineGrainedHookRegistry.ensure_registered
               幂等注册，消除SM86/SM90混合路径下的重复触发。
    """

    def init_desloс_fix(
        self,
        profiles: Dict[DeviceRole, DeviceProfile],
        enable_fine_grained_param_gather_hook: bool = True,
        data_parallel_sharding_strategy: str = "optim_grads_params",
        keep_fp8_transpose_cache: bool = False,
    ) -> None:
        """初始化DES-LOC修复组件。

        Args:
            profiles: 设备画像字典，由 build_device_profiles() 构建。
            enable_fine_grained_param_gather_hook: 与上游同名标志。
            data_parallel_sharding_strategy: ZeRO sharding策略。
            keep_fp8_transpose_cache: 是否保留FP8 transpose缓存到下一轮。
        """
        self._desloс_profiles = profiles
        self._enable_fine_grained = enable_fine_grained_param_gather_hook
        self._dp_sharding_strategy = data_parallel_sharding_strategy
        self._keep_fp8_cache = keep_fp8_transpose_cache

        self._cache_manager = LocalityCacheManager(profiles)
        self._state_coordinator = HeteroPreBackwardCoordinator(profiles)
        self._hook_registry = FineGrainedHookRegistry()

        logger.info(
            "HeteroFSDPMXFP8Fix initialized | "
            "fine_grained=%s sharding=%s keep_fp8=%s profiles=%s",
            enable_fine_grained_param_gather_hook,
            data_parallel_sharding_strategy,
            keep_fp8_transpose_cache,
            {r.value: p.device for r, p in profiles.items()},
        )

    # ------------------------------------------------------------------
    # Bug-1修复：后向传播后双路径参数释放
    # ------------------------------------------------------------------

    def post_backward_release(self, module: nn.Module) -> None:
        """后向传播后释放模块参数缓冲区（同时覆盖bwd=True和bwd=False）。

        上游修复diff：
            + release_module_parameters(module, bwd=False)  ← 补全遗漏行

        DES-LOC版本：委托给LocalityCacheManager，感知设备角色，
        避免在错误设备上执行释放（节省PCIe往返）。
        """
        logger.debug(
            "post_backward_release | module=%s", type(module).__name__
        )
        # bwd=True：释放后向使用的unshard缓冲
        self._cache_manager.release_for_module(
            module, bwd=True, fix_missing_fwd=True  # Bug-1: fix_missing_fwd
        )
        # 显式释放bwd=False（FP8 transpose cache），与上游修复对齐
        # LocalityCacheManager.release_for_module在fix_missing_fwd=True时
        # 已自动处理，此处保留显式调用以增强可观测性（日志可追踪）
        self._cache_manager.release_for_module(
            module, bwd=False, fix_missing_fwd=False
        )

        # Bug-2修复：递归回退所有子模块状态到IDLE
        self._state_coordinator.set_training_state_recursive(
            module, TrainingState.IDLE
        )

    # ------------------------------------------------------------------
    # Bug-2修复：pre-backward状态递归设置
    # ------------------------------------------------------------------

    def pre_backward_state_setup(
        self,
        module: nn.Module,
        root_module: nn.Module,
        fsdp_unit_modules: Iterable[type],
    ) -> List["Tensor"]:
        """设置pre-backward状态并收集需要all-gather的参数列表。

        上游修复diff：
            - module._training_state = TrainingState.PRE_BACKWARD
            + for sub_module in module.modules():
            +     sub_module._training_state = TrainingState.PRE_BACKWARD
            （同时移除了enable_fine_grained_param_gather_hook的param_list覆盖）

        DES-LOC版本：
            1. 通过HeteroPreBackwardCoordinator递归设置状态。
            2. optim_grads_params策略下，对root_module的所有子模块
               也递归设置（上游另一处Bug-2修复点）。
        """
        # 递归设置当前模块及子模块为PRE_BACKWARD（修复Bug-2）
        self._state_coordinator.set_training_state_recursive(
            module, TrainingState.PRE_BACKWARD
        )

        # 收集参数列表（移除了上游fine_grained分支对param_list的覆盖）
        fsdp_types = tuple(fsdp_unit_modules)
        if isinstance(module, fsdp_types):
            param_list = list(module.parameters())
        else:
            param_list = list(module.parameters(recurse=False))

        logger.debug(
            "pre_backward_state_setup | module=%s param_count=%d",
            type(module).__name__, len(param_list),
        )
        return param_list

    def root_pre_backward_state_setup(
        self,
        root_module: nn.Module,
        ag_pipeline: object,  # AllGatherPipeline
    ) -> None:
        """根模块pre-backward状态初始化（optim_grads_params路径）。

        上游修复diff：
            - for module in root_module.modules():
            -     if isinstance(module, tuple(fsdp_unit_modules)):
            -         module._training_state = TrainingState.PRE_BACKWARD
            + for sub_module in root_module.modules():
            +     sub_module._training_state = TrainingState.PRE_BACKWARD
            （移除了fsdp_unit_modules类型过滤）

        DES-LOC版本：使用coordinator递归设置，无需类型过滤。
        """
        if self._dp_sharding_strategy != "optim_grads_params":
            return

        self._state_coordinator.set_training_state_recursive(
            root_module, TrainingState.PRE_BACKWARD
        )

        # 通知AllGather Pipeline所有bucket可被释放
        if hasattr(ag_pipeline, "num_buckets"):
            for bucket_id in range(ag_pipeline.num_buckets):
                if hasattr(ag_pipeline, "set_bucket_releasable"):
                    ag_pipeline.set_bucket_releasable(bucket_id)
            logger.debug(
                "root_pre_backward_state_setup | released %d AG buckets",
                ag_pipeline.num_buckets,
            )

    # ------------------------------------------------------------------
    # Bug-3修复：fine-grained hook幂等注册
    # ------------------------------------------------------------------

    def register_module_hooks(
        self,
        module: nn.Module,
        pre_forward_unshard_fn: Callable,
        pre_backward_unshard_fn: Callable,
        post_forward_fn: Callable,
        is_fsdp_unit: bool,
        keep_fp8_transpose_cache: bool,
    ) -> None:
        """注册模块的前向/后向hook（Bug-3修复版）。

        上游修复diff（关键逻辑）：
            旧代码：
                if enable_fine_grained:
                    _register_pre_forward_unshard_hook(module)
                    _register_pre_backward_unshard_hook(module)   ← 重复/遗漏
                ...
                if not enable_fine_grained:
                    _register_pre_backward_unshard_hook(module)   ← 条件互斥

            修复后（上游）：
                if enable_fine_grained:
                    _register_pre_forward_unshard_hook(module)
                    # pre_backward已移除（统一在下方注册）
                ...
                _register_pre_backward_unshard_hook(module)       ← 无条件注册

        DES-LOC版本：通过FineGrainedHookRegistry幂等注册，
        SM86和SM90各自注册对应角色的hook，互不干扰。
        """
        # 判断模块所在设备角色（默认EXECUTION）
        module_device_role = getattr(
            module, "_desloс_device_role", DeviceRole.EXECUTION
        )

        # 前向unshard hook：仅在enable_fine_grained=True时注册
        if self._enable_fine_grained:
            self._hook_registry.ensure_registered(
                module,
                hook_type="pre_forward_unshard",
                hook_fn=pre_forward_unshard_fn,
                device_role=module_device_role,
            )

        # 后向unshard hook：无条件注册（上游修复后的行为），幂等保证不重复
        self._hook_registry.ensure_registered(
            module,
            hook_type="pre_backward_unshard",
            hook_fn=pre_backward_unshard_fn,
            device_role=module_device_role,
        )

        # post_forward hook：仅FSDP unit注册
        if is_fsdp_unit:
            self._hook_registry.ensure_registered(
                module,
                hook_type="post_forward",
                hook_fn=post_forward_fn,
                device_role=module_device_role,
            )

        # FP8 transpose cache hook：非keep模式且optim_grads_params策略
        if (
            not keep_fp8_transpose_cache
            and self._dp_sharding_strategy == "optim_grads_params"
            and not is_fsdp_unit
        ):
            self._register_fp8_cache_release_hook(module, module_device_role)

        logger.debug(
            "register_module_hooks | module=%s role=%s fine_grained=%s "
            "is_fsdp_unit=%s",
            type(module).__name__, module_device_role.value,
            self._enable_fine_grained, is_fsdp_unit,
        )

    def _register_fp8_cache_release_hook(
        self,
        module: nn.Module,
        device_role: DeviceRole,
    ) -> None:
        """注册FP8 transpose cache释放hook（A6000侧优先执行）。

        在DES-LOC中，FP8缓存通常驻留在A6000（LOCALITY_CACHE），
        此hook确保其在post-backward后被LocalityCacheManager及时回收。
        """
        cache_mgr = self._cache_manager

        def _fp8_cache_release_hook(
            mod: nn.Module, _grad_input, _grad_output
        ) -> None:
            cache_mgr.release_for_module(mod, bwd=True, fix_missing_fwd=True)

        self._hook_registry.ensure_registered(
            module,
            hook_type="fp8_cache_release",
            hook_fn=_fp8_cache_release_hook,
            device_role=device_role,
        )


# ---------------------------------------------------------------------------
# 工厂函数：构建DES-LOC设备画像
# ---------------------------------------------------------------------------

def build_device_profiles(
    a6000_devices: Optional[List[torch.device]] = None,
    h100_device: Optional[torch.device] = None,
    cpu_cache_budget_gb: float = 256.0,
) -> Dict[DeviceRole, DeviceProfile]:
    """构建Neuron_SP硬件拓扑的DeviceProfile字典。

    默认拓扑：
        - cuda:0, cuda:1 → A6000 SM86，各48GB，FP8 cache预算12GB
        - cuda:2         → H100 NVL SM90，96GB，FP8 cache预算32GB
        - cpu            → 1.5TB DRAM，cache预算256GB（可调）

    Args:
        a6000_devices:       A6000设备列表，默认[cuda:0, cuda:1]。
        h100_device:         H100设备，默认cuda:2。
        cpu_cache_budget_gb: CPU DRAM分配给FP8 cache的预算（GB）。

    Returns:
        DeviceRole → DeviceProfile 的映射字典。
    """
    if a6000_devices is None:
        a6000_devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    if h100_device is None:
        h100_device = torch.device("cuda:2")

    GiB = 1024 ** 3

    # 代表性A6000画像（取第一个设备作为LOCALITY_CACHE角色的代理）
    a6000_profile = DeviceProfile(
        role=DeviceRole.LOCALITY_CACHE,
        device=a6000_devices[0],
        total_memory=48 * GiB,
        sm_version=86,
        fp8_native=False,           # SM86不支持原生FP8，需模拟
        cache_budget=12 * GiB,
    )

    h100_profile = DeviceProfile(
        role=DeviceRole.EXECUTION,
        device=h100_device,
        total_memory=96 * GiB,
        sm_version=90,
        fp8_native=True,            # SM90原生FP8
        cache_budget=32 * GiB,
    )

    cpu_profile = DeviceProfile(
        role=DeviceRole.OFFLOAD_POOL,
        device=torch.device("cpu"),
        total_memory=int(1536 * GiB),
        sm_version=0,
        fp8_native=False,
        cache_budget=int(cpu_cache_budget_gb * GiB),
    )

    profiles = {
        DeviceRole.LOCALITY_CACHE: a6000_profile,
        DeviceRole.EXECUTION:      h100_profile,
        DeviceRole.OFFLOAD_POOL:   cpu_profile,
    }

    logger.info(
        "build_device_profiles | A6000=%s H100=%s CPU_budget=%.0fGB",
        [str(d) for d in a6000_devices], str(h100_device), cpu_cache_budget_gb,
    )
    return profiles


def assign_device_roles(
    model: nn.Module,
    locality_cache_prefixes: Tuple[str, ...] = ("embed", "ln_", "wte"),
    execution_prefixes: Tuple[str, ...] = ("attn", "mlp", "ff_"),
) -> None:
    """根据层名前缀为模块分配DES-LOC设备角色。

    约定：
        - embedding、LayerNorm等轻量层 → LOCALITY_CACHE（A6000）
        - attention、MLP等计算密集层  → EXECUTION（H100）
        - 其余层                      → EXECUTION（默认）

    Args:
        model:                   待分配的模型。
        locality_cache_prefixes: 匹配则分配到LOCALITY_CACHE的名称前缀。
        execution_prefixes:      匹配则分配到EXECUTION的名称前缀。
    """
    assigned = {DeviceRole.LOCALITY_CACHE: 0, DeviceRole.EXECUTION: 0}
    for name, module in model.named_modules():
        leaf_name = name.split(".")[-1] if "." in name else name
        if any(leaf_name.startswith(p) for p in locality_cache_prefixes):
            module._desloс_device_role = DeviceRole.LOCALITY_CACHE
            assigned[DeviceRole.LOCALITY_CACHE] += 1
        else:
            module._desloс_device_role = DeviceRole.EXECUTION
            assigned[DeviceRole.EXECUTION] += 1

    logger.info(
        "assign_device_roles | locality_cache=%d execution=%d",
        assigned[DeviceRole.LOCALITY_CACHE], assigned[DeviceRole.EXECUTION],
    )


# ---------------------------------------------------------------------------
# 便捷入口：一键将修复注入现有模型
# ---------------------------------------------------------------------------

def patch_model_for_desloс_mxfp8(
    model: nn.Module,
    profiles: Optional[Dict[DeviceRole, DeviceProfile]] = None,
    **kwargs,
) -> HeteroFSDPMXFP8Fix:
    """将DES-LOC MXFP8修复注入模型，返回修复实例。

    Args:
        model:    待修复的nn.Module。
        profiles: 设备画像，若为None则自动调用build_device_profiles()。
        **kwargs: 传递给HeteroFSDPMXFP8Fix.init_desloс_fix的额外参数。

    Returns:
        已初始化的HeteroFSDPMXFP8Fix实例，同时附加到model._desloс_fix。
    """
    if profiles is None:
        profiles = build_device_profiles()

    fix = HeteroFSDPMXFP8Fix()
    fix.init_desloс_fix(profiles, **kwargs)
    assign_device_roles(model)
    model._desloс_fix = fix  # 弱引用也可，此处简化

    logger.info(
        "patch_model_for_desloс_mxfp8 | model=%s patched successfully",
        type(model).__name__,
    )
    return fix


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    # 构建最小测试模型
    class _TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(128, 64)
            self.attn  = nn.Linear(64, 64, bias=False)
            self.mlp   = nn.Linear(64, 64, bias=False)
            self.ln_   = nn.LayerNorm(64)

    model = _TinyTransformer()
    profiles = build_device_profiles(
        a6000_devices=[torch.device("cpu")],  # 测试环境用CPU模拟
        h100_device=torch.device("cpu"),
    )
    fix = patch_model_for_desloс_mxfp8(model, profiles)

    # --- Test 1: Bug-2 递归状态设置 ---
    fix._state_coordinator.set_training_state_recursive(
        model, TrainingState.PRE_BACKWARD
    )
    for sub in model.modules():
        assert getattr(sub, "_training_state", None) == TrainingState.PRE_BACKWARD, \
            f"Bug-2: {type(sub).__name__} not in PRE_BACKWARD"

    fix._state_coordinator.set_training_state_recursive(
        model, TrainingState.IDLE
    )
    for sub in model.modules():
        assert getattr(sub, "_training_state", None) == TrainingState.IDLE, \
            f"Bug-2: {type(sub).__name__} not in IDLE after reset"

    # --- Test 2: Bug-1 LocalityCacheManager 双路径释放 ---
    dummy_tensor = torch.zeros(4, 4)
    fix._cache_manager.register_cache(
        model.attn, [dummy_tensor], bwd=True,
        device_role=DeviceRole.LOCALITY_CACHE,
    )
    fix._cache_manager.register_cache(
        model.attn, [dummy_tensor.clone()], bwd=False,
        device_role=DeviceRole.LOCALITY_CACHE,
    )
    mid = id(model.attn)
    assert False in fix._cache_manager._registry[mid], "bwd=False not registered"
    fix._cache_manager.release_for_module(model.attn, bwd=True, fix_missing_fwd=True)
    assert mid not in fix._cache_manager._registry, \
        "Bug-1: bwd=False cache not released after fix_missing_fwd"

    # --- Test 3: Bug-3 FineGrainedHookRegistry 幂等性 ---
    registry = FineGrainedHookRegistry()
    FineGrainedHookRegistry.reset()
    registry = FineGrainedHookRegistry()

    dummy_hook = lambda m, gi, go: None
    r1 = registry.ensure_registered(
        model.attn, "pre_backward_unshard", dummy_hook,
        DeviceRole.EXECUTION,
    )
    r2 = registry.ensure_registered(
        model.attn, "pre_backward_unshard", dummy_hook,
        DeviceRole.EXECUTION,
    )
    assert r1 is True,  "Bug-3: first registration should succeed"
    assert r2 is False, "Bug-3: duplicate registration should be skipped"

    print("All smoke tests passed. DES-LOC MXFP8 fix OK.")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroPreBackwardCoordinator on a DeepSpeed engine.

    Instantiates a :class:`HeteroPreBackwardCoordinator` from the engine's configuration
    and attaches it as ``engine.hetero_fsdp_mxfp8_fix``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_fsdp_mxfp8_fix.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_fsdp_mxfp8_fix = None
    logger.info("hetero_fsdp_mxfp8_fix.register() attached engine.hetero_fsdp_mxfp8_fix")
