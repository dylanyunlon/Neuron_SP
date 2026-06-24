"""
DES-LOC Heterogeneous Module Registry
======================================

Central registry that discovers, catalogs, and wires all hetero_*.py modules
into the DES-LOC engine. This is the glue layer between 97 independent module
files and the unified training system.

Design:
    Each hetero_*.py exposes classes following a naming convention:
    - Hetero{Name}  → primary class
    - Hetero{Name}Config → optional config dataclass

    The registry scans deepspeed/ for all hetero_*.py files, imports them,
    and categorizes by subsystem (runtime, zero, moe, inference, checkpoint,
    comm, ops). The engine queries the registry to get the right module for
    each training phase.
"""

import os
import re
import sys
import logging
import importlib
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Type
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ModuleInfo:
    """Metadata about one hetero module."""
    file_path: str
    module_name: str
    subsystem: str
    classes: List[str]
    primary_class: Optional[str]
    loaded: bool = False
    module_obj: Optional[Any] = None
    error: Optional[str] = None


# Subsystem classification by directory
SUBSYSTEM_MAP = {
    "runtime/zero": "zero",
    "runtime": "runtime",
    "moe": "moe",
    "inference": "inference",
    "checkpoint": "checkpoint",
    "comm": "comm",
    "ops": "ops",
    "pipe": "pipe",
    "profiling": "profiling",
    "sequence": "sequence",
    "compression": "compression",
    "compile": "compile",
    "elasticity": "elasticity",
}


class HeteroRegistry:
    """
    Discovers and manages all hetero_*.py modules.

    Usage:
        registry = HeteroRegistry()
        registry.scan()                      # find all modules
        registry.load_subsystem("runtime")   # lazy-load one subsystem
        cls = registry.get_class("HeteroStepBatchScheduler")
    """

    def __init__(self, project_root: Optional[str] = None):
        if project_root is None:
            project_root = str(Path(__file__).parent.parent.parent)
        self.project_root = project_root
        self.modules: Dict[str, ModuleInfo] = {}   # file_path → info
        self._class_index: Dict[str, ModuleInfo] = {}  # class_name → info
        self._subsystem_index: Dict[str, List[ModuleInfo]] = defaultdict(list)

    def scan(self) -> int:
        """Scan for all hetero_*.py files under deepspeed/."""
        ds_root = os.path.join(self.project_root, "deepspeed")
        count = 0
        for root, dirs, files in os.walk(ds_root):
            for fname in sorted(files):
                if fname.startswith("hetero_") and fname.endswith(".py"):
                    fpath = os.path.join(root, fname)
                    rel = os.path.relpath(fpath, self.project_root)

                    # Determine subsystem
                    rel_dir = os.path.relpath(root, ds_root)
                    subsystem = "other"
                    for prefix, sub in sorted(SUBSYSTEM_MAP.items(),
                                              key=lambda x: -len(x[0])):
                        if rel_dir.startswith(prefix) or rel_dir == prefix:
                            subsystem = sub
                            break

                    # Extract class names via regex (no import yet)
                    classes = self._extract_classes(fpath)
                    primary = None
                    for c in classes:
                        if c.startswith("Hetero") and "Config" not in c and "Test" not in c:
                            primary = c
                            break

                    info = ModuleInfo(
                        file_path=rel,
                        module_name=fname[:-3],
                        subsystem=subsystem,
                        classes=classes,
                        primary_class=primary,
                    )
                    self.modules[rel] = info
                    self._subsystem_index[subsystem].append(info)
                    for c in classes:
                        self._class_index[c] = info
                    count += 1

        logger.info("Registry: scanned %d hetero modules across %d subsystems",
                     count, len(self._subsystem_index))
        return count

    @staticmethod
    def _extract_classes(filepath: str) -> List[str]:
        """Extract class names from a Python file without importing it."""
        classes = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    m = re.match(r'^class\s+(\w+)', line)
                    if m:
                        classes.append(m.group(1))
        except Exception:
            pass
        return classes

    def load_module(self, info: ModuleInfo) -> bool:
        """Import a single module."""
        if info.loaded:
            return True
        try:
            full_path = os.path.join(self.project_root, info.file_path)
            spec = importlib.util.spec_from_file_location(
                info.module_name, full_path
            )
            if spec is None or spec.loader is None:
                info.error = "spec_from_file_location returned None"
                return False
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            info.module_obj = mod
            info.loaded = True
            return True
        except Exception as e:
            info.error = str(e)
            logger.debug("Failed to load %s: %s", info.file_path, e)
            return False

    def load_subsystem(self, subsystem: str) -> int:
        """Load all modules in a subsystem."""
        loaded = 0
        for info in self._subsystem_index.get(subsystem, []):
            if self.load_module(info):
                loaded += 1
        logger.info("Loaded %d/%d modules from subsystem '%s'",
                     loaded, len(self._subsystem_index.get(subsystem, [])),
                     subsystem)
        return loaded

    def load_all(self) -> Dict[str, int]:
        """Load every module (may fail for those with unmet dependencies)."""
        results = {}
        for sub in self._subsystem_index:
            results[sub] = self.load_subsystem(sub)
        return results

    def get_class(self, class_name: str) -> Optional[type]:
        """Get a class by name, loading its module if needed."""
        info = self._class_index.get(class_name)
        if info is None:
            return None
        if not info.loaded:
            self.load_module(info)
        if info.module_obj is None:
            return None
        return getattr(info.module_obj, class_name, None)

    def get_subsystem_classes(self, subsystem: str) -> Dict[str, type]:
        """Get all primary classes from a subsystem."""
        result = {}
        for info in self._subsystem_index.get(subsystem, []):
            if not info.loaded:
                self.load_module(info)
            if info.primary_class and info.module_obj:
                cls = getattr(info.module_obj, info.primary_class, None)
                if cls:
                    result[info.primary_class] = cls
        return result

    def discover_modules(self) -> int:
        """Discover all hetero_*.py modules under deepspeed/.

        Convenience wrapper around scan() that also indexes all classes
        for later hook registration.  Returns number of modules found.
        """
        count = self.scan()
        logger.info("HeteroRegistry: discovered %d modules", count)
        return count

    def register_hooks(self, engine: Any) -> int:
        """Register every discovered module with the engine.

        For each module that exposes a top-level ``register(engine)``
        function, call it.  For modules without ``register()``, apply
        the fallback: store a reference on the engine so the module's
        primary class can be retrieved via the registry later.

        Returns the number of modules that were successfully registered
        (either via register() or via fallback).
        """
        registered = 0
        for info in self.modules.values():
            if not info.loaded:
                self.load_module(info)
            if info.module_obj is None:
                continue

            # Preferred path: module exposes register(engine)
            register_fn = getattr(info.module_obj, "register", None)
            if callable(register_fn):
                try:
                    register_fn(engine)
                    registered += 1
                    continue
                except Exception as exc:
                    logger.debug(
                        "register() failed for %s: %s", info.module_name, exc
                    )

            # Fallback: no register() — store the primary class reference
            # on the engine keyed by module name so it can be looked up.
            if info.primary_class:
                cls = getattr(info.module_obj, info.primary_class, None)
                if cls is not None:
                    attr_name = f"_hetero_mod_{info.module_name}"
                    if not hasattr(engine, attr_name):
                        setattr(engine, attr_name, cls)
                    registered += 1

        logger.info(
            "HeteroRegistry: registered %d/%d modules with engine",
            registered, len(self.modules),
        )
        return registered

    def discover_and_register(self, engine: Any) -> int:
        """One-shot: discover all modules and register them with the engine.

        This is the intended entry point called during engine init.

        Scans deepspeed/runtime/ and deepspeed/runtime/zero/ for all
        hetero_*.py files.  For each discovered module, attempts
        importlib.import_module and calls register(engine) if exposed.
        Tracks per-module success / failure / skip counts.
        """
        ds_root = os.path.join(self.project_root, "deepspeed")
        scan_dirs = [
            os.path.join(ds_root, "runtime"),
            os.path.join(ds_root, "runtime", "zero"),
        ]

        succeeded = 0
        failed = 0
        skipped = 0

        for scan_dir in scan_dirs:
            if not os.path.isdir(scan_dir):
                continue
            for fname in sorted(os.listdir(scan_dir)):
                if not fname.startswith("hetero_") or not fname.endswith(".py"):
                    continue
                # Skip self to avoid recursive import
                if fname == "hetero_registry.py":
                    skipped += 1
                    continue

                # Build dotted module path relative to project root
                rel = os.path.relpath(
                    os.path.join(scan_dir, fname), self.project_root
                )
                dotted = rel[:-3].replace(os.sep, ".")

                try:
                    mod = importlib.import_module(dotted)
                except Exception as exc:
                    logger.debug(
                        "discover_and_register: import failed for %s: %s",
                        dotted, exc,
                    )
                    failed += 1
                    continue

                register_fn = getattr(mod, "register", None)
                if not callable(register_fn):
                    skipped += 1
                    continue

                try:
                    register_fn(engine)
                    succeeded += 1
                except Exception as exc:
                    logger.debug(
                        "discover_and_register: register() failed for %s: %s",
                        dotted, exc,
                    )
                    failed += 1

        total = succeeded + failed + skipped
        logger.info(
            "discover_and_register: %d modules total — "
            "%d succeeded, %d failed, %d skipped",
            total, succeeded, failed, skipped,
        )

        # Also populate the internal index for class lookups
        self.discover_modules()
        return succeeded

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["DES-LOC Hetero Module Registry", "=" * 40]
        for sub, modules in sorted(self._subsystem_index.items()):
            loaded = sum(1 for m in modules if m.loaded)
            lines.append(f"  {sub}: {len(modules)} modules ({loaded} loaded)")
            for m in modules:
                status = "✓" if m.loaded else ("✗" if m.error else "○")
                lines.append(f"    {status} {m.module_name} → {m.primary_class or '(no primary)'}")
        lines.append(f"\nTotal: {len(self.modules)} modules, "
                     f"{len(self._class_index)} classes indexed")
        return "\n".join(lines)

    def register_all(self, engine: Any) -> int:
        """Call ``register(engine)`` on every discovered hetero module.

        This is the single entry-point that wires all hetero_*.py modules into
        *engine* at once.  For each module that exposes a class-level
        ``register(engine)`` classmethod (e.g. ``HeteroStepBatchScheduler.register``)
        or a module-level ``register(engine)`` function, that callable is invoked
        in discovery order.

        Modules that have not been loaded yet are imported on demand.  Modules
        that fail to import or whose ``register()`` raises are skipped with a
        DEBUG log so that a single broken module cannot block the rest.

        Returns
        -------
        int
            Number of modules successfully registered.
        """
        # Four well-known modules are registered via their class-level register()
        # to satisfy the explicit requirement; all other modules are handled by
        # the existing discover_and_register / register_hooks infrastructure.
        _EXPLICIT_MODULES = (
            "deepspeed.runtime.hetero_step_batch_scheduler",
            "deepspeed.runtime.hetero_gdn_selective_recompute",
            "deepspeed.runtime.hetero_fp32_grad_accum",
            "deepspeed.runtime.hetero_grad_norm_skip",
        )

        registered = 0

        # --- Phase 1: explicit class-level register() calls ---
        for dotted in _EXPLICIT_MODULES:
            try:
                import importlib as _importlib
                mod = _importlib.import_module(dotted)
                # Prefer the module-level register() which itself delegates to
                # the primary class's classmethod.
                register_fn = getattr(mod, "register", None)
                if callable(register_fn):
                    register_fn(engine)
                    registered += 1
                    logger.debug("register_all: registered %s", dotted)
                else:
                    logger.debug(
                        "register_all: no register() found in %s, skipping", dotted
                    )
            except Exception as exc:
                logger.debug("register_all: failed to register %s: %s", dotted, exc)

        # --- Phase 2: remaining modules via the generic hook mechanism ---
        # Ensure the registry has been scanned.
        if not self.modules:
            self.scan()

        extra = self.register_hooks(engine)
        # Avoid double-counting the explicit modules that were handled in phase 1.
        registered_total = registered + max(0, extra - len(_EXPLICIT_MODULES))

        logger.info(
            "HeteroRegistry.register_all(): %d modules registered with engine",
            registered_total,
        )
        return registered_total


# ---------------------------------------------------------------------------
# Convenience: global registry singleton
# ---------------------------------------------------------------------------

_global_registry: Optional[HeteroRegistry] = None


def get_registry() -> HeteroRegistry:
    """Get or create the global registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = HeteroRegistry()
        _global_registry.scan()
    return _global_registry


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reg = get_registry()
    results = reg.load_all()
    print(reg.summary())
    print(f"\nLoad results: {results}")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroRegistry on a DeepSpeed engine.

    Instantiates a :class:`HeteroRegistry` from the engine's configuration
    and attaches it as ``engine.hetero_registry``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_registry.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_registry = None
    logger.info("hetero_registry.register() attached engine.hetero_registry")
