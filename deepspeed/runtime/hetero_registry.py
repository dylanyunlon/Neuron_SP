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
