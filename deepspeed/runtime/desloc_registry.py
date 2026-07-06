"""
HeteroRegistry: auto-discovers hetero_* modules and exposes a unified registry.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from deepspeed.runtime.desloc_engine import DesLocEngine

logger = logging.getLogger(__name__)

_REGISTRY_BASE = "deepspeed"
_HETERO_PREFIX = "hetero_"
_SENTINEL = object()  # Used to distinguish "attribute absent" from "attribute is None"


class HeteroRegistry:
    """
    Auto-discovers all hetero_*.py modules under the deepspeed package tree
    and exposes them through a unified registry dict.

    Modules are expected to optionally expose:
        - REGISTRY_NAME: str
        - register(engine): callable that receives the engine instance
    """

    def __init__(self) -> None:
        self._modules: Dict[str, Any] = {}
        self._hooks: Dict[str, Any] = {}

    def discover(self, base_package: str = _REGISTRY_BASE) -> None:
        """
        Walk the base_package tree and import every module whose name starts
        with hetero_.  Collects REGISTRY_NAME and register() if present.

        Args:
            base_package: Top-level package name to search.
        """
        try:
            base_mod = importlib.import_module(base_package)
        except ImportError:
            logger.warning("Base package '%s' not importable; skipping discovery.", base_package)
            return

        base_path = getattr(base_mod, "__path__", [])
        found = 0
        for finder, mod_name, is_pkg in pkgutil.walk_packages(
            path=base_path,
            prefix=base_package + ".",
            onerror=lambda e: logger.debug("pkgutil walk error: %s", e),
        ):
            short = mod_name.split(".")[-1]
            if not short.startswith(_HETERO_PREFIX):
                continue
            try:
                mod = importlib.import_module(mod_name)
                key = getattr(mod, "REGISTRY_NAME", mod_name)
                self._modules[key] = mod
                found += 1
                logger.debug("Registered hetero module: %s -> %s", mod_name, key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to import hetero module %s: %s", mod_name, exc)

        logger.info("HeteroRegistry: discovered %d hetero_* modules.", found)

    def register_hooks(self, engine: "DesLocEngine") -> int:
        """
        Activate every discovered hetero_* module against the engine.

        Two activation paths are supported:
          1. Preferred — the module exposes a top-level ``register(engine)``
             function which is invoked directly.
          2. Fallback  — the module has no ``register()`` hook, in which case
             its primary ``Hetero*`` class is attached to the engine under
             ``_hetero_mod_<module_name>`` so it can be retrieved later via
             the registry.  This ensures even passive extension modules are
             discoverable from the engine instance.

        A module is counted as *actually activated* only when the engine
        attribute that ``register()`` is expected to set is present **and
        non-None**.  Modules whose ``register()`` merely installs a ``None``
        placeholder (deferred initialisation) are tracked separately so the
        caller can distinguish "hooked but pending" from "fully initialised".

        Returns:
            The number of modules that are fully initialised on the engine
            (engine attribute exists and is not None).
        """
        for key, mod in self._modules.items():
            if key in self._hooks:
                continue

            register_fn = getattr(mod, "register", None)
            if callable(register_fn):
                try:
                    register_fn(engine)
                    self._hooks[key] = mod
                    logger.debug("Hook registered from module: %s", key)
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Hook registration failed for %s: %s", key, exc)

            primary_cls = None
            for attr_name in getattr(mod, "__all__", None) or dir(mod):
                if not attr_name.startswith("Hetero") or "Config" in attr_name:
                    continue
                candidate = getattr(mod, attr_name, None)
                if isinstance(candidate, type) and candidate.__module__ == mod.__name__:
                    primary_cls = (attr_name, candidate)
                    break

            if primary_cls is not None:
                attr_name, cls = primary_cls
                short = mod.__name__.rsplit(".", 1)[-1]
                engine_attr = f"_hetero_mod_{short}"
                if not hasattr(engine, engine_attr):
                    setattr(engine, engine_attr, cls)
                self._hooks[key] = mod
                logger.debug(
                    "Hook fallback for %s: attached %s as engine.%s",
                    key, attr_name, engine_attr,
                )

        # Count only modules that are genuinely initialised on the engine:
        # their engine attribute must exist and be non-None.  Modules whose
        # register() only installed a None placeholder are "hooked" (present
        # in self._hooks) but not yet "activated".
        activated = self._count_activated(engine)
        hooked = len(self._hooks)

        logger.info(
            "HeteroRegistry: %d/%d hetero_* modules fully initialised on engine "
            "(%d hooked, %d pending placeholder).",
            activated, len(self._modules), hooked, hooked - activated,
        )
        return activated

    def _count_activated(self, engine: "DesLocEngine") -> int:
        """Return the number of hooked modules that are fully initialised.

        A module is considered *fully initialised* when the engine carries a
        non-None attribute for it.  The attribute name is derived by
        convention: for modules registered via ``register(engine)`` DeepSpeed
        attaches the instance as ``engine.hetero_<short_module_name>`` (e.g.
        ``engine.hetero_fp32_grad_accum``).  For fallback-attached modules the
        attribute is ``engine._hetero_mod_<short_module_name>``.

        Modules that only set a ``None`` placeholder during ``register()``
        are counted as *pending*, not activated.
        """
        count = 0
        for key, mod in self._hooks.items():
            short = mod.__name__.rsplit(".", 1)[-1]  # e.g. "hetero_fp32_grad_accum"

            # Preferred path: engine.hetero_<name> (strip the hetero_ prefix
            # already present in short so we don't double it).
            preferred_attr = short  # e.g. "hetero_fp32_grad_accum"
            fallback_attr = f"_hetero_mod_{short}"  # e.g. "_hetero_mod_hetero_fp32_grad_accum"

            for attr in (preferred_attr, fallback_attr):
                val = getattr(engine, attr, _SENTINEL)
                if val is not _SENTINEL and val is not None:
                    count += 1
                    break

        return count

    def get(self, name: str) -> Optional[Any]:
        """Retrieve a registered module by its registry name."""
        return self._modules.get(name)

    def __len__(self) -> int:
        return len(self._modules)
