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

        Three activation paths are tried in order for each module:

          1. Preferred — the module exposes a top-level ``register(engine)``
             function which is invoked directly.
          2. Alias     — the module has no ``register()`` but does have a
             ``register_*`` function (e.g. ``register_deslock_checkpoint_strategy``).
             The first such callable is invoked.  This handles modules that
             predate the naming convention without requiring a shim.
          3. Fallback  — no registration callable exists; the module's primary
             ``Hetero*`` class (excluding pure-config dataclasses named
             ``Hetero*Config`` *only when* a non-config class is also present)
             is attached to the engine as ``_hetero_mod_<module_name>`` so it
             remains reachable via the registry.

        A module is counted as *hooked* once any of the three paths succeeds.
        The returned count reflects hooked modules; use ``_count_activated``
        for the subset that have also been fully initialised (engine attribute
        non-None).

        Returns:
            The number of modules successfully hooked in this call (modules
            already hooked in a prior call are excluded from the count).
        """
        newly_hooked = 0
        for key, mod in self._modules.items():
            if key in self._hooks:
                continue

            # --- Path 1: canonical register(engine) ---
            register_fn = getattr(mod, "register", None)
            if callable(register_fn):
                try:
                    register_fn(engine)
                    self._hooks[key] = mod
                    newly_hooked += 1
                    logger.debug("Hook registered via register() for module: %s", key)
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Hook registration failed for %s: %s", key, exc)

            # --- Path 2: register_*(engine) alias ---
            # Some modules (e.g. hetero_async_checkpoint_load) expose a
            # register_<specific_name>(engine) function instead of the generic
            # register().  Find the first top-level callable whose name starts
            # with "register_" and call it.
            alias_fn = None
            for attr_name in dir(mod):
                if not attr_name.startswith("register_"):
                    continue
                candidate = getattr(mod, attr_name, None)
                if callable(candidate) and getattr(candidate, "__module__", None) == mod.__name__:
                    alias_fn = (attr_name, candidate)
                    break

            if alias_fn is not None:
                alias_name, fn = alias_fn
                try:
                    fn(engine)
                    self._hooks[key] = mod
                    newly_hooked += 1
                    logger.debug(
                        "Hook registered via alias %s() for module: %s", alias_name, key
                    )
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Hook alias %s() failed for %s: %s", alias_name, key, exc
                    )

            # --- Path 3: attach primary Hetero* class as engine attribute ---
            # Scan __all__ first (explicit exports), then fall back to dir().
            # Exclude pure enum/config-only classes (name ends with "Config"
            # *and* no non-config Hetero* class has been found yet in this
            # module) so that a module like hetero_checkpoint_config, whose
            # only public Hetero class IS a config, still gets attached.
            names = list(getattr(mod, "__all__", None) or []) or [
                n for n in dir(mod) if n.startswith("Hetero")
            ]

            hetero_classes = []
            for attr_name in names:
                if not attr_name.startswith("Hetero"):
                    continue
                candidate = getattr(mod, attr_name, None)
                if isinstance(candidate, type) and candidate.__module__ == mod.__name__:
                    hetero_classes.append((attr_name, candidate))

            # Prefer non-Config classes; fall back to Config classes when
            # that's all the module has.
            primary_cls = None
            non_config = [(n, c) for n, c in hetero_classes if "Config" not in n]
            primary_cls = (non_config or hetero_classes or [None])[0]  # type: ignore[assignment]

            if primary_cls is not None:
                attr_name, cls = primary_cls
                short = mod.__name__.rsplit(".", 1)[-1]
                engine_attr = f"_hetero_mod_{short}"
                if not hasattr(engine, engine_attr):
                    setattr(engine, engine_attr, cls)
                self._hooks[key] = mod
                newly_hooked += 1
                logger.debug(
                    "Hook fallback for %s: attached %s as engine.%s",
                    key, attr_name, engine_attr,
                )
            else:
                logger.warning(
                    "HeteroRegistry: module %s has no register(), register_*(), "
                    "or Hetero* class — skipping.",
                    key,
                )

        hooked_total = len(self._hooks)
        activated = self._count_activated(engine)

        logger.info(
            "HeteroRegistry: %d/%d hetero_* modules hooked on engine "
            "(%d newly hooked this pass; %d fully initialised, %d pending placeholder).",
            hooked_total, len(self._modules),
            newly_hooked, activated, hooked_total - activated,
        )
        return newly_hooked

    def _count_activated(self, engine: "DesLocEngine") -> int:
        """Return the number of hooked modules that are fully initialised.

        A module is counted as *fully initialised* when **any** of the
        following is true:

        * The engine carries a non-None attribute derived from the module name
          (``engine.hetero_<short>`` for register()-registered modules, or
          ``engine._hetero_mod_<short>`` for fallback-attached ones).
        * The module was hooked via a ``register_*()`` alias and set at least
          one non-None ``engine.hetero_*`` or ``engine._hetero_mod_*``
          attribute (passive modules that only provide configuration classes
          and don't write engine attributes are also counted, since attaching
          the class itself is a successful activation).

        Modules that only set a ``None`` placeholder during ``register()``
        are counted as *pending*, not activated.
        """
        count = 0
        for key, mod in self._hooks.items():
            short = mod.__name__.rsplit(".", 1)[-1]  # e.g. "hetero_fp32_grad_accum"

            # Check for engine attribute set by register() or alias path.
            preferred_attr = short                       # e.g. "hetero_fp32_grad_accum"
            fallback_attr  = f"_hetero_mod_{short}"     # e.g. "_hetero_mod_hetero_fp32_grad_accum"

            activated = False
            for attr in (preferred_attr, fallback_attr):
                val = getattr(engine, attr, _SENTINEL)
                if val is not _SENTINEL and val is not None:
                    activated = True
                    break

            # For passive modules (config-only, no engine attr written), the
            # mere presence of a class attached via the fallback path counts as
            # activated — the module is reachable through the registry.
            if not activated:
                fallback_val = getattr(engine, fallback_attr, _SENTINEL)
                if fallback_val is not _SENTINEL:
                    # Attached as a class reference (the fallback path stores
                    # the class, not an instance, so isinstance(..., type) is True).
                    if isinstance(fallback_val, type):
                        activated = True

            if activated:
                count += 1

        return count

    def get(self, name: str) -> Optional[Any]:
        """Retrieve a registered module by its registry name."""
        return self._modules.get(name)

    def __len__(self) -> int:
        return len(self._modules)
