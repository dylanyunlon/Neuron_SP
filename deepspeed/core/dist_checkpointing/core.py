# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_FNAME = 'metadata.json'
MANIFEST_FNAME = 'manifest.json'

logger = logging.getLogger(__name__)


class CheckpointingException(Exception):
    pass


@dataclass
class CheckpointingConfig:
    """ Documents backends used in the checkpoint. """
    sharded_backend: str
    sharded_backend_version: int = 1
    common_backend: str = 'torch'
    common_backend_version: int = 1


# Insight I4: versioned checkpoint schema (Megatron ab-3.3)
# ---------------------------------------------------------------------------
# CheckpointManifest replaces ad-hoc getattr()-based defensive reads on the
# old CheckpointingConfig.  Every field carries a safe default so that a
# checkpoint written by an older version of the code can still be loaded by
# a newer reader without raising AttributeError or KeyError.
#
# Upgrade policy:
#   - Adding a new field: give it a default value → old checkpoints load fine.
#   - Removing a field: keep it as Optional with default=None for one release
#     cycle so the reader can emit a deprecation warning rather than crash.
#   - Changing semantics: bump `version` and document the mapping below.
#
# Version history:
#   1 – initial release; sharded_backend + common_backend parity with
#       CheckpointingConfig.
#   2 – added parallel_config dict; model_dtype; extra_metadata escape hatch.
# ---------------------------------------------------------------------------
@dataclass
class CheckpointManifest:
    """Versioned manifest for distributed checkpoints.

    All fields have defaults so that loading a checkpoint produced by an older
    codebase (which may lack newer fields) never raises an error.  New fields
    must always be added with a default value; never add a positional/required
    field.

    Attributes:
        version: Schema version, incremented when field semantics change.
        sharded_backend: Name of the sharded-tensor storage backend
            (e.g. 'zarr', 'torch').
        sharded_backend_version: Version of the sharded backend.
        common_backend: Storage backend used for the non-sharded common dict.
        common_backend_version: Version of the common backend.
        parallel_config: Snapshot of parallel-strategy configuration at save
            time (tp_size, pp_size, dp_size, …).  Used to detect resharding
            mismatches on load.  Empty dict means not recorded.
        model_dtype: String representation of the primary model parameter
            dtype, e.g. 'torch.bfloat16'.  Empty string means not recorded.
        extra_metadata: Arbitrary key/value pairs for future extensions.
            Readers MUST ignore unknown keys in this dict.
    """
    # Schema version – bump when field semantics change (not just additions).
    version: int = 2

    # Backend identity (mirrors CheckpointingConfig for backward compat).
    sharded_backend: str = ''
    sharded_backend_version: int = 1
    common_backend: str = 'torch'
    common_backend_version: int = 1

    # Parallel strategy snapshot – all new fields go here with defaults.
    parallel_config: Dict[str, Any] = field(default_factory=dict)

    # Model dtype recorded at save time.
    model_dtype: str = ''

    # Open-ended escape hatch: future fields land here first, graduate to
    # top-level attributes in the next major version bump.
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory: build from the legacy CheckpointingConfig so existing save
    # paths can adopt CheckpointManifest incrementally.
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpointing_config(
        cls, config: 'CheckpointingConfig', **kwargs
    ) -> 'CheckpointManifest':
        """Construct a CheckpointManifest from a legacy CheckpointingConfig.

        Any extra keyword arguments are forwarded as field overrides, which
        allows callers to populate parallel_config or model_dtype without
        touching the legacy config object.
        """
        return cls(
            sharded_backend=config.sharded_backend,
            sharded_backend_version=config.sharded_backend_version,
            common_backend=config.common_backend,
            common_backend_version=config.common_backend_version,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CheckpointManifest':
        """Deserialise, ignoring unknown keys so old readers aren't broken."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        unknown = set(d) - known
        if unknown:
            logger.debug(
                "CheckpointManifest: ignoring unknown keys from disk: %s", unknown
            )
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Manifest I/O helpers
# ---------------------------------------------------------------------------

def save_manifest(manifest: CheckpointManifest, checkpoint_dir: str) -> None:
    """Persist the CheckpointManifest alongside the existing metadata.json.

    The manifest is stored as MANIFEST_FNAME ('manifest.json') in the
    checkpoint directory.  Writing is intentionally separate from
    save_config() so that old-format checkpoints remain readable by code
    that does not yet understand the manifest.
    """
    # Insight I4: versioned checkpoint schema (Megatron ab-3.3)
    manifest_path = Path(checkpoint_dir, MANIFEST_FNAME)
    with manifest_path.open('w') as f:
        json.dump(manifest.to_dict(), f, indent=2)


def load_manifest(checkpoint_dir: str) -> Optional[CheckpointManifest]:
    """Load a CheckpointManifest if present, else return None.

    Callers that need a non-None value for old checkpoints can fall back to
    ``CheckpointManifest.from_checkpointing_config(maybe_load_config(...))``.
    """
    # Insight I4: versioned checkpoint schema (Megatron ab-3.3)
    manifest_path = Path(checkpoint_dir, MANIFEST_FNAME)
    if not manifest_path.exists():
        return None
    with manifest_path.open() as f:
        raw = json.load(f)
    return CheckpointManifest.from_dict(raw)


def check_is_distributed_checkpoint(checkpoint_dir):
    return maybe_load_config(checkpoint_dir) is not None


def maybe_load_config(checkpoint_dir: str) -> Optional[CheckpointingConfig]:
    config_path = Path(checkpoint_dir, CONFIG_FNAME)
    if not config_path.exists():
        return None
    with config_path.open() as f:
        config_dict = json.load(f)
    # Insight I4: versioned checkpoint schema (Megatron ab-3.3)
    # Strip unknown keys so old CheckpointingConfig fields don't cause
    # TypeError when a manifest-enriched metadata.json is read back by a
    # code path that still uses CheckpointingConfig directly.
    known_fields = {
        'sharded_backend', 'sharded_backend_version',
        'common_backend', 'common_backend_version',
    }
    filtered = {k: v for k, v in config_dict.items() if k in known_fields}
    return CheckpointingConfig(**filtered)


def save_config(config: CheckpointingConfig, checkpoint_dir: str):
    config_path = Path(checkpoint_dir, CONFIG_FNAME)
    with config_path.open('w') as f:
        json.dump(asdict(config), f)
print('[M1437]')
