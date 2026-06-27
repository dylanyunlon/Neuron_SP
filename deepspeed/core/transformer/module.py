# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""MegatronModule — base class for all transformer sub-modules."""

from __future__ import annotations

from abc import ABC
from typing import Optional

import torch.nn as nn


class MegatronModule(nn.Module, ABC):
    """Base module class. All transformer components inherit from this.

    Provides:
    * ``self.config`` — the TransformerConfig for this module.
    * ``sharded_state_dict()`` — distributed checkpoint helper that
      attaches TP sharding metadata to tensor-parallel parameters.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> dict:
        """Return state dict with sharding metadata for distributed checkpointing.

        Iterates over all parameters and buffers, attaching TP sharding metadata
        when the parameter is marked as ``tensor_model_parallel``, then recurses
        into all child submodules.

        The result is a flat dict keyed by qualified name where each value is
        a plain dict::

            {
                "param": <Tensor>,
                "shape": <tuple>,
                "tp_shard": {"dim": int, "stride": int},   # only when TP-sharded
                "sharded_offsets": <tuple>,                 # only when PP offset set
            }

        Args:
            prefix: Dot-separated name prefix prepended to every key.
            sharded_offsets: Pipeline-parallel shard offsets.
            metadata: Optional extra metadata forwarded to child modules.

        Returns:
            Flat dict of sharded state entries.
        """
        sharded_sd: dict = {}

        # --- Parameters / buffers on this module --------------------------
        for name, param in self.named_parameters(recurse=False):
            key = f"{prefix}{name}"
            entry: dict = {"param": param, "shape": tuple(param.shape)}
            if getattr(param, "tensor_model_parallel", False):
                entry["tp_shard"] = {
                    "dim": getattr(param, "partition_dim", 0),
                    "stride": getattr(param, "partition_stride", 1),
                }
            if sharded_offsets:
                entry["sharded_offsets"] = sharded_offsets
            sharded_sd[key] = entry

        for name, buf in self.named_buffers(recurse=False):
            key = f"{prefix}{name}"
            sharded_sd[key] = {"param": buf, "shape": tuple(buf.shape)}

        # --- Recurse into child submodules --------------------------------
        for child_name, child_module in self.named_children():
            child_prefix = f"{prefix}{child_name}."
            if isinstance(child_module, MegatronModule):
                sharded_sd.update(
                    child_module.sharded_state_dict(
                        prefix=child_prefix,
                        sharded_offsets=sharded_offsets,
                        metadata=metadata,
                    )
                )
            else:
                for pname, param in child_module.named_parameters():
                    key = f"{child_prefix}{pname}"
                    sharded_sd[key] = {"param": param, "shape": tuple(param.shape)}

        return sharded_sd
