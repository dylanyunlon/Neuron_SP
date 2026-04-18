# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Collection of DeepSpeed configuration utilities
"""
import collections
import json
import torch
from functools import reduce
from pydantic import BaseModel, ConfigDict, field_serializer

from deepspeed.utils import logger


class DeepSpeedConfigModel(BaseModel):
    """
    This class should be used as a base for all DeepSpeed configs. It extends
    pydantic.BaseModel to allow for deprecated fields. To enable this feature,
    add deprecated=True to pydantic.Field:

    my_dep_field: int = Field(0, deprecated=True)

    Deprecated Field kwargs:
    - deprecated: [True|False], default False
        Enables / Disables deprecated fields
    - deprecated_msg: str, default ""
        Message to include with deprecation warning
    - new_param: str, default ""
        Name of the field replacing the deprecated field
    - set_new_param: [True|False], default True
        If new_param is provided, enables setting the value of that param with
        deprecated field value
    - new_param_fn: callable, default (lambda x: x)
        If new_param is provided and set_new_param is True, this function will
        modify the value of the deprecated field before placing that value in
        the new_param field

    Example:
        my_new_field is replacing a deprecated my_old_field. The expected type
        for my_new_field is int while the expected type for my_old_field is
        str. We want to maintain backward compatibility with our configs, so we
        define the fields with:

        class MyExampleConfig(DeepSpeedConfigModel):
            my_new_field: int = 0
            my_old_field: str = Field('0',
                                      deprecated=True,
                                      new_param='my_new_field',
                                      new_param_fn=(lambda x: int(x)))
    """

    def __init__(self, strict=False, **data):
        if (not strict):  # This is temporary until we refactor all DS configs, allows HF to load models
            data = {k: v for k, v in data.items() if (v != "auto" or k == "replace_method")}
        super().__init__(**data)
        self._deprecated_fields_check()

    def _process_deprecated_field(self, dep_field):
        # Get information about the deprecated field
        pydantic_config = self
        fields_set = pydantic_config.model_fields_set
        kwargs = type(pydantic_config).model_fields[dep_field].json_schema_extra
        new_param_fn = kwargs.get("new_param_fn", lambda x: x)
        param_value = new_param_fn(getattr(pydantic_config, dep_field))
        new_field = kwargs.get("new_param", "")
        dep_msg = kwargs.get("deprecated_msg", "")
        if dep_field in fields_set:
            logger.warning(f"Config parameter {dep_field} is deprecated" +
                           (f" use {new_field} instead" if new_field else "") + (f". {dep_msg}" if dep_msg else ""))
            # Check if there is a new param and if it should be set with a value
            if new_field and kwargs.get("set_new_param", True):
                # Remove the deprecate field if there is a replacing field
                try:
                    delattr(pydantic_config, dep_field)
                except Exception as e:
                    logger.error(f"Tried removing deprecated '{dep_field}' from config")
                    raise e

                # Set new param value
                new_param_nested = new_field.split(".")
                if len(new_param_nested) > 1:
                    # If the new param exists in a subconfig, we need to get
                    # the fields set for that subconfig
                    pydantic_config = reduce(getattr, new_param_nested[:-1], pydantic_config)
                    fields_set = pydantic_config.model_fields_set
                new_param_name = new_param_nested[-1]
                assert (
                    new_param_name not in fields_set
                ), f"Cannot provide deprecated parameter '{dep_field}' and replacing parameter '{new_field}' together"
                # A custom function for converting the old param value to new param value can be provided
                try:
                    setattr(pydantic_config, new_param_name, param_value)
                except Exception as e:
                    logger.error(f"Tried setting value for '{new_field}' with value from deprecated '{dep_field}'")
                    raise e

    def _deprecated_fields_check(self):
        fields = type(self).model_fields
        for field_name, field_info in fields.items():
            if field_info.json_schema_extra and field_info.json_schema_extra.get("deprecated", False):
                self._process_deprecated_field(field_name)

    model_config = ConfigDict(
        validate_default=True,
        validate_assignment=True,
        use_enum_values=True,
        populate_by_name=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    @field_serializer("dtype", check_fields=False)
    def serialize_torch_dtype(dtype: torch.dtype) -> str:
        return str(dtype)


def get_config_default(config, field_name):
    assert field_name in config.model_fields, f"'{field_name}' is not a field in {config}"
    assert not config.model_fields.get(
        field_name).is_required(), f"'{field_name}' is a required field and does not have a default value"
    return config.model_fields.get(field_name).get_default()


class pp_int(int):
    """
    A wrapper for integers that will return a custom string or comma-formatted
    string of the integer. For example, print(pp_int(1e5)) will return
    "10,000". This is useful mainly for auto-generated documentation purposes.
    """

    def __new__(cls, val, custom_print_str=None):
        inst = super().__new__(cls, val)
        inst.custom_print_str = custom_print_str
        return inst

    def __repr__(self):
        if hasattr(self, "custom_print_str") and self.custom_print_str:
            return self.custom_print_str
        return f"{self.real:,}"


# adapted from https://stackoverflow.com/a/50701137/9201239
class ScientificNotationEncoder(json.JSONEncoder):
    """
    This class overrides ``json.dumps`` default formatter.

    This version keeps everything as normal except formats numbers bigger than 1e3 using scientific notation.

    Just pass ``cls=ScientificNotationEncoder`` to ``json.dumps`` to activate it

    """

    def iterencode(self, o, _one_shot=False, level=0):
        indent = self.indent if self.indent is not None else 4
        prefix_close = " " * level * indent
        level += 1
        prefix = " " * level * indent
        if isinstance(o, bool):
            return "true" if o else "false"
        elif isinstance(o, float) or isinstance(o, int):
            if o > 1e3:
                return f"{o:e}"
            else:
                return f"{o}"
        elif isinstance(o, collections.abc.Mapping):
            x = [f'\n{prefix}"{k}": {self.iterencode(v, level=level)}' for k, v in o.items()]
            return "{" + ", ".join(x) + f"\n{prefix_close}" + "}"
        elif isinstance(o, collections.abc.Sequence) and not isinstance(o, str):
            return f"[{ ', '.join(map(self.iterencode, o)) }]"
        return "\n, ".join(super().iterencode(o, _one_shot))


class DeepSpeedConfigObject(object):
    """
    For json serialization
    """

    def repr(self):
        return self.__dict__

    def __repr__(self):
        return json.dumps(
            self.__dict__,
            sort_keys=True,
            indent=4,
            cls=ScientificNotationEncoder,
        )


def get_scalar_param(param_dict, param_name, param_default_value):
    return param_dict.get(param_name, param_default_value)


def get_list_param(param_dict, param_name, param_default_value):
    return param_dict.get(param_name, param_default_value)


def get_dict_param(param_dict, param_name, param_default_value):
    return param_dict.get(param_name, param_default_value)


def dict_raise_error_on_duplicate_keys(ordered_pairs):
    """Reject duplicate keys."""
    d = dict((k, v) for k, v in ordered_pairs)
    if len(d) != len(ordered_pairs):
        counter = collections.Counter([pair[0] for pair in ordered_pairs])
        keys = [key for key, value in counter.items() if value > 1]
        raise ValueError("Duplicate keys in DeepSpeed config: {}".format(keys))
    return d


# ═══════════════════════════════════════════════════════════════
# DES-LOC Configuration Validation & Utilities (M192)
# ═══════════════════════════════════════════════════════════════
import warnings as _m192_warnings
import math as _m192_math


def validate_desloc_config(desloc_dict):
    """Validate DES-LOC sub-dictionary from ds_config.

    Performs comprehensive validation:
    1. Half-life ordering: Kx >= 1, Ku >= Kx, Kv >= Ku
    2. Clip radius sanity: rho > 0
    3. Outer optimizer compatibility
    4. Muon compatibility (Kv ignored when muon_compat=True)
    5. Nesterov momentum bounds
    6. Warmup steps non-negative

    Returns validated dict with defaults filled in.

    Knuth critique (user perspective):
    - If Kx > total_steps, DES-LOC never syncs → divergence risk
    - If clip_radius too small, gradient signal destroyed
    - If nesterov_momentum >= 1.0, Nesterov diverges

    Knuth critique (system perspective):
    - muon_compat + Kv != Ku wastes config space
    - probabilistic sync with small Kx adds variance without saving comm
    """
    if not isinstance(desloc_dict, dict):
        return desloc_dict

    # Fill defaults
    d = dict(desloc_dict)
    d.setdefault('enabled', False)
    d.setdefault('Kx', 32)
    d.setdefault('Ku', 96)
    d.setdefault('Kv', 192)
    d.setdefault('clip_radius', 1.0)
    d.setdefault('outer_optimizer', 'averaging')
    d.setdefault('nesterov_momentum', 0.9)
    d.setdefault('nesterov_lr', 1.0)
    d.setdefault('muon_compat', False)
    d.setdefault('warmup_sync_steps', 0)
    d.setdefault('comm_logging', False)
    d.setdefault('probabilistic_sync', False)

    if not d['enabled']:
        return d

    kx, ku, kv = d['Kx'], d['Ku'], d['Kv']

    # 1. Basic bounds
    assert kx >= 1, f"DES-LOC Kx must be >= 1, got {kx}"
    assert ku >= 1, f"DES-LOC Ku must be >= 1, got {ku}"
    assert kv >= 1, f"DES-LOC Kv must be >= 1, got {kv}"

    # 2. Half-life ordering
    if ku < kx:
        _m192_warnings.warn(
            f"DES-LOC: Ku ({ku}) < Kx ({kx}). "
            f"Paper recommends Ku >= Kx (first moment decays faster).",
            UserWarning)
    if kv < ku:
        _m192_warnings.warn(
            f"DES-LOC: Kv ({kv}) < Ku ({ku}). "
            f"Paper recommends Kv >= Ku (second moment is most stable).",
            UserWarning)

    # 3. Clip radius
    assert d['clip_radius'] > 0,         f"DES-LOC: clip_radius must be > 0, got {d['clip_radius']}"

    # 4. Outer optimizer
    outer = d['outer_optimizer']
    assert outer in ('averaging', 'nesterov'),         f"DES-LOC: outer_optimizer must be 'averaging' or 'nesterov', got '{outer}'"

    # 5. Nesterov bounds
    if outer == 'nesterov':
        mom = d['nesterov_momentum']
        assert 0.0 < mom < 1.0,             f"DES-LOC: nesterov_momentum must be in (0, 1), got {mom}"
        olr = d['nesterov_lr']
        assert olr > 0,             f"DES-LOC: nesterov_lr must be > 0, got {olr}"

    # 6. Muon compatibility
    if d['muon_compat']:
        if kv != ku:
            _m192_warnings.warn(
                f"DES-LOC muon_compat=True: Kv ({kv}) is ignored for Muon "
                f"(single-momentum optimizer). Setting Kv=Ku={ku}.",
                UserWarning)
            d['Kv'] = ku

    # 7. Warmup steps
    assert d['warmup_sync_steps'] >= 0,         f"DES-LOC: warmup_sync_steps must be >= 0, got {d['warmup_sync_steps']}"

    return d


def compute_desloc_comm_estimate(param_bytes, total_steps, Kx=32, Ku=96, Kv=192,
                                  world_size=1, bandwidth_gbps=25.0):
    """Estimate DES-LOC communication cost and compare with DDP.

    Returns dict with:
    - ddp_total_bytes: what DDP would communicate
    - desloc_total_bytes: what DES-LOC communicates
    - reduction_factor: ddp / desloc
    - estimated_time_saved_sec: wallclock savings estimate

    This is a planning tool — actual savings depend on overlap.
    """
    # DDP: allreduce all params every step (3 states: grad, exp_avg, exp_avg_sq)
    bytes_per_allreduce = param_bytes * 2 * (world_size - 1) / max(world_size, 1)
    ddp_total = bytes_per_allreduce * 3 * total_steps

    # DES-LOC: allreduce at different rates
    x_syncs = total_steps // max(Kx, 1)
    u_syncs = total_steps // max(Ku, 1)
    v_syncs = total_steps // max(Kv, 1)
    desloc_total = bytes_per_allreduce * (x_syncs + u_syncs + v_syncs)

    bandwidth_bps = bandwidth_gbps * 1e9 / 8
    ddp_time = ddp_total / max(bandwidth_bps, 1)
    desloc_time = desloc_total / max(bandwidth_bps, 1)

    return {
        'ddp_total_bytes': int(ddp_total),
        'desloc_total_bytes': int(desloc_total),
        'reduction_factor': round(ddp_total / max(desloc_total, 1), 2),
        'x_syncs': x_syncs,
        'u_syncs': u_syncs,
        'v_syncs': v_syncs,
        'ddp_time_sec': round(ddp_time, 2),
        'desloc_time_sec': round(desloc_time, 2),
        'time_saved_sec': round(ddp_time - desloc_time, 2),
    }


def format_desloc_config_summary(config_dict):
    """Format DES-LOC configuration as human-readable summary."""
    if not isinstance(config_dict, dict) or not config_dict.get('enabled'):
        return "DES-LOC: disabled"

    lines = ["DES-LOC Configuration:"]
    lines.append(f"  Sync periods: Kx={config_dict.get('Kx', '?')}, "
                 f"Ku={config_dict.get('Ku', '?')}, "
                 f"Kv={config_dict.get('Kv', '?')}")
    lines.append(f"  Clip radius (rho): {config_dict.get('clip_radius', 1.0)}")
    lines.append(f"  Outer optimizer: {config_dict.get('outer_optimizer', 'averaging')}")
    if config_dict.get('outer_optimizer') == 'nesterov':
        lines.append(f"  Nesterov momentum: {config_dict.get('nesterov_momentum', 0.9)}")
        lines.append(f"  Nesterov lr: {config_dict.get('nesterov_lr', 1.0)}")
    if config_dict.get('muon_compat'):
        lines.append("  Muon compatibility: enabled (Kv=Ku)")
    if config_dict.get('warmup_sync_steps', 0) > 0:
        lines.append(f"  Warmup sync steps: {config_dict['warmup_sync_steps']}")

    # Half-life info
    beta1, beta2 = 0.9, 0.999  # defaults
    h1 = -1.0 / _m192_math.log2(beta1) if 0 < beta1 < 1 else float('inf')
    h2 = -1.0 / _m192_math.log2(beta2) if 0 < beta2 < 1 else float('inf')
    lines.append(f"  Half-life ratio (h_v/h_u): {h2/max(h1, 1e-6):.0f}x "
                 f"(h_u={h1:.0f}, h_v={h2:.0f} steps)")

    return "\n".join(lines)


def desloc_config_to_env_vars(config_dict):
    """Convert DES-LOC config dict to environment variable dict.

    For use with launcher (runner.py, launch.py) to propagate
    DES-LOC settings to worker processes.
    """
    if not isinstance(config_dict, dict):
        return {}
    env = {}
    if config_dict.get('enabled'):
        env['DESLOC_ENABLED'] = '1'
        env['DESLOC_KX'] = str(config_dict.get('Kx', 32))
        env['DESLOC_KU'] = str(config_dict.get('Ku', 96))
        env['DESLOC_KV'] = str(config_dict.get('Kv', 192))
        env['DESLOC_CLIP_RADIUS'] = str(config_dict.get('clip_radius', 1.0))
        env['DESLOC_OUTER_OPT'] = config_dict.get('outer_optimizer', 'averaging')
        if config_dict.get('muon_compat'):
            env['DESLOC_MUON_COMPAT'] = '1'
    return env


def desloc_config_from_env():
    """Read DES-LOC config from environment variables.

    Inverse of desloc_config_to_env_vars().
    """
    import os
    if os.environ.get('DESLOC_ENABLED', '').lower() not in ('1', 'true'):
        return {'enabled': False}
    return {
        'enabled': True,
        'Kx': int(os.environ.get('DESLOC_KX', 32)),
        'Ku': int(os.environ.get('DESLOC_KU', 96)),
        'Kv': int(os.environ.get('DESLOC_KV', 192)),
        'clip_radius': float(os.environ.get('DESLOC_CLIP_RADIUS', 1.0)),
        'outer_optimizer': os.environ.get('DESLOC_OUTER_OPT', 'averaging'),
        'muon_compat': os.environ.get('DESLOC_MUON_COMPAT', '') == '1',
    }


# End M192
