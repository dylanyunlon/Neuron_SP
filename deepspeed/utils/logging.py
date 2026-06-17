# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
import logging
import sys
import os
import torch
from deepspeed.utils.torch import required_torch_version

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LoggerFactory:

    @staticmethod
    def create_logger(name=None, level=logging.WARNING):
        """create a logger

        Args:
            name (str): name of the logger
            level: level of logger

        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] "
                                      "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        if required_torch_version(min_version=2.6) and os.getenv("DISABLE_LOGS_WHILE_COMPILING", "0") == "1":
            excluded_set = {
                item.strip()
                for item in os.getenv("LOGGER_METHODS_TO_EXCLUDE_FROM_DISABLE", "").split(",")
            }
            ignore_set = {'info', 'debug', 'error', 'warning', 'critical', 'exception', 'isEnabledFor'} - excluded_set
            for method in ignore_set:
                original_logger = getattr(logger_, method)
                torch._dynamo.config.ignore_logger_methods.add(original_logger)
        return logger_


logger = LoggerFactory.create_logger(name="DeepSpeed", level=logging.WARNING)


@functools.lru_cache(None)
def warning_once(*args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    logger.warning(*args, **kwargs)


logger.warning_once = warning_once


def print_configuration(args, name):
    logger.info("{}:".format(name))
    for arg in sorted(vars(args)):
        dots = "." * (29 - len(arg))
        logger.info("  {} {} {}".format(arg, dots, getattr(args, arg)))


def get_dist_msg(message, ranks=None):
    from deepspeed import comm as dist
    """Return a message with rank prefix when one of following conditions is met:

      + not dist.is_initialized()
      + dist.get_rank() in ranks if ranks is not None or ranks = [-1]

    If neither is met, `None` is returned.

    Example: "hello" => "[Rank 0] hello"

    Args:
        message (str)
        ranks (list)
    """
    should_log = not dist.is_initialized()
    ranks = ranks or []
    my_rank = dist.get_rank() if dist.is_initialized() else -1
    if ranks and not should_log:
        should_log = ranks[0] == -1
        should_log = should_log or (my_rank in set(ranks))
    if should_log:
        return "[Rank {}] {}".format(my_rank, message)
    else:
        return None


def log_dist(message, ranks=None, level=logging.INFO):
    """Log message when get_dist_msg() deems it should be logged, see its docstring for details.

    Args:
        message (str)
        ranks (list)
        level (int)
    """
    final_message = get_dist_msg(message, ranks)
    if final_message is not None:
        logger.log(level, final_message)


def print_dist(message, ranks=None):
    """print message when get_dist_msg() deems it should be logged, see its docstring for details.

    Use this function instead of `log_dist` when the log level shouldn't impact whether the message should be printed or not.

    Args:
        message (str)
        ranks (list)
    """
    final_message = get_dist_msg(message, ranks)
    if final_message is not None:
        print(final_message)


@functools.lru_cache(None)
def _log_dist_once_cached(message, ranks_key, level):
    ranks_arg = list(ranks_key) if ranks_key is not None else None
    log_dist(message, ranks=ranks_arg, level=level)


def log_dist_once(message, ranks=None, level=logging.INFO):
    # Identical to `log_dist`, but will emit each unique message only once per process.
    # ranks is a list which is unhashable, so convert to tuple for caching
    ranks_key = tuple(ranks) if ranks is not None else None
    _log_dist_once_cached(message, ranks_key, level)


logger.log_dist_once = log_dist_once


def print_json_dist(message, ranks=None, path=None):
    from deepspeed import comm as dist
    """Print message when one of following condition meets

    + not dist.is_initialized()
    + dist.get_rank() in ranks if ranks is not None or ranks = [-1]

    Args:
        message (str)
        ranks (list)
        path (str)

    """
    should_log = not dist.is_initialized()
    ranks = ranks or []
    my_rank = dist.get_rank() if dist.is_initialized() else -1
    if ranks and not should_log:
        should_log = ranks[0] == -1
        should_log = should_log or (my_rank in set(ranks))
    if should_log:
        message['rank'] = my_rank
        import json
        with open(path, 'w') as outfile:
            json.dump(message, outfile)
            os.fsync(outfile)


def get_log_level_from_string(log_level_str):
    """converts a log level string into its numerical equivalent. e.g. "info" => `logging.INFO`
    """
    log_level_str = log_level_str.lower()
    if log_level_str not in log_levels:
        raise ValueError(
            f"{log_level_str} is not one of the valid logging levels. Valid log levels are {log_levels.keys()}.")
    return log_levels[log_level_str]


def set_log_level_from_string(log_level_str, custom_logger=None):
    """Sets a log level in the passed `logger` and its handlers from string. e.g. "info" => `logging.INFO`

    Args:
        log_level_str: one of 'debug', 'info', 'warning', 'error', 'critical'
        custom_logger: if `None` will use the default `logger` object
    """
    log_level = get_log_level_from_string(log_level_str)
    if custom_logger is None:
        custom_logger = logger
    custom_logger.setLevel(log_level)
    for handler in custom_logger.handlers:
        handler.setLevel(log_level)


def get_current_level():
    """
    Return logger's current log level
    """
    return logger.getEffectiveLevel()


def should_log_le(max_log_level_str):
    """
    Args:
        max_log_level_str: maximum log level as a string

    Returns ``True`` if the current log_level is less or equal to the specified log level. Otherwise ``False``.

    Example:

        ``should_log_le("info")`` will return ``True`` if the current log level is either ``logging.INFO`` or ``logging.DEBUG``
    """

    if not isinstance(max_log_level_str, str):
        raise ValueError(f"{max_log_level_str} is not a string")

    max_log_level = get_log_level_from_string(max_log_level_str)
    return get_current_level() <= max_log_level


# ---------------------------------------------------------------------------
# DES-LOC rank utilities  (mirrors Megatron 6cf285b23 _rank_utils.py,
# reinterpreted for DeepSpeed distributed state)
# ---------------------------------------------------------------------------

def _safe_get_rank() -> int:
    """Return current process rank without hard-requiring dist to be initialised.

    Mirrors ``megatron.core._rank_utils.safe_get_rank``: tries
    torch.distributed first, then falls back to the ``RANK`` env-var, then 0.
    """
    try:
        import torch.distributed as _dist
        if _dist.is_initialized():
            return _dist.get_rank()
    except Exception:
        pass
    try:
        return int(os.environ.get("RANK", 0))
    except (ValueError, TypeError):
        return 0


def log_single_rank(logger_obj: logging.Logger, level: int, msg: str,
                    *args, rank: int = 0, **kwargs) -> None:
    """Emit *msg* on a single rank only.

    Mirrors ``megatron.core._rank_utils.log_single_rank`` / the
    ``log_single_rank`` helper scattered across Megatron utils (6cf285b23),
    reinterpreted for the DeepSpeed runtime: no ``megatron.core`` import
    path is required.

    Args:
        logger_obj: Standard :class:`logging.Logger` to write through.
        level:      ``logging`` level constant (e.g. ``logging.WARNING``).
        msg:        Message string.
        *args:      Extra positional args forwarded to ``logger_obj.log``.
        rank:       Rank that should emit the log.  Defaults to 0.
        **kwargs:   Extra keyword args forwarded to ``logger_obj.log``.
    """
    if _safe_get_rank() == rank:
        logger_obj.log(level, msg, *args, **kwargs)


# ---------------------------------------------------------------------------
# TierAwareLogger — DES-LOC heterogeneous GPU diagnostic logger
# (mirrors Megatron 6cf285b23 logging-cleanup intent, reinterpreted for
#  DES-LOC: rank-0 may live on an A6000 or an H100, so logs must carry
#  a GPU-tier prefix so operators can tell which tier is reporting)
# ---------------------------------------------------------------------------

def _detect_gpu_tier() -> str:
    """Detect the GPU tier label for the current rank's device.

    Returns a short human-readable label such as ``"H100"``, ``"A6000"``,
    ``"A100"``, ``"V100"``, or ``"GPU"`` (generic fallback).  The label is
    derived from the CUDA device name reported by ``torch.cuda``; no
    external registry or Megatron import is needed.

    Decision boundary (SM major):
      - SM 9.x  (Hopper)  → "H100" / "H200" matched by name, else "H-class"
      - SM 8.x  (Ampere)  → "A100" / "A6000" matched by name, else "A-class"
      - SM 7.x  (Volta/Turing) → "V100"/"T4"  matched by name, else "V-class"
      - else              → "GPU"
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "CPU"
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        props = torch.cuda.get_device_properties(local_rank)
        name: str = props.name  # e.g. "NVIDIA H100 80GB HBM3"
        sm_major: int = props.major
        # Fine-grained name matching first
        for label in ("H200", "H100", "A6000", "A100", "A40", "A10",
                      "V100", "T4", "L40", "L4", "RTX 4090", "RTX 3090"):
            if label.replace(" ", "").lower() in name.replace(" ", "").lower():
                return label
        # Coarse SM-based fallback
        if sm_major >= 9:
            return "H-class"
        if sm_major >= 8:
            return "A-class"
        if sm_major >= 7:
            return "V-class"
        return "GPU"
    except Exception:
        return "GPU"


class TierAwareLogger(logging.LoggerAdapter):
    """Logger adapter that prepends ``[<GPU-tier>]`` to every log record.

    In a DES-LOC heterogeneous cluster (A6000 + H100) rank-0 may land on
    either GPU tier.  Without a tier prefix, a warning like
    "checkpoint mismatch" is ambiguous: was it the A6000 node or the H100
    node?  ``TierAwareLogger`` solves this by appending the tier label once
    at construction time and injecting it into every subsequent message.

    Usage::

        from deepspeed.utils.logging import TierAwareLogger
        import logging, logging.getLogger

        _base = logging.getLogger(__name__)
        logger = TierAwareLogger(_base)

        logger.info("initialised")
        # → "[A6000] initialised"   (on an A6000 rank)
        # → "[H100] initialised"    (on an H100 rank)

    The adapter also exposes ``log_single_rank`` as a convenience method so
    callers don't need to import it separately::

        logger.log_single_rank(logging.WARNING, "only rank-0 sees this")

    Attributes:
        tier_label (str): GPU tier string, e.g. ``"H100"`` or ``"A6000"``.
    """

    def __init__(self, base_logger: logging.Logger, extra: dict = None):
        self.tier_label: str = _detect_gpu_tier()
        super().__init__(base_logger, extra or {})

    # ------------------------------------------------------------------
    # LoggerAdapter protocol
    # ------------------------------------------------------------------

    def process(self, msg, kwargs):
        """Prepend ``[<tier>]`` to the log message."""
        return f"[{self.tier_label}] {msg}", kwargs

    # ------------------------------------------------------------------
    # Rank-filtered helpers
    # ------------------------------------------------------------------

    def log_single_rank(self, level: int, msg: str, *args,
                        rank: int = 0, **kwargs) -> None:
        """Emit tier-prefixed *msg* on *rank* only.

        Combines rank filtering (mirrors Megatron ``log_single_rank``) with
        the tier prefix so a single call handles both concerns.
        """
        if _safe_get_rank() == rank:
            self.log(level, msg, *args, **kwargs)

    def info_rank0(self, msg: str, *args, **kwargs) -> None:
        """Shorthand: INFO on rank 0 only."""
        self.log_single_rank(logging.INFO, msg, *args, **kwargs)

    def warning_rank0(self, msg: str, *args, **kwargs) -> None:
        """Shorthand: WARNING on rank 0 only."""
        self.log_single_rank(logging.WARNING, msg, *args, **kwargs)

    def error_rank0(self, msg: str, *args, **kwargs) -> None:
        """Shorthand: ERROR on rank 0 only."""
        self.log_single_rank(logging.ERROR, msg, *args, **kwargs)


def get_tier_aware_logger(name: str = "DeepSpeed",
                          level: int = logging.WARNING) -> TierAwareLogger:
    """Create (or retrieve) a :class:`TierAwareLogger` wrapping *name*.

    This is the recommended entry-point for DES-LOC components that need
    heterogeneous-cluster-aware logging.  The underlying
    :class:`logging.Logger` is shared via the standard logging registry, so
    calling this function twice with the same *name* returns adapters over
    the same base logger.

    Args:
        name:  Logger name (usually ``__name__``).
        level: Initial log level; only applied if the base logger is new.

    Returns:
        :class:`TierAwareLogger` wrapping the named logger.

    Example::

        logger = get_tier_aware_logger(__name__)
        logger.warning_rank0("checkpoint load failed")
        # → "[H100] checkpoint load failed"  (rank 0, H100 node)
    """
    base = LoggerFactory.create_logger(name=name, level=level)
    return TierAwareLogger(base)
