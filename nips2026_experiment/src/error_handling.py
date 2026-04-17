#!/usr/bin/env python3
"""
===============================================================================
M043: Error Handling and Recovery Module
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module provides comprehensive error handling, retry mechanisms,
graceful degradation, and recovery strategies for distributed training.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Features:
- Custom exception hierarchy for distributed training
- Retry decorators with exponential backoff
- Graceful degradation handlers
- Checkpoint-based recovery
- Error aggregation and reporting
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M043"

import os
import sys
import json
import traceback
import functools
import time
import signal
import threading
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    List, Dict, Optional, Any, Callable, TypeVar, Union,
    Tuple, Type, Generic
)
from datetime import datetime, timedelta
from enum import Enum, auto
from contextlib import contextmanager
from collections import deque
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# PART 1: EXCEPTION HIERARCHY
# =============================================================================

class DESLOCException(Exception):
    """Base exception for DES-LOC framework."""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
        }


class CommunicationError(DESLOCException):
    """Errors in distributed communication."""
    pass


class SynchronizationError(DESLOCException):
    """Errors during synchronization operations."""
    pass


class CheckpointError(DESLOCException):
    """Errors during checkpoint save/load."""
    pass


class OutOfMemoryError(DESLOCException):
    """GPU/CPU out of memory errors."""
    pass


class ConfigurationError(DESLOCException):
    """Configuration validation errors."""
    pass


class TimeoutError(DESLOCException):
    """Operation timeout errors."""
    pass


class ConvergenceError(DESLOCException):
    """Training convergence issues."""
    pass


class DataLoadingError(DESLOCException):
    """Data loading and preprocessing errors."""
    pass


class HardwareError(DESLOCException):
    """Hardware-related errors (GPU failures, etc.)."""
    pass


# =============================================================================
# PART 2: ERROR SEVERITY AND RECOVERY ACTIONS
# =============================================================================

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    DEBUG = auto()      # Informational, no action needed
    WARNING = auto()    # Potential issue, continue with caution
    ERROR = auto()      # Recoverable error, attempt recovery
    CRITICAL = auto()   # Severe error, may need to abort
    FATAL = auto()      # Unrecoverable, must abort


class RecoveryAction(Enum):
    """Possible recovery actions."""
    IGNORE = auto()           # Continue without action
    RETRY = auto()            # Retry the operation
    SKIP = auto()             # Skip current item/batch
    FALLBACK = auto()         # Use fallback method
    CHECKPOINT_RESTORE = auto()  # Restore from checkpoint
    REDUCE_BATCH = auto()     # Reduce batch size
    RESTART_WORKER = auto()   # Restart failed worker
    ABORT = auto()            # Abort training


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    step: int = 0
    rank: int = 0
    world_size: int = 1
    batch_idx: int = 0
    epoch: int = 0
    checkpoint_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    exception_type: str
    message: str
    severity: ErrorSeverity
    context: ErrorContext
    traceback: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    recovery_action: Optional[RecoveryAction] = None
    recovered: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'exception_type': self.exception_type,
            'message': self.message,
            'severity': self.severity.name,
            'context': asdict(self.context),
            'traceback': self.traceback,
            'timestamp': self.timestamp,
            'recovery_action': self.recovery_action.name if self.recovery_action else None,
            'recovered': self.recovered,
        }


# =============================================================================
# PART 3: RETRY DECORATORS
# =============================================================================

class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        on_retry: Optional[Callable[[Exception, int], None]] = None,
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.on_retry = on_retry
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random())
        
        return delay


def retry(config: RetryConfig = None) -> Callable[[F], F]:
    """Decorator for retry with exponential backoff."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = config.get_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        
                        if config.on_retry:
                            config.on_retry(e, attempt)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed for {func.__name__}"
                        )
            
            raise last_exception
        
        return wrapper
    
    return decorator


def retry_on_oom(
    reduce_factor: float = 0.5,
    min_batch_size: int = 1,
    max_attempts: int = 3,
) -> Callable[[F], F]:
    """Decorator to handle OOM by reducing batch size."""
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, batch_size: int = None, **kwargs):
            current_batch_size = batch_size
            
            for attempt in range(max_attempts):
                try:
                    if current_batch_size is not None:
                        kwargs['batch_size'] = current_batch_size
                    return func(*args, **kwargs)
                except (RuntimeError, MemoryError) as e:
                    if 'out of memory' in str(e).lower() or isinstance(e, MemoryError):
                        if current_batch_size is None or current_batch_size <= min_batch_size:
                            raise OutOfMemoryError(
                                f"OOM even with minimum batch size: {min_batch_size}",
                                {'original_error': str(e)}
                            )
                        
                        current_batch_size = max(
                            min_batch_size,
                            int(current_batch_size * reduce_factor)
                        )
                        logger.warning(
                            f"OOM detected. Reducing batch size to {current_batch_size}"
                        )
                        
                        # Clear CUDA cache
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except ImportError:
                            pass
                    else:
                        raise
            
            raise OutOfMemoryError("Failed after all OOM recovery attempts")
        
        return wrapper
    
    return decorator


# =============================================================================
# PART 4: TIMEOUT HANDLING
# =============================================================================

class TimeoutHandler:
    """Handles operation timeouts."""
    
    def __init__(self, timeout_seconds: float, error_message: str = "Operation timed out"):
        self.timeout_seconds = timeout_seconds
        self.error_message = error_message
        self._timer: Optional[threading.Timer] = None
        self._timed_out = False
    
    def _timeout_handler(self):
        """Called when timeout occurs."""
        self._timed_out = True
        logger.error(f"Timeout after {self.timeout_seconds}s: {self.error_message}")
    
    def __enter__(self):
        self._timed_out = False
        self._timer = threading.Timer(self.timeout_seconds, self._timeout_handler)
        self._timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._timer:
            self._timer.cancel()
        
        if self._timed_out:
            raise TimeoutError(self.error_message, {
                'timeout_seconds': self.timeout_seconds
            })
        
        return False
    
    @property
    def timed_out(self) -> bool:
        return self._timed_out


@contextmanager
def timeout(seconds: float, message: str = "Operation timed out"):
    """Context manager for operation timeout."""
    handler = TimeoutHandler(seconds, message)
    with handler:
        yield handler


# =============================================================================
# PART 5: ERROR AGGREGATOR
# =============================================================================

class ErrorAggregator:
    """Aggregates and analyzes errors across training."""
    
    def __init__(self, max_history: int = 1000, error_threshold: int = 10):
        self.max_history = max_history
        self.error_threshold = error_threshold
        self.errors: deque = deque(maxlen=max_history)
        self.error_counts: Dict[str, int] = {}
        self.recovery_stats: Dict[RecoveryAction, int] = {}
        self._lock = threading.Lock()
    
    def record(self, error: ErrorRecord):
        """Record an error."""
        with self._lock:
            self.errors.append(error)
            
            # Update counts
            key = error.exception_type
            self.error_counts[key] = self.error_counts.get(key, 0) + 1
            
            if error.recovery_action:
                self.recovery_stats[error.recovery_action] = \
                    self.recovery_stats.get(error.recovery_action, 0) + 1
    
    def should_abort(self) -> bool:
        """Check if training should abort based on error frequency."""
        with self._lock:
            # Check total errors in recent window
            recent_errors = sum(
                1 for e in self.errors
                if e.severity in (ErrorSeverity.ERROR, ErrorSeverity.CRITICAL)
            )
            
            if recent_errors >= self.error_threshold:
                return True
            
            # Check for fatal errors
            fatal_count = sum(
                1 for e in self.errors
                if e.severity == ErrorSeverity.FATAL
            )
            
            return fatal_count > 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        with self._lock:
            severity_counts = {}
            for e in self.errors:
                sev = e.severity.name
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            
            return {
                'total_errors': len(self.errors),
                'by_type': dict(self.error_counts),
                'by_severity': severity_counts,
                'recovery_actions': {
                    k.name: v for k, v in self.recovery_stats.items()
                },
                'should_abort': self.should_abort(),
            }
    
    def save_report(self, path: Path):
        """Save error report to file."""
        with self._lock:
            report = {
                'summary': self.get_summary(),
                'errors': [e.to_dict() for e in self.errors],
                'generated_at': datetime.now().isoformat(),
            }
            
            with open(path, 'w') as f:
                json.dump(report, f, indent=2)


# =============================================================================
# PART 6: RECOVERY STRATEGIES
# =============================================================================

class RecoveryStrategy:
    """Base class for recovery strategies."""
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this strategy can handle the error."""
        raise NotImplementedError
    
    def recover(self, error: Exception, context: ErrorContext) -> RecoveryAction:
        """Attempt recovery and return action taken."""
        raise NotImplementedError


class CheckpointRecovery(RecoveryStrategy):
    """Recovery by restoring from checkpoint."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        return context.checkpoint_path is not None or self._find_latest_checkpoint() is not None
    
    def recover(self, error: Exception, context: ErrorContext) -> RecoveryAction:
        checkpoint = context.checkpoint_path or self._find_latest_checkpoint()
        
        if checkpoint and Path(checkpoint).exists():
            logger.info(f"Recovering from checkpoint: {checkpoint}")
            return RecoveryAction.CHECKPOINT_RESTORE
        
        return RecoveryAction.ABORT
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file."""
        if not self.checkpoint_dir.exists():
            return None
        
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        
        return str(max(checkpoints, key=lambda p: p.stat().st_mtime))


class GradualDegradation(RecoveryStrategy):
    """Recovery by gradually reducing resource usage."""
    
    def __init__(self, min_batch_size: int = 1, min_workers: int = 1):
        self.min_batch_size = min_batch_size
        self.min_workers = min_workers
        self.current_reduction_level = 0
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        return isinstance(error, (OutOfMemoryError, MemoryError)) or \
               'out of memory' in str(error).lower()
    
    def recover(self, error: Exception, context: ErrorContext) -> RecoveryAction:
        self.current_reduction_level += 1
        
        if self.current_reduction_level <= 3:
            logger.info(f"Reducing batch size (level {self.current_reduction_level})")
            return RecoveryAction.REDUCE_BATCH
        
        return RecoveryAction.ABORT


class CommunicationRecovery(RecoveryStrategy):
    """Recovery from communication failures."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.retry_count = 0
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        return isinstance(error, (CommunicationError, SynchronizationError))
    
    def recover(self, error: Exception, context: ErrorContext) -> RecoveryAction:
        self.retry_count += 1
        
        if self.retry_count <= self.max_retries:
            logger.info(f"Retrying communication (attempt {self.retry_count})")
            return RecoveryAction.RETRY
        
        logger.warning("Communication recovery failed, restarting worker")
        self.retry_count = 0
        return RecoveryAction.RESTART_WORKER


# =============================================================================
# PART 7: ERROR HANDLER
# =============================================================================

class ErrorHandler:
    """Central error handling coordinator."""
    
    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        error_log_path: Optional[Path] = None,
    ):
        self.aggregator = ErrorAggregator()
        self.strategies: List[RecoveryStrategy] = []
        self.error_log_path = error_log_path
        
        # Add default strategies
        if checkpoint_dir:
            self.strategies.append(CheckpointRecovery(checkpoint_dir))
        self.strategies.append(GradualDegradation())
        self.strategies.append(CommunicationRecovery())
    
    def add_strategy(self, strategy: RecoveryStrategy):
        """Add a recovery strategy."""
        self.strategies.append(strategy)
    
    def handle(
        self,
        error: Exception,
        context: ErrorContext,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
    ) -> RecoveryAction:
        """Handle an error and determine recovery action."""
        # Create error record
        record = ErrorRecord(
            exception_type=type(error).__name__,
            message=str(error),
            severity=severity,
            context=context,
            traceback=traceback.format_exc(),
        )
        
        # Log the error
        log_level = {
            ErrorSeverity.DEBUG: logging.DEBUG,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.FATAL: logging.CRITICAL,
        }.get(severity, logging.ERROR)
        
        logger.log(log_level, f"Error in {context.operation}: {error}")
        
        # Try recovery strategies
        action = RecoveryAction.ABORT
        for strategy in self.strategies:
            if strategy.can_recover(error, context):
                action = strategy.recover(error, context)
                if action != RecoveryAction.ABORT:
                    record.recovered = True
                    break
        
        record.recovery_action = action
        self.aggregator.record(record)
        
        # Check if we should abort
        if self.aggregator.should_abort():
            logger.critical("Error threshold exceeded, aborting training")
            action = RecoveryAction.ABORT
        
        # Save error log if configured
        if self.error_log_path:
            self.aggregator.save_report(self.error_log_path)
        
        return action
    
    @contextmanager
    def catch(
        self,
        context: ErrorContext,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        reraise: bool = True,
    ):
        """Context manager for catching and handling errors."""
        try:
            yield
        except Exception as e:
            action = self.handle(e, context, severity)
            
            if action == RecoveryAction.ABORT and reraise:
                raise
            elif action == RecoveryAction.IGNORE:
                pass
            elif reraise:
                raise


# =============================================================================
# PART 8: GRACEFUL SHUTDOWN
# =============================================================================

class GracefulShutdown:
    """Handles graceful shutdown on signals."""
    
    def __init__(self, cleanup_func: Optional[Callable] = None):
        self.cleanup_func = cleanup_func
        self.shutdown_requested = False
        self._original_handlers: Dict[int, Any] = {}
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        
        if self.cleanup_func:
            try:
                self.cleanup_func()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        # Register signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._original_handlers[sig] = signal.signal(sig, self._signal_handler)
            except (OSError, ValueError):
                pass  # Signal handling not available (e.g., not main thread)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original handlers
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass
        
        return False


# =============================================================================
# PART 9: MAIN / DEMO
# =============================================================================

def demo():
    """Demonstrate error handling capabilities."""
    print("=" * 70)
    print("DES-LOC Error Handling Demo")
    print("=" * 70)
    
    # Setup error handler
    handler = ErrorHandler(
        checkpoint_dir=Path("./checkpoints"),
        error_log_path=Path("./error_log.json"),
    )
    
    # Demo retry decorator
    @retry(RetryConfig(max_attempts=3, initial_delay=0.1))
    def flaky_operation():
        import random
        if random.random() < 0.7:
            raise CommunicationError("Random failure")
        return "Success!"
    
    # Demo error handling context
    context = ErrorContext(
        operation="demo_operation",
        step=100,
        rank=0,
    )
    
    try:
        with handler.catch(context):
            raise OutOfMemoryError("Simulated OOM")
    except OutOfMemoryError:
        print("Caught and handled OOM error")
    
    # Print summary
    print("\nError Summary:")
    print(json.dumps(handler.aggregator.get_summary(), indent=2))
    
    print("\n[M043] Error Handling Demo - COMPLETED")


if __name__ == "__main__":
    demo()
