# optimizer_param_scheduler.py

import math


class OptimizerParamScheduler:
    """
    Scheduler for optimizer learning rate and weight decay.

    Supports:
      - lr_decay_style: 'cosine', 'linear', 'constant', 'WSD'
      - Linear warmup from init_lr to max_lr over lr_warmup_steps
      - Decay from max_lr to min_lr over lr_decay_steps
      - tier_lr_multiplier: final LR is multiplied by this factor
      - Weight decay: linear increase from start_wd to end_wd over wd_incr_steps
        with wd_incr_style in ('linear', 'cosine', 'constant')
      - Checkpoint save/restore via state_dict / load_state_dict
    """

    def __init__(
        self,
        optimizer,
        init_lr,
        max_lr,
        min_lr,
        lr_warmup_steps,
        lr_decay_steps,
        lr_decay_style,
        start_wd,
        end_wd,
        wd_incr_steps,
        wd_incr_style,
        use_checkpoint_opt_param_scheduler=True,
        override_opt_param_scheduler=False,
        wsd_decay_steps=None,
        tier_lr_multiplier=1.0,
    ):
        self.optimizer = optimizer

        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps

        assert lr_decay_style in ('cosine', 'linear', 'constant', 'WSD'), \
            f"Unsupported lr_decay_style: {lr_decay_style}"
        self.lr_decay_style = lr_decay_style

        self.start_wd = start_wd
        self.end_wd = end_wd
        self.wd_incr_steps = wd_incr_steps

        assert wd_incr_style in ('linear', 'cosine', 'constant'), \
            f"Unsupported wd_incr_style: {wd_incr_style}"
        self.wd_incr_style = wd_incr_style

        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        self.override_opt_param_scheduler = override_opt_param_scheduler

        # WSD: steps spent in the final decay phase
        self.wsd_decay_steps = wsd_decay_steps if wsd_decay_steps is not None else 0

        self.tier_lr_multiplier = tier_lr_multiplier

        self.num_steps = 0

        # Validate
        assert self.min_lr >= 0.0, "min_lr must be non-negative"
        assert self.max_lr >= self.min_lr, "max_lr must be >= min_lr"
        assert self.lr_warmup_steps >= 0, "lr_warmup_steps must be non-negative"
        assert self.lr_decay_steps >= 0, "lr_decay_steps must be non-negative"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lr_scale(self) -> float:
        """
        Return a value in [0, 1] representing how far through decay we are,
        based on the current step and decay style.
        """
        step = self.num_steps

        # ---- Warmup phase ----
        if self.lr_warmup_steps > 0 and step <= self.lr_warmup_steps:
            # Linear warmup: 0 → 1
            return float(step) / float(self.lr_warmup_steps)

        # After warmup
        steps_after_warmup = step - self.lr_warmup_steps

        if self.lr_decay_style == 'constant':
            return 1.0

        if self.lr_decay_style == 'WSD':
            return self._wsd_lr_scale(steps_after_warmup)

        # ---- Decay phase (linear / cosine) ----
        if self.lr_decay_steps == 0:
            return 1.0

        decay_ratio = min(float(steps_after_warmup) / float(self.lr_decay_steps), 1.0)

        if self.lr_decay_style == 'linear':
            # 1 → 0 linearly
            coeff = 1.0 - decay_ratio
        elif self.lr_decay_style == 'cosine':
            # cosine annealing: 1 → 0
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        else:
            coeff = 1.0

        return coeff

    def _wsd_lr_scale(self, steps_after_warmup: int) -> float:
        """
        Warmup-Stable-Decay schedule (after warmup).

        Layout after warmup:
          [0 … lr_decay_steps - wsd_decay_steps)  → stable at max_lr  (scale = 1)
          [lr_decay_steps - wsd_decay_steps … lr_decay_steps)          → cosine decay
          >= lr_decay_steps                                              → min_lr
        """
        if self.lr_decay_steps <= 0:
            return 1.0

        stable_steps = max(self.lr_decay_steps - self.wsd_decay_steps, 0)

        if steps_after_warmup < stable_steps:
            # Stable phase
            return 1.0

        decay_start = steps_after_warmup - stable_steps
        if self.wsd_decay_steps > 0:
            decay_ratio = min(float(decay_start) / float(self.wsd_decay_steps), 1.0)
        else:
            decay_ratio = 1.0

        # Cosine decay 1 → 0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return coeff

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_lr(self, param_group) -> float:
        """
        Compute the learning rate for *param_group*.

        If the group contains a ``lr_mult`` key the base LR is additionally
        multiplied by that factor (useful for per-layer scaling).
        """
        scale = self._get_lr_scale()

        # Interpolate between min_lr and max_lr
        lr = self.min_lr + scale * (self.max_lr - self.min_lr)

        # During warmup we interpolate from init_lr instead of min_lr
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            warmup_ratio = float(self.num_steps) / float(self.lr_warmup_steps)
            lr = self.init_lr + warmup_ratio * (self.max_lr - self.init_lr)

        # Apply tier multiplier (e.g. 0.8 for A6000, 1.0 for H100)
        lr = lr * self.tier_lr_multiplier

        # Per-group learning rate multiplier (optional)
        lr_mult = param_group.get('lr_mult', 1.0)
        lr = lr * lr_mult

        return lr

    def get_wd(self) -> float:
        """
        Compute the weight decay for the current step.
        """
        if self.wd_incr_steps == 0 or self.start_wd == self.end_wd:
            return self.end_wd

        wd_ratio = min(float(self.num_steps) / float(self.wd_incr_steps), 1.0)

        if self.wd_incr_style == 'constant':
            coeff = 1.0
        elif self.wd_incr_style == 'linear':
            coeff = wd_ratio
        elif self.wd_incr_style == 'cosine':
            # Cosine increase: 0 → 1
            coeff = 0.5 * (1.0 - math.cos(math.pi * wd_ratio))
        else:
            coeff = wd_ratio

        wd = self.start_wd + coeff * (self.end_wd - self.start_wd)
        return wd

    def step(self, increment: int = 1):
        """Advance the scheduler by *increment* steps and update optimizer."""
        self.num_steps += increment
        self._update_optimizer()

    def _update_optimizer(self):
        """Push the current LR / WD values into all optimizer param groups."""
        wd = self.get_wd()
        for group in self.optimizer.param_groups:
            group['lr'] = self.get_lr(group)
            if group.get('weight_decay', None) is not None:
                group['weight_decay'] = wd

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def get_last_lr(self) -> list:
        """Return last computed LR for each param group.

        Compatibility shim: desloc_engine calls scheduler.get_last_lr()[0]
        which is a torch LambdaLR method. We emulate it here.
        """
        return [group.get('lr', self.max_lr) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Return a dict containing all scheduler state needed for checkpointing."""
        return {
            'num_steps': self.num_steps,
            'init_lr': self.init_lr,
            'max_lr': self.max_lr,
            'min_lr': self.min_lr,
            'lr_warmup_steps': self.lr_warmup_steps,
            'lr_decay_steps': self.lr_decay_steps,
            'lr_decay_style': self.lr_decay_style,
            'start_wd': self.start_wd,
            'end_wd': self.end_wd,
            'wd_incr_steps': self.wd_incr_steps,
            'wd_incr_style': self.wd_incr_style,
            'wsd_decay_steps': self.wsd_decay_steps,
            'tier_lr_multiplier': self.tier_lr_multiplier,
        }

    def load_state_dict(self, state_dict: dict):
        """
        Restore scheduler state from *state_dict*.

        Behaviour is controlled by the two flags set at construction time:

        * ``override_opt_param_scheduler=True``: ignore the checkpoint and
          keep the current (freshly constructed) hyper-parameters.  Only
          ``num_steps`` is restored so that the step counter stays in sync.
        * ``use_checkpoint_opt_param_scheduler=False``: skip loading
          entirely (caller manages the scheduler externally).
        """
        if not self.use_checkpoint_opt_param_scheduler:
            return

        if self.override_opt_param_scheduler:
            # Only restore the step counter; keep all other params as-is.
            if 'num_steps' in state_dict:
                self.num_steps = state_dict['num_steps']
            self._update_optimizer()
            return

        # Full restore
        self.num_steps = state_dict.get('num_steps', self.num_steps)
        self.init_lr = state_dict.get('init_lr', self.init_lr)
        self.max_lr = state_dict.get('max_lr', self.max_lr)
        self.min_lr = state_dict.get('min_lr', self.min_lr)
        self.lr_warmup_steps = state_dict.get('lr_warmup_steps', self.lr_warmup_steps)
        self.lr_decay_steps = state_dict.get('lr_decay_steps', self.lr_decay_steps)
        self.lr_decay_style = state_dict.get('lr_decay_style', self.lr_decay_style)
        self.start_wd = state_dict.get('start_wd', self.start_wd)
        self.end_wd = state_dict.get('end_wd', self.end_wd)
        self.wd_incr_steps = state_dict.get('wd_incr_steps', self.wd_incr_steps)
        self.wd_incr_style = state_dict.get('wd_incr_style', self.wd_incr_style)
        self.wsd_decay_steps = state_dict.get('wsd_decay_steps', self.wsd_decay_steps)
        self.tier_lr_multiplier = state_dict.get('tier_lr_multiplier', self.tier_lr_multiplier)

        self._update_optimizer()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OptimizerParamScheduler("
            f"lr_decay_style={self.lr_decay_style}, "
            f"max_lr={self.max_lr}, min_lr={self.min_lr}, "
            f"warmup_steps={self.lr_warmup_steps}, "
            f"decay_steps={self.lr_decay_steps}, "
            f"num_steps={self.num_steps}, "
            f"tier_lr_multiplier={self.tier_lr_multiplier})"
        )
