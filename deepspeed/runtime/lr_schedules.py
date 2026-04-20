# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Implementation of learning rate schedules.

Taken and modified from PyTorch v1.0.1 source
https://github.com/pytorch/pytorch/blob/v1.1.0/torch/optim/lr_scheduler.py
"""

import argparse
from torch.optim import Optimizer
import math
from deepspeed.utils import logger
from torch import tensor, is_tensor

LR_SCHEDULE = 'lr_schedule'
LR_RANGE_TEST = 'LRRangeTest'
ONE_CYCLE = 'OneCycle'
WARMUP_LR = 'WarmupLR'
WARMUP_DECAY_LR = 'WarmupDecayLR'
WARMUP_COSINE_LR = 'WarmupCosineLR'
VALID_LR_SCHEDULES = [LR_RANGE_TEST, ONE_CYCLE, WARMUP_LR, WARMUP_DECAY_LR, WARMUP_COSINE_LR]

LR_RANGE_TEST_MIN_LR = 'lr_range_test_min_lr'
LR_RANGE_TEST_STEP_RATE = 'lr_range_test_step_rate'
LR_RANGE_TEST_STEP_SIZE = 'lr_range_test_step_size'
LR_RANGE_TEST_STAIRCASE = 'lr_range_test_staircase'

EDGE_VALUE = 'edge_value'
MID_VALUE = 'mid_value'

CYCLE_FIRST_STEP_SIZE = 'cycle_first_step_size'
CYCLE_FIRST_STAIR_COUNT = 'cycle_first_stair_count'
CYCLE_SECOND_STEP_SIZE = 'cycle_second_step_size'
CYCLE_SECOND_STAIR_COUNT = 'cycle_second_stair_count'
DECAY_STEP_SIZE = 'decay_step_size'

CYCLE_MIN_LR = 'cycle_min_lr'
CYCLE_MAX_LR = 'cycle_max_lr'
DECAY_LR_RATE = 'decay_lr_rate'

CYCLE_MIN_MOM = 'cycle_min_mom'
CYCLE_MAX_MOM = 'cycle_max_mom'
DECAY_MOM_RATE = 'decay_mom_rate'

WARMUP_MIN_LR = 'warmup_min_lr'
WARMUP_MAX_LR = 'warmup_max_lr'
WARMUP_NUM_STEPS = 'warmup_num_steps'
WARMUP_TYPE = 'warmup_type'
WARMUP_LOG_RATE = 'log'
WARMUP_LINEAR_RATE = 'linear'

WARMUP_MIN_RATIO = 'warmup_min_ratio'
COS_MIN_RATIO = 'cos_min_ratio'

TOTAL_NUM_STEPS = 'total_num_steps'


def add_tuning_arguments(parser):
    group = parser.add_argument_group('Convergence Tuning', 'Convergence tuning configurations')

    # LR scheduler
    group.add_argument('--lr_schedule', type=str, default=None, help='LR schedule for training.')

    # Learning rate range test
    group.add_argument("--lr_range_test_min_lr", type=float, default=0.001, help='Starting lr value.')
    group.add_argument("--lr_range_test_step_rate", type=float, default=1.0, help='scaling rate for LR range test.')
    group.add_argument("--lr_range_test_step_size", type=int, default=1000, help='training steps per LR change.')
    group.add_argument("--lr_range_test_staircase",
                       type=bool,
                       default=False,
                       help='use staircase scaling for LR range test.')

    # OneCycle schedule
    group.add_argument("--cycle_first_step_size",
                       type=int,
                       default=1000,
                       help='size of first step of 1Cycle schedule (training steps).')
    group.add_argument("--cycle_first_stair_count",
                       type=int,
                       default=-1,
                       help='first stair count for 1Cycle schedule.')
    group.add_argument("--cycle_second_step_size",
                       type=int,
                       default=-1,
                       help='size of second step of 1Cycle schedule (default first_step_size).')
    group.add_argument("--cycle_second_stair_count",
                       type=int,
                       default=-1,
                       help='second stair count for 1Cycle schedule.')
    group.add_argument("--decay_step_size",
                       type=int,
                       default=1000,
                       help='size of intervals for applying post cycle decay (training steps).')

    # 1Cycle LR
    group.add_argument("--cycle_min_lr", type=float, default=0.01, help='1Cycle LR lower bound.')
    group.add_argument("--cycle_max_lr", type=float, default=0.1, help='1Cycle LR upper bound.')
    group.add_argument("--decay_lr_rate", type=float, default=0.0, help='post cycle LR decay rate.')

    # 1Cycle Momentum
    group.add_argument('--cycle_momentum', default=False, action='store_true', help='Enable 1Cycle momentum schedule.')
    group.add_argument("--cycle_min_mom", type=float, default=0.8, help='1Cycle momentum lower bound.')
    group.add_argument("--cycle_max_mom", type=float, default=0.9, help='1Cycle momentum upper bound.')
    group.add_argument("--decay_mom_rate", type=float, default=0.0, help='post cycle momentum decay rate.')

    # Warmup LR
    group.add_argument('--warmup_min_lr', type=float, default=0, help='WarmupLR minimum/initial LR value')
    group.add_argument('--warmup_max_lr', type=float, default=0.001, help='WarmupLR maximum LR value.')
    group.add_argument('--warmup_num_steps', type=int, default=1000, help='WarmupLR step count for LR warmup.')
    group.add_argument('--warmup_type',
                       type=str,
                       default=WARMUP_LOG_RATE,
                       help='WarmupLR increasing function during warmup')

    # WarmUP cos LR
    group.add_argument("--warmup_min_ratio", type=float, default=0.01, help='Cosine LR lower bound.')
    group.add_argument("--cos_min_ratio", type=float, default=0.01, help='Cosine LR lower bound.')

    return parser


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = add_tuning_arguments(parser)

    lr_sched_args, unknown_args = parser.parse_known_args()
    return lr_sched_args, unknown_args


def override_lr_range_test_params(args, params):
    if hasattr(args, LR_RANGE_TEST_MIN_LR) and args.lr_range_test_min_lr is not None:
        params[LR_RANGE_TEST_MIN_LR] = args.lr_range_test_min_lr

    if hasattr(args, LR_RANGE_TEST_STEP_RATE) and args.lr_range_test_step_rate is not None:
        params[LR_RANGE_TEST_STEP_RATE] = args.lr_range_test_step_rate

    if hasattr(args, LR_RANGE_TEST_STEP_SIZE) and args.lr_range_test_step_size is not None:
        params[LR_RANGE_TEST_STEP_SIZE] = args.lr_range_test_step_size

    if hasattr(args, LR_RANGE_TEST_STAIRCASE) and args.lr_range_test_staircase is not None:
        params[LR_RANGE_TEST_STAIRCASE] = args.lr_range_test_staircase


def override_1cycle_params(args, params):
    if hasattr(args, CYCLE_FIRST_STEP_SIZE) and args.cycle_first_step_size is not None:
        params[CYCLE_FIRST_STEP_SIZE] = args.cycle_first_step_size

    if hasattr(args, CYCLE_FIRST_STAIR_COUNT) and args.cycle_first_stair_count is not None:
        params[CYCLE_FIRST_STAIR_COUNT] = args.cycle_first_stair_count

    if hasattr(args, CYCLE_SECOND_STEP_SIZE) and args.cycle_second_step_size is not None:
        params[CYCLE_SECOND_STEP_SIZE] = args.cycle_second_step_size

    if hasattr(args, CYCLE_SECOND_STAIR_COUNT) and args.cycle_second_stair_count is not None:
        params[CYCLE_SECOND_STAIR_COUNT] = args.cycle_second_stair_count

    if hasattr(args, DECAY_STEP_SIZE) and args.decay_step_size is not None:
        params[DECAY_STEP_SIZE] = args.decay_step_size

    # 1Cycle LR params
    if hasattr(args, CYCLE_MIN_LR) and args.cycle_min_lr is not None:
        params[CYCLE_MIN_LR] = args.cycle_min_lr

    if hasattr(args, CYCLE_MAX_LR) and args.cycle_max_lr is not None:
        params[CYCLE_MAX_LR] = args.cycle_max_lr

    if hasattr(args, DECAY_LR_RATE) and args.decay_lr_rate is not None:
        params[DECAY_LR_RATE] = args.decay_lr_rate

    # 1Cycle MOM params
    if hasattr(args, CYCLE_MIN_MOM) and args.cycle_min_mom is not None:
        params[CYCLE_MIN_MOM] = args.cycle_min_mom

    if hasattr(args, CYCLE_MAX_MOM) and args.cycle_max_mom is not None:
        params[CYCLE_MAX_MOM] = args.cycle_max_mom

    if hasattr(args, DECAY_MOM_RATE) and args.decay_mom_rate is not None:
        params[DECAY_MOM_RATE] = args.decay_mom_rate


def override_warmupLR_params(args, params):
    if hasattr(args, WARMUP_MIN_LR) and args.warmup_min_lr is not None:
        params[WARMUP_MIN_LR] = args.warmup_min_lr

    if hasattr(args, WARMUP_MAX_LR) and args.warmup_max_lr is not None:
        params[WARMUP_MAX_LR] = args.warmup_max_lr

    if hasattr(args, WARMUP_NUM_STEPS) and args.warmup_num_steps is not None:
        params[WARMUP_NUM_STEPS] = args.warmup_num_steps

    if hasattr(args, WARMUP_TYPE) and args.warmup_type is not None:
        params[WARMUP_TYPE] = args.warmup_type


def override_params(args, params):
    # LR range test params
    override_lr_range_test_params(args, params)

    # 1Cycle params
    override_1cycle_params(args, params)

    # WarmupLR params
    override_warmupLR_params(args, params)


def get_config_from_args(args):
    if not hasattr(args, LR_SCHEDULE) or args.lr_schedule is None:
        return None, '--{} not specified on command line'.format(LR_SCHEDULE)

    if args.lr_schedule not in VALID_LR_SCHEDULES:
        return None, '{} is not supported LR schedule'.format(args.lr_schedule)

    config = {}
    config['type'] = args.lr_schedule
    config['params'] = {}

    if args.lr_schedule == LR_RANGE_TEST:
        override_lr_range_test_params(args, config['params'])
    elif args.lr_schedule == ONE_CYCLE:
        override_1cycle_params(args, config['params'])
    else:
        override_warmupLR_params(args, config['params'])

    return config, None


def get_lr_from_config(config):
    if 'type' not in config:
        return None, 'LR schedule type not defined in config'

    if 'params' not in config:
        return None, 'LR schedule params not defined in config'

    lr_schedule = config['type']
    lr_params = config['params']

    if lr_schedule not in VALID_LR_SCHEDULES:
        return None, '{} is not a valid LR schedule'.format(lr_schedule)

    if lr_schedule == LR_RANGE_TEST:
        return lr_params[LR_RANGE_TEST_MIN_LR], ''
    if lr_schedule == ONE_CYCLE:
        return lr_params[CYCLE_MAX_LR], ''
    # Warmup LR
    return lr_params[WARMUP_MAX_LR], ''


def update_lr(param_groups, lrs):
    for param_group, lr in zip(param_groups, lrs):
        # new LR should match the type of current LR for scalar and Tensor LR support
        if is_tensor(param_group['lr']):
            lr = tensor([lr], device=param_group['lr'].device)
        param_group['lr'] = lr
    return [group['lr'] for group in param_groups]


"""
Only optimizers that are subclass of torch.optim.Optimizer are supported. So check the passed optimizer and wrapped
optimizer to see if requirement is satisfied.
TODO: Looking under the hood to examine the wrapped optimizer is a hack that requires a better long-term fix.
"""


def get_torch_optimizer(optimizer):
    if isinstance(optimizer, Optimizer):
        return optimizer

    if hasattr(optimizer, 'optimizer') and isinstance(optimizer.optimizer, Optimizer):
        return optimizer.optimizer

    raise TypeError('{} is not a subclass of torch.optim.Optimizer'.format(type(optimizer).__name__))


class LRRangeTest(object):
    """Sets the learning rate of each parameter group according to
    learning rate range test (LRRT) policy. The policy increases learning
    rate starting from a base value with a constant frequency, as detailed in
    the paper `A disciplined approach to neural network hyper-parameters: Part 1 <https://arxiv.org/abs/1803.09820>`_

    LRRT policy is used for finding maximum LR that trains a model without divergence, and can be used to
    configure the LR boundaries for Cyclic LR schedules.

    LRRT changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_range_test_min_lr (float or list): Initial learning rate which is the
            lower boundary in the range test for each parameter group.
        lr_range_test_step_size (int): Interval of training steps to increase learning rate. Default: 2000
        lr_range_test_step_rate (float): Scaling rate for range test. Default: 1.0
        lr_range_test_staircase (bool): Scale in staircase fashion, rather than continuous. Default: False.
        last_batch_iteration (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_batch_iteration=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = LRRangeTest(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()

        _A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay:
        https://arxiv.org/abs/1803.09820
"""

    def __init__(self,
                 optimizer: Optimizer,
                 lr_range_test_min_lr: float = 1e-3,
                 lr_range_test_step_size: int = 2000,
                 lr_range_test_step_rate: float = 1.0,
                 lr_range_test_staircase: bool = False,
                 last_batch_iteration: int = -1):

        self.optimizer = get_torch_optimizer(optimizer)

        if isinstance(lr_range_test_min_lr, list) or isinstance(lr_range_test_min_lr, tuple):
            if len(lr_range_test_min_lr) != len(self.optimizer.param_groups):
                raise ValueError("expected {} lr_range_test_min_lr, got {}".format(len(self.optimizer.param_groups),
                                                                                   len(lr_range_test_min_lr)))
            self.min_lr = list(lr_range_test_min_lr)
        else:
            self.min_lr = [lr_range_test_min_lr] * len(self.optimizer.param_groups)

        self.step_size = lr_range_test_step_size
        self.step_rate = lr_range_test_step_rate
        self.last_batch_iteration = last_batch_iteration
        self.staircase = lr_range_test_staircase
        self.interval_fn = self._staircase_interval if lr_range_test_staircase else self._continuous_interval

        if last_batch_iteration == -1:
            self._last_lr = update_lr(self.optimizer.param_groups, self.min_lr)

    def _staircase_interval(self):
        return math.floor(float(self.last_batch_iteration + 1) / self.step_size)

    def _continuous_interval(self):
        return float(self.last_batch_iteration + 1) / self.step_size

    def _get_increase(self):
        return (1 + self.step_rate * self.interval_fn())

    def get_lr(self):
        lr_increase = self._get_increase()
        return [lr_range_test_min_lr * lr_increase for lr_range_test_min_lr in self.min_lr]

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        assert getattr(self, '_last_lr', None) is not None, "need to call step() first"
        return self._last_lr

    def step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        self._last_lr = update_lr(self.optimizer.param_groups, self.get_lr())

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']


class OneCycle(object):
    """Sets the learning rate of each parameter group according to
    1Cycle learning rate policy (1CLR). 1CLR is a variation of the
    Cyclical Learning Rate (CLR) policy that involves one cycle followed by
    decay. The policy simultaneously cycles the learning rate (and momentum)
    between two boundaries with a constant frequency, as detailed in
    the paper `A disciplined approach to neural network hyper-parameters`_.

    1CLR policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This implementation was adapted from the github repo: `PyTorch <https://github.com/pytorch/pytorch>`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        cycle_min_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        cycle_max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (cycle_max_lr - cycle_min_lr).
            The lr at any cycle is the sum of cycle_min_lr
            and some scaling of the amplitude; therefore
            cycle_max_lr may not actually be reached depending on
            scaling function.
        decay_lr_rate(float): Decay rate for learning rate. Default: 0.
        cycle_first_step_size (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        cycle_second_step_size (int): Number of training iterations in the
            decreasing half of a cycle. If cycle_second_step_size is None,
            it is set to cycle_first_step_size. Default: None
        cycle_first_stair_count(int): Number of stairs in first half of cycle phase. This means
        lr/mom are changed in staircase fashion. Default 0, means staircase disabled.
        cycle_second_stair_count(int): Number of stairs in second half of cycle phase. This means
        lr/mom are changed in staircase fashion. Default 0, means staircase disabled.
        decay_step_size (int): Intervals for applying decay in decay phase. Default: 0, means no decay.
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'cycle_min_mom' and 'cycle_max_mom'.
            Default: True
        cycle_min_mom (float or list): Initial momentum which is the
            lower boundary in the cycle for each parameter group.
            Default: 0.8
        cycle_max_mom (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (cycle_max_mom - cycle_min_mom).
            The momentum at any cycle is the difference of cycle_max_mom
            and some scaling of the amplitude; therefore
            cycle_min_mom may not actually be reached depending on
            scaling function. Default: 0.9
        decay_mom_rate (float): Decay rate for momentum. Default: 0.
        last_batch_iteration (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_batch_iteration=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = OneCycle(optimizer, 0.0001, 0.0010)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()


    .. _A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay: https://arxiv.org/abs/1803.09820
    """

    def __init__(self,
                 optimizer,
                 cycle_min_lr,
                 cycle_max_lr,
                 decay_lr_rate=0.,
                 cycle_first_step_size=2000,
                 cycle_second_step_size=None,
                 cycle_first_stair_count=0,
                 cycle_second_stair_count=None,
                 decay_step_size=0,
                 cycle_momentum=True,
                 cycle_min_mom=0.8,
                 cycle_max_mom=0.9,
                 decay_mom_rate=0.,
                 last_batch_iteration=-1):

        self.optimizer = get_torch_optimizer(optimizer)

        # Initialize cycle shape
        self._initialize_cycle(cycle_first_step_size, cycle_second_step_size, cycle_first_stair_count,
                               cycle_second_stair_count, decay_step_size)

        # Initialize cycle lr
        self._initialize_lr(self.optimizer, cycle_min_lr, cycle_max_lr, decay_lr_rate, last_batch_iteration)

        # Initialize cyclic momentum
        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            self._initialize_momentum(self.optimizer, cycle_min_mom, cycle_max_mom, decay_mom_rate,
                                      last_batch_iteration)
        # Initialize batch iteration tracker
        self.last_batch_iteration = last_batch_iteration

    # Configure cycle shape

    def _initialize_cycle(self, cycle_first_step_size, cycle_second_step_size, cycle_first_stair_count,
                          cycle_second_stair_count, decay_step_size):
        cycle_first_step_size = float(cycle_first_step_size)
        cycle_second_step_size = float(
            cycle_second_step_size) if cycle_second_step_size is not None else cycle_first_step_size

        self.total_size = cycle_first_step_size + cycle_second_step_size
        self.step_ratio = cycle_first_step_size / self.total_size
        self.first_stair_count = cycle_first_stair_count
        self.second_stair_count = cycle_first_stair_count if cycle_second_stair_count is None else cycle_second_stair_count
        self.decay_step_size = decay_step_size

        if math.isclose(self.decay_step_size, 0):
            self.skip_lr_decay = True
            self.skip_mom_decay = True
        else:
            self.skip_lr_decay = False
            self.skip_mom_decay = False

    # Configure lr schedule
    def _initialize_lr(self, optimizer, cycle_min_lr, cycle_max_lr, decay_lr_rate, last_batch_iteration):
        self.min_lrs = [cycle_min_lr] * len(optimizer.param_groups)
        if last_batch_iteration == -1:
            for lr, group in zip(self.min_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = [cycle_max_lr] * len(optimizer.param_groups)
        self.decay_lr_rate = decay_lr_rate

        if math.isclose(self.decay_lr_rate, 0):
            self.skip_lr_decay = True

    # Configure momentum schedule
    def _initialize_momentum(self, optimizer, cycle_min_mom, cycle_max_mom, decay_mom_rate, last_batch_iteration):
        if 'betas' not in optimizer.defaults:
            optimizer_name = type(optimizer).__name__
            logger.warning(
                f"cycle_momentum is disabled because optimizer {optimizer_name} does not support momentum, no betas attribute in defaults"
            )
            self.cycle_momentum = False
            return

        self.decay_mom_rate = decay_mom_rate
        self.min_moms = [(cycle_min_mom, 0.99)] * len(optimizer.param_groups)
        self.max_moms = [(cycle_max_mom, 0.99)] * len(optimizer.param_groups)

        if last_batch_iteration == -1:
            for momentum, group in zip(self.min_moms, optimizer.param_groups):
                group['betas'] = momentum

        if math.isclose(self.decay_mom_rate, 0):
            self.skip_mom_decay = True

    def _get_scale_factor(self):
        batch_iteration = (self.last_batch_iteration + 1)
        cycle = math.floor(1 + batch_iteration / self.total_size)
        x = 1. + batch_iteration / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        return scale_factor

    def _get_cycle_mom(self):
        scale_factor = self._get_scale_factor()
        momentums = []
        for base_betas, max_betas in zip(self.min_moms, self.max_moms):
            cycle_min_mom = base_betas[0]
            cycle_max_mom = max_betas[0]
            base_height = (cycle_max_mom - cycle_min_mom) * scale_factor
            momentum = cycle_max_mom - base_height
            momentums.append((momentum, base_betas[1]))
        return momentums

    def _get_cycle_lr(self):
        scale_factor = self._get_scale_factor()
        lrs = []
        for cycle_min_lr, cycle_max_lr in zip(self.min_lrs, self.max_lrs):
            base_height = (cycle_max_lr - cycle_min_lr) * scale_factor
            lr = cycle_min_lr + base_height
            lrs.append(lr)

        return lrs

    def _get_decay_mom(self, decay_batch_iteration):
        if self.skip_mom_decay:
            return self.max_moms

        decay_interval = decay_batch_iteration / self.decay_step_size
        mom_decay_factor = (1 + self.decay_mom_rate * decay_interval)
        momentums = [(beta0 * mom_decay_factor, beta1) for beta0, beta1 in self.max_moms]

        return momentums

    def _get_decay_lr(self, decay_batch_iteration):
        """Calculates the learning rate at batch index. This function is used
        after the cycle completes and post cycle decaying of lr/mom is enabled.
        This function treats `self.last_batch_iteration` as the last batch index.
        """
        if self.skip_lr_decay:
            return self.min_lrs

        decay_interval = decay_batch_iteration / self.decay_step_size
        lr_decay_factor = (1 + self.decay_lr_rate * decay_interval)
        lrs = [cycle_min_lr / lr_decay_factor for cycle_min_lr in self.min_lrs]

        return lrs

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_batch_iteration` as the last batch index.
        """
        if self.last_batch_iteration < self.total_size:
            return self._get_cycle_lr()
        return self._get_decay_lr(self.last_batch_iteration - self.total_size + 1)

    def get_mom(self):
        """Calculates the momentum at batch index. This function treats
        `self.last_batch_iteration` as the last batch index.
        """
        if not self.cycle_momentum:
            return None

        if self.last_batch_iteration < self.total_size:
            return self._get_cycle_mom()
        return self._get_decay_mom(self.last_batch_iteration - self.total_size + 1)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        assert getattr(self, '_last_lr', None) is not None, "need to call step() first"
        return self._last_lr

    def step(self, batch_iteration=None):
        """ Updates the optimizer with the learning rate for the last batch index.
        `self.last_batch_iteration` is treated as the last batch index.

        If self.cycle_momentum is true, also updates optimizer momentum.
        """
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1

        self.last_batch_iteration = batch_iteration
        self._last_lr = update_lr(self.optimizer.param_groups, self.get_lr())

        if self.cycle_momentum:
            momentums = self.get_mom()
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['betas'] = momentum

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']


class WarmupLR(object):
    """Increase the learning rate of each parameter group from min lr to max lr
        over warmup_num_steps steps, and then fix at max lr.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_min_lr (float or list): minimum learning rate. Default: 0
            warmup_max_lr (float or list): maximum learning rate. Default: 0.001
            warmup_num_steps (int): number of steps to warm up from min_lr to max_lr. Default: 1000
            warmup_type {‘log’, ‘linear’}: increasing function from min_lr to max_lr during warmup. Default: log
            last_batch_iteration (int): The index of the last batch. Default: -1.
        Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> scheduler = WarmupLR(optimizer)
            >>> data_loader = torch.utils.data.DataLoader(...)
            >>> for epoch in range(10):
            >>>     for batch in data_loader:
            >>>         train_batch(...)
            >>>         scheduler.step()

    """

    def __init__(self,
                 optimizer: Optimizer,
                 warmup_min_lr: float = 0.0,
                 warmup_max_lr: float = None,
                 warmup_num_steps: int = 1000,
                 warmup_type: str = WARMUP_LOG_RATE,
                 last_batch_iteration: int = -1):

        self.optimizer = get_torch_optimizer(optimizer)

        if warmup_max_lr is None:
            warmup_max_lr = [group['lr'] for group in self.optimizer.param_groups][0]

        self.min_lrs = self._format_param(self.optimizer, warmup_min_lr, "min_lr")
        self.max_lrs = self._format_param(self.optimizer, warmup_max_lr, "max_lr")
        self.delta_lrs = [big - small for big, small in zip(self.max_lrs, self.min_lrs)]
        self.warmup_num_steps = max(2, warmup_num_steps)
        # Currently only support linear and log function
        if warmup_type not in {WARMUP_LOG_RATE, WARMUP_LINEAR_RATE}:
            logger.warning(f"Using unknown warmup_type: {warmup_type}. The increasing function "
                           f"is set to default (log)")
            warmup_type = WARMUP_LOG_RATE
        self.warmup_type = warmup_type
        self.inverse_log_warm_up = 1.0 / math.log(self.warmup_num_steps)
        self.last_batch_iteration = last_batch_iteration
        # Initialize lr in optimizer
        if last_batch_iteration == -1:
            self._last_lr = update_lr(self.optimizer.param_groups, self.get_lr())

    def get_lr(self):
        if self.last_batch_iteration < 0:
            logger.warning("Attempting to get learning rate from scheduler before it has started")
            return self.min_lrs
        gamma = self._get_gamma()
        return [min_lr + (delta_lr * gamma) for min_lr, delta_lr in zip(self.min_lrs, self.delta_lrs)]

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        assert getattr(self, '_last_lr', None) is not None, "need to call step() first"
        return self._last_lr

    def step(self, last_batch_iteration=None):
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        self._last_lr = update_lr(self.optimizer.param_groups, self.get_lr())

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']

    def _get_gamma(self):
        if self.last_batch_iteration < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                return self.inverse_log_warm_up * math.log(self.last_batch_iteration + 1)
            elif self.warmup_type == WARMUP_LINEAR_RATE:
                return self.last_batch_iteration / self.warmup_num_steps
        return 1.0

    def _format_param(self, optimizer, param_value, param_name):
        if isinstance(param_value, list) or isinstance(param_value, tuple):
            if len(param_value) != len(optimizer.param_groups):
                raise ValueError("expected {} value for {}, got {}".format(len(optimizer.param_groups), param_name,
                                                                           FileNotFoundError(param_value)))
            return list(param_value)
        return [param_value] * len(optimizer.param_groups)


class WarmupDecayLR(WarmupLR):
    """Increase the learning rate of each parameter group from min lr to max lr
        over warmup_num_steps steps, and then decay at linear rate over the remaining training steps.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_num_steps (int): total number of training steps
            warmup_min_lr (float or list): minimum learning rate. Default: 0
            warmup_max_lr (float or list): maximum learning rate. Default: 0.001
            warmup_num_steps (int): number of steps to warm up from min_lr to max_lr. Default: 1000
            warmup_type {‘log’, ‘linear’}: increasing function from min_lr to max_lr during warmup. Default: log
            last_batch_iteration (int): The index of the last batch. Default: -1.
        Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> scheduler = WarmupDecayLR(optimizer, 1000000)
            >>> data_loader = torch.utils.data.DataLoader(...)
            >>> for epoch in range(10):
            >>>     for batch in data_loader:
            >>>         train_batch(...)
            >>>         scheduler.step()

    """

    def __init__(self,
                 optimizer: Optimizer,
                 total_num_steps: int,
                 warmup_min_lr: float = 0.0,
                 warmup_max_lr: float = 0.001,
                 warmup_num_steps: int = 1000,
                 warmup_type: str = WARMUP_LOG_RATE,
                 last_batch_iteration: int = -1):

        self.total_num_steps = total_num_steps
        super(WarmupDecayLR, self).__init__(optimizer, warmup_min_lr, warmup_max_lr, warmup_num_steps, warmup_type,
                                            last_batch_iteration)
        if self.total_num_steps < self.warmup_num_steps:
            logger.warning('total_num_steps {} is less than warmup_num_steps {}'.format(
                total_num_steps, warmup_num_steps))

    def _get_gamma(self):
        if self.last_batch_iteration < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                return self.inverse_log_warm_up * math.log(self.last_batch_iteration + 1)
            elif self.warmup_type == WARMUP_LINEAR_RATE:
                return self.last_batch_iteration / self.warmup_num_steps
        return max(
            0.0,
            float(self.total_num_steps - self.last_batch_iteration) /
            float(max(1.0, self.total_num_steps - self.warmup_num_steps)))


class WarmupCosineLR(object):
    """Increase the learning rate of each parameter group from min lr ratio to max lr ratio
        over warmup_num_steps steps, and then decay at cosine rate over the remaining training steps to min cosine ratio.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_num_steps (int): total number of training steps
            warmup_min_ratio (float or list): warmup start learning rate ratio. Default: 0
            warmup_num_steps (int): number of steps to warm up from warmup_min_ratio to 1.0. Default: 1000
            warmup_type {‘log’, ‘linear’}: increasing function from min_lr to max_lr during warmup. Default: log
            cos_min_ratio (float): cosine end learning rate ratio. Default: 0.0001
            last_batch_iteration (int): The index of the last batch. Default: -1.
        Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> scheduler = WarmupCosineLR(optimizer, 1000000)
            >>> data_loader = torch.utils.data.DataLoader(...)
            >>> for epoch in range(10):
            >>>     for batch in data_loader:
            >>>         train_batch(...)
            >>>         scheduler.step()

    """

    def __init__(self,
                 optimizer: Optimizer,
                 total_num_steps: int,
                 warmup_min_ratio: float = 0.0,
                 warmup_num_steps: int = 1000,
                 cos_min_ratio: float = 0.0001,
                 warmup_type: str = WARMUP_LOG_RATE,
                 last_batch_iteration: int = -1):

        self.optimizer = get_torch_optimizer(optimizer)

        self.total_num_steps = total_num_steps
        self.last_batch_iteration = last_batch_iteration
        self.cos_min_ratio = cos_min_ratio

        self.warmup_type = warmup_type
        self.warmup_min_ratio = warmup_min_ratio
        self.warmup_num_steps = max(2, warmup_num_steps)
        self.inverse_log_warm_up = 1.0 / math.log(self.warmup_num_steps)

        if self.total_num_steps < self.warmup_num_steps:
            logger.warning('total_num_steps {} is less than warmup_num_steps {}'.format(
                total_num_steps, warmup_num_steps))
        self.org_lrs = [group['lr'] for group in self.optimizer.param_groups]

        # Initialize lrs in optimizer groups
        if last_batch_iteration == -1:
            self._last_lr = update_lr(self.optimizer.param_groups, self.get_lr())

    def get_lr_ratio(self):
        if self.last_batch_iteration < 0:
            logger.warning("Attempting to get learning rate from scheduler before it has started")
            return [0.0]

        if self.last_batch_iteration < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                ratio = self.inverse_log_warm_up * math.log(self.last_batch_iteration + 1)
            elif self.warmup_type == WARMUP_LINEAR_RATE:
                ratio = self.last_batch_iteration / self.warmup_num_steps
            ratio_delta = 1. - self.warmup_min_ratio
            ratio = self.warmup_min_ratio + ratio * ratio_delta
            return ratio

        real_last_step = self.last_batch_iteration - self.warmup_num_steps + 1
        real_total_steps = self.total_num_steps - self.warmup_num_steps
        ratio_delta = 1. - self.cos_min_ratio
        ratio = (1 + math.cos(math.pi * real_last_step / real_total_steps)) / 2
        ratio = max(0.0, self.cos_min_ratio + ratio_delta * ratio)
        return ratio

    def step(self, last_batch_iteration=None):
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        self._last_lr = update_lr(self.optimizer.param_groups, self.get_lr())

    def get_lr(self):
        if self.last_batch_iteration < 0:
            logger.warning("Attempting to get learning rate from scheduler before it has started")
            return [0.0]
        lr_ratio = self.get_lr_ratio()
        return [org_lr * lr_ratio for org_lr in self.org_lrs]

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        assert getattr(self, '_last_lr', None) is not None, "need to call step() first"
        return self._last_lr

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']

    def _format_param(self, optimizer, param_value, param_name):
        if isinstance(param_value, list) or isinstance(param_value, tuple):
            if len(param_value) != len(optimizer.param_groups):
                raise ValueError("expected {} value for {}, got {}".format(len(optimizer.param_groups), param_name,
                                                                           FileNotFoundError(param_value)))
            return list(param_value)
        return [param_value] * len(optimizer.param_groups)


# =========================================================================

# =========================================================================
# DES-LOC LR Schedule Extensions (Section 4.1 + Section A.1)
# =========================================================================

class DeslocWSDSchedule:
    """Warmup-Stable-Decay schedule aligned to DES-LOC Kx boundaries.
    Ref: Section 4.1 - 'Learning rates use the WSD schedule.'
    Ensures LR transitions happen at sync boundaries for consistency."""

    def __init__(self, warmup_steps=512, stable_steps=None, decay_steps=None,
                 peak_lr=6e-4, min_lr=6e-5, Kx=32):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps or warmup_steps * 10
        self.decay_steps = decay_steps or warmup_steps * 2
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.Kx = max(Kx, 1)

    def get_lr(self, step):
        """Compute LR at given step."""
        if step < self.warmup_steps:
            return self.peak_lr * step / max(self.warmup_steps, 1)
        elif step < self.warmup_steps + self.stable_steps:
            return self.peak_lr
        else:
            decay_progress = (step - self.warmup_steps - self.stable_steps)
            decay_progress = min(decay_progress / max(self.decay_steps, 1), 1.0)
            import math
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def align_to_kx(self, step):
        """Round step to nearest Kx boundary."""
        if self.Kx <= 1:
            return step
        return (step // self.Kx) * self.Kx


def desloc_lr_warmup_aligned(base_warmup, Kx, min_warmup=64):
    """Align warmup period to Kx boundary.
    Ref: Section A.1 - TWARM=512 steps.
    Ensures warmup ends at a sync boundary."""
    if Kx <= 1:
        return max(base_warmup, min_warmup)
    aligned = ((base_warmup + Kx - 1) // Kx) * Kx
    return max(aligned, min_warmup)


# =========================================================================
# M216: DES-LOC Learning Rate Schedule Integration (Section 4.1 + A.1)
# =========================================================================

import math
from collections import OrderedDict


class DeslocAdaptiveWSD:
    """Adaptive WSD schedule that adjusts based on DES-LOC sync patterns.

    Core idea: On sync steps (t % Kx == 0), the effective learning rate
    may need adjustment because the gradient is an average across all workers.
    On non-sync steps, each worker uses only its local gradient.

    Ref: Theorem 1 step-size restriction:
      eta_0 = 1/(4L) * min(1-beta, 1/sqrt(psi * max(1, B^2-1)))
    The psi factor depends on Kx, Ku, so LR must respect this bound.
    """

    def __init__(self, base_lr=6e-4, warmup_steps=512, total_steps=100000,
                 min_lr=6e-5, Kx=32, beta1=0.9, beta2=0.999):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.Kx = max(Kx, 1)
        self.beta1 = beta1
        self.beta2 = beta2
        self._step = 0
        self._history = []

    def _psi(self):
        """Compute psi factor for step-size bound."""
        px = 1.0 / self.Kx
        pu = 1.0 / (3 * self.Kx)
        beta = self.beta1
        numer = 4.0 * (1.0 - px) * (1.0 - beta) * (1.0 - pu)
        denom = px * px * 6.0 * (1.0 - (1.0 - pu) * beta)
        if abs(denom) < 1e-15:
            return float('inf')
        return numer / denom

    def max_allowed_lr(self, L=1.0, B_sq=1.0):
        """Compute maximum LR from Theorem 1 step-size restriction."""
        psi = self._psi()
        term1 = 1.0 - self.beta1
        inner = psi * max(1.0, B_sq - 1.0)
        term2 = 1.0 / math.sqrt(max(inner, 1e-15))
        return min(term1, term2) / (4.0 * max(L, 1e-15))

    def get_lr(self, step=None):
        """Compute LR at given step with WSD phases."""
        if step is None:
            step = self._step
        # Warmup phase
        if step < self.warmup_steps:
            return self.base_lr * step / max(self.warmup_steps, 1)
        # Stable phase (80% of remaining)
        stable_end = self.warmup_steps + int(0.8 * (self.total_steps - self.warmup_steps))
        if step < stable_end:
            return self.base_lr
        # Decay phase (cosine)
        decay_steps = self.total_steps - stable_end
        progress = (step - stable_end) / max(decay_steps, 1)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self):
        """Advance one step and return current LR."""
        lr = self.get_lr()
        self._history.append(lr)
        self._step += 1
        return lr

    def state_dict(self):
        return {'step': self._step, 'base_lr': self.base_lr, 'Kx': self.Kx}

    def load_state_dict(self, sd):
        self._step = sd.get('step', 0)


class DeslocLRAligner:
    """Align LR schedule transitions to Kx boundaries.

    Problem: If LR changes mid-way between sync boundaries, different
    workers may use different LRs for their local steps. On the next
    sync, the averaged parameters reflect a mix of LR values.

    Solution: Snap LR transitions to the nearest Kx boundary.
    Ref: Section A.1 warmup alignment.
    """

    def __init__(self, Kx=32):
        self.Kx = max(Kx, 1)

    def align_step(self, step):
        """Round step down to nearest Kx boundary."""
        if self.Kx <= 1:
            return step
        return (step // self.Kx) * self.Kx

    def align_warmup(self, warmup_steps, min_warmup=64):
        """Align warmup end to Kx boundary (round up)."""
        if self.Kx <= 1:
            return max(warmup_steps, min_warmup)
        aligned = ((warmup_steps + self.Kx - 1) // self.Kx) * self.Kx
        return max(aligned, min_warmup)

    def align_checkpoint_interval(self, interval):
        """Align checkpoint interval to LCM of Kx and interval."""
        if self.Kx <= 1:
            return interval
        return self._lcm(interval, self.Kx)

    @staticmethod
    def _lcm(a, b):
        return abs(a * b) // math.gcd(a, b)


class DeslocCyclicalKxSchedule:
    """Cyclical Kx schedule that varies sync frequency during training.

    Hypothesis: Early in training, more frequent sync (small Kx) helps
    convergence. Later, larger Kx saves communication without hurting quality.

    This implements a linear ramp from Kx_start to Kx_end over training.
    """

    def __init__(self, Kx_start=4, Kx_end=128, total_steps=100000,
                 warmup_steps=512):
        self.Kx_start = max(Kx_start, 1)
        self.Kx_end = max(Kx_end, Kx_start)
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def get_Kx(self, step):
        """Get Kx at given step."""
        if step < self.warmup_steps:
            return 1  # DDP during warmup
        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        Kx = self.Kx_start + progress * (self.Kx_end - self.Kx_start)
        # Round to nearest power of 2
        p = 1
        while p < Kx:
            p *= 2
        return min(p, self.Kx_end)


class DeslocLRScaler:
    """Scale LR based on effective batch size under DES-LOC.

    When Kx > 1, each worker processes Kx local steps before averaging.
    The effective batch size is approximately batch_size * Kx.
    Linear scaling rule: LR should scale with sqrt(effective_batch).

    Ref: Section 5.5 — outer LR tuned following Goyal et al.
    """

    def __init__(self, base_lr, base_batch_size, scaling='sqrt'):
        self.base_lr = base_lr
        self.base_batch = base_batch_size
        self.scaling = scaling

    def scaled_lr(self, actual_batch_size, Kx=1):
        """Compute scaled LR for given batch size and Kx."""
        effective = actual_batch_size * max(Kx, 1)
        ratio = effective / max(self.base_batch, 1)
        if self.scaling == 'linear':
            return self.base_lr * ratio
        elif self.scaling == 'sqrt':
            return self.base_lr * math.sqrt(ratio)
        else:
            return self.base_lr


class DeslocScheduleValidator:
    """Validate LR schedule meets DES-LOC convergence requirements.

    Checks:
    1. Peak LR respects Theorem 1 step-size bound
    2. Warmup is long enough (>= Kx steps)
    3. LR transitions align to Kx boundaries
    4. Decay is smooth (no discontinuities)
    """

    def __init__(self, schedule, Kx, beta1=0.9):
        self.schedule = schedule
        self.Kx = Kx
        self.beta1 = beta1

    def check_all(self, total_steps, L=1.0):
        """Run all validation checks."""
        issues = []
        # Check peak LR
        peak = max(self.schedule.get_lr(s) for s in range(total_steps))
        max_lr = DeslocAdaptiveWSD(Kx=self.Kx, beta1=self.beta1).max_allowed_lr(L)
        if peak > max_lr * 1.1:
            issues.append(f'Peak LR {peak:.6f} exceeds bound {max_lr:.6f}')
        # Check warmup length
        warmup = getattr(self.schedule, 'warmup_steps', 0)
        if warmup < self.Kx:
            issues.append(f'Warmup {warmup} < Kx {self.Kx}')
        return issues

    return max(min_warmup, aligned)


# =============================================================================
# M233 (Claude-15): WSD phase loss annotation for Figure 1
# Ref: Section 4.1 — WSD schedule (warmup/stable/decay)
# Ref: Section 5.4 — loss curve shows phase transitions
# Ref: NKI-FA draw_plot.py — annotated data points
# =============================================================================


class DeslocWSDPhaseAnnotator:
    """Annotate loss curve with WSD schedule phase boundaries for Figure 1.

    The WSD (Warmup-Stable-Decay) schedule has three distinct phases.
    Figure 1 loss curves should show vertical lines at phase transitions
    with annotations like 'warmup→stable at step 512'.

    From Section 4.1: WSD schedule with linear warmup, constant LR,
    then cosine/linear decay. Phase boundaries affect DES-LOC sync
    behavior: during warmup, Kx=1 (full DDP); after warmup, Kx>1.

    Usage:
        annotator = DeslocWSDPhaseAnnotator(
            warmup_steps=512, total_steps=5000, decay_start=4000)
        phase = annotator.get_phase(step=100)  # -> 'warmup'
        boundaries = annotator.get_boundaries()  # for Figure 1 vlines
    """

    PHASES = ('warmup', 'stable', 'decay')

    def __init__(self, warmup_steps=512, total_steps=5000,
                 decay_start=None, decay_style='cosine',
                 desloc_warmup=None):
        """Initialize annotator.

        Args:
            warmup_steps: LR warmup duration
            total_steps: total training steps
            decay_start: step where decay begins (default: total - total/5)
            decay_style: 'cosine' or 'linear'
            desloc_warmup: DES-LOC Kx=1 warmup (may differ from LR warmup)
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_start = decay_start or max(
            warmup_steps + 1, total_steps - total_steps // 5)
        self.decay_style = decay_style
        self.desloc_warmup = desloc_warmup or warmup_steps

    def get_phase(self, step):
        """Get WSD phase for a given step.

        Returns: str — 'warmup', 'stable', or 'decay'
        """
        if step < self.warmup_steps:
            return 'warmup'
        elif step < self.decay_start:
            return 'stable'
        else:
            return 'decay'

    def get_boundaries(self):
        """Get phase transition points for Figure 1 annotations.

        Returns: list of dicts:
            [{'step': 512, 'from': 'warmup', 'to': 'stable',
              'label': 'warmup→stable'},
             {'step': 4000, 'from': 'stable', 'to': 'decay',
              'label': 'stable→decay'}]
        """
        boundaries = []
        if self.warmup_steps > 0:
            boundaries.append({
                'step': self.warmup_steps,
                'from': 'warmup', 'to': 'stable',
                'label': f'warmup→stable (step {self.warmup_steps})',
                'desloc_note': f'DES-LOC activates Kx>1 at step {self.desloc_warmup}',
            })
        if self.decay_start < self.total_steps:
            boundaries.append({
                'step': self.decay_start,
                'from': 'stable', 'to': 'decay',
                'label': f'stable→decay (step {self.decay_start})',
                'desloc_note': 'Kx maintained during decay',
            })
        return boundaries

    def get_lr_at_step(self, step, base_lr=6e-4):
        """Compute WSD learning rate at a given step.

        Args:
            step: training step
            base_lr: peak learning rate

        Returns: float — learning rate value
        """
        import math
        if step < self.warmup_steps:
            return base_lr * (step + 1) / max(1, self.warmup_steps)
        elif step < self.decay_start:
            return base_lr
        else:
            progress = (step - self.decay_start) / max(
                1, self.total_steps - self.decay_start)
            if self.decay_style == 'cosine':
                return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                return base_lr * (1.0 - progress)

    def annotate_loss_data(self, steps, losses):
        """Add phase annotations to loss curve data.

        Returns: list of (step, loss, phase) triples.
        Useful for color-coding loss curve by WSD phase in Figure 1.
        """
        annotated = []
        for step, loss in zip(steps, losses):
            phase = self.get_phase(step)
            annotated.append((step, loss, phase))
        return annotated

    def get_figure1_vline_specs(self):
        """Return vertical line specs for matplotlib.

        Usage:
            for vline in annotator.get_figure1_vline_specs():
                ax.axvline(x=vline['x'], color=vline['color'],
                           linestyle=vline['ls'], label=vline['label'])
        """
        specs = []
        colors = {'warmup→stable': '#e74c3c', 'stable→decay': '#3498db'}
        for b in self.get_boundaries():
            key = f'{b["from"]}→{b["to"]}'
            specs.append({
                'x': b['step'],
                'color': colors.get(key, '#95a5a6'),
                'ls': '--',
                'label': b['label'],
                'alpha': 0.7,
            })
        return specs

    def emit_nkifa_log(self, stream=None):
        """Write phase info in NKI-FA format."""
        def _w(line):
            if stream:
                stream.write(line + '\n')
            else:
                print(line)
        _w('### wsd_phases ###')
        _w(f'warmup_steps: {self.warmup_steps}')
        _w(f'decay_start: {self.decay_start}')
        _w(f'total_steps: {self.total_steps}')
        _w(f'decay_style: {self.decay_style}')
        _w(f'desloc_warmup: {self.desloc_warmup}')
        for b in self.get_boundaries():
            _w(f'boundary: step={b["step"]} {b["label"]}')

    def get_phase_durations(self):
        """Get duration of each phase as fraction of total training."""
        warmup_frac = self.warmup_steps / max(1, self.total_steps)
        decay_frac = (self.total_steps - self.decay_start) / max(1, self.total_steps)
        stable_frac = 1.0 - warmup_frac - decay_frac
        return {
            'warmup': round(warmup_frac, 4),
            'stable': round(max(0, stable_frac), 4),
            'decay': round(decay_frac, 4),
        }
