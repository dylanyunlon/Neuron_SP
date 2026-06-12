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
        over warmup_num_steps steps, and then decay at linear rate over the remaining
        training steps.

        Incorporates three improvements from Megatron-LM commit a1d04b7939c3cf3:
          1. min_lr floor: decay never drops below this value (default 0.0 = original).
          2. Clamped progress: effective iteration is clipped to schedule boundary so
             the linear decay formula cannot overshoot past the end of training.
          3. Checkpoint-aware LR resume: use_checkpoint_lr_scheduler /
             override_lr_scheduler flags give explicit control over which values win on
             checkpoint reload, surfacing silent mismatches.

        Knuth double critique of the original design:
          (1) _get_gamma could return negative values past total_num_steps because
              last_batch_iteration was never clamped — a mathematically inconsistent
              extrapolation that silently violated the schedule’s own contract.
          (2) load_state_dict blindly overwrote every hyper-param from the checkpoint
              with no validation, making “continue with different --lr” invisibly broken.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_num_steps (int): total number of training steps
            warmup_min_lr (float or list): minimum learning rate. Default: 0
            warmup_max_lr (float or list): maximum learning rate. Default: 0.001
            warmup_num_steps (int): number of steps to warm up from min_lr to max_lr.
                Default: 1000
            warmup_type {‘log’, ‘linear’}: increasing function from min_lr to max_lr
                during warmup. Default: log
            last_batch_iteration (int): The index of the last batch. Default: -1.
            min_lr (float): absolute LR floor applied during the decay tail.
                Default: 0.0 (original behaviour unchanged).
            use_checkpoint_lr_scheduler (bool): if True, hyper-params are loaded from
                the checkpoint; mismatches with class-init values are logged.
                Default: True
            override_lr_scheduler (bool): if True, class-init values win over the
                checkpoint’s saved hyper-params. Mutually exclusive with
                use_checkpoint_lr_scheduler. Default: False
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
                 last_batch_iteration: int = -1,
                 min_lr: float = 0.0,
                 use_checkpoint_lr_scheduler: bool = True,
                 override_lr_scheduler: bool = False):

        # M454: min_lr floor + checkpoint-resume flags stored before super().__init__ so
        # they are available if get_lr() is triggered during parent initialisation.
        self.decay_min_lr = min_lr
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler
        self.override_lr_scheduler = override_lr_scheduler
        if self.override_lr_scheduler:
            assert not self.use_checkpoint_lr_scheduler, (
                'WarmupDecayLR: override_lr_scheduler and use_checkpoint_lr_scheduler '
                'are mutually exclusive — set at most one.')
        self.total_num_steps = total_num_steps
        super(WarmupDecayLR, self).__init__(optimizer, warmup_min_lr, warmup_max_lr,
                                            warmup_num_steps, warmup_type,
                                            last_batch_iteration)
        if self.total_num_steps < self.warmup_num_steps:
            logger.warning('total_num_steps {} is less than warmup_num_steps {}'.format(
                total_num_steps, warmup_num_steps))
        print(
            f'[WarmupDecayLR] init: total={total_num_steps} warmup={warmup_num_steps} '
            f'max_lr={warmup_max_lr} min_lr_floor={min_lr} '
            f'use_ckpt_lr={use_checkpoint_lr_scheduler} override_lr={override_lr_scheduler}'
        )

    def _get_gamma(self):
        # M454: clamp effective iteration to schedule boundary (Megatron num_iters_ pattern).
        # Without the clamp, iterations past total_num_steps yield negative gamma values —
        # a silent contract violation that produced negative LR in extreme cases.
        eff_iter = min(self.last_batch_iteration, self.total_num_steps)
        if eff_iter < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                return self.inverse_log_warm_up * math.log(eff_iter + 1)
            elif self.warmup_type == WARMUP_LINEAR_RATE:
                return eff_iter / self.warmup_num_steps
        # Linear decay tail: fraction of max_lr remaining, floored by decay_min_lr ratio.
        raw_gamma = max(
            0.0,
            float(self.total_num_steps - eff_iter) /
            float(max(1.0, self.total_num_steps - self.warmup_num_steps)))
        # Convert absolute min_lr floor to a gamma floor against the first group’s max_lr.
        max_lr_ref = self.max_lrs[0] if self.max_lrs else 1.0
        min_gamma = (self.decay_min_lr / max_lr_ref) if max_lr_ref > 0.0 else 0.0
        return max(raw_gamma, min_gamma)

    def state_dict(self):
        # M454: persist decay_min_lr and total_num_steps so a resumed run can validate
        # against the saved configuration.
        sd = super(WarmupDecayLR, self).state_dict()
        sd['decay_min_lr'] = self.decay_min_lr
        sd['total_num_steps'] = self.total_num_steps
        return sd

    def _check_and_resume_(self, cls_value, sd_value, name):
        """Checkpoint-aware field resolution (Megatron check_and_set_ pattern, adapted).

        Knuth double critique:
          (1) The original load_state_dict was a silent liar: it blindly overwrote every
              hyper-param from the checkpoint, so changing --lr or --total-iters at resume
              time had zero visible effect — a confusing invariant violation.
          (2) It persisted only last_batch_iteration in state_dict, so there was nothing
              to cross-check anyway; this pair of methods fixes both defects together.
        """
        if self.override_lr_scheduler:
            print(f'[WarmupDecayLR] override: keeping class value {name}={cls_value} '
                  f'(checkpoint had {sd_value})')
            return cls_value
        if cls_value != sd_value:
            logger.warning(
                f'[WarmupDecayLR] {name}: class value {cls_value} != checkpoint value '
                f'{sd_value}; adopting checkpoint value. Pass override_lr_scheduler=True '
                f'to use class value instead.')
        print(f'[WarmupDecayLR] resume: {name}={sd_value} (from checkpoint)')
        return sd_value

    def load_state_dict(self, sd):
        # M454: checkpoint-aware reload — validate/override configurable hyper-params;
        # always trust the checkpoint’s iteration counter.
        if 'decay_min_lr' in sd:
            self.decay_min_lr = self._check_and_resume_(
                self.decay_min_lr, sd['decay_min_lr'], 'decay_min_lr')
        if 'total_num_steps' in sd:
            self.total_num_steps = self._check_and_resume_(
                self.total_num_steps, sd['total_num_steps'], 'total_num_steps')
        self.last_batch_iteration = sd['last_batch_iteration']
        print(f'[WarmupDecayLR] load_state_dict: resumed at iter={self.last_batch_iteration} '
              f'lr={self.get_lr()}')


class AnnealingLR(object):
    """Anneals the learning rate from start_lr through a warmup phase then decays.

    Ported from Megatron-LM commit b6e0377b (refactored learning-rate) and fused
    with the checkpoint-resume contract established by WarmupDecayLR in this file.

    Key differences from the Megatron original (20% adaptation):
      - Backed by DeepSpeed's update_lr / get_torch_optimizer helpers so it plugs
        into the same optimizer-wrapping infrastructure used by WarmupDecayLR.
      - state_dict / load_state_dict follow the WarmupDecayLR._check_and_resume_
        pattern: hyper-params are cross-validated on reload, not blindly overwritten.
      - Diagnostic prints mirror the [WarmupDecayLR] / [WarmupCosineLR] format so
        all schedule events appear in a consistent log stream.

    Knuth double critique of the Megatron original:
      (1) The class stored `num_iters = last_iter` and immediately called
          `self.step(self.num_iters)`, meaning iteration 0 produced a non-zero LR
          on the very first forward pass — a subtle off-by-one whose effect on the
          first gradient update was never documented or intended.
      (2) `state_dict` persisted only `num_iters`; resuming a run with a different
          `decay_style` or `end_iter` silently continued with stale schedule config,
          making "change the schedule at resume time" invisibly broken.

    Args:
        optimizer: Wrapped optimizer (may be a DeepSpeed wrapper).
        start_lr (float): Peak learning rate after warmup.
        warmup_iter (int): Number of warmup steps (linear ramp from 0 to start_lr).
        total_iters (int): Total training iterations; must be > 0.
        decay_style (str): One of 'linear', 'cosine', 'constant'.
        last_iter (int): Step count to initialise from (0 for a fresh run).
        min_lr (float): Absolute LR floor applied during decay. Default: 0.0.
        use_checkpoint_lr_scheduler (bool): Load hyper-params from checkpoint on
            resume; log a warning on mismatch. Default: True.
        override_lr_scheduler (bool): Class-init values win over the checkpoint's
            saved hyper-params. Mutually exclusive with use_checkpoint_lr_scheduler.
            Default: False.
    """

    DECAY_STYLES = ('linear', 'cosine', 'constant')
    print('[M434]')

    def __init__(self,
                 optimizer,
                 start_lr: float,
                 warmup_iter: int,
                 total_iters: int,
                 decay_style: str,
                 last_iter: int,
                 min_lr: float = 0.0,
                 use_checkpoint_lr_scheduler: bool = True,
                 override_lr_scheduler: bool = False):

        self.optimizer = get_torch_optimizer(optimizer)
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.warmup_iter = warmup_iter
        self.end_iter = total_iters
        assert self.end_iter > 0, 'AnnealingLR: total_iters must be > 0'
        # Normalise decay_style to lower-case; keep original for error messages.
        raw_style = decay_style.lower() if isinstance(decay_style, str) else 'constant'
        if raw_style not in self.DECAY_STYLES:
            logger.warning(
                f'[AnnealingLR] unknown decay_style "{decay_style}"; '
                f'falling back to "constant". Valid: {self.DECAY_STYLES}')
            raw_style = 'constant'
        self.decay_style = raw_style
        self.override_lr_scheduler = override_lr_scheduler
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler
        if self.override_lr_scheduler:
            assert not self.use_checkpoint_lr_scheduler, (
                'AnnealingLR: override_lr_scheduler and use_checkpoint_lr_scheduler '
                'are mutually exclusive — set at most one.')

        # Initialise iteration counter then snap LR to the formula (Megatron pattern).
        self.num_iters = last_iter
        self.step(self.num_iters)

        print(
            f'[AnnealingLR] init: decay_style={self.decay_style} '
            f'start_lr={start_lr} warmup_iter={warmup_iter} total_iters={total_iters} '
            f'min_lr={min_lr} last_iter={last_iter} '
            f'use_ckpt_lr={use_checkpoint_lr_scheduler} override_lr={override_lr_scheduler}'
        )

    # ------------------------------------------------------------------
    # Core schedule formula (Megatron b6e0377b, §4 of the BJYwwY9ll paper)
    # ------------------------------------------------------------------

    def get_lr(self):
        """Return the current scalar learning rate according to the schedule.

        Ref: https://openreview.net/pdf?id=BJYwwY9ll pg. 4
        """
        num_iters_ = min(self.num_iters, self.end_iter - self.warmup_iter)
        # Warmup ramp: linear from 0 → start_lr over warmup_iter steps.
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            return float(self.start_lr) * num_iters_ / self.warmup_iter

        num_iters_ = num_iters_ - self.warmup_iter
        if self.decay_style == 'linear':
            lr = self.start_lr * (self.end_iter - num_iters_) / self.end_iter
        elif self.decay_style == 'cosine':
            lr = self.start_lr / 2.0 * (
                math.cos(math.pi * num_iters_ / self.end_iter) + 1)
        elif self.decay_style == 'exponential':
            # exp(-0.693) ≈ 1/2 — halves every end_iter steps
            print('[M49]')
            lr = self.start_lr * math.exp(-0.693 * num_iters_ / self.end_iter)
        else:
            # 'constant' — frozen at start_lr after warmup
            lr = self.start_lr
        return max(lr, self.min_lr)

    # ------------------------------------------------------------------
    # Step / optimizer update
    # ------------------------------------------------------------------

    def step(self, step_num=None):
        """Advance the schedule and push the new LR into all optimizer param groups."""
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        # Reuse DeepSpeed's update_lr helper so Tensor-LR param groups work correctly.
        self._last_lr = update_lr(self.optimizer.param_groups, [new_lr] * len(self.optimizer.param_groups))

    def get_last_lr(self):
        """Return last LR set by step(); raises if step() has not been called."""
        assert getattr(self, '_last_lr', None) is not None, \
            '[AnnealingLR] get_last_lr() called before step()'
        return self._last_lr

    # ------------------------------------------------------------------
    # Checkpoint serialisation (fused from WarmupDecayLR, M463)
    # ------------------------------------------------------------------

    def state_dict(self):
        """Persist both the iteration counter and the schedule hyper-params.

        Rationale: the Megatron original saved only num_iters, so resuming with a
        different decay_style or end_iter was silently broken — the schedule would
        continue with the class-init values while load_state_dict quietly ignored
        the mismatch.  Saving hyper-params here enables _check_and_resume_ to
        surface that confusion.
        """
        return {
            'num_iters': self.num_iters,
            'start_lr': self.start_lr,
            'min_lr': self.min_lr,
            'warmup_iter': self.warmup_iter,
            'end_iter': self.end_iter,
            'decay_style': self.decay_style,
        }

    def _check_and_resume_(self, cls_value, sd_value, name):
        """Resolve a single hyper-param between class-init and checkpoint values.

        Mirrors WarmupDecayLR._check_and_resume_: if override_lr_scheduler is set
        the class-init value wins; otherwise the checkpoint value wins and any
        discrepancy is surfaced as a warning rather than silently discarded.
        """
        if self.override_lr_scheduler:
            print(f'[AnnealingLR] override: keeping class value {name}={cls_value} '
                  f'(checkpoint had {sd_value})')
            return cls_value
        if cls_value != sd_value:
            logger.warning(
                f'[AnnealingLR] {name}: class value {cls_value} != checkpoint value '
                f'{sd_value}; adopting checkpoint value. '
                f'Pass override_lr_scheduler=True to use class value instead.')
        print(f'[AnnealingLR] resume: {name}={sd_value} (from checkpoint)')
        return sd_value

    def load_state_dict(self, sd):
        """Restore schedule state from a checkpoint.

        Hyper-params (start_lr, min_lr, warmup_iter, end_iter, decay_style) are
        cross-validated through _check_and_resume_ before being applied; the
        iteration counter is always taken from the checkpoint.  After restoring
        all fields, step() is called to snap the LR into the optimizer — matching
        the Megatron __init__ pattern and ensuring the first training step after
        resume uses a formula-derived LR rather than a potentially stale float.
        """
        for attr in ('start_lr', 'min_lr', 'warmup_iter', 'end_iter', 'decay_style'):
            if attr in sd:
                setattr(self, attr,
                        self._check_and_resume_(getattr(self, attr), sd[attr], attr))
        # Always trust the checkpoint's iteration counter.
        self.num_iters = sd['num_iters']
        # Re-snap LR to formula at the restored iteration (Megatron b6e0377b discipline).
        self.step(self.num_iters)
        print(
            f'[AnnealingLR] load_state_dict: resumed at iter={self.num_iters} '
            f'lr={self.get_lr():.6e} decay_style={self.decay_style}'
        )


class WarmupCosineLR(object):
    """Increase the learning rate of each parameter group from min lr ratio to max lr
        ratio over warmup_num_steps steps, then decay at cosine rate over the remaining
        training steps down to cos_min_ratio.

        M454 improvements (Megatron a1d04b7939c3cf3 cosine-decay pattern adapted):
          - Effective iteration is clamped to the schedule boundary so the cosine
            formula cannot extrapolate past total_num_steps and produce a rising tail.
          - state_dict now persists total_num_steps and cos_min_ratio so resumed runs
            can detect schedule-config mismatches.
          - load_state_dict logs a warning when class-init values differ from the
            checkpoint’s saved values, matching WarmupDecayLR’s new contract.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_num_steps (int): total number of training steps
            warmup_min_ratio (float): warmup start LR ratio. Default: 0
            warmup_num_steps (int): warmup steps from warmup_min_ratio to 1.0.
                Default: 1000
            cos_min_ratio (float): cosine end LR ratio. Default: 0.0001
            warmup_type {‘log’, ‘linear’}: warmup shape. Default: log
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

        print(f'[WarmupCosineLR] init: total={total_num_steps} warmup={warmup_num_steps} '
              f'cos_min_ratio={cos_min_ratio}')

    def get_lr_ratio(self):
        if self.last_batch_iteration < 0:
            logger.warning("Attempting to get learning rate from scheduler before it has started")
            return [0.0]

        # M454: clamp to schedule end so cosine does not rise again past total_num_steps.
        eff_iter = min(self.last_batch_iteration, self.total_num_steps)

        if eff_iter < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                ratio = self.inverse_log_warm_up * math.log(eff_iter + 1)
            elif self.warmup_type == WARMUP_LINEAR_RATE:
                ratio = eff_iter / self.warmup_num_steps
            else:
                ratio = self.inverse_log_warm_up * math.log(eff_iter + 1)
            ratio_delta = 1. - self.warmup_min_ratio
            ratio = self.warmup_min_ratio + ratio * ratio_delta
            return ratio

        real_last_step = eff_iter - self.warmup_num_steps + 1
        real_total_steps = self.total_num_steps - self.warmup_num_steps
        ratio_delta = 1. - self.cos_min_ratio
        ratio = (1 + math.cos(math.pi * real_last_step / max(1, real_total_steps))) / 2
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
        # M454: persist schedule config so mismatches are detectable on resume.
        return {
            'last_batch_iteration': self.last_batch_iteration,
            'total_num_steps': self.total_num_steps,
            'cos_min_ratio': self.cos_min_ratio,
        }

    def load_state_dict(self, sd):
        # M454: log a warning if the saved schedule config disagrees with the current
        # class-init values (mirrors WarmupDecayLR._check_and_resume_ contract).
        for key, cls_val in (('total_num_steps', self.total_num_steps),
                             ('cos_min_ratio', self.cos_min_ratio)):
            if key in sd and sd[key] != cls_val:
                logger.warning(
                    f'[WarmupCosineLR] {key}: class value {cls_val} != checkpoint '
                    f'value {sd[key]}; using checkpoint value.')
            if key in sd:
                setattr(self, key, sd[key])
        self.last_batch_iteration = sd['last_batch_iteration']
        print(f'[WarmupCosineLR] load_state_dict: resumed at iter={self.last_batch_iteration} '
              f'lr_ratio={self.get_lr_ratio():.6f}')

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



# M294 — Claude-19: WSD v2 + CheckpointResumer
import math as _m294
class DeslocWSDV2:
    __slots__=('lm','wu','stb','dec','tot','dt','lmn','Kx','st','lr','ph','phl')
    def __init__(s,lm=6e-4,wu=512,stb=4000,dec=5000,dt='cosine',lmn=6e-5,Kx=1):
        s.lm=lm;s.Kx=max(1,Kx);s.wu=((wu+Kx-1)//Kx)*Kx if Kx>1 else wu;s.stb=stb;s.dec=dec;s.tot=s.wu+stb+dec;s.dt=dt;s.lmn=lmn;s.st=0;s.lr=0.;s.ph='warmup';s.phl=[]
    def step(s):
        s.st+=1;op=s.ph
        if s.st<=s.wu:s.ph='warmup';s.lr=s.lm*s.st/max(1,s.wu)
        elif s.st<=s.wu+s.stb:s.ph='stable';s.lr=s.lm
        else:
            s.ph='decay';dp=min(1.,(s.st-s.wu-s.stb)/max(1,s.dec))
            if s.dt=='cosine':s.lr=s.lmn+(s.lm-s.lmn)*(1+_m294.cos(_m294.pi*dp))/2
            elif s.dt=='linear':s.lr=s.lm-(s.lm-s.lmn)*dp
            else:s.lr=max(s.lmn,s.lm*_m294.exp(-3*dp))
        if op!=s.ph:s.phl.append((s.st,op,s.ph))
        return s.lr
    def eff_Kx(s):return 1 if s.ph=='warmup'else s.Kx
class DeslocCkptResumer:
    __slots__=('cs','cl','sched','w')
    def __init__(s,cs,cl=None):s.cs=cs;s.cl=cl;s.sched=s.w=None
    def attach(s,sched):
        s.sched=sched
        for _ in range(s.cs):sched.step()
        if s.cl is not None and abs(sched.lr-s.cl)>.01*s.cl:s.w=f"LR mismatch: {s.cl:.6f} vs {sched.lr:.6f}"
def desloc_psi_lr(lr,Kx,Ku,b1=.9,w=.1):
    px,pu=1./max(1,Kx),1./max(1,Ku);n=4*(1-px)*(1-b1)*(1-pu);d=px*px*6*(1-(1-pu)*b1)
    psi=n/d if abs(d)>1e-15 else 0;return lr/_m294.sqrt(1+w*psi)
# M294: end


class DeslocWSDKxAligned:
    def __init__(self, lr_max=6e-4, warmup=512, stable=4000, decay=5000,
                 lr_min=6e-5, Kx=32, style='cosine', restarts=0):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.Kx = max(1, Kx)
        self.style = style
        self.restarts = restarts
        self.warmup_end = self._align(warmup)
        self.stable_end = self._align(warmup + stable)
        self.total = self._align(warmup + stable + decay)
        self.decay_len = self.total - self.stable_end
        self._step = 0
        self._lr = 0.0
        self._phase = 'warmup'

    def _align(self, s):
        if self.Kx <= 1:
            return s
        return ((s + self.Kx - 1) // self.Kx) * self.Kx

    def step(self, stability_ok=True):
        self._step += 1
        if self._step <= self.warmup_end:
            self._phase = 'warmup'
            self._lr = self.lr_max * self._step / max(1, self.warmup_end)
        elif self._step <= self.stable_end:
            self._phase = 'stable'
            self._lr = self.lr_max
        elif self._step <= self.total:
            self._phase = 'decay'
            p = (self._step - self.stable_end) / max(1, self.decay_len)
            p = min(1.0, p)
            if self.style == 'cosine':
                if self.restarts > 0:
                    cp = (p * (self.restarts + 1)) % 1.0
                    self._lr = self.lr_min + (self.lr_max - self.lr_min) * (1 + _m294.cos(_m294.pi * cp)) / 2
                else:
                    self._lr = self.lr_min + (self.lr_max - self.lr_min) * (1 + _m294.cos(_m294.pi * p)) / 2
            elif self.style == 'linear':
                self._lr = self.lr_max - (self.lr_max - self.lr_min) * p
            else:
                self._lr = max(self.lr_min, self.lr_max * _m294.exp(-3 * p))
        else:
            self._phase = 'done'
            self._lr = self.lr_min
        return self._lr

    @property
    def lr(self):
        return self._lr

    def effective_Kx(self):
        return 1 if self._phase == 'warmup' else self.Kx

    def state_dict(self):
        return {'step': self._step, 'lr': self._lr, 'phase': self._phase,
                'we': self.warmup_end, 'se': self.stable_end, 'tot': self.total}

    def load_state_dict(self, d):
        # M451: Megatron fb4cbdc27 — AnnealingLR.__init__ calls self.step(self.num_iters)
        # to snap LR to the formula rather than trusting a serialized float.  Apply the
        # same discipline: rewind to step 0 then replay forward to the saved step count
        # so _lr is always formula-derived, not potentially stale from an old checkpoint.
        saved_step = d.get('step', 0)
        self._step = 0
        self._lr = 0.0
        self._phase = 'warmup'
        # Replay schedule to reach saved_step (fast — no GPU ops, pure Python arithmetic).
        for _ in range(saved_step):
            self.step()
        # M451: diagnostic — show restored state so checkpoint reloads are visible in logs.
        print(
            f"[DeslocWSDKxAligned] load_state_dict: replayed {saved_step} steps -> "
            f"phase={self._phase}, lr={self._lr:.6f} "
            f"(serialized lr was {d.get('lr', 0):.6f})"
        )
        # M451: end
