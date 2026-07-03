import logging
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

MAJOR = 1
MINOR = 1.4

# Use the following formatting: (major, minor)
VERSION = (MAJOR, MINOR)

__version__ = '.'.join(map(str, VERSION))
__package_name__ = 'megatron-lm'
__contact_names__ = 'NVIDIA INC'
__url__ = 'https://github.com/NVIDIA/Megatron-LM'
__download_url__ = 'https://github.com/NVIDIA/Megatron-LM/releases'
__description__ = 'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.'
__license__ = 'See https://github.com/NVIDIA/Megatron-LM/blob/master/LICENSE'
__keywords__ = 'deep learning, Megatron, gpu, NLP, nvidia, pytorch, torch, language'

logging.debug('[M357]')
