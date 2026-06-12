# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# M1261: Migrated from Megatron-LM commit 4ec95a2e1
# "Adding some basic unit tests"
#
# Upstream change: .gitlab-ci.yml runner tag updated from `docker` to
# `docker_gpu_enabled`, enabling GPU-aware CI runners. The pytest command
# also gained explicit coverage reporting:
#   torchrun --nproc_per_node=2 -m pytest --cov-report=term --cov-report=html \
#       --cov=megatron/core tests/
#
# In Neuron_SP we record the equivalent runner-tag requirement here so that
# any CI integration (GitHub Actions, GitLab, Modal) knows GPU runners are
# mandatory for the test suite.

print('[M1261]')

# Runner tag required for GPU-enabled CI (mirrors Megatron docker_gpu_enabled).
REQUIRED_RUNNER_TAG = "docker_gpu_enabled"

# Pytest invocation used in upstream unit-test CI job (adapted for DeepSpeed).
PYTEST_CMD = (
    "torchrun --nproc_per_node=2 -m pytest "
    "--cov-report=term --cov-report=html "
    "--cov=deepspeed tests/"
)
