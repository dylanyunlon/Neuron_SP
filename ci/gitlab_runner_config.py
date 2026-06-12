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

# ---------------------------------------------------------------------------
# M1395: Migrated from Megatron-LM commit 2fb158351
# "print exit code"
#
# Upstream change (.gitlab-ci.yml, cleanup.selene job):
#   - find ${SELENE_ADLR_CI_PATH} -type d -ctime +20 | grep -v data | wc -l
#   + find ${SELENE_ADLR_CI_PATH} -type d -ctime +20 | grep -v data | wc -l || echo exit code $?
#
# The || echo exit code $? guard prevents the shell from aborting on a
# non-zero find/grep exit (e.g. when the directory is empty or grep finds
# no matches).  In Neuron_SP CI helpers we apply the same pattern:
# wrap any pipeline that may legitimately return non-zero in an explicit
# exit-code echo so the log remains informative rather than failing the job.
# ---------------------------------------------------------------------------

print('[M1395]')


def count_old_dirs(base_path: str, ctime_days: int = 20) -> int:
    """Count directories older than *ctime_days* days under *base_path*.

    Mirrors the cleanup.selene CI step with the M1395 exit-code guard:
        find <path> -type d -ctime +N | grep -v data | wc -l || echo exit code $?

    Returns the count, or -1 if the path does not exist / find fails.
    """
    import subprocess

    cmd = (
        f"find {base_path} -type d -ctime +{ctime_days} "
        f"| grep -v data | wc -l || echo exit code $?"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout.strip()
    if output.startswith("exit code"):
        print(f"[M1395] cleanup scan returned: {output}")
        return -1
    try:
        return int(output)
    except ValueError:
        return -1


# Runner tag required for GPU-enabled CI (mirrors Megatron docker_gpu_enabled).
REQUIRED_RUNNER_TAG = "docker_gpu_enabled"

# Pytest invocation used in upstream unit-test CI job (adapted for DeepSpeed).
PYTEST_CMD = (
    "torchrun --nproc_per_node=2 -m pytest "
    "--cov-report=term --cov-report=html "
    "--cov=deepspeed tests/"
)
