# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""deepspeed.core.transformer.experimental_attention_variant — experimental attention variants.

Ported from Megatron-LM (megatron.core → deepspeed.core). Addresses #111-115.
"""

from deepspeed.core.transformer.experimental_attention_variant.absorbed_mla import (  # noqa: F401
    AbsorbedMLASelfAttention,
    AbsorbedMLASelfAttentionSubmodules,
)
from deepspeed.core.transformer.experimental_attention_variant.dsa import (  # noqa: F401
    DSAttention,
    DSAttentionSubmodules,
    DSAIndexer,
    DSAIndexerSubmodules,
)
