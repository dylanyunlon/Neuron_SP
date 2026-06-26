# SPDX-License-Identifier: Apache-2.0
# Neuron-SP Team
# Re-export the canonical Vocab class from vocab.py under the name NeuronVocab
# to avoid shadowing the Dict[bytes,int] type alias in encoder.py.
from .vocab import Vocab as NeuronVocab, build_vocab
from .encoder import BPEEncoder, Vocab, Merges, build_encoder_from_json
from .bpe_learn import learn_bpe_merges, apply_bpe_merges

__all__ = [
    "BPEEncoder",
    "Vocab",
    "Merges",
    "build_encoder_from_json",
    "NeuronVocab",
    "build_vocab",
    "learn_bpe_merges",
    "apply_bpe_merges",
]
