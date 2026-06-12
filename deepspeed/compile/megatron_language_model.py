# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1344: Megatron 6b50a8c64 — assertion check for T5 and untied embeddings
# Source: megatron/model/language_model.py (NVIDIA/Megatron-LM commit 6b50a8c64)
# Author: Jimmy Zhang <jiemingz@nvidia.com>  Date: 2023-04-03
#
# Mapping: megatron/model/language_model.py → deepspeed/compile/megatron_language_model.py
#          (project convention: megatron/model/* → deepspeed/compile/)
#
# Change ported from language_model.py TransformerLanguageModel.__init__():
#
#   BEFORE: super(TransformerLanguageModel, self).__init__(
#               share_word_embeddings=not args.untie_embeddings_and_output_weights)
#
#   AFTER:  if args.untie_embeddings_and_output_weights: assert not add_decoder
#           super(TransformerLanguageModel, self).__init__(
#               share_word_embeddings=not args.untie_embeddings_and_output_weights)
#
# Rationale: when untie_embeddings_and_output_weights is True, the model uses
# separate input and output embedding matrices.  This mode is incompatible with
# the T5 decoder (add_decoder=True) because the cross-stage embedding
# synchronisation path in MegatronModule.initialize_word_embeddings() assumes
# tied weights.  The assertion catches the invalid combination early so the
# error message is clear rather than producing a silent weight mismatch or a
# confusing hang during distributed initialisation.
#
# 20% adaptation: exposed as a standalone guard function
# assert_no_decoder_with_untied_embeddings() so that DeepSpeed model-init
# code can call it directly without instantiating TransformerLanguageModel.
# Adds print('[M1344]') marker.
# ---------------------------------------------------------------------------

print('[M1344]')


def assert_no_decoder_with_untied_embeddings(args, add_decoder):
    """Assert that untied embeddings and T5 decoder are not used together.

    Megatron 6b50a8c64 language_model.py TransformerLanguageModel.__init__():
      if args.untie_embeddings_and_output_weights: assert not add_decoder

    When untie_embeddings_and_output_weights=True the input embedding and the
    output projection use independent weight matrices.  The T5 decoder path
    (add_decoder=True) relies on MegatronModule.initialize_word_embeddings()
    to synchronise tied weights across pipeline stages; that synchronisation
    is undefined when weights are untied.  Catching this combination at
    construction time produces a clear error instead of a silent correctness
    bug or a distributed deadlock.

    Args:
        args: namespace with untie_embeddings_and_output_weights attribute.
        add_decoder (bool): True if the model is being built with a decoder
            (T5-style encoder-decoder architecture).

    Raises:
        AssertionError: if untie_embeddings_and_output_weights is True and
            add_decoder is also True.
    """
    untie = getattr(args, 'untie_embeddings_and_output_weights', False)
    if untie:
        assert not add_decoder, (
            "untie_embeddings_and_output_weights=True is incompatible with "
            "add_decoder=True (T5).  Tied embeddings are required for the "
            "cross-stage embedding synchronisation used by the T5 decoder."
        )
    print(f'[M1344] assert_no_decoder_with_untied_embeddings: '
          f'untie={untie} add_decoder={add_decoder} — OK')
