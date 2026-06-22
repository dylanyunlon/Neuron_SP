"""
Pretrain GPT on GitHub Commit Data.

Entry point for pretraining a code model on structured git commit data,
using Megatron's distributed training framework.

Hardware target:
  GPU0: RTX A6000 48GB (PCIe) — PP stage 0, ~20% layers
  GPU1: RTX A6000 48GB (PCIe) — PP stage 1, ~20% layers
  GPU2: H100 NVL 96GB (PCIe)  — PP stage 2, ~60% layers

Training data: Git commits formatted with commit_tokenizer.py
  <COMMIT><MSG>...</MSG><FILE>...<ADD>/<DEL>/<CTX>...</FILE></COMMIT>

Usage:
  python pretrain_commit.py \\
    --commit-data-path /data/commit_corpus/ \\
    --num-layers 32 \\
    --hidden-size 4096 \\
    --num-attention-heads 32 \\
    --seq-length 2048 \\
    --micro-batch-size 1 \\
    --global-batch-size 8 \\
    --pipeline-model-parallel-size 3 \\
    --bf16
"""

import os
import sys
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch

# Megatron imports
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    pretrain,
    print_rank_0,
)
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.datasets.commit_dataset import (
    CommitDataset,
    CommitDatasetConfig,
    build_commit_datasets,
)
from megatron.core.datasets.commitpack_streaming_dataset import (
    CommitPackStreamingDataset,
    CommitPackStreamingConfig,
    build_commitpack_dataloader,
)
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)


def model_provider(pre_process=True, post_process=True):
    """
    Build GPT model for commit pretraining.

    For heterogeneous PP:
    - 7B model: 32 layers, hidden=4096, heads=32
    - PP stage 0 (A6000): layers 0-5   (6 layers, ~20%)
    - PP stage 1 (A6000): layers 6-11  (6 layers, ~20%)
    - PP stage 2 (H100):  layers 12-31 (20 layers, ~60%)
    """
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    args = get_args()

    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        params_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        bf16=args.bf16,
        fp16=args.fp16,
    )

    model = GPTModel(
        config=config,
        transformer_layer_spec=None,  # uses default
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.seq_length,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=True,
    )

    return model


def get_batch(data_iterator):
    """
    Generate a batch from CommitDataset.

    Returns: tokens, labels, loss_mask, attention_mask, position_ids
    """
    args = get_args()

    # Check pipeline stage
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()

    # Only first and last PP stages need data
    if pp_rank != 0 and pp_rank != pp_size - 1:
        return None, None, None, None, None

    if data_iterator is None:
        return None, None, None, None, None

    # Get batch from iterator
    batch = next(data_iterator)

    tokens = batch['tokens'].cuda(non_blocking=True)
    labels = batch['labels'].cuda(non_blocking=True)
    loss_mask = batch['loss_mask'].cuda(non_blocking=True)
    position_ids = batch['position_ids'].cuda(non_blocking=True)

    # Create causal attention mask
    attention_mask = None  # FlashAttention handles this internally
    # For non-flash: create lower-triangular mask
    if not getattr(args, 'use_flash_attn', False):
        seq_length = tokens.size(1)
        attention_mask = torch.tril(
            torch.ones(
                (1, seq_length, seq_length),
                device=tokens.device,
                dtype=torch.bool,
            )
        )

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """
    Commit-aware loss function.

    Uses the per-token loss_mask from CommitDataset:
    - <ADD>/<DEL> content: weight 1.0 (primary prediction target)
    - <MSG> content: weight 1.0 (commit message understanding)
    - <CTX> content: weight 0.5 (context, less important)
    - Structural tokens: weight 0.0 (don't predict these)
    """
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()

    # Weighted cross-entropy
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)

    # Reporting
    report = {
        'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)]),
    }

    return loss, num_tokens, report


def forward_step(data_iterator, model):
    """Forward step for commit pretraining."""
    timers = get_timers()
    args = get_args()

    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(
        tokens,
        position_ids,
        attention_mask,
        labels=labels,
    )

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build CommitDatasets for train/valid/test."""
    args = get_args()

    print_rank_0("> building commit datasets ...")

    commit_data_path = getattr(args, 'commit_data_path', None)
    if commit_data_path is None:
        # Fallback to standard data path
        commit_data_path = getattr(args, 'data_path', [None])[0]

    if commit_data_path is None:
        raise ValueError(
            "Must specify --commit-data-path or --data-path for commit pretraining"
        )

    fim_rate = getattr(args, 'fim_rate', 0.0)

    train_ds, valid_ds, test_ds = build_commit_datasets(
        data_path=commit_data_path,
        seq_length=args.seq_length,
        tokenizer=None,  # Will use CommitTokenizer when trained
        fim_rate=fim_rate,
        seed=args.seed,
    )

    print_rank_0(f"  train: {len(train_ds)} samples")
    print_rank_0(f"  valid: {len(valid_ds)} samples")
    print_rank_0(f"  test:  {len(test_ds)} samples")

    return train_ds, valid_ds, test_ds


def train_valid_test_datasets_provider_streaming(train_val_test_num_samples):
    """
    Build CommitPack streaming datasets for train/valid/test.

    Used when --commitpack-streaming is set.  Returns an IterableDataset
    for train and falls back to the static CommitDataset for valid/test
    (streaming validation is not yet supported by Megatron's eval loop).
    """
    args = get_args()
    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()

    languages_str = getattr(args, 'commitpack_languages', 'python')
    languages = [l.strip() for l in languages_str.split(',')]

    resume_path = getattr(args, 'commitpack_resume_path', None)
    tokenizer_name = getattr(args, 'commitpack_tokenizer', 'gpt2')

    print_rank_0(
        f"> building CommitPack streaming dataset: "
        f"langs={languages}, rank={rank}/{world_size}"
    )

    train_config = CommitPackStreamingConfig(
        languages=languages,
        split="train",
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        rank=rank,
        world_size=world_size,
        tokenizer_name=tokenizer_name,
        resume_path=resume_path,
        prefetch_batches=getattr(args, 'commitpack_prefetch_batches', 8),
        tokenizer_workers=getattr(args, 'commitpack_tokenizer_workers', 8),
        numa_aware=getattr(args, 'commitpack_numa_aware', True),
    )
    train_ds = CommitPackStreamingDataset(train_config)

    # valid / test: reuse static CommitDataset if commit_data_path available
    commit_data_path = getattr(args, 'commit_data_path', None)
    if commit_data_path:
        _, valid_ds, test_ds = build_commit_datasets(
            data_path=commit_data_path,
            seq_length=args.seq_length,
            tokenizer=None,
            seed=args.seed,
        )
    else:
        # Minimal dummy datasets so Megatron eval loop doesn't crash
        valid_ds = train_ds
        test_ds = train_ds

    return train_ds, valid_ds, test_ds


def add_commit_args(parser):
    """Add commit-specific arguments."""
    group = parser.add_argument_group(title='commit pretraining')
    group.add_argument(
        '--commit-data-path',
        type=str,
        default=None,
        help='Path to formatted commit corpus for static dataset (file or directory)',
    )
    group.add_argument(
        '--fim-rate',
        type=float,
        default=0.0,
        help='Fill-in-the-Middle transformation probability (0.0-1.0)',
    )
    group.add_argument(
        '--ctx-loss-weight',
        type=float,
        default=0.5,
        help='Loss weight for <CTX> (context) lines in diffs',
    )
    # ── CommitPack streaming 参数 ─────────────────────────────────────────────
    group.add_argument(
        '--commitpack-streaming',
        action='store_true',
        default=False,
        help='Use CommitPack 4TB HuggingFace streaming dataset instead of local files',
    )
    group.add_argument(
        '--commitpack-languages',
        type=str,
        default='python',
        help='Comma-separated list of CommitPack language subsets, e.g. "python,javascript,typescript"',
    )
    group.add_argument(
        '--commitpack-tokenizer',
        type=str,
        default='gpt2',
        help='HuggingFace tokenizer name or local path for CommitPack streaming',
    )
    group.add_argument(
        '--commitpack-resume-path',
        type=str,
        default=None,
        help='Base path for CommitPack resume JSON files (suffix _rankN.json added automatically)',
    )
    group.add_argument(
        '--commitpack-prefetch-batches',
        type=int,
        default=8,
        dest='commitpack_prefetch_batches',
        help='Number of batches to prefetch asynchronously into GPU memory',
    )
    group.add_argument(
        '--commitpack-tokenizer-workers',
        type=int,
        default=8,
        dest='commitpack_tokenizer_workers',
        help='Number of CPU threads for parallel tokenization (default: 8, EPYC can handle 32+)',
    )
    group.add_argument(
        '--commitpack-numa-aware',
        action='store_true',
        default=True,
        dest='commitpack_numa_aware',
        help='Pin prefetch threads to NUMA node matching each GPU (default: True)',
    )
    return parser


if __name__ == "__main__":
    # Register custom args
    from megatron.training.arguments import parse_args
    extra_args_provider = add_commit_args

    # Select dataset provider based on --commitpack-streaming flag
    import sys
    use_streaming = '--commitpack-streaming' in sys.argv

    dataset_provider = (
        train_valid_test_datasets_provider_streaming
        if use_streaming
        else train_valid_test_datasets_provider
    )

    pretrain(
        dataset_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=extra_args_provider,
    )
