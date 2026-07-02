"""
CommitDataset: Megatron-compatible dataset for pretraining on GitHub commits.

Loads formatted commit text (from commit_tokenizer.py), tokenizes,
and serves samples compatible with Megatron's GPT training loop.

Key design choices:
- Respects <COMMIT> boundaries (never splits mid-commit)
- Commit-aware loss masking: <ADD>/<DEL> full weight, <CTX> half, structural tokens zero
- Optional FIM (Fill-in-the-Middle) transformation
- Supports heterogeneous GPU pipeline parallelism
"""

import os
import random
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CommitDatasetConfig:
    """Configuration for CommitDataset."""

    def __init__(
        self,
        data_path: str,
        seq_length: int = 2048,
        seed: int = 42,
        fim_rate: float = 0.0,
        ctx_loss_weight: float = 0.5,
        structural_loss_weight: float = 0.0,
        split: str = "train",
        split_ratio: Tuple[float, float, float] = (0.98, 0.01, 0.01),
    ):
        self.data_path = data_path
        self.seq_length = seq_length
        self.seed = seed
        self.fim_rate = fim_rate
        self.ctx_loss_weight = ctx_loss_weight
        self.structural_loss_weight = structural_loss_weight
        self.split = split
        self.split_ratio = split_ratio


# Token IDs for loss masking (will be set by tokenizer)
# These are placeholder indices — actual IDs assigned after BPE training
STRUCTURAL_TOKENS = {
    "<COMMIT>", "</COMMIT>", "<MSG>", "</MSG>", "<AUTHOR>", "</AUTHOR>",
    "<FILE>", "</FILE>", "<HUNK>", "</HUNK>",
    "<ADD>", "</ADD>", "<DEL>", "</DEL>", "<CTX>", "</CTX>",
    "<README>", "</README>", "<TREE>", "</TREE>", "<REF>", "</REF>",
    "<RENAME>", "<DELETE>", "<CREATE>", "<BINARY>",
    "<pad>", "<bos>", "<eos>", "<sep>",
}

# Tokens whose CONTENT gets full loss weight
FULL_WEIGHT_REGIONS = {"<ADD>", "<DEL>", "<MSG>"}

# Tokens whose CONTENT gets reduced weight
REDUCED_WEIGHT_REGIONS = {"<CTX>"}


class CommitDataset(Dataset):
    """
    Dataset that loads pre-formatted commit text and serves tokenized
    samples for autoregressive pretraining.

    Each sample is a sequence of token IDs of length `seq_length`,
    packed from one or more commits (never splitting mid-commit).

    Returns dict with:
        tokens:       [seq_length]   input token IDs
        labels:       [seq_length]   target token IDs (tokens shifted left by 1)
        loss_mask:    [seq_length]   per-token loss weight
        position_ids: [seq_length]   position indices
    """

    def __init__(self, config: CommitDatasetConfig, tokenizer=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.seq_length = config.seq_length

        # Load and parse commits
        logger.info(f"Loading commit data from {config.data_path}")
        self.commits = self._load_commits(config.data_path)

        # Split
        rng = random.Random(config.seed)
        indices = list(range(len(self.commits)))
        rng.shuffle(indices)

        n = len(indices)
        train_end = int(n * config.split_ratio[0])
        val_end = train_end + int(n * config.split_ratio[1])

        if config.split == "train":
            self.indices = indices[:train_end]
        elif config.split == "valid":
            self.indices = indices[train_end:val_end]
        elif config.split == "test":
            self.indices = indices[val_end:]
        else:
            self.indices = indices

        # Pack commits into fixed-length samples
        self.samples = self._pack_samples()

        logger.info(
            f"CommitDataset [{config.split}]: "
            f"{len(self.commits)} commits → {len(self.samples)} samples "
            f"(seq_length={config.seq_length})"
        )

    def _load_commits(self, data_path: str) -> List[str]:
        """Load formatted commit text, split by <COMMIT>...</COMMIT> boundaries."""
        commits = []

        if os.path.isdir(data_path):
            files = sorted(
                os.path.join(data_path, f)
                for f in os.listdir(data_path)
                if f.endswith('.txt')
            )
        else:
            files = [data_path]

        for fpath in files:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            # Split on </COMMIT> to get individual commits
            parts = content.split('</COMMIT>')
            for part in parts:
                part = part.strip()
                if '<COMMIT>' in part:
                    # Re-add the closing tag
                    commit_text = part[part.index('<COMMIT>'):] + '\n</COMMIT>'
                    commits.append(commit_text)

        return commits

    def _tokenize_commit(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Tokenize a commit and compute per-token loss weights.

        Returns:
            token_ids: list of int
            loss_weights: list of float (same length)
        """
        if self.tokenizer is None:
            # Fallback: simple char-level for testing
            token_ids = [ord(c) % 32000 for c in text]
            loss_weights = [1.0] * len(token_ids)
            return token_ids, loss_weights

        token_ids = self.tokenizer.encode(text)

        # Compute loss weights based on structural context
        loss_weights = []
        current_region = None
        special_ids = getattr(self.tokenizer, 'special_token_ids', {})
        id_to_token = {v: k for k, v in special_ids.items()}

        for tid in token_ids:
            token_str = id_to_token.get(tid, None)

            if token_str is not None:
                # This is a structural token
                if token_str in FULL_WEIGHT_REGIONS:
                    current_region = "full"
                elif token_str in REDUCED_WEIGHT_REGIONS:
                    current_region = "reduced"
                elif token_str.startswith("</"):
                    current_region = None

                # Structural tokens themselves get low/zero weight
                loss_weights.append(self.config.structural_loss_weight)
            else:
                # Content token — weight depends on region
                if current_region == "full":
                    loss_weights.append(1.0)
                elif current_region == "reduced":
                    loss_weights.append(self.config.ctx_loss_weight)
                else:
                    loss_weights.append(0.5)  # default for unlabeled regions

        return token_ids, loss_weights

    def _apply_fim(
        self, token_ids: List[int], loss_weights: List[float]
    ) -> Tuple[List[int], List[float]]:
        """
        Apply Fill-in-the-Middle transformation.

        Randomly select a span within <ADD> regions and rearrange:
        [prefix] [suffix] [middle] → model learns to fill in the middle.
        """
        if random.random() > self.config.fim_rate:
            return token_ids, loss_weights

        # Simple FIM: pick a random split point in the middle third
        n = len(token_ids)
        if n < 10:
            return token_ids, loss_weights

        # Split into prefix, middle, suffix
        split1 = random.randint(n // 4, n // 2)
        split2 = random.randint(n // 2, 3 * n // 4)

        prefix = token_ids[:split1]
        middle = token_ids[split1:split2]
        suffix = token_ids[split2:]

        prefix_w = loss_weights[:split1]
        middle_w = loss_weights[split1:split2]
        suffix_w = loss_weights[split2:]

        # FIM token IDs (if tokenizer has them)
        special_ids = getattr(self.tokenizer, 'special_token_ids', {})
        fim_pre = special_ids.get('<fim_prefix>', 0)
        fim_suf = special_ids.get('<fim_suffix>', 0)
        fim_mid = special_ids.get('<fim_middle>', 0)

        # Rearrange: <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle
        new_ids = [fim_pre] + prefix + [fim_suf] + suffix + [fim_mid] + middle
        new_weights = [0.0] + prefix_w + [0.0] + [0.0] * len(suffix) + [0.0] + middle_w

        return new_ids, new_weights

    def _pack_samples(self) -> List[Tuple[List[int], List[float]]]:
        """
        Pack tokenized commits into fixed-length samples.

        Never splits a commit across samples. If a commit exceeds
        seq_length, truncate the diff portion (keep <MSG>).
        """
        samples = []
        current_ids = []
        current_weights = []

        for idx in self.indices:
            commit_text = self.commits[idx]
            token_ids, loss_weights = self._tokenize_commit(commit_text)

            # Apply FIM
            token_ids, loss_weights = self._apply_fim(token_ids, loss_weights)

            # Truncate if single commit exceeds seq_length
            if len(token_ids) > self.seq_length:
                token_ids = token_ids[:self.seq_length]
                loss_weights = loss_weights[:self.seq_length]

            # Check if adding this commit would exceed seq_length
            if len(current_ids) + len(token_ids) > self.seq_length:
                # Pad current sample and save
                if current_ids:
                    samples.append(self._pad_sample(current_ids, current_weights))
                current_ids = token_ids
                current_weights = loss_weights
            else:
                current_ids.extend(token_ids)
                current_weights.extend(loss_weights)

        # Flush remaining
        if current_ids:
            samples.append(self._pad_sample(current_ids, current_weights))

        return samples

    def _pad_sample(
        self, token_ids: List[int], loss_weights: List[float]
    ) -> Tuple[List[int], List[float]]:
        """Pad or truncate to exactly seq_length."""
        pad_id = 0  # <pad> token
        n = len(token_ids)

        if n >= self.seq_length:
            return token_ids[:self.seq_length], loss_weights[:self.seq_length]

        # Pad
        token_ids = token_ids + [pad_id] * (self.seq_length - n)
        loss_weights = loss_weights + [0.0] * (self.seq_length - n)
        return token_ids, loss_weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        token_ids, loss_weights = self.samples[idx]

        tokens = torch.tensor(token_ids, dtype=torch.long)
        loss_mask = torch.tensor(loss_weights, dtype=torch.float)

        # Autoregressive: labels = tokens shifted left by 1
        labels = torch.roll(tokens, shifts=-1)
        labels[-1] = 0  # pad the last position

        # Loss mask: also shift to align with labels
        loss_mask_shifted = torch.roll(loss_mask, shifts=-1)
        loss_mask_shifted[-1] = 0.0

        # Position IDs
        position_ids = torch.arange(self.seq_length, dtype=torch.long)

        return {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask_shifted,
            'position_ids': position_ids,
        }


def build_commit_datasets(
    data_path: str,
    seq_length: int = 2048,
    tokenizer=None,
    fim_rate: float = 0.0,
    seed: int = 42,
) -> Tuple[CommitDataset, CommitDataset, CommitDataset]:
    """
    Build train/valid/test CommitDatasets.

    Args:
        data_path: Path to formatted commit corpus (file or directory)
        seq_length: Sequence length for training
        tokenizer: CommitTokenizer instance
        fim_rate: Probability of applying FIM transformation
        seed: Random seed for reproducibility

    Returns:
        (train_dataset, valid_dataset, test_dataset)
    """
    datasets = []
    for split in ["train", "valid", "test"]:
        config = CommitDatasetConfig(
            data_path=data_path,
            seq_length=seq_length,
            seed=seed,
            fim_rate=fim_rate if split == "train" else 0.0,
            split=split,
        )
        ds = CommitDataset(config, tokenizer=tokenizer)
        datasets.append(ds)

    return tuple(datasets)


if __name__ == "__main__":
    # Quick test
    config = CommitDatasetConfig(
        data_path="/home/claude/commit_corpus/commits.txt",
        seq_length=512,
    )
    ds = CommitDataset(config)
    print(f"Dataset size: {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        print(f"Sample keys: {list(sample.keys())}")
        for k, v in sample.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        print(f"  tokens[:20]: {sample['tokens'][:20].tolist()}")
        print(f"  loss_mask sum: {sample['loss_mask'].sum().item():.1f}")
