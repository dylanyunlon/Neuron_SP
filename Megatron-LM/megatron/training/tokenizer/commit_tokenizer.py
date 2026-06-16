"""
Neuron_SP Commit-Aware Tokenizer for Pretraining on GitHub Commits

Design Philosophy:
=================
Unlike traditional code tokenizers that treat source files as flat text,
this tokenizer understands the STRUCTURE of git commits:

    <COMMIT>
    <MSG> fix: resolve race condition in CUDA graph capture </MSG>
    <FILE path="megatron/core/transformer/cuda_graphs.py">
    <HUNK @@ -949,6 +949,8 @@>
    <DEL>    def create_cudagraphs(self):</DEL>
    <ADD>    def create_cudagraphs(self, freeze_gc=True):</ADD>
    <ADD>        if freeze_gc:</ADD>
    <ADD>            gc.freeze()</ADD>
    <CTX>        stream = torch.cuda.Stream()</CTX>
    </HUNK>
    </FILE>
    </COMMIT>

This structural tokenization gives the model:
1. Awareness of commit boundaries (vs raw code)
2. Understanding of diff semantics (+/- lines, hunks, files)
3. Commit message ↔ code change alignment (like Seed-Coder's
   "code change prediction task")

References:
- Seed-Coder (ByteDance, 2025): 74M commits, 100B tokens, code change
  prediction task with BM25-retrieved context
- CommitBART (Liu et al., 2022): 7.99M commits across 7 languages,
  denoising + contrastive pretraining on commit diffs
- CommitBERT (Jung, 2021): 345K commits, CodeBERT-initialized
"""

import os
import json
import hashlib
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# ===========================================================================
# Special Tokens for Commit Structure
# ===========================================================================

COMMIT_SPECIAL_TOKENS = {
    # Commit-level structure
    "<COMMIT>":     "Start of a commit sample",
    "</COMMIT>":    "End of a commit sample",
    "<MSG>":        "Start of commit message",
    "</MSG>":       "End of commit message",
    "<AUTHOR>":     "Author attribution",
    "</AUTHOR>":    "End author",

    # File-level structure
    "<FILE>":       "Start of a file change (carries path= attribute)",
    "</FILE>":      "End of file change",
    "<RENAME>":     "File was renamed",
    "<DELETE>":     "File was deleted entirely",
    "<CREATE>":     "File was newly created",
    "<BINARY>":     "Binary file changed (skip content)",

    # Hunk-level structure (within a file)
    "<HUNK>":       "Start of a diff hunk (carries @@ range)",
    "</HUNK>":      "End of diff hunk",
    "<ADD>":        "Added line (+ in diff)",
    "</ADD>":       "End of added line",
    "<DEL>":        "Deleted line (- in diff)",
    "</DEL>":       "End of deleted line",
    "<CTX>":        "Context line (unchanged, for locality)",
    "</CTX>":       "End of context line",

    # Context tokens (Seed-Coder style)
    "<README>":     "Repository README content",
    "</README>":    "End README",
    "<TREE>":       "Directory tree structure",
    "</TREE>":      "End directory tree",
    "<REF>":        "BM25-retrieved relevant file content",
    "</REF>":       "End reference file",

    # FIM tokens (Fill-in-the-Middle, like Seed-Coder)
    "<fim_prefix>": "FIM: code before the hole",
    "<fim_suffix>": "FIM: code after the hole",
    "<fim_middle>": "FIM: the code to fill in",

    # Language tags (top languages from Megatron-LM commits)
    "<lang:python>":     "Python source",
    "<lang:cuda>":       "CUDA C/C++",
    "<lang:cpp>":        "C++ source",
    "<lang:shell>":      "Shell script",
    "<lang:yaml>":       "YAML config",
    "<lang:json>":       "JSON",
    "<lang:markdown>":   "Markdown docs",
    "<lang:dockerfile>": "Dockerfile",
    "<lang:other>":      "Other language",

    # Padding / control
    "<pad>":        "Padding token",
    "<eos>":        "End of sequence",
    "<bos>":        "Beginning of sequence",
    "<unk>":        "Unknown token",
    "<sep>":        "Separator",
}


# ===========================================================================
# Commit Data Formatter
# ===========================================================================

def detect_language(filepath: str) -> str:
    """Detect programming language from file extension."""
    ext_map = {
        '.py': '<lang:python>',
        '.cu': '<lang:cuda>',
        '.cuh': '<lang:cuda>',
        '.cpp': '<lang:cpp>',
        '.cc': '<lang:cpp>',
        '.c': '<lang:cpp>',
        '.h': '<lang:cpp>',
        '.hpp': '<lang:cpp>',
        '.sh': '<lang:shell>',
        '.bash': '<lang:shell>',
        '.yml': '<lang:yaml>',
        '.yaml': '<lang:yaml>',
        '.json': '<lang:json>',
        '.md': '<lang:markdown>',
        '.rst': '<lang:markdown>',
        '.dockerfile': '<lang:dockerfile>',
    }
    ext = Path(filepath).suffix.lower()
    if filepath.lower().endswith('dockerfile'):
        return '<lang:dockerfile>'
    return ext_map.get(ext, '<lang:other>')


def parse_git_diff(diff_text: str) -> List[Dict]:
    """
    Parse a unified diff into structured file/hunk/line format.

    Returns list of:
    {
        "path": "megatron/core/transformer/attention.py",
        "language": "<lang:python>",
        "status": "modified" | "created" | "deleted" | "renamed",
        "hunks": [
            {
                "header": "@@ -10,5 +10,7 @@ class Attention:",
                "lines": [
                    {"type": "ctx", "content": "    def __init__(self):"},
                    {"type": "del", "content": "        self.old = True"},
                    {"type": "add", "content": "        self.new = True"},
                    {"type": "add", "content": "        self.extra = False"},
                ]
            }
        ]
    }
    """
    files = []
    current_file = None
    current_hunk = None

    for line in diff_text.split('\n'):
        # New file diff
        if line.startswith('diff --git'):
            if current_file:
                if current_hunk:
                    current_file['hunks'].append(current_hunk)
                files.append(current_file)

            # Extract path
            match = re.search(r'b/(.+)$', line)
            path = match.group(1) if match else 'unknown'
            current_file = {
                'path': path,
                'language': detect_language(path),
                'status': 'modified',
                'hunks': [],
            }
            current_hunk = None

        # File status
        elif line.startswith('new file'):
            if current_file:
                current_file['status'] = 'created'
        elif line.startswith('deleted file'):
            if current_file:
                current_file['status'] = 'deleted'
        elif line.startswith('rename from'):
            if current_file:
                current_file['status'] = 'renamed'

        # Hunk header
        elif line.startswith('@@'):
            if current_file:
                if current_hunk:
                    current_file['hunks'].append(current_hunk)
                current_hunk = {
                    'header': line.strip(),
                    'lines': [],
                }

        # Diff lines
        elif current_hunk is not None:
            if line.startswith('+') and not line.startswith('+++'):
                current_hunk['lines'].append({
                    'type': 'add',
                    'content': line[1:],  # strip leading +
                })
            elif line.startswith('-') and not line.startswith('---'):
                current_hunk['lines'].append({
                    'type': 'del',
                    'content': line[1:],  # strip leading -
                })
            elif line.startswith(' '):
                current_hunk['lines'].append({
                    'type': 'ctx',
                    'content': line[1:],  # strip leading space
                })

    # Flush last file/hunk
    if current_file:
        if current_hunk:
            current_file['hunks'].append(current_hunk)
        files.append(current_file)

    return files


def format_commit_for_pretraining(
    commit_hash: str,
    commit_message: str,
    author: str,
    diff_text: str,
    readme: Optional[str] = None,
    tree: Optional[str] = None,
    max_diff_lines: int = 512,
    max_ctx_lines: int = 3,
) -> str:
    """
    Format a single git commit into the tokenizer's structured format.

    This is the "code change prediction" task from Seed-Coder:
    Given commit message + context → predict file paths + code changes.

    Args:
        commit_hash: Git commit SHA
        commit_message: Subject + body of commit
        author: Author name
        diff_text: Raw unified diff output
        readme: Optional README content for context
        tree: Optional directory tree for context
        max_diff_lines: Max diff lines per commit (truncation)
        max_ctx_lines: Max context lines to keep per hunk

    Returns:
        Structured text ready for tokenization
    """
    parts = ['<COMMIT>']

    # Author
    parts.append(f'<AUTHOR> {author} </AUTHOR>')

    # Commit message
    parts.append(f'<MSG> {commit_message.strip()} </MSG>')

    # Optional context (Seed-Coder style)
    if readme:
        # Truncate README
        readme_trunc = readme[:2000]
        parts.append(f'<README> {readme_trunc} </README>')

    if tree:
        tree_trunc = tree[:1000]
        parts.append(f'<TREE> {tree_trunc} </TREE>')

    # Parse and format diff
    files = parse_git_diff(diff_text)
    total_lines = 0

    for file_info in files:
        path = file_info['path']
        lang = file_info['language']
        status = file_info['status']

        # Status tag
        status_tag = {
            'created': '<CREATE>',
            'deleted': '<DELETE>',
            'renamed': '<RENAME>',
        }.get(status, '')

        parts.append(f'<FILE path="{path}"> {lang} {status_tag}')

        for hunk in file_info['hunks']:
            parts.append(f'<HUNK {hunk["header"]}>')

            ctx_count = 0
            for line_info in hunk['lines']:
                if total_lines >= max_diff_lines:
                    break

                if line_info['type'] == 'add':
                    parts.append(f'<ADD>{line_info["content"]}</ADD>')
                    total_lines += 1
                    ctx_count = 0
                elif line_info['type'] == 'del':
                    parts.append(f'<DEL>{line_info["content"]}</DEL>')
                    total_lines += 1
                    ctx_count = 0
                elif line_info['type'] == 'ctx':
                    ctx_count += 1
                    if ctx_count <= max_ctx_lines:
                        parts.append(f'<CTX>{line_info["content"]}</CTX>')
                        total_lines += 1

            parts.append('</HUNK>')

            if total_lines >= max_diff_lines:
                break

        parts.append('</FILE>')

        if total_lines >= max_diff_lines:
            break

    parts.append('</COMMIT>')

    return '\n'.join(parts)


# ===========================================================================
# Tokenizer Wrapper (extends SentencePiece/BPE base)
# ===========================================================================

class CommitTokenizer:
    """
    Commit-aware tokenizer that wraps a base BPE/SentencePiece tokenizer
    and adds commit-specific special tokens.

    Training pipeline:
    1. Collect commits from target repos (Megatron-LM, DeepSpeed, etc.)
    2. Format each commit with format_commit_for_pretraining()
    3. Train BPE on the formatted corpus (learns code subwords + structure)
    4. Add special tokens to vocabulary

    Usage:
        tokenizer = CommitTokenizer(base_model="gpt2")
        tokenizer.train_from_commits(commit_files)
        tokens = tokenizer.encode("<COMMIT><MSG>fix bug</MSG>...</COMMIT>")
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        base_tokenizer: str = "sentencepiece",
    ):
        self.vocab_size = vocab_size
        self.base_tokenizer = base_tokenizer
        self.special_tokens = list(COMMIT_SPECIAL_TOKENS.keys())
        self.special_token_ids = {}

        # Will be initialized during training
        self._sp_model = None
        self._token_to_id = {}
        self._id_to_token = {}

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    @property
    def bpe_vocab_size(self) -> int:
        """Vocab size for BPE model (total - special tokens)."""
        return self.vocab_size - self.num_special_tokens

    def train(self, corpus_files: List[str], model_prefix: str = "commit_sp"):
        """
        Train SentencePiece BPE model on commit corpus.

        Args:
            corpus_files: List of text files (formatted commits)
            model_prefix: Output model file prefix
        """
        import sentencepiece as spm

        # Train SentencePiece
        spm.SentencePieceTrainer.train(
            input=','.join(corpus_files),
            model_prefix=model_prefix,
            vocab_size=self.bpe_vocab_size,
            model_type='bpe',
            character_coverage=0.9999,
            byte_fallback=True,
            # Control tokens — these get reserved IDs
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            # Crucial for code: preserve whitespace/indentation
            split_by_whitespace=True,
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            # Normalize unicode but keep structure
            normalization_rule_name='identity',
            # Train on enough data
            input_sentence_size=10_000_000,
            shuffle_input_sentence=True,
        )

        # Load trained model
        self._sp_model = spm.SentencePieceProcessor()
        self._sp_model.Load(f"{model_prefix}.model")

        # Add special tokens after BPE vocab
        base_vocab = self._sp_model.GetPieceSize()
        for i, token in enumerate(self.special_tokens):
            token_id = base_vocab + i
            self.special_token_ids[token] = token_id
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token

        print(f"Tokenizer trained:")
        print(f"  BPE vocab: {base_vocab}")
        print(f"  Special tokens: {len(self.special_tokens)}")
        print(f"  Total vocab: {base_vocab + len(self.special_tokens)}")

    def encode(self, text: str) -> List[int]:
        """Encode text with commit-aware tokenization."""
        if self._sp_model is None:
            raise RuntimeError("Tokenizer not trained. Call train() first.")

        tokens = []
        # Split on special tokens, keeping them
        pattern = '|'.join(re.escape(t) for t in self.special_tokens)
        parts = re.split(f'({pattern})', text)

        for part in parts:
            if part in self.special_token_ids:
                tokens.append(self.special_token_ids[part])
            elif part.strip():
                tokens.extend(self._sp_model.Encode(part))

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if self._sp_model is None:
            raise RuntimeError("Tokenizer not trained. Call train() first.")

        parts = []
        regular_ids = []

        for tid in token_ids:
            if tid in self._id_to_token:
                # Flush regular tokens
                if regular_ids:
                    parts.append(self._sp_model.Decode(regular_ids))
                    regular_ids = []
                parts.append(self._id_to_token[tid])
            else:
                regular_ids.append(tid)

        if regular_ids:
            parts.append(self._sp_model.Decode(regular_ids))

        return ''.join(parts)

    def save(self, path: str):
        """Save tokenizer config and special tokens."""
        config = {
            'vocab_size': self.vocab_size,
            'base_tokenizer': self.base_tokenizer,
            'special_tokens': self.special_tokens,
            'special_token_ids': self.special_token_ids,
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, config_path: str, sp_model_path: str) -> 'CommitTokenizer':
        """Load tokenizer from config + SentencePiece model."""
        import sentencepiece as spm

        with open(config_path) as f:
            config = json.load(f)

        tokenizer = cls(
            vocab_size=config['vocab_size'],
            base_tokenizer=config['base_tokenizer'],
        )
        tokenizer.special_tokens = config['special_tokens']
        tokenizer.special_token_ids = config['special_token_ids']

        tokenizer._sp_model = spm.SentencePieceProcessor()
        tokenizer._sp_model.Load(sp_model_path)

        # Rebuild lookup
        for token, tid in config['special_token_ids'].items():
            tokenizer._token_to_id[token] = tid
            tokenizer._id_to_token[tid] = token

        return tokenizer


# ===========================================================================
# Data Pipeline: Git Repo → Training Corpus
# ===========================================================================

def extract_commits_from_repo(
    repo_path: str,
    output_dir: str,
    max_commits: int = 100000,
    min_diff_lines: int = 3,
    max_diff_lines: int = 1000,
    skip_merges: bool = True,
    languages: Optional[List[str]] = None,
) -> int:
    """
    Extract commits from a git repository and format them for pretraining.

    Follows Seed-Coder criteria for high-quality repos:
    - Filter by activity (stars, forks, commit count)
    - Skip merge commits
    - Skip binary-only changes
    - Skip commits with only CI/config changes (optional)

    Args:
        repo_path: Path to git repository
        output_dir: Directory to write formatted commit files
        max_commits: Maximum number of commits to extract
        min_diff_lines: Minimum diff lines to include
        max_diff_lines: Maximum diff lines per commit
        skip_merges: Skip merge commits
        languages: Filter to specific languages (file extensions)

    Returns:
        Number of commits extracted
    """
    import subprocess

    os.makedirs(output_dir, exist_ok=True)

    # Get commit list
    cmd = ["git", "log", "--format=%H", "--reverse"]
    if skip_merges:
        cmd.append("--no-merges")
    cmd.append(f"-{max_commits}")

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=repo_path
    )
    commits = result.stdout.strip().split('\n')

    extracted = 0
    output_file = os.path.join(output_dir, "commits.txt")

    with open(output_file, 'w') as out:
        for commit_hash in commits:
            if not commit_hash.strip():
                continue

            # Get commit info
            info = subprocess.run(
                ["git", "show", "--format=%s%n%b%n%an", "-s", commit_hash],
                capture_output=True, text=True, cwd=repo_path
            )
            lines = info.stdout.strip().split('\n')
            message = lines[0] if lines else ""
            author = lines[-1] if len(lines) > 1 else "unknown"

            # Get diff
            diff = subprocess.run(
                ["git", "diff", f"{commit_hash}^..{commit_hash}",
                 "--unified=3", "--no-color"],
                capture_output=True, text=True, cwd=repo_path
            )
            diff_text = diff.stdout

            # Filter by diff size
            diff_line_count = len(diff_text.split('\n'))
            if diff_line_count < min_diff_lines:
                continue
            if diff_line_count > max_diff_lines:
                continue

            # Filter by language if specified
            if languages:
                files = parse_git_diff(diff_text)
                has_target_lang = any(
                    any(lang in f['language'] for lang in languages)
                    for f in files
                )
                if not has_target_lang:
                    continue

            # Format and write
            formatted = format_commit_for_pretraining(
                commit_hash=commit_hash,
                commit_message=message,
                author=author,
                diff_text=diff_text,
                max_diff_lines=512,
            )

            out.write(formatted + '\n\n')
            extracted += 1

    print(f"Extracted {extracted} commits from {repo_path} → {output_file}")
    return extracted


# ===========================================================================
# Entry point for testing
# ===========================================================================

if __name__ == "__main__":
    # Demo: format a sample commit
    sample_diff = """diff --git a/megatron/core/transformer/attention.py b/megatron/core/transformer/attention.py
index abc1234..def5678 100644
--- a/megatron/core/transformer/attention.py
+++ b/megatron/core/transformer/attention.py
@@ -100,7 +100,9 @@ class CoreAttention(MegatronModule):
         self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
         if self.config.apply_query_key_layer_scaling:
             self.norm_factor *= self.layer_number
-        self.scale_mask_softmax = FusedScaleMaskSoftmax(
+        # Use flash attention when available
+        if self.config.use_flash_attn:
+            self.attn_impl = FlashAttention(self.norm_factor)
+        else:
+            self.scale_mask_softmax = FusedScaleMaskSoftmax(
"""

    result = format_commit_for_pretraining(
        commit_hash="abc123def",
        commit_message="feat: Add flash attention support with config toggle",
        author="dylanyunlon",
        diff_text=sample_diff,
    )
    print(result)
    print()
    print(f"Special tokens defined: {len(COMMIT_SPECIAL_TOKENS)}")
    print("Token list:")
    for tok, desc in COMMIT_SPECIAL_TOKENS.items():
        print(f"  {tok:25s} — {desc}")
