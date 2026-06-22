# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
stackv2_commits.py — The Stack v2 PR/commit subset adapter (M761-M775)

Converts The Stack v2 PR/commit parquet records into DES-LOC pretraining
format, applying dedup, filtering, and PII removal consistent with the
Stack v2 pipeline (StarCoder2, arXiv:2402.19173).

Unified output format (DES-LOC diff tokens):
    <|diff_start|>
    <|lang|>python
    <|file_path|> src/foo.py
    <|old|>
    <old file content or context lines>
    <|new|>
    <new file content or context lines>
    <|msg|> fix: handle edge case in parser
    <|diff_end|>

For PRs with multiple files, each file gets its own diff block concatenated
into one sample (up to the 100K character hard cap).

Usage (streaming from HF Hub, requires HF token + accepted agreement):
    from datasets.bigcode.the_stack_v2.stackv2_commits import StackV2CommitAdapter

    adapter = StackV2CommitAdapter()
    for sample in adapter.stream_hf(split="train", max_samples=1000):
        print(sample["text"][:200])

Usage (local parquet files):
    for sample in adapter.stream_parquet("/data/stackv2_commits/*.parquet"):
        print(sample["text"][:200])

Smoke-test:
    python stackv2_commits.py --samples 5 --source dummy
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
import struct
import sys
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Format constants (DES-LOC unified diff tokens)
# ---------------------------------------------------------------------------

DIFF_START  = "<|diff_start|>"
DIFF_END    = "<|diff_end|>"
LANG_TAG    = "<|lang|>"
FILE_PATH   = "<|file_path|>"
OLD_MARKER  = "<|old|>"
NEW_MARKER  = "<|new|>"
MSG_MARKER  = "<|msg|>"

# Hard limits (task spec)
MAX_CHARS            = 100_000      # characters per sample
MIN_CHANGED_LINES    = 10           # minimum meaningful diff size
MAX_FILES_PER_COMMIT = 50           # safety cap on per-commit file count

# Merge commit patterns to skip
_MERGE_PATTERNS = re.compile(
    r"^Merge (pull request|branch|remote-tracking branch|tag)\b",
    re.IGNORECASE,
)

# PII patterns (Stack v2 paper §3.3 — simplified regex pass)
_PII_PATTERNS = [
    # Email addresses
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    # IPv4
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # Generic secret-ish tokens: hex strings ≥ 32 chars
    re.compile(r"\b[0-9a-fA-F]{32,}\b"),
    # AWS/GCP-style access keys
    re.compile(r"\b(?:AKIA|ASIA|AROA|AIDA)[0-9A-Z]{16}\b"),
]
_PII_PLACEHOLDER = "<REDACTED>"


# ---------------------------------------------------------------------------
# Language normaliser
# ---------------------------------------------------------------------------

# Maps common extension/language strings to normalised tags
_LANG_MAP: Dict[str, str] = {
    "py":         "python",
    "python":     "python",
    "js":         "javascript",
    "javascript": "javascript",
    "ts":         "typescript",
    "typescript": "typescript",
    "rb":         "ruby",
    "ruby":       "ruby",
    "go":         "go",
    "rs":         "rust",
    "rust":       "rust",
    "cpp":        "cpp",
    "c++":        "cpp",
    "cc":         "cpp",
    "c":          "c",
    "java":       "java",
    "kt":         "kotlin",
    "kotlin":     "kotlin",
    "cs":         "csharp",
    "csharp":     "csharp",
    "php":        "php",
    "sh":         "shell",
    "bash":       "shell",
    "zsh":        "shell",
    "scala":      "scala",
    "swift":      "swift",
    "r":          "r",
    "lua":        "lua",
    "hs":         "haskell",
    "haskell":    "haskell",
    "ml":         "ocaml",
    "ocaml":      "ocaml",
    "ex":         "elixir",
    "exs":        "elixir",
    "elixir":     "elixir",
    "clj":        "clojure",
    "clojure":    "clojure",
    "jl":         "julia",
    "julia":      "julia",
    "tex":        "latex",
    "latex":      "latex",
    "md":         "markdown",
    "markdown":   "markdown",
    "yaml":       "yaml",
    "yml":        "yaml",
    "json":       "json",
    "toml":       "toml",
    "sql":        "sql",
    "dockerfile": "dockerfile",
    "makefile":   "makefile",
}


def _normalise_lang(raw: str) -> str:
    """Return a normalised language tag (lowercase, no spaces)."""
    if not raw:
        return "unknown"
    key = raw.lower().strip().lstrip(".")
    return _LANG_MAP.get(key, key.replace(" ", "_"))


def _lang_from_path(path: str) -> str:
    """Infer language tag from file extension."""
    ext = Path(path).suffix.lstrip(".").lower()
    return _normalise_lang(ext) if ext else "unknown"


# ---------------------------------------------------------------------------
# PII scrubbing
# ---------------------------------------------------------------------------

def _strip_pii(text: str) -> str:
    for pattern in _PII_PATTERNS:
        text = pattern.sub(_PII_PLACEHOLDER, text)
    return text


# ---------------------------------------------------------------------------
# Diff rendering
# ---------------------------------------------------------------------------

def _count_changed_lines(old: str, new: str) -> int:
    """Count lines that differ (added + removed)."""
    old_lines = set(old.splitlines())
    new_lines = set(new.splitlines())
    added   = sum(1 for l in new.splitlines() if l not in old_lines)
    removed = sum(1 for l in old.splitlines() if l not in new_lines)
    return added + removed


def _render_file_diff(
    path: str,
    old: str,
    new: str,
    lang: str,
    message: str,
) -> str:
    """Render a single-file diff in DES-LOC format."""
    old = _strip_pii(old)
    new = _strip_pii(new)
    return (
        f"{DIFF_START}\n"
        f"{LANG_TAG}{lang}\n"
        f"{FILE_PATH} {path}\n"
        f"{OLD_MARKER}\n{old}\n"
        f"{NEW_MARKER}\n{new}\n"
        f"{MSG_MARKER} {message}\n"
        f"{DIFF_END}"
    )


def _render_sample(files: List[Dict], message: str) -> str:
    """
    Concatenate per-file diffs for a multi-file commit/PR.

    files: list of {path, old, new, lang}
    """
    parts = []
    for f in files:
        lang = f.get("lang") or _lang_from_path(f.get("path", ""))
        parts.append(_render_file_diff(
            path=f.get("path", ""),
            old=f.get("old", ""),
            new=f.get("new", ""),
            lang=lang,
            message=message,
        ))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _is_merge_commit(message: str) -> bool:
    return bool(_MERGE_PATTERNS.match(message.strip()))


def _total_changed_lines(files: List[Dict]) -> int:
    total = 0
    for f in files:
        total += _count_changed_lines(f.get("old", ""), f.get("new", ""))
    return total


# ---------------------------------------------------------------------------
# directory_id-based dedup (Stack v2 strategy)
# ---------------------------------------------------------------------------

class DirectoryIdDeduplicator:
    """
    Tracks seen directory_id hashes and returns False for duplicates.

    Stack v2 deduplication uses the SWH directory_id (a SHA1 over the
    directory tree).  When the raw ID is unavailable we fall back to
    a SHA-256 of the concatenated file blobs, which approximates the
    intent of deduplicating identical directory contents.
    """

    def __init__(self) -> None:
        self._seen: Set[str] = set()

    def is_new(self, record: Dict) -> bool:
        did = record.get("directory_id") or record.get("dir_id")
        if not did:
            # Fallback: hash the sorted (path, blob_id) pairs
            key_parts = sorted(
                (f.get("path", ""), f.get("blob_id", ""))
                for f in record.get("files", [])
            )
            did = hashlib.sha256(json.dumps(key_parts).encode()).hexdigest()
        if did in self._seen:
            return False
        self._seen.add(did)
        return True

    @property
    def seen_count(self) -> int:
        return len(self._seen)


# ---------------------------------------------------------------------------
# Record normaliser — handles HF parquet schema variants
# ---------------------------------------------------------------------------

def _normalise_record(raw: Dict) -> Optional[Dict]:
    """
    Normalise a raw Stack v2 PR/commit record into internal format:
        {
            "directory_id": str | None,
            "message": str,
            "files": [{"path", "old", "new", "lang", "blob_id"}, ...],
        }

    Returns None if the record is structurally unusable.
    """
    message = (
        raw.get("commit_message")
        or raw.get("message")
        or raw.get("subject")
        or raw.get("title")
        or ""
    ).strip()

    if not message:
        return None

    # ----- files -----
    files: List[Dict] = []

    # Layout A: head_base_files list (process_commit_pairs.py output)
    hbf = raw.get("head_base_files")
    if hbf:
        for entry in hbf[:MAX_FILES_PER_COMMIT]:
            path = entry.get("head_path") or entry.get("base_path") or ""
            old  = entry.get("base_content") or ""
            new  = entry.get("head_content") or ""
            lang = _normalise_lang(
                entry.get("head_language") or entry.get("base_language") or ""
            ) or _lang_from_path(path)
            files.append({
                "path":    path,
                "old":     old,
                "new":     new,
                "lang":    lang,
                "blob_id": entry.get("head_blob_id") or entry.get("base_blob_id") or "",
            })

    # Layout B: flat old_contents / new_contents (CommitPack-style)
    elif raw.get("old_contents") is not None or raw.get("new_contents") is not None:
        path = (
            raw.get("new_file")
            or raw.get("old_file")
            or raw.get("filename")
            or ""
        )
        lang = _normalise_lang(
            raw.get("lang") or raw.get("language") or ""
        ) or _lang_from_path(path)
        files.append({
            "path":    path,
            "old":     raw.get("old_contents") or "",
            "new":     raw.get("new_contents") or "",
            "lang":    lang,
            "blob_id": "",
        })

    # Layout C: diffs list (some GHArchive-derived formats)
    elif raw.get("diffs"):
        for d in raw["diffs"][:MAX_FILES_PER_COMMIT]:
            path = d.get("filename") or d.get("path") or ""
            files.append({
                "path":    path,
                "old":     d.get("old") or d.get("before") or "",
                "new":     d.get("new") or d.get("after") or "",
                "lang":    _lang_from_path(path),
                "blob_id": d.get("sha") or "",
            })

    if not files:
        return None

    return {
        "directory_id": raw.get("directory_id") or raw.get("dir_id"),
        "message":      message,
        "files":        files,
    }


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class StackV2CommitAdapter:
    """
    Converts Stack v2 PR/commit records to DES-LOC pretraining format.

    Filtering (task spec §2):
      - max_chars: 100K character hard cap per sample
      - drop merge commits
      - drop samples with < min_changed_lines meaningful diff lines
      - directory_id-based deduplication

    PII: email, IPv4, long hex strings, AWS key patterns → <REDACTED>

    Args:
        max_chars:         character cap per formatted sample (default 100K)
        min_changed_lines: minimum changed lines to keep (default 10)
        dedup:             enable directory_id dedup (default True)
    """

    def __init__(
        self,
        max_chars:          int  = MAX_CHARS,
        min_changed_lines:  int  = MIN_CHANGED_LINES,
        dedup:              bool = True,
    ) -> None:
        self.max_chars         = max_chars
        self.min_changed_lines = min_changed_lines
        self._deduplicator     = DirectoryIdDeduplicator() if dedup else None

        # Counters for diagnostics
        self.stats: Dict[str, int] = {
            "seen":           0,
            "kept":           0,
            "drop_merge":     0,
            "drop_lines":     0,
            "drop_chars":     0,
            "drop_dedup":     0,
            "drop_bad_record": 0,
        }

    # ------------------------------------------------------------------
    # Public streaming APIs
    # ------------------------------------------------------------------

    def stream_records(self, records: Iterable[Dict]) -> Iterator[Dict]:
        """
        Process an iterable of raw records (dicts) and yield formatted samples.

        Each yielded sample dict contains:
            "text"    : formatted DES-LOC string
            "lang"    : primary language (from first changed file)
            "message" : commit/PR message
            "_source" : "stackv2_commits"
        """
        for raw in records:
            self.stats["seen"] += 1
            sample = self._process_one(raw)
            if sample is not None:
                self.stats["kept"] += 1
                yield sample

    def stream_parquet(
        self,
        pattern: str,
        max_samples: Optional[int] = None,
    ) -> Iterator[Dict]:
        """
        Stream from local parquet files matching a glob pattern.

        Requires: pyarrow
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required: pip install pyarrow")

        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No parquet files matched: {pattern}")

        count = 0
        for fpath in files:
            table = pq.read_table(fpath)
            for batch in table.to_batches():
                for row in batch.to_pylist():
                    if max_samples and count >= max_samples:
                        return
                    sample = next(iter(self.stream_records([row])), None)
                    if sample is not None:
                        yield sample
                        count += 1

    def stream_hf(
        self,
        dataset_id: str      = "bigcode/the-stack-v2",
        split:      str      = "train",
        max_samples: Optional[int] = None,
        hf_token:   Optional[str] = None,
    ) -> Iterator[Dict]:
        """
        Stream directly from the HuggingFace Hub (requires accepted agreement).

        Requires: datasets
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets is required: pip install datasets")

        kwargs: Dict = {"streaming": True, "split": split, "trust_remote_code": True}
        if hf_token:
            kwargs["token"] = hf_token

        ds = load_dataset(dataset_id, **kwargs)
        count = 0
        for raw in ds:
            if max_samples and count >= max_samples:
                break
            sample = next(iter(self.stream_records([raw])), None)
            if sample is not None:
                yield sample
                count += 1

    # ------------------------------------------------------------------
    # Dummy data generator (smoke tests / offline dev)
    # ------------------------------------------------------------------

    @staticmethod
    def dummy_records(n: int = 10) -> List[Dict]:
        """Generate n synthetic Stack v2-style commit records for testing."""
        records = []
        for i in range(n):
            records.append({
                "directory_id": f"deadbeef{i:08x}",
                "commit_message": f"fix: handle edge case #{i} in parser",
                "head_base_files": [
                    {
                        "head_path": f"src/module_{i}.py",
                        "base_path": f"src/module_{i}.py",
                        "head_content": (
                            f"def func_{i}(x):\n"
                            f"    # fixed\n"
                            + "    y = x + 1\n" * 5
                            + f"    return y\n"
                        ),
                        "base_content": (
                            f"def func_{i}(x):\n"
                            + "    y = x\n" * 5
                            + f"    return y\n"
                        ),
                        "head_language": "Python",
                        "base_language": "Python",
                        "head_blob_id": f"blob{i}head",
                        "base_blob_id": f"blob{i}base",
                    }
                ],
            })
        # One merge commit (should be dropped)
        records.append({
            "directory_id": "merge_commit_id",
            "commit_message": "Merge pull request #42 from user/branch",
            "head_base_files": [
                {
                    "head_path": "file.py",
                    "base_path": "file.py",
                    "head_content": "x = 1",
                    "base_content": "x = 0",
                    "head_language": "Python",
                    "base_language": "Python",
                    "head_blob_id": "a",
                    "base_blob_id": "b",
                }
            ],
        })
        return records

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_one(self, raw: Dict) -> Optional[Dict]:
        norm = _normalise_record(raw)
        if norm is None:
            self.stats["drop_bad_record"] += 1
            return None

        message = norm["message"]
        files   = norm["files"]

        # Filter: merge commits
        if _is_merge_commit(message):
            self.stats["drop_merge"] += 1
            return None

        # Filter: meaningful diff size
        if _total_changed_lines(files) < self.min_changed_lines:
            self.stats["drop_lines"] += 1
            return None

        # Dedup
        if self._deduplicator is not None and not self._deduplicator.is_new(norm):
            self.stats["drop_dedup"] += 1
            return None

        # Render
        text = _render_sample(files, message)

        # Filter: character cap
        if len(text) > self.max_chars:
            self.stats["drop_chars"] += 1
            return None

        primary_lang = files[0]["lang"] if files else "unknown"
        return {
            "text":    text,
            "lang":    primary_lang,
            "message": message,
            "_source": "stackv2_commits",
        }

    def print_stats(self) -> None:
        s = self.stats
        kept  = s["kept"]
        seen  = s["seen"]
        ratio = kept / seen if seen else 0.0
        print(f"[StackV2CommitAdapter] seen={seen}  kept={kept}  ({ratio:.1%})")
        print(f"  drop_merge={s['drop_merge']}  drop_lines={s['drop_lines']}  "
              f"drop_chars={s['drop_chars']}  drop_dedup={s['drop_dedup']}  "
              f"drop_bad_record={s['drop_bad_record']}")
        if self._deduplicator:
            print(f"  dedup_set_size={self._deduplicator.seen_count}")


# ---------------------------------------------------------------------------
# Smoke-test CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="StackV2CommitAdapter smoke test")
    parser.add_argument("--samples",  type=int,  default=5,     help="Number of samples")
    parser.add_argument("--source",   type=str,  default="dummy",
                        choices=["dummy", "hf", "parquet"])
    parser.add_argument("--parquet",  type=str,  default=None,  help="Glob for parquet files")
    parser.add_argument("--hf-token", type=str,  default=None)
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args()

    adapter = StackV2CommitAdapter(dedup=not args.no_dedup)

    if args.source == "dummy":
        print(f"=== Dummy smoke test ({args.samples} samples) ===\n")
        records = StackV2CommitAdapter.dummy_records(args.samples)
        for i, sample in enumerate(adapter.stream_records(records)):
            print(f"--- Sample {i} | lang={sample['lang']} ---")
            print(sample["text"][:600])
            print()

    elif args.source == "parquet":
        if not args.parquet:
            print("--parquet glob required for source=parquet", file=sys.stderr)
            sys.exit(1)
        for i, sample in enumerate(adapter.stream_parquet(args.parquet, args.samples)):
            print(f"--- Sample {i} | lang={sample['lang']} ---")
            print(sample["text"][:600])
            print()

    elif args.source == "hf":
        for i, sample in enumerate(
            adapter.stream_hf(max_samples=args.samples, hf_token=args.hf_token)
        ):
            print(f"--- Sample {i} | lang={sample['lang']} ---")
            print(sample["text"][:600])
            print()

    adapter.print_stats()
