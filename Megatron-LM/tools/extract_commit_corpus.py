#!/usr/bin/env python3
"""
Extract commit corpus from multiple GitHub repositories.

Targets high-quality ML/systems repos following Seed-Coder criteria:
- Skip merge commits, reverts, CI-only, docs-only
- Filter by diff size (not too small, not too huge)
- Dedup by commit message + diff hash
- Output formatted for CommitTokenizer

Usage:
    python extract_commit_corpus.py \
        --repos NVIDIA/Megatron-LM microsoft/DeepSpeed huggingface/transformers \
        --output-dir /data/commit_corpus \
        --max-commits-per-repo 50000
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from megatron.training.tokenizer.commit_tokenizer import (
    format_commit_for_pretraining,
    parse_git_diff,
    detect_language,
)


# High-quality ML/systems repos
DEFAULT_REPOS = [
    "NVIDIA/Megatron-LM",
    "microsoft/DeepSpeed",
    "huggingface/transformers",
    "pytorch/pytorch",
    "facebookresearch/fairseq",
    "EleutherAI/gpt-neox",
    "google-research/t5x",
    "salesforce/CodeGen",
    "bigscience-workshop/Megatron-DeepSpeed",
    "NVIDIA/NeMo",
]

# Skip patterns for low-quality commits
SKIP_PATTERNS = re.compile(
    r'^(Merge |Revert |ci[:\(]|chore[:\(]|docs[:\(]|'
    r'style[:\(]|build[:\(]|'
    r'fix typo|Update README|bump version|'
    r'auto-generated|Automatic |Bot: )',
    re.IGNORECASE,
)


def clone_or_pull(repo: str, clone_dir: str) -> str:
    """Clone repo if not exists, else pull latest."""
    repo_name = repo.split('/')[-1]
    repo_path = os.path.join(clone_dir, repo_name)

    if os.path.exists(repo_path):
        print(f"  Pulling {repo}...")
        subprocess.run(
            ["git", "pull", "--quiet"],
            cwd=repo_path,
            capture_output=True,
        )
    else:
        print(f"  Cloning {repo}...")
        subprocess.run(
            ["git", "clone", "--quiet", f"https://github.com/{repo}.git", repo_path],
            capture_output=True,
        )

    return repo_path


def get_commit_list(repo_path: str, max_commits: int) -> list:
    """Get list of non-merge commit hashes, oldest first."""
    result = subprocess.run(
        ["git", "log", "--format=%H|%s|%an", "--reverse", "--no-merges",
         f"-{max_commits}"],
        capture_output=True, text=True, cwd=repo_path,
    )
    commits = []
    for line in result.stdout.strip().split('\n'):
        if '|' not in line:
            continue
        parts = line.split('|', 2)
        if len(parts) == 3:
            commits.append({
                'hash': parts[0],
                'subject': parts[1],
                'author': parts[2],
            })
    return commits


def should_skip(subject: str) -> bool:
    """Check if commit should be skipped based on message."""
    return bool(SKIP_PATTERNS.match(subject))


def get_diff(repo_path: str, commit_hash: str) -> str:
    """Get unified diff for a commit."""
    result = subprocess.run(
        ["git", "diff", f"{commit_hash}^..{commit_hash}",
         "--unified=3", "--no-color", "--no-ext-diff"],
        capture_output=True, text=True, cwd=repo_path,
        timeout=30,
    )
    return result.stdout


def diff_hash(diff_text: str) -> str:
    """Compute hash of diff for deduplication."""
    # Normalize: strip line numbers and whitespace
    normalized = re.sub(r'@@ .+ @@', '@@', diff_text)
    normalized = re.sub(r'\s+', ' ', normalized)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def extract_repo(
    repo: str,
    clone_dir: str,
    output_dir: str,
    max_commits: int,
    min_diff_lines: int,
    max_diff_lines: int,
    seen_hashes: set,
) -> dict:
    """
    Extract and format commits from a single repo.

    Returns statistics dict.
    """
    repo_name = repo.split('/')[-1]
    repo_path = clone_or_pull(repo, clone_dir)

    commits = get_commit_list(repo_path, max_commits)
    print(f"  {repo}: {len(commits)} non-merge commits")

    output_file = os.path.join(output_dir, f"{repo_name}_commits.txt")
    stats = {
        'repo': repo,
        'total': len(commits),
        'skipped_pattern': 0,
        'skipped_size': 0,
        'skipped_dedup': 0,
        'skipped_error': 0,
        'extracted': 0,
        'tokens_est': 0,
        'languages': Counter(),
    }

    with open(output_file, 'w', encoding='utf-8') as out:
        for i, commit in enumerate(commits):
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{len(commits)}...")

            # Skip by pattern
            if should_skip(commit['subject']):
                stats['skipped_pattern'] += 1
                continue

            # Get diff
            try:
                diff_text = get_diff(repo_path, commit['hash'])
            except Exception:
                stats['skipped_error'] += 1
                continue

            # Skip by size
            diff_lines = len(diff_text.split('\n'))
            if diff_lines < min_diff_lines or diff_lines > max_diff_lines:
                stats['skipped_size'] += 1
                continue

            # Dedup
            dh = diff_hash(diff_text)
            if dh in seen_hashes:
                stats['skipped_dedup'] += 1
                continue
            seen_hashes.add(dh)

            # Track languages
            files = parse_git_diff(diff_text)
            for f in files:
                lang = f['language'].replace('<lang:', '').replace('>', '')
                stats['languages'][lang] += 1

            # Get full commit message
            msg_result = subprocess.run(
                ["git", "log", "--format=%s%n%b", "-1", commit['hash']],
                capture_output=True, text=True, cwd=repo_path,
            )
            full_message = msg_result.stdout.strip()

            # Format
            formatted = format_commit_for_pretraining(
                commit_hash=commit['hash'],
                commit_message=full_message,
                author=commit['author'],
                diff_text=diff_text,
                max_diff_lines=512,
            )

            out.write(formatted + '\n\n')
            stats['extracted'] += 1
            stats['tokens_est'] += len(formatted.split())  # rough word count

    print(f"  → {stats['extracted']} commits extracted to {output_file}")
    return stats


def main():
    parser = argparse.ArgumentParser(description='Extract commit corpus')
    parser.add_argument(
        '--repos', nargs='+', default=DEFAULT_REPOS,
        help='GitHub repos to extract from',
    )
    parser.add_argument(
        '--output-dir', type=str, default='/data/commit_corpus',
        help='Output directory for formatted commits',
    )
    parser.add_argument(
        '--clone-dir', type=str, default='/tmp/repos',
        help='Directory to clone repos into',
    )
    parser.add_argument(
        '--max-commits-per-repo', type=int, default=50000,
        help='Max commits to extract per repo',
    )
    parser.add_argument(
        '--min-diff-lines', type=int, default=5,
        help='Minimum diff lines to include',
    )
    parser.add_argument(
        '--max-diff-lines', type=int, default=2000,
        help='Maximum diff lines per commit',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.clone_dir, exist_ok=True)

    seen_hashes = set()
    all_stats = []

    print(f"=== Extracting commits from {len(args.repos)} repos ===\n")

    for repo in args.repos:
        try:
            stats = extract_repo(
                repo=repo,
                clone_dir=args.clone_dir,
                output_dir=args.output_dir,
                max_commits=args.max_commits_per_repo,
                min_diff_lines=args.min_diff_lines,
                max_diff_lines=args.max_diff_lines,
                seen_hashes=seen_hashes,
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"  ERROR on {repo}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"CORPUS SUMMARY")
    print(f"{'='*60}")

    total_extracted = 0
    total_tokens = 0
    all_languages = Counter()

    for s in all_stats:
        print(f"\n{s['repo']}:")
        print(f"  Total commits: {s['total']}")
        print(f"  Extracted: {s['extracted']}")
        print(f"  Skipped (pattern): {s['skipped_pattern']}")
        print(f"  Skipped (size): {s['skipped_size']}")
        print(f"  Skipped (dedup): {s['skipped_dedup']}")
        print(f"  Est. tokens: {s['tokens_est']:,}")
        total_extracted += s['extracted']
        total_tokens += s['tokens_est']
        all_languages.update(s['languages'])

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_extracted} commits, ~{total_tokens:,} tokens")
    print(f"\nLanguage distribution:")
    for lang, count in all_languages.most_common(15):
        print(f"  {lang:15s}: {count:6d} files")

    # Write stats
    import json
    stats_path = os.path.join(args.output_dir, "corpus_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({
            'total_commits': total_extracted,
            'total_tokens_est': total_tokens,
            'repos': [s['repo'] for s in all_stats],
            'languages': dict(all_languages.most_common()),
            'per_repo': [{k: v for k, v in s.items() if k != 'languages'}
                         for s in all_stats],
        }, f, indent=2)
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()
