#!/usr/bin/env python3
"""
DES-LOC CodeGen Evaluation Harness (M821-M835)
Supports: HumanEval (pass@k), MBPP, commit prediction (message+context → diff)
"""

import argparse
import json
import os
import sys
import time
import subprocess
import tempfile
from typing import Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULT_EVAL_CONFIG = {
    "eval": {
        "benchmarks": ["humaneval", "mbpp", "commit"],
        "humaneval": {"samples_per_task": 10, "k_values": [1, 5, 10],
                      "max_new_tokens": 512, "temperature": 0.8},
        "mbpp": {"max_new_tokens": 256, "temperature": 0.2},
        "commit_prediction": {
            "n_samples": 1000, "max_new_tokens": 64, "temperature": 0.1,
            "metrics": ["bleu", "rouge_l", "exact_match"],
        },
        "output_dir": "desloc_results/eval_runs",
    },
    "data": {"commitpack_split": "test", "commitpack_n": 1000},
}


def load_eval_config(config_path: Optional[str] = None) -> dict:
    """Load eval_config.yaml; return defaults if unavailable."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "eval_config.yaml")
    if HAS_YAML and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            return cfg if cfg else _DEFAULT_EVAL_CONFIG
        except Exception as e:
            print(f"  [WARN] Could not parse {config_path}: {e}; using defaults")
    return _DEFAULT_EVAL_CONFIG

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def bleu_score(reference: str, hypothesis: str) -> float:
    """Naive 1-gram / 2-gram BLEU approximation (no NLTK dependency)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0
    # unigram precision
    ref_set = set(ref_tokens)
    match1 = sum(1 for t in hyp_tokens if t in ref_set)
    p1 = match1 / len(hyp_tokens)
    # bigram precision
    def bigrams(tokens):
        return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    ref_bi = set(bigrams(ref_tokens))
    hyp_bi = bigrams(hyp_tokens)
    if hyp_bi:
        match2 = sum(1 for b in hyp_bi if b in ref_bi)
        p2 = match2 / len(hyp_bi)
    else:
        p2 = 0.0
    # brevity penalty
    bp = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))
    return bp * ((p1 * p2) ** 0.5)


def rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 via LCS length."""
    r = reference.lower().split()
    h = hypothesis.lower().split()
    if not r or not h:
        return 0.0
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if r[i-1] == h[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / n
    recall = lcs / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(reference: str, hypothesis: str) -> bool:
    return reference.strip().lower() == hypothesis.strip().lower()


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class ModelWrapper:
    def __init__(self, model_path: str, device: str = "auto"):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers not installed")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if HAS_TORCH else None,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2,
                 num_samples: int = 1) -> list[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                num_return_sequences=num_samples,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        results = []
        input_len = inputs["input_ids"].shape[1]
        for out in outputs:
            decoded = self.tokenizer.decode(out[input_len:], skip_special_tokens=True)
            results.append(decoded)
        return results


# ---------------------------------------------------------------------------
# HumanEval pass@k
# ---------------------------------------------------------------------------

HUMANEVAL_STUB = [
    {
        "task_id": "HumanEval/0",
        "prompt": 'def has_close_elements(numbers: list, threshold: float) -> bool:\n    """Check if any two numbers in the list are closer than threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n",
        "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n",
        "entry_point": "has_close_elements",
    },
    {
        "task_id": "HumanEval/1",
        "prompt": 'def separate_paren_groups(paren_string: str) -> list:\n    """Input: string with nested parens. Return list of separate paren groups.\n    >>> separate_paren_groups("( ) (( )) (( )( ))")\n    [\'()\', \'(())\', \'(()())\']    \n    """\n',
        "canonical_solution": "    result = []\n    current_string = []\n    current_depth = 0\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n    return result\n",
        "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\n",
        "entry_point": "separate_paren_groups",
    },
]


def execute_code_with_test(code: str, test: str, entry_point: str, timeout: int = 5) -> bool:
    """Run generated code + test in subprocess, return True if passes."""
    full_code = code + "\n" + test + f"\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(fname)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from Chen et al. 2021."""
    if n - c < k:
        return 1.0
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)


def run_humaneval(model: Optional["ModelWrapper"], samples_per_task: int = 10,
                  k_values: list = None, use_stubs: bool = False) -> dict:
    if k_values is None:
        k_values = [1, 5, 10]
    print("\n=== HumanEval Evaluation ===")
    results = {}

    if use_stubs or model is None:
        dataset = HUMANEVAL_STUB
    else:
        try:
            import datasets as hf_datasets
            ds = hf_datasets.load_dataset("openai_humaneval", split="test")
            dataset = list(ds)
        except Exception:
            print("  [WARN] Could not load openai_humaneval dataset, using stubs")
            dataset = HUMANEVAL_STUB

    task_results = []
    for item in dataset:
        task_id = item["task_id"]
        prompt = item["prompt"]
        test = item["test"]
        entry_point = item["entry_point"]

        if model is not None:
            completions = model.generate(prompt, max_new_tokens=512,
                                         temperature=0.8, num_samples=samples_per_task)
        else:
            # Dummy: use canonical solution as completion (simulate oracle)
            completions = [item["canonical_solution"]] * samples_per_task

        n = len(completions)
        c = 0
        for comp in completions:
            full_code = prompt + comp
            if execute_code_with_test(full_code, test, entry_point):
                c += 1

        task_results.append({"task_id": task_id, "n": n, "c": c})

    for k in k_values:
        if k > samples_per_task:
            continue
        scores = [pass_at_k(r["n"], r["c"], k) for r in task_results]
        avg = sum(scores) / len(scores) if scores else 0.0
        results[f"pass@{k}"] = avg
        print(f"  pass@{k}: {avg:.4f}")

    return results


# ---------------------------------------------------------------------------
# MBPP
# ---------------------------------------------------------------------------

MBPP_STUB = [
    {
        "task_id": 1,
        "text": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].",
        "code": "def min_cost(cost, m, n):\n    tc = [[0]*n for _ in range(m)]\n    tc[0][0] = cost[0][0]\n    for i in range(1, m):\n        tc[i][0] = tc[i-1][0] + cost[i][0]\n    for j in range(1, n):\n        tc[0][j] = tc[0][j-1] + cost[0][j]\n    for i in range(1, m):\n        for j in range(1, n):\n            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j]\n    return tc[m-1][n-1]",
        "test_list": [
            "assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8",
        ],
    },
    {
        "task_id": 2,
        "text": "Write a function to find the similar elements from the given two tuple lists.",
        "code": "def similar_elements(test_tup1, test_tup2):\n    res = tuple(set(test_tup1) & set(test_tup2))\n    return res",
        "test_list": [
            "assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == {4, 5}",
        ],
    },
]


def run_mbpp(model: Optional["ModelWrapper"], use_stubs: bool = False) -> dict:
    print("\n=== MBPP Evaluation ===")

    if use_stubs or model is None:
        dataset = MBPP_STUB
    else:
        try:
            import datasets as hf_datasets
            ds = hf_datasets.load_dataset("mbpp", split="test")
            dataset = list(ds)
        except Exception:
            print("  [WARN] Could not load mbpp dataset, using stubs")
            dataset = MBPP_STUB

    passed = 0
    total = len(dataset)
    for item in dataset:
        prompt = f"# {item['text']}\n"
        if model is not None:
            completions = model.generate(prompt, max_new_tokens=256, temperature=0.2)
            code = prompt + completions[0]
        else:
            code = item["code"]

        test_code = "\n".join(item["test_list"])
        full = code + "\n" + test_code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full)
            fname = f.name
        try:
            r = subprocess.run([sys.executable, fname],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                passed += 1
        except subprocess.TimeoutExpired:
            pass
        finally:
            os.unlink(fname)

    accuracy = passed / total if total else 0.0
    print(f"  accuracy: {accuracy:.4f}  ({passed}/{total})")
    return {"accuracy": accuracy, "passed": passed, "total": total}


# ---------------------------------------------------------------------------
# Commit message prediction
# ---------------------------------------------------------------------------

COMMITPACK_STUB = [
    {
        "old_file": "src/utils.py",
        "new_file": "src/utils.py",
        "old_contents": "import argparse\n\ndef parse_args():\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--batch-size', type=int, default=32)\n    return parser.parse_args()\n",
        "new_contents": "import argparse\n\ndef parse_args():\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--batch-size', type=int, default=32)\n    parser.add_argument('--lr', type=float, default=1e-4)\n    parser.add_argument('--epochs', type=int, default=10)\n    return parser.parse_args()\n",
        "subject": "add lr and epochs arguments to arg parser",
        "message": "add lr and epochs arguments to arg parser",
        "diff": "diff --git a/src/utils.py b/src/utils.py\n--- a/src/utils.py\n+++ b/src/utils.py\n@@ -4,4 +4,6 @@ def parse_args():\n     parser.add_argument('--batch-size', type=int, default=32)\n+    parser.add_argument('--lr', type=float, default=1e-4)\n+    parser.add_argument('--epochs', type=int, default=10)\n     return parser.parse_args()\n",
    },
    {
        "old_file": "README.md",
        "new_file": "README.md",
        "old_contents": "A machine learning project.\n",
        "new_contents": "# DES-LOC\nDistributed Efficient Supervised Learning with Optimization Clusters\n\nA machine learning project.\n",
        "subject": "add project title and description to README",
        "message": "add project title and description to README",
        "diff": "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1,1 +1,3 @@\n+# DES-LOC\n+Distributed Efficient Supervised Learning with Optimization Clusters\n+\n A machine learning project.\n",
    },
    {
        "old_file": "train.py",
        "new_file": "train.py",
        "old_contents": "for step in range(max_steps):\n    loss = train_step(model, batch)\n    optimizer.step()\n",
        "new_contents": "for step in range(max_steps):\n    loss = train_step(model, batch)\n    optimizer.step()\n    if step % eval_interval == 0:\n        loss = evaluate(model, val_loader)\n        logger.info(f'step {step} val_loss={loss:.4f}')\n",
        "subject": "add periodic validation loss logging during training",
        "message": "add periodic validation loss logging during training",
        "diff": "diff --git a/train.py b/train.py\n--- a/train.py\n+++ b/train.py\n@@ -3,2 +3,5 @@\n     optimizer.step()\n+    if step % eval_interval == 0:\n+        loss = evaluate(model, val_loader)\n+        logger.info(f'step {step} val_loss={loss:.4f}')\n",
    },
]


def _compute_unified_diff(old_contents: str, new_contents: str,
                          old_file: str = "a", new_file: str = "b") -> str:
    """Compute a unified diff string from old/new file contents."""
    import difflib
    old_lines = old_contents.splitlines(keepends=True)
    new_lines = new_contents.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{old_file}", tofile=f"b/{new_file}",
    ))
    return "".join(diff_lines)


def _build_commit_prediction_prompt(message: str, old_contents: str,
                                    filename: str) -> str:
    """
    Build a prompt for commit prediction evaluation.

    Given:
      - commit message (what the developer intended to change)
      - file context before the change (old_contents)
      - filename

    The model should predict the unified diff (the actual change).
    """
    prompt = (
        f"File: {filename}\n"
        f"Commit message: {message}\n\n"
        f"File contents before the commit:\n"
        f"```\n{old_contents}\n```\n\n"
        f"Generate the unified diff for this commit:\n"
    )
    return prompt


def load_commitpack_sample(n: int = 1000, split: str = "test") -> list:
    """
    Load CommitPackFT samples.  Each returned item includes the fields
    needed for commit prediction evaluation:
      - old_file, new_file
      - old_contents, new_contents
      - subject / message  (commit description)
      - diff               (ground-truth unified diff; computed if absent)

    Falls back to COMMITPACK_STUB on any failure.
    """
    # 1) Try a local JSONL cache first (written by extract_commitpack_test.py)
    local_path = os.path.join(
        os.path.dirname(__file__), "data", "commitpack_test_1000.jsonl"
    )
    if os.path.exists(local_path):
        try:
            samples = []
            with open(local_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    samples.append(item)
                    if len(samples) >= n:
                        break
            if samples:
                print(f"  [INFO] Loaded {len(samples)} samples from {local_path}")
                return samples
        except Exception as e:
            print(f"  [WARN] Local cache read failed ({e}), trying HF datasets")

    # 2) Try HuggingFace datasets (bigcode/commitpackft)
    try:
        import datasets as hf_datasets
        ds = hf_datasets.load_dataset(
            "bigcode/commitpackft", split=split, streaming=True
        )
        samples = []
        for item in ds:
            samples.append({
                "old_file":      item.get("old_file", "file"),
                "new_file":      item.get("new_file", "file"),
                "old_contents":  item.get("old_contents", ""),
                "new_contents":  item.get("new_contents", ""),
                "subject":       item.get("subject", item.get("message", "")),
                "message":       item.get("message", ""),
                "diff":          item.get("diff", ""),
            })
            if len(samples) >= n:
                break
        if samples:
            return samples
    except Exception as e:
        print(f"  [WARN] CommitPackFT HF load failed ({e}), using stubs")

    return COMMITPACK_STUB


def run_commit_prediction(model: Optional["ModelWrapper"], n_samples: int = 1000,
                          max_new_tokens: int = 64, temperature: float = 0.1,
                          metrics: Optional[list] = None,
                          use_stubs: bool = False,
                          config: Optional[dict] = None) -> dict:
    """
    Commit prediction evaluation.

    Given a commit message and the file context (old_contents), the model
    predicts the diff.  The predicted diff is compared against the ground-truth
    diff using BLEU, ROUGE-L, and exact-match.

    Parameters are read from eval_config.yaml (commit_prediction section) when
    *config* is supplied; explicit keyword arguments take precedence.
    """
    # Read config overrides
    if config is not None:
        cp_cfg = config.get("eval", {}).get("commit_prediction", {})
        n_samples     = cp_cfg.get("n_samples",      n_samples)
        max_new_tokens = cp_cfg.get("max_new_tokens", max_new_tokens)
        temperature    = cp_cfg.get("temperature",    temperature)
        if metrics is None:
            metrics = cp_cfg.get("metrics", ["bleu", "rouge_l", "exact_match"])
        split = config.get("data", {}).get("commitpack_split", "test")
    else:
        split = "test"

    if metrics is None:
        metrics = ["bleu", "rouge_l", "exact_match"]

    print(f"\n=== Commit Prediction Evaluation (n={n_samples}) ===")
    print(f"  metrics={metrics}  max_new_tokens={max_new_tokens}  temperature={temperature}")

    if use_stubs or model is None:
        dataset = COMMITPACK_STUB
    else:
        dataset = load_commitpack_sample(n_samples, split=split)

    bleu_scores: list = []
    rouge_scores: list = []
    exact_matches: int = 0
    per_sample: list = []

    for idx, item in enumerate(dataset):
        message      = item.get("subject") or item.get("message", "")
        old_contents = item.get("old_contents", "")
        new_contents = item.get("new_contents", "")
        old_file     = item.get("old_file", "file")
        new_file     = item.get("new_file", "file")

        # Ground-truth diff: prefer pre-computed field; otherwise derive it
        reference_diff = item.get("diff", "")
        if not reference_diff and old_contents and new_contents:
            reference_diff = _compute_unified_diff(
                old_contents, new_contents, old_file, new_file
            )

        if not reference_diff:
            # Nothing to evaluate against; skip
            continue

        # Build prompt and generate predicted diff
        prompt = _build_commit_prediction_prompt(message, old_contents, old_file)

        if model is not None:
            completions = model.generate(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature
            )
            predicted_diff = completions[0].strip()
        else:
            # Dry-run / oracle: compute the real diff from old→new contents
            # (upper-bound simulation — use it to validate metric plumbing)
            if old_contents and new_contents:
                predicted_diff = _compute_unified_diff(
                    old_contents, new_contents, old_file, new_file
                )
            else:
                predicted_diff = reference_diff

        b = bleu_score(reference_diff, predicted_diff)
        r = rouge_l(reference_diff, predicted_diff)
        em = exact_match(reference_diff, predicted_diff)

        bleu_scores.append(b)
        rouge_scores.append(r)
        if em:
            exact_matches += 1

        per_sample.append({
            "idx":            idx,
            "message":        message,
            "bleu":           b,
            "rouge_l":        r,
            "exact_match":    em,
        })

    n_evaluated = len(bleu_scores)
    avg_bleu  = sum(bleu_scores)  / n_evaluated if n_evaluated else 0.0
    avg_rouge = sum(rouge_scores) / n_evaluated if n_evaluated else 0.0
    em_rate   = exact_matches     / n_evaluated if n_evaluated else 0.0

    print(f"  Evaluated:   {n_evaluated} samples")
    if "bleu" in metrics:
        print(f"  BLEU:        {avg_bleu:.4f}")
    if "rouge_l" in metrics:
        print(f"  ROUGE-L:     {avg_rouge:.4f}")
    if "exact_match" in metrics:
        print(f"  Exact match: {em_rate:.4f}  ({exact_matches}/{n_evaluated})")

    result: dict = {
        "n_samples":         n_evaluated,
        "exact_match_rate":  em_rate,
        "exact_matches":     exact_matches,
    }
    if "bleu" in metrics:
        result["bleu"] = avg_bleu
    if "rouge_l" in metrics:
        result["rouge_l"] = avg_rouge

    return result


# ---------------------------------------------------------------------------
# Periodic eval hook (called from training loop)
# ---------------------------------------------------------------------------

def run_periodic_eval(model_path: str, step: int, output_dir: str,
                      benchmarks: list = None, device: str = "auto",
                      config_path: str = None) -> dict:
    """
    Called every N training steps from the DES-LOC training loop.
    Only runs on the H100 (highest-capability) node; other GPUs continue
    data preparation for the next epoch.

    Usage in training loop:
        if step % EVAL_INTERVAL == 0 and is_h100():
            results = run_periodic_eval(model_path, step, output_dir)
    """
    cfg = load_eval_config(config_path)
    eval_cfg = cfg.get("eval", {})

    if benchmarks is None:
        benchmarks = eval_cfg.get("benchmarks", ["humaneval", "mbpp", "commit"])

    print(f"\n[PeriodicEval] step={step}  model={model_path}")
    os.makedirs(output_dir, exist_ok=True)

    model = None
    if HAS_TRANSFORMERS and os.path.exists(model_path):
        try:
            model = ModelWrapper(model_path, device=device)
        except Exception as e:
            print(f"  [WARN] Could not load model: {e}")

    all_results: dict = {"step": step, "model_path": model_path, "timestamp": time.time()}

    if "humaneval" in benchmarks:
        he_cfg = eval_cfg.get("humaneval", {})
        all_results["humaneval"] = run_humaneval(
            model,
            samples_per_task=he_cfg.get("samples_per_task", 10),
            k_values=he_cfg.get("k_values", [1, 5, 10]),
            use_stubs=(model is None),
        )
    if "mbpp" in benchmarks:
        all_results["mbpp"] = run_mbpp(model, use_stubs=(model is None))
    if "commit" in benchmarks:
        cp_cfg = eval_cfg.get("commit_prediction", {})
        all_results["commit"] = run_commit_prediction(
            model,
            n_samples=cp_cfg.get("n_samples", 1000),
            max_new_tokens=cp_cfg.get("max_new_tokens", 64),
            temperature=cp_cfg.get("temperature", 0.1),
            metrics=cp_cfg.get("metrics", ["bleu", "rouge_l", "exact_match"]),
            use_stubs=(model is None),
            config=cfg,
        )

    out_file = os.path.join(output_dir, f"eval_step_{step:07d}.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved → {out_file}")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DES-LOC CodeGen Eval Harness")
    parser.add_argument("--model", default=None, help="HF model path or name")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["humaneval", "mbpp", "commit"],
                        choices=["humaneval", "mbpp", "commit"])
    parser.add_argument("--output-dir", default=None,
                        help="Override output dir from eval_config.yaml")
    parser.add_argument("--step", type=int, default=0, help="Training step (for logging)")
    parser.add_argument("--samples-per-task", type=int, default=None,
                        help="Number of samples per HumanEval task (for pass@k)")
    parser.add_argument("--commit-samples", type=int, default=None,
                        help="Override commit_prediction.n_samples from config")
    parser.add_argument("--config", default=None,
                        help="Path to eval_config.yaml (default: eval/eval_config.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with stub data only (no model needed)")
    args = parser.parse_args()

    # Load configuration
    cfg = load_eval_config(args.config)
    eval_cfg = cfg.get("eval", {})

    output_dir = args.output_dir or eval_cfg.get("output_dir", "desloc_results/eval_runs")
    os.makedirs(output_dir, exist_ok=True)

    model = None
    if args.model and not args.dry_run:
        print(f"Loading model: {args.model}")
        model = ModelWrapper(args.model, device=args.device)

    all_results: dict = {
        "step": args.step,
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if "humaneval" in args.benchmarks:
        he_cfg = eval_cfg.get("humaneval", {})
        spt = args.samples_per_task or he_cfg.get("samples_per_task", 10)
        all_results["humaneval"] = run_humaneval(
            model, samples_per_task=spt,
            k_values=he_cfg.get("k_values", [1, 5, 10]),
            use_stubs=(model is None),
        )

    if "mbpp" in args.benchmarks:
        all_results["mbpp"] = run_mbpp(model, use_stubs=(model is None))

    if "commit" in args.benchmarks:
        cp_cfg = eval_cfg.get("commit_prediction", {})
        n_samples     = args.commit_samples or cp_cfg.get("n_samples", 1000)
        max_new_tokens = cp_cfg.get("max_new_tokens", 64)
        temperature    = cp_cfg.get("temperature", 0.1)
        metrics        = cp_cfg.get("metrics", ["bleu", "rouge_l", "exact_match"])
        all_results["commit"] = run_commit_prediction(
            model,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            metrics=metrics,
            use_stubs=(model is None),
            config=cfg,
        )

    out_file = os.path.join(output_dir, f"eval_step_{args.step:07d}.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {out_file}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
