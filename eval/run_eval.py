#!/usr/bin/env python3
"""
DES-LOC CodeGen Evaluation Harness (M821-M835)
Supports: HumanEval (pass@k), MBPP, commit message prediction
"""

import argparse
import json
import os
import sys
import time
import random
import re
import subprocess
import tempfile
from pathlib import Path
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
        "diff": "diff --git a/src/utils.py b/src/utils.py\n--- a/src/utils.py\n+++ b/src/utils.py\n@@ -10,6 +10,10 @@ def parse_args():\n+    parser.add_argument('--lr', type=float, default=1e-4)\n+    parser.add_argument('--epochs', type=int, default=10)\n",
        "message": "add lr and epochs arguments to arg parser",
    },
    {
        "diff": "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1,3 +1,5 @@\n+# DES-LOC\n+Distributed Efficient Supervised Learning with Optimization Clusters\n",
        "message": "add project title and description to README",
    },
    {
        "diff": "diff --git a/train.py b/train.py\n--- a/train.py\n+++ b/train.py\n@@ -45,3 +45,7 @@\n+    if step % eval_interval == 0:\n+        loss = evaluate(model, val_loader)\n+        logger.info(f'step {step} val_loss={loss:.4f}')\n",
        "message": "add periodic validation loss logging during training",
    },
]


def load_commitpack_sample(n: int = 1000) -> list:
    """Attempt to load CommitPackFT; fall back to stub on failure."""
    try:
        import datasets as hf_datasets
        ds = hf_datasets.load_dataset(
            "bigcode/commitpackft", split="test", streaming=True
        )
        samples = []
        for item in ds:
            samples.append({"diff": item.get("diff", ""), "message": item.get("message", "")})
            if len(samples) >= n:
                break
        if samples:
            return samples
    except Exception as e:
        print(f"  [WARN] CommitPackFT load failed ({e}), using stubs")
    return COMMITPACK_STUB


def run_commit_prediction(model: Optional["ModelWrapper"], n_samples: int = 1000,
                          use_stubs: bool = False) -> dict:
    print(f"\n=== Commit Message Prediction (n={n_samples}) ===")

    if use_stubs or model is None:
        dataset = COMMITPACK_STUB
    else:
        dataset = load_commitpack_sample(n_samples)

    bleu_scores = []
    rouge_scores = []
    exact_matches = 0

    for item in dataset:
        diff = item["diff"]
        reference = item["message"]

        if model is not None:
            prompt = f"<diff>\n{diff}\n</diff>\nCommit message:"
            completions = model.generate(prompt, max_new_tokens=64, temperature=0.1)
            hypothesis = completions[0].strip().split("\n")[0]
        else:
            # Oracle: return reference directly (upper bound simulation)
            hypothesis = reference

        bleu_scores.append(bleu_score(reference, hypothesis))
        rouge_scores.append(rouge_l(reference, hypothesis))
        if exact_match(reference, hypothesis):
            exact_matches += 1

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    em_rate = exact_matches / len(dataset) if dataset else 0.0

    print(f"  BLEU:        {avg_bleu:.4f}")
    print(f"  ROUGE-L:     {avg_rouge:.4f}")
    print(f"  Exact match: {em_rate:.4f}  ({exact_matches}/{len(dataset)})")

    return {
        "bleu": avg_bleu,
        "rouge_l": avg_rouge,
        "exact_match_rate": em_rate,
        "n_samples": len(dataset),
    }


# ---------------------------------------------------------------------------
# Periodic eval hook (called from training loop)
# ---------------------------------------------------------------------------

def run_periodic_eval(model_path: str, step: int, output_dir: str,
                      benchmarks: list = None, device: str = "auto") -> dict:
    """
    Called every N training steps from the DES-LOC training loop.
    Only runs on the H100 (highest-capability) node; other GPUs continue
    data preparation for the next epoch.

    Usage in training loop:
        if step % EVAL_INTERVAL == 0 and is_h100():
            results = run_periodic_eval(model_path, step, output_dir)
    """
    if benchmarks is None:
        benchmarks = ["humaneval", "mbpp", "commit"]

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
        all_results["humaneval"] = run_humaneval(model, samples_per_task=10,
                                                  use_stubs=(model is None))
    if "mbpp" in benchmarks:
        all_results["mbpp"] = run_mbpp(model, use_stubs=(model is None))
    if "commit" in benchmarks:
        all_results["commit"] = run_commit_prediction(model, n_samples=1000,
                                                       use_stubs=(model is None))

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
    parser.add_argument("--output-dir", default="desloc_results/eval_runs")
    parser.add_argument("--step", type=int, default=0, help="Training step (for logging)")
    parser.add_argument("--samples-per-task", type=int, default=10,
                        help="Number of samples per HumanEval task (for pass@k)")
    parser.add_argument("--commit-samples", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with stub data only (no model needed)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
        all_results["humaneval"] = run_humaneval(
            model, samples_per_task=args.samples_per_task,
            use_stubs=(model is None)
        )

    if "mbpp" in args.benchmarks:
        all_results["mbpp"] = run_mbpp(model, use_stubs=(model is None))

    if "commit" in args.benchmarks:
        all_results["commit"] = run_commit_prediction(
            model, n_samples=args.commit_samples,
            use_stubs=(model is None)
        )

    out_file = os.path.join(args.output_dir, f"eval_step_{args.step:07d}.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {out_file}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
