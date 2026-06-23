# Neuron_SP — 7B commit-model pretrain on ags1 heterogeneous cluster
SHELL := /bin/bash
.DEFAULT_GOAL := help

# ── Environment ──────────────────────────────────────────────
.PHONY: setup
setup: ## One-time conda + pip setup on ags1
	bash setup_ags1.sh

# ── Data ─────────────────────────────────────────────────────
.PHONY: data data-small data-count
data: ## Download and prepare full CommitPack (4 TB, takes hours)
	bash datasets/bigcode/pull_all_datasets.sh
	python data/prepare_commits.py --tokenizer-name bigcode/starcoder

data-small: ## Prepare a small sample (10K commits) for testing
	python data/prepare_commits.py --num-samples 10000 --tokenizer-name bigcode/starcoder

data-count: ## Count tokens in prepared .bin files
	python tools/count_tokens.py --path data/

# ── Training ─────────────────────────────────────────────────
.PHONY: train-70m train-1b train-7b train-7b-resume
train-70m: ## Quick single-GPU 70M test (synthetic data, 100 steps)
	python run_pretrain.py --model-size 70m --steps 100 --log-every 10

train-1b: ## Single-GPU 1B test (synthetic, 50 steps)
	python run_pretrain.py --model-size 1b --steps 50 --gradient-checkpointing --log-every 10

train-7b: ## Full 5-GPU 7B FSDP training
	bash launch_7b.sh

train-7b-resume: ## Resume 7B from latest checkpoint
	@LATEST=$$(ls -t checkpoints/step_*.pt 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then echo "No checkpoint found"; exit 1; fi; \
	echo "Resuming from $$LATEST"; \
	bash launch_7b.sh --resume $$LATEST

# ── Scaling law ──────────────────────────────────────────────
.PHONY: scaling-law
scaling-law: ## Run scaling law experiments (70M → 1B)
	bash scripts/launch_scaling_law.sh

# ── Evaluation ───────────────────────────────────────────────
.PHONY: eval eval-sample
eval: ## Run commit-prediction eval on latest checkpoint
	@LATEST=$$(ls -t checkpoints/step_*.pt 2>/dev/null | head -1); \
	python eval/run_eval.py --model-path $$LATEST

eval-sample: ## Generate sample commit completions
	@LATEST=$$(ls -t checkpoints/step_*.pt 2>/dev/null | head -1); \
	python tools/generate_sample.py --checkpoint $$LATEST --prompt "fix: correct off-by-one error in"

# ── Benchmark ────────────────────────────────────────────────
.PHONY: bench bench-all
bench: ## Benchmark MFU on GPU 2 (H100)
	python benchmark_mfu.py --gpu-id 2

bench-all: ## Profile memory for 7B model
	python tools/profile_memory.py

# ── Paper ────────────────────────────────────────────────────
.PHONY: paper
paper: ## Compile NeurIPS 2026 LaTeX paper
	cd FAUST_nips2026 && $(MAKE)

# ── Convert ──────────────────────────────────────────────────
.PHONY: convert-hf
convert-hf: ## Convert checkpoint to HuggingFace format
	@LATEST=$$(ls -t checkpoints/step_*.pt 2>/dev/null | head -1); \
	python tools/convert_to_hf.py --checkpoint-path $$LATEST --output-dir models/neuron-sp-7b-hf

# ── Housekeeping ─────────────────────────────────────────────
.PHONY: test clean help
test: ## Run pytest suite
	python -m pytest tests/ -v --timeout=30

clean: ## Remove checkpoints, logs, caches
	rm -rf checkpoints/ logs/*.log __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
