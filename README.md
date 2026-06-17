# Structured Inference with Large Language Gibbs
This repository contains the code for the paper "Structured Inference with Large Language Gibbs" (Under review, ICML 2026 SPIGM Workshop, paper will be available soon).

<p align="center"><img src="assets/fig1.png" width="1000"/></p>

---
> **Abstract:**  The knowledge encoded in large language models (LLMs) can serve as a substrate for structured reasoning over variables describing a complex world, but accessing this knowledge in a probabilistically coherent manner poses a difficult inference problem. We propose Large Language Gibbs, a scheme for structured probabilistic inference that uses conditional distributions of an LLM as transition operators. Rather than sampling structured objects through single-pass autoregressive generation, we iteratively resample individual variables conditioned on others using an LLM's next-token conditionals. This approach avoids order-dependent biases and produces a stationary distribution that reflects a compromise between all local conditionals. We apply this approach to sampling from synthetic distributions, consistent reasoning tasks, and Bayesian structure learning. The results suggest that the use of LLM conditionals in MCMC is a practical alternative to one-pass generation for structured probabilistic inference under a world prior accessible through noisy LLM conditionals.


## Installation

We recommend using [uv](https://docs.astral.sh/uv/) to install dependencies and run the project.

First, run the following command to install the dependencies (this will automatically create `.venv` in the root directory):

```bash
uv sync
```


## Experiment1: Sampling from simple distributions (§4)

Sampling from simple distributions like a uniform and a Gaussian distribution.

### Basic setup

The experiments query an OpenAI-compatible chat/completions API.

- Local server: pass `--port`; the script uses `http://localhost:<port>/v1`.
- Remote server: pass both `--base_url` and `--api_key`.

Known model names are listed in `utils.py`; if your model is not listed, you should add it to the dictionary.

### Usage

We assume that you have a local server running on port 8000, with the model `meta-llama/Llama-3.1-8B`, and the target is the Gaussian distribution.

Run a single seed using Gibbs sampling:

```bash
uv run python sampling/run.py \
  --target gaussian \
  --model_name meta-llama/Llama-3.1-8B \
  --port 8000 \
  --seed 0
  --methods gibbs
```

Existing result files are not overwritten; matching runs are skipped.

`run_multiseed.sh` launches experiments for multiple methods and seeds in parallel: independent and batch sampling, Gibbs with block sizes 1 and 4, and Barker/Gambling variants (only when using an instruct model). For example (5 seeds):
```bash
bash sampling/run_multiseed.sh gaussian meta-llama/Llama-3.1-8B 8000 5
```

The script starts many jobs concurrently, so make sure the backing API server can handle the requested load.

After sampling, visualise the results with:

```bash
uv run python sampling/make_plot.py
```


## Experiment2: Consistent reasoning tasks (§5.1)


## Experiment3: Bayesian structure learning (§5.2)

Learning directed acyclic graph (DAG) structure from data with [DAG-GFlowNet](https://github.com/tristandeleu/jax-dag-gflownet) using LLM priors.


### Basic setup

This experiment requires additional dependencies beyond the base project install. Run one of the following commands depending on your system's CUDA version:

```bash
# GPU with CUDA 12
uv sync --extra structure-learning-cuda12

# GPU with CUDA 13
uv sync --extra structure-learning-cuda13

# CPU (JAX without GPU)
uv sync --extra structure-learning
```


### Usage

#### Step 0: Download datasets

The datasets and meta-data files used in the paper are provided in the `structure_learning/datasets`.
To test with other datasets from [pgmpy](https://github.com/pgmpy/pgmpy), use `structure_learning/get_pgmpy_dataset.py` to download and save the datasets. Note that you should modify `meta_data.json` for each dataset appropriately to give LLMs enough information about the dataset.


#### Step 1: Generate synthetic data using LLMs


... an OpenAI-compatible chat/completions API is required ...


#### Step 2: Train DAG-GFlowNet with LLM data as priors

