# Bayesian structure learning

This directory contains the experiments for Bayesian structure learning (§5.2): learning directed acyclic graph (DAG) structure from data with DAG-GFlowNet, including priors informed by LLM conditionals.

## Installation

These experiments require additional dependencies beyond the base project install. From the repository root, run one of the following commands depending on your system's CUDA version:

```bash
# GPU with CUDA 12
uv sync --extra structure-learning-cuda12

# GPU with CUDA 13
uv sync --extra structure-learning-cuda13

# CPU (JAX without GPU)
uv sync --extra structure-learning
```


## Usage

Enter this directory before running the commands below:

```bash
cd structure_learning
```

### Step 0: Download datasets

The datasets and meta-data files used in the paper are provided in the `structure_learning/datasets`.
To test with other datasets from [pgmpy](https://github.com/pgmpy/pgmpy), use `get_pgmpy_dataset.py` to download and save the datasets. Note that you should modify `meta_data.json` for each dataset appropriately to give LLMs enough information about the dataset.


### Step 1: Generate synthetic data using LLMs


... an OpenAI-compatible chat/completions API is required ...


### Step 2: Train DAG-GFlowNet with LLM data as priors

