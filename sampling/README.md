# Sampling from simple distributions

This directory contains the experiments for sampling from simple distributions (§4): a uniform and a Gaussian distribution.

## Basic setup

Enter this directory before running the commands below:
```bash
cd sampling
```

The experiments query an OpenAI-compatible chat/completions API.

- Local server: pass `--port`; the script uses `http://localhost:<port>/v1`.
- Remote server: pass both `--base_url` and `--api_key`.

Known model names are listed in `utils.py`; if your model is not listed, you should add it to the dictionary.

## Usage

We assume that you have a local server running on port 8000, with the model `meta-llama/Llama-3.1-8B`, and the target is the Gaussian distribution.

Run a single seed using Gibbs sampling:

```bash
uv run python run.py \
  --target gaussian \
  --model_name meta-llama/Llama-3.1-8B \
  --port 8000 \
  --seed 0
  --methods gibbs
```

Existing result files are not overwritten; matching runs are skipped.

`run_multiseed.sh` launches experiments for multiple methods and seeds in parallel: independent and batch sampling, Gibbs with block sizes 1 and 4, and Barker/Gambling variants (only when using an instruct model). For example (5 seeds):
```bash
bash run_multiseed.sh gaussian meta-llama/Llama-3.1-8B 8000 5
```

The script starts many jobs concurrently, so make sure the backing API server can handle the requested load.

After sampling, visualise the results with:

```bash
uv run python make_plot.py
```
