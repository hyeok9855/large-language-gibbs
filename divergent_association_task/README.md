# Divergent Association Task (DAT)

Name 10 nouns as unrelated to each other as possible (Olson et al., 2021). Score =
mean pairwise GloVe cosine distance x 100 over the first 7 valid words; answers
with fewer than 7 valid words are unscorable. Human mean ~78.

Protocol follows Bellemare-Pepin et al. (2024, arXiv:2405.13012): their prompt
verbatim (`utils.dat_prompt`), temperature 1, 500 answers per model, free-form
replies parsed into words and scored post hoc with Olson's scorer (`dat.py`,
vendored). No system prompt, no output constraints, no rejection sampling.

- **direct**: one free-form reply per answer (the paper's protocol). Base models
  continue the prompt after `Response:` and a `1.` list marker (completions API).
- **gibbs**: initialised from a direct answer; each step shows the other 9 words
  as a numbered list in an assistant prefill and draws entry 10 (constrained to a
  single word), sweeping the 10 slots in random order. Defaults: 25 chains,
  burn-in 50 steps, thinning 20 (2 sweeps), 20 retained states per chain.

```
bash divergent_association_task/download_assets.sh                  # GloVe + dictionary (~5GB)
sbatch divergent_association_task/sbatch.sh <model> [temp] [n] [seed]  # vLLM + run.py + evaluate.py
python divergent_association_task/evaluate.py --plot                # results/summary.json, dat_summary.png
python divergent_association_task/make_tables.py                    # LaTeX table
```

Earlier ablations (JSON/continuation formats, rejection sampling, list formats)
are archived in `results_ablations_2026-08/` and `REPORT_ablations_2026-08.md`.
