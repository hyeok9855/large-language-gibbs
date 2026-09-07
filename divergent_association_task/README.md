# Divergent Association Task (DAT)

Name 10 nouns as unrelated to each other as possible (Olson et al., 2021). Score = mean pairwise GloVe cosine distance x 100 over the first 7 valid words; answers with fewer than 7 valid words are unscorable. Human mean ~78.

Prompt from Bellemare-Pepin et al. (2024, arXiv:2405.13012), verbatim (`utils.dat_prompt`), temperature 1, 500 answers per model, scored post hoc with Olson's scorer (`dat.py`, vendored). No system prompt; both methods write the answer through the same assistant prefill and single-word grammar, so they differ only in the sampling scheme.

- **direct**: the answer built one entry at a time by continuing the partial list in an assistant prefill (ancestral order), redrawn whole until scorable.
- **gibbs**: initialised from a direct answer; each step shows the other 9 words as a list in an assistant prefill and draws the next entry (constrained to a single word), sweeping the 10 slots in random order.

- `--list_format` picks how the list is written: `numbered` (`1. w\n2.`), `bullet` (`- w\n-`), `comma` (`w1, w2,`) or `bracket` (`[w1, w2,`); results are tagged `direct_<format>` and `gibbs_<format>`.
- `--answer_prefix` (off by default) opens the prefill with `Here are ten words:` on its own line. The inline formats have no anchor at the first entry without it, so instruct models start a prose reply that the word grammar slices into entries. Tagged `_prefix`.

```
bash divergent_association_task/download_assets.sh                          # GloVe + dictionary (~5GB)
python divergent_association_task/run.py --model_name <hf id> --port 8000   # direct + gibbs, all four formats
python divergent_association_task/evaluate.py --plot                        # results/summary.json, dat_summary.png
python divergent_association_task/make_tables.py                            # LaTeX table
```
