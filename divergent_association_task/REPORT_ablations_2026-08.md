# Divergent Association Task (DAT): Gibbs vs. Direct

## What the DAT is

The Divergent Association Task (Olson et al., 2021, *PNAS* 118(25)) is a 4-minute
creativity test: name **10 single-word nouns that are as unrelated to each other as
possible** (no proper nouns, no specialised vocabulary). The score is the average
pairwise **cosine distance x 100** between GloVe (840B, 300d) embeddings of the
**first 7 valid words** — valid meaning the word survives a dictionary check after
cleanup. Scores in practice run ~50 (related words: *arm, eyes, feet, ...*) to ~95+
(*hippo, jumper, machinery, prickle, ...*); the human average is ~78 (Olson et al.'s
N=8,572 sample; ~80-81 in the follow-up 100k-participant norm study).

Both papers in `related_works/` administer exactly this task to LLMs with the
one-pass prompt "write 10 nouns as irrelevant from each other as possible":

- **Chen & Ding (arXiv:2310.11158)**: GPT-4 scores 89.1 with greedy decoding
  (above 96% of humans); smaller open models score well below the human mean.
  Sampling (top-p) raises scores of weaker models but adds instability.
- **Bellemare-Pepin et al. (arXiv:2405.13012)**: across ~35 chat models at 500
  runs each, only the strongest (GPT-4 ~83-84, Llama-3.1 ~85) beat the average
  human (~81), and none reach the top-10% human mean (~92). Outputs are highly
  repetitive (e.g. GPT-4-turbo used "ocean" in >90% of its answers).

## Why it is a Gibbs-vs-Direct benchmark

A DAT answer is a structured object with a *mutual* constraint: every word must be
far from every other. One-pass autoregressive generation (**Direct**, the protocol
of both papers) produces words left-to-right — word 1 is generated blind, and only
later words can adapt to earlier ones. **Large Language Gibbs** instead treats the
answer as 10 variables `word_1..word_10` and repeatedly resamples one word from the
LLM conditional p(word_i | all other words), so every word eventually conditions on
the full context. The DAT score is an *external, objective* functional of the joint
sample — exactly the kind of quantity the paper's thesis says should improve when
order-dependent bias is removed.

## Setup

`divergent_association_task/` mirrors the `sampling/` conventions (priorbot
priors + an OpenAI-compatible vLLM server, structured JSON outputs):

| File | Role |
|---|---|
| `run.py` | Driver. `--methods direct gibbs`, `--n_samples`, `--n_chains`, `--seed`, `--temperature`, `--n_words`. Skips existing result files. |
| `templates.py` | DAT instructions (Olson wording as adapted by the two papers) as joint prompt (Direct / Gibbs init), conditional prompt (Gibbs kernel), and partial-JSON continuation prompts. Base and instruct variants. |
| `continuation.py` | Word-continuation LLM client for the `*_continuation` methods: extends `sampling/continuation_llm.py` with pattern-constrained string fields. |
| `utils.py` | Model registry, paths, `word_1..word_N` JSON schema (single lowercase word per property, enforced by vLLM structured outputs). |
| `dat.py` | Olson et al.'s reference scorer, vendored verbatim (MIT) from github.com/jayolson/divergent-association-task. |
| `evaluate.py` | Scores all results (GloVe, first 7 valid of 10), writes `results/summary.json` + `results/dat_summary.png`. CPU only. |
| `download_assets.sh` | Fetches GloVe 840B 300d + the scorer dictionary into `assets/` (~5.3GB). |
| `sbatch.sh` | SLURM job: vLLM server + both methods x seeds + evaluation. |

- **Direct** = `LLMPrior`: one LLM call generates the whole 10-word answer as JSON
  (fixed order `word_1..word_10`), matching the papers' protocol.
- **Gibbs** = `GibbsLLMPrior` (single-site, random-permutation systematic sweep):
  the chain is initialised with one Direct draw, then each step shows the other 9
  words and asks for the missing one. Defaults: thinning = 20 steps (2 sweeps)
  between retained samples, burn-in = 50 steps (5 sweeps). A small
  `RecordedGibbsLLMPrior` subclass also saves the full chain so evaluation can plot
  DAT score against Gibbs step (step 0 = the Direct initialisation).
- **Direct/Gibbs continuation** = the same two schemes, but every word is produced
  by *literal continuation of the partial JSON object* (as in `sampling/`
  `direct_continuation` / `gibbs_continuation`): the model continues
  `{"word_1": "tundra", "word_2":` one field per LLM call (instruct models via
  assistant prefill with `continue_final_message`, base models via the completions
  API), constrained to the shape regex `` ?"<word>"[,}]?[ \n]?``. The optional
  leading space and trailing delimiter are load-bearing tokenizer escapes: BPE
  merges `" \""` on the way in and fuses the closing quote with the next JSON
  delimiter (`",`, `"}`) on the way out. A first run whose grammar admitted only a
  bare closing quote degenerated into letter concatenations pinned at the length
  cap ("booktreasuremenugrav") in most answers - the trailing-delimiter allowance
  is what fixed it.

Scoring is the papers' exact pipeline: the vendored scorer reproduces the reference
values from Olson's repository (cat/dog distance 0.1983, example lists 50 / 78 / 95).
Answers with fewer than 7 valid unique words score `None` and are excluded (counted
in `n_valid`).

### Prompts

Verbatim as sent to the model (instruct models get system prompt "You are a
helpful assistant." and answer via the chat API; `[... word_2..word_9 ...]`
elides eight identical schema properties). The **joint prompt** — used for every
Direct sample and for each Gibbs chain initialisation:

> Write 10 words that are as different from each other as possible, in all
> meanings and uses of the words. Rules: Only single words in English. Only nouns
> (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or
> places). No specialised vocabulary (e.g., no technical terms). Respond with
> JSON that follows this schema: {"type": "object", "properties": {"word_1":
> {"type": "string", "pattern": "^[a-z][a-z-]{0,18}[a-z]$"}, [... word_2..word_9
> ...], "word_10": {"type": "string", "pattern": "^[a-z][a-z-]{0,18}[a-z]$"}},
> "required": ["word_1", ..., "word_10"]}

The **conditional prompt** — the Gibbs kernel, one call per step (this instance is
step 100 of a real recorded chain, seed 42, resampling `word_10`):

> Write 10 words that are as different from each other as possible, in all
> meanings and uses of the words. Rules: Only single words in English. Only nouns
> (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or
> places). No specialised vocabulary (e.g., no technical terms). You have already
> written these words: {"word_1": "tundra", "word_2": "quartz", "word_3":
> "toothbrush", "word_4": "salad", "word_5": "pestle", "word_6": "bread",
> "word_7": "fountain", "word_8": "cloud", "word_9": "gravel"}. Write the
> remaining word so that all 10 words are as different from each other as
> possible. Respond with JSON that follows this schema: {"type": "object",
> "properties": {"word_10": {"type": "string", "pattern":
> "^[a-z][a-z-]{0,18}[a-z]$"}}, "required": ["word_10"]}

The **continuation prompt** (both `*_continuation` methods) asks for the same
answer but hands the model a partial JSON object to continue, one field per call
- instruct models see the instructions plus "Write the answer as a JSON object
with keys (word_1, ..., word_10)." as the user turn and continue an assistant
prefill such as:

> {"word_1": "tundra", "word_2": "quartz", "word_3": "toothbrush", "word_4":

In the Gibbs-continuation conditional the prefill holds the other 9 words and the
resampled key comes last. Generation is constrained to one word-shaped JSON
string (`` ?"<word>"[,}]?[ \n]?``).

Base models get completion-style variants of the same instructions (no system
prompt, `/completions` endpoint): the joint prompt ends with "Here is an answer,
formatted as JSON:" and the conditional shows the partial answer and ends with
"Here is the remaining word of the same answer, formatted as JSON:"; the
continuation prompt ends with the partial JSON itself (see `templates.py`).

Budget note: per retained answer, Direct costs 1 LLM call (10 words); Gibbs costs
`thinning` = 20 calls of 1 word each (~2x the generated tokens, plus prompt reads,
which vLLM prefix caching makes cheap since all prompts share the instruction
prefix). The continuation variants pay 1 call per word: 10 per Direct-continuation
answer, and the same 20 per retained Gibbs-continuation answer (plus a 10-call
initialisation).

## Reproducing

```bash
# one-time: scoring assets (~2GB download, needs internet)
bash divergent_association_task/download_assets.sh

# full experiment: vLLM server + all four methods x 3 seeds + scoring
sbatch divergent_association_task/sbatch.sh                                   # Llama-3.1-8B-Instruct, T=1.0
sbatch divergent_association_task/sbatch.sh meta-llama/Llama-3.1-8B 1.0 3     # base model
sbatch divergent_association_task/sbatch.sh allenai/Olmo-3.1-32B-Instruct 1.0 3
sbatch divergent_association_task/sbatch.sh allenai/Olmo-3-32B-Think 1.0 3    # see the caveat under Results
sbatch divergent_association_task/sbatch.sh allenai/Olmo-3-1125-32B 1.0 3

# or manually against a running server
uv run python divergent_association_task/run.py --port 8000 --seed 42
uv run python divergent_association_task/evaluate.py --plot
```

Defaults: 100 answers per method per seed, seeds 42-44, 4 chains, temperature 1.0,
10 words per answer, all four methods (`--methods` selects a subset). Per seed
this is 100 direct calls, 1,000 direct-continuation calls, and 4 x 551 Gibbs
steps per Gibbs variant.

## Results

SLURM jobs 3622350 + 3624552 (Llama-3.1-8B-Instruct), 3624600 (Llama-3.1-8B),
3624692 (Olmo-3-1125-32B), 3624693 (Olmo-3-32B-Think); T=1.0, seeds 42-44, 300
answers per method per model, one uniform protocol for all models (no
reasoning accommodations; `--manual_reasoning` exists but is deliberately
unused). The Olmo-3-32B-Think rows carry a prompt-template caveat that makes
its chat and continuation numbers non-comparable - see the caveat in its
subsection below. Full details: `results/summary.json`; one plot per model:
`results/dat_summary_<model>_temp1.0.png` (score boxplots, score along the
chain, and the fraction of chain states that are valid answers).

Cross-model summary (DAT mean +/- std over valid answers; human average 78.4):

| model | direct | direct_cont. | gibbs | gibbs_cont. |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 83.07 +/- 4.28 | 83.43 +/- 3.63 | 83.44 +/- 3.35 | **86.37 +/- 3.82** |
| Olmo-3-32B-Think | 74.55 +/- 4.99 | 74.88 +/- 4.83 | **80.16 +/- 5.12** | 75.21 +/- 6.50 |
| Llama-3.1-8B (base) | 81.24 +/- 9.82 | 81.13 +/- 8.76 | invalid (1/300) | invalid (159/300) |
| Olmo-3-1125-32B (base) | 76.44 +/- 7.61 | 76.73 +/- 7.54 | invalid (1/300) | invalid (29/300) |

Two regularities: **on both instruct-class models, exactly one Gibbs variant
delivers a large, significant gain over both Direct baselines** - but which
conditional format wins is model-dependent (continuation for Llama-Instruct,
chat for Olmo-Think; details below). And **on both base models the Gibbs chains
drift out of the space of valid answers** - the instability first seen on
Llama-3.1-8B replicates at 32B. A third protocol - rejection sampling against
the scorer's lexicon - repairs the base-model instability and takes base
models to the top of this table (86.86 / 86.13); see "Fixing the drift" under
Base models.

### Llama-3.1-8B-Instruct

All 1,200 answers had >= 7 valid words.

| method | DAT (pooled) | per-seed means | min | 5th pct | below human avg |
|---|---|---|---|---|---|
| direct | 83.07 +/- 4.28 | 83.27 / 82.57 / 83.37 | 61.5 | 75.7 | 12.7% |
| direct_continuation | 83.43 +/- 3.63 | 83.32 / 83.62 / 83.35 | 73.1 | 76.8 | 11.7% |
| gibbs  | 83.44 +/- 3.35 | 83.65 / 82.81 / 83.87 | 75.0 | 78.0 | 6.0% |
| **gibbs_continuation** | **86.37 +/- 3.82** | 86.54 / 86.12 / 86.45 | 73.9 | 79.5 | 2.7% |

Example answers (worst / median / best per method, scores from the reference
scorer):

| | score | answer |
|---|---|---|
| direct, worst | 61.5 | space, galaxy, nebula, star, sun, comet, moon, horizon, mountain, pond |
| direct, median | 83.3 | cloud, triangle, ocean, pine, radio, tartan, suite, nasal, karate, sheep |
| direct, best | 94.2 | butterfly, toolbox, kilogram, novelty, vilify, structure, tornado, citizen, fragment, disclose |
| gibbs, worst | 75.0 | ocean, hill, cloud, fountain, basin, kitchen, ruler, tiger, visa, ember |
| gibbs, median | 83.5 | boulder, harmony, foam, scent, thread, cloud, triangle, whirlpool, silence, citadel |
| gibbs, best | 91.6 | flagon, infinity, fungus, ticket, flipper, snack, whisper, cloud, cemetery, canvas |
| direct_continuation, worst | 73.1 | cloud, key, ocean, sand, knight, storm, diamond, piano, maple, sphinx |
| direct_continuation, median | 83.5 | cloud, fiddle, star, sand, phone, library, snake, butter, piano, mountain |
| direct_continuation, best | 91.3 | garden, piano, triangle, kitten, obliteration, milestone, honeybee, sandstorm, whistle, nostalgia |
| gibbs_continuation, worst | 73.9 | skeleton, torch, unger, ocean, shadow, novelty, horizon, sword, equipment, soap |
| gibbs_continuation, median | 86.6 | nose, cloud, advice, harmony, xylophone, passport, sunset, network, avalanche, sand |
| gibbs_continuation, best | 95.0 | granola, memoir, suture, galaxy, avalanche, sandcastle, thermostat, gargoyle, syrup, brush |

The direct worst answer is the order-bias failure mode in miniature: the pass
opens with "space" and autoregressively locks into an astronomy theme that later
words cannot undo. The gibbs worst is merely a loose answer - under conditional
resampling a collapsed theme is unstable, because each themed word gets
re-drawn conditioned on the rest and escapes. The gibbs_continuation answers
also range over rarer vocabulary (granola, suture, gargoyle) than the
chat-formatted methods, which mostly recycle the assistant's favourite words;
its worst answer contains the run's one non-word ("unger", dropped by the
scorer's dictionary check).

- **All methods beat the human average** (78.4); direct, direct_continuation and
  gibbs land where Bellemare-Pepin et al. place the strongest chat models
  (~83-85), and **gibbs_continuation clears them by ~3 points** (86.37; Welch
  t = 9.9, p ~ 1e-21, Cohen's d = 0.8 against every other method, consistent in
  all three seeds). Continuation alone does nothing: direct_continuation vs
  direct is +0.36 (p = 0.27) - the gain needs Gibbs *and* continuation together.
- **Chat-Gibbs vs direct** (both chat-formatted): means statistically
  indistinguishable (Welch p = 0.24) but variances differ (Levene p = 0.0017) -
  Gibbs collapses the low tail (min 75.0 vs 61.5; below-human 6.0% vs 12.7%).
  Chen & Ding frame sampling as a creativity-instability trade-off; the Gibbs
  stationary distribution keeps the mean while removing the instability - the
  one-pass failure modes (an early bad word later words cannot undo) get
  resampled away. gibbs_continuation pushes the tail further (2.7% below human).
- **Chain behaviour**: chat-gibbs is flat from step 0 (mixes around its
  direct-init level; benefit is concentration, not climb). gibbs_continuation
  visibly **climbs**: it starts at its direct_continuation init level (~83) and
  reaches its ~86-88 stationary level within ~25 single-word updates (2-3
  sweeps) - the first case where the chain's stationary distribution clearly
  beats its own initialisation.
- **Word repetition** (the papers' mode-collapse observation, reproduced):
  "cloud" appears in 86% of direct, 92% of direct_continuation, and 78% of
  chat-gibbs answers - but only **23% under gibbs_continuation** (top-10 words:
  12.9% of word slots vs 22-25%; 1,169 unique words vs 911-998). A plausible
  reading: the chat-formatted conditional stays inside the RLHF assistant's
  favourite-word modes, so chat-Gibbs mixes around the same joint as direct,
  while the raw JSON-continuation conditional taps the broader pretraining
  distribution; Gibbs's mutual conditioning then turns that extra diversity
  into higher pairwise distance.
- **Cost**: per seed - direct 100 calls/31s, direct_continuation 1,000/23s,
  gibbs 2,204/100s, gibbs_continuation 2,240/54s (continuation calls are
  single-word with short prefix-cached prompts, so more calls != more time).

### Olmo-3-32B-Think

Run under the same protocol as every other model: the constrained JSON starts
at token 1, so the Think model emits no think tokens (which also makes it as
fast as a plain instruct model: the whole 4-method run took ~15 min).

> **Caveat (found after these runs; superseded by an Olmo-3.1-32B-Instruct
> re-run).** This model's chat template ends its generation prompt with
> `<|im_start|>assistant\n<think>`, and the server ran without
> `--reasoning-parser`, so the grammar applies from the first generated token:
> the model is *inside an unterminated think block* and can never close it.
> Two consequences.
> (a) The `direct`/`gibbs` rows below are genuinely reasoning-free, but sampled
> off-distribution - visible in the retry rate, 18-932 rejected draws per job
> (jobs 3624693 / 3621600 / 3622338) against 0-5 for Llama-3.1-8B-Instruct in
> the same harness, mostly whitespace-flooded truncations, whitespace being the
> only free token the JSON grammar leaves.
> (b) The continuation methods send `add_generation_prompt=False,
> continue_final_message=True`, so they get *no* `<think>` at all. The chat and
> continuation columns for this model therefore differ by prompt prefix as well
> as by conditional format, which confounds the "mirror image of Llama-Instruct"
> reading below. Every other model in the table is unaffected (no think prefill
> in their templates).

| method | DAT | valid |
|---|---|---|
| direct | 74.55 +/- 4.99 | 300/300 |
| direct_continuation | 74.88 +/- 4.83 | 300/300 |
| **gibbs** | **80.16 +/- 5.12** | 297/300 |
| gibbs_continuation | 75.21 +/- 6.50 | 279/300 |

- **Direct scores below the human average** (74.55; 79% of answers below 78.4)
  - consistent with Bellemare-Pepin et al.'s finding that reasoning models are
  the weak DAT performers (DeepSeek-R1 ~72, o4-mini ~75, Gemini-2.5-Pro ~76).
- **Chat-Gibbs is the winner here, and it is the largest Gibbs gain of any
  model**: +5.61 over direct (Welch t = 13.5, p ~ 1e-36, d = 1.11; per-seed
  80.46 / 80.28 / 79.73), lifting the model from well below to above the human
  average (below-human answers drop from 79% to 35%). The chain climbs slowly -
  it reaches its ~80-82 stationary level after ~100-300 single-word updates
  (10-30 sweeps), versus ~25 updates for Llama's gibbs_continuation - so
  burn-in matters for this model.
- **The winning format is the mirror image of Llama-Instruct**:
  gibbs_continuation gains nothing here (+0.34 vs direct_continuation,
  p = 0.48, with mild drift: 279/300 valid). Olmo's pretraining continuation
  distribution stays on common concrete nouns (top words: tree, book, water),
  so continuation conditionals add no diversity; its chat conditional, by
  contrast, is diffuse enough for the chain to move (direct's top word covers
  47% of answers vs 86% for Llama-Instruct). Where the chain moves to is
  itself interesting: chat-Gibbs shifts the vocabulary toward abstracta -
  "void" appears in 60% of its answers, plus "thought", "infinity", "concept" -
  i.e. it semi-collapses onto a new mode of words that are far from
  *everything*, a legitimate DAT strategy (repetition is redistributed, not
  eliminated).
- Together with Llama-Instruct this pins down the general pattern: **Gibbs
  improves DAT through whichever conditional family is broad enough to move
  while staying anchored to valid answers** - which family that is (chat vs
  continuation) depends on where the model's conditional mass lives.

Example answers (scores from the reference scorer):

| | score | answer |
|---|---|---|
| direct, median | 74.8 | rock, cloud, fire, water, book, chair, tree, car, light, darkness |
| gibbs, median | 80.6 | flow, void, galaxy, sunshine, void, cactus, idea, chaos, metamorphosis, apple |
| gibbs, best | 91.9 | brick, bubblegum, void, sea, choosable, xenon, velvet, abstract, sequences, infinity |

### Base models (Llama-3.1-8B and Olmo-3-1125-32B)

The base runs are the control for the RLHF-mode interpretation above: a base
model has no assistant modes, and all four methods run through the completions
API, so any chat-vs-continuation difference reduces to pure format.

| model | method | DAT (valid answers only) | valid |
|---|---|---|---|
| Llama-3.1-8B | direct | 81.24 +/- 9.82 | 271/300 |
| Llama-3.1-8B | direct_continuation | 81.13 +/- 8.76 | 277/300 |
| Llama-3.1-8B | gibbs | (n=1 - meaningless) | **1/300** |
| Llama-3.1-8B | gibbs_continuation | 85.33 (survivorship-biased) | **159/300** |
| Olmo-3-1125-32B | direct | 76.44 +/- 7.61 | 288/300 |
| Olmo-3-1125-32B | direct_continuation | 76.73 +/- 7.54 | 286/300 |
| Olmo-3-1125-32B | gibbs | (n=1 - meaningless) | **1/300** |
| Olmo-3-1125-32B | gibbs_continuation | 86.04 (survivorship-biased) | **29/300** |

- **Direct works, Gibbs breaks.** One-pass methods stay ~90% valid and score
  81 with 2-3x the instruct variance (worst answer 28.6; occasional theme
  collapse like a row of ten blood-compounds). But the **chat-format Gibbs
  chain random-walks out of the English lexicon**: the fraction of valid chain
  states falls from 0.92 at step 0 to 0.25 by step 50 and 0.00 from step ~100
  on. A seed-42 chain: step 0 "apple, banana, band, chair, ..." -> step 50
  "heterographies, perazineuraedifierat, ..." -> step 550
  "fgklcqnohisvrejudapm, qrcpnebfauhgikdlmjvt, ..." (one top-frequency "word"
  is literally "xyzxyzxyzxyzxyzxyzxy"). Continuation-Gibbs drifts too, only
  slower (validity 1.0 -> 0.33 by the end; real words but increasingly
  non-nouns: "never", "should", "pale").
- **The instability replicates at 32B, faster**: Olmo-3-1125-32B chat-Gibbs
  validity hits 0.00 by step ~50 (vs ~100 for Llama-8B) and even its
  continuation-Gibbs reaches 0.00 by step ~300 (29/300 answers survive
  thinning). Scale does not rescue base-model Gibbs; both one-pass methods
  stay ~95% valid on the same model.
- **Why**: the DAT objective is maximised by out-of-vocabulary strings (Chen &
  Ding's surprisal confound taken to its limit), and a base conditional has no
  instruction-following pressure to stay in the lexicon - each resample
  conditioned on already-odd words makes odder words more likely, and 551
  steps compound what a 1-step Direct sample never encounters. Iterating LLM
  conditionals inherits their fidelity: RLHF is what anchors the instruct
  chains to well-formed noun lists (validity stays at 300/300 there), so on
  the instruct model Gibbs can only redistribute mass across valid answers,
  while on the base model the chain's stationary distribution is not
  concentrated on valid answers at all.
- **The mode-collapse control confirms the interpretation**: base direct's
  most frequent word appears in 7% of answers for Llama ("value") and 26% for
  Olmo ("apple") vs "cloud" in 86% on Llama-Instruct - the extreme repetition
  the papers observed is a post-training artifact, and the gibbs_continuation
  win on Llama-Instruct is exactly the regime where the RLHF anchor keeps
  answers valid while the continuation conditional escapes the anchor's
  favourite words.
- **Practical upshot**: run DAT Gibbs on instruction-tuned conditionals, or
  restrict base-model kernels to the lexicon - implemented and validated as
  `--reject_invalid` / `--reject_duplicates` below.

#### Why the chains lose validity (transition forensics)

Every Gibbs run stores its full chains and which key each step resampled, so
every conditional draw can be classified (`analyze_chain_validity.py`; numbers
below for Llama-3.1-8B). Non-words do all the damage - duplicate rates are
<2% of draws everywhere, and ~0 in states.

- **The per-draw failure rate is strongly context-sensitive.** P(new word is a
  non-word | k non-words among the other 9), continuation kernel: 0.035 (k=0)
  -> 0.149 (k=1) -> 0.31 (k=3) -> 0.99 (k=9); chat kernel: 0.099 -> 0.21 ->
  0.40 -> 0.99. One bad word in context quadruples the odds of minting the
  next one: bad begets worse.
- **The conditional prompt format is not the problem.** From a *clean* context
  the Gibbs kernel is as good as the ancestral generator (3.5-4% non-words vs
  6-14% per word within direct passes, whose own rate rises with prefix
  length). What differs is *exposure*: a direct pass calls the generator 10
  times on its self-generated clean prefix and stops; the chain calls it 551
  times on its own output, so the context-sensitive error compounds.
- **Corruption outruns repair (base), and the register explains chat vs
  continuation.** Once a position holds a non-word, the chat kernel repairs it
  with probability 0.013 vs 0.174 for continuation - a gibberish "partial
  answer" makes the chat conditional continue in gibberish register, while
  the continuation prefill's local shape (short quoted strings in JSON) still
  pulls toward real words. Hence chat collapses by step ~100 and continuation
  drifts slowly. On instruct the asymmetry flips (corrupt 0.010/0.017, repair
  0.985/0.957), pinning chains to valid answers - and its rare "non-words" are
  mostly scorer-vocabulary misses like "fryingpan" or "fibonacci", not
  gibberish.
- **The feedback is sufficient to explain the whole effect**: a birth-death
  simulation driven only by the measured P(non-word | k) tables reproduces
  the observed drift (chat: simulated 6.6/9.6/9.9 mean non-words at steps
  50/150/300 vs observed 7.5/9.8/10.0; continuation matches shape and scale).
  No additional mechanism - duplicates, key order, thinning - is needed.

#### Fixing the drift: rejection sampling (`--reject_invalid`, `--reject_duplicates`)

The forensics point at the remedy: from a *clean* context every kernel is fine,
so restricting each conditional to scorable words cuts the feedback loop at its
root. `--reject_invalid` resamples any draw containing a word the scorer cannot
embed (lexicon = `dat.Model().vectors`, cached as `assets/valid_words.txt`;
whole-answer rejection for direct, per-word for the other methods);
`--reject_duplicates` additionally resamples draws that repeat a word, in-draw
or against the prompt context (hyphen-insensitive, mirroring the scorer's
uniqueness rule). Both are applied identically to all four methods and never
see a GloVe distance - they restrict the task's support (which the scorer
already enforces at evaluation time), they do not optimise the metric. Every
run reports its **acceptance rate** (the `accept` column in `evaluate.py`, and
`reject_stats` in each payload): a rejected score is only meaningful alongside
how hard the sampler was pushed to produce it. Results carry `_reject` /
`_dedup` tags (SLURM jobs 3625109/3625110 and 3625182/3625183; reproduce with
`REJECT=true DEDUP=true sbatch divergent_association_task/sbatch.sh <model> 1.0 3`).

| model | method | `_reject`: DAT (valid) | `_reject_dedup`: DAT (valid, accept) |
|---|---|---|---|
| Llama-3.1-8B | direct | 79.63 +/- 10.05 (290/300) | 79.81 +/- 10.08 (300/300, 0.58) |
| Llama-3.1-8B | direct_continuation | 80.48 +/- 9.70 (296/300) | 81.62 +/- 9.37 (300/300, 0.94) |
| Llama-3.1-8B | **gibbs** | **86.86 +/- 5.39 (300/300)** | **86.40 +/- 5.23 (300/300, 0.84)** |
| Llama-3.1-8B | gibbs_continuation | 83.50 +/- 8.52 (253/300) | 84.79 +/- 6.25 (300/300, 0.95) |
| Olmo-3-1125-32B | direct | 75.29 +/- 7.62 (294/300) | 75.75 +/- 7.98 (300/300, 0.82) |
| Olmo-3-1125-32B | direct_continuation | 76.54 +/- 7.68 (289/300) | 75.77 +/- 7.85 (300/300, 0.98) |
| Olmo-3-1125-32B | gibbs | 86.85 (**50/300**) | 84.59 +/- **11.48** (300/300, **0.73**) |
| Olmo-3-1125-32B | **gibbs_continuation** | **86.13 +/- 5.77 (279/300)** | **85.89 +/- 5.73 (300/300, 0.92)** |

(Acceptance rates exist only for the `_dedup` runs; the earlier `_reject` jobs
predate the diagnostic.)

- **Rejection works exactly as the forensics predicted**: non-words go extinct
  (zero across all rejected runs - residual invalidity in the `_reject` column
  is pure duplicates, which `_dedup` then removes), and where the kernel is
  healthy, acceptance stays high, so the sampler runs at its clean-context
  regime at negligible cost.
- **Headline: base models + lexicon-restricted Gibbs match or beat the best
  instruct result** (86.37). Llama-8B chat-Gibbs: 86.86, +7.23 over
  direct_reject (Welch t = 10.8, p ~ 3e-24, d = 0.90), with essentially no
  repetition (top word in 5 of 3,000 slots). Olmo-32B continuation-Gibbs:
  86.13, +9.59 over its direct baseline (t = 16.8, p ~ 2e-51, **d = 1.41, the
  largest effect in the benchmark**). As before, each model has one healthy
  kernel format and the win comes through it.
- **The degenerate kernel launders instead of healing - the cautionary cell.**
  Olmo chat-Gibbs under `_reject` duplicate-collapses onto dictionary-legal
  filler ("xxx" in 1,911 of ~3,000 retained slots, plus "null", "undefined"):
  50/300 valid. Adding `_dedup` makes it *look* fixed (300/300) - but the
  kernel routed around both constraints, enumerating **distinct** filler
  variants (roman numerals "xxxvii"/"xxxii"/"xxxviii", "answer", "word",
  "something") and dictionary-valid non-nouns ("questionably", "filed") that
  the scorer cannot police (it has no part-of-speech check). The tell is the
  diagnostic triple: depressed acceptance (0.73, pooled - late-chain decay is
  worse), doubled variance (11.48), and unnatural top-words. Constraints
  cannot fix a kernel that wants to escape the task; validity alone certifies
  nothing. This cell should be read as "sampler degenerate", not "84.59".
- **On the benchmark-hacking question**: rejection is support restriction, not
  metric optimisation - uniform across methods, score-blind, and enforcing
  only what the scorer already enforces. Empirically it *deflates* one-pass
  scores (direct 81.24 -> 79.63) by removing the high-surprisal words whose
  inflated distances Chen & Ding flagged. Two disclosures still belong next to
  any rejected number: the lexicon is the *scorer's embeddable vocabulary*
  rather than a neutral English dictionary (mild evaluation leakage; an
  independent word list would be cleaner), and rejected runs are a distinct
  protocol - not comparable to the papers' numbers, and "beats the human
  average" carries the qualifier "given externally enforced lexical
  compliance".

## Limitations / notes

- The DAT instruction wording is held identical between the joint and conditional
  prompts; only the "you have already written / write the remaining word" framing
  differs. Any score difference is attributable to the inference procedure, not
  the instructions.
- Word-level constrained decoding (lowercase single words) removes the papers'
  parse-failure/adherence confound; the dictionary check in scoring still rejects
  non-words, and duplicates simply reduce `n_valid`.
- `n_words=10, minimum=7` follows Olson et al.; both papers confirm scores are
  stable to that choice.
