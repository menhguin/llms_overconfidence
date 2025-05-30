# LLMS Overconfidence – Reproduction Guide (Anonymised Version)

This repo contains all code, prompts and configs used in the NeurIPS-25 paper *Two LLMs Debate, Both Are Certain They've Won* (authors anonymised for peer review). Follow the steps below to reproduce the experiments **out-of-the-box** on any machine that can reach the OpenRouter API.

---

## 1.  Quick-Start (1 command)

```powershell
# Ensure you are in the llms_overconfidence directory
# (e.g., cd <path-to-supplementary-material>/llms_overconfidence)

# You can obtain an OpenRouter API key from https://openrouter.ai/
# Provide your OpenRouter key (once per shell) **or** put it in a `.env` file here
$Env:OPENROUTER_API_KEY = "sk-..."  # bash/zsh: export OPENROUTER_API_KEY="sk-..."
```

At this point you are ready to run **any** experiment script.

---

## 2.  Smoke-Test: one debate, one model

To verify your key / env are correct, run a **single** self-debate between two copies of Claude-3.7-Sonnet:

```powershell
python -m src.scripts.test_single_debate
```
*Runtime*: ~60 s • *Cost*: < $0.05 • *Output*: `experiments/test_single/claude_sonnet_vs_self.json`

---

## 3.  Full experiments used in the paper

| Command | What happens | Approx. time | Cost* |
|---------|--------------|--------------|-------|
| `python -m src.scripts.replicate_self_debate` | 3× self-debate ablations (private / public / anchored) for every model in `config_data/debate_models.json` | 30 min | $4 |
| `python -m src.scripts.multi_round` | 60 cross-model debates + AI-judge verdicts | 2 h | $9 |

\*Costs were measured May-2025; they will vary with model pricing.

All artefacts are written under `experiments/` and a live log to `tournament.log`.

---

## 3-b.  Additional experiment scripts (Appendix)

| Script | Purpose (as reported in the paper's appendix) | Typical time | Est. cost |
|--------|----------------------------------------------|--------------|-----------|
| `python -m src.scripts.four_round_debate` | 4-round debate variant used for robustness check (App. G) | ~45 min | $2 |
| `python -m src.scripts.redteam_debate` | Self-red-team prompting ablation that reduces escalation (Table 8) | ~25 min | $1 |
| `python -m src.scripts.opening_hypothesis_tests` | Tests whether models' *initial* confidence exceeds 50 % across motions | 5 min | <$0.50 |
| `python -m src.scripts.mismatch_confidence` | Detects divergence between private chain-of-thought & public bet (§ 3.5) | 10 min | <$1 |
| `python -m src.scripts.both_overconfident` | Counts logically impossible finales where both sides > 75 % | 2 min | — |

These scripts assume debates generated by the main runs already exist under `experiments/`; they will simply analyse or extend them and emit CSV/PNG outputs in the repo root.

---

## 4.  Customising a run

*   **Change models** → edit `config_data/debate_models.json` (remove rows or add new IDs).
*   **Change topics** → edit `config_data/topic_list.json`.
*   **Change debate prompts / rubric** → `config_data/debate_prompts.yaml`.
*   **Change number of rounds** → open `src/core/config.py` and set `num_rounds`.

---

## 5.  Troubleshooting cheatsheet

| Problem | Fix |
|---------|-----|
| `KeyError: 'OPENROUTER_API_KEY'` | Export the key in the current shell **or** place `.env` with `OPENROUTER_API_KEY=...` in the repo root. |
| `ModuleNotFoundError: src` | The code assumes you are running commands from the root of the `llms_overconfidence` directory. Ensure your terminal's current working directory is correct. |
| Messages about `Get-Content` / `cat` | Ignore – they were caused by piping output in earlier instructions. Current commands run without pipes. |
| Debate JSON shows `-1` speeches | Transient model failure. Re-run the affected debate or lower concurrency. |

---

## 6.  Citation (anonymised)

```
@article{ANONYMISED FOR PEER REVIEW,
  title   = {Two LLMs Debate, Both Are Certain They've Won},
  author  = {ANONYMISED FOR PEER REVIEW},
  journal = {Neural Information Processing Systems},
  year    = {2025}
}
```

---

## 7.  License
MIT for code; prompt text is adapted from public-domain debate materials. Model outputs are subject to the providers' terms.
