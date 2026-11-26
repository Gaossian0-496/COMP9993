# Tool-augmented multi-step retrosynthesis (tool-all)

Source for the **tool-all** arm of a study on whether giving a language-model retrosynthesis planner
*callable tools over a reaction corpus* improves multi-step route finding. The planner is an LLM
searching an AND/OR tree (an AOT\*-style search); the tools let it ask the corpus questions before
committing a step.

This repository is the portable subset: the code needed to run that arm, with no scheduler, cluster
path or site assumption anywhere in it.

## What the tools do

Four tool blocks, all config-gated. None of them leaks the oracle's answer.

| Tool | Config block | What the planner gets |
|---|---|---|
| `get_known_splits_of_this_molecule` | `cut_tool` | Corpus splits of the current molecule, ranked by largest remaining piece, capped at 0.9 of it |
| `how_is_this_made` | `mine_tool` | How each motif is *made*: corpus frequency, what its precursors carry over background, whether anything closes its ring systems |
| `validate_route` | `check_tool` | The route checker applied to the model's **own** proposed step |
| `check_my_route`, `my_rejected_steps` | `validate_tool` | A format-only check, and this molecule's rejected first steps with the reason each was refused |

**Leakage control.** No tool returns an oracle-suggested disconnection, and none reveals
purchasability — `cut_tool.show_buyable` is false, because purchasability is the oracle's own
termination test and handing it over would be giving away the answer.

Two design points that came from getting it wrong first:

- `cut_tool` ranks candidate splits by size and filters at 0.9 of the product, so it can only ever
  propose steps that make the molecule **smaller**. Ring-forming steps and functional-group
  interconversions are unreachable by that ranking by construction. `how_is_this_made` exists to
  cover them: on one fused-heteroaromatic target the steps the solved route needed sit at 125–170%
  of the product, the templates for them exist, and no amount of size-ranked searching surfaces them.
- `how_is_this_made` ranks motifs by **rarity and never filters**. An earlier support floor deleted
  the glycosidic bond, the silyl ether and the trityl group from a nucleoside and left behind "the
  methoxy comes from an alkyl halide".

## Results on the primary benchmark

100 hard targets, 250 LLM calls per target, compared per-molecule head-to-head against the same
baseline on the same targets.

| Arm | Solved | Matched vs baseline |
|---|---|---|
| Baseline (no tools) | 83 / 100 | — |
| Baseline, second run | 80 / 100 | the noise floor, ±3 |
| Tool-disconnection (`cut_tool` only, no skills) | **96 / 100** | 13 W / 0 L, **+13** |
| Tool-all (this config) | **96 / 100** | 13 W / 0 L, **+13** |

Both tool arms land on the same figure. The cleaner attribution is tool-disconnection, which carries
no skill library at all, so its **+13 with no losses** is the gain from a single tool. Tool-all adds
the remaining four tools and the skill path and does not improve on it.

A second run of the baseline scores 80, so treat anything inside ±3 as noise.

Raw pass rates across arms are **not** comparable unless the arms finished the same molecules.
Compare matched per-molecule wins and losses.

## Requirements

- Python 3.10 and an OpenAI-compatible inference server (tested with SGLang serving a 27B model).
  `pip install -r requirements.txt` for the Python side. The versions there are pinned rather than
  floated because RDKit and rdchiral change template matching between releases, which changes which
  steps the oracle accepts.
- **`scscore/`** — `optimizer.py` imports `scscore.scscore.standalone_model_numpy` at import time.
  Not vendored here (it is a third-party model of several hundred MB). Obtain it from the SCScore
  project and place it at the repository root, or the package will not import.
- **`dataset/`** — reaction templates, the purchasable inventory and the RAG route corpus. Not
  included: the inventory alone is ~23M compounds. The loader expects a `dataset/` directory.
- **A skill library** — `examples/config_tool_all.yaml` sets `skill_library: skill_library_pg_v2.json`.
  That file is **not** in this repository. Either supply one of that name, or set `skill_repair` and
  `skill_gate` to false and `skill_library` to null, which runs the tool arm without the skill path.
  The tool-disconnection row above was produced without any skill library.

## Running it

```bash
# start an OpenAI-compatible server for the model in the config, then:
python main.py --method planning \
    --config_default examples/config_tool_all.yaml \
    --dataset_name pistachio_hard \
    --port <server port> \
    --output_dir results/tool_all \
    --workers <n>
```

`--workers N` runs N threads in **one** process sharing a single copy of the loaded data. Do not
launch N separate processes: the shared data is ~25 GB, so N processes means N copies.

Per target the run writes a route buffer (`results_planning_<smiles>_0.yaml`), a `.done` marker and a
`.status.json` (`PASS` / `BUDGET` / `EARLY_TERM`), a tree dump, and a JSONL of every LLM call. A
restart skips any target that already has a `.done`.

### Environment

| Variable | Effect |
|---|---|
| `LLM_SYN_JOBS_DIR` | Where the `min_remaining` tail-stop writes its sentinel. Defaults to `<repo>/../jobs`. |

## Layout

```
main.py                 entry point: dataset load, shared data, per-target loop, threaded workers
run.py                  GA loop: initialization, modification, stall gate
optimizer.py            Oracle, scoring, route checking, the single LLM choke point
route.py                a validated reaction chain
modification_prompt.py  prompt construction and feedback

aot_search.py           the AND/OR search: expansion, selection, retries, tool dispatch
aot_tree.py             tree state (AndOrTree); reactions is a dict keyed "product>>reactants"

agentic_tools.py        tool schemas and dispatch
disconnections.py       corpus splits of a molecule
split_pool.py           candidate split ranking
checker_feedback.py     per-step rejection reasons
skill_miner.py          how_is_this_made, and the write-only skill writer
skill_matcher.py        RDKit substructure gates
skill_memory.py         skill library load, lifecycle, rendering
skill_signature.py      reaction-centre signatures
skill_repair.py         skill-guided repair of a rejected step

examples/config_tool_all.yaml   the tool-all configuration
```

`skill_miner.py` is named for the skill path but supplies `how_is_this_made`, which is a **tool**:
it is required even with the skill path switched off.

## Not included, and why

- **Advisor arms.** A second model advising the planner, in several forms. All were neutral or
  negative and all are closed. Their call sites remain in `aot_search.py` as a record of what was
  tried, behind config blocks absent from the shipped config; the modules themselves are optional
  imports, so the package loads without them (`aot_search.ADVISORS_AVAILABLE` reports which).
- **`describe_this_molecule`** (`mol_tool`). Defaults to off and is absent from the config; its
  backing module is an optional import.
- **Multimodal, prompt-ablation and skill-memory arms.** The code paths that survive are config-gated
  and off; the arms are not part of this distribution.
- **Run outputs, datasets, models, skill stores.** Reproducible from the code, or third-party, or
  far too large.

## Licence

MIT, in `LICENSE`, carrying the upstream copyright of LLM-Syn-Planner (Copyright (c) 2025 Haorui
Wang) alongside this work's.

## Provenance

Built on **LLM-Syn-Planner** (Wang et al., ICML 2025) — a genetic algorithm over LLM-proposed routes
with retrieval of similar corpus routes, validated against reaction templates and a purchasable
inventory. Five files are modified from that upstream (`main.py`, `run.py`, `optimizer.py`,
`route.py`, `modification_prompt.py`); the search, the tools and the skill code are new.

The search follows the released **AOT\*** code for its mechanics.
