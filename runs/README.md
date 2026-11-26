# Run artefacts for the three arms reported in the paper

Produced by the code in this repository on `pistachio_hard`, 100 targets, a budget of 250 LLM
calls per target and a 24 h per-target time cap. One directory per arm.

| Directory | Arm | Solved | Matched vs `baseline_star` |
|---|---|---|---|
| `baseline/` | AND/OR search, no tools, no skills | 79 / 100 | — |
| `baseline_star/` | the control this work compares against | 83 / 100 | — |
| `baseline_star_tools/` | the five tools and the skill path (`examples/config_tool_all.yaml`) | **96 / 100** | 13 W / 0 L, **+13** |

A second run of `baseline_star` scores 80, so treat anything inside **±3** as noise. Compare arms
as matched per-molecule wins and losses, never as a difference of pass rates: an arm that finishes
a different subset of targets will mislead you.

## What is in each directory

**`<smiles>_0.status.json`** — one per target, the verdict and what it cost.

| Field | Meaning |
|---|---|
| `status` | `PASS`, `BUDGET` (spent all 250 calls) or `EARLY_TERM` (hit the time cap) |
| `passed` | whether a route was found in which every leaf is purchasable |
| `termination` | `found_pass`, `llm_call_budget`, `time_cap` |
| `llm_calls` | calls charged to the budget; tool calls are **not** charged |
| `n_routes` | routes in the buffer at the end |
| `tree` | `molecules`, `reactions`, `frontier` (never expanded), `solved_molecules`, `dead` |
| `fix_stats` | counters, including per-tool invocation counts (`cut_tool_calls`, `mine_tool_calls`, `validate_tool_calls`, `rejected_tool_calls`) |
| `n_attempts`, `solved_on_attempt` | attempts used, and which one succeeded |

Tool counters are **invocations, not calls that invoked**: one planner turn may call a tool several
times before it answers, so these sum to more than `llm_calls`.

**`results_planning_<smiles>_0.yaml`** — the route buffer. Routes are **sorted by score, not
chronologically**, so no trajectory can be read from them. A score of `0.0` is solved; more
negative is worse.

**`aot_expansions_<pid>.jsonl`** — one row per expansion, in the order they happened. Carries
`target_idx`, `depth`, `molecule`, `outcome`, `steps`, `valid`, `attached_new`, the per-row tool
counters and the timestamp. There is **no parent link**, so the search tree cannot be
reconstructed from these rows; use the `tree` block in the status file for its final shape.

## What is not here

**The LLM transcripts.** `llm_responses_<pid>.jsonl` holds every prompt and completion with token
counts, `finish_reason` and latency. They run to 147–221 MB per arm, which is too large to version,
so they are excluded by `.gitignore`. Ask if you need them.

**The dataset.** Reaction templates, the purchasable inventory and the RAG route corpus are
third-party and far too large; see the top-level README.

## Caveats

Timestamps in the JSONL files are UTC. Filenames have `/` stripped from the SMILES, so a lookup
by SMILES needs the same substitution. A target with no `.status.json` was never completed.
