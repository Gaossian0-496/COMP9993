"""AOT* search mode: LLM-generated routes mapped onto an AND-OR tree.

Selected by config `search: aot` (main.py); without it the original LLM-Syn-Planner
GA runs unchanged. Plan and rationale: docs/plan_aot_star.md.

Port of the released AOT* (github.com/ShawnKS/AOTstar, MIT; read-only reference
clone of the released AOT* code). Search mechanics follow the
RELEASE (user decision 2026-09-28): flat UCB over frontier reactions divided by
(1 + path_burden * unsolved_penalty); chemistry-only node value
max(0, 1 + oracle.reward/14); `average` backprop; depth <= 20; <= 5 re-expansions
per molecule; 3 initial root calls; <= 5 LLM retries per expansion (8 per root
call); failed expansions mark the molecule dead; the root is re-expanded when the
frontier is empty; a solved molecule's other branches leave the frontier.
"Salvage" (the release's drift fix) is included behind `aot.salvage` (default on).

Held identical to baseline A for a fair comparison (user decision): the LLM, its
sampling and logging (query_LLM), the initialization prompt (build_init_prompt_parts,
verbatim upstream), RAG example sampling, the template validator (sanitize), the
inventory and the SCScore oracle.

Deliberate differences from the release, each for correctness or fairness:
- Budget = LLM calls (every call, including retries), `aot.llm_call_budget`.
- A valid step is attached under the molecule it actually decomposes (canonical
  SMILES match within the route's reachable set). The release attaches the first
  valid step under the expanded molecule even when that step decomposes a different
  molecule.
- No solved-subroute cache shared across target molecules (per-molecule budgets).
- Identical reactions are merged into one AND node (aot_tree).
"""
import ast
import copy
import json
import os
import random
import re
import sys
import time
from difflib import SequenceMatcher

import numpy as np
from rdkit import Chem, DataStructs
import rdkit.Chem.AllChem as AllChem
from rdchiral.initialization import rdchiralReactants

import agentic_tools as agt
# The ADVISOR ARMS are not part of this distribution. Every advisor arm was closed as neutral or
# negative, and no reported tool arm reaches one: each call site is behind a config block that is
# absent from the shipped config. With the modules missing the package still imports and the tool
# arms run unchanged.
try:
    import advisor as adv
    import advisor_live as al
    import advisor_plan9 as ap9
    import advisor_prune as apr
    import advisor_steer as ast_
    import advisor_ask
    import advisor_select as asel
    import advisor_view as av
    import advisor_tools as at
    ADVISORS_AVAILABLE = True
except ImportError:
    adv = al = ap9 = apr = ast_ = advisor_ask = asel = av = at = None
    ADVISORS_AVAILABLE = False
import disconnections as dc
import joint_path as jp
try:
    import mol_facts as mf   # backs describe_this_molecule; off by default, not shipped
except ImportError:
    mf = None
import reopen
import self_notes as sn
import situational_rules as srul
import split_pool as sp
import switch_choice as sc
from aot_tree import AndOrTree
from optimizer import extract_molecules_from_output, _APPEND_LOCK
from modification_prompt import redundant_reaction
import threading
from datetime import datetime as _dt
from run import planning_Optimizer, build_init_prompt_parts, clean_last_step, process_reaction_routes, \
    extract_route_block
from utils import sanitize_smiles, is_reaction_in_dict, run_retro
from syntheseus import Molecule

AOT_DEFAULTS = {
    "llm_call_budget": 250,
    "c_param": 0.5,
    "unsolved_penalty": 0.3,
    "depth_penalty": 0.0,         # 2026-10-20 arm bfd: >0 divides the selection score by (1 + depth_penalty * depth)
    "max_depth": 20,
    "max_reexpansions": 5,
    "num_initial_routes": 3,
    "max_expansion_retries": 5,
    "max_root_retries": 8,
    "backprop": "average",
    "salvage": True,
    "tried_note": False,          # fix C: tell the LLM which reactions the tree already has for this molecule
    "salvage_tanimoto_weight": 0.7,
    "skill_repair": False,        # 2026-10-12: AutoGuide-style optional hint on stuck expansions (skill_repair.py)
    "skill_library": "skill_library_repair.json",
    "skill_max": 2,
    "skill_gate": False,          # 2026-10-13: only offer entries whose structural gate (RDKit SMARTS) passes on the molecule
    "skill_fixed_k": 0,           # gated mode: this many hints come from the fixed rank (specificity, support), no LLM call
    "skill_rotate": False,        # never show the same entry twice for one molecule
    # 2026-11-20 (user: "deterministically add all that is relevant + llm selection up to 5 more"):
    #   "gated"   = the selector only ever sees gate-PASSING entries (released behaviour; every finished
    #               gated arm, e.g. A_capsskills, was run this way and stays reproducible)
    #   "library" = after skill_fixed_k has taken every gate-passing entry, the selector is offered the
    #               REST OF THE LIBRARY, including entries whose gate did not fire. Needed because with
    #               fixed_k >= the number of gate-passing entries the old menu is empty and the selector
    #               has no job at all; and because a substructure gate cannot see, for example, that a
    #               protecting group must come off before a ring can be closed.
    "skill_menu": "gated",
    # 2026-11-21 (user: "Yes do it"): add each REJECTED proposal to the tried list inside the retry
    #   loop, so attempt N+1 is at least told what attempt N just offered and lost.
    #
    #   The released retry is pure resampling: `tried` is built ONCE before the loop from the
    #   reactions already ATTACHED to the molecule, and never appended to. A failed attempt attaches
    #   nothing, so it never becomes a tree reaction, so it never enters `tried` -- attempt 5 sees the
    #   same prompt, byte for byte, as attempt 1. Measured consequence (hard100_ctrl_v2): acceptance
    #   falls 25.7% -> 23.2% -> 19.9% -> 18.0% -> 14.6% across attempts 1-5, over 2,257 retry calls,
    #   about a third of the whole budget. Sampling-diversity work calls this being trapped in a local
    #   cluster (arXiv 2502.11027); it is the mildest possible fix for it.
    #
    #   This says only "you already offered this" -- NOT why the checker refused it. That is
    #   `checker_feedback` below, and keeping the two separate is the point: this arm isolates
    #   "do not repeat yourself" from "here is the diagnosis".
    "retry_tried": False,
    "checker_feedback": False,    # 2026-10-14: on a rejected expansion, one follow-up call with the checker's diagnostics
    "checker_feedback_max": 1,    # follow-up calls per rejected answer; NOT counted in llm_call_budget (user decision)
    "junk_guard": False,          # 2026-10-16: a new unsolved reactant that is not a real molecule (junk_molecule) is dead at once
    "max_iterations": 10_000,     # guard only; the LLM-call budget is the real limit
}

# Arm notes (try 12, 2026-10-26): one self-written note per target, carried across every route call for that
# target and checkpointed so a walltime cut does not wipe it. Written in the SAME completion as the route, so
# the arm costs no extra oracle calls. max_words 0 = no instructed limit (the user's decision).
# Config `aot.termination` (2026-11-08; off by default = identical behaviour). The release closes a branch only
# when one expansion's 5 LLM tries all attach nothing (0-7 of ~200 molecules per failed strat28 search), the
# 5-reaction cap never binds, and failure is never passed up the tree, so the failed strat28 searches end with
# 92-100% of their reactions still selectable (docs/experiment_history.md, 2026-11-08).
#   no_split_dead     : before any LLM call on a selected reaction's reactants, run the checker's own template test
#                       (disconnections.template_split_exists). A molecule that NO template decomposes is dead at
#                       once (proven: the checker can never accept a step on it) and costs no call. A target with
#                       no split ends the search: EARLY_TERM, termination "root_proven_dead".
#   propagate_failure : AND-OR failure propagation (aot_tree.failure_state / closed_by_failure). A reaction with a
#                       failed reactant fails; a molecule that can get no new reaction and whose reactions all
#                       failed fails; reactions only failed branches need are no longer selected. Recomputed every
#                       iteration, so a later reaction revives a molecule. Nothing failed = release selection.
#   propagate_from    : "proven" = only no_split_dead molecules seed propagation (default); "all" = also molecules
#                       dead because an expansion's LLM tries all failed (not proof: tries 4-5 still attach ~50%).
TERMINATION_DEFAULTS = {"no_split_dead": False, "propagate_failure": False, "propagate_from": "proven"}

# Config `aot.advisor_steer` (arm steer, 2026-11-08; off by default = identical behaviour; advisor_steer.py has the
# reasoning). After every non-root expansion of molecule A (selected through reaction P), the same Qwen server with
# a fresh context chooses the next move: SIDEWAYS (re-expand A: a different reaction for it), DEEPER (A's newest
# reaction, the search's picker chooses which reactant), ELSEWHERE (the search's best reaction outside P's subtree)
# and, with allow_remove, REMOVE (P and everything only it needs are closed for good; needs
# aot.termination.propagate_failure). Moves that are not possible are not offered; with fewer than 2 there is no
# call. Unparseable answer or any error = the search's own pick. Calls are NOT counted in llm_call_budget (user).
#   context: "local"  = SMILES, size, SCScore, buyability, reactions tried, branch path
#            "splits" = local + per molecule: known splits, the biggest checker-accepted cut, one step from buyable
#            "rc5"    = local + per molecule: rc's own list (up to 5 checker-accepted cuts, biggest first)
# Every decision -> aot_steer_<pid>.jsonl (offered order, answer tail, move, the search's own pick for comparison).
STEER_DEFAULTS = {"enabled": False, "context": "local", "allow_remove": False, "temperature": 0.7,
                  "show_tried": 6, "show_siblings": 6,
                  # review (8 Nov): a failed expansion adds no reaction, so nothing else stops an advisor that keeps
                  # answering SIDEWAYS from spending a target's whole budget on one dead-end molecule. The release
                  # allows 0 retries (dead after one failed expansion); this allows this many, back to back.
                  "max_failed_retries": 1,
                  # arm steerRgate (2026-11-14, user: "build the gated steerR"). 0 = the 8 Nov arms: the advisor steers
                  # from the first expansion. > 0 = steer ONLY ONCE THE SEARCH HAS STALLED on this molecule, i.e. after
                  # this many of its LLM calls are spent without a solve; until then the release's blind UCB runs and
                  # the release's dead-marking applies, so the molecule is handled exactly as the control handles it.
                  # Why: on strat28 steerR's whole net gain is 3 rescues in the "unsolved" stratum (molecules the
                  # control fails at 250 calls), landing LATE at 50/135/175 calls -- while its one loss is a molecule
                  # the control solved at 66 calls and always-on steering drove to BUDGET. A gate below ~66 and above
                  # the control's median solve (36) should keep the rescues and drop the derailment. The threshold is
                  # the hard budget-fraction trigger run.py's stall gate settled on (inject_after_calls, default 40):
                  # its own comment records that the score-plateau signal almost never fired (1 of 9 molecules), so a
                  # call count is blunt but is the signal that actually triggers.
                  "after_calls": 0}

NOTES_DEFAULTS = {
    "enabled": False,
    "max_words": 0,               # 0 = uncapped; the guard below is what keeps the prompt inside the window
    "reserve_tokens": 4096,       # kept free ON TOP of the generation cap (llm.max_tokens)
}


def junk_molecule(smiles):
    """A carbon carrying >= 2 terminal [O-] (e.g. [O-]C([O-])[O-], R-OC([O-])[O-]): not a real
    intermediate. Advisor run 2026-10-16, idx 23: Qwen's Boc/carbonate steps produced these, each
    expansion attached more of them, so they never went dead and took 64 of its last 103 calls.
    Hits 0 intermediates on the 322 solved routes of 6 AOT runs. Carboxylates (one [O-] per carbon),
    nitro groups, sulfates and phosphates are not matched."""
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return False
    return any(a.GetSymbol() == "C" and
               sum(1 for b in a.GetNeighbors()
                   if b.GetSymbol() == "O" and b.GetFormalCharge() < 0 and b.GetDegree() == 1) >= 2
               for a in m.GetAtoms())


# Config `aot.advisor` (2026-10-14; off by default = identical behaviour). Endpoints come from the
# environment (ADVISOR_ETHER0_URL / ADVISOR_RETRODFM_URL), set by the job script, so a run without
# its servers fails at start instead of silently running a no-advice arm.
ADVISOR_DEFAULTS = {
    "enabled": False,
    "advisors": ["ether0", "retrodfm"],
    "stuck_fails": 2,            # trigger 1: >= this many attempts on the molecule that attached nothing
    "near_whole_depth": 3,       # trigger 2: path length >= this ...
    "near_whole_size": 0.9,      # ... and the molecule is still >= this fraction of the target's heavy atoms
    "max_per_molecule": 2,       # advice rounds per molecule
    "refresh_after": 2,          # a 2nd round needs >= this many further attempts
    "max_attempts_shown": 10,
    "plan_of_attack": False,     # try 8 (2026-10-17): plan + facts + one exchange; needs advisors [interns2]
    # try 9 (2026-10-18, the ap9 module): the plan is discussed before the first call and every replan_every
    # budget calls, and shown in every route prompt. Needs plan_of_attack (its facts and stuck advice).
    "plan_at_start": False,
    "replan_every": 0,           # 0 = never re-plan
    "plan_rounds": 3,            # Qwen replies per discussion (an advisor revision between them unless Qwen agrees)
    "plan_max_tokens": 24000,    # the advisor's cap in the discussion (lowered per call to fit advisor_context)
    "plan_reply_max_tokens": 2000,   # Qwen's cap in the discussion
    "advisor_context": 40960,    # the advisor server's --context-length
    # try 10 (2026-10-18, advisor_tools.py): Intern calls tools (find_disconnections, check_step, is_buyable,
    # similar_routes) in every call instead of getting the facts list. Needs plan_of_attack and a server started with
    # --tool-call-parser qwen3_coder.
    "trust_framing": False,      # arm adv9t (2026-10-20): drop the wrong-rate warning, default to following
    # arm joint (try 11, 2026-10-20, joint_path.py): symmetric co-design. Both models get the SAME route prompt,
    # propose independently, then converge on one path that may be neither proposal. Needs advisors [interns2].
    "joint": False,
    "joint_rounds": 20,          # a generous cap, discouraged in the prompt (user: "max rounds but discourage it")
    "joint_at_root": True,       # always discuss the target itself
    "joint_every": 25,           # and every N budget calls for any molecule (0 = root only)
    "joint_reserve": 6144,       # tokens kept free in the advisor's context to generate its turn
    "joint_qwen_max_tokens": 20480,   # Qwen's discussion turn is a ROUTE (median 8,676), not a 1k reply
    "tools": False,
    "max_tokens": 8192,
    "timeout_s": 900,
}

# Config `aot.real_cuts` (arm rc, 2026-10-18; off by default = identical behaviour; disconnections.py): the best
# checker-accepted known steps that are not near-copies go into Qwen's prompt for the molecule it is decomposing.
# Arm `ask` (2026-11-15): a PULL advisor, offered to the planner as a tool only once the target is stuck.
# `dry_gate` consecutive expansions that attach nothing opens it (5 fires on 11 of 100 targets and catches all 4
# failures on hard100_cutselR; a call-count gate measures elapsed effort, not lack of progress). CONTINUE is the
# default answer, the advisor NAMES a destination, and destinations carry the CHECKER-ACCEPTED split count --
# the three faults measured in the push advisors. See advisor_ask.py for the full reasoning.
# Arm `check` (2026-11-15): the ROUTE CHECKER offered to the planner as a tool, so it can test a step before
# spending a budgeted call on it. Nothing is handed over -- no splits, no suggestions, no chemistry; the tool only
# answers yes/no about the model's OWN idea, with the same per-step feedback a rejection would have produced.
# The measured target: without a split list the planner writes about four rejected steps per accepted one, and
# every rejection costs one of the 250 calls. The tool is free, so that waste can be spent for nothing instead.
# This is the push/pull counterpart of the disconnection arms: the checker is currently run FOR the model, after
# the fact. `max_iters` caps the tool loop inside one call, because the model calls an available tool on 98-99%
# of turns and an uncapped validator could grind.
_VALIDATE_ROUTE_SCHEMA = next(t for t in agt.TOOL_SCHEMAS if t["function"]["name"] == "validate_route")

CHECK_TOOL_DEFAULTS = {"enabled": False, "max_iters": 6}
CHECK_TOOL_NOTE = (
    "\nTOOL AVAILABLE: validate_route(route). Before you commit to an answer you may check a candidate route "
    "with the same checker that will score it, and get a per-step report of what is wrong. It gives you no "
    "chemistry and no suggestions -- it only tells you whether YOUR step would be accepted, and why not. Calling "
    "it is free and does not use your budget, so it is better to test a step and fix it than to submit one that "
    "fails. Your final answer must still be a route in the <ROUTE> format above.\n")

# arm valtool (2026-11-21, user: "a validation tool that the model can use to fix any issues before submission /
# understand previous failures"). A callable tool that does two leakage-free things: (1) checks a candidate route
# for FORMAT problems -- missing <ROUTE> block, not a list of step dicts, a missing field, an invalid SMILES (with
# the specific RDKit reason), a malformed reaction string -- so the model can fix it before it is thrown away; and
# (2) lists the first steps already REJECTED for this molecule this expansion, so the model does not resubmit them.
# It NEVER runs the route checker, suggests no disconnection, and exposes no purchasability, so unlike
# check_tool/validate_route it adds no oracle signal. In the valtool arm it REPLACES validate_route. The format
# logic is exactly the validation we added over upstream (extract_route_block + the parse gate + validate_smiles'
# _smiles_detail), surfaced to the model instead of only used to silently discard a bad answer.
VALIDATE_TOOL_DEFAULTS = {"enabled": False}
# Two SEPARATE tools (user, 2026-11-21: "maybe have it as separate tools the validation + the retry list"), both
# gated by aot.validate_tool.enabled. check_my_route = format only; my_rejected_steps = this molecule's history.
VALIDATE_TOOL_SCHEMA = {"type": "function", "function": {
    "name": "check_my_route",
    "description": ("Check a candidate route for FORMAT problems before you submit it -- a missing <ROUTE> block, a "
                    "block that is not a list of step dicts, a step missing a required field, an invalid SMILES (with "
                    "the specific reason), or a malformed reaction string -- so you can fix it before it is thrown "
                    "away. It says NOTHING about whether the chemistry is accepted, only whether the route string you "
                    "pass it is well-formed. Calling it is free. USE IT LAST, once your route is complete. THIS TOOL "
                    "DOES NOT SUBMIT ANYTHING: the route you pass it is an argument, not an answer, and nothing you "
                    "give it is ever read as your reply. A clean result means the string you handed over is readable "
                    "-- it does NOT mean you have answered. After it replies, correct anything it reports and then "
                    "WRITE THE FINAL ROUTE OUT IN FULL inside <ROUTE></ROUTE> in your reply. If you stop after "
                    "calling this tool, your answer is lost and the call is wasted."),
    "parameters": {"type": "object", "properties": {
        "route": {"type": "string", "description": "A candidate route: the same list of step dicts you would put "
                  "inside <ROUTE></ROUTE>."}},
        "required": ["route"]}}}
REJECTED_TOOL_SCHEMA = {"type": "function", "function": {
    "name": "my_rejected_steps",
    "description": ("List the first steps you have ALREADY had rejected for the molecule you are decomposing right "
                    "now, each with the reason it was rejected, so you do not propose them again and can see what "
                    "went wrong. Takes no argument and always answers for that one molecule. It reports only your "
                    "own rejected attempts -- it suggests nothing. Calling it is free."),
    "parameters": {"type": "object", "properties": {}}}}
# check_my_route is always offered when the arm is on; my_rejected_steps and its note line are offered ONLY once
# this molecule has at least one rejected attempt (user, 2026-11-21: "only available when there is a previous
# failure"), so an empty tool is never put in front of the model.
VALIDATE_TOOL_NOTE = (
    "\nTOOL AVAILABLE: check_my_route(route) -- check a candidate route for FORMAT problems (missing route block, "
    "unreadable structure, a missing field, an invalid SMILES, a malformed reaction) before you submit, so you can "
    "fix it. It tells you only whether the route string you pass it is well-formed, NOT whether the chemistry is "
    "accepted. Calling it is free. Use it LAST, once your route is complete. IT DOES NOT SUBMIT ANYTHING -- what you "
    "pass it is an argument, not an answer, and a clean result does NOT mean you have answered. After it replies, fix "
    "anything it reports and then write the finished route out in full inside <ROUTE></ROUTE> in your reply, in the "
    "format above. If you stop after the tool call, your answer is lost.\n")
REJECTED_TOOL_NOTE = (
    "\nTOOL AVAILABLE: my_rejected_steps() -- you have already had at least one step rejected for this molecule; "
    "call it to see those steps and why each failed, so you do not repeat them. Calling it is free.\n")

# continue_extra_tries (user 2026-11-16: "if we continue give it another up to 3 tries (enforce it) if it's at
# limit"): when the advisor answers CONTINUE during an expansion and that expansion then uses up its
# max_expansion_retries without attaching, it gets up to this many MORE tries on the same molecule, right away,
# before the molecule is written off. Granted once per expansion, only for an explicit CONTINUE (not a parse
# fallback), and every extra try is charged to the budget. Measured basis: in hard100_cutselR the 5th try still
# attaches 38% of the time (control 39%), so the cap writes off molecules that still had a fair chance.
# 0 = the cap as released.
ASK_DEFAULTS = {"enabled": False, "dry_gate": 5, "max_calls": 6, "from_search": 5, "from_coverage": 5,
                "check_uncached": 4, "show_tried": 6, "notes": True, "temperature": 0.7,
                "continue_extra_tries": 0,
                # arm askR (2026-11-18): the situational rules mined by analysis/mine_rules.py are shown to
                # the advisor with their evidence (situational_rules.py / .json), and `allow_stop` lets it
                # answer STOP -- "this line is hopeless, give the molecule up now" -- which no advisor here
                # could say before. Both default off, so arm `ask`/`ask2` behaviour is untouched.
                "rules": False, "rules_file": "situational_rules.json", "allow_stop": False}

REAL_CUTS_DEFAULTS = {
    "enabled": False,
    "max_piece": 0.9,            # only steps whose largest piece is < 90% of the molecule (not near-copies)
    "show": 5,                   # at most this many, biggest cut first
    "check_max": 60,             # candidates the route checker is run on, per molecule
}

# Arm ragcut (2026-11-13, user: "what if like rag cut tool gives splits of similar molecules"). rc/cuttool/cutselR all
# hand over splits OF THE MOLECULE ITSELF, pre-accepted by the same checker that scores the run; the 13 Nov audit found
# 80-85% of solved-route steps had been handed over verbatim, and the arm's effect may be exactness rather than any
# transfer by the model. ragcut supplies the SAME retrieval source the baseline already uses (_sample_examples: routes
# of similar molecules, pushed into every expansion prompt) distilled to one line each -- the first step of the route,
# i.e. how that similar molecule was disconnected -- and NOTHING about this molecule. Nothing is applied or checked
# against it, so the model must do the transfer. Sits between ctrl (whole routes, sampled) and rc (verified own splits).
RAG_CUTS_DEFAULTS = {
    "enabled": False,
    "show": 5,                   # deterministic top-N by Tanimoto (rc's count), most similar first -- NOT RAG's sampling
    "min_sim": 0.0,              # drop analogues below this similarity (0 = show the best `show` whatever they score)
}

# Arms nollm / cuttool / cutsel (2026-11-01; all off by default = identical behaviour). All three use rc's source
# (disconnections.best_steps: checker-accepted known splits, biggest cut first), so each differs from rc in ONE way:
#   aot.no_llm   the tool WITHOUT the LLM: each expansion takes the biggest such split not yet in the tree.
#   aot.cut_tool rc's list PULLED: Qwen may call get_known_splits(smiles) when it feels stuck (free, not budgeted).
#   aot.cut_select WHO CHOOSES: a separate Qwen call sees EVERY checker-accepted split (near-copies included) and
#                picks `pick`; those go to the main prompt in rc's exact block, biggest cut first.
# Arm advsel (2026-11-01; off by default): the advisor owns the SELECTION phase -- which open branch to
# expand next -- and nothing else (advisor_select.py has the two presentations and the reasoning).
# order: "shuffled" (propose: independent judgement) | "ranked" (refine: improve the search's ranking)
#        | "both" (shadow only: ask BOTH at the same decision -> a paired comparison, identical candidates)
# mode:  "shadow" (log the pick, UCB still drives) | "drive" (expand the advisor's pick)
ADV_SELECT_DEFAULTS = {
    "enabled": False,
    "order": "both",
    "mode": "shadow",
    "rows": 10,
    "rejects": True,        # show the attempts split into writing vs chemistry failures (try 13's block)
    "temperature": 0.7,
    "seed": 0,
}

# Arm advprune (2026-11-01; off by default): an advisor says which frontier nodes are NOT worth
# expanding, and they are DE-PRIORITISED, never removed (advisor_prune.py has the evidence and the
# reasoning). Asked every `every` expansions when the frontier is at least `min_frontier` wide; the
# advisor is Qwen on the planner's own server and its calls are not charged to the oracle budget.
ADV_PRUNE_DEFAULTS = {
    "enabled": False,
    "every": 10,            # expansions between asks
    "min_frontier": 4,      # never ask when the frontier is this narrow (nothing to triage)
    "rows": 12,             # frontier rows shown, in the search's own ranking
    "max_skip": 4,          # at most this many marks per ask
    "factor": 5.0,          # marked nodes' selection score is divided by this
    "temperature": 0.7,
}

NO_LLM_DEFAULTS = {"enabled": False, "max_piece": 0.9, "show": 1000, "check_max": 400}
# scope (2026-11-13, user: "another arm limiting cut tool to its own molecules only"): "any" = the 1 Nov arm, the model
# names any molecule it likes -- which is how it chains, 69-71% of its follow-up calls asking about a fragment the tool
# had just handed it in the same turn, building a multi-step route before answering. "self" offers a DIFFERENT tool that
# takes NO argument and answers only for the molecule being expanded (user: "new tool? that doesn't let it specify a
# molecule?"), so the chaining is impossible rather than refused -- no refusal path to confuse with "no splits found".
CUT_TOOL_DEFAULTS = {"enabled": False, "max_piece": 0.9, "show": 5, "check_max": 60, "scope": "any",
                     "wording": "default", "show_buyable": True}
# show_buyable (2026-11-15, user: "remove purchasability check here"). True = every finished arm. False drops the
# `buyable` field from the tool's result: the SPLITS are identical, the model is simply not told which pieces can
# already be bought. That annotation is the oracle's own termination test answered in advance and for free, on
# several candidates at once, while the baseline learns the same thing only by proposing a route and paying a
# budgeted call for the feedback. 41-46% of non-empty tool results contain a split whose every piece is buyable,
# so it is load-bearing often enough to confound "verified steps help" with "we told it when to stop".
# wording (2026-11-15, user: "remove 'the route checker accepts each split as written' and also make 'Call it when
# you are stuck' more prominent and imperative"). "default" = the wording every finished arm ran with, so those
# results stay comparable. "stuck" = two changes, both testable:
#   * the acceptance promise is REMOVED. The tool still returns only checker-accepted splits -- nothing about the
#     data changes -- but the model is no longer told so. This separates the VALUE of pre-verified steps from the
#     model's TRUST in a claim about them. Related precedent: arm adv9t raised advisor compliance from 5% to 42%
#     by changing trust framing and moved the outcome not at all (20/28 vs 21).
#   * "call it when you are stuck" moves to the FRONT as an instruction, with an explicit do-not-call case. The
#     measured problem it targets: the model calls the tool on 98-99% of turns, not when stuck, so the existing
#     trailing sentence is plainly not read as a gate.
# max_piece (2026-11-08, user chose option a): 1.01 = the 1 Nov arm as built (near-copies INCLUDED, the literal
# reading of "the FULL list"); 0.9 = rc's near-copy filter applied BEFORE the checker, so Qwen still chooses among
# every real cut. Measured (analysis/split_cost.py, job 9714802): 94-100% of candidates are near-copies, checker
# time per molecule 548 s -> 139 s with the filter; the selector prompt itself tells Qwen to avoid near-copies.
# show (2026-11-08, user: "top 25"): 10**6 = the 1 Nov arm as built (no early stop: every accepted split); 25 = stop
# once 25 accepted real cuts are found. The 1 Nov run passed ~110 splits per molecule to the selector and 58% of the
# splits it then handed the planner were near-copies; 25 keeps a genuine choice (5x what rc shows) at about rc's cost.
CUT_SELECT_DEFAULTS = {"enabled": False, "pick": 5, "check_max": 400, "max_piece": 1.01, "show": 10 ** 6}

# Config `aot.step_quality` (arm stepq, 2026-11-18; absent or disabled = the released score, unchanged).
#
# The search's only test of a step is that the route checker accepts it, so a step that shaves 3% off a
# molecule is scored like one that halves it, and the branch built from trimming keeps looking as
# attractive as a real one. Measured on the two targets that failed in slow15_ask2: 176 and 143 ACCEPTED
# steps, trees 20 and 16 deep, 93 and 78 distinct molecules, and no route -- against a median of 21
# expansions, 10 molecules and depth 5 on the targets that solved. Those searches were not stuck; every
# step was legal and none was progress.
#
# `near` is deliberately the same 0.9 that `disconnections` uses to decide a template disconnection is a
# real cut (cut_select.max_piece): the model's own steps are now held to the standard the enumerated ones
# already are. A step whose largest piece is >= `near` of the molecule has its score multiplied by
# `penalty`, which lowers the whole branch's UCB value and makes the search drift away from it on its
# own -- no LLM call, no advice, and it applies to EVERY expansion rather than the ~3% an advisor is
# asked about (see the arm `ask` / `ask2` entries in docs/experiment_history.md).
STEP_QUALITY_DEFAULTS = {"enabled": False, "near": 0.9, "penalty": 0.5}

# Config `aot.branch_quality` (arm branchq, 2026-11-20; absent or disabled = the released score, unchanged).
#
# WHY THIS EXISTS: arm stepq (above) was the WRONG UNIT, and its own run says so. It finished 13/15,
# matched 0W/0L against cutselR -- it changed every route (all 13 differed, net -7 steps, which is noise)
# but moved no molecule across the line, despite the penalty firing 502 times.
#
# The reason is selectivity. stepq judges ONE STEP in isolation: largest piece >= 0.9 of its parent. That
# fires on 54% of every step taken and on 48% of the steps inside searches that go on to SUCCEED -- a
# 1.3x separation (64% of steps in failed searches vs 48% in solved ones). Halving the score of half the
# steps in a winning search is closer to a uniform rescaling than to a judgement. Worse, its largest
# single category (36.6% of all steps) is steps where the molecule GREW going backwards, which is
# ordinary retrosynthesis: a deprotection read in reverse adds the group back.
#
# The feature that actually separated failure from success in the rule mining (3 runs, n=1012, AUC 0.709,
# monotone across 7 bands 18% -> 73% on-path, and the root split of a held-out decision tree in all 3
# folds) is CUMULATIVE and per-BRANCH, not per-step:
#
#     shrink_rate = (1 - heavy(molecule)/heavy(target)) / depth
#
# i.e. the share of the TARGET's mass removed per step taken SO FAR. `shrink_rate <= 0.06` fires on 68%
# of failed-target expansions against 23% of solved ones -- a 3.0x separation, vs stepq's 1.3x. A run of
# individually ordinary small steps that is collectively going nowhere is exactly what this catches and
# what a per-step threshold cannot see, by construction.
#
# WHAT IT DOES: for the largest UNSOLVED piece a step produces, at that piece's own depth, compute
# shrink_rate against the target. If it is <= `floor`, multiply the step's score by `penalty`. Negative
# values (the piece is bigger than the target) are penalised by the same test without a special case --
# `grows_past_the_target` is the worst-behaved group in the mined data (14.1% on-path, 2.8x coverage).
#
# `min_depth` exists because the ratio is unstable where it has had no room to move: at depth 1 a single
# legitimate deprotection can look like a stall. The mining excluded depth 0 outright (the target is
# trivially on its own route; excluding it moved the base rate 52.0% -> 41.6%).
#
# WHAT THIS IS NOT: not an advisor (no LLM call, no prompt), and not a frontier re-ranking. It is the same
# intervention point as stepq -- `_score` -- with the measurement computed at the unit the evidence was
# actually gathered. The counters `branchq_seen` / `branchq_trim` record reach and firing rate so selectivity
# can be read off the run itself rather than re-derived.
BRANCH_QUALITY_DEFAULTS = {"enabled": False, "floor": 0.06, "penalty": 0.5, "min_depth": 2}

# Config `aot.reopen` (arm reopen, 2026-11-18; absent or disabled = released behaviour). Full reasoning,
# and the measurements behind every number here, in reopen.py.
#   after_calls     the threshold: the first ask/reopen happens once this many LLM calls have been spent
#                   with the target still unsolved, and the k-th at after_calls * k.
#   max_reopens     how many extra openings may be forced in total.
#   min_budget_left a reopen needs MORE than this many calls still unspent, or it cannot finish anything.
#   decide          "counter" = reopen on the threshold alone; "llm" = ask the model at the threshold and
#                   let it decline (one advisor call, not charged to the budget, as elsewhere here).
#   openings        how many fresh opening moves to generate per reopen (arm reopenP, 2026-11-19, user:
#                   "ask for 3 more different molecules to start with"). Each costs a root call.
#   choose          who picks which of them the search pursues next: "none" leaves the search's own
#                   ordering alone; "llm" asks the advisor, which is shown each candidate in the units
#                   the situational skills are defined over, plus the skills themselves.
REOPEN_DEFAULTS = {"enabled": False, "after_calls": 100, "max_reopens": 2, "min_budget_left": 50,
                   "decide": "counter", "show_tried": 6, "openings": 1, "choose": "none",
                   # arm reopenG (2026-11-20, user: "try it, protect it"): guaranteed selections inside the
                   # reopened opening's subtree, 0 = released selection (what reopenC/L/P ran with).
                   #
                   # WHY. reopenC/L/P all LOST molecules and never won one, and the logs say the mechanism
                   # was never tested. Every loss was a target the reopen fired on (3/3, 2/2, 1/1 -- no
                   # loss without one), and after each reopen only 1-2% of the ~150 remaining expansions
                   # touched depth <= 1, where the new opening lived: targets 6/10/12 went from median
                   # depth 7/8/6 to 12/14/11 and reached depth 19/18/17 under the OLD branches. A fresh
                   # opening has no visit count, so burden-divided UCB hands it straight back to branches
                   # carrying 100 calls of accumulated value. The arm paid for an option the scorer then
                   # discarded -- so it spent budget and changed nothing, which is exactly how it lost
                   # three molecules the control solved at 136, 34 and 216 calls.
                   #
                   # 30 is ~12% of the 250-call budget and about twice the median calls-to-solve on this
                   # dataset (cutselR median 6, p90 31), so a viable opening has a realistic chance to
                   # finish inside the quota. The quota is dropped the moment the subtree has nothing
                   # expandable, so it can never stall the search.
                   "protect": 0}

# Config `aot.split_pool` (2026-11-11; off by default = the work runs in-process exactly as before, split_pool.py).
# Job 9717221: the split search does not run in parallel at all -- per-item latency 106.6 s at 1 thread vs 3702.7 s
# at 28 (34.7x), throughput no better than one thread, because the scan and the checker hold the GIL and our 28
# workers are threads in ONE process. `enabled` forks `processes` children once, before the worker threads start,
# and each molecule's search is answered by one of them while the calling thread waits in a syscall that releases
# the GIL. `wait_s` 0 = when every child is busy the caller does the work itself rather than queue behind them.
SPLIT_POOL_DEFAULTS = {"enabled": False, "processes": 4, "timeout_s": 900, "wait_s": 0.0}


def split_pool_settings(aot, config):
    """The merged `aot.split_pool` settings, validated. Raises ValueError on anything that cannot work OR that
    would let a pooled verdict differ from the in-process one. main.py calls this at startup, so a bad config
    stops the job before the data load instead of failing inside every split search (review, 12 Nov).

    Why the two extra conditions: a child checks routes against its fork-time copy of `visited_molecules` /
    `dead_molecules`, which `rank_reactants` uses to order a template's outcomes. `prefer_model_reactants` moves
    the model's own outcome to the front whenever the template produces it, so that order cannot change whether a
    step is "accepted as written"; switch choice reads and writes per-call state the child does not share."""
    aot = aot if isinstance(aot, dict) else {}
    config = config if isinstance(config, dict) else {}
    blk = aot.get("split_pool") or {}
    unknown = set(blk) - set(SPLIT_POOL_DEFAULTS)
    if unknown:
        raise ValueError(f"aot.split_pool: unknown keys {sorted(unknown)}")
    c = dict(SPLIT_POOL_DEFAULTS, **blk)
    if int(c["processes"]) < 1 or float(c["timeout_s"]) <= 0 or float(c["wait_s"]) < 0:
        raise ValueError("aot.split_pool needs processes >= 1, timeout_s > 0 and wait_s >= 0")
    if c["enabled"]:
        if not (config.get("prefer_model_reactants") or config.get("prefer_reproducing_template")):
            raise ValueError("aot.split_pool needs prefer_model_reactants: true -- without it the checker's "
                             "verdict depends on visited/dead molecules a pool child does not share")
        if (aot.get("switch_choice") or {}).get("enabled"):
            raise ValueError("aot.split_pool cannot run with aot.switch_choice: the checker then uses per-call "
                             "switch state a pool child does not share")
    return c

# Arm steer's "one step from buyable" probe checks at most this many all-buyable candidates (_steer_cuts). The pool
# pre-checks the SAME number in the child, so the caller's loop is all cache hits; one constant, so they cannot drift.
STEER_BUYABLE_MAX = 10
# Config `aot.mol_tool` (arm moltool, 2026-11-19; absent or disabled = baseline). A no-argument tool that
# answers, for the molecule the planner is working on RIGHT NOW, with deterministic RDKit facts about it --
# formula, mass, element counts, ring system (sizes, aromaticity, fused/spiro/bridgehead, macrocycle),
# stereocentres assigned and unassigned, rotatable bonds, H-bond donors/acceptors, TPSA, logP, sp3 fraction,
# complexity, functional groups with counts, and the Murcko scaffold.
#
# WHY IT EXISTS. Arm ctoolself gave the planner the same no-argument shape but answered with the checker's
# accepted DISCONNECTIONS, and it beat the control by +13 matched (+10 on the replicate). What that does not
# tell us is whether the gain came from the tool mechanism -- context the model cannot compute, delivered on
# demand -- or from the oracle's own disconnections being handed back. This arm holds the mechanism fixed and
# removes every disconnection, template, precedent and purchasability fact, leaving only what RDKit can state
# about the molecule as it stands. If it moves the result, computed context is what matters; if it does not,
# the splits were doing the work. Deliberately paired with ctoolself, not with cutselR.
# `image` adds a second tool, see_this_molecule(), that draws the molecule and hands the planner the
# picture (2026-11-19, user: "allow the model to also pull the image of the molecule via tools"). It needs the
# vision encoder -- the server must run WITHOUT --text-only -- and llm.multimodal true. NOTE the prior: on the
# 9B runs images did not help and UNLABELLED images hurt by ~30%, so the pictures here are LABELLED
# (build_labeled_image_content_items) and are PULLED on demand rather than pushed into every prompt, which is
# the one configuration those runs never tested.
# Arm `mine` (2026-11-22, user: "a mechanism that calls the template repo and passes back the relevant ones as
# skills"). A SECOND self-scoped tool beside the disconnection tool, asking the opposite question. The
# disconnection tool answers "which bonds of this molecule can be broken", ranked by largest_piece ASCENDING and
# filtered at max_piece 0.9, so it can only ever return steps that make the molecule SMALLER. This one answers
# "how do the features of this molecule get MADE", which reaches the two classes that ranking excludes by
# construction: ring-forming steps (precursor LARGER than the product -- 12.9% of the corpus, 95% of those
# larger) and functional-group interconversions (precursor the SAME size). It runs no checker, suggests no
# disconnection and exposes no purchasability, so it carries the same leakage argument as the disconnection
# tool. Cost: a substructure scan plus dict lookups, 0.5 ms per node measured, against a few hundred ms for the
# disconnection tool -- 3.2 s for all 6,353 nodes of a 100-target run, so it is free at search scale.
# `wording` (2026-11-23, user: "update the how this is made tool to match stuck wording of cuttool"). Same tool,
# same facts; only what the model is TOLD differs, exactly as cut_tool.wording does. "default" = the wording arm
# `mine` ran with, kept so hard100_mine stays comparable. "stuck" leads with the imperative gate and an explicit
# do-NOT-call case, and drops the "Free: not charged against your budget" line -- measured 2026-11-23, the
# ungated wording is called on 93.8% of first attempts and its rate per planner call is flat (0.90-1.01) across
# every difficulty band, i.e. reflexively rather than on demand. The same rewrite took the disconnection tool
# from 98-99% of turns to 66%.
MINE_TOOL_DEFAULTS = {"enabled": False, "derive_skills": False, "max_new_skills": 40, "wording": "default",
                      # ADMISSION 2 (one skill per rare motif) as a switch, default ON so every arm run so
                      # far keeps its exact behaviour. Off, several molecules may each contribute a skill
                      # about the SAME motif: the corpus gets redundant entries that a later consolidation
                      # pass merges, which is the right trade when the rules cap the harvest at one skill
                      # per motif and the dataset only has 19 rare motifs across 100 targets (2026-11-24,
                      # user: "allow more than 1 please, we can combine via dream later").
                      "dedupe_motifs": True}

MINE_TOOL_SCHEMA = {"type": "function", "function": {
    "name": "how_is_this_made",
    "description": ("How the parts of the molecule you are disconnecting right now are MADE. Takes no argument "
                    "and always answers for that one molecule. For each notable motif it carries -- a ring "
                    "system, a linkage, a functional group -- it says how often recorded reactions involve that "
                    "motif, and which groups their starting materials carry more often than chance. A motif "
                    "flagged rare is where the unusual chemistry in your molecule is. It may also report that "
                    "nothing known closes a ring system, which means that system has to be built up a ring at a "
                    "time rather than cut across. It suggests NO disconnection, says nothing about whether a "
                    "route is accepted, and does not submit anything. Free: not charged against your budget."),
    "parameters": {"type": "object", "properties": {}, "required": []}}}

MINE_TOOL_NOTE = (
    "\nTOOL AVAILABLE: how_is_this_made(). It tells you how the parts of the molecule you are disconnecting are "
    "MADE -- which of its motifs are common and which are rare, what the starting materials of those reactions "
    "usually carry, and whether anything known closes its ring systems at all. A rare motif is where the "
    "unusual chemistry is. It suggests no disconnection and submits nothing; what the facts imply for your "
    "route is for you to judge. Calling it is free and does not count against your budget.\n")

# `wording: stuck`. Identical facts, gated description -- modelled line for line on CUT_TOOL_SCHEMA_SELF_STUCK:
# the gate leads as an imperative, there is an explicit do-NOT-call case, and the free-of-charge invitation is
# gone (it is still free; the model is simply no longer urged to spend it).
MINE_TOOL_SCHEMA_STUCK = {"type": "function", "function": {
    "name": "how_is_this_made",
    "description": ("CALL THIS ONLY WHEN YOU ARE STUCK on how to disconnect the molecule you were asked to "
                    "decompose -- you cannot see a disconnection you trust, or your earlier proposals for it were "
                    "rejected. If you can already see a good disconnection, do NOT call it: answer directly. It "
                    "takes no argument and always answers for that one molecule, and tells you how its parts are "
                    "MADE. For each notable motif it carries -- a ring system, a linkage, a functional group -- it "
                    "says how often recorded reactions involve that motif, and which groups their starting "
                    "materials carry more often than chance. A motif flagged rare is where the unusual chemistry in "
                    "your molecule is. It may also report that nothing known closes a ring system, which means that "
                    "system has to be built up a ring at a time rather than cut across. It suggests NO "
                    "disconnection, says nothing about whether a route is accepted, and does not submit anything."),
    "parameters": {"type": "object", "properties": {}, "required": []}}}

MINE_TOOL_NOTE_STUCK = (
    "\nTOOL AVAILABLE: how_is_this_made(). CALL IT ONLY WHEN YOU ARE STUCK on how to disconnect the molecule you "
    "were asked to decompose -- you cannot see a disconnection you trust, or your earlier proposals for it were "
    "rejected. If you can already see a good route, do NOT call it: answer directly. It tells you how the parts of "
    "that molecule are MADE -- which of its motifs are common and which are rare, what the starting materials of "
    "those reactions usually carry, and whether anything known closes its ring systems at all. A rare motif is "
    "where the unusual chemistry is. It suggests no disconnection and submits nothing; what the facts imply for "
    "your route is for you to judge. Your final answer must still be a route in the <ROUTE> format above.\n")

MOL_TOOL_DEFAULTS = {"enabled": False, "max_calls": 0, "image": False, "max_images": 2}

MOL_TOOL_SCHEMA = {"type": "function", "function": {
    "name": "describe_this_molecule",
    "description": ("Deterministic chemical facts about the molecule you are being asked to disconnect right "
                    "now: formula, molecular weight, element counts, every ring with its size and whether it "
                    "is aromatic, fused/spiro/bridgehead atoms, macrocycles, stereocentres (assigned and "
                    "unassigned), rotatable bonds, hydrogen-bond donors and acceptors, TPSA, logP, the "
                    "fraction of sp3 carbon, a complexity index, the functional groups present with their "
                    "counts, and the Murcko scaffold. All computed from the structure -- it contains no "
                    "disconnection, no reaction, no precedent and no purchasability information. Takes no "
                    "arguments. Free: it is not charged against your call budget."),
    "parameters": {"type": "object", "properties": {}, "required": []}}}

IMAGE_TOOL_SCHEMA = {"type": "function", "function": {
    "name": "see_this_molecule",
    "description": ("Draw the molecule you are being asked to disconnect right now and show it to you as a "
                    "labelled 2-D structure diagram. Useful when the SMILES is long and you want to see the "
                    "ring system, the stereochemistry or where the substituents sit. Takes no arguments. "
                    "Free: it is not charged against your call budget."),
    "parameters": {"type": "object", "properties": {}, "required": []}}}

IMAGE_TOOL_NOTE = (
    "\nTOOL AVAILABLE: see_this_molecule(). It draws the molecule you are disconnecting and shows you the "
    "structure as a picture. Use it when the SMILES is hard to read -- a large ring system, dense "
    "stereochemistry, many substituents. Calling it is free and does not count against your budget.\n")

MOL_TOOL_NOTE = (
    "\nTOOL AVAILABLE: describe_this_molecule(). It returns deterministic chemical facts about the molecule "
    "you are disconnecting -- its formula and mass, its ring system, where its stereocentres are, which "
    "functional groups it contains and how many of each, its scaffold, and several computed properties. It "
    "suggests nothing: there are no disconnections or reactions in the answer, and what the facts imply for "
    "the retrosynthesis is for you to judge. Calling it is free and does not count against your budget.\n")

CUT_TOOL_SCHEMA = {"type": "function", "function": {
    "name": "get_known_splits",
    "description": ("Known one-step splits of a molecule, found by applying recorded USPTO reactions and reaction "
                    "templates to it. The route checker accepts each split as written. Sorted with the biggest "
                    "simplification first; pieces that can be bought are marked. Call it when you are stuck."),
    "parameters": {"type": "object", "properties": {
        "smiles": {"type": "string", "description": "SMILES of the molecule you want to split"}},
        "required": ["smiles"]}}}
CUT_TOOL_NOTE = (
    "\nTOOL AVAILABLE: get_known_splits(smiles). If you are stuck on how to disconnect a molecule -- you cannot "
    "see a disconnection you trust, or your earlier proposals for it were rejected -- you may call it. It returns "
    "known one-step splits of that molecule, found by applying recorded USPTO reactions; the route checker accepts "
    "each as written; they are sorted with the biggest simplification first, and pieces that can be bought are "
    "marked. Calling it is optional and costs nothing. If you can already see a good route, answer directly. "
    "Your final answer must still be a route in the <ROUTE> format above.\n")
# Arm ctoolself (2026-11-13): the same tool with its argument removed, so it can only ever answer for the molecule
# being decomposed. Wording stays as close to CUT_TOOL_NOTE as the missing argument allows.
CUT_TOOL_SCHEMA_SELF = {"type": "function", "function": {
    "name": "get_known_splits_of_this_molecule",
    "description": ("Known one-step splits OF THE MOLECULE YOU ARE DECOMPOSING, found by applying recorded USPTO "
                    "reactions and reaction templates to it. The route checker accepts each split as written. Sorted "
                    "with the biggest simplification first; pieces that can be bought are marked. Takes no argument: "
                    "it always answers for the molecule you were asked to decompose. Call it when you are stuck."),
    "parameters": {"type": "object", "properties": {}}}}
# Arm cuttool/ctoolself, `wording: stuck` (2026-11-15). Same tool, same data; only what the model is told
# differs. The acceptance promise is gone and the gate leads.
CUT_TOOL_SCHEMA_STUCK = {"type": "function", "function": {
    "name": "get_known_splits",
    "description": ("CALL THIS ONLY WHEN YOU ARE STUCK on how to disconnect a molecule -- you cannot see a "
                    "disconnection you trust, or your earlier proposals for it were rejected. If you can already "
                    "see a good disconnection, do NOT call it: answer directly. It returns known one-step splits "
                    "of the molecule you name, found by applying recorded USPTO reactions and reaction templates "
                    "to it, sorted with the biggest simplification first; pieces that can be bought are marked."),
    "parameters": {"type": "object", "properties": {
        "smiles": {"type": "string", "description": "SMILES of the molecule you want to split"}},
        "required": ["smiles"]}}}
CUT_TOOL_SCHEMA_SELF_STUCK = {"type": "function", "function": {
    "name": "get_known_splits_of_this_molecule",
    "description": ("CALL THIS ONLY WHEN YOU ARE STUCK on how to disconnect the molecule you were asked to "
                    "decompose -- you cannot see a disconnection you trust, or your earlier proposals for it were "
                    "rejected. If you can already see a good disconnection, do NOT call it: answer directly. It "
                    "takes no argument and always answers for that one molecule. It returns known one-step splits "
                    "of it, found by applying recorded USPTO reactions and reaction templates, sorted with the "
                    "biggest simplification first; pieces that can be bought are marked."),
    "parameters": {"type": "object", "properties": {}}}}
CUT_TOOL_NOTE_STUCK = (
    "\nTOOL AVAILABLE: get_known_splits(smiles). CALL IT ONLY WHEN YOU ARE STUCK -- you cannot see a "
    "disconnection you trust, or your earlier proposals for that molecule were rejected. If you can already see a "
    "good route, do NOT call it: answer directly. It returns known one-step splits of the molecule you name, found "
    "by applying recorded USPTO reactions; they are sorted with the biggest simplification first, and pieces that "
    "can be bought are marked. Calling it costs nothing. Your final answer must still be a route in the <ROUTE> "
    "format above.\n")
CUT_TOOL_NOTE_SELF_STUCK = (
    "\nTOOL AVAILABLE: get_known_splits_of_this_molecule(). CALL IT ONLY WHEN YOU ARE STUCK on how to disconnect "
    "the molecule you were asked to decompose -- you cannot see a disconnection you trust, or your earlier "
    "proposals for it were rejected. If you can already see a good route, do NOT call it: answer directly. It "
    "takes no argument and always answers for that one molecule: it cannot tell you anything about the pieces it "
    "returns. It gives known one-step splits of that molecule, found by applying recorded USPTO reactions; they "
    "are sorted with the biggest simplification first, and pieces that can be bought are marked. Calling it costs "
    "nothing. Your final answer must still be a route in the <ROUTE> format above.\n")
CUT_TOOL_NOTE_SELF = (
    "\nTOOL AVAILABLE: get_known_splits_of_this_molecule(). If you are stuck on how to disconnect the molecule you "
    "were asked to decompose -- you cannot see a disconnection you trust, or your earlier proposals for it were "
    "rejected -- you may call it. It takes no argument and always answers for that one molecule: it cannot tell you "
    "anything about the pieces it returns. It gives known one-step splits of that molecule, found by applying "
    "recorded USPTO reactions; the route checker accepts each as written; they are sorted with the biggest "
    "simplification first, and pieces that can be bought are marked. Calling it is optional and costs nothing. If "
    "you can already see a good route, answer directly. Your final answer must still be a route in the <ROUTE> "
    "format above.\n")


def _one_step_route(smiles, pieces, why):
    """A route in the planner's own step format, so the no-LLM arm goes through the same mapping and checks."""
    return [{"Molecule set": repr([smiles]), "Rational": why, "Product": repr([smiles]),
             "Reaction": repr([f"{smiles}>>{'.'.join(pieces)}"]), "Reactants": repr(list(pieces)),
             "Updated molecule set": repr(list(pieces))}]


# Config `aot.switch_choice` (2026-10-15; off by default = upstream silent switch). See switch_choice.py.
SWITCH_DEFAULTS = {
    "enabled": False,
    "max_options": 3,            # near-miss database reactions offered (user decision: "different" never offered)
    "history": True,             # annotate the tried list + show the route with who wrote each step
    "temperature": 0.7,          # the choice call (free: NOT counted in llm_call_budget, user decision)
}


def _fp(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return AllChem.GetMorganFingerprint(m, 2, useFeatures=False) if m is not None else None


class AOTOptimizer(planning_Optimizer):

    def __init__(self, args=None, config=None, shared_data=None):
        super().__init__(args=args, config=config, shared_data=shared_data)
        cfg = dict(AOT_DEFAULTS)
        cfg.update((self.llm_config.get("aot") or {}))
        self.aot = cfg
        if self._adv_cfg()["enabled"]:
            self._adv_make_pool()

    # ------------------------------------------------------------------ helpers
    def _purchasable(self, canon_smi):
        try:
            return bool(self.inventory.is_purchasable(Molecule(canon_smi)))
        except Exception:
            return False

    def _new_tree(self, target):
        return AndOrTree(target, canon=sanitize_smiles, purchasable=self._purchasable,
                         c_param=self.aot["c_param"],
                         unsolved_penalty=self.aot["unsolved_penalty"],
                         max_depth=self.aot["max_depth"],
                         max_reexpansions=self.aot["max_reexpansions"],
                         backprop=self.aot["backprop"],
                         depth_penalty=self.aot["depth_penalty"],
                         propagate_failure=bool(self._term_cfg()["propagate_failure"]),
                         propagate_from=self._term_cfg()["propagate_from"])

    def _sample_examples(self, smiles):
        """Baseline RAG, applied to the molecule being expanded: 3 train/val routes
        sampled without replacement with probability proportional to Tanimoto
        similarity (as in planning_Optimizer.initialization)."""
        self._last_rag = None
        fp = _fp(smiles)
        if fp is None:
            return "\n"
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, self._rag_all_fps), dtype=float)
        cand = np.arange(len(sims))
        k = int(self._current_config.get('rag_top_k') or 0) if getattr(self, '_current_config', None) else 0
        if k and k < len(sims):   # RAG fix: only the k nearest routes (as the AOT* release, top 50)
            cand = np.argpartition(-sims, k)[:k]
        w = sims[cand]
        total = w.sum()
        if total <= 0:
            return "\n"
        idx = cand[np.random.choice(len(cand), p=w / total, size=min(3, len(cand)), replace=False)]
        # fix-effect logging: similarity of the examples shown vs the best available
        self._last_rag = {"rag_sim_shown": round(float(sims[idx].mean()), 3),
                          "rag_sim_best": round(float(sims.max()), 3)}
        examples = ""
        for i in idx:
            examples += "<ROUTE>\n" + str(process_reaction_routes(self._rag_route_list[i])) + "\n</ROUTE>\n"
        return examples + "\n"

    # ------------------------------------------------------------------ advisors
    def _adv_cfg(self):
        return dict(ADVISOR_DEFAULTS, **((self.aot.get("advisor") or {}) if isinstance(self.aot, dict) else {}))

    def _adv_make_pool(self):
        c = self._adv_cfg()
        self._adv_pool = adv.AdvisorPool(c["advisors"], max_tokens=c["max_tokens"], timeout=c["timeout_s"])

    def _adv_state(self):
        """Per-molecule advisor state (checkpointed): attempt history per molecule, advice given,
        and who supplied each tree edge (chemist vs template database)."""
        return self.__dict__.setdefault("_adv", {"hist": {}, "advice": {}, "src": {}})

    def _adv_path(self, smiles):
        """Steps from the target down to `smiles` (first-found parent chain), target first."""
        tree, steps, seen = self._aot_tree, [], set()
        node = tree.molecules.get(smiles)
        src = self._adv_state()["src"]
        while node is not None and node.smiles not in seen and node.parent_reactions:
            seen.add(node.smiles)
            a = node.parent_reactions[0]
            steps.append({"product": a.product, "reactants": list(a.reactants),
                          "source": adv.SRC_TEMPLATE if src.get(a.key) == "template" else adv.SRC_CHEMIST})
            node = a.parent_molecule
        return steps[::-1]

    # ------------------------------------------------------------------ switch choice
    def _sw_cfg(self):
        return dict(SWITCH_DEFAULTS, **((self.aot.get("switch_choice") or {}) if isinstance(self.aot, dict) else {}))

    def _sw_state(self):
        """Who wrote each tree edge (checkpointed): reaction key -> {src: own | chosen | template,
        orig: the model's own step-1 reaction when the database supplied the edge}."""
        return self.__dict__.setdefault("_sw", {"src": {}})

    def _sw_path(self, smiles):
        """Steps from the target down to `smiles` (first-found parent chain) with their authors."""
        tree, steps, seen = self._aot_tree, [], set()
        node = tree.molecules.get(smiles)
        src = self._sw_state()["src"]
        while node is not None and node.smiles not in seen and node.parent_reactions:
            seen.add(node.smiles)
            a = node.parent_reactions[0]
            rec = src.get(a.key) or {}
            steps.append({"product": a.product, "reactants": list(a.reactants),
                          "src": rec.get("src", sc.SRC_OWN), "orig": rec.get("orig")})
            node = a.parent_molecule
        return steps[::-1]

    def _switch_choose(self, tree, or_node, depth, snap):
        """Config `aot.switch_choice`: step 1 of the answer just mapped would have been switched
        silently upstream. Offer the near-miss database reactions (if any) and let the model pick one
        or NONE, in ONE free follow-up call (not counted in llm_call_budget); the pick is attached
        exactly as the silent switch would attach it. Everything goes to aot_switch_<pid>.jsonl and
        fix_stats (switch_*)."""
        pend = getattr(self, "_switch_pending", None)
        self._switch_pending = None
        if not pend or sanitize_smiles(pend["product"]) != or_node.smiles:
            return []
        c = self._sw_cfg()
        self._fx_bump("switch_events")
        silent = pend.get("silent")
        if silent:
            self._fx_bump("switch_silent_close" if silent["label"] in sc.CLOSE else "switch_silent_different")
        # offer only options that would really attach (dry run; the step can still fail for other
        # reasons, e.g. an invalid SMILES elsewhere in it: probe 2026-10-15, 11/24 picks did not attach)
        near = pend["options"]
        opts = [o for o in near if self.switch_option_attaches([or_node.smiles], snap, pend["product"], o)]
        opts = opts[:int(c["max_options"])]
        entry = {"molecule": or_node.smiles, "reason": pend["reason"], "reaction": pend["reaction"],
                 "reactants": pend["reactants"], "options": opts, "near_misses": len(near), "silent": silent,
                 "scanned": pend.get("scanned"), "applicable": pend.get("applicable"),
                 "different": pend.get("different"), "prompt": None, "raw": None, "choice": None,
                 "attached": 0}
        created = []
        if not near:
            self._fx_bump("switch_no_close")
        elif not opts:
            self._fx_bump("switch_no_valid")
        elif self._time_left():
            step = snap[0] if snap else {}
            prompt = sc.choice_prompt(self._aot_tree.root.smiles, or_node.smiles, pend["reaction"],
                                      str(step.get("Rational", "")), pend["reason"], opts,
                                      self._sw_path(or_node.smiles) if c["history"] else [])
            self._fx_bump("switch_asked")
            _, answer = self.query_LLM(prompt, temperature=float(c["temperature"]))
            choice = sc.parse_choice(answer, len(opts))
            entry.update(prompt=prompt, raw=answer, choice=choice)
            if choice is None:
                self._fx_bump("switch_unparsed")
            elif choice == "NONE":
                self._fx_bump("switch_none")
            else:
                o = opts[choice - 1]
                self._fx_bump("switch_chosen")
                self._switch_forced = {"product": pend["product"], "key": o["key"], "template": o["template"],
                                       "reactants": o["reactants"], "orig": pend["reaction"]}
                try:
                    created = self._map_route(tree, or_node, copy.deepcopy(snap), depth)
                finally:
                    self._switch_forced = None
                    self._switch_pending = None
                entry["attached"] = len(created)
                if created:
                    self._fx_bump("switch_attached")
        self._log_switch(entry)
        return created

    def _log_switch(self, entry):
        try:
            entry = {"ts": _dt.utcnow().isoformat(timespec="seconds"),
                     "target_idx": getattr(self, "tb_target_idx", -1),
                     "worker": threading.current_thread().name, "llm_calls": self._aot_llm_calls, **entry}
            path = os.path.join(self.args.output_dir, f"aot_switch_{os.getpid()}.jsonl")
            with _APPEND_LOCK, open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[aot] switch log failed: {e}")

    @staticmethod
    def _heavy(smi):
        m = Chem.MolFromSmiles(smi) if smi else None
        return m.GetNumHeavyAtoms() if m is not None else 0

    @staticmethod
    def _rejected_step(snap):
        """arm retrytried: the first step of a rejected answer, in the same `product>>reactants`
        shape `tried` already uses, so the prompt's existing wording covers it. -> None when the
        answer had no readable first step (nothing useful to tell the next attempt)."""
        if not snap:
            return None
        step = snap[0]
        try:
            prod = (ast.literal_eval(step["Product"]) or [None])[0]
            rts = ast.literal_eval(step["Reactants"]) or []
        except Exception:
            return None
        if not prod or not rts:
            return None
        return f"{prod}>>{'.'.join(str(r) for r in rts)}"

    def _record_attempt(self, or_node, snap, created):
        """Remember this attempt (reaction, reasoning, the checker's verdict in words) for the
        advisors. No-op unless advisors are on."""
        # arm advsel also needs the per-molecule attempt history (the writing-vs-chemistry split is the
        # information try 13 showed matters most for judging whether a branch is exhausted). Still off
        # for the baseline, so the baseline is untouched.
        if not (self._adv_cfg()["enabled"]
                or self._cfg_of("advisor_select", ADV_SELECT_DEFAULTS)["enabled"]):
            return
        att = {"reaction": None, "reactants": None, "rational": "", "attached": bool(created)}
        s1 = getattr(self, "_last_step1", None)
        if getattr(self, "_last_route_outcome", None) != "ok" or not snap or s1 is None:
            att.update(verdict="unreadable", words="the answer had no readable route")
        else:
            step = snap[0]
            try:
                att["reaction"] = (ast.literal_eval(step["Reaction"]) or [None])[0]
                att["reactants"] = ast.literal_eval(step["Reactants"])
            except Exception:
                pass
            att["rational"] = str(step.get("Rational", ""))[:600]
            att["verdict"], att["words"] = adv.step1_verdict(
                step, bool(s1[0]), s1[1], getattr(self, "_last_final_reactants", None), sanitize_smiles)
        self._adv_state()["hist"].setdefault(or_node.smiles, []).append(att)

    def _advisor_block(self, smiles, question=""):
        """Notes for the molecule's next prompt ("" = none). Triggers advice when the molecule is
        stuck (>= stuck_fails READABLE attempts that attached nothing, or >= 1 readable attempt on
        a molecule at path length >= near_whole_depth that is still >= near_whole_size of the
        target); advice is free (not in llm_call_budget) and cached per molecule
        (<= max_per_molecule rounds). Unreadable answers (format failures, ~24% of calls) are shown
        to the advisors but do not make a molecule "stuck": the advisors cannot fix formatting."""
        self._last_advisor = None
        c = self._adv_cfg()
        if not c["enabled"]:
            return ""
        st = self._adv_state()
        hist = st["hist"].get(smiles, [])
        readable = [a for a in hist if a["verdict"] != "unreadable"]
        have = st["advice"].get(smiles)
        why = None
        if readable:
            fails = sum(1 for a in readable if not a["attached"])
            path = self._adv_path(smiles)
            root = self._aot_tree.root.smiles
            near = (len(path) >= c["near_whole_depth"] and
                    self._heavy(smiles) >= c["near_whole_size"] * max(self._heavy(root), 1))
            why = "fails" if fails >= c["stuck_fails"] else ("near_whole" if near else None)
        due = why and (have is None or (have["count"] < c["max_per_molecule"] and
                                        len(readable) - have["n_at"] >= c["refresh_after"]))
        if due and self._time_left():
            have = self._advise(smiles, hist, len(readable), path, why, c, question)
            self._last_advisor = {"advisor": "new", "advisor_trigger": why, "advisor_n": len(have["notes"])}
        elif have and have["notes"]:
            self._last_advisor = {"advisor": "reused", "advisor_n": len(have["notes"])}
        if not have or not have["notes"]:
            return ""
        self._fx_bump("advisor_notes_shown")
        if c["plan_of_attack"]:
            note = have["notes"][0][1]
            self._last_advisor.update(advisor_stance=note.get("stance"), advisor_verdict=note.get("verdict"))
            if getattr(self, "_plan9_shown", False):      # try 9: the plan is already in the prompt
                note = dict(note, plan=None)
            return al.render_block(note, trust=bool(c.get("trust_framing")))
        return adv.notes_block(have["notes"])

    def _advise(self, smiles, hist, n_readable, path, why, c, question=""):
        """One advice round: ask every advisor (in parallel), log everything, cache the notes."""
        if c["plan_of_attack"]:
            return self._advise_r8(smiles, hist, n_readable, path, why, c, question)
        from concurrent.futures import ThreadPoolExecutor
        if getattr(self, "_adv_pool", None) is None:
            self._adv_make_pool()
        item = {"target": self._aot_tree.root.smiles, "node": smiles, "path": path,
                "attempts": hist[-int(c["max_attempts_shown"]):]}
        names = list(c["advisors"])
        prompts = {n: adv.advisor_prompt(item, n) for n in names}
        self._fx_bump("advisor_triggered")
        with ThreadPoolExecutor(len(names)) as ex:
            futs = {n: ex.submit(self._adv_pool.ask, n, prompts[n]) for n in names}
            res = {n: f.result() for n, f in futs.items()}
        notes = []
        for n in names:
            r = res[n]
            self._fx_bump("advisor_calls")
            note = adv.parse_note(r["text"], r["finish"]) if r["text"] else {}
            usable = bool(note.get("review") or note.get("direction_clean"))
            self._fx_bump("advisor_usable" if usable else "advisor_failed")
            if usable:
                note["attempt_verdicts"] = [a["verdict"] for a in item["attempts"]]   # Attempt N = item's N-th
                notes.append((n, note))
            self._log_advisor(smiles, n, why, len(path), prompts[n], r, note)
        prev = self._adv_state()["advice"].get(smiles)
        rec = {"n_at": n_readable, "count": (prev["count"] if prev else 0) + 1, "notes": notes}
        self._adv_state()["advice"][smiles] = rec
        return rec

    def _log_advisor(self, smiles, name, why, path_len, prompt, r, note):
        try:
            entry = {"ts": _dt.utcnow().isoformat(timespec="seconds"),
                     "target_idx": getattr(self, "tb_target_idx", -1),
                     "worker": threading.current_thread().name, "molecule": smiles, "advisor": name,
                     "trigger": why, "path_len": path_len, "llm_calls": self._aot_llm_calls,
                     "prompt": prompt, "raw": r["text"], "finish": r["finish"], "tokens": r["tokens"],
                     "latency_s": r["latency_s"], "error": r["error"], **note}
            path = os.path.join(self.args.output_dir, f"aot_advisor_{os.getpid()}.jsonl")
            with _APPEND_LOCK, open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[aot] advisor log failed: {e}")

    # ------------------------------------------------------------------ joint co-design (try 11, arm joint)
    def _joint_due(self, smiles, c):
        """Is a joint discussion due for this route call? Root (if joint_at_root) or every joint_every calls."""
        st = self._adv_state().setdefault("joint", {"last": None})
        tree = getattr(self, "_aot_tree", None)
        if c["joint_at_root"] and tree is not None and smiles == tree.root.smiles and st["last"] is None:
            return True
        every = int(c["joint_every"] or 0)
        return every > 0 and (st["last"] is None or self._aot_llm_calls - st["last"] >= every)

    def _joint_route(self, smiles, question):
        """Arm joint: Qwen and Intern propose from the SAME prompt and converge. -> the agreed route block
        or None (not due, no usable proposal, or no time). All discussion calls are free: only the final
        route call that writes the agreed path counts toward llm_call_budget."""
        c = self._adv_cfg()
        if not (c["enabled"] and c["joint"]) or not self._joint_due(smiles, c) or not self._time_left():
            return None
        if getattr(self, "_adv_pool", None) is None:
            self._adv_make_pool()
        name = list(c["advisors"])[0]
        ask_adv = ap9.ask_advisor_fn(al.ask_messages, self._adv_pool, name, c["plan_max_tokens"],
                                     c["advisor_context"], self._estimate_message_tokens)
        ask_q = lambda msgs: ap9.qwen_chat(self, msgs, c["joint_qwen_max_tokens"])
        res, trace = jp.discuss(question, ask_q, ask_adv, extract_route_block, rounds=c["joint_rounds"],
                                system=ap9.QWEN_SYSTEM, shuffle=self._joint_shuffle,
                                context=c["advisor_context"], estimate=self._estimate_message_tokens,
                                reserve=c["joint_reserve"], keep_going=self._time_left)
        for e in trace:
            self._fx_bump("joint_calls")
            self._log_adv8(smiles, name, "joint", 0, e)
        self._adv_state()["joint"]["last"] = self._aot_llm_calls
        self._fx_bump("joint_discussions")
        self._fx_bump(f"joint_status_{res['status']}")
        if res["differs_from_both"]:
            self._fx_bump("joint_differs_from_both")
        self._last_joint = {"joint_status": res["status"], "joint_rounds": res["rounds"],
                            "joint_differs": res["differs_from_both"]}
        return res["route"] if res["usable"] else None

    def _joint_shuffle(self, pairs):
        """Hide which proposal is whose by position: a per-call coin flip (random, as the run's sampling is)."""
        return pairs if random.random() < 0.5 else list(reversed(pairs))

    # ------------------------------------------------------------------ plan-of-attack advisor (try 8)
    def _adv8_state(self):
        """Per-optimizer scratch of the try-8 advisor (not checkpointed): RetroSim outcomes per molecule and
        the template cache. The per-target plans live in the checkpointed advisor state under "plan"."""
        return self.__dict__.setdefault("_adv8", {"jx": {}, "outcomes": {}})

    def _adv8_routes(self, smi, target):
        """The 3 most similar train/val routes (never one whose product is the final target)."""
        fp = _fp(smi)
        if fp is None:
            return []
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, self._rag_all_fps), dtype=float)
        tcan, out = sanitize_smiles(target) or target, []
        for j in np.argsort(-sims)[:50]:
            if sanitize_smiles(self._rag_route_list[j][0].split(">>")[0]) == tcan:
                continue
            out.append((float(sims[j]), al.ap.route_steps(self._rag_route_list[j])))
            if len(out) == al.ap.N_ROUTES:
                break
        return out

    def _log_adv8(self, smiles, name, why, path_len, entry):
        r = dict(entry["r"])
        for k in ("text", "finish", "tokens", "latency_s", "error"):
            r.setdefault(k, None)
        extra = {"tool_calls": r["tool_calls"]} if "tool_calls" in r else {}      # try 10
        self._log_advisor(smiles, name, why, path_len, entry["prompt"], r, {"stage": entry["stage"], **entry["parsed"], **extra})

    def _advise_r8(self, smiles, hist, n_readable, path, why, c, question):
        """One try-8 advice round (aot.advisor.plan_of_attack): the target's plan (once, at its first round;
        again at a later round only if the server gave no answer), facts for this molecule, then direction ->
        Qwen's reply (advisor_live.qwen_reply, as the probe) -> the advisor's answer. All free (outside
        llm_call_budget). Any failure leaves the molecule without advice; it never costs the search a call."""
        if getattr(self, "_adv_pool", None) is None:
            self._adv_make_pool()
        name = list(c["advisors"])[0]
        root = self._aot_tree.root.smiles
        st = self._adv_state()
        plans = st.setdefault("plan", {})
        w = self._adv8_state()
        tools = bool(c["tools"])
        ask = (lambda msgs: self._adv10_ask(name, msgs, smiles)) if tools else (lambda msgs: al.ask_messages(self._adv_pool, name, msgs))
        tf = {"transform": at.tool_prompt} if tools else {}                       # try 10: tools replace the facts
        similar = lambda smi: self._adv8_routes(smi, root)
        self._fx_bump("advisor_triggered")

        def facts_of(node, proposed):
            if tools:
                return None
            try:
                return al.node_facts(self, node, proposed, similar, w, sanitize_smiles, self._heavy)
            except Exception as e:
                print(f"[adv8] facts failed on {node}: {e}")
                self._fx_bump("adv8_facts_failed")
                return None

        if root not in plans or plans[root].get("error"):     # asked again only if the server gave no answer
            plan, trace = al.make_plan(root, facts_of(root, []), ask, **tf)
            plans[root] = plan
            self._fx_bump("advisor_calls")
            self._fx_bump("adv8_plan_usable" if plan["usable"] else "adv8_plan_failed")
            if plan["error"]:
                self._fx_bump("adv8_plan_error")
            for e in trace:
                self._log_adv8(root, name, "plan", 0, e)
        proposed = [x for a in hist for x in (a.get("reactants") or [])]
        item = {"target": root, "node": smiles, "path": path, "attempts": hist}
        facts = facts_of(smiles, proposed)
        extra = {"plan_for_qwen": None} if getattr(self, "_plan9_shown", False) else {}   # try 9: already in question
        note, trace = al.run_round(item, question, plans[root]["text"], facts, ask,
                                   lambda prompt: al.qwen_reply(self, prompt),
                                   trust=bool(c.get("trust_framing")), **extra, **tf)
        for e in trace:
            self._fx_bump("advisor_calls" if e["stage"] != "reply" else "adv8_replies")
            self._log_adv8(smiles, name, why, len(path), e)
        notes = []
        if note is not None:
            note["attempt_verdicts"] = [a["verdict"] for a in hist[-al.ap.MAX_ATTEMPTS:]]
            notes.append((name, note))
            self._fx_bump("advisor_usable")
            self._fx_bump("adv8_reply_usable" if note["reply"] else "adv8_reply_unusable")
            if note["reply"]:
                self._fx_bump("adv8_answer_usable" if note["answer"] else "adv8_answer_unusable")
        else:
            self._fx_bump("advisor_failed")
        prev = st["advice"].get(smiles)
        rec = {"n_at": n_readable, "count": (prev["count"] if prev else 0) + 1, "notes": notes}
        st["advice"][smiles] = rec
        return rec

    # ------------------------------------------------------------------ advisor with tools (try 10)
    def _adv10_ask(self, name, msgs, smiles):
        """One Intern call with tools (advisor_tools.ask_with_tools): no cap on tool calls; stops when Intern answers,
        when the molecule's time is up, or when the conversation no longer fits advisor_context. Each tool call is
        logged to aot_tools_<pid>.jsonl and counted (adv10_*)."""
        c = self._adv_cfg()
        cli, model = self._adv_pool._client(name)
        root = self._aot_tree.root.smiles
        stage = at.stage_of(msgs)
        runner = at.Runner(self, root, sanitize_smiles, self.__dict__.setdefault("_dc", {}))
        base = {"target_idx": getattr(self, "tb_target_idx", -1), "worker": threading.current_thread().name,
                "llm_calls": self._aot_llm_calls, "stage": stage, "molecule": root if stage == "plan" else smiles}
        path = os.path.join(self.args.output_dir, f"aot_tools_{os.getpid()}.jsonl")

        def log(tool, arguments, result, ok, seconds):
            self._fx_bump("adv10_tool_calls")
            self._fx_bump(f"adv10_tool_{tool}" if tool in {t["function"]["name"] for t in at.TOOLS} else "adv10_tool_unknown")
            if not ok:
                self._fx_bump("adv10_tool_errors")
            at.log_call(path, base, tool, arguments, result, ok, seconds)
        r = at.ask_with_tools(cli, model, msgs, runner, adv.MAX_TOKENS.get(name, c["max_tokens"]), c["advisor_context"],
                              self._time_left, log=log, sampling=adv.SAMPLING[name], extra=adv.extra_body(name))
        self._fx_bump("adv10_asks")
        if r["error"]:
            self._fx_bump("adv10_ask_errors")
        return r

    # ------------------------------------------------------------------ real cuts in Qwen's prompt (arm rc)
    def _rc_cfg(self):
        return dict(REAL_CUTS_DEFAULTS, **((self.aot.get("real_cuts") or {}) if isinstance(self.aot, dict) else {}))

    def _rag_cuts_block(self, smiles):
        """Arm ragcut (aot.rag_cuts): the first step of each of the `show` most similar RAG routes -- how molecules
        LIKE this one were disconnected. The baseline's own retrieval (self._rag_all_fps, built from the route
        targets, so the first step's product is exactly what the similarity was measured against), distilled to one
        line per route and ranked deterministically instead of sampled.

        Nothing here is applied to `smiles` or checked against it -- that is arm rc. A route whose first-step product
        IS this molecule is skipped: that would be the answer, not an analogy. Free (no LLM call, no checker run),
        cached per molecule. "" when there is nothing to show, so the prompt then equals the baseline's."""
        self._last_ragcut = None
        c = self._cfg_of("rag_cuts", RAG_CUTS_DEFAULTS)
        if not c["enabled"]:
            return ""
        fp = _fp(smiles)
        if fp is None or not getattr(self, "_rag_all_fps", None):
            return ""
        cache = self.__dict__.setdefault("_dc", {}).setdefault("ragcut", {})
        if smiles not in cache:
            t = time.time()
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, self._rag_all_fps), dtype=float)
            me, items, skipped = sanitize_smiles(smiles) or smiles, [], 0
            show, floor = int(c["show"]), float(c["min_sim"])
            for j in np.argsort(-sims):
                if len(items) >= show or float(sims[j]) < floor:
                    break
                steps = self._rag_route_list[j] if j < len(self._rag_route_list) else None
                first = steps[0] if steps else ""
                if ">>" not in (first or ""):
                    continue
                prod, reac = first.split(">>", 1)
                if (sanitize_smiles(prod) or prod) == me:     # the analogue IS the molecule: that is the answer
                    skipped += 1
                    continue
                items.append((float(sims[j]), prod, reac))
            cache[smiles] = (items, skipped, round(time.time() - t, 1))
        items, skipped, secs = cache[smiles]
        self._last_ragcut = {"ragcut_shown": len(items), "ragcut_self_skipped": skipped, "ragcut_s": secs,
                             "ragcut_sim_best": round(items[0][0], 3) if items else 0.0}
        if items:
            self._fx_bump("ragcut_block_shown")
        return dc.rag_cuts_block(smiles, items)

    def _real_cuts_block(self, smiles):
        """Arm rc (aot.real_cuts): up to `show` checker-accepted known steps for the molecule being decomposed that
        are not near-copies, biggest cut first (disconnections.best_steps), computed once per molecule; "" when there
        are none, so the prompt then equals the baseline's. The search is code, not an LLM call: free."""
        self._last_rc = None
        c = self._rc_cfg()
        if not c["enabled"]:
            return ""
        st = self.__dict__.setdefault("_dc", {})
        cache = st.setdefault("rc", {})
        t = time.time()
        if smiles not in cache:
            try:
                cache[smiles] = self._best_steps(smiles, c["max_piece"], c["show"], c["check_max"])
            except Exception as e:                     # the search must never cost the expansion its call
                print(f"[rc] search failed on {smiles}: {e}")
                self._fx_bump("rc_failed")
                cache[smiles] = ([], 0, 0)
            self._fx_bump("rc_searched")
        steps, checked, n = cache[smiles]
        self._last_rc = {"rc_shown": len(steps), "rc_checked": checked, "rc_candidates": n,
                         "rc_s": round(time.time() - t, 1)}
        if steps:
            self._fx_bump("rc_block_shown")
        return dc.real_cuts_block(smiles, steps)

    # ------------------------------------------------------------------ arm advsel (1 Nov)
    def _advisor_select(self, tree, iteration):
        """Arm advsel. Returns the AND node to expand (drive mode) or None (off, shadow, or any failure,
        in which case the search's own selection stands). Never raises."""
        self._last_sel = None
        c = self._cfg_of("advisor_select", ADV_SELECT_DEFAULTS)
        if not c["enabled"]:
            return None
        cands = [a for a in tree.leaf_and_nodes
                 if a.depth < tree.max_depth and not a.is_fully_solved()
                 and tree.has_expandable_reactants(a)]
        if len(cands) < 2:                      # nothing to choose between
            return None
        mode, order = str(c["mode"]), str(c["order"])
        orders = ["ranked", "shuffled"] if order == "both" else [order]
        if mode == "drive" and len(orders) > 1:
            raise ValueError("aot.advisor_select: order 'both' is shadow-only (only one can drive)")
        cuts = av.root_cuts(tree, 6)
        asked, drive_node = [], None
        for o in orders:
            rng = asel.rng_for(c["seed"], tree.root.smiles, iteration)
            hist = self._adv_state()["hist"] if c["rejects"] else None
            rows, nodes = asel.candidate_rows(tree, cands, int(c["rows"]), o, rng, hist)
            q = asel.question(tree.root.smiles, rows, len(cands), cuts, o)
            answer = None
            try:
                self._fx_bump(f"advsel_asked_{o}")
                _, answer = self.query_LLM(q, temperature=float(c["temperature"]))
            except Exception as e:              # a selection review must never cost the expansion
                print(f"[advsel] {o} review failed: {e}")
                self._fx_bump("advsel_failed")
            i, why, how = asel.parse_pick(answer, len(rows), o)
            rec = {"order": o, "parse": how, "why": why,
                   "ucb_rank_of_pick": rows[i]["rank"] if i is not None else None,
                   "shown_position": (i + 1) if i is not None else None,
                   "agrees_with_ucb": (rows[i]["rank"] == 1) if i is not None else None,
                   "product": rows[i]["product"] if i is not None else None}
            asked.append(rec)
            if mode == "drive" and i is not None:
                drive_node = nodes[i]
        self._last_sel = {"sel_iter": iteration, "sel_frontier": len(cands), "sel_shown": int(c["rows"]),
                          "sel_mode": mode, "sel_asks": asked}
        return drive_node

    # ------------------------------------------------------------------ arm advprune (1 Nov)
    def _prune_review(self, tree, iteration):
        """Arm advprune: ask which open branches are not worth expanding and de-prioritise those.

        Never removes a candidate and never raises: a failed or unparseable answer leaves the search
        exactly as it was. Marks from the previous ask are dropped first, so they expire by design."""
        c = self._cfg_of("advisor_prune", ADV_PRUNE_DEFAULTS)
        self._last_prune = None
        if not c["enabled"] or iteration % max(1, int(c["every"])) != 0:
            return
        tree.deprioritised = {}        # marks live for ONE review window, cleared even when nothing is asked
        rows, n_front = av.frontier_rows(tree, int(c["rows"]), marked=False)
        if n_front < int(c["min_frontier"]):
            return
        q = apr.question(tree.root.smiles, rows, n_front, av.root_cuts(tree, 6), int(c["max_skip"]))
        answer = None
        try:
            self._fx_bump("prune_asked")
            _, answer = self.query_LLM(q, temperature=float(c["temperature"]))
        except Exception as e:                       # a review failure must never cost the expansion
            print(f"[advprune] review failed: {e}")
            self._fx_bump("prune_failed")
        idx, reasons, how = apr.parse_skip(answer, len(rows), int(c["max_skip"]))
        by_score = [a for a in tree.leaf_and_nodes
                    if a.depth < tree.max_depth and not a.is_fully_solved()
                    and tree.has_expandable_reactants(a)]
        by_score.sort(key=lambda a: (tree.selection_score(a, marked=False), -a.depth, a.key), reverse=True)  # = select()
        marked = []
        for i in idx:
            if i < len(by_score):
                tree.deprioritised[by_score[i].key] = float(c["factor"])
                marked.append({"product": by_score[i].product, "reactants": list(by_score[i].reactants),
                               "depth": by_score[i].depth, "rank": i + 1, "why": reasons.get(i, "")})
        if marked:
            self._fx_bump("prune_marked")
        self._last_prune = {"prune_iter": iteration, "prune_frontier": n_front, "prune_shown": len(rows),
                            "prune_parse": how, "prune_n_marked": len(marked), "prune_marked": marked}

    # ------------------------------------------------------------------ arm steer (2026-11-08)
    def _steer_cfg(self):
        c = self._cfg_of("advisor_steer", STEER_DEFAULTS)
        if c["context"] not in ("local", "splits", "rc5"):
            raise ValueError(f"aot.advisor_steer.context must be local, splits or rc5, not {c['context']!r}")
        if c["enabled"] and c["allow_remove"] and not self._term_cfg()["propagate_failure"]:
            raise ValueError("aot.advisor_steer.allow_remove needs aot.termination.propagate_failure (REMOVE closes "
                             "the branch through failure propagation)")
        if int(c["after_calls"]) < 0:
            raise ValueError(f"aot.advisor_steer.after_calls must be >= 0, not {c['after_calls']!r}")
        return c

    # ------------------------------------------------------------------ arm `ask`: the pull advisor
    def _check_tool_cfg(self):
        return self._cfg_of("check_tool", CHECK_TOOL_DEFAULTS)

    def _check_tool_on(self):
        return bool(self._check_tool_cfg()["enabled"])

    def _validate_tool_on(self):
        return bool(self._cfg_of("validate_tool", VALIDATE_TOOL_DEFAULTS)["enabled"])

    def _cut_tool_schema(self, c):
        """The tool definition sent to the model. `wording` changes only what it is TOLD: the data
        returned is identical either way (see CUT_TOOL_DEFAULTS)."""
        stuck = str(c.get("wording", "default")) == "stuck"
        if c["scope"] == "self":
            return CUT_TOOL_SCHEMA_SELF_STUCK if stuck else CUT_TOOL_SCHEMA_SELF
        return CUT_TOOL_SCHEMA_STUCK if stuck else CUT_TOOL_SCHEMA

    def _cut_tool_note(self, c):
        stuck = str(c.get("wording", "default")) == "stuck"
        if c["scope"] == "self":
            return CUT_TOOL_NOTE_SELF_STUCK if stuck else CUT_TOOL_NOTE_SELF
        return CUT_TOOL_NOTE_STUCK if stuck else CUT_TOOL_NOTE

    def _ask_cfg(self):
        return self._cfg_of("advisor_ask", ASK_DEFAULTS)

    def _ask_open(self):
        """Is the advisor tool offered to the planner right now? Only once this target has had
        `dry_gate` consecutive expansions that attached nothing, and only while calls remain."""
        c = self._ask_cfg()
        if not c["enabled"]:
            return False
        if int(getattr(self, "_ask_calls", 0)) >= int(c["max_calls"]):
            return False
        return int(getattr(self, "_ask_dry", 0)) >= int(c["dry_gate"])

    def _ask_note_dry(self, attached):
        """Called once per LLM call from _log_expansion: any attach resets the stall counter.

        Guarded on `aot` because it now runs inside the expansion logger, which must never raise --
        the logger is also driven by test stubs and by partially built optimizers."""
        if not isinstance(getattr(self, "aot", None), dict) or not self._ask_cfg()["enabled"]:
            return
        self._ask_dry = 0 if attached else int(getattr(self, "_ask_dry", 0)) + 1

    def _ask_log(self, entry):
        try:
            path = os.path.join(self.args.output_dir, f"aot_ask_{os.getpid()}.jsonl")
            # `_dt` is the module's imported name (from datetime import datetime as _dt). Writing
            # `datetime.datetime.utcnow()` here raised NameError on EVERY call, and the except below
            # swallowed it -- so the first advisor run would have finished with an empty log.
            # target_idx: `tb_target_idx` is what every other log here uses. This read `_target_idx`, which
            # nothing sets, so job 9801161 logged target_idx=None on every advisor call (fixed 2026-11-16).
            entry = dict(ts=_dt.utcnow().isoformat(timespec="seconds"),
                         target_idx=getattr(self, "tb_target_idx", -1),
                         worker=threading.current_thread().name, **entry)
            with _APPEND_LOCK, open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[ask] log failed: {e}")

    # ------------------------------------------------------------------ arm reopen (see reopen.py)
    def _reopen_cfg(self):
        return self._cfg_of("reopen", REOPEN_DEFAULTS)

    def _reopen_log(self, entry):
        try:
            path = os.path.join(self.args.output_dir, f"aot_reopen_{os.getpid()}.jsonl")
            entry = dict(ts=_dt.utcnow().isoformat(timespec="seconds"),
                         target_idx=getattr(self, "tb_target_idx", -1),
                         worker=threading.current_thread().name, **entry)
            with _APPEND_LOCK, open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[reopen] log failed: {e}")

    def _reopen_due(self):
        """Is a reopen decision due now? The threshold steps on every ASK, so a declined ask is not
        re-asked immediately, and `max_reopens` / `min_budget_left` bound what can follow."""
        c = self._reopen_cfg()
        if not c["enabled"]:
            return False
        if int(getattr(self, "_reopens", 0)) >= int(c["max_reopens"]):
            return False
        budget = int(self.aot["llm_call_budget"])
        if budget - int(self._aot_llm_calls) <= int(c["min_budget_left"]):
            return False
        step = int(c["after_calls"])
        if step <= 0:
            return False
        return self._aot_llm_calls >= step * (int(getattr(self, "_reopen_asks", 0)) + 1)

    def _reopen_decide(self, tree):
        """True = force a new opening. `counter` always says yes; `llm` asks the model and defaults to no
        on any failure, because the arm must never be able to make things worse by accident."""
        c = self._reopen_cfg()
        self._reopen_asks = int(getattr(self, "_reopen_asks", 0)) + 1
        base = {"calls": self._aot_llm_calls, "asks": self._reopen_asks,
                "reopens": int(getattr(self, "_reopens", 0)), "decide": str(c["decide"])}
        if str(c["decide"]).lower() != "llm":
            self._reopen_log(dict(event="reopen", move="REOPEN", why="counter", status="ok", **base))
            return True
        openings = []
        for r in tree.root.child_reactions:
            try:
                openings.append(f"{tree.root.smiles}>>{'.'.join(m.smiles for m in r.child_molecules)}")
            except Exception:
                continue
        deepest = max((a.depth for a in tree.leaf_and_nodes), default=0)
        open_unsolved = sum(1 for a in tree.leaf_and_nodes
                            if not a.is_fully_solved() and tree.has_expandable_reactants(a))
        try:
            q = reopen.question(target=getattr(self, "_aot_target", "") or tree.root.smiles,
                                calls=self._aot_llm_calls, budget=int(self.aot["llm_call_budget"]),
                                reopens=int(getattr(self, "_reopens", 0)), openings=openings,
                                tried=(getattr(self, "_ask_recent", []) or [])[-int(c["show_tried"]):],
                                deepest=deepest, unsolved_open=open_unsolved)
            _, answer = self.query_LLM(q, temperature=0.7)
        except Exception as e:
            self._reopen_log(dict(event="reopen", move="STAY", why=f"ask failed: {e}"[:200],
                                  status="error", **base))
            self._fx_bump("reopen_ask_failed")
            return False
        move, why, status = reopen.parse(answer)
        self._reopen_log(dict(event="reopen", move=move, why=str(why)[:300], status=status,
                              n_openings=len(openings), deepest=deepest,
                              unsolved_open=open_unsolved, **base))
        self._fx_bump("reopen_" + move.lower())
        return move == "REOPEN"

    def _reopen_now(self, tree):
        """One forced root expansion. The tree is KEPT: the new opening competes with the old ones, so a
        late solve already under way is not discarded. Returns True if a decision was acted on."""
        if not self._reopen_decide(tree):
            return False
        c = self._reopen_cfg()
        self._reopens = int(getattr(self, "_reopens", 0)) + 1
        self._fx_bump("reopen_done")
        fresh = []
        for _ in range(max(int(c["openings"]), 1)):
            if not self._budget_left():
                break
            nodes = self._expand(tree, tree.root, 0, self.aot["max_root_retries"])
            self._evaluate_new(tree, nodes)
            fresh.extend(nodes or [])
        # arm reopenG: make the search actually TRY what it just paid for (see REOPEN_DEFAULTS["protect"])
        prot = int(c.get("protect") or 0)
        if prot > 0 and fresh:
            try:
                tree.protect([a.key for a in fresh], prot)
                self._fx_bump("reopen_protected")
            except Exception as e:
                print(f"[reopen] could not protect the new opening: {e}")
        self._reopen_log({"event": "reopened", "calls": self._aot_llm_calls,
                          "reopens": self._reopens, "new_reactions": len(fresh),
                          "openings_asked": int(c["openings"]), "protect": prot})
        if str(c["choose"]).lower() == "llm" and len(fresh) > 1:
            self._reopen_choose(tree, fresh)
        return True

    def _reopen_candidate(self, tag, a):
        """One opening described in the units the situational skills use, so the advisor can apply them."""
        pieces = [m.smiles for m in a.child_molecules]
        atoms = [self._heavy(p) for p in pieces]
        whole = max(self._heavy(getattr(self, "_aot_target", "") or ""), 1)
        big = pieces[atoms.index(max(atoms))] if atoms else None
        steps = self._ask_cached_steps().get(big) if big else None
        return {"tag": tag, "pieces": pieces, "n_pieces": len(pieces),
                "largest_share": (max(atoms) / whole) if atoms else None,
                "buyable": sum(1 for p in pieces if self._purchasable(p)),
                "splits": None if steps is None else len(steps),
                "stereo": self._stereo_count(big), "node": a}

    def _reopen_choose(self, tree, fresh):
        """Arm reopenP: the advisor picks which fresh opening the search pursues next. The others stay in
        the tree and keep competing -- the pick is a priority, not an exclusion, so a wrong choice costs
        ordering rather than losing a branch. A missing or unusable answer leaves the search's own order
        alone rather than guessing one."""
        cands = []
        for n, a in enumerate(fresh, 1):
            try:
                cands.append(self._reopen_candidate(f"o{n}", a))
            except Exception:
                continue
        if len(cands) < 2:
            return
        skills = ""
        try:
            best = min(cands, key=lambda x: x.get("largest_share") or 1.0)
            sit = {"depth": 1, "size_ratio": best.get("largest_share"),
                   "shrink_rate": (1.0 - (best.get("largest_share") or 1.0)),
                   "splits": best.get("splits"), "tries": 0, "calls": self._aot_llm_calls,
                   "stereo": best.get("stereo")}
            lib = srul.load(str(self._ask_cfg()["rules_file"]))
            skills = srul.render(srul.matches(sit, str(self._ask_cfg()["rules_file"])),
                                 lib.get("base_rate"))
        except Exception:
            skills = ""
        try:
            q = reopen.choose_question(
                target=getattr(self, "_aot_target", "") or tree.root.smiles,
                target_atoms=self._heavy(getattr(self, "_aot_target", "") or ""),
                calls=self._aot_llm_calls, budget=int(self.aot["llm_call_budget"]),
                candidates=[{k: v for k, v in x.items() if k != "node"} for x in cands],
                skills=skills)
            _, answer = self.query_LLM(q, temperature=0.7)
        except Exception as e:
            self._reopen_log({"event": "choose", "status": "error", "why": str(e)[:200],
                              "calls": self._aot_llm_calls})
            self._fx_bump("reopen_choose_failed")
            return
        tag, why, status = reopen.parse_choice(answer, tuple(x["tag"] for x in cands))
        pick = next((x for x in cands if x["tag"] == tag), None)
        if pick is not None:
            self._ask_pending = pick["node"]      # honoured once by the main loop's selection
            self._fx_bump("reopen_choose_ok")
        else:
            self._fx_bump("reopen_choose_unusable")
        self._reopen_log({"event": "choose", "status": status, "tag": tag, "why": str(why)[:300],
                          "calls": self._aot_llm_calls,
                          "candidates": [{k: v for k, v in x.items() if k != "node"} for x in cands]})

    def _ask_destinations(self, tree):
        """The shortlist: a UNION of the search's own top rows and the rows with the most
        checker-accepted splits, each row carrying both facts.

        Taking it from the search's ranking alone is what arm advsel did, and a frontier whose
        high-ranked rows are all dead ends gives the advisor no way out. Accepted-split counts come
        from the cache where the search already has them; at most `check_uncached` are computed
        here, and the rest are reported as unknown rather than zero -- showing a wrong number is the
        fault this arm exists to fix.
        """
        c = self._ask_cfg()
        cands = [a for a in tree.leaf_and_nodes
                 if a.depth < tree.max_depth and not a.is_fully_solved()
                 and tree.has_expandable_reactants(a)]
        if tree.propagate_failure:
            closed = tree.closed_by_failure()
            cands = [a for a in cands if a.key not in closed]
        # user 2026-11-16, "go and continue is separate": never offer the molecule being worked on now. In a toy
        # search it was offered back 3 times in 19, always as c1, so "GO c1" meant "stay" -- logged as a move,
        # then discarded when the failed expansion retired that node. Filtered HERE so `omitted` is right too.
        cur = getattr(self, "_ask_expanding", None)
        if cur is not None:
            cands = [a for a in cands
                     if (lambda m: m is None or m.smiles != cur)(tree.expansion_target(a))]
        rows, seen = [], set()
        by_search = sorted(cands, key=lambda a: (-tree.selection_score(a), a.depth, a.key))

        def mol_of(a):
            return tree.expansion_target(a)

        cc = self._cfg_of("cut_select", CUT_SELECT_DEFAULTS)
        steps_of = self._ask_cached_steps()

        def splits_known(smi):
            s = steps_of.get(smi)
            return None if s is None else len(s)

        def row_for(a, m, from_search, from_coverage):
            """One shortlist row. `size` and `best_cut` were added 2026-11-17: without them the advisor
            read accepted-split count as a quality score, and that count rises with molecule size and is
            capped at cut_select.show, so it is highest on the target itself. On arm `ask` 14 of 15 GO
            answers named the SHALLOWEST molecule offered and 11 named the root -- i.e. "start over".
            `buyable` was wrong here until the same date: it was hard-coded False and only ever corrected
            on rows whose splits had to be computed below, so cached rows told the advisor that nothing
            was purchasable no matter what the checker had found."""
            steps = steps_of.get(m.smiles)
            best, buy = self._ask_step_facts(steps)
            return {"node": a, "smiles": m.smiles, "depth": a.depth,
                    "tried": len(m.child_reactions),
                    "splits": None if steps is None else len(steps),
                    "size": self._heavy(m.smiles),
                    "best_cut": best, "buyable": buy,
                    "from_search": from_search, "from_coverage": from_coverage}

        # pass 1: the search's own order
        for a in by_search[:int(c["from_search"])]:
            m = mol_of(a)
            if m is None or m.smiles in seen:
                continue
            seen.add(m.smiles)
            rows.append(row_for(a, m, True, False))
        # pass 2: the rows we already know have the most accepted splits
        with_known = [a for a in by_search if (lambda m: m is not None and splits_known(m.smiles))(mol_of(a))]
        with_known.sort(key=lambda a: -(splits_known(mol_of(a).smiles) or 0))
        for a in with_known[:int(c["from_coverage"])]:
            m = mol_of(a)
            if m is None:
                continue
            if m.smiles in seen:
                for r in rows:
                    if r["smiles"] == m.smiles:
                        r["from_coverage"] = True
                continue
            seen.add(m.smiles)
            rows.append(row_for(a, m, False, True))
        # fill a bounded number of unknown counts, rather than guessing them
        budget = int(c["check_uncached"])
        for r in rows:
            if r["splits"] is None and budget > 0:
                budget -= 1
                try:
                    steps, _, _ = self._cuts(r["smiles"], float(cc["max_piece"]), int(cc["show"]),
                                             cc["check_max"], "cutsel")
                    r["splits"] = len(steps)
                    r["best_cut"], r["buyable"] = self._ask_step_facts(steps)
                except Exception:
                    pass
        for n, r in enumerate(rows, 1):
            r["tag"] = f"c{n}"
        return rows, max(len(cands) - len(rows), 0)

    @staticmethod
    def _ask_step_facts(steps):
        """(best_cut, buyable) over a molecule's accepted splits: the smallest `largest_piece_percent`
        -- how much the best available step actually simplifies the molecule -- and whether any piece
        can already be bought. Tolerant of a step missing either field: one odd entry must not blank
        the whole shortlist, which is what `_ask_destinations`'s except-clause would otherwise do."""
        best, buy = None, False
        for s in steps or ():
            if not isinstance(s, dict):
                continue
            p = s.get("largest_piece_percent")
            if isinstance(p, (int, float)) and not isinstance(p, bool):
                best = p if best is None else min(best, p)
            if s.get("buyable"):
                buy = True
        return best, buy

    def _ask_cutsel_key(self):
        """(max_piece, show, check_max) for this arm -- the last three fields of `_cuts`'s cache key.
        None when the config cannot be read, and callers then fall back to matching the molecule alone;
        the tool body must never raise, and a half-built object is not a reason to."""
        try:
            cc = self._cfg_of("cut_select", CUT_SELECT_DEFAULTS)
            return (float(cc["max_piece"]), int(cc["show"]), int(cc["check_max"]))
        except Exception:
            return None

    def _ask_cached_steps(self):
        """{smiles: accepted splits} from the cutsel cache, restricted to THIS arm's settings.

        `_cuts` keys on (smiles, max_piece, show, check_max), so the same molecule can hold entries
        computed under different settings -- different numbers, which must not be reported as this
        arm's (2026-11-17). A key that carries no settings at all cannot be mis-attributed, so it is
        taken as it is."""
        ckey = self._ask_cutsel_key()
        cache = self.__dict__.setdefault("_dc", {}).get("cutsel", {}) or {}
        out = {}
        for k, v in cache.items():
            if not (isinstance(k, tuple) and k and isinstance(v, tuple) and v):
                continue
            if ckey is not None and len(k) == 4 and tuple(k[1:]) != ckey:
                continue
            out[k[0]] = v[0]
        return out

    def _ask_current_steps(self):
        """The cached accepted splits of the molecule being expanded, or None if never computed."""
        cur = getattr(self, "_ask_expanding", None)
        return None if cur is None else self._ask_cached_steps().get(cur)

    def _ask_current_splits(self):
        """Accepted-split count for the molecule being expanded, read from the SAME cache the shortlist
        uses, so the two numbers in the prompt are comparable. None = never computed; it is never
        reported as 0, because showing a wrong count is the fault this arm exists to fix."""
        steps = self._ask_current_steps()
        return None if steps is None else len(steps)

    def _ask_current_row(self):
        """The `current` block of the prompt: the same facts, measured the same way, as a shortlist row,
        so the advisor compares like with like. Added 2026-11-17 alongside `size`/`best_cut` on the
        destinations -- a comparison where only one side carries size is what made the advisor read a
        big molecule's long split list as quality."""
        cur = getattr(self, "_ask_expanding", None)
        if cur is None:
            return None
        steps = self._ask_current_steps()
        best, buy = self._ask_step_facts(steps)
        return {"smiles": cur,
                "splits": None if steps is None else len(steps),
                "size": self._heavy(cur),
                "best_cut": best, "buyable": buy,
                "depth": getattr(self, "_ask_expanding_depth", None),
                "tried": int(getattr(self, "_ask_expanding_tried", 0))}

    @staticmethod
    def _stereo_count(smi):
        """Stereocentres, assigned or not. The one structural property that discriminated (0-2 centres
        reach a route 47-51% of the time, 3+ only 28-31%); every other descriptor tried came back at
        AUC ~0.5, see situational_rules.json `_what_did_not_work`."""
        try:
            m = Chem.MolFromSmiles(smi) if smi else None
            if m is None:
                return None
            return len(Chem.FindMolChiralCenters(m, includeUnassigned=True,
                                                 useLegacyImplementation=False))
        except Exception:
            return None

    def _ask_rules_block(self, cur_row):
        """Arm askR: the situational rules that match where the planner is now, rendered with their
        evidence. "" when the arm is off, nothing matches, or the molecule's situation is not measurable.

        The situation is described only by what the search already knows at this moment -- depth, size
        relative to the target, accepted splits, tries spent here, calls spent on the target. Molecule
        type is deliberately absent: conditioning the mined rules on functional groups made them WEAKER
        (best FG rule 6.4% on-path against 1.5% for the plain situational rule), so the functional group
        was a passenger and the situation carried the signal. See situational_rules.json `_caveats`.
        """
        c = self._ask_cfg()
        if not c["rules"] or not cur_row:
            return ""
        tsize = self._heavy(getattr(self, "_aot_target", "") or "")
        size = cur_row.get("size") or 0
        ratio = (size / tsize) if (tsize and size) else None
        depth = cur_row.get("depth")
        sit = {"depth": depth,
               "splits": cur_row.get("splits"),          # None stays None: never matched as 0
               "tries": cur_row.get("tried"),
               "calls": self._aot_llm_calls,
               "size_ratio": ratio,
               # shrink_rate: the share of the TARGET's mass removed per step taken so far. Validated
               # 2026-11-18 as the strongest single feature available here (AUC 0.709 on whether a
               # molecule ends up in the winning route, monotone across seven bands, and consistent in
               # three runs independently), which is why the library's main rules are built on it.
               "shrink_rate": ((1.0 - ratio) / max(int(depth or 0), 1)) if ratio is not None else None,
               "stereo": self._stereo_count(cur_row.get("smiles"))}
        try:
            lib = srul.load(str(c["rules_file"]))
            hits = srul.matches(sit, str(c["rules_file"]))
        except Exception as e:
            print(f"[askR] rules failed: {e}")
            return ""
        if not hits:
            self._fx_bump("ask_rules_none")
            return ""
        self._fx_bump("ask_rules_matched")
        self._ask_log({"event": "rules", "calls": self._aot_llm_calls, "situation": sit,
                       "matched": [r.get("id") for r in hits],
                       "advice": [r.get("advice") for r in hits]})
        return srul.render(hits, lib.get("base_rate"))

    def _ask_advisor_result(self, finished, asked=""):
        """The tool body. Returns a JSON string for the planner; never raises."""
        c = self._ask_cfg()
        tree = getattr(self, "_aot_tree", None)
        if tree is None:
            return json.dumps({"error": "no search in progress"})
        # 2026-11-17: the tool stays offered for the rest of the agentic loop, so the planner asked twice
        # inside ONE turn on 3 of the 6 targets that used the advisor -- 6 of 20 calls -- and on idx 24 the
        # two replies named DIFFERENT destinations, only one of which could be followed. A repeat within the
        # same turn now returns the first answer and spends no allowance. `_ask_turn_answer` is cleared at
        # the top of _ask_route, which is the turn boundary.
        repeat = getattr(self, "_ask_turn_answer", None)
        if repeat is not None:
            self._fx_bump("ask_repeat_in_turn")
            self._ask_log({"event": "ask_repeat", "calls": self._aot_llm_calls,
                           "finished": str(finished)[:300], "asked": str(asked)[:200],
                           "ask_call": int(getattr(self, "_ask_calls", 0))})
            return repeat
        self._ask_calls = int(getattr(self, "_ask_calls", 0)) + 1
        try:
            rows, omitted = self._ask_destinations(tree)
        except Exception as e:
            print(f"[ask] shortlist failed: {e}")
            rows, omitted = [], 0
        notes = list(getattr(self, "_ask_notes", []))
        tried = (getattr(self, "_ask_recent", []) or [])[-int(c["show_tried"]):]
        cur_row = self._ask_current_row()
        dry_at_call = int(getattr(self, "_ask_dry", 0))   # read BEFORE the reset below, so the log
        q = advisor_ask.question(                         # records how stuck the target actually was
            target=getattr(self, "_aot_target", ""), calls=self._aot_llm_calls,
            budget=self.aot["llm_call_budget"], finished=finished, asked=asked,
            dry=dry_at_call, tried=tried,
            destinations=[{k: v for k, v in r.items() if k != "node"} for r in rows],
            notes=notes, omitted=omitted, extra_tries=int(c["continue_extra_tries"]),
            target_size=self._heavy(getattr(self, "_aot_target", "") or ""),
            split_cap=int(self._cfg_of("cut_select", CUT_SELECT_DEFAULTS)["show"]),
            current=cur_row, rules=self._ask_rules_block(cur_row),
            allow_stop=bool(c["allow_stop"]))
        answer = None
        try:
            _, answer = self.query_LLM(q, temperature=float(c["temperature"]))
        except Exception as e:                      # an advisor failure must never cost the expansion
            print(f"[ask] advisor call failed: {e}")
        move, tag, note, why, status = advisor_ask.parse(answer, tuple(r["tag"] for r in rows),
                                                         allow_stop=bool(c["allow_stop"]))
        if note and c["notes"]:
            self.__dict__.setdefault("_ask_notes", []).append(note)
        self._ask_pending = None
        if move == "CONTINUE" and status == "ok":
            # only the advisor's own judgement grants the extra tries -- an unreadable answer also becomes
            # CONTINUE (the no-change fallback), and that must not spend budget on its behalf
            self._ask_continue = True
        if move == "STOP" and status == "ok":
            # arm askR: give the molecule up. _expand drops its remaining tries, and the main loop's
            # existing `failed` path then marks it dead and retires the reaction -- the same path an
            # exhausted expansion takes, so no separate retirement logic can disagree with it.
            self._ask_stop = True
            self._ask_continue = False
        if move == "GO" and tag:
            node = next((r["node"] for r in rows if r["tag"] == tag), None)
            if node is not None:
                self._ask_pending = node          # honoured once by the main loop's selection
        self._ask_dry = 0                          # the advice itself is a fresh start
        self._ask_log({"event": "ask", "calls": self._aot_llm_calls, "dry": dry_at_call,
                       "finished": str(finished)[:300], "asked": str(asked)[:200], "move": move, "tag": tag,
                       "status": status, "why": why[:300], "note": note[:300],
                       "n_destinations": len(rows), "omitted": omitted,
                       "destinations": [{k: v for k, v in r.items() if k != "node"} for r in rows],
                       "ask_call": self._ask_calls})
        self._fx_bump(f"ask_{move.lower()}")
        out = advisor_ask.result_json(move, tag, why,
                                      [{k: v for k, v in r.items() if k != "node"} for r in rows],
                                      extra_tries=(int(c["continue_extra_tries"]) if status == "ok" else 0))
        self._ask_turn_answer = out      # a second call in this same turn gets this back, free
        return out

    def _steer_active(self):
        """Arm steerRgate: is the advisor steering RIGHT NOW on this molecule? `after_calls` 0 (the 8 Nov arms) =
        from the first expansion. Otherwise the advisor stays out until the molecule has spent that many LLM calls
        without solving -- the stall gate. While it is out, every steer branch in the search loop is skipped, so the
        release's blind selection AND the release's dead-marking run: the molecule is treated exactly as the control
        treats it, which is what makes a matched comparison against the control meaningful before the gate opens.

        `_aot_llm_calls` is the per-molecule counter (reset for each target, restored on walltime resume), so the
        gate is per molecule and survives a chunk boundary."""
        if not getattr(self, "_steer_on", False):
            return False
        after = int(self._steer_cfg()["after_calls"])
        return after <= 0 or self._aot_llm_calls >= after

    def _steer_log(self, entry):
        self._aot_jsonl("steer", entry)

    def _steer_sc(self, smiles):
        try:
            return float(self.oracle.sc_Oracle.get_score_from_smi(smiles)[1])
        except Exception:
            return None

    def _steer_cuts(self, smiles):
        """rc's list for this molecule, computed ONCE per molecule and shared by both fact contexts.

        Review fixes (8 Nov, second pass):
          * the checker stops at rc's `show` PLUS max_reexpansions, so that after the splits the molecule already
            holds are filtered out (_steer_mol), up to rc's `show` unused ones remain -- otherwise a molecule whose
            only accepted cut was already attached was reported as having no real cut at all;
          * "one step from buyable" tests every candidate split with all pieces buyable, near-copies included
            (a deprotection to a buyable compound is a near-copy), up to 10 through the checker, as before the
            first rewrite narrowed it to the 5 biggest real cuts.
        -> ([{pieces, pct, buyable}], real-cut candidates (unchecked count), one_step_buyable)."""
        c = self._cfg_of("real_cuts", REAL_CUTS_DEFAULTS)
        extra = int(self.aot.get("max_reexpansions", 5)) if isinstance(self.aot, dict) else 5
        shown, _, n_real = self._cuts(smiles, float(c["max_piece"]), int(c["show"]) + extra, int(c["check_max"]),
                                      "steer", buyable_max=STEER_BUYABLE_MAX)
        cuts = [{"pieces": x["pieces"], "pct": x["largest_piece_percent"], "buyable": x["buyable"]} for x in shown]
        st = self.__dict__.setdefault("_dc", {})
        one, tried = False, 0
        for x in dc.search(self, smiles, sanitize_smiles, st):     # cached by _cuts above
            if tried >= STEER_BUYABLE_MAX:
                break
            if x["pieces"] and all(self._purchasable(p) for p in x["pieces"]):
                tried += 1
                if dc.accepted(self, smiles, x["pieces"], sanitize_smiles, st):
                    one = True
                    break
        return cuts, n_real, one

    def _steer_mol(self, smiles, tgt_atoms, ctx, state=None, used=()):
        """One molecule's facts. `used` = pieces of reactions the tree ALREADY holds for it, so a split that is
        already a reaction is not offered as new (review). Facts are cached per molecule and context."""
        cache = self.__dict__.setdefault("_steer_cache", {})
        k = (smiles, ctx)
        if k not in cache:
            atoms = dc._heavy(smiles)
            m = {"smiles": smiles, "atoms": atoms, "pct": round(100 * atoms / max(tgt_atoms, 1)),
                 "sc": self._steer_sc(smiles)}
            if ctx in ("splits", "rc5"):
                try:
                    cuts, n_real, one = self._steer_cuts(smiles)
                except Exception as e:
                    print(f"[steer] cut facts failed on {smiles}: {e}")
                    cuts, n_real, one = [], None, False
                m["_cuts"], m["_n_real"], m["_one"] = cuts, n_real, one
            cache[k] = m
        m = dict(cache[k])
        if state:
            m["state"] = state
        if "_cuts" in m:
            allc = m.pop("_cuts")
            fresh = [x for x in allc if tuple(sorted(x["pieces"])) not in used]
            n_real, one = m.pop("_n_real"), m.pop("_one")
            show = int(self._cfg_of("real_cuts", REAL_CUTS_DEFAULTS)["show"])
            all_used = bool(allc) and not fresh             # it has real cuts, all already tried
            if ctx == "rc5":
                m["rc"] = fresh[:show]
                m["rc_all_used"] = all_used
            else:
                m["splits"] = {"n": n_real, "best": fresh[0] if fresh else None, "one_step_buyable": one,
                               "all_used": all_used}
        return m

    @staticmethod
    def _steer_used(node):
        """The reactant sets the tree already holds for this molecule (so they are not shown as unused splits)."""
        return {tuple(sorted(x.reactants)) for x in node.child_reactions}

    def _steer_state(self, tree, node):
        """Why this molecule can or cannot still be worked on -- the most direct 'this branch is stuck' signal."""
        if node.purchasable:
            return "buyable"
        if node.is_solved:
            return "made"
        if node.smiles in tree.proven_dead:
            return "DEAD: no template can split it"
        if node.smiles in tree.dead:
            return "DEAD: every attempt failed"
        n = len(node.child_reactions)
        if n >= tree.max_reexpansions:
            return f"{n} reactions tried, at the limit: no new reaction can be added"
        return "never tried" if n == 0 else f"{n} reactions tried, all unfinished"

    @staticmethod
    def _steer_rxn(tree, x, label):
        pieces = [{"smiles": ch.smiles, "status": ("buyable" if ch.purchasable else "made" if ch.is_solved
                                                   else "dead" if ch.smiles in tree.dead else "not made")}
                  for ch in x.child_molecules]
        if x.is_fully_solved():
            status = "solved"
        elif any(p["status"] == "dead" for p in pieces):
            status = "blocked: a reactant is dead"
        else:
            status = f"{sum(1 for p in pieces if p['status'] == 'not made')} still to make"
        return {"product_label": label, "pieces": pieces, "status": status}

    @staticmethod
    def _steer_path(tree, A, P):
        """Molecules from the target down to A (through P, then up the cheapest parents) and the reactions."""
        mols, rxns, seen, r = [A], [], {id(A)}, P
        while r is not None:
            rxns.append(r)
            m = r.parent_molecule
            if id(m) in seen:
                break
            seen.add(id(m))
            mols.append(m)
            if m is tree.root or not m.parent_reactions:
                break
            r = min(m.parent_reactions, key=lambda p: (sum(1 for c in p.child_molecules if c is not m and not c.is_solved),
                                                       p.depth, p.key))
        return mols[::-1], rxns

    def _steer_facts(self, tree, A, P, opts, branch, c):
        ctx = c["context"]
        tgt_atoms = dc._heavy(tree.root.smiles)
        unmade = lambda n: self._steer_mol(n.smiles, tgt_atoms, ctx, self._steer_state(tree, n),
                                           self._steer_used(n))
        path, path_rxns = self._steer_path(tree, A, P)
        on_path, sib, seen_sib = {m.smiles for m in path}, [], set()
        for x in path_rxns:
            for ch in x.child_molecules:
                if not ch.is_solved and ch.smiles not in on_path and ch.smiles not in seen_sib:
                    seen_sib.add(ch.smiles)
                    sib.append(ch)          # nodes, not smiles: the state line needs the node
        used_A = self._steer_used(A)
        tries = self.__dict__.setdefault("_steer_tries", {}).get(A.smiles, [])
        last = tries[max((i for i, (att, _) in enumerate(tries) if att == 1), default=0):]
        f = {"target": {"smiles": tree.root.smiles, "atoms": tgt_atoms},
             "calls_used": self._aot_llm_calls, "budget": self.aot["llm_call_budget"],
             "path": [{"depth": d, "smiles": m.smiles} for d, m in enumerate(path)],
             # siblings get local facts only: split facts cost a 3-25 s template search per new molecule
             # siblings get local facts only: the cut search costs 3-25 s per new molecule
             "siblings": [self._steer_mol(x.smiles, tgt_atoms, "local", self._steer_state(tree, x))
                          for x in sib[:int(c["show_siblings"])]],
             "current": self._steer_mol(A.smiles, tgt_atoms, ctx, self._steer_state(tree, A), used_A),
             "n_tried": len(A.child_reactions),
             "tried": [self._steer_rxn(tree, x, "A") for x in A.child_reactions[-int(c["show_tried"]):]],
             "max_reactions": tree.max_reexpansions,
             "last_attempt": {"calls": len(last), "attached": sum(1 for _, k in last if k == "attached"),
                              "format": sum(1 for _, k in last if k == "format"),
                              "rejected": sum(1 for _, k in last if k == "rejected")}}
        if "DEEPER" in opts:
            d = opts["DEEPER"][0]
            f["deeper"] = {"reaction": self._steer_rxn(tree, d, "A"), "unmade": [unmade(ch) for ch in
                           d.child_molecules if not ch.is_solved]}
        if "PARTNER" in opts:
            _, pm = opts["PARTNER"]
            f["partner"] = {"reaction": self._steer_rxn(tree, P, P.product), "mol": unmade(pm)}
        if "ELSEWHERE" in opts:
            e = opts["ELSEWHERE"][0]
            f["elsewhere"] = {"reaction": self._steer_rxn(tree, e, e.product), "depth": e.depth,
                              "unmade": [unmade(ch) for ch in e.child_molecules if not ch.is_solved]}
        if "REMOVE" in opts:
            # only reactions that nothing outside P's branch still needs are really closed (review)
            closed = tree.closed_by_failure_if(P) if hasattr(tree, "closed_by_failure_if") else branch
            f["remove"] = {"reaction": self._steer_rxn(tree, P, P.product), "n_below": max(0, len(closed) - 1)}
        return f

    @staticmethod
    def _steer_category(tree, own, A, P, branch, opts):
        """Which move the search's own pick amounts to (logged next to the advisor's choice)."""
        if own is None:
            return "none"
        if "DEEPER" in opts and own is opts["DEEPER"][0]:
            return "deeper"
        if own is P and tree.expansion_target(P) is A:
            return "sideways"
        if "PARTNER" in opts and own is P and tree.expansion_target(P) is opts["PARTNER"][1]:
            return "partner"          # review: was logged 'same_branch', so PARTNER agreement was never measured
        return "same_branch" if own.key in branch else "elsewhere"

    def _steer_choose(self, tree, iteration):
        """Arm steer: ask where to go after the last expansion. -> (reaction, forced expansion target or None),
        or (None, None) = the search's own pick. Never raises.

        After an expansion that attached nothing, the release writes the molecule off at once. Here that write-off
        is applied PROVISIONALLY first, every option and the search's own pick are computed on that state, and it
        is undone only if the advisor chooses SIDEWAYS (try the molecule again). Code review, 8 Nov: building the
        options on the pre-write-off state let PARTNER and ELSEWHERE point at reactions the write-off then doomed,
        and logged a search pick the search would never make."""
        c = self._steer_cfg()
        pos, self._steer_pos = getattr(self, "_steer_pos", None), None
        if not pos:
            return None, None
        A, P = pos["mol"], pos["parent"]
        failed = bool(pos.get("failed"))
        entry = {"event": "decision", "iteration": iteration, "context": c["context"], "molecule": A.smiles,
                 "after_failed_expansion": failed}
        prov = None
        try:
            if A.is_solved or A is tree.root or P.key not in tree.reactions or P.is_fully_solved():
                if failed:
                    self._steer_writeoff(tree, A, P, None)
                return None, None
            could_retry = (failed and A.smiles not in tree.dead and tree.selectable(P)
                           and len(A.child_reactions) < tree.max_reexpansions
                           and (getattr(self, "_steer_streak", None) or {}).get(A.smiles, 0)
                           <= int(c["max_failed_retries"]))
            if failed:
                prov = self._steer_provisional(tree, A, P)   # options below see the release's state
            opts = {}
            if failed:
                if could_retry:
                    opts["SIDEWAYS"] = (P, A)
            elif (A.smiles not in tree.dead and len(A.child_reactions) < tree.max_reexpansions
                  and tree.selectable(P)):
                opts["SIDEWAYS"] = (P, A)
            deep = [tree.reactions[k] for k in pos["created"]
                    if k in tree.reactions and tree.reactions[k].parent_molecule is A]
            deep = [x for x in deep if tree.selectable(x) and tree.expansion_target(x) is not None]
            if deep:
                opts["DEEPER"] = (deep[0], None)
            if not failed and tree.selectable(P):
                partner = next((ch for ch in P.child_molecules
                                if ch is not A and not ch.is_solved and ch.smiles not in tree.dead
                                and len(ch.child_reactions) < tree.max_reexpansions), None)
                if partner is not None:
                    opts["PARTNER"] = (P, partner)   # the other molecule the SAME reaction needs (user: option a)
            # leaving means leaving: P's branch AND everything under A, which may also hang under another reaction
            # (review: ELSEWHERE could otherwise be the same reaction as DEEPER)
            branch = tree.branch_keys(P) | tree.subtree_keys(A)
            alt = tree.select(exclude=branch)
            if alt is not None:
                opts["ELSEWHERE"] = (alt, None)
            if c["allow_remove"] and not failed and tree.selectable(P):
                opts["REMOVE"] = (None, None)        # only while the branch is still live (review)
            own = tree.select()
            entry.update(depth=P.depth + 1,
                         search_pick=self._steer_category(tree, own, A, P, branch, opts),
                         search_pick_key=(own.key if own is not None else None),
                         search_pick_target=(getattr(tree.expansion_target(own), "smiles", None)
                                             if own is not None else None))
            if len(opts) < 2:
                self._fx_bump("steer_skipped")
                entry.update(outcome="skipped", offered=sorted(opts))
                self._steer_log(entry)
                if failed:
                    self._steer_writeoff(tree, A, P, prov)
                return None, None
            order = random.sample(list(opts), len(opts))      # shuffled: advselshadow had a 20% first-row bias
            q = ast_.prompt(self._steer_facts(tree, A, P, opts, branch, c), order)
            t = time.time()
            self._fx_bump("steer_asked")
            answer = None
            try:
                _, answer = self.query_LLM(q, temperature=float(c["temperature"]))
            except Exception as e:                       # the decision must never cost the expansion
                entry["llm_error"] = repr(e)[:200]
            mv, how = ast_.parse(answer, order)
            entry.update(offered=order, move=mv, parse=how, latency_s=round(time.time() - t, 1),
                         prompt_chars=len(q), answer_tail=(answer or "")[-400:])
            if mv is None:
                self._fx_bump("steer_fallback")
                entry["outcome"] = "fallback"
                self._steer_log(entry)
                if failed:
                    self._steer_writeoff(tree, A, P, prov)
                return None, None
            self._fx_bump("steer_" + mv.lower())
            if mv == "SIDEWAYS" and failed:
                self._steer_undo(tree, A, P, prov)          # the one outcome that keeps A alive
            elif failed:
                self._steer_writeoff(tree, A, P, prov)
            if mv == "REMOVE":
                closed = tree.closed_by_failure_if(P)
                tree.judged_failed.add(P.key)     # permanent; closed_by_failure drops P's branch from selection
                tree.retire(P)
                mols = sorted({m.smiles for k in closed if k in tree.reactions
                               for m in [tree.reactions[k].parent_molecule] + tree.reactions[k].child_molecules})
                entry.update(outcome="removed", removed=P.key, removed_below=max(0, len(closed) - 1),
                             removed_molecules=mols[:200])
                self._steer_log(entry)
                return None, None                 # the search's own pick, now without P's branch
            a, forced = opts[mv]
            entry.update(outcome="steered", chosen=a.key)
            self._steer_log(entry)
            return a, forced
        except Exception as e:
            self._fx_bump("steer_error")
            entry.update(outcome="error", error=repr(e)[:300])
            self._steer_log(entry)
            if failed:
                try:
                    self._steer_writeoff(tree, A, P, prov)   # an error falls back to the release's behaviour
                except Exception:
                    pass
            return None, None

    @staticmethod
    def _steer_provisional(tree, A, P):
        """Apply the release's write-off to the TREE only (dead + retire), remembering how to undo it. The oracle's
        dead_molecules count is bumped only when the write-off is final (_steer_writeoff)."""
        undo = {"was_dead": A.smiles in tree.dead, "was_open": P in tree.leaf_and_nodes}
        tree.dead.add(A.smiles)
        tree.retire(P)
        return undo

    @staticmethod
    def _steer_undo(tree, A, P, prov):
        if prov is None:
            return
        if not prov["was_dead"]:
            tree.dead.discard(A.smiles)
        if prov["was_open"]:
            tree.leaf_and_nodes.add(P)

    def _steer_writeoff(self, tree, A, P, prov):
        """Make the release's write-off final: A dead (tree and oracle), P retired."""
        if prov is None:
            if A.smiles in tree.dead:
                return
            tree.dead.add(A.smiles)
            tree.retire(P)
        self.update_dead_molecules(A.smiles)
        self._fx_bump("steer_deferred_dead")

    def _steer_ckpt(self):
        """Serialisable steer state for the checkpoint (the pending decision and the failure streaks)."""
        p = getattr(self, "_steer_pos", None)
        return {"pos": ({"mol": p["mol"].smiles, "parent": p["parent"].key, "created": list(p["created"]),
                         "failed": bool(p.get("failed"))} if p else None),
                "streak": dict(getattr(self, "_steer_streak", {}) or {})}

    # ------------------------------------------------------------------ termination fix (2026-11-08)
    def _term_cfg(self):
        c = self._cfg_of("termination", TERMINATION_DEFAULTS)
        if c["propagate_from"] not in ("proven", "all"):
            raise ValueError(f"aot.termination.propagate_from must be 'proven' or 'all', not {c['propagate_from']!r}")
        return c

    def _template_split_exists(self, smiles):
        """The checker's own template test, cached per target (disconnections.template_split_exists); run in a
        pool child when `aot.split_pool.enabled` and one is free (it is the same full template scan, so it pays
        the same GIL tax), else right here. The verdict is cached HERE either way, so the child never has to
        answer the same molecule twice."""
        st = self.__dict__.setdefault("_dc", {})
        cache = st.setdefault("tsplit", {})
        t = time.time()
        if smiles in cache:
            return cache[smiles]
        if self._split_pool_cfg()["enabled"]:
            v = sp.template_split_exists(smiles)
            if v is not sp.MISS:
                self._fx_bump("split_pool_hit")
                self._fx_bump("tsplit_s", round(time.time() - t, 2))
                cache[smiles] = v
                return v
            self._fx_bump("split_pool_miss")
        r = dc.template_split_exists(self, smiles, sanitize_smiles, st)
        self._fx_bump("tsplit_s", round(time.time() - t, 2))
        return r

    def _check_no_split(self, tree, nodes, depth):
        """Run the template test on each not-yet-tested, unsolved, not-dead molecule in `nodes`; mark every one
        that no template decomposes as dead and proven dead (no LLM call). -> True if any of `nodes` is now
        proven dead. All are tested (not just the first), so siblings feed propagation too."""
        st = self.__dict__.setdefault("_dc", {}).setdefault("tsplit", {})
        hit = False
        for c in sorted(nodes, key=lambda n: n.smiles):
            if c.is_solved:
                continue
            if c.smiles in tree.proven_dead:
                hit = True
                continue
            if c.smiles in tree.dead:
                continue
            fresh = c.smiles not in st
            r = self._template_split_exists(c.smiles)
            if fresh:
                self._fx_bump("tsplit_checked")
                if r is None:
                    self._fx_bump("tsplit_unknown")
            if r is False:
                tree.dead.add(c.smiles)
                tree.proven_dead.add(c.smiles)
                self.update_dead_molecules(c.smiles)
                self._fx_bump("no_split_dead")
                self._log_term({"event": "no_split_dead", "molecule": c.smiles, "depth": depth,
                                "is_root": c is tree.root})
                hit = True
        return hit

    def _aot_jsonl(self, kind, entry, target=None):
        """One row in aot_<kind>_<pid>.jsonl (termination events, steer decisions). Separate from aot_expansions,
        whose rows are one per planner LLM call."""
        try:
            entry = {"ts": _dt.utcnow().isoformat(timespec="seconds"),
                     "target_idx": getattr(self, "tb_target_idx", -1),
                     "target": target if target is not None else getattr(self, "_aot_target", None),
                     "worker": threading.current_thread().name, "llm_calls": self._aot_llm_calls, **entry}
            path = os.path.join(self.args.output_dir, f"aot_{kind}_{os.getpid()}.jsonl")
            with _APPEND_LOCK, open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[aot] {kind} log failed: {e}")

    def _log_term(self, entry):
        self._aot_jsonl("termination", entry)

    # ------------------------------------------------------------------ arms nollm / cuttool / cutsel (1 Nov)
    def _cfg_of(self, key, defaults):
        return dict(defaults, **((self.aot.get(key) or {}) if isinstance(self.aot, dict) else {}))

    # ------------------------------------------------------------------ the split search, in-process or pooled
    def _split_pool_cfg(self):
        return split_pool_settings(self.aot, getattr(self, "llm_config", None))

    def _best_steps(self, smiles, max_piece, show, check_max, buyable_max=0):
        """`disconnections.best_steps`, in a pool child when `aot.split_pool.enabled` and one is free, else right
        here. -> (steps, checked, n_candidates), identical either way; raises exactly as the direct call does.

        The child computes on a fresh state and sends back the two caches it filled (`disconnections.prefetch`);
        they are merged into ours WITHOUT overwriting anything we already hold, so our own values stay
        authoritative and every later `dc.search` / `dc.accepted` for this molecule is a cache hit -- which is
        what keeps the work out of this process instead of merely doing it twice."""
        st = self.__dict__.setdefault("_dc", {})
        if self._split_pool_cfg()["enabled"]:
            out = sp.best_steps(smiles, max_piece, show, check_max, buyable_max)
            if out is not sp.MISS:
                for key in ("cands", "verdicts"):
                    mine = st.setdefault(key, {})
                    for k, v in out[key].items():
                        mine.setdefault(k, v)
                self._fx_bump("split_pool_hit")
                return out["steps"], out["checked"], out["n"]
            self._fx_bump("split_pool_miss")
        return dc.best_steps(self, smiles, sanitize_smiles, st, float(max_piece), int(show), int(check_max))

    def _cuts(self, smiles, max_piece, show, check_max, tag, buyable_max=0):
        """rc's source, cached per molecule and setting. A failed search returns no cuts, never an exception."""
        st = self.__dict__.setdefault("_dc", {})
        cache = st.setdefault(tag, {})
        k = (smiles, float(max_piece), int(show), int(check_max))
        if k not in cache:
            self._split_pool_cfg()   # a config error must surface, not be swallowed below as "no cuts" (review 12 Nov)
            try:
                cache[k] = self._best_steps(smiles, max_piece, show, check_max, buyable_max)
            except Exception as e:
                print(f"[{tag}] search failed on {smiles}: {e}")
                self._fx_bump(f"{tag}_failed")
                cache[k] = ([], 0, 0)
        return cache[k]

    def _cut_route(self, smiles, tried):
        """Arm nollm: the biggest checker-accepted known split of `smiles` that the tree does not hold yet."""
        c = self._cfg_of("no_llm", NO_LLM_DEFAULTS)
        steps, checked, n = self._cuts(smiles, c["max_piece"], c["show"], c["check_max"], "nollm")
        canon = lambda xs: tuple(sorted(sanitize_smiles(x) or x for x in xs if x))
        done = set()
        for t in tried or []:
            if ">>" in t:
                rhs = t.split(">>", 1)[1].split()[0] if t.split(">>", 1)[1].split() else ""
                done.add(canon(rhs.split(".")))
        # Splits proposed earlier for this molecule are skipped too: _expand computes `tried` ONCE before
        # its retry loop, so a split that attached nothing would otherwise be re-proposed on every retry,
        # each one charged, while lower-ranked splits were never tried (code review, 1 Nov).
        # keyed by TARGET too: one optimizer serves many targets per worker, and two targets can share an
        # intermediate -- a split proposed for one target's search must not be skipped in another's
        root = getattr(getattr(self, "_aot_tree", None), "root", None)
        proposed = self.__dict__.setdefault("_nollm_proposed", {}).setdefault(
            (root.smiles if root is not None else "", smiles), set())
        for rank, s in enumerate(steps, 1):
            key = canon(s["pieces"])
            if key in done or key in proposed:
                continue
            proposed.add(key)
            self._last_cuts = {"nollm_rank": rank, "nollm_available": len(steps), "nollm_candidates": n}
            self._last_route_outcome = "ok"
            self._last_question = f"[no-llm] {smiles}"
            return _one_step_route(smiles, s["pieces"], "no-LLM baseline: biggest checker-accepted known split "
                                                         "not yet in the tree")
        self._last_cuts = {"nollm_rank": None, "nollm_available": len(steps), "nollm_candidates": n}
        self._last_route_outcome = "no_cut"
        self._fx_bump("nollm_exhausted")
        return None

    def _mine_tool_on(self):
        return bool(self._cfg_of("mine_tool", MINE_TOOL_DEFAULTS)["enabled"])

    def _mine_tool_stuck(self):
        """`wording` changes only what the model is TOLD; the facts returned are identical either way."""
        return str(self._cfg_of("mine_tool", MINE_TOOL_DEFAULTS).get("wording", "default")) == "stuck"

    def _mine_tool_schema(self):
        return MINE_TOOL_SCHEMA_STUCK if self._mine_tool_stuck() else MINE_TOOL_SCHEMA

    def _mine_tool_note(self):
        return MINE_TOOL_NOTE_STUCK if self._mine_tool_stuck() else MINE_TOOL_NOTE

    def _mine_tool_result(self):
        """Arm mine: the no-argument tool body. Answers for the molecule bound by `_expand`, exactly as the
        self-scoped disconnection tool does, so the model cannot chain it across fragments. Never raises."""
        try:
            self._fx_bump("mine_tool_calls")
            # Bound by _ask_route / _expand, same as the other self-scoped tools. `_cut_tool_self_smiles` is set
            # whenever the self-scoped disconnection tool is on (this arm always runs with it); `_val_tool_smiles`
            # is the valtool binding and covers the checker-feedback repair call too.
            smi = (getattr(self, "_cut_tool_self_smiles", None)
                   or getattr(self, "_val_tool_smiles", None))
            if not smi:
                self._fx_bump("mine_tool_unbound")
                return json.dumps({"error": "no molecule is being decomposed right now"})
            import skill_miner
            return skill_miner.how_is_this_made(smi)
        except Exception as e:
            return json.dumps({"error": f"how_is_this_made failed: {type(e).__name__}"})

    def _derive_skill_context(self, idx):
        """Everything the search recorded for ONE target, pulled back out of the per-process jsonl:
        (what the disconnection tool offered, the planner's own reasoning). Read-only, best effort."""
        offered, plans = [], []
        try:
            import glob as _glob
            for p in _glob.glob(os.path.join(self.args.output_dir, "llm_responses_*.jsonl")):
                with open(p) as fh:
                    for ln in fh:
                        try:
                            d = json.loads(ln)
                        except Exception:
                            continue
                        if d.get("target_idx") != idx:
                            continue
                        for t in (d.get("tool_trace") or []):
                            r = str(t.get("result", ""))[:400]
                            if r and "splits" in r:
                                offered.append(r)
                        for tag in ("PLAN", "EXPLANATION"):
                            for m in re.finditer(rf"<{tag}>(.*?)</{tag}>", str(d.get("response", "")), re.S):
                                plans.append(m.group(1).strip()[:600])
        except Exception:
            pass
        return ("\n".join(offered[:20]) or "(none)",
                "\n".join(plans[-8:]) or "(none recorded)")

    def _derive_skill(self, tree, solved, base):
        """Arm mine, `mine_tool.derive_skills` (2026-11-22, user: "we do get a skill repo at the end. we can
        mine, just don't give it access?").

        WRITE-ONLY, and that is the whole design. It runs AFTER the molecule's .status.json is written, so the
        search is over; the skill it writes goes to a per-run JSONL and is NEVER loaded back into the live
        library during this run. Nothing the model saw changes, so:
          * the search is byte-for-byte the arm it would have been without derivation, and the match against
            hard100_ctoolselfS survives;
          * results cannot depend on dataset order -- molecule 100 faced exactly what molecule 1 faced;
          * one run yields both the tool measurement and the skill repo for the NEXT run to apply frozen.
        The call is outside `llm_call_budget` (the molecule is finished and scored; its budget is spent and
        recorded in meta), and failure is swallowed -- a skill writer must never change a recorded outcome.
        """
        cfg = self._cfg_of("mine_tool", MINE_TOOL_DEFAULTS)
        if not cfg.get("derive_skills"):
            return
        try:
            import skill_miner
            out = os.path.join(self.args.output_dir, "derived_skills.jsonl")
            # The cap AND the already-derived motif set both come from the file, not from memory: one
            # optimizer is reused per worker thread, so an in-memory set would dedupe only within a worker
            # and would be lost across a walltime resubmit.
            n, done_motifs = 0, set()
            try:
                with open(out) as fh:
                    for ln in fh:
                        if not ln.strip():
                            continue
                        n += 1
                        try:
                            done_motifs.add(json.loads(ln).get("derived_from_motif", ""))
                        except Exception:
                            pass
            except FileNotFoundError:
                pass
            if n >= int(cfg.get("max_new_skills", MINE_TOOL_DEFAULTS["max_new_skills"])):
                self._fx_bump("derive_skipped_cap")
                return
            target = tree.root.smiles if tree else None
            if not target:
                return
            # ADMISSION 1: a RARE motif must have fired. Routine chemistry is not a skill, and the molecules
            # with nothing rare are the ones the disconnection tool already serves.
            facts = json.loads(skill_miner.how_is_this_made(target))
            findings = facts.get("findings") or []
            rare = [f for f in findings
                    if "rare" in f.get("how_often", "") or "finding" in f]
            if not rare:
                self._fx_bump("derive_skipped_nothing_rare")
                return
            # ADMISSION 2: don't re-derive what a stored gate already covers. Tested over EVERY rare motif,
            # not just the rarest: the skill is gated on the motif the WRITER names, which need not be
            # rare[0], so testing rare[0] alone let the same gate be derived again and again (code review
            # 2026-11-22). If every rare motif here is already covered, nothing new can come of the call,
            # so skip it before paying for it; the motif the writer actually picks is re-checked below.
            dedupe = bool(cfg.get("dedupe_motifs", True))
            rare_motifs = {f.get("motif", "") for f in rare if f.get("motif")}
            if dedupe and rare_motifs and rare_motifs <= done_motifs:
                self._fx_bump("derive_skipped_motif_seen")
                return
            # `AOTTree.reactions` is a DICT (key -> ANDNode), not a list. Slicing it raised
            # "TypeError: unhashable type: 'slice'" on EVERY molecule that cleared admission 1 -- 32 of the
            # 100 in hard100_mine -- and the failure is swallowed by the handler below, so both mine and
            # mineall finished having derived nothing at all. The unit test missed it because its fake tree
            # used `reactions = ["A>>B"]`; it now builds a real AOTTree (2026-11-24).
            nodes = list(tree.reactions.values()) if tree else []
            steps = "\n".join(
                f"{i+1}. {getattr(a, 'product', '?')}>>{'.'.join(getattr(a, 'reactants', None) or [])}"
                for i, a in enumerate(nodes[:30])) or "(no accepted steps)"
            rej = "\n".join(str(r)[:300] for r in (getattr(self, "_val_tool_rejected", None) or [])[:20]) \
                or "(none recorded)"
            # THE CONTEXT (user, 2026-11-22: "make sure when you're deriving the skill, you get all the
            # context, not just the tree itself"). The per-process jsonl already holds every call for this
            # target: the model's own <PLAN>/<EXPLANATION> text, and the full tool_trace -- which is the only
            # record of what the disconnection tool OFFERED, so it is also how admission rule 3 is applied
            # (never write a skill about a disconnection the tool already hands over for free).
            offered, reasoning = self._derive_skill_context(getattr(self, "tb_target_idx", -1))
            prompt = skill_miner.derivation_prompt(
                target, solved, steps, rej, json.dumps(findings, indent=1), offered, reasoning)
            reply = (self.query_LLM(prompt, temperature=0.2) or "")
            if isinstance(reply, tuple):
                reply = reply[0] or ""
            if "NO SKILL" in reply.upper():          # the writer is ALLOWED to decline; that is the cap
                self._fx_bump("derive_declined")     # that keeps the store from filling with boilerplate
                return
            m = re.search(r"\{.*\}", reply, re.S)
            if not m:
                self._fx_bump("derive_unparseable")
                return
            body = json.loads(m.group(0))
            # The gate is still built in CODE, but from the motif the skill SAYS it is about, validated
            # against the motifs actually found in this molecule. Forcing the rarest motif instead would
            # gate a skill on chemistry it does not discuss -- which is the hand-written skill 4 failure
            # (an authored gate that matched 1 molecule in 450) arriving by a different route.
            named = str(body.get("motif", "")).strip()
            allowed = {f.get("motif", "") for f in findings}
            if named not in allowed:
                self._fx_bump("derive_motif_not_in_evidence")
                return
            if dedupe and named in done_motifs:      # ADMISSION 2, on the motif that will actually gate it
                self._fx_bump("derive_skipped_motif_seen")
                return
            skill = skill_miner.attach_gate(body, named)
            if not skill:                            # no SMARTS for it -> no gate we can derive -> no skill
                self._fx_bump("derive_no_gate")
                return
            skill["target"] = target
            skill["solved"] = bool(solved)
            with open(out, "a") as fh:               # append: concurrent workers each add one line
                fh.write(json.dumps(skill) + "\n")
            self._fx_bump("derive_written")
        except Exception as e:
            self._fx_bump("derive_failed")
            print(f"[mine] skill derivation failed (ignored): {type(e).__name__}: {e}", flush=True)

    def _mol_tool_cfg(self):
        return self._cfg_of("mol_tool", MOL_TOOL_DEFAULTS)

    def _mol_tool_on(self):
        return bool(self._mol_tool_cfg()["enabled"])

    def _mol_tool_result(self):
        """Arm moltool: the no-argument tool body. Answers for the molecule bound by `_expand`, which spans
        both the main route call and any checker-feedback repair. Never raises -- a tool that can kill an
        expansion is worse than no tool."""
        smi = getattr(self, "_mol_tool_smiles", None)
        if not smi:
            return json.dumps({"error": "describe_this_molecule is only available while a molecule is "
                                        "being disconnected"})
        c = self._mol_tool_cfg()
        cap = int(c["max_calls"] or 0)
        n = int(getattr(self, "_mol_tool_calls", 0))
        if cap and n >= cap:
            return json.dumps({"error": f"describe_this_molecule has already been called {n} times for "
                                        f"this molecule"})
        self._mol_tool_calls = n + 1
        self._fx_bump("mol_tool_call")
        try:
            out = mf.facts(smi)
        except Exception as e:                       # mol_facts already guards, this is the last resort
            self._fx_bump("mol_tool_failed")
            return json.dumps({"error": f"describe_this_molecule failed: {e}"})
        try:
            self._mol_tool_log_row({"molecule": smi, "calls": self._aot_llm_calls,
                                    "tool_call": self._mol_tool_calls,
                                    "heavy_atoms": out.get("heavy_atoms"),
                                    "n_functional_groups": len(out.get("functional_groups") or {})})
        except Exception:
            pass
        return json.dumps(out)

    def _image_tool_on(self):
        c = self._mol_tool_cfg()
        return bool(c["enabled"] and c["image"])

    def _image_tool_result(self):
        """Arm moltool (image half): records the molecule to draw and returns TEXT. The picture itself is
        appended by the agentic loop in optimizer.query_LLM as its own user message, because a tool result
        is a string in the chat protocol and cannot carry an image. Never raises."""
        smi = getattr(self, "_mol_tool_smiles", None)
        if not smi:
            return json.dumps({"error": "see_this_molecule is only available while a molecule is being "
                                        "disconnected"})
        c = self._mol_tool_cfg()
        cap = int(c["max_images"] or 0)
        n = int(getattr(self, "_image_tool_calls", 0))
        if cap and n >= cap:
            return json.dumps({"error": f"see_this_molecule has already drawn this molecule {n} times"})
        self._image_tool_calls = n + 1
        self._fx_bump("image_tool_call")
        self._image_tool_pending = [smi]        # the loop attaches it, then clears this
        try:
            self._mol_tool_log_row({"molecule": smi, "calls": self._aot_llm_calls,
                                    "image_call": self._image_tool_calls, "kind": "image"})
        except Exception:
            pass
        return json.dumps({"drawn": smi, "note": "The labelled structure diagram follows as an image."})

    def _mol_tool_log_row(self, entry):
        path = os.path.join(self.args.output_dir, f"aot_moltool_{os.getpid()}.jsonl")
        entry = dict(ts=_dt.utcnow().isoformat(timespec="seconds"),
                     target_idx=getattr(self, "tb_target_idx", -1),
                     worker=threading.current_thread().name, **entry)
        with _APPEND_LOCK, open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _cut_tool_cfg(self):
        """aot.cut_tool, with `scope` validated: an unknown scope must stop the run, not silently fall back to the
        unrestricted tool and make the arm a duplicate of cuttool."""
        c = self._cfg_of("cut_tool", CUT_TOOL_DEFAULTS)
        if c["scope"] not in ("any", "self"):
            raise ValueError(f"aot.cut_tool.scope must be 'any' or 'self', not {c['scope']!r}")
        return c

    def _cut_tool_result_self(self):
        """Arm ctoolself: what the NO-ARGUMENT tool returns -- the splits of the molecule being decomposed, and of
        nothing else. `_ask_route` binds that molecule for the duration of the call (one optimizer per worker
        thread), so the model cannot reach a fragment it was just handed."""
        smi = getattr(self, "_cut_tool_self_smiles", None)
        if not smi:
            self._fx_bump("cut_tool_self_unbound")
            return json.dumps({"error": "no molecule is being decomposed right now"})
        return self._cut_tool_result(smi)

    def _validate_tool_result(self, route_str=None):
        """arm valtool, tool 1 (check_my_route): a FORMAT report on the candidate. Pure syntax -- it never runs
        the route checker, suggests no disconnection, and exposes no purchasability. JSON, never raises."""
        try:
            self._fx_bump("validate_tool_calls")
            if route_str is None or str(route_str).strip() == "":
                return json.dumps({"error": "pass the candidate route to check, as the 'route' argument"})
            rep = self._syntax_report(str(route_str))
            # WORDING (2026-11-22, user: "fix it all and resubmit"). The old note called the ARGUMENT "the answer"
            # ("the answer is well-formed"), which is exactly the conflation that caused the failure: of the 87
            # valtool answers with no usable <ROUTE> block, 76 had just been told ok, and the model then ended its
            # turn believing it had answered. So: never call the argument an answer, and put the strongest reminder
            # on the ok=True branch, not the ok=False one -- a clean result is the dangerous reply, not a dirty one.
            rep["note"] = ("the route string you passed me is well-formed. This is NOT a submission and NOT an "
                           "answer, and it says nothing about whether the chemistry is accepted -- only that what "
                           "you handed me is readable" if rep["ok"] else
                           "the route string you passed me has a format problem; fix it before you submit")
            rep["submitted"] = False
            # Repeated in the RESULT because that is the last thing the model reads before it decides to stop.
            rep["next"] = ("NOTHING HAS BEEN SUBMITTED. You have not answered yet. Write the finished route out in "
                           "full inside <ROUTE></ROUTE> in your reply now"
                           + ("" if rep["ok"] else ", after fixing the problems above")
                           + " -- if you stop here your answer is lost and this call is wasted.")
            return json.dumps(rep)
        except Exception as e:
            return json.dumps({"error": f"check_my_route failed: {e}"})

    def _rejected_tool_result(self):
        """arm valtool, tool 2 (my_rejected_steps): the first steps already rejected for the bound molecule this
        expansion, each with the reason the checker ALREADY gave (no fresh checker call, no suggested
        disconnection, no purchasability). JSON, never raises."""
        try:
            self._fx_bump("rejected_tool_calls")
            rej = list(getattr(self, "_val_tool_rejected", []) or [])
            note = ("these steps have already been rejected for this molecule; do not propose them again"
                    if rej else "nothing has been rejected yet for this molecule")
            return json.dumps({"rejected_steps": rej, "note": note})
        except Exception as e:
            return json.dumps({"error": f"my_rejected_steps failed: {e}"})

    def _rejection_reason(self):
        """A short, leakage-free reason the last PARSED attempt was rejected, read from what the checker already
        computed for it (self._last_step1) -- no fresh checker call and no suggested disconnection. Purchasability
        rejections are deliberately NOT reported, since that would hand over the oracle's own stopping test."""
        s1 = getattr(self, "_last_step1", None)
        if s1 is None:
            return "a step was malformed"
        ok, info = s1
        if ok:
            return "this step is already in the search tree"
        if info.get("invalid_updated_mol_id"):
            return "a molecule in the step is not a valid SMILES"
        if not info.get("product_inside", True):
            return "the step's product was not the molecule being decomposed"
        if not info.get("reaction_existance"):
            return "this step matches no known USPTO reaction"
        if not info.get("reaction_valid"):
            return "the reaction does not apply to this molecule"
        if not info.get("updated_set_valid"):
            return "the reactants do not match what the reaction produces"
        return "rejected by the route checker"

    def _syntax_report(self, route_str):
        """Leakage-free structural check of a candidate route: tag present, parses to a list of step dicts,
        required fields present, every SMILES syntactically valid, reaction string well-formed. No route
        checker, no chemistry verdict, no purchasability. -> {ok: bool, problems: [str, ...]}."""
        from agentic_tools import _smiles_detail
        body = extract_route_block(route_str) if "<ROUTE>" in route_str else route_str
        if body is None:
            return {"ok": False, "problems": ["no <ROUTE>...</ROUTE> block found"]}
        try:
            route = ast.literal_eval(body.strip())
        except Exception as e:
            return {"ok": False, "problems": [f"the route is not a readable list of steps ({type(e).__name__})"]}
        if not isinstance(route, list) or not route:
            return {"ok": False, "problems": ["the route is not a non-empty list of steps"]}
        REQ = ("Molecule set", "Product", "Reaction", "Reactants", "Updated molecule set")
        problems = []
        for i, step in enumerate(route, 1):
            if not isinstance(step, dict):
                problems.append(f"step {i}: not a dictionary")
                continue
            for f in REQ:
                if f not in step:
                    problems.append(f"step {i}: missing field '{f}'")
            smis = []
            for f in ("Molecule set", "Product", "Reactants", "Updated molecule set"):
                if f not in step:
                    continue
                try:
                    val = ast.literal_eval(step[f]) if isinstance(step[f], str) else step[f]
                    if isinstance(val, (list, tuple)):
                        smis.extend(str(x) for x in val if x)
                    elif val:
                        smis.append(str(val))
                except Exception:
                    problems.append(f"step {i}: field '{f}' is not a readable list")
            if "Reaction" in step:
                try:
                    rx = ast.literal_eval(step["Reaction"]) if isinstance(step["Reaction"], str) else step["Reaction"]
                    rx = rx[0] if isinstance(rx, (list, tuple)) and rx else rx
                    if not isinstance(rx, str) or ">>" not in rx:
                        problems.append(f"step {i}: reaction is not of the form 'product>>reactants'")
                    else:
                        smis.extend(p for half in rx.split(">>") for p in half.split(".") if p)
                except Exception:
                    problems.append(f"step {i}: reaction string is not readable")
            for smi in smis:
                valid, _canon, reason = _smiles_detail(smi)
                if not valid:
                    problems.append(f"step {i}: invalid SMILES '{smi}' ({reason})")
        # De-duplicate, order-preserving (2026-11-22, user: "fix it all"). The same bad SMILES occurs in several
        # fields of a step (Product / Reaction / Reactants / Updated molecule set), so one typo was reported up to
        # four times verbatim -- seen in the run as three identical "step 3: invalid SMILES 'CO2'" lines. Repeats
        # add nothing the model can act on and pad the context it reads just before it has to rewrite the route.
        problems = list(dict.fromkeys(problems))
        return {"ok": not problems, "problems": problems}

    def _cut_tool_result(self, smiles):
        """Arm cuttool: what get_known_splits returns -- rc's list for that molecule (JSON, never raises)."""
        try:
            c = self._cfg_of("cut_tool", CUT_TOOL_DEFAULTS)
            mol = sanitize_smiles(smiles) or smiles
            steps, _, _ = self._cuts(mol, c["max_piece"], c["show"], c["check_max"], "cuttool")
            self._fx_bump("cut_tool_calls")
            self.__dict__.setdefault("_cut_tool_log", []).append({"smiles": mol, "n": len(steps)})
            # `wording: stuck` removes the acceptance promise, and it has to go from the RESULT too, not
            # only the tool description and the prompt note -- otherwise the tool re-asserts it on every
            # call and the ablation measures nothing. `show_buyable: false` drops the purchasability
            # annotation: the splits are unchanged, the model is simply not told which pieces can already
            # be bought, which is the oracle's own stopping test answered in advance.
            stuck = str(c.get("wording", "default")) == "stuck"
            show_buy = bool(c.get("show_buyable", True))
            out = []
            for s_ in steps:
                row = {"reaction": f"{mol}>>{'.'.join(s_['pieces'])}",
                       "largest_piece_percent": s_["largest_piece_percent"]}
                if show_buy:
                    row["buyable"] = s_["buyable"]
                out.append(row)
            if not steps:
                note = "no known split of this molecule was found"
            elif stuck:
                note = "biggest simplification first"
            else:
                note = "the route checker accepts each split as written; biggest simplification first"
            return json.dumps({"molecule": mol, "splits": out, "note": note})
        except Exception as e:
            return json.dumps({"error": f"get_known_splits failed: {e}"})

    def _select_cuts_block(self, smiles):
        """Arm cutsel: a separate Qwen call sees EVERY checker-accepted split of the molecule and picks `pick`; the
        main prompt gets them in rc's block, biggest cut first, so only WHO CHOSE differs from rc. Computed once per
        molecule; the selector call is not charged to the oracle budget (logged)."""
        c = self._cfg_of("cut_select", CUT_SELECT_DEFAULTS)
        if not c["enabled"]:
            return ""
        if self._rc_cfg()["enabled"]:
            raise ValueError("aot.cut_select replaces aot.real_cuts: enable one of them, not both")
        st = self.__dict__.setdefault("_dc", {})
        cache = st.setdefault("cutsel_block", {})
        if smiles not in cache:
            full, checked, n = self._cuts(smiles, float(c["max_piece"]), int(c["show"]), c["check_max"], "cutsel")
            info = {"cutsel_full": len(full), "cutsel_candidates": n, "cutsel_called": False}
            chosen = full
            if len(full) > int(c["pick"]):
                info["cutsel_called"] = True
                self._fx_bump("cutsel_calls")
                answer = None
                try:
                    scope = (None if float(c["max_piece"]) > 1 and int(c["show"]) >= 10 ** 6 else
                             f"the {len(full)} biggest of its known one-step splits that are real cuts (splits that "
                             f"keep 90% or more of the molecule are left out)")
                    _, answer = self.query_LLM(dc.select_prompt(smiles, full, int(c["pick"]), scope),
                                               temperature=0.7)
                except Exception as e:                  # a selector failure must never cost the expansion its call
                    print(f"[cutsel] selector failed on {smiles}: {e}")
                    self._fx_bump("cutsel_failed")
                idx = dc.parse_selection(answer, len(full), int(c["pick"]))
                info.update(cutsel_parsed=len(idx), cutsel_filled=int(c["pick"]) - len(idx))
                picked = [full[i] for i in idx]
                for s in full:                          # fill from size order if the answer gave fewer
                    if len(picked) >= int(c["pick"]):
                        break
                    if s not in picked:
                        picked.append(s)
                chosen = sorted(picked, key=lambda s: s["largest_piece_percent"])
                info["cutsel_ranks"] = [full.index(s) + 1 for s in chosen]   # 1 = rc's own first choice
            cache[smiles] = (dc.real_cuts_block(smiles, chosen), info)
        block, info = cache[smiles]
        self._last_cuts = dict(info)
        return block

    # ------------------------------------------------------------------ rolling plan (try 9)
    def _plan9_on(self):
        c = self._adv_cfg()
        return bool(c["enabled"] and c["plan_of_attack"] and c["plan_at_start"])

    def _plan9_block(self):
        """Try 9 (aot.advisor.plan_at_start): the target's plan for this route prompt (every prompt, "" = none).
        Discussed before the first call and again every replan_every budget calls (ap9.discuss). A plan
        call the server never answered is asked again at the next call (at most 2 retries per plan); an unusable
        answer waits for the next re-plan. A failed re-plan keeps the previous plan."""
        self._plan9_shown, self._last_plan9 = False, None
        if not self._plan9_on():
            return ""
        c, st = self._adv_cfg(), self._adv_state()
        p = st.get("plan9")
        calls = self._aot_llm_calls
        every = int(c["replan_every"] or 0)
        due = (p is None or (p.get("error") and p.get("errors", 0) <= 2)
               or (every > 0 and calls - p["at"] >= every))
        if due and self._time_left():
            p = self._plan9_discuss(p)
        block = ap9.plan_block(p, p["shown_at"] if p else 0, trust=bool(c.get("trust_framing")))
        if block:
            self._plan9_shown = True
            self._fx_bump("adv9_plan_shown")
            self._last_plan9 = {"plan_v": p["n"], "plan_at": p["shown_at"], "plan_status": p.get("status")}
        return block

    def _adv9_facts(self, smiles):
        """Try 8's facts for one molecule (same cache and similar-routes lookup); None if they fail."""
        root = self._aot_tree.root.smiles
        try:
            return al.node_facts(self, smiles, [], lambda smi: self._adv8_routes(smi, root), self._adv8_state(),
                                 sanitize_smiles, self._heavy)
        except Exception as e:
            print(f"[adv9] facts failed on {smiles}: {e}")
            self._fx_bump("adv9_facts_failed")
            return None

    def _plan9_discuss(self, prev):
        """One plan discussion (the start plan if prev is None, else a re-plan); -> the plan record now in use.
        All calls are free (outside llm_call_budget) and logged to aot_advisor_<pid>.jsonl (stages plan9_*)."""
        if getattr(self, "_adv_pool", None) is None:
            self._adv_make_pool()
        c, st = self._adv_cfg(), self._adv_state()
        tree, calls = self._aot_tree, self._aot_llm_calls
        root, name = tree.root.smiles, list(c["advisors"])[0]
        replan = prev is not None and prev.get("n", 0) > 0
        state = None
        if calls > 0:                       # any plan written after the search started sees the search so far
            def fails_of(smi):
                rd = [a for a in st["hist"].get(smi, []) if a["verdict"] != "unreadable"]
                failed = [a for a in rd if not a["attached"]]
                return len(failed), (failed[-1]["words"] if failed else "")
            try:
                state = ap9.search_state(tree, calls, self.aot["llm_call_budget"], self._heavy, self._adv_path, fails_of)
            except Exception as e:
                print(f"[adv9] search state failed: {e}")
                state = f"- LLM calls used: {calls} of {self.aot['llm_call_budget']}."
        ask = ap9.ask_advisor_fn(al.ask_messages, self._adv_pool, name, c["plan_max_tokens"], c["advisor_context"],
                                 self._estimate_message_tokens)
        plan, trace = ap9.discuss(root, self._adv9_facts(root), ask,
                                  lambda msgs: ap9.qwen_chat(self, msgs, c["plan_reply_max_tokens"]),
                                  rounds=c["plan_rounds"], state=state,
                                  prev_plan=prev["text"] if prev and prev.get("usable") else None,
                                  trust=bool(c.get("trust_framing")))
        why = "replan" if replan else "plan_start"
        for e in trace:
            self._fx_bump("adv9_replies" if e["stage"] == "plan9_reply" else "advisor_calls")
            self._log_adv8(root, name, why, 0, e)
        self._fx_bump("adv9_replans" if replan else "adv9_plans")
        self._fx_bump(f"adv9_{plan['status']}")
        if plan["error"]:
            self._fx_bump("adv9_plan_error")
        n = (prev or {}).get("n", 0)
        if plan["usable"]:
            rec = dict(plan, at=calls, shown_at=calls, n=n + 1)
            st.setdefault("plan", {})[root] = {"text": plan["text"], "usable": True, "error": False}   # try 8 reads it
        elif prev and prev.get("usable"):
            rec = dict(prev, at=calls)                    # keep the plan in use; wait for the next re-plan
        else:
            rec = dict(plan, at=calls, shown_at=calls, n=n,
                       errors=(prev or {}).get("errors", 0) + (1 if plan["error"] else 0))
        st["plan9"] = rec
        return rec

    def _repair_hint(self, smiles, tried, attempt):
        """Skill repair (config `aot.skill_repair`): only for a stuck expansion, i.e. a retry
        after a failed attempt or a re-expansion of a molecule with tried reactions. One free
        selector call (not counted in llm_call_budget) picks <= skill_max library entries or
        NONE; the hint text is appended to the expansion prompt."""
        self._last_skill = None
        if not self.aot.get("skill_repair") or not (attempt > 1 or tried):
            return ""
        import skill_repair as sr
        if getattr(self, "_skill_lib", None) is None:
            self._skill_lib = sr.load_library(self.aot["skill_library"])
        lib, k = self._skill_lib, int(self.aot["skill_max"])
        # the previous attempt's failure reason only exists on a retry; on the first attempt of a
        # re-expansion _last_route_outcome belongs to some other molecule
        last = getattr(self, "_last_route_outcome", None) if attempt > 1 else None
        fixed, shown = [], None
        if self.aot.get("skill_gate") or self.aot.get("skill_rotate"):
            if getattr(self, "_skills_shown", None) is None:
                self._skills_shown = {}
            shown = self._skills_shown.setdefault(smiles, set()) if self.aot.get("skill_rotate") else set()
        if self.aot.get("skill_gate"):
            # hints 1..fixed_k: best gated entries by fixed rank (no LLM); the selector then chooses
            # up to k more from the REMAINING gated entries, so it can only pick what fits the molecule
            gated = sr.eligible(smiles, lib, shown)
            fixed = [e["id"] for _, e in gated[:int(self.aot.get("skill_fixed_k", 0))]]
            # 2026-11-20 (user: "deterministically add all that is relevant + llm selection up to 5 more"):
            # with skill_fixed_k >= the number of gate-passing entries, EVERY relevant skill is added without
            # an LLM call, which previously left the selector an empty menu and no job. The menu is now the
            # rest of the LIBRARY, so the selector can still offer a skill whose structural gate did not fire
            # -- a gate is a cheap substructure test and cannot see, for example, that a cap must come off
            # before a ring can be closed. `skill_rotate` still prevents repeats within a molecule.
            if str(self.aot.get("skill_menu", "gated")).lower() == "library":
                menu = [e for e in lib if e["id"] not in fixed and (not shown or e["id"] not in shown)]
            else:
                menu = [e for _, e in gated if e["id"] not in fixed]
        else:
            menu = [e for e in lib if not shown or e["id"] not in shown]
        ids, state, answer = [], "", None
        if menu and k > 0:
            prompt = sr.build_selector_prompt(smiles, tried, last, menu, k)
            self._fx_bump("skill_selector_calls")
            try:
                _, answer = self.query_LLM(prompt, temperature=0.0)
            except Exception as e:   # a selector failure must never cost the expansion its normal call
                print(f"[aot] skill selector failed: {e}")
                answer = None
            ids, state = sr.parse_selection(answer, menu, k)
        picked = fixed + ids
        if shown is not None:
            shown.update(picked)
        self._last_skill = {"skill_ids": picked, "skill_state": state, "skill_answered": bool(answer)}
        if fixed:
            self._last_skill["skill_fixed"] = fixed
        self._fx_bump("skill_picked" if picked else "skill_none")
        return sr.render_hint(picked, lib)

    # ------------------------------------------------------------------ self-notes (try 12, arm notes)
    def _notes_cfg(self):
        c = (self.aot.get("self_notes") or {}) if isinstance(self.aot, dict) else {}
        return dict(NOTES_DEFAULTS, **(c if isinstance(c, dict) else {}))

    def _notes_state(self):
        """The target's own note (checkpointed): one note shared by every expansion in this target's search."""
        return self.__dict__.setdefault("_nt", {"text": None, "fields": {}, "writes": 0})

    def _notes_block(self, question):
        """The note as it goes into THIS route prompt, plus the request to rewrite it. `question` is the prompt
        built so far, which is what the guard measures the note against."""
        c = self._notes_cfg()
        if not c["enabled"]:
            return ""
        st = self._notes_state()
        llm = self.llm_config.get("llm", {}) if isinstance(self.llm_config, dict) else {}
        context = int(llm.get("context_length", 32768))
        reserve = int(llm.get("max_tokens", 20480)) + int(c["reserve_tokens"])
        shown, shrink, truncated = sn.guard(st["text"], sn.estimate(question), context, reserve)
        if shrink:
            self._fx_bump("notes_shrink_asked")
        if truncated:
            self._fx_bump("notes_truncated")
        self._last_notes = {"notes_writes": st["writes"], "notes_tokens": sn.estimate(st["text"] or ""),
                            "notes_shrink": bool(shrink), "notes_truncated": bool(truncated)}
        return sn.render_block(shown, shrink, truncated) + sn.ask_block(c["max_words"])

    def _notes_capture(self, smiles, answer):
        """Store the <NOTES> block the model just wrote. An answer with no usable block leaves the previous
        note standing, so a bad write never destroys what the target already had."""
        c = self._notes_cfg()
        if not c["enabled"]:
            return
        st = self._notes_state()
        d = sn.parse_notes(answer)
        if d["usable"]:
            st["text"], st["fields"] = d["text"], d["fields"]
            st["writes"] += 1
            self._fx_bump("notes_written")
        else:
            self._fx_bump("notes_absent")
        self._log_notes(smiles, d)

    def _log_notes(self, smiles, d):
        """One line per route call -> aot_notes_<pid>.jsonl, so the note can be read back move by move."""
        try:
            st = self._notes_state()
            entry = {"ts": _dt.utcnow().isoformat(timespec="seconds"),
                     "target_idx": getattr(self, "tb_target_idx", -1),
                     "worker": threading.current_thread().name, "molecule": smiles,
                     "llm_calls": self._aot_llm_calls, "usable": d["usable"],
                     "fields": sorted(d["fields"]), "note": d["text"], "writes": st["writes"],
                     **(getattr(self, "_last_notes", None) or {})}
            path = os.path.join(self.args.output_dir, f"aot_notes_{os.getpid()}.jsonl")
            with _APPEND_LOCK, open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[aot] notes log failed: {e}")

    def _llm_route(self, smiles, tried=None, attempt=1):
        """One LLM call proposing a whole route for `smiles`. Counts toward the budget
        even if the answer is unusable. Returns a parsed, cleaned route or None and
        records why in self._last_route_outcome.

        Fix C (`aot.tried_note`): append the reactions the tree already holds for
        this molecule, in the GA modification prompt's own wording
        (modification_prompt.redundant_reaction), so repeat expansions of a molecule
        are asked for something new instead of re-proposing the same disconnection."""
        # both are stale-prone: when the joint discussion fires, _advisor_block never runs, and vice versa
        self._last_advisor = self._last_joint = None
        self._last_cuts = None
        if self._cfg_of("no_llm", NO_LLM_DEFAULTS)["enabled"]:     # arm nollm: the tool without the LLM
            self._aot_llm_calls += 1                                 # one expansion = one budget unit, as for Qwen
            return self._cut_route(smiles, tried)
        examples = self._sample_examples(smiles)
        initial_prompt, task_description, requirements = build_init_prompt_parts(smiles, examples)
        question = initial_prompt + requirements + task_description
        question += self._repair_hint(smiles, tried, attempt)
        if self.aot.get("tried_note") and tried:
            question += redundant_reaction([smiles], {smiles: list(tried)})
            self._fx_bump("tried_note_used")
        question += self._real_cuts_block(smiles)
        question += self._rag_cuts_block(smiles)                  # arm ragcut ("" when off)
        question += self._select_cuts_block(smiles)               # arm cutsel ("" when off)
        _ctc = self._cut_tool_cfg()
        if _ctc["enabled"]:                                        # arm cuttool / ctoolself
            question += self._cut_tool_note(_ctc)
        if self._ask_open():                                       # arm ask: only once the target is stuck
            question += advisor_ask.ASK_TOOL_NOTE
        if self._mine_tool_on():                                   # arm mine: how this molecule's motifs are MADE
            question += self._mine_tool_note()
        if self._mol_tool_on():                                    # arm moltool: RDKit facts, on demand
            question += MOL_TOOL_NOTE
        if self._image_tool_on():                                  # ... and the picture, on demand
            question += IMAGE_TOOL_NOTE
        if self._check_tool_on():                                  # arm check: the checker, on demand
            question += CHECK_TOOL_NOTE
        if self._validate_tool_on():                               # arm valtool: format check (always)
            question += VALIDATE_TOOL_NOTE
            if getattr(self, "_val_tool_rejected", None):          # ... rejection history, only once one exists
                question += REJECTED_TOOL_NOTE
        if self._sw_cfg()["enabled"] and self._sw_cfg()["history"]:
            question += sc.route_block(self._sw_path(smiles))
        question += self._plan9_block()
        agreed = self._joint_route(smiles, question)        # arm joint (try 11); None = not due / no agreement
        if agreed:
            question += jp.final_ask(agreed)
        else:
            question += self._advisor_block(smiles, question)
        question += self._notes_block(question)   # arm notes (try 12): the note, then the request to rewrite it
        self._aot_llm_calls += 1
        self._last_question = question
        return self._ask_route(question, smiles)

    def _ask_route(self, question, smiles=None):
        """One LLM call + parse (no budget accounting). Keeps the answer's route block in
        self._last_route_body for checker feedback."""
        self._last_route_body = None
        self._ask_turn_answer = None     # arm ask: one planner turn = one advisor answer (see _ask_advisor_result)
        _ct = self._cut_tool_cfg()
        _tools = []
        if self._ask_open():                       # arm ask: the pull advisor, offered only while stuck
            _tools.append(advisor_ask.ASK_TOOL_SCHEMA)
        if self._check_tool_on():                  # arm check: the route checker, on demand
            _tools.append(_VALIDATE_ROUTE_SCHEMA)
        if self._validate_tool_on():               # arm valtool: format check always; history only if one exists
            _tools.append(VALIDATE_TOOL_SCHEMA)
            if getattr(self, "_val_tool_rejected", None):
                _tools.append(REJECTED_TOOL_SCHEMA)
        if self._mine_tool_on():                   # arm mine: how this molecule's motifs are made
            _tools.append(self._mine_tool_schema())
        if self._mol_tool_on():                    # arm moltool: deterministic facts about this molecule
            _tools.append(MOL_TOOL_SCHEMA)
        if self._image_tool_on():                  # ... and its picture
            _tools.append(IMAGE_TOOL_SCHEMA)
        if _ct["enabled"]:                                                  # arm cuttool: only this one tool
            log = self.__dict__.setdefault("_cut_tool_log", [])
            n0 = len(log)
            # arm ctoolself: the no-argument tool is bound to the molecule under expansion by _expand (which spans
            # both this call and any checker-feedback repair, where smiles is None), so it is never bound here.
            _self_scope = _ct["scope"] == "self"
            _, answer = self.query_LLM(question, temperature=0.7,
                                       tools=([self._cut_tool_schema(_ct)] + _tools),
                                       tool_ctx=self)
            self._last_cuts = {"cut_tool_calls": len(log) - n0,
                               "cut_tool_on_this_molecule": sum(1 for x in log[n0:] if smiles and x["smiles"] ==
                                                                (sanitize_smiles(smiles) or smiles))}
        elif _tools:                                # arm ask without a cut tool: the advisor is the only tool
            _, answer = self.query_LLM(question, temperature=0.7, tools=_tools, tool_ctx=self)
        else:
            _, answer = self.query_LLM(question, temperature=0.7)
        if smiles is not None:
            self._notes_capture(smiles, answer)   # arm notes: before any route parsing can return early
        la = getattr(self, "_last_advisor", None)
        # only when a note was shown (advisor_n >= 1): a round that produced no note shows nothing
        if answer and isinstance(la, dict) and la.get("advisor_n") and self._adv_cfg()["plan_of_attack"]:
            la["advice_line"] = al.ap.parse_advice_line(answer) or "none"      # Qwen's FOLLOW / ADAPT / REJECT
        if not answer:
            self._last_route_outcome = "no_answer"
            return None
        body = extract_route_block(answer)
        self._last_route_body = body
        if body is None:
            self._last_route_outcome = "no_route_tag"     # includes runaways cut at max_tokens
            return None
        try:
            route = ast.literal_eval(body)
            if not isinstance(route, list) or not route or not isinstance(route[0], dict):
                self._last_route_outcome = "unparseable"
                return None
            self._last_route_outcome = "ok"
            return clean_last_step(route)
        except Exception:
            self._last_route_outcome = "unparseable"
            return None

    def _budget_left(self):
        if self._aot_llm_calls >= self.aot["llm_call_budget"]:
            return False
        deadline = getattr(self, "_attempt_deadline", None)
        return deadline is None or time.time() <= deadline

    # ------------------------------------------------------------------ mapping
    def _map_route(self, tree, or_node, route, base_depth):
        """Validate a route with the baseline validator and attach its steps.
        Returns the list of NEW AND nodes; per-step accounting goes to
        self._last_map_stats (fix A, logged per expansion)."""
        st = {"steps": len(route), "valid": 0, "attached_new": 0, "duplicate": 0,
              "unreachable": 0, "salvaged": 0, "invalid": 0}
        self._last_map_stats = st
        self._last_step1 = None
        self._last_final_reactants = None
        self._switch_pending = None
        track = self._adv_cfg()["enabled"] or self._sw_cfg()["enabled"]
        orig = copy.deepcopy(route) if track else None       # sanitize rewrites steps in place
        try:
            checked, evaluation = self.sanitize([or_node.smiles], route, True)
        except Exception:
            st["invalid"] = len(route)
            return []
        if evaluation:
            self._last_step1 = (bool(evaluation[0][1]), evaluation[0][2])   # for checker feedback
            if track and evaluation[0][1]:
                try:
                    self._last_final_reactants = ast.literal_eval(checked[0]["Reactants"])
                except Exception:
                    pass
        created = []
        reachable = {or_node.smiles: base_depth}      # molecule -> depth for its reactions

        def reach(a):
            for c in a.child_molecules:
                reachable.setdefault(c.smiles, a.depth + 1)

        for k, (_, ok, info) in enumerate(evaluation):
            step = checked[k]
            try:
                prod = extract_molecules_from_output(step["Product"])
                prod = sanitize_smiles(prod[0]) if prod else None
            except Exception:
                prod = None
            node = tree.molecules.get(prod) if prod in reachable else None
            if ok:
                st["valid"] += 1
            if ok and node is not None and not node.is_solved:
                try:
                    reactants = ast.literal_eval(step["Reactants"])
                except Exception:
                    continue
                a, new = tree.add_reaction(node, reactants, reachable[prod],
                                           reaction=info.get("reaction"))
                if a is not None:
                    # A re-proposed (merged) reaction still makes its reactants
                    # reachable, so the answer's later steps can attach under them.
                    reach(a)
                    if new:
                        if track:
                            self._note_edge_source(a, orig, k, reactants)
                        created.append(a)
                        st["attached_new"] += 1
                    else:
                        st["duplicate"] += 1
                continue
            if ok:
                st["unreachable"] += 1         # valid, but its product is not open in this route's subtree
                continue
            st["invalid"] += 1
            if k > 0 and self.aot["salvage"]:
                a = self._salvage(tree, route[k] if k < len(route) else step, prod, reachable)
                if a is not None:
                    if track:
                        self._adv_state()["src"][a.key] = "template"
                        self._sw_state()["src"][a.key] = {"src": sc.SRC_SALVAGE, "orig": None}
                    created.append(a)
                    st["salvaged"] += 1
                    reach(a)
        return created

    def _note_edge_source(self, a, orig, k, reactants):
        """Remember whether an attached reaction is the chemist's own or the checker's substitute."""
        try:
            mine = {sanitize_smiles(x) for x in ast.literal_eval(orig[k]["Reactants"])}
        except Exception:
            mine = None
        kept = {sanitize_smiles(x) for x in reactants}
        self._adv_state()["src"][a.key] = "chemist" if mine == kept else "template"
        forced = getattr(self, "_switch_forced", None)
        if mine == kept:
            rec = {"src": sc.SRC_OWN, "orig": None}
        else:
            try:
                own = (ast.literal_eval(orig[k]["Reaction"]) or [None])[0]
            except Exception:
                own = None
            rec = {"src": sc.SRC_CHOSEN if (forced and k == 0) else sc.SRC_TEMPLATE, "orig": own}
        self._sw_state()["src"][a.key] = rec

    def _salvage(self, tree, step, llm_product, reachable):
        """Released AOT* salvage (skip-and-continue + aggressive product matching):
        a later step that failed in the chain (the previous step was invalid or its
        reactants were auto-corrected) is re-targeted at a molecule actually in this
        route's reachable set -- the exact canonical match first, otherwise ranked by
        0.7*Tanimoto + 0.3*string similarity -- and accepted only if the step's
        reaction matches a USPTO template (exact lookup, as for later chain steps)
        that applies to that molecule."""
        try:
            reaction = ast.literal_eval(step["Reaction"])[0]
        except Exception:
            return None
        found, key = is_reaction_in_dict(reaction, self.template_dict)
        if not found or key is None:
            return None
        template = self.original_template_dict[key]
        open_nodes = [s for s in reachable if s in tree.molecules and not tree.molecules[s].is_solved]
        if not open_nodes:
            return None
        if llm_product in open_nodes:
            ranked = [llm_product]
        else:
            pfp = _fp(llm_product) if llm_product else None
            if pfp is None:
                return None
            w = self.aot["salvage_tanimoto_weight"]
            scored = []
            for s in open_nodes:
                sfp = _fp(s)
                if sfp is None:
                    continue
                sim = w * DataStructs.TanimotoSimilarity(pfp, sfp) + \
                    (1 - w) * SequenceMatcher(None, llm_product, s).ratio()
                scored.append((sim, s))
            ranked = [s for _, s in sorted(scored, reverse=True)]
        for s in ranked:
            try:
                outs = run_retro(rdchiralReactants(s), template)
                if not outs:
                    continue
                if len(outs) > 1:
                    outs = self.rank_reactants(outs)
                reactants = [sanitize_smiles(r) for r in outs[0]]
                if not reactants or any(r is None for r in reactants):
                    continue
            except Exception:
                continue
            a, new = tree.add_reaction(tree.molecules[s], reactants, reachable[s], reaction=template)
            if a is not None and new:
                return a
        return None

    # ------------------------------------------------------------------ scoring
    def _step_quality(self, a):
        """Arm stepq: the factor `_score` is multiplied by, 1.0 unless the step barely simplified the
        molecule. Config absent or disabled -> 1.0, i.e. the released score exactly. See
        STEP_QUALITY_DEFAULTS for the measurement this answers."""
        c = self._cfg_of("step_quality", STEP_QUALITY_DEFAULTS)
        if not c["enabled"]:
            return 1.0
        parent = getattr(a, "parent_molecule", None)
        kids = [k.smiles for k in getattr(a, "child_molecules", [])]
        if parent is None or not kids:
            return 1.0
        whole = self._heavy(parent.smiles)
        if whole <= 0:
            return 1.0                      # unparseable parent: no measurement, so no penalty
        if max(self._heavy(k) for k in kids) < float(c["near"]) * whole:
            return 1.0
        self._fx_bump("stepq_trim")
        return float(c["penalty"])

    def _branch_shrink(self, a):
        """Arm branchq: `shrink_rate` for the largest UNSOLVED piece this step produces -- the share of the
        TARGET's mass removed per step taken so far, which is the unit the mined evidence used.
        -> None when it cannot be measured (no target, unparseable, above nothing, shallower than
        `min_depth`), and a measurement is never invented: an unmeasurable branch is never penalised."""
        c = self._cfg_of("branch_quality", BRANCH_QUALITY_DEFAULTS)
        if not c["enabled"]:
            return None
        target = getattr(self, "_aot_target", "") or ""
        whole = self._heavy(target)
        if whole <= 0:
            return None
        unsolved = [k.smiles for k in getattr(a, "child_molecules", []) if not k.is_solved]
        if not unsolved:
            return None                     # a finished step: every piece purchasable, nothing to judge
        sizes = [self._heavy(s) for s in unsolved]
        if not sizes or max(sizes) <= 0:
            return None                     # nothing parseable: no measurement, so no penalty
        # the piece's OWN depth: the step sits at a.depth, so what it produces is one level deeper
        depth = int(getattr(a, "depth", 0) or 0) + 1
        if depth < int(c["min_depth"]):
            return None
        return (1.0 - (max(sizes) / float(whole))) / float(depth)

    def _branch_quality(self, a):
        """The factor `_score` is multiplied by for arm branchq: 1.0 unless this branch has removed too
        little of the target for how deep it already is. See BRANCH_QUALITY_DEFAULTS for the evidence."""
        sh = self._branch_shrink(a)
        if sh is None:
            return 1.0
        c = self._cfg_of("branch_quality", BRANCH_QUALITY_DEFAULTS)
        self._fx_bump("branchq_seen")
        if sh > float(c["floor"]):
            return 1.0
        self._fx_bump("branchq_trim")
        return float(c["penalty"])

    def _score(self, tree, a):
        """Released `_evaluate_and_node_chemical` (chemistry only, depth_decay 1), times the arm-stepq and
        arm-branchq factors (both 1.0 when those arms are off, so this is the release)."""
        unsolved = [c.smiles for c in a.child_molecules if not c.is_solved]
        if not unsolved:
            return 1.0                      # every piece purchasable: a finished step, never penalised
        try:
            r = self.oracle.reward(self.inventory, unsolved, self.visited_molecules, self.dead_molecules)
            s = max(0.0, 1.0 + r / 14.0)
        except Exception:
            s = 0.4
        return s * self._step_quality(a) * self._branch_quality(a)

    def _evaluate_new(self, tree, nodes):
        for a in nodes:
            if self.aot.get("junk_guard"):
                # a dead child dooms the reaction (has_expandable_reactants), so it is never selected
                for c in a.child_molecules:
                    if not c.is_solved and c.smiles not in tree.dead and junk_molecule(c.smiles):
                        tree.dead.add(c.smiles)
                        self.update_dead_molecules(c.smiles)
                        self._fx_bump("junk_dead")
            s = self._score(tree, a)
            tree.init_stats(a, s)
            tree.backpropagate(a, s)

    def _refresh_solved(self, tree):
        """Solved propagation + released `_reevaluate_affected_and_nodes`."""
        newly = tree.update_solved()
        for o in newly:
            for p in o.parent_reactions:
                s = self._score(tree, p)
                p.feasibility_score = s
                p.visit_count += 1
                p.total_value += s
                p.average_value = p.total_value / p.visit_count
                if tree.backprop == "andor":
                    tree.backpropagate(p, s)
        return newly

    # ------------------------------------------------------------------ expansion
    def _expand(self, tree, or_node, depth, retries):
        created = []
        tried = [f"{or_node.smiles}>>{'.'.join(a.reactants)}" for a in or_node.child_reactions]
        if self._sw_cfg()["enabled"] and self._sw_cfg()["history"]:
            src = self._sw_state()["src"]
            tried = [sc.annotate_tried(t, src.get(a.key)) for t, a in zip(tried, or_node.child_reactions)]
        # arm ctoolself: the no-argument tool answers for the molecule under expansion. Bind it HERE, not in
        # _ask_route, so it stays bound across BOTH the main ask (_llm_route) and any checker-feedback repair
        # (which calls _ask_route without a smiles arg); cleared in finally so it never leaks to the next node.
        _self_tool = self._cut_tool_cfg()["enabled"] and self._cut_tool_cfg()["scope"] == "self"
        if _self_tool:
            self._cut_tool_self_smiles = or_node.smiles
        # arm moltool: same binding discipline as ctoolself -- bound here so it covers the main route call
        # AND any checker-feedback repair, cleared in finally so it never leaks to the next molecule.
        _mol_tool = self._mol_tool_on()
        if _mol_tool:
            self._mol_tool_smiles = or_node.smiles
            self._mol_tool_calls = 0
            self._image_tool_calls = 0
            self._image_tool_pending = None
        # arm valtool: check_my_route answers for the molecule under expansion and accumulates the first steps
        # rejected for it this expansion; reset here (covers the main call and any repair) and cleared in finally
        # so neither leaks to the next molecule.
        _val_tool = self._validate_tool_on()
        if _val_tool:
            self._val_tool_smiles = or_node.smiles
            self._val_tool_rejected = []
        # arm ask (user, 2026-11-16: "go and continue is separate"): the advisor's shortlist must not offer the
        # molecule the planner is on right now. Staying is CONTINUE; every GO must point somewhere else. Bound
        # for the whole expansion and cleared in finally, as for ctoolself above.
        _ask_on = self._ask_cfg()["enabled"]
        if _ask_on:
            self._ask_expanding = or_node.smiles
            self._ask_expanding_depth = depth           # shown to the advisor beside its destinations
            self._ask_expanding_tried = len(or_node.child_reactions)
            self._ask_continue = False          # set by _ask_advisor_result on an explicit CONTINUE
            self._ask_stop = False              # ... and STOP, when aot.advisor_ask.allow_stop is on
        try:
            # arm ask: `limit` can grow ONCE, by continue_extra_tries, when the advisor said CONTINUE in this
            # expansion and the normal tries are used up. Without the arm this is exactly range(1, retries + 1).
            limit, extended, attempt = retries, False, 0
            while attempt < limit:
                attempt += 1
                if not self._budget_left():
                    break
                # arm askR: the advisor answered STOP for this molecule. Give up the remaining tries now.
                # The main loop's existing `failed` path then marks it dead and retires the reaction -- the
                # same route a genuinely exhausted expansion takes, so no new retirement logic is needed.
                if getattr(self, "_ask_stop", False):
                    self._fx_bump("ask_stop_honoured")
                    break
                self._last_map_stats = {}
                self._last_step1 = None
                route = self._llm_route(or_node.smiles, tried=tried, attempt=attempt)
                snap = copy.deepcopy(route) if route else None     # sanitize rewrites steps in place
                if route:
                    created = self._map_route(tree, or_node, route, depth)
                    if self._sw_cfg()["enabled"] and getattr(self, "_switch_pending", None):
                        created = created + self._switch_choose(tree, or_node, depth, snap)
                # arm retrytried: tell the NEXT attempt what this one offered and lost. Released
                # behaviour (off) rebuilds nothing, so attempt N+1 sees attempt N's exact prompt.
                if self.aot.get("retry_tried") and not created:
                    try:
                        rj = self._rejected_step(snap)
                        if rj and rj not in tried:
                            tried = tried + [rj]
                            self._fx_bump("retry_tried_added")
                    except Exception as e:
                        print(f"[aot] retry_tried could not record the rejection: {e}")
                # arm valtool: remember each rejected first step WITH the reason the checker already gave, so
                # my_rejected_steps() can show both. Reason read from _last_step1 (no fresh checker call).
                if _val_tool and not created:
                    try:
                        rj = self._rejected_step(snap)
                        if rj and all(x.get("step") != rj for x in self._val_tool_rejected):
                            self._val_tool_rejected.append({"step": rj, "reason": self._rejection_reason()})
                    except Exception:
                        pass
                self._record_attempt(or_node, snap, created)
                if _ask_on:                     # after the call, so the try in which CONTINUE was said shows it
                    self._last_ask = {"ask_continue": bool(getattr(self, "_ask_continue", False)),
                                      "ask_extra_try": attempt > retries}
                self._log_expansion(or_node, depth, attempt, created, len(tried))
                if created:
                    break
                if route is None and getattr(self, "_last_route_outcome", None) == "no_cut":
                    break       # arm nollm: no split left for this molecule; further retries would only charge
                if self.aot.get("checker_feedback"):
                    created = self._checker_feedback(tree, or_node, depth, attempt, snap, len(tried))
                    if created:
                        break
                if (_ask_on and attempt == limit and not extended and getattr(self, "_ask_continue", False)
                        and int(self._ask_cfg()["continue_extra_tries"]) > 0):
                    # 2026-11-17: only extend where there is something to work with. On arm `ask` both
                    # extensions landed on one molecule with 0 accepted splits and attached 0 of 6. Unknown
                    # (never checked) does not qualify either -- it is not evidence that options exist.
                    if not self._ask_current_splits():
                        self._fx_bump("ask_extend_blocked_no_splits")
                    else:
                        limit += int(self._ask_cfg()["continue_extra_tries"])
                        extended = True
                        self._fx_bump("ask_continue_extended")
            return created
        finally:
            # arm askR: ENFORCE the STOP here. Relying on the main loop's `failed` path was not enough --
            # that path needs `nodes` to come back empty, so a STOP given after the expansion had already
            # attached something would have been recorded and then ignored. Marking the molecule dead is
            # what actually binds: expansion_target() skips dead molecules and has_expandable_reactants()
            # stops offering any reaction whose unsolved reactant is dead, so the branch leaves the
            # frontier for good whatever else happened in this expansion.
            if _ask_on and getattr(self, "_ask_stop", False):
                try:
                    tree.dead.add(or_node.smiles)
                    self.update_dead_molecules(or_node.smiles)
                    self._fx_bump("ask_stop_enforced")
                    self._ask_log({"event": "stop_enforced", "molecule": or_node.smiles,
                                   "depth": depth, "calls": self._aot_llm_calls,
                                   "attached_before_stop": len(created)})
                except Exception as e:
                    print(f"[askR] stop enforcement failed: {e}")
            if _mol_tool:
                self._mol_tool_smiles = None
            if _val_tool:
                self._val_tool_smiles = None
            if _self_tool:
                self._cut_tool_self_smiles = None
            if _ask_on:
                self._ask_expanding = None
                self._ask_expanding_depth = None
                self._ask_expanding_tried = 0
                self._ask_continue = False
                self._last_ask = None

    # ------------------------------------------------------------------ checker feedback
    def _time_left(self):
        deadline = getattr(self, "_attempt_deadline", None)
        return deadline is None or time.time() <= deadline

    def _checker_feedback(self, tree, or_node, depth, attempt, route, n_tried):
        """Config `aot.checker_feedback` (2026-10-14): when an answer gave the tree nothing
        because of a diagnosable error (format, unreadable SMILES, or step 1 rejected), ask
        again with the checker's diagnostics, the rejected route and the closest valid
        disconnections the checker found (checker_feedback.py). Up to
        `checker_feedback_max` follow-ups; they are NOT counted in llm_call_budget (user
        decision) but are counted in fix_stats.feedback_calls and logged with `feedback`."""
        import checker_feedback as cf
        question = getattr(self, "_last_question", None)
        created = []
        for k in range(1, int(self.aot.get("checker_feedback_max", 1)) + 1):
            if question is None or not self._time_left():
                break
            try:
                problems, closest = self._diagnose(or_node.smiles, route)
            except Exception as e:           # diagnostics must never cost the search anything
                print(f"[aot] checker feedback diagnosis failed: {e}")
                problems, closest = [], []
            if not problems:
                break
            prev, body = getattr(self, "_last_route_outcome", None), getattr(self, "_last_route_body", None)
            self._fx_bump("feedback_calls")
            self._last_map_stats, self._last_step1 = {}, None
            route = self._ask_route(cf.build_repair_prompt(question, or_node.smiles, body, problems, closest))
            snap = copy.deepcopy(route) if route else None
            created = self._map_route(tree, or_node, route, depth) if route else []
            self._last_feedback = {"feedback": k, "feedback_of": prev,
                                   "feedback_problems": len(problems), "feedback_closest": len(closest)}
            self._log_expansion(or_node, depth, attempt, created, n_tried)
            self._last_feedback = None
            if created:
                self._fx_bump("feedback_attached")
                break
            route = snap
        return created

    def _diagnose(self, node, route):
        """(problems, closest) for the answer just mapped; ([], []) = nothing to fix."""
        import checker_feedback as cf
        outcome = getattr(self, "_last_route_outcome", None)
        body = getattr(self, "_last_route_body", None)
        if outcome == "no_route_tag":
            return cf.diagnose_format(None), []
        if outcome == "unparseable":
            return (cf.diagnose_format(body) or
                    ["The <ROUTE> block could not be read as a list of steps."]), []
        if outcome != "ok" or not route:
            return [], []                       # no answer at all: nothing to diagnose
        s1 = getattr(self, "_last_step1", None)
        if s1 is None:                          # the validator raised on a malformed step
            return (cf.diagnose_format(body) or
                    ["A step is malformed: every step needs the fields " +
                     ", ".join(f"'{f}'" for f in cf.LIST_FIELDS) + "."]), []
        ok, info = s1
        if ok:
            return [], []                       # step 1 valid (e.g. a duplicate): not an error
        problems = cf.step1_smiles_problems(route[0])
        if not info.get("product_inside", True):
            problems.append(f"Step 1 'Product' must be exactly the molecule to decompose, {node}; "
                            f"you wrote {info.get('product')}.")
        if not problems:
            try:
                rxn = ast.literal_eval(route[0]["Reaction"])[0]
            except Exception:
                rxn = route[0].get("Reaction")
            if not info.get("reaction_existance"):
                problems.append(f"No USPTO reaction template matches step 1's reaction ({rxn}), and no "
                                f"similar known reaction applies to {node}.")
            elif not info.get("reaction_valid"):
                problems.append(f"The closest USPTO template to step 1's reaction ({rxn}) does not apply "
                                f"to {node}: its reaction pattern is not present in this molecule.")
            elif not info.get("updated_set_valid"):
                problems.append(f"A USPTO template applies to {node}, but its products differ from your "
                                f"reactants {info.get('reactants')}.")
            else:
                problems.append("Step 1 was rejected by the route checker.")
        return problems, self._closest_disconnections(node, info)

    def _closest_disconnections(self, node, info, k=2):
        """Outcomes of the template the checker itself matched to step 1 (exact lookup or
        blurry_search), applied to the molecule: the nearest disconnection known to be valid."""
        tpl = info.get("reaction") if info.get("reaction_existance") else None
        if not tpl or ">>" not in str(tpl):
            return []
        try:
            outs = run_retro(rdchiralReactants(node), tpl)
            if len(outs) > 1:
                outs = self.rank_reactants(outs)
        except Exception:
            return []
        res = []
        for o in outs[:k]:
            r = [sanitize_smiles(x) for x in o]
            if r and None not in r:
                res.append(f"{node}>>{'.'.join(r)}")
        return res

    def _log_expansion(self, or_node, depth, attempt, created, n_tried):
        """Fix A: one line per LLM call of an expansion -> aot_expansions_<pid>.jsonl."""
        # arm `ask`: the stall counter ticks HERE, once per LLM call, because that is the unit the
        # gate was measured on. It was first wired per EXPANSION, which bundles up to
        # max_expansion_retries calls and counts as progress if any one of them attaches -- in that
        # unit the longest dry run on the 15-target set is 2, so a gate of 5 could never open and the
        # advisor was never offered once in a full run. Counted per call the same set reaches 5 on 6
        # of 15 targets. (2026-11-16)
        self._ask_note_dry(bool(created))
        try:
            entry = {
                "ts": _dt.utcnow().isoformat(timespec="seconds"),
                "target_idx": getattr(self, "tb_target_idx", -1),
                "worker": threading.current_thread().name,
                "molecule": or_node.smiles, "is_root": or_node is self._aot_tree.root,
                "depth": depth, "attempt": attempt, "llm_calls": self._aot_llm_calls,
                "outcome": getattr(self, "_last_route_outcome", None),
                "n_tried_before": n_tried, "new_reactions": len(created),
                **(getattr(self, "_last_map_stats", None) or {}),
                **(getattr(self, "_last_rag", None) or {}),
                **(getattr(self, "_last_skill", None) or {}),
                **(getattr(self, "_last_feedback", None) or {}),
                **(getattr(self, "_last_advisor", None) or {}),
                **(getattr(self, "_last_plan9", None) or {}),
                **(getattr(self, "_last_rc", None) or {}),
                **(getattr(self, "_last_ragcut", None) or {}),
                **(getattr(self, "_last_cuts", None) or {}),
                **(getattr(self, "_last_prune", None) or {}),
                **(getattr(self, "_last_sel", None) or {}),
                **(getattr(self, "_last_joint", None) or {}),
                **(getattr(self, "_last_ask", None) or {}),
            }
            # _last_prune / _last_sel describe the ITERATION, but this runs once per attempt; consume them so
            # retries do not re-log the same marks and picks (code review, 1 Nov: it multiplied the counts)
            self._last_prune = self._last_sel = None
            if self._steer_active():                     # arm steer: what the last expansion of each molecule did
                k = "attached" if created else ("format" if entry["outcome"] not in (None, "ok") else "rejected")
                self.__dict__.setdefault("_steer_tries", {}).setdefault(or_node.smiles, []).append((attempt, k))
            path = os.path.join(self.args.output_dir, f"aot_expansions_{os.getpid()}.jsonl")
            with _APPEND_LOCK, open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[aot] expansion log failed: {e}")

    # ------------------------------------------------------------------ search
    def _optimize(self, target, route_list, all_fps, config):
        self._rag_all_fps = all_fps
        self._rag_route_list = route_list
        self._current_config = config
        raw_target = target                      # file names use the target as given
        target = sanitize_smiles(target) or target

        resume = getattr(self, "_resume_state", None)
        self._resume_state = None
        if resume is not None and resume.get("stage") == "aot":
            self._ckpt_restore_common(resume)
            tree = resume["tree"]
            tree.canon, tree.purchasable = sanitize_smiles, self._purchasable
            tree.propagate_failure = bool(self._term_cfg()["propagate_failure"])   # termination fix (config wins)
            tree.propagate_from = self._term_cfg()["propagate_from"]
            self._aot_llm_calls = resume["llm_calls"]
            self._fx = dict(resume.get("fix_stats") or {})   # counters survive walltime resume
            self._adv = resume.get("advisor_state") or {"hist": {}, "advice": {}, "src": {}}
            self._sw = resume.get("switch_state") or {"src": {}}
            # arm notes: the target's note must survive a walltime cut, or the second chunk starts blind
            self._nt = resume.get("notes_state") or {"text": None, "fields": {}, "writes": 0}
            # arm ask (review 2026-11-16): optimize() calls reset() before this, which zeroes all of it. Without
            # the restore a target crossing a 12h boundary got a fresh 6-call cap (up to 12 on one target), lost
            # the advisor's notes -- so it could re-send the search somewhere it had already ruled out -- and had
            # its gate shut until it re-earned `dry_gate` dry calls. The run is on the 15 SLOWEST targets, which
            # are exactly the ones that cross boundaries. _ask_pending is a node of `tree`; both are in the same
            # pickle, so its identity survives, and the main loop re-validates it against the frontier anyway.
            _rs = resume.get("reopen_state") or {}
            self._reopens = int(_rs.get("reopens", 0))
            self._reopen_asks = int(_rs.get("asks", 0))
            _as = resume.get("ask_state")
            if _as:
                self._ask_dry = int(_as.get("dry", 0))
                self._ask_calls = int(_as.get("calls", 0))
                self._ask_notes = list(_as.get("notes") or [])
                self._ask_recent = list(_as.get("recent") or [])
                self._ask_pending = _as.get("pending")
            init_done, iteration = resume["init_done"], resume["iteration"]
        else:
            tree = self._new_tree(target)
            self._aot_llm_calls = 0
            self._reopens = self._reopen_asks = 0     # arm reopen: per target, never carried between them
            self._adv = {"hist": {}, "advice": {}, "src": {}}
            self._sw = {"src": {}}
            self._nt = {"text": None, "fields": {}, "writes": 0}   # arm notes: fresh note per target
            init_done, iteration = 0, 0
        self._aot_tree = tree
        self._dc = {}                            # disconnections caches (try 10 / arm rc): per target, not checkpointed
        # switch choice: the checker (optimizer._step1_substitute) stops switching silently
        self._switch_choice_on = self._sw_cfg()["enabled"]
        self._switch_max_options = int(self._sw_cfg()["max_options"])
        self._switch_forced = self._switch_pending = None
        self._aot_solved_at = None
        self._aot_term_reason = None             # termination fix: why an unsolved search stopped early
        self._aot_target = raw_target            # carried on every aot_<kind>_<pid>.jsonl row
        steer = self._steer_cfg()
        self._steer_on = bool(steer["enabled"])  # arm steer: per-target state
        self._steer_pos, self._steer_tries, self._steer_cache, self._steer_streak = None, {}, {}, {}
        if self._steer_on and resume is not None and resume.get("stage") == "aot" and resume.get("steer_state"):
            # a walltime cut between an expansion and its decision must not lose the deferred write-off (review)
            ss = resume["steer_state"]
            self._steer_streak = dict(ss.get("streak") or {})
            sp = ss.get("pos")
            if sp and sp["mol"] in tree.molecules and sp["parent"] in tree.reactions:
                self._steer_pos = {"mol": tree.molecules[sp["mol"]], "parent": tree.reactions[sp["parent"]],
                                   "created": list(sp["created"]), "failed": bool(sp["failed"])}
        term = self._term_cfg()
        if term["no_split_dead"] and not tree.root.is_solved and self._check_no_split(tree, [tree.root], 0):
            self._aot_term_reason = "root_proven_dead"   # no checker template decomposes the target at all

        def checkpoint():
            # Deep AND-OR graphs can exceed the default pickle recursion depth.
            sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
            self._ckpt_save("aot", tree=tree, llm_calls=self._aot_llm_calls,
                            init_done=init_done, iteration=iteration,
                            fix_stats=dict(getattr(self, "_fx", {})),
                            advisor_state=(self._adv_state() if (self._adv_cfg()["enabled"] or self._cfg_of(
                                "advisor_select", ADV_SELECT_DEFAULTS)["enabled"]) else None),   # advsel uses the hist too
                            switch_state=self._sw_state() if self._sw_cfg()["enabled"] else None,
                            notes_state=self._notes_state() if self._notes_cfg()["enabled"] else None,
                            steer_state=self._steer_ckpt() if self._steer_on else None,
                            ask_state=({"dry": int(getattr(self, "_ask_dry", 0)),
                                        "calls": int(getattr(self, "_ask_calls", 0)),
                                        "notes": list(getattr(self, "_ask_notes", []) or []),
                                        "recent": list(getattr(self, "_ask_recent", []) or [])[-50:],
                                        "pending": getattr(self, "_ask_pending", None)}
                                       if self._ask_cfg()["enabled"] else None),
                            # arm reopen: without this a target crossing a 12h boundary gets a fresh
                            # max_reopens and a fresh threshold, exactly the fault fixed for arm ask.
                            reopen_state=({"reopens": int(getattr(self, "_reopens", 0)),
                                           "asks": int(getattr(self, "_reopen_asks", 0))}
                                          if self._reopen_cfg()["enabled"] else None))

        # Initial expansion: several independent root calls for diverse directions.
        while (init_done < self.aot["num_initial_routes"] and not tree.root.is_solved
               and self._aot_term_reason is None
               and self._budget_left()):
            nodes = self._expand(tree, tree.root, 0, self.aot["max_root_retries"])
            self._evaluate_new(tree, nodes)
            init_done += 1
            checkpoint()

        while (self._budget_left() and iteration < self.aot["max_iterations"]
               and self._aot_term_reason is None):
            self._refresh_solved(tree)
            if tree.root.is_solved:
                break
            iteration += 1
            if self._reopen_due() and self._reopen_now(tree):   # arm reopen (no-op when off)
                checkpoint()
                continue
            self._prune_review(tree, iteration)      # arm advprune (no-op when off)
            a = self._advisor_select(tree, iteration)   # arm advsel (None when off/shadow)
            forced_target = None
            if a is None and self._steer_active():       # arm steerRgate: out until the stall gate opens
                a, forced_target = self._steer_choose(tree, iteration)   # arm steer ((None, None) = search's pick)
            if a is None and getattr(self, "_ask_pending", None) is not None:
                a, self._ask_pending = self._ask_pending, None      # arm ask: honoured once, then released
                if a not in tree.leaf_and_nodes or a.is_fully_solved():
                    a = None
            if a is None:
                a = tree.select()
            if a is None:
                # Frontier empty: re-expand the root (released behaviour).
                self._evaluate_new(tree, self._expand(tree, tree.root, 0, self.aot["max_root_retries"]))
                checkpoint()
                continue
            if term["propagate_failure"] and tree.last_term:
                self._log_term(dict(event="iteration", iteration=iteration, **tree.last_term))
            if term["no_split_dead"] and self._check_no_split(
                    tree, [c for c in a.child_molecules if not c.is_solved], a.depth + 1):
                tree.retire(a)       # a reactant is proven dead: this reaction can never be solved; no call spent
                checkpoint()
                continue
            target_node = forced_target or tree.expansion_target(a)   # arm steer SIDEWAYS forces the molecule
            if target_node is None:
                tree.retire(a)
                checkpoint()
                continue
            self.update_visited_molecules([target_node.smiles])
            nodes = self._expand(tree, target_node, a.depth + 1, self.aot["max_expansion_retries"])
            self.__dict__.setdefault("_ask_recent", []).append(
                {"molecule": target_node.smiles,
                 "outcome": (f"attached {len(nodes)} new reaction(s)" if nodes else "nothing accepted")})
            failed = not nodes and self._budget_left()
            if failed and not self._steer_active():
                # Genuinely failed (not just out of budget): the molecule is dead.
                tree.dead.add(target_node.smiles)
                self.update_dead_molecules(target_node.smiles)
                tree.retire(a)
            self._evaluate_new(tree, nodes)
            if self._steer_active():
                # arm steer: the advisor decides BEFORE the molecule is written off, so "try A again or move on"
                # (the user's own question) is a decision it can make. _steer_choose applies the release's
                # dead-marking itself whenever it does not choose SIDEWAYS.
                self._steer_pos = {"mol": target_node, "parent": a, "created": [x.key for x in nodes],
                                   "failed": failed}
                # back-to-back failed expansions of this molecule (caps SIDEWAYS after failure)
                self._steer_streak[target_node.smiles] = (self._steer_streak.get(target_node.smiles, 0) + 1
                                                          if failed else 0)
            checkpoint()

        self._refresh_solved(tree)
        if tree.root.is_solved:
            self._aot_solved_at = self._aot_llm_calls
            sol = tree.solution()
            # Record the solved tree so optimize()'s PASS test (a 0.0 buffer entry) sees it.
            self.oracle.route_buffer[json.dumps(sol, sort_keys=True)] = [0.0, self._aot_llm_calls]
        self._write_tree(raw_target)

    def _tree_dump(self, tree):
        """EVERY molecule and EVERY reaction in the finished search, flat, for post-hoc analysis.

        Added 2026-11-18 (user: "Why are we not keeping everything?"). Until now the only record kept of a
        finished search was its WINNING route: `solution_steps` plus aggregate stats. The full AND/OR tree
        lived in the resume checkpoint, which `Optimizer._ckpt_clear` deletes the moment an attempt
        finishes -- correctly, because a stale checkpoint would make a completed molecule resume -- so
        every branch that was tried and abandoned was lost. That is precisely the record needed to learn
        when a line of work is hopeless: a solved tree tells you what worked, and nothing told you what
        did not. The three arms analysed on 2026-11-18 could only be mined for per-expansion COUNTS.

        Flat lists, not the nested shape `solution` uses, because every consumer of this is a tabulation:
        one row per reaction with its product, its pieces, its template, where it sat and how it scored.
        Cheap: a hard target's tree is a few hundred nodes, so this is tens of KB per molecule.
        """
        try:
            on_path = set()
            for prod, reacts in (tree.solution_steps() or []):
                on_path.add((prod, tuple(sorted(reacts))))
            mols, rxns, seen_m, seen_r = [], [], set(), set()
            stack, seen_nodes = [tree.root], set()
            while stack:
                o = stack.pop()
                if id(o) in seen_nodes:
                    continue
                seen_nodes.add(id(o))
                if o.smiles not in seen_m:
                    seen_m.add(o.smiles)
                    mols.append({"smiles": o.smiles, "solved": bool(o.is_solved),
                                 "dead": o.smiles in tree.dead,
                                 "n_reactions": len(o.child_reactions)})
                for a in o.child_reactions:
                    k = getattr(a, "key", None)
                    if k in seen_r:
                        continue
                    seen_r.add(k)
                    pieces = [c.smiles for c in a.child_molecules]
                    rxns.append({"key": k, "product": o.smiles, "pieces": pieces,
                                 "template": getattr(a, "reaction", None),
                                 "depth": a.depth, "visits": a.visit_count,
                                 "value": round(float(a.average_value), 4),
                                 "score": round(float(a.feasibility_score), 4),
                                 "solved": bool(a.is_fully_solved()),
                                 "on_path": (o.smiles, tuple(sorted(pieces))) in on_path})
                    stack.extend(a.child_molecules)
            return {"molecules": mols, "reactions": rxns,
                    "dead_molecules": sorted(tree.dead),
                    "root": tree.root.smiles}
        except Exception as e:                      # a dump failure must never cost us the result file
            print(f"[aot] tree dump failed: {e}")
            return {"error": str(e)[:200]}

    def _write_tree(self, target):
        tree = self._aot_tree
        suffix = ("%s_%s_%s" % (self.model_name, target, getattr(self, "seed", 0))).replace("/", "")
        path = os.path.join(self.args.output_dir, "results_" + suffix + ".tree.json")
        out = {"solved": tree.root.is_solved, "llm_calls": self._aot_llm_calls,
               "solution": tree.solution(), "solution_steps": tree.solution_steps(),
               "stats": tree.stats(), "full_tree": self._tree_dump(tree)}
        try:
            with open(path, "w") as f:
                json.dump(out, f, indent=1)
        except Exception as e:
            print(f"[aot] failed to write tree: {e}")

    def write_done_marker(self, suffix, n_attempts=1, solved_on_attempt=None, termination=None):
        """Same files/fields as the baseline; status from the tree. BUDGET = the
        LLM-call budget was used up."""
        tree = getattr(self, "_aot_tree", None)
        solved = bool(tree is not None and tree.root.is_solved)
        calls = getattr(self, "_aot_llm_calls", 0)
        if solved:
            status, termination = "PASS", "found_pass"
        elif calls >= self.aot["llm_call_budget"]:
            status, termination = "BUDGET", "llm_call_budget"
        else:
            status = "EARLY_TERM"
            # termination fix: a proven-dead target outranks optimize()'s generic "budget_or_iters"
            termination = getattr(self, "_aot_term_reason", None) or termination or "search_exhausted"
        meta = {
            "status": status, "termination": termination, "passed": solved,
            "search": "aot", "llm_calls": calls,
            "llm_call_budget": self.aot["llm_call_budget"],
            "pass_steps": getattr(self, "_aot_solved_at", None) if solved else None,
            "n_routes": len(tree.reactions) if tree else 0,
            "tree": tree.stats() if tree else None,
            "fix_stats": dict(getattr(self, "_fx", {})),
            "n_attempts": n_attempts, "solved_on_attempt": solved_on_attempt,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
        }
        base = os.path.join(self.args.output_dir, "results_" + suffix.replace("/", ""))
        with open(base + ".status.json", "w") as f:
            json.dump(meta, f, indent=2)
        self._derive_skill(tree, solved, base)     # WRITE-ONLY; see _derive_skill. Before .done so a
        # ...and re-dump, because `meta` was built BEFORE derivation ran, so every counter the admission
        # rules bump (derive_declined / _skipped_cap / _failed / ...) was invisible in the results. That is
        # how a crash on all 32 eligible molecules went unnoticed for two full runs (2026-11-24). The status
        # file is still written first, so a walltime cut here cannot lose an already-scored outcome.
        meta["fix_stats"] = dict(getattr(self, "_fx", {}))
        with open(base + ".status.json", "w") as f:
            json.dump(meta, f, indent=2)
        open(base + ".done", "w").close()          # walltime cut between the two cannot lose the skill.
