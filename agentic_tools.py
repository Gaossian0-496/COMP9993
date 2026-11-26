"""Agentic validation tools for Run G (native SGLang tool-calling).

These tools let the LLM validate the pieces of a retrosynthesis route *while it is
drafting* — propose -> check -> correct, in-context — instead of generating a full
route blind and only learning it is invalid after sanitize()/the oracle.

Every tool is a thin wrapper over validators that already exist in utils.py /
optimizer.py; no chemistry logic is reimplemented here.

Budget integrity (critical): none of these tools may write to oracle.route_buffer.
- check_reaction_in_uspto / check_purchasable / validate_smiles / run_retro: pure.
- validate_route: runs Optimizer.sanitize() (check_route/map_reaction/...), which scores
  *nothing* — scoring is a separate self.rewards() call in run.py.
- estimate_complexity: wraps Oracle.get_oracle_score(), which only *reads*
  len(route_buffer); it never appends.
Full-route scoring (Oracle.score_route / evaluate) is deliberately NOT exposed: it would
either blow the max_oracle_calls budget or turn Run G into unbounded oracle-in-the-loop
search, breaking the apples-to-apples comparison with the A_27B baseline.
"""

import ast
import heapq
import json
import re

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdChemReactions
from rdchiral.initialization import rdchiralReactants

from utils import (
    sanitize_smiles,
    run_retro as _run_retro,
    is_reaction_in_dict,
    check_availability,
    process_reaction_routes,
    smiles_to_reaction,
)
from modification_prompt import get_feedback

# Cap tool-result payloads (what the MODEL sees, fed back as role:"tool" messages) so a
# chatty tool can't blow up the context window across many tool turns. Per-tool because
# validate_route's per-step report is the feedback the model needs to fix the route (give
# it lots of room), while run_retro can return many reactant sets and is the real explosion
# risk (keep it tight). Everything else gets a generous default.
# Kept modest: canary v4 showed large tool results balloon the conversation across many
# tool turns and make each generation pathologically slow (context re-prefill dominates).
_RESULT_CAP_DEFAULT = 4000
_RESULT_CAP_BY_TOOL = {
    "validate_route": 6000,
    "run_retro": 3000,
    "suggest_disconnections": 4000,
    "check_reaction_in_uspto": 6000,   # now also returns nearest-precedent templates
    "search_skills": None,             # uncapped: full strategies returned verbatim (user directive)
}


# --------------------------------------------------------------------------------------
# OpenAI tool schemas (passed as tools=[...] to chat.completions.create)
# --------------------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "validate_smiles",
            "description": (
                "Check whether one or more SMILES strings are valid, parseable molecules. When a "
                "SMILES is invalid the result gives the SPECIFIC reason (e.g. valence error, "
                "kekulization failure, bad ring closure) so you can correct it. Use this to confirm "
                "a product, reactant, or building block is a real molecule before committing it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of SMILES strings to validate.",
                    }
                },
                "required": ["smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_reaction_in_uspto",
            "description": (
                "Check whether a proposed backward reaction has a matching reaction template "
                "in the USPTO dataset. Routes are only accepted if every disconnection is a "
                "known USPTO reaction. If there is NO match, the result explains why (any invalid "
                "SMILES in the reaction) AND returns the nearest real USPTO disconnections to adapt "
                "toward — so use the feedback to fix the step, don't just retry blindly. Reaction "
                "must be in the form 'Product>>Reactant1.Reactant2'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reaction": {
                        "type": "string",
                        "description": "Reaction SMILES, e.g. 'CCOC(=O)c1ccccc1>>CCO.OC(=O)c1ccccc1'.",
                    }
                },
                "required": ["reaction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_retro",
            "description": (
                "Apply a retro reaction template to a product and return the reactant set(s) "
                "it produces. Use this to verify a disconnection actually decomposes the "
                "product into the reactants you expect."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Product SMILES to decompose.",
                    },
                    "template": {
                        "type": "string",
                        "description": "Retro reaction template/SMILES 'Product>>Reactant1.Reactant2'.",
                    },
                },
                "required": ["product", "template"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_purchasable",
            "description": (
                "Check whether molecules are commercially purchasable building blocks. A route "
                "is only complete when every leaf is purchasable, so use this to decide whether "
                "an intermediate still needs to be decomposed further."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of SMILES strings to check for purchasability.",
                    }
                },
                "required": ["smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_complexity",
            "description": (
                "Estimate the synthetic complexity (SC/SA blend) of molecules. Lower is "
                "simpler / closer to a purchasable building block. Use this to choose which "
                "disconnection makes the intermediates simpler."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of SMILES strings to score.",
                    }
                },
                "required": ["smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_route",
            "description": (
                "Validate a complete candidate route end-to-end and get a per-step report of "
                "structural errors (invalid SMILES, reaction not in USPTO, reaction cannot "
                "happen, molecule-set mismatch, non-purchasable leaves). Call this on your full "
                "route before emitting the final <ROUTE> to catch problems while you can still "
                "fix them. Does NOT consume the oracle budget."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "string",
                        "description": (
                            "The route as the same Python-literal list of step dicts you would "
                            "put inside <ROUTE></ROUTE> (e.g. \"[{'Molecule set': \\\"['CC=O']\\\", "
                            "'Product': ..., 'Reaction': ..., 'Reactants': ..., "
                            "'Updated molecule set': ...}]\")."
                        ),
                    }
                },
                "required": ["route"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_disconnections",
            "description": (
                "Given a product SMILES, return real USPTO reactions used to make molecules "
                "SIMILAR to it (retrieved by structural similarity). Use this when you are unsure "
                "how to disconnect a molecule — instead of guessing a reaction and checking, look "
                "up precedent first. Works for ANY intermediate, not just the original target."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "description": "Product SMILES to find disconnection precedents for."},
                    "k": {"type": "integer", "description": "How many similar precedents to return (1-5, default 3)."},
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_skills",
            "description": (
                "Retrieve learned retrosynthesis STRATEGY skills relevant to a molecule you are "
                "trying to disconnect. Each skill is a high-level disconnection strategy distilled "
                "from many real USPTO reactions, with worked SMILES examples. Call this ON DEMAND "
                "when you are unsure how to break down an intermediate and want proven strategies "
                "for its functional groups — the store is searched by the reactive groups your "
                "molecule carries, and only DISTINCT strategies are returned (no duplicates)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string", "description": "SMILES of the molecule/intermediate you want disconnection strategies for."},
                    "requirement": {"type": "string", "description": "Optional: a short phrase describing what you need (e.g. 'form the amide bond', 'cleave the aryl ether') to bias which strategies are returned."},
                    "k": {"type": "integer", "description": "How many strategies to return IN FULL (1-4, default 3). Additional matching strategies are always listed by name under 'more_available'."},
                },
                "required": ["smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_route",
            "description": (
                "Submit your FINAL retrosynthesis route. Call this exactly once, as your last "
                "action, when you have a route you are reasonably confident in (partial routes "
                "are acceptable — they get refined later). This is how you finish: provide the "
                "route as structured steps and stop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "array",
                        "description": "Ordered list of retrosynthesis steps (first step starts from the target).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "molecule_set": {"type": "array", "items": {"type": "string"},
                                                 "description": "SMILES in the working set at this step (first step = [target])."},
                                "product": {"type": "string", "description": "The molecule from molecule_set being disconnected this step."},
                                "reaction": {"type": "string", "description": "Backward reaction SMILES 'Product>>Reactant1.Reactant2'."},
                                "reactants": {"type": "array", "items": {"type": "string"}, "description": "Reactant SMILES produced."},
                                "updated_molecule_set": {"type": "array", "items": {"type": "string"},
                                                         "description": "Working set after removing product and adding reactants."},
                                "rational": {"type": "string", "description": "Brief reasoning for this step (optional)."},
                            },
                            "required": ["molecule_set", "product", "reaction", "reactants", "updated_molecule_set"],
                        },
                    }
                },
                "required": ["route"],
            },
        },
    },
]


# --------------------------------------------------------------------------------------
# Tool implementations. Each takes (arguments: dict, ctx) and returns a JSON string.
# ctx is the optimizer instance (has .inventory, .template_dict, .oracle).
# --------------------------------------------------------------------------------------

def _smiles_detail(smi):
    """Return (valid, canonical, reason). When invalid, `reason` is the SPECIFIC RDKit
    problem (valence error, kekulization failure, bad ring closure, parse error) rather than
    a bare False — so the model can actually fix it."""
    if not smi or not isinstance(smi, str):
        return False, None, "empty or non-string SMILES"
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return False, None, "could not parse SMILES (syntax error — check brackets, ring-closure digits, valences)"
    problems = Chem.DetectChemistryProblems(mol)
    if problems:
        return False, None, "; ".join(p.Message() for p in problems)
    canon = sanitize_smiles(smi)
    if canon is None:
        return False, None, "failed RDKit sanitization (chemistry is inconsistent)"
    return True, canon, None


def _tool_validate_smiles(arguments, ctx):
    smiles = arguments.get("smiles") or []
    if isinstance(smiles, str):
        smiles = [smiles]
    results = []
    for smi in smiles:
        valid, canon, reason = _smiles_detail(smi)
        entry = {"smiles": smi, "valid": valid, "canonical": canon}
        if not valid:
            entry["reason"] = reason
        results.append(entry)
    return {"results": results}


def _nearest_reaction_templates(reaction, ctx, k=3):
    """Top-k USPTO reaction templates most similar to `reaction` by difference-fingerprint
    Tanimoto (same machinery as the optimizer's blurry_search). Gives the model a concrete
    'closest known disconnections' list to steer toward when its proposal isn't a match."""
    all_fps = getattr(ctx, "all_reaction_fps", None)
    rlist = getattr(ctx, "reaction_list", None)
    if not all_fps or rlist is None:
        return []
    try:
        rxn_obj = smiles_to_reaction(reaction)
        if rxn_obj is None:
            return []
        fp = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn_obj)
        sims = DataStructs.BulkTanimotoSimilarity(fp, list(all_fps))
        top = heapq.nlargest(k, range(len(sims)), key=lambda i: sims[i])
        return [{"similarity": round(float(sims[i]), 3), "template": rlist[i]} for i in top]
    except Exception:
        return []


def _tool_check_reaction_in_uspto(arguments, ctx):
    reaction = arguments.get("reaction", "")
    in_uspto, template_key = is_reaction_in_dict(reaction, ctx.template_dict)
    out = {"reaction": reaction, "in_uspto": bool(in_uspto), "template_key": template_key}
    if not in_uspto:
        # Detail WHY it might not match + give the closest real disconnections to adapt toward.
        valid_parts = []
        try:
            prod, reac = reaction.split(">>")
            for part, label in ((prod, "product"), (reac, "reactants")):
                for s in part.split("."):
                    ok, _c, why = _smiles_detail(s)
                    if not ok:
                        valid_parts.append(f"{label} '{s}': {why}")
        except Exception:
            valid_parts.append("reaction is not in 'Product>>Reactant1.Reactant2' form")
        if valid_parts:
            out["smiles_problems"] = valid_parts
        out["nearest_precedents"] = _nearest_reaction_templates(reaction, ctx, k=3)
        out["note"] = ("No exact USPTO precedent for this disconnection. Fix any SMILES problems "
                       "above, or adapt your disconnection toward one of the nearest_precedents.")
    return out


def _tool_run_retro(arguments, ctx):
    product = arguments.get("product", "")
    template = arguments.get("template", "")
    ok, _c, why = _smiles_detail(product)
    if not ok:
        return {"product": product, "template": template, "reactant_sets": [],
                "error": f"invalid product SMILES: {why}"}
    try:
        product_rd = rdchiralReactants(product)
        outputs = _run_retro(product_rd, template)
    except Exception as e:
        return {"product": product, "template": template, "error": f"{type(e).__name__}: {e}",
                "reactant_sets": []}
    out = {"product": product, "template": template, "n_outcomes": len(outputs),
           "reactant_sets": outputs}
    if not outputs:
        out["note"] = ("This template does not apply to this product — the reaction pattern "
                       "doesn't match, so this disconnection is not valid here. Try a different "
                       "disconnection (use suggest_disconnections).")
    return out


def _tool_check_purchasable(arguments, ctx):
    smiles = arguments.get("smiles") or []
    if isinstance(smiles, str):
        smiles = [smiles]
    unpurchasable = check_availability(smiles, ctx.inventory)
    unset = set(unpurchasable)
    results = []
    for smi in smiles:
        purch = smi not in unset
        entry = {"smiles": smi, "purchasable": purch}
        if not purch:
            # Complexity tells the model how far this leaf is from a buyable building block.
            try:
                entry["complexity"] = round(float(ctx.oracle.get_oracle_score(smi)), 2)
            except Exception:
                pass
        results.append(entry)
    out = {"results": results, "non_purchasable": unpurchasable}
    if unpurchasable:
        out["note"] = ("Non-purchasable molecules must be decomposed further. 'complexity' is "
                       "lower for molecules closer to a commercially available building block.")
    return out


def _tool_estimate_complexity(arguments, ctx):
    smiles = arguments.get("smiles") or []
    if isinstance(smiles, str):
        smiles = [smiles]
    results = []
    for smi in smiles:
        try:
            score = float(ctx.oracle.get_oracle_score(smi))
            results.append({"smiles": smi, "complexity": score})
        except Exception as e:
            results.append({"smiles": smi, "error": f"{type(e).__name__}: {e}"})
    return {"results": results, "note": "lower complexity = simpler / closer to purchasable"}


def validate_route_list(route, ctx):
    """Core route validator (shared by the validate_route tool and the submit_route gate).
    Takes an already-parsed list of pipeline step dicts, runs Optimizer.sanitize() (which does
    NOT score, so route_buffer is untouched), and returns a structured per-step report incl.
    `all_steps_valid` — False if any step has a structural problem (invalid SMILES, reaction not
    in USPTO, reaction can't happen, or molecule-set mismatch)."""
    if not isinstance(route, list) or not route:
        return {"parsed": False, "all_steps_valid": False,
                "error": "Route must be a non-empty list of step dicts."}
    try:
        starting_list = ast.literal_eval(route[0]["Molecule set"])
    except Exception as e:
        return {"parsed": False, "all_steps_valid": False,
                "error": f"Could not read step 0 'Molecule set': {e}"}
    # 2026-11-15: this is a SPECULATIVE check -- the model is asking whether a step would be
    # accepted, not proposing it. `sanitize` runs the real checker, and `check_route` /
    # `check_route_extra` both add to `explored_reaction`, so without this a step the model merely
    # ASKED about would be recorded as already explored and could be suppressed when the search
    # later wanted it. The fix-effect counters would likewise count speculative checks as real
    # repairs. Same save/restore the existing dry run uses (advisor_live.dry_accepts).
    _saved = (set(getattr(ctx, "explored_reaction", set()) or set()),
              getattr(ctx, "_switch_pending", None),
              dict(getattr(ctx, "_fx", {}) or {}))
    try:
        _checked, evaluation = ctx.sanitize(starting_list, route, False)
    except Exception as e:
        return {"parsed": True, "all_steps_valid": False,
                "error": f"Validation failed: {type(e).__name__}: {e}"}
    finally:
        ctx.explored_reaction, ctx._switch_pending, ctx._fx = _saved

    steps = []
    all_ok = True
    for tup in evaluation:
        step_idx, ok, info = tup[0], tup[1], tup[2]
        ok = bool(ok)
        all_ok = all_ok and ok
        entry = {"step": int(step_idx), "ok": ok}
        if not ok:
            try:
                _prev, msg = get_feedback(info.get("molecule_set", []), info, ctx.inventory)
                entry["error"] = msg.strip()
            except Exception as e:
                entry["error"] = f"(could not build feedback: {type(e).__name__}: {e})"
        steps.append(entry)

    last_info = evaluation[-1][2] if evaluation else {}
    final_unpurchasable = []
    if last_info.get("check_availability") and last_info.get("unavailable_mol_id"):
        ums = last_info.get("updated_molecule_set", [])
        final_unpurchasable = [ums[i] for i in last_info["unavailable_mol_id"] if i < len(ums)]

    return {
        "parsed": True,
        "all_steps_valid": all_ok,
        "n_steps": len(steps),
        "steps": steps,
        "final_non_purchasable": final_unpurchasable,
        "route_complete": all_ok and not final_unpurchasable,
    }


def _tool_validate_route(arguments, ctx):
    route_str = arguments.get("route", "")
    try:
        route = ast.literal_eval(route_str) if isinstance(route_str, str) else route_str
    except Exception as e:
        return {"parsed": False, "error": f"Could not parse route: {type(e).__name__}: {e}"}
    return validate_route_list(route, ctx)


def _tool_suggest_disconnections(arguments, ctx):
    product = arguments.get("product", "")
    try:
        k = int(arguments.get("k", 3) or 3)
    except Exception:
        k = 3
    k = max(1, min(k, 5))
    all_fps = getattr(ctx, "_rag_all_fps", None)
    route_list = getattr(ctx, "_rag_route_list", None)
    if not all_fps or route_list is None:
        return {"product": product, "error": "precedent index unavailable in this context"}
    mol = Chem.MolFromSmiles(product)
    if mol is None:
        return {"product": product, "error": "invalid product SMILES"}
    # Same similarity machinery the optimizer uses for RAG retrieval (Morgan2 + Tanimoto),
    # but on-demand for any intermediate the model is reasoning about.
    fp = AllChem.GetMorganFingerprint(mol, 2, useFeatures=False)
    sims = DataStructs.BulkTanimotoSimilarity(fp, list(all_fps))
    top = heapq.nlargest(k, range(len(sims)), key=lambda i: sims[i])
    suggestions = []
    for idx in top:
        try:
            steps = process_reaction_routes(route_list[idx])
            rxns = [s.get("Reaction") for s in steps][:4]
            suggestions.append({"similarity": round(float(sims[idx]), 3), "precedent_reactions": rxns})
        except Exception as e:
            suggestions.append({"similarity": round(float(sims[idx]), 3), "error": str(e)})
    return {
        "product": product,
        "note": ("These are real USPTO reactions used to make molecules SIMILAR to your product; "
                 "adapt the disconnection type to your exact structure (then confirm with "
                 "check_reaction_in_uspto)."),
        "suggestions": suggestions,
    }


def _tool_search_skills(arguments, ctx):
    """On-demand skill retrieval. Reuses the SAME chemistry-semantic machinery the
    injection path uses (FG-applicability matcher + reaction-signature diversity
    gate), so a model-initiated pull returns the same quality of skills as a
    stall-triggered push -- just when the model decides it needs them."""
    smiles = (arguments.get("smiles") or "").strip()
    requirement = (arguments.get("requirement") or "").strip()
    try:
        k = int(arguments.get("k", 3) or 3)
    except Exception:
        k = 3
    k = max(1, min(k, 4))
    if not getattr(ctx, "_skill_memory_enabled", False) or not getattr(ctx, "_skill_memory", None):
        return {"error": "skill memory is not enabled in this run"}
    if Chem.MolFromSmiles(smiles) is None:
        return {"smiles": smiles, "error": "invalid SMILES"}
    sm = ctx._sm
    sig = ctx._sig
    skills = [s for s in ctx._skill_memory.get("skills", [])
              if s.get("state") != sm.STATE_ELIMINATED]
    if not skills:
        return {"smiles": smiles, "skills": [], "note": "skill store is empty"}
    # 1. FG-applicability prefilter (chemistry-semantic, no LLM) over the whole store.
    pool_n = int(ctx._skill_cfg.get("prefilter_pool", 400))
    try:
        pool = ctx._tanimoto_prefilter(smiles, skills, pool_n)
    except Exception:
        pool = list(skills[:pool_n])
    # 2. Optional requirement keyword bias: float skills whose name/trigger mentions it.
    if requirement:
        rw = [w for w in re.split(r"\W+", requirement.lower()) if len(w) > 2]
        if rw:
            def hit(s):
                text = (s.get("name", "") + " " + s.get("trigger", "") + " "
                        + s.get("instructions", "")).lower()
                return sum(w in text for w in rw)
            pool = sorted(pool, key=hit, reverse=True)
    # 3. Diversity gate: one skill per reaction-centre signature, so results are
    #    DISTINCT strategies (not paraphrases of one). Does not pad with dupes.
    depth = 12  # keep enough distinct strategies to also list the extras as pointers
    if bool(ctx._skill_cfg.get("diversify_injection", True)):
        try:
            pool = sig.diversify(pool, depth, per_sig=int(ctx._skill_cfg.get("diversity_per_sig", 1)))
        except Exception:
            pool = pool[:depth]
    else:
        pool = pool[:depth]

    # Return the top-k IN FULL (name + when + strategy + detailed description +
    # worked examples), then the NEXT distinct matches as NAME-ONLY pointers.
    # Deliberately NOT the full store: the model gets a few complete strategies to
    # act on plus visibility that more exist, and can re-query (narrow via
    # `requirement`) to pull a different few -- rather than being handed everything.
    full = []
    for s in pool[:k]:
        exs = []
        for e in sorted(s.get("examples", []) or [],
                        key=lambda e: e.get("gt_similarity", 0.0), reverse=True)[:2]:
            steps = list(e.get("steps") or [])
            if steps:
                exs.append(steps[0])
        full.append({
            "name": s.get("name", ""),
            "when": s.get("trigger", ""),
            "strategy": s.get("instructions", ""),
            "details": s.get("description", ""),
            "examples": exs,
        })
    more = [{"name": s.get("name", ""), "when": s.get("trigger", "")}
            for s in pool[k:k+8]]
    return {
        "smiles": smiles,
        "note": ("Top strategies for this molecule's functional groups, in full. Adapt one to your "
                 "exact structure, then confirm the disconnection with check_reaction_in_uspto. "
                 "More matching strategies are listed by name under 'more_available' — call "
                 "search_skills again with a `requirement` phrase to pull any of them in full."),
        "skills": full,
        "more_available": more,
    }


def build_submit_route(arguments):
    """Convert a submit_route tool call into the pipeline's <ROUTE> text. Returns
    (route_text_or_None, result_json_str, pipeline_steps_or_None). The route comes from
    STRUCTURED tool arguments, so there is no free-text <ROUTE> regex/ast fragility. On
    malformed input returns (None, error-json, None) so the model can see the problem and retry."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except Exception as e:
            return None, json.dumps({"submitted": False, "error": f"could not parse arguments: {e}"}), None
    steps = (arguments or {}).get("route") or []
    if not isinstance(steps, list) or not steps:
        return None, json.dumps({"submitted": False, "error": "route must be a non-empty list of steps"}), None

    pipeline_steps = []
    for n, s in enumerate(steps):
        if not isinstance(s, dict):
            return None, json.dumps({"submitted": False, "error": f"step {n} is not an object"}), None
        try:
            mol_set = list(s["molecule_set"])
            product = s["product"]
            if isinstance(product, list):
                product = product[0] if product else ""
            reaction = s["reaction"]
            if isinstance(reaction, list):
                reaction = reaction[0] if reaction else ""
            reactants = list(s["reactants"])
            updated = list(s["updated_molecule_set"])
        except (KeyError, TypeError) as e:
            return None, json.dumps({"submitted": False, "error": f"step {n} missing/!invalid field: {e}"}), None
        # Pipeline expects each field as a string that ast.literal_eval can parse back.
        pipeline_steps.append({
            "Molecule set": str(mol_set),
            "Rational": str(s.get("rational", "")),
            "Product": str([product]),
            "Reaction": str([reaction]),
            "Reactants": str(reactants),
            "Updated molecule set": str(updated),
        })
    route_text = "<ROUTE>\n" + repr(pipeline_steps) + "\n</ROUTE>"
    return route_text, json.dumps({"submitted": True, "n_steps": len(pipeline_steps)}), pipeline_steps


_DISPATCH = {
    "validate_smiles": _tool_validate_smiles,
    "check_reaction_in_uspto": _tool_check_reaction_in_uspto,
    "run_retro": _tool_run_retro,
    "check_purchasable": _tool_check_purchasable,
    "estimate_complexity": _tool_estimate_complexity,
    "validate_route": _tool_validate_route,
    "suggest_disconnections": _tool_suggest_disconnections,
    "search_skills": _tool_search_skills,
}


def execute_tool(name, arguments, ctx):
    """Dispatch a tool call and return a JSON string (always — never raises).

    name:       tool function name from the model's tool_call.
    arguments:  dict (already JSON-parsed) of the call arguments. A raw JSON string is
                also accepted for convenience.
    ctx:        the optimizer instance (provides .inventory, .template_dict, .oracle, .sanitize).
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except Exception as e:
            return json.dumps({"error": f"Could not parse tool arguments: {e}"})
    if not isinstance(arguments, dict):
        arguments = {}
    if name == "get_known_splits":       # arm cuttool (aot_search, aot.cut_tool, 2026-11-01): the optimizer answers
        fn = getattr(ctx, "_cut_tool_result", None)
        if fn is None:
            return json.dumps({"error": "get_known_splits is not available in this run"})
        return fn(str(arguments.get("smiles", "")))
    if name == "get_known_splits_of_this_molecule":   # arm ctoolself (aot.cut_tool.scope: self, 2026-11-13): no
        fn = getattr(ctx, "_cut_tool_result_self", None)   # argument -- the optimizer answers for the bound molecule
        if fn is None:
            return json.dumps({"error": "get_known_splits_of_this_molecule is not available in this run"})
        return fn()

    if name == "check_my_route":           # arm valtool (aot.validate_tool) tool 1: FORMAT report only
        fn = getattr(ctx, "_validate_tool_result", None)   # leakage-free: no checker, no disconnection, no buyability
        if fn is None:
            return json.dumps({"error": "check_my_route is not available in this run"})
        return fn(arguments.get("route"))

    if name == "my_rejected_steps":        # arm valtool tool 2: this molecule's rejected steps + reasons
        fn = getattr(ctx, "_rejected_tool_result", None)
        if fn is None:
            return json.dumps({"error": "my_rejected_steps is not available in this run"})
        return fn()

    if name == "how_is_this_made":         # arm mine (aot_search, aot.mine_tool, 2026-11-22): how this
        fn = getattr(ctx, "_mine_tool_result", None)   # molecule's motifs are MADE, from the recorded corpus.
        if fn is None:                                 # No checker, no disconnection, no purchasability.
            return json.dumps({"error": "how_is_this_made is not available in this run"})
        return fn()

    if name == "describe_this_molecule":   # arm moltool (aot_search, aot.mol_tool, 2026-11-19): deterministic
        fn = getattr(ctx, "_mol_tool_result", None)   # RDKit facts about the molecule under expansion; no
        if fn is None:                                # argument, and no disconnection in the answer
            return json.dumps({"error": "describe_this_molecule is not available in this run"})
        return fn()

    if name == "see_this_molecule":       # arm moltool image half (aot.mol_tool.image, 2026-11-19): records
        fn = getattr(ctx, "_image_tool_result", None)   # the molecule to draw; query_LLM attaches the picture
        if fn is None:
            return json.dumps({"error": "see_this_molecule is not available in this run"})
        return fn()

    if name == "ask_advisor":            # arm ask (aot_search, aot.advisor_ask, 2026-11-15): the pull advisor
        fn = getattr(ctx, "_ask_advisor_result", None)
        if fn is None:
            return json.dumps({"error": "ask_advisor is not available in this run"})
        return fn(str(arguments.get("finished", "")), str(arguments.get("question", "")))

    fn = _DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool '{name}'. Available: {list(_DISPATCH)}"})

    try:
        result = fn(arguments, ctx)
    except Exception as e:
        result = {"error": f"Tool '{name}' raised {type(e).__name__}: {e}"}

    out = json.dumps(result, default=str)
    cap = _RESULT_CAP_BY_TOOL.get(name, _RESULT_CAP_DEFAULT)
    if cap and len(out) > cap:
        out = json.dumps({
            "truncated": True,
            "note": f"result exceeded {cap} chars and was truncated",
            "partial": out[:cap],
        })
    return out
