"""Known one-step splits ("disconnections") of a molecule, from both reaction sources (2026-10-18).

Why: try 8 showed Intern a fixed list of facts that was 77% near-copies (steps whose largest piece keeps >= 90%
of the molecule), and Qwen's next step succeeded 72% of the time when the list held a real cut vs 44% when it held
only near-copies (docs/experiment_history.md, "Advisor problems re-validated"). A timing job on 9 real molecules
(scratchpad time_disconnections.py, job 9548232) found both sources quick enough to search live: the 40k set in
0.6-2.3 s, the 270k checker templates in 3-25 s; the 270k templates gave 76-81 real cuts on 28-43-atom molecules
against 4-6 from the 40k set, and both sources are 87-93% near-copies, so the steps are sorted biggest cut first.

Two users (both config-gated, off by default):
  - arm "rc" (aot.real_cuts): the best checker-accepted steps that are not near-copies go into Qwen's own prompt;
  - try 10 (aot.advisor.tools): Intern looks them up itself (advisor_tools.py).

Sources (both read-only shared data of the optimizer):
  - "40k": RetroSim over the USPTO-50k train set (advisor_plan.retrosim_outcomes, as try 8's facts): reactions on the
    100 most similar recorded products, their templates applied to the molecule; score = precedent strength;
  - "270k": the route checker's templates (optimizer.original_template_dict, retro SMARTS product>>reactants): every
    template whose product side matches the molecule (optimizer.template_dict's compiled patterns) is applied with
    rdchiral; support = how many templates give the same pieces.
Everything is computed by code; nothing comes from a model. "Accepted" = the search's own validator accepts the
one-step route as written (advisor_live.dry_accepts, which leaves no state behind)."""
from rdkit import Chem
from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReactants, rdchiralReaction

import re

# Advisor arms are not part of this distribution; these imports are allowed to fail.
try:
    import advisor_live as al
    import advisor_plan as ap
    ADVISORS_AVAILABLE = True
except ImportError:
    al = ap = None
    ADVISORS_AVAILABLE = False

from utils import run_retro

NEAR = 0.9          # largest piece >= 90% of the molecule = a near-copy (as try 8's facts)
REAL = 0.7          # largest piece < 70% = a real cut (as the analyses)


def template_split_exists(opt, smiles, canon, state):
    """Termination fix (2026-11-08, config aot.termination.no_split_dead): can ANY template the route checker
    could use decompose `smiles` (canonical)? The checker accepts a step only when run_retro(product, template)
    gives an output (utils.verify_reaction_step), and the template comes from one of TWO sources:
      - the 270k dictionary (optimizer._lookup_template, and blurry_search in _step1_substitute);
      - RetroSim: templates extracted from the 100 most similar USPTO-50k train reactions (rule_based_search via
        sim_based_rag.do_one, the step-1 fallback when blurry_search finds nothing; key '99999999999').
    The census (analysis/no_split_census.py, job 9714318) caught the second source: O=CCl sits on a found ctrl
    route through a RetroSim template ([CH;+0:1]>>Cl-[C;H0;+0:1]) that no dictionary template matches. So both
    are tested, the 270k first (most molecules stop there), RetroSim only when the 270k gives nothing. False
    means no answer for this molecule can ever pass the checker.

    Uses the checker's own call (utils.run_retro, which groups a multi-fragment product side) on every template.
    The compiled product-side patterns are only a necessary-condition pre-filter, and a template whose pattern
    failed to compile is run, never skipped. A template that raises is skipped, as the checker would reject it
    too. An output only counts if every piece is a valid molecule and the molecule is not among its own pieces
    (the tree refuses that step). Stops at the first usable output, so only dead molecules pay for a full
    scan (3-25 s). -> True / False / None (could not decide; callers treat None as alive).
    Cached per molecule in state["tsplit"]."""
    cache = state.setdefault("tsplit", {})
    if smiles in cache:
        return cache[smiles]
    res = None
    try:
        m = Chem.MolFromSmiles(smiles) if smiles else None
        if m is not None:
            rct = rdchiralReactants(smiles)
            res = False
            for k, (_reac, prods) in opt.template_dict.items():
                if prods and all(p is not None for p in prods) and not all(m.HasSubstructMatch(p) for p in prods):
                    continue
                tpl = opt.original_template_dict.get(k)
                if not tpl:
                    continue
                try:
                    outs = run_retro(rct, tpl)
                except Exception:
                    continue
                for out in outs:
                    pieces = [canon(x) for x in out]
                    if pieces and all(p is not None for p in pieces) and smiles not in pieces:
                        res = True
                        break
                if res:
                    break
            if res is False:
                # the RetroSim source (as ap.retrosim_outcomes = sim_based_rag.do_one, keeping the precursors).
                # Any failure here leaves the molecule undecided (alive), never dead.
                res = None
                found = False
                for prec, _score, _tpl in ap.retrosim_outcomes(smiles, opt.datasub, state.setdefault("jx", {})):
                    pieces = [canon(x) for x in prec.split(".")]
                    if pieces and all(p is not None for p in pieces) and smiles not in pieces:
                        found = True
                        break
                res = found
    except Exception as e:
        print(f"[disconnections] template_split_exists failed on {smiles}: {e}")
        res = None
    cache[smiles] = res
    return res


def _heavy(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return m.GetNumHeavyAtoms() if m is not None else 0


def search(opt, smiles, canon, state):
    """All known one-step splits of `smiles` (canonical), sorted biggest cut first (largest piece ascending, then
    template support, then precedent). Each: {pieces (canonical, sorted), largest (fraction of the molecule's heavy
    atoms), support (270k templates giving it), precedent (best 40k RetroSim score or None), sources}. Cached per
    molecule in state["cands"]; never raises (a failing source is skipped and counted in state["errors"])."""
    cache = state.setdefault("cands", {})
    if smiles in cache:
        return cache[smiles]
    errors = state.setdefault("errors", {})
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        cache[smiles] = []
        return []
    ha, found = max(m.GetNumHeavyAtoms(), 1), {}

    def add(prec, src, score=None):
        pieces = [canon(x) for x in prec.split(".")]
        if not pieces or any(p is None for p in pieces):
            return
        key = tuple(sorted(set(pieces)))
        if smiles in key:
            return                                   # the molecule among its own pieces: no step
        c = found.get(key)
        if c is None:
            c = found[key] = {"pieces": list(key), "largest": max(_heavy(p) for p in key) / ha, "support": 0,
                              "precedent": None, "sources": []}
        if src not in c["sources"]:
            c["sources"].append(src)
        if src == "270k":
            c["support"] += 1
        if score is not None and (c["precedent"] is None or score > c["precedent"]):
            c["precedent"] = round(float(score), 3)

    try:
        for prec, score, _tpl in ap.retrosim_outcomes(smiles, opt.datasub, state.setdefault("jx", {})):
            add(prec, "40k", score)
    except Exception as e:
        errors["40k"] = errors.get("40k", 0) + 1
        print(f"[disconnections] 40k search failed on {smiles}: {e}")
    try:
        rct = rdchiralReactants(smiles)
        for k, (_reac, prods) in opt.template_dict.items():
            if not prods or any(p is None for p in prods) or not all(m.HasSubstructMatch(p) for p in prods):
                continue
            try:
                outs = rdchiralRun(rdchiralReaction(opt.original_template_dict[k]), rct, combine_enantiomers=False)
            except Exception:
                continue
            for prec in outs:
                add(prec, "270k")
    except Exception as e:
        errors["270k"] = errors.get("270k", 0) + 1
        print(f"[disconnections] 270k search failed on {smiles}: {e}")
    out = sorted(found.values(), key=lambda c: (c["largest"], -c["support"], -(c["precedent"] or 0), c["pieces"]))
    cache[smiles] = out
    return out


def accepted(opt, smiles, pieces, canon, state):
    """The route checker's verdict on smiles >> pieces as written (cached per step)."""
    v = state.setdefault("verdicts", {})
    key = (smiles, tuple(pieces))
    if key not in v:
        v[key] = bool(al.dry_accepts(opt, smiles, list(pieces), canon))
    return v[key]


def describe(opt, smiles, c, canon, state):
    """One step as the tools / the prompt block report it."""
    return {"pieces": list(c["pieces"]), "largest_piece_percent": round(100 * c["largest"]),
            "buyable": [p for p in c["pieces"] if opt._purchasable(p)],
            "checker_accepts": accepted(opt, smiles, c["pieces"], canon, state),
            "templates": c["support"], "precedent": c["precedent"], "sources": list(c["sources"])}


def best_steps(opt, smiles, canon, state, max_piece=NEAR, show=5, check_max=60):
    """Arm rc: up to `show` checker-accepted steps whose largest piece is < max_piece, biggest cut first; the
    checker runs on at most `check_max` candidates. -> (steps described, candidates checked, total < max_piece)."""
    cands = [c for c in search(opt, smiles, canon, state) if c["largest"] < max_piece]
    shown, checked = [], 0
    for c in cands:
        if len(shown) >= show or checked >= check_max:
            break
        checked += 1
        if accepted(opt, smiles, c["pieces"], canon, state):
            shown.append(describe(opt, smiles, c, canon, state))
    return shown, checked, len(cands)


def prefetch(opt, smiles, canon, max_piece=NEAR, show=5, check_max=60, buyable_max=0):
    """`best_steps` plus the two caches it filled, so a caller in ANOTHER PROCESS can install them and get
    exactly the result an in-process call would have given (split_pool.py, config `aot.split_pool`).

    Runs on a fresh state, so nothing here depends on what the caller has cached. Only the two picklable caches
    come back: `cands` (the candidate scan, `search`) and `verdicts` (the checker, `accepted`). `jx` is left
    behind on purpose -- it holds RetroSim working objects, which are the caller's business and need not pickle.

    `buyable_max` (> 0 for arm steer): after `best_steps`, run the checker on candidates whose every piece is
    purchasable, near-copies included, in `search`'s own order and at most `buyable_max` of them -- the same
    loop `aot_search._steer_cuts` runs, so that loop finds every verdict already cached and spends no checker
    time in the caller's process. -> {"steps", "checked", "n", "cands", "verdicts"}."""
    st = {}
    steps, checked, n = best_steps(opt, smiles, canon, st, max_piece, show, check_max)
    tried = 0
    for c in search(opt, smiles, canon, st):
        if tried >= int(buyable_max):
            break
        if c["pieces"] and all(opt._purchasable(p) for p in c["pieces"]):
            tried += 1
            if accepted(opt, smiles, c["pieces"], canon, st):
                break
    return {"steps": steps, "checked": checked, "n": n,
            "cands": st.get("cands", {}), "verdicts": st.get("verdicts", {})}


def real_cuts_block(smiles, steps):
    """What Qwen's prompt gets in arm rc ("" when there is nothing to show, so the prompt equals the baseline's)."""
    if not steps:
        return ""
    lines = ["", f"KNOWN ONE-STEP SPLITS of the molecule you are decomposing ({smiles}), found by applying recorded "
                 "USPTO reactions and reaction templates to it; the route checker accepts each as written. The "
                 "largest piece is given as a share of the molecule; smaller means a bigger simplification:"]
    for n, s in enumerate(steps, 1):
        buy = f"; buyable: {', '.join(s['buyable'])}" if s["buyable"] else ""
        lines.append(f"  {n}. {smiles}>>{'.'.join(s['pieces'])}   (largest piece {s['largest_piece_percent']}%{buy})")
    lines.append("These are options, not instructions: use one if it fits your plan.")
    return "\n".join(lines) + "\n"


def rag_cuts_block(smiles, items):
    """Arm ragcut (2026-11-13): how molecules SIMILAR to this one were disconnected -- the first step of each of the
    most similar RAG routes, shown UNAPPLIED and unchecked. Unlike real_cuts_block these are not steps of `smiles`
    and the route checker has accepted none of them for it; the model must transfer the idea. "" when there is
    nothing to show, so the prompt then equals the baseline's. `items` = [(similarity, product, reactants)]."""
    if not items:
        return ""
    lines = ["", "HOW SIMILAR MOLECULES WERE DISCONNECTED. Each line below is the FIRST STEP of a recorded USPTO "
                 f"route for a molecule that resembles the one you are decomposing ({smiles}), most similar first. "
                 "Each is a DIFFERENT molecule, shown exactly as recorded: the route checker has NOT accepted any "
                 "of these for your molecule, and none of them is a split of it. They are analogies to adapt:"]
    for n, (sim, prod, reac) in enumerate(items, 1):
        lines.append(f"  {n}. {prod}>>{reac}   (similarity to your molecule {sim:.2f})")
    lines.append("These are options, not instructions: adapt one if the same disconnection suits your molecule.")
    return "\n".join(lines) + "\n"


def select_prompt(smiles, steps, pick, scope=None):
    """Arm cutsel: the separate Qwen call that sees EVERY checker-accepted split and picks `pick` for the main model.
    `scope` (8 Nov, arm cutselR): replaces "the FULL list of its known one-step splits" when the list is not full
    (cutselR shows the 25 biggest real cuts); None keeps the 1 Nov prompt word for word."""
    what = scope or "the FULL list of its known one-step splits"
    lines = [f"You are helping plan a retrosynthesis of this molecule: {smiles}",
             "",
             f"Below is {what}, found by applying recorded USPTO reactions and "
             "reaction templates; the route checker accepts each as written. The largest piece is given as a share of "
             "the molecule (smaller means a bigger simplification); 'buyable' marks pieces that can be bought.", ""]
    for n, s in enumerate(steps, 1):
        buy = f"; buyable: {', '.join(s['buyable'])}" if s["buyable"] else ""
        lines.append(f"  {n}. {smiles}>>{'.'.join(s['pieces'])}   (largest piece {s['largest_piece_percent']}%{buy})")
    lines += ["",
              f"Choose the {pick} splits that are the BEST and BIGGEST disconnections for making this molecule: prefer "
              "splits that simplify it the most and that a chemist would actually choose, and avoid near-copies that "
              "only change a small group. A planning model will be shown only the splits you choose.",
              f"Answer with one line: SELECTED: <{pick} numbers from the list, comma-separated>"]
    return "\n".join(lines)


def parse_selection(answer, n, pick):
    """-> up to `pick` distinct 0-based indices from the last 'SELECTED:' line; [] when there is none."""
    if not answer:
        return []
    m = re.findall(r"SELECTED:\s*([0-9,\s]+)", answer)
    if not m:
        return []
    out = []
    for tok in re.findall(r"\d+", m[-1]):
        i = int(tok) - 1
        if 0 <= i < n and i not in out:
            out.append(i)
        if len(out) >= pick:
            break
    return out
