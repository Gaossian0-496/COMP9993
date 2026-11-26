"""Evolving skill-structured strategy memory for Run S (v2).

Port of the mechanism from arXiv 2605.29237 ("Evolving Skill-Structured Attack
Memory Enhances LLM Jailbreaking", "MemoAttack") onto the retrosynthesis planner
— minus the jailbreaking application. Replaces the single flat markdown strategy
guide (Run A 27B v3/v4) with a categorized store of strategy "skills".

v2 upgrades (supervisor review):
  * HIERARCHICAL, EXAMPLE-RICH skill unit — each skill pairs a generalisable
    instruction with a template and CONCRETE worked examples (real target +
    ordered steps + reasoning), faithful to the paper's skill+template+evidence
    unit. (v1 forbade concrete examples; v2 requires them.)
  * Contextual THOMPSON SAMPLING selection (paper-faithful) — the only runtime
    selection policy. The Beta posterior draw is the "randomly give skills"
    mechanism; no epsilon-random / softmax add-ons.
  * GROUND-TRUTH WEIGHTING — skills born from / reinforced by routes close to the
    dataset's ground-truth precursor accumulate `gt_weight`, which raises their
    selection score and protects them from cleanup.
  * Paper's FOUR lifecycle states — candidate / active / retired / ELIMINATED
    (a real stored tombstone state, not physical deletion). Physical removal is a
    SEPARATE "storage cleanup" step, off by default (unbounded store; the 40-cap
    is gone).

This module is pure logic with stdlib-only deps (json/os/math/random/fcntl/
difflib) so it is testable standalone with no LLM/network/GPU. The LLM extraction
and all wiring live in optimizer.py / run.py.

Memory store JSON schema:
    {
      "version": 2,
      "total_molecules": <int>,
      "skills": [
        {"id": "s0001", "category": <one of FIXED_CATEGORIES>,
         "name": "<short label>",
         "trigger": "<when this lesson applies; used for matching>",
         "instructions": "<IF ... THEN ... ELSE ... + reasoning (conditional)>",
         "template": "<reusable disconnection pattern, e.g. SMARTS-ish>",
         "examples": [                       # EVIDENCE: concrete worked examples
            {"target_smiles": "...", "steps": ["step1: ...", ...],
             "reasoning": "...", "from_idx": <int>,
             "gt_similarity": <float 0-1>, "outcome": "PASS"|"FAIL"}
         ],
         "reasoning": "<general why-this-works>",
         "state": "candidate"|"active"|"retired"|"eliminated",
         "trials": <int>, "wins": <float>,   # wins = accumulated reward
         "gt_weight": <float>,               # accumulated ground-truth closeness
         "created_idx": <int>, "updated_idx": <int>}
      ]
    }

Lifecycle (faithful to MemoAttack §Lifecycle-Driven Memory Evolution):
  - candidate:  newly invented/mutated, on probation (unproven).
  - active:     promoted after enough positive evidence; the working pool.
  - retired:    demoted active whose evidence went weak; kept as a low-priority
                probe (retired-penalty in selection), can reactivate.
  - eliminated: retired skill that exhausted recovery (repeated underperformance);
                NO LONGER PARTICIPATES IN SELECTION but stays in the store as a
                tombstone. This is a logical state, distinct from physical removal.
  Storage cleanup (separate from the lifecycle): physically removes eliminated /
  aged-retired units only when over a hard `max_total` — default 0 = no-op, so the
  store grows unbounded ("context is not an issue"). Never removes a skill that
  carries ground-truth evidence.
"""

import os
import json
import math
import random
import fcntl
import difflib
from datetime import datetime as _datetime

# Fixed taxonomy — stable bandit "structure". `general` is the catch-all.
FIXED_CATEGORIES = [
    "leaf_purchasability",
    "disconnection_strategy",
    "uspto_template_adherence",
    "functional_group_interconversion",
    "route_depth",
    "stereochemistry",
    "ring_coupling_chemistry",
    "general",
]

_DEDUP_RATIO = 0.80   # difflib ratio (on instructions) at/above which two skills are "the same"
_NAME_DEDUP_RATIO = 0.85   # OR: same category + name this similar => same skill (catches
                           # same-reaction skills whose instruction wording differs)
_TEMPLATE_DEDUP_RATIO = 0.90  # OR: same category + near-identical template => same skill
_MAX_EXAMPLES = 0     # cap on concrete examples kept per skill (0 = uncapped; keep ALL)
_PROVENANCE_CAP = 20  # cap on provenance/lineage trail entries (oldest dropped)

# Lifecycle states (paper: candidate -> active -> retired -> eliminated).
STATE_CANDIDATE = "candidate"
STATE_ACTIVE = "active"
STATE_RETIRED = "retired"
STATE_ELIMINATED = "eliminated"
LIFECYCLE_STATES = (STATE_CANDIDATE, STATE_ACTIVE, STATE_RETIRED, STATE_ELIMINATED)


# ---------------------------------------------------------------------------
# Persistence (mirrors optimizer.py:_update_strategy_guide lock/atomic/history)
# ---------------------------------------------------------------------------

def _empty_memory():
    return {"version": 2, "total_molecules": 0, "skills": []}


def load_memory(path):
    """Read the memory JSON. Returns a fresh empty store if absent/unreadable.

    Backfills v2 fields so a v1 store migrates transparently on first load."""
    if not path or not os.path.exists(path):
        return _empty_memory()
    try:
        with open(path) as f:
            mem = json.load(f)
    except Exception:
        return _empty_memory()
    if not isinstance(mem, dict):
        return _empty_memory()
    mem.setdefault("version", 2)
    mem.setdefault("total_molecules", 0)
    mem.setdefault("skills", [])
    for s in mem["skills"]:
        s.setdefault("trials", 0)
        s.setdefault("wins", 0.0)
        s.setdefault("category", "general")
        s.setdefault("trigger", "")
        s.setdefault("state", STATE_CANDIDATE)
        s.setdefault("created_idx", -1)
        s.setdefault("updated_idx", s.get("created_idx", -1))
        # v2 fields
        s.setdefault("name", "")
        s.setdefault("instructions", s.get("lesson", ""))  # migrate old lesson
        s.setdefault("template", "")
        s.setdefault("examples", [])
        s.setdefault("reasoning", "")
        s.setdefault("gt_weight", 0.0)
        s.setdefault("origin_gt", 0.0)
        s.setdefault("provenance", [])
    return mem


def atomic_write_memory(path, mem, history_dir=None, tag=""):
    """Atomically write the memory JSON and optionally snapshot to history_dir.

    Caller is responsible for holding the fcntl lock (see lock_path())."""
    tmp = path + f".tmp.{os.getpid()}"
    payload = json.dumps(mem, indent=2)
    with open(tmp, "w") as f:
        f.write(payload)
    os.replace(tmp, path)
    if history_dir:
        try:
            os.makedirs(history_dir, exist_ok=True)
            ts_safe = _datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
            snap = os.path.join(history_dir, f"{ts_safe}_pid{os.getpid()}_{tag}.json")
            with open(snap, "w") as f:
                f.write(payload)
        except Exception:
            pass


def lock_path(path):
    """Sidecar lock file path (never lock the store itself — atomic rename
    re-creates its inode each write)."""
    return path + ".lock"


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _winrate(skill):
    t = skill.get("trials", 0)
    return (skill.get("wins", 0.0) / t) if t > 0 else 0.0


def _gt_norm(skill):
    """Average ground-truth closeness per trial, in [0,1] — kept comparable to a
    Beta draw so it never swamps the stochastic term (see thompson_select)."""
    t = skill.get("trials", 0)
    if t <= 0:
        return 0.0
    return max(0.0, min(1.0, skill.get("gt_weight", 0.0) / t))


def _selectable(mem):
    """Skills eligible for injection: everything except eliminated tombstones."""
    return [s for s in mem.get("skills", []) if s.get("state") != STATE_ELIMINATED]


# ---------------------------------------------------------------------------
# UCB (retained only as the no-LLM pool-prefilter fallback + for unit tests)
# ---------------------------------------------------------------------------

def ucb_score(skill, total_trials, c):
    """UCB1 score. Unseen skills (trials==0) return +inf to force exploration."""
    t = skill.get("trials", 0)
    if t <= 0:
        return float("inf")
    total = max(total_trials, 1)
    return _winrate(skill) + c * math.sqrt(math.log(total) / t)


def category_share(mem, skill):
    """Fraction of total trials that landed in this skill's category — usage-share
    control so one dominant category can't crowd out the injection slate."""
    skills = mem.get("skills", [])
    total = sum(s.get("trials", 0) for s in skills)
    if total <= 0:
        return 0.0
    cat = skill.get("category", "general")
    cat_trials = sum(s.get("trials", 0) for s in skills if s.get("category") == cat)
    return cat_trials / total


def ranking_score(skill, mem, total_trials, c=1.4, lambda_ret=0.5, usage_w=0.3):
    """Lifecycle-aware UCB selection score (deterministic). Unseen skills keep
    UCB=+inf so candidates are probed; retired skills take a penalty; the
    usage-share term down-weights an over-represented category."""
    base = ucb_score(skill, total_trials, c)
    if base == float("inf"):
        return base
    if skill.get("state") == STATE_RETIRED:
        base -= lambda_ret
    base -= usage_w * category_share(mem, skill)
    return base


def lifecycle_select(mem, k, c=1.4, lambda_ret=0.5, usage_w=0.3):
    """Deterministic UCB-based selection (excludes eliminated). Used as the no-LLM
    fallback to pre-filter the contextual candidate pool and by unit tests — NOT a
    runtime policy (runtime uses thompson_select)."""
    skills = _selectable(mem)
    if not skills or k <= 0:
        return []
    total_trials = sum(s.get("trials", 0) for s in skills)
    scored = sorted(
        skills,
        key=lambda s: ranking_score(s, mem, total_trials, c, lambda_ret, usage_w),
        reverse=True,
    )
    return scored[:k]


# ---------------------------------------------------------------------------
# Contextual Thompson sampling (paper-faithful runtime selection policy)
# ---------------------------------------------------------------------------

def thompson_select(mem, k, relevance=None, relevance_weight=1.0, gt_pref=0.5,
                    retired_penalty=0.5, rng=None):
    """Select up to k skills by contextual Thompson sampling (the ONLY runtime
    policy; MemoAttack uses contextual Thompson sampling and nothing extra).

    For each selectable skill draw theta ~ Beta(wins+1, (trials-wins)+1) and score

        score = relevance_weight * relevance  +  theta
                + gt_pref * gt_norm  -  retired_penalty*[retired]

    The Beta draw IS the "randomly give skills" mechanism: an unseen skill is
    Beta(1,1) = Uniform(0,1) with a wide posterior, so cold skills surface often;
    the posterior narrows as evidence accrues. `relevance` is an optional
    {skill_id: float in [0,1]} map (contextual term B(x_u,m)); missing -> 0.5.

    Term-scaling note: theta, relevance and gt_norm are each ~[0,1], so the
    stochastic term is never swamped by the deterministic terms. Returns skill
    dicts (references into mem['skills'])."""
    skills = _selectable(mem)
    if not skills or k <= 0:
        return []
    rnd = rng or random
    relevance = relevance or {}

    def score(s):
        wins = s.get("wins", 0.0)
        trials = s.get("trials", 0)
        a = max(wins, 0.0) + 1.0
        b = max(trials - wins, 0.0) + 1.0
        theta = rnd.betavariate(a, b)
        rel = relevance.get(s.get("id"), 0.5)
        # Ground-truth term: the better of (a) downstream reinforcement on
        # gt-close molecules and (b) the skill's own birth-route closeness, so a
        # skill distilled from a near-ground-truth route ranks up even before it
        # has been tried. Both terms are in [0,1] -> never swamps theta.
        gt_term = max(_gt_norm(s), s.get("origin_gt", 0.0))
        val = relevance_weight * rel + theta + gt_pref * gt_term
        if s.get("state") == STATE_RETIRED:
            val -= retired_penalty
        return val

    return sorted(skills, key=score, reverse=True)[:k]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_one(skill, max_examples):
    lines = []
    name = (skill.get("name") or "").strip()
    instr = (skill.get("instructions") or skill.get("lesson") or "").strip()
    trig = (skill.get("trigger") or "").strip()
    tmpl = (skill.get("template") or "").strip()
    why = (skill.get("reasoning") or "").strip()
    header = f"- [{name}]" if name else "-"
    lines.append(header)
    if trig:
        lines.append(f"  WHEN: {trig}")
    if instr:
        lines.append(f"  DO: {instr}")
    desc = (skill.get("description") or "").strip()
    if desc:
        lines.append(f"  DETAIL: {desc}")
    if tmpl:
        lines.append(f"  PATTERN: {tmpl}")
    if why:
        lines.append(f"  WHY: {why}")
    # Best concrete examples first (highest ground-truth similarity).
    exs = sorted(skill.get("examples", []) or [],
                 key=lambda e: e.get("gt_similarity", 0.0), reverse=True)
    # max_examples <= 0 (or None) => render ALL examples (no cap).
    shown = exs if (not max_examples or max_examples <= 0) else exs[:max_examples]
    for e in shown:
        tgt = (e.get("target_smiles") or "").strip()
        gt = e.get("gt_similarity", 0.0)
        outc = e.get("outcome", "")
        lines.append(f"  EXAMPLE (gt_sim {gt:.2f}{(', ' + outc) if outc else ''}): "
                     f"target {tgt}")
        for st in (e.get("steps") or []):
            lines.append(f"      {st}")
        er = (e.get("reasoning") or "").strip()
        if er:
            lines.append(f"      reasoning: {er}")
    return lines


def render_skills(selected, max_examples=2):
    """Render selected skills as the prompt block, grouped by category, using the
    same delimiters the planner already greps for in llm_responses_*.jsonl."""
    if not selected:
        return ""
    by_cat = {}
    for s in selected:
        by_cat.setdefault(s.get("category", "general"), []).append(s)
    lines = ["\n=== RETROSYNTHESIS STRATEGY GUIDE ==="]
    for cat in FIXED_CATEGORIES:
        if cat not in by_cat:
            continue
        lines.append(f"\n## {cat.replace('_', ' ').title()}")
        for s in by_cat[cat]:
            lines.extend(_render_one(s, max_examples))
    lines.append("\n=== END STRATEGY GUIDE ===\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Credit assignment
# ---------------------------------------------------------------------------

def _add_provenance(skill, record):
    """Append a provenance/lineage record, keeping at most _PROVENANCE_CAP
    (oldest dropped). Records who created / merged into / credited a skill so the
    store is auditable."""
    trail = skill.setdefault("provenance", [])
    trail.append(record)
    if len(trail) > _PROVENANCE_CAP:
        del trail[:len(trail) - _PROVENANCE_CAP]


def _provenance_record(action, idx, attempt, example):
    """Build a 'created'/'merged' provenance record from the attached example."""
    ex = example or {}
    return {
        "idx": idx, "attempt": attempt, "action": action,
        "outcome": ex.get("outcome", ""),
        "gt_sim": round(float(ex.get("gt_similarity", 0.0) or 0.0), 3),
    }


def _ex_gt(example):
    """Ground-truth closeness of the route an example was built from, in [0,1]."""
    return max(0.0, min(1.0, float((example or {}).get("gt_similarity", 0.0) or 0.0)))


def apply_reward(mem, selected_ids, reward, gt_sim=0.0, idx=-1, attempt=1):
    """Credit assignment: for each injected skill bump trials (+1), wins
    (+reward), gt_weight (+gt_sim), and append a 'credited' provenance record.
    Returns the count actually updated."""
    ids = set(selected_ids or [])
    n = 0
    for s in mem.get("skills", []):
        if s.get("id") in ids:
            s["trials"] = s.get("trials", 0) + 1
            s["wins"] = s.get("wins", 0.0) + float(reward)
            s["gt_weight"] = s.get("gt_weight", 0.0) + max(0.0, float(gt_sim))
            _add_provenance(s, {
                "idx": idx, "attempt": attempt, "action": "credited",
                "reward": round(float(reward), 3), "gt_sim": round(float(gt_sim), 3),
            })
            n += 1
    return n


# ---------------------------------------------------------------------------
# Evolution: add / merge
# ---------------------------------------------------------------------------

def _normalise(text):
    return " ".join((text or "").lower().split())


def _next_id(mem):
    mx = 0
    for s in mem.get("skills", []):
        sid = s.get("id", "")
        if sid.startswith("s") and sid[1:].isdigit():
            mx = max(mx, int(sid[1:]))
    return f"s{mx + 1:04d}"


def _dedup_key(instructions):
    # Dedup on instruction text only; name/trigger are cosmetic and may differ
    # between two extractions of the same underlying rule.
    return _normalise(instructions or "")


def _add_example(skill, example):
    """Append a concrete example, keeping at most _MAX_EXAMPLES (highest
    gt_similarity retained)."""
    if not example or not (example.get("target_smiles") or example.get("steps")):
        return
    exs = skill.setdefault("examples", [])
    exs.append(example)
    # _MAX_EXAMPLES <= 0 => keep ALL examples (no cap).
    if _MAX_EXAMPLES and len(exs) > _MAX_EXAMPLES:
        exs.sort(key=lambda e: e.get("gt_similarity", 0.0), reverse=True)
        del exs[_MAX_EXAMPLES:]


def _apply_merge(s, idx, name, trigger, template, reasoning, example, attempt,
                 action="merged"):
    """Fold a new extraction into an existing skill `s`: refresh recency, backfill
    empty cosmetic fields (never overwrite), append the concrete example, bump
    origin_gt, and record provenance. Shared by difflib merge_or_add and the
    LLM-driven merge_into."""
    s["updated_idx"] = idx
    if not (s.get("name", "") or "").strip():
        s["name"] = name
    if not (s.get("trigger", "") or "").strip():
        s["trigger"] = trigger
    if not (s.get("template", "") or "").strip():
        s["template"] = template
    if not (s.get("reasoning", "") or "").strip():
        s["reasoning"] = reasoning
    _add_example(s, example)
    # Track the BEST ground-truth closeness this skill has ever been distilled
    # from (retention/ranking preference; see thompson_select).
    s["origin_gt"] = max(s.get("origin_gt", 0.0), _ex_gt(example))
    _add_provenance(s, _provenance_record(action, idx, attempt, example))
    return ("merged", s)


def merge_into(mem, skill_id, category, instructions, idx, name="", trigger="",
               template="", reasoning="", example=None, attempt=1):
    """Merge a new extraction into a SPECIFIC existing skill chosen upstream
    (e.g. by an LLM semantic-dedup call). If `skill_id` is missing or eliminated
    in the current store (staleness — the snapshot the LLM saw may have changed
    under the lock), fall back to the mechanical merge_or_add so nothing is lost.
    Returns ("merged"|"added"|"skipped", skill)."""
    if skill_id:
        for s in mem.get("skills", []):
            if s.get("id") == skill_id and s.get("state") != STATE_ELIMINATED:
                return _apply_merge(s, idx, (name or "").strip(),
                                    (trigger or "").strip(), (template or "").strip(),
                                    (reasoning or "").strip(), example, attempt,
                                    action="merged_llm")
    return merge_or_add(mem, category, instructions, idx, name=name, trigger=trigger,
                        template=template, reasoning=reasoning, example=example,
                        attempt=attempt)


def merge_or_add(mem, category, instructions, idx, name="", trigger="",
                 template="", reasoning="", example=None, attempt=1):
    """Add a new skill, or merge into an existing near-duplicate within the same
    category (difflib ratio >= _DEDUP_RATIO on instruction text). Returns
    ("added"|"merged"|"skipped", skill).

    On merge: append the new concrete example, refresh recency, and BACKFILL empty
    fields (name/trigger/template/reasoning) but never overwrite existing ones."""
    category = category if category in FIXED_CATEGORIES else "general"
    instructions = (instructions or "").strip()
    name = (name or "").strip()
    trigger = (trigger or "").strip()
    template = (template or "").strip()
    reasoning = (reasoning or "").strip()
    if not instructions:
        return ("skipped", None)
    # ---- Chemistry-keyed dedup (exact reaction SMARTS).
    # When the template is a real, machine-extracted reaction SMARTS (the stereo
    # corpus: rdchiral supplies it from the atom mapping), IDENTITY OF THE REACTION
    # is the only correct dedup key. Prose similarity is not: it merged an amide-
    # from-acid with an amide-from-acyl-chloride (verified), which is the same
    # failure that filed an N-methylation under a lactam skill (s0144). Exact
    # SMARTS match => same skill; anything else => a new skill. Category is NOT
    # scoped here (the old scoping let one reaction live twice under two
    # categories -- 37 such pairs measured).
    if is_reaction_smarts(template):
        for s in mem.get("skills", []):
            if s.get("state") == STATE_ELIMINATED:
                continue
            if (s.get("template") or "").strip() == template:
                return _apply_merge(s, idx, name, trigger, template, reasoning,
                                    example, attempt)
        return _add_new(mem, category, instructions, idx, name, trigger,
                        template, reasoning, example, attempt)

    # ---- Legacy prose dedup (pre-existing stores whose templates are the old
    # generic R1/Ar/X placeholders and cannot be compared as chemistry).
    key = _dedup_key(instructions)
    for s in mem.get("skills", []):
        if s.get("category") != category:
            continue
        if s.get("state") == STATE_ELIMINATED:
            continue  # don't resurrect tombstones via merge
        if is_reaction_smarts(s.get("template", "")):
            continue  # never prose-merge into a chemistry-keyed skill
        ratio = difflib.SequenceMatcher(
            None, key, _dedup_key(s.get("instructions", "") or s.get("lesson", ""))
        ).ratio()
        # Same skill if EITHER the instructions match, OR (within-category) the
        # name or template is near-identical -- catches same-reaction skills whose
        # instruction wording differs (e.g. 3x "Amide bond formation ...").
        name_ratio = (difflib.SequenceMatcher(None, _dedup_key(name),
                      _dedup_key(s.get("name", ""))).ratio()
                      if name and s.get("name") else 0.0)
        tmpl_ratio = (difflib.SequenceMatcher(None, _dedup_key(template),
                      _dedup_key(s.get("template", ""))).ratio()
                      if template and s.get("template") else 0.0)
        if (ratio >= _DEDUP_RATIO or name_ratio >= _NAME_DEDUP_RATIO
                or tmpl_ratio >= _TEMPLATE_DEDUP_RATIO):
            return _apply_merge(s, idx, name, trigger, template, reasoning,
                                example, attempt)
    return _add_new(mem, category, instructions, idx, name, trigger, template,
                    reasoning, example, attempt)


def is_reaction_smarts(template):
    """True if `template` is a real machine-extracted reaction SMARTS (has a '>>'
    and atom maps), as opposed to the legacy LLM-written generic pattern
    ('R-C(=O)-O-CH2-Ar => ...'). Only real SMARTS can be dedup'd as chemistry."""
    t = (template or "").strip()
    return bool(t) and ">>" in t and ":" in t and "=>" not in t


def _add_new(mem, category, instructions, idx, name, trigger, template,
             reasoning, example, attempt):
    skill = {
        "id": _next_id(mem),
        "category": category,
        "name": name,
        "trigger": trigger,
        "instructions": instructions,
        "template": template,
        "examples": [],
        "reasoning": reasoning,
        "state": STATE_CANDIDATE,   # newborn on probation
        "trials": 0,
        "wins": 0.0,
        "gt_weight": 0.0,
        "origin_gt": _ex_gt(example),   # birth-route ground-truth closeness
        "created_idx": idx,
        "updated_idx": idx,
        "provenance": [],
    }
    _add_example(skill, example)
    _add_provenance(skill, _provenance_record("created", idx, attempt, example))
    mem.setdefault("skills", []).append(skill)
    return ("added", skill)


# ---------------------------------------------------------------------------
# Lifecycle transitions (paper's four states)
# ---------------------------------------------------------------------------

def evolve_lifecycle(mem, promote_min_trials=3, promote_winrate=0.5,
                     retire_min_trials=4, retire_winrate=0.25,
                     eliminate_min_trials=6, eliminate_winrate=0.05,
                     gt_protect=0.5):
    """Evidence-based state transitions (MemoAttack). Mutates skills in place;
    returns {"promoted": [...], "retired": [...], "reactivated": [...],
    "eliminated": [...]}.

    - candidate -> active  (promotion):    trials>=promote_min & winrate>=promote_winrate
    - retired   -> active  (reactivation): same positive-evidence test
    - active    -> retired (retirement):   trials>=retire_min  & winrate<retire_winrate
    - retired   -> eliminated (elimination): repeated underperformance
        (trials>=eliminate_min & winrate<=eliminate_winrate) AND the skill's
        birth route was not close to ground truth (origin_gt < gt_protect).
        Reactivation is checked first, so a recovering retired skill is never
        eliminated in the same pass. (gt_protect is a closeness THRESHOLD in
        [0,1], not an epsilon — skills distilled from near-ground-truth routes
        are kept regardless of winrate.)

    Retirement/elimination are soft w.r.t. the store (the skill stays); only
    storage_cleanup physically removes anything. Eliminated skills no longer
    participate in selection."""
    tr = {"promoted": [], "retired": [], "reactivated": [], "eliminated": []}
    for s in mem.get("skills", []):
        st = s.get("state", STATE_CANDIDATE)
        if st == STATE_ELIMINATED:
            continue
        t = s.get("trials", 0)
        wr = _winrate(s)
        if st in (STATE_CANDIDATE, STATE_RETIRED):
            if t >= promote_min_trials and wr >= promote_winrate:
                s["state"] = STATE_ACTIVE
                tr["reactivated" if st == STATE_RETIRED else "promoted"].append(s.get("id"))
                continue
        if st == STATE_ACTIVE:
            if t >= retire_min_trials and wr < retire_winrate:
                s["state"] = STATE_RETIRED
                tr["retired"].append(s.get("id"))
                continue
        if st == STATE_RETIRED:
            if (t >= eliminate_min_trials and wr <= eliminate_winrate
                    and s.get("origin_gt", 0.0) < gt_protect):
                s["state"] = STATE_ELIMINATED
                tr["eliminated"].append(s.get("id"))
    return tr


def storage_cleanup(mem, max_total=0, gt_protect=0.5):
    """Physical removal under capacity pressure (paper: "storage cleanup rather
    than the lifecycle itself"). Removes eliminated units first, then aged retired
    units, and ONLY while len(skills) > max_total. Default max_total=0 -> no-op
    (unbounded store; the old 40-cap is gone). Never removes a skill whose birth
    route was close to ground truth (origin_gt >= gt_protect). Returns removed
    ids."""
    skills = mem.get("skills", [])
    removed = []
    if not max_total or max_total <= 0:
        return removed
    while len(skills) > max_total:
        elim = [s for s in skills if s.get("state") == STATE_ELIMINATED
                and s.get("origin_gt", 0.0) < gt_protect]
        retired = [s for s in skills if s.get("state") == STATE_RETIRED
                   and s.get("origin_gt", 0.0) < gt_protect]
        if elim:
            victim = min(elim, key=lambda s: (s.get("updated_idx", -1), _winrate(s)))
        elif retired:
            victim = min(retired, key=lambda s: (s.get("updated_idx", -1), _winrate(s)))
        else:
            break  # nothing safe to remove (protect actives/candidates/gt)
        skills.remove(victim)
        removed.append(victim.get("id"))
    mem["skills"] = skills
    return removed
