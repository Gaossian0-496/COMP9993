"""Situational rules as SKILLS: "in this situation, stop" / "in this situation, keep going".

Mined by analysis/mine_rules.py from 11,728 logged expansions and stored in situational_rules.json with
their evidence attached (n, measured rate, lift against a 41.6% base). This module only matches a rule to
the situation the search is in; it decides nothing. The advisor is shown the matching rules and their
numbers and makes the call, because a rule mined from our own solved runs is evidence, not a verdict --
see the `_caveats` field in the library.

A rule's `advice` is "stop", "continue", or "caution" -- the last for a property of the chemistry rather
than of the search's choices (stereocentre density), which lowers the prior without being grounds to
retire anything.

A rule's `when` is a map of feature -> [low, high], inclusive on both ends. A feature the situation does
not know (most often `splits`, which is None until the checker has run on that molecule) does NOT match:
absence of a measurement is never treated as a value, which is the same rule the pull advisor follows for
accepted-split counts.
"""
import json
import os

_CACHE = {}


def load(path="situational_rules.json"):
    """Parsed library, cached per path. A missing or broken file yields no rules rather than raising:
    the arm must degrade to "no rules matched", never take down a search."""
    if path in _CACHE:
        return _CACHE[path]
    full = path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    try:
        with open(full) as f:
            lib = json.load(f)
        rules = [r for r in (lib.get("rules") or []) if isinstance(r.get("when"), dict)]
        out = {"rules": rules, "base_rate": lib.get("_base_rate"), "error": None}
    except Exception as e:
        out = {"rules": [], "base_rate": None, "error": str(e)[:200]}
    _CACHE[path] = out
    return out


def _in(value, bounds):
    if value is None or not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return False
    try:
        return float(bounds[0]) <= float(value) <= float(bounds[1])
    except (TypeError, ValueError):
        return False


def matches(situation, path="situational_rules.json"):
    """Every rule whose `when` is satisfied by `situation`, worst-rate first so the strongest stop
    signal reads first. `situation` keys: depth, size_ratio, splits, tries, calls."""
    lib = load(path)
    hits = []
    for r in lib["rules"]:
        when = r["when"]
        if when and all(_in(situation.get(k), v) for k, v in when.items()):
            hits.append(r)
    hits.sort(key=lambda r: float((r.get("evidence") or {}).get("on_path", 1.0)))
    return hits


def render(hits, base_rate=None):
    """The prompt block, composed from the skill fields (name / applies / why / plan) the way the
    hand-written library renders, with the measurement attached so the advisor can weigh it. Empty list
    -> "" so the prompt is unchanged when nothing matches."""
    if not hits:
        return ""
    L = ["SITUATIONAL SKILLS THAT MATCH WHERE THE PLANNER IS NOW. Each was mined from this planner's own "
         "past runs: `n` expansions were in that situation and `reached a route` is the share of those "
         "molecules that ended up in the route which actually solved its target"
         + (f", against an overall rate of {round(100 * base_rate)}%." if base_rate else ".")]
    for r in hits:
        ev = r.get("evidence") or {}
        L.append("")
        L.append(f"  [{str(r.get('advice', '?')).upper()}] {r.get('name', r.get('id', ''))}")
        if r.get("applies"):
            L.append(f"        applies when: {r['applies']}")
        if r.get("why"):
            L.append(f"        why: {r['why']}")
        if r.get("avoid"):
            L.append(f"        do NOT apply it when: {r['avoid']}")
        for step in (r.get("plan") or []):
            L.append(f"        - {step}")
        bits = []
        if ev.get("n"):
            bits.append(f"n={ev['n']}")
        if ev.get("on_path") is not None:
            bits.append(f"reached a route {round(100 * float(ev['on_path']))}% of the time")
        if ev.get("lift") is not None:
            bits.append(f"{ev['lift']}x the overall rate")
        cov = ev.get("coverage") or {}
        if cov.get("ratio"):
            bits.append(f"fires {cov['ratio']}x more often inside targets that FAILED "
                        f"({round(100 * cov['on_failed'])}% of their expansions) than inside ones that "
                        f"solved ({round(100 * cov['on_solved'])}%)")
        if bits:
            L.append("        measured: " + "; ".join(bits))
        for ex in (r.get("examples") or [])[:2]:
            L.append(f"        example ({ex.get('outcome', '?')}): {ex.get('molecule', '')} "
                     f"at depth {ex.get('depth', '?')}, {round(100 * float(ex.get('size_ratio', 0)))}% of "
                     f"its target's size"
                     + (f" -- {ex['note']}" if ex.get("note") else ""))
    L.append("")
    L.append("  These are measured tendencies from targets that solved, not proof about this molecule: the "
             "rates say how often STAYING paid off, never what would have happened had the search stopped. "
             "Weigh them with everything else above.")
    return "\n".join(L)
