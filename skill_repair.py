"""Repair skills for AOT (2026-10-12), AutoGuide-style selection.

When an AOT expansion is stuck (a retry after a failed attempt, or a re-expansion of a
molecule that already has unsolved reactions) one short selector call looks at the
molecule and what has been tried, and picks at most `max_skills` entries from the
library (or NONE). The chosen entry is appended to the normal expansion prompt as an
OPTIONAL idea. First attempts never see a skill, and the expansion prompt is otherwise
unchanged. Pure logic, no LLM client: the caller supplies the selector's answer.

Library entry: {id, kind, name, applies, avoid, why, plan[]} (analysis/build_skill_library.py).
"""
import json
import re


def load_library(path):
    with open(path) as f:
        lib = json.load(f)
    ids = [e["id"] for e in lib]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate skill ids in library")
    return lib


_MOLS = {}


def _mol(smiles):
    if smiles not in _MOLS:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        _MOLS[smiles] = Chem.MolFromSmiles(smiles or "")
    return _MOLS[smiles]


def gate_specificity(smiles, entry):
    """Structural gate (2026-10-13). `entry["gate"]` is a list of SMARTS; the entry is eligible iff
    EVERY pattern matches the molecule. Returns the summed heavy-atom count of the patterns (more
    atoms = a more specific match), 0 if the gate fails or the entry has no gate or the molecule does
    not parse. Replaces the selector LLM's reading of a raw SMILES, which named groups the molecule did
    not contain in 20% of its one-line descriptions (audit of run A_capsskills).

    2026-11-20: ALL-match, was ANY-match. Every one of the 19 gated entries in skill_library_gated.json
    carries exactly ONE pattern, so for them the two readings are identical and nothing changes. The
    change exists because a conjunction cannot be written inside one SMARTS without hand-coding every
    ring shape: "a glycoside AND a fluorine on the sugar" narrows that skill from 5 matching targets to
    3 and drops its exposure to already-solved molecules from 3 to 1. Alternatives within a single
    pattern are still written the old way, as [$(a),$(b)] inside one string."""
    from rdkit import Chem
    mol = _mol(smiles)
    if mol is None:
        return 0
    gates = entry.get("gate") or []
    if not gates:
        return 0
    total = 0
    for g in gates:
        n = _gate_term(mol, g)
        if not n:
            return 0                      # one miss and the entry is not eligible at all
        total += n
    return total


def _gate_term(mol, g):
    """One term of a gate. -> its specificity contribution, or 0 if it does not hold.

    2026-11-20: three forms, because a plain SMARTS can only ask "is this present?" and two of the
    skills in skill_library_pg.json need to ask "are there TWO of them?" and "is this ring bridged?",
    which are COUNTS and cannot be written as a substructure at all.
      "SMARTS"                                      must match
      {"any_of": [SMARTS, ...], "min_distinct": k}  at least k of these DISTINCT patterns match --
                                                    "two different protecting groups", which counting
                                                    matches of one OR'd pattern would get wrong (two
                                                    Boc groups are two matches but one kind of cap)
      {"descriptor": name, "min": k}                an RDKit ring count is at least k; only the two
                                                    names below, so a library file cannot call anything
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    if isinstance(g, str):
        q = Chem.MolFromSmarts(g)
        return q.GetNumAtoms() if (q is not None and mol.HasSubstructMatch(q)) else 0
    if not isinstance(g, dict):
        return 0
    if "any_of" in g:
        pats = [Chem.MolFromSmarts(s) for s in (g.get("any_of") or [])]
        hit = [q for q in pats if q is not None and mol.HasSubstructMatch(q)]
        need = int(g.get("min_distinct", 1))
        return sum(q.GetNumAtoms() for q in hit) if len(hit) >= need else 0
    if "descriptor" in g:
        fns = {"bridgehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms,
               "spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms}
        fn = fns.get(g["descriptor"])
        if fn is None:
            return 0
        try:
            v = fn(mol)
        except Exception:
            return 0
        return int(v) if v >= int(g.get("min", 1)) else 0
    return 0


def eligible(smiles, lib, shown=()):
    """[(specificity, entry)] for gated entries whose gate passes and that were not already shown
    for this molecule, best first: specificity, then corpus support, then id (deterministic)."""
    shown = set(shown)
    out = []
    for e in lib:
        if e["id"] in shown:
            continue
        s = gate_specificity(smiles, e)
        if s:
            out.append((s, e))
    out.sort(key=lambda t: (-t[0], -(t[1].get("support") or 0), t[1]["id"]))
    return out


def menu_text(lib):
    return "\n".join(f"{e['id']}: {e['name']} | applies when: {e['applies']} | avoid when: {e['avoid']}"
                     for e in lib)


def build_selector_prompt(smiles, tried, last_outcome, lib, max_skills=2):
    tried_txt = "\n".join(f"- {t}" for t in (tried or [])[:8]) or "- (none recorded)"
    why = {"no_answer": "the model gave no answer", "no_route_tag": "the answer had no route",
           "unparseable": "the route could not be parsed"}.get(last_outcome or "", "")
    return (
        "You choose strategy hints for a retrosynthesis planner that is stuck on one molecule.\n\n"
        f"MOLECULE: {smiles}\n"
        f"REACTIONS ALREADY TRIED FOR IT (none led to a solved route):\n{tried_txt}\n"
        + (f"LAST ATTEMPT FAILED BECAUSE: {why}\n" if why else "")
        + "\nHINT MENU (id: name | when it applies | when to avoid it):\n" + menu_text(lib) + "\n\n"
        "Task: (1) in one sentence, describe the molecule's reactive groups and what could be "
        "blocking the tried disconnections; (2) pick AT MOST "
        f"{max_skills} hint ids whose 'applies when' clearly holds for this molecule and whose "
        "'avoid when' does not. Prefer picking nothing to picking a hint that only loosely fits.\n"
        "Answer in exactly two lines:\nSTATE: <one sentence>\nPICK: <id>, <id>   (or PICK: NONE)\n")


def parse_selection(answer, lib, max_skills=2):
    """-> (ids, state). Unknown ids are dropped; no PICK line or NONE -> []."""
    if not answer:
        return [], ""
    state = ""
    m = re.search(r"STATE:\s*(.+)", answer)
    if m:
        state = m.group(1).strip()[:400]
    picks = re.findall(r"PICK:\s*(.+)", answer)
    if not picks:
        return [], state
    known = {e["id"] for e in lib}
    ids = []
    for tok in re.split(r"[,\s]+", picks[-1].strip()):
        tok = tok.strip("`*[]().")
        if tok in known and tok not in ids:
            ids.append(tok)
    return ids[:max_skills], state


def render_hint(ids, lib):
    by = {e["id"]: e for e in lib}
    items = []
    for i in ids:
        e = by[i]
        steps = "; ".join(e.get("plan") or [])
        # 2026-11-20: `examples` is optional and carries worked precedent from our own runs (which arm
        # solved it, in how many calls and steps). An entry may have `plan`, `examples`, or neither.
        egs = e.get("examples") or []
        items.append(f"- {e['name']}. {e['why']}" + (f" Typical sequence: {steps}." if steps else "")
                     + ("".join(f" Worked example: {x}" for x in egs) if egs else "")
                     + f" Do not use it when: {e['avoid']}")
    if not items:
        return ""
    return ("\nOPTIONAL STRATEGY IDEAS (earlier attempts on this molecule did not lead to a solved route; "
            "consider these only if they genuinely fit, and ignore them otherwise):\n"
            + "\n".join(items) + "\n")
