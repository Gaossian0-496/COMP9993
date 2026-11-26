"""Canonical reaction-centre signature: the durable identity of a skill.

WHY THIS EXISTS (2026-07-20). Skills were previously identified by the
functional-group delta of their *example molecule* (`distill_skills_highlevel.
cluster_key`). That key is contaminated by accessory groups: three verbatim
carbamate disconnections landed in three different groups merely because one
example contained an aniline and another an extra ether, so they were never
merged and all three were injected into the same prompt. 95% of the distilled
store turned out to be redundant this way, and the k injected slots were spent
on paraphrases of ONE idea.

The fix is to identify a skill by WHAT THE REACTION DOES, not by what molecule
it happened to be demonstrated on. The rdchiral template carries atom maps, so
the reaction centre is recoverable exactly: align the two sides by map number
and keep only the bonds whose order CHANGES. Context atoms (the C=C hanging off
one example but not another) never enter the signature, because they do not
change.

signature("[C:5]-[O:6]-[C:3](=[O:4])-[N:2]-[C:1]>>[C:1]-[N:2]=[C:3]=[O:4].[C:5]-[OH:6]")
    -> "C-N:1>2|C-O:1>0"      (C-N becomes a double bond; C-O breaks)

The same string is produced for every carbamate->isocyanate+alcohol template
regardless of substitution, and a different string for anything that is not that
transformation. Stable across store rebuilds, so it is safe as a persisted key.
"""
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

_ORDER = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
          Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}


def _side_bonds(smarts):
    """{frozenset(map_a, map_b): (order, elem_a, elem_b)} for one side of a
    reaction SMARTS. Atoms without a map number are context we cannot align, so
    bonds touching them are skipped."""
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        return None
    out = {}
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        m1, m2 = a1.GetAtomMapNum(), a2.GetAtomMapNum()
        if not m1 or not m2:
            continue
        e1 = (a1.GetSymbol() or "*").upper()
        e2 = (a2.GetSymbol() or "*").upper()
        e1, e2 = sorted((e1, e2))
        out[frozenset((m1, m2))] = (_ORDER.get(b.GetBondType(), 1), e1, e2)
    return out


def reaction_signature(template):
    """Canonical signature of a retro reaction SMARTS 'product>>reactants'.

    Returns "" when the template is unparseable or nothing changes (both are
    treated as 'no usable identity' by callers, which then fall back)."""
    if not template or ">>" not in template:
        return ""
    prod_s, react_s = template.split(">>", 1)
    prod, react = _side_bonds(prod_s), _side_bonds(react_s)
    if prod is None or react is None:
        return ""
    changes = []
    for key in set(prod) | set(react):
        p = prod.get(key)
        r = react.get(key)
        po = p[0] if p else 0            # 0 = bond absent on that side
        ro = r[0] if r else 0
        if po == ro:
            continue                     # unchanged -> context, not the centre
        e1, e2 = (p or r)[1], (p or r)[2]
        changes.append(f"{e1}-{e2}:{po}>{ro}")
    return "|".join(sorted(changes))


def skill_signature(skill, source_lookup=None):
    """Signature for a stored skill.

    Uses the skill's own template when present (stereo/template stores). For
    distilled high-level skills (template == "") it falls back to the majority
    signature of the source skills recorded in provenance.member_ids, resolved
    through `source_lookup` ({skill_id: source_skill})."""
    sig = reaction_signature(skill.get("template") or "")
    if sig:
        return sig
    if not source_lookup:
        return ""
    from collections import Counter
    c = Counter()
    for p in (skill.get("provenance") or []):
        for mid in (p.get("member_ids") or []):
            src = source_lookup.get(mid)
            if not src:
                continue
            s = reaction_signature(src.get("template") or "")
            if s:
                c[s] += 1
    return c.most_common(1)[0][0] if c else ""


def dedup_key(skill, sig_key="signature"):
    """Effective identity used to detect duplicate skills at injection.

    Prefers the canonical reaction signature. Falls back to the FG-delta group
    hash for skills whose source template never parsed (97/1167 of the distilled
    store): that key is the contaminated one this module exists to replace, but
    for those skills it is the only identity available, and 9 of their groups do
    contain real duplicates -- a weak key beats no dedup at all. Returns "" only
    when neither exists, which means 'never dedupe this'."""
    return (skill.get(sig_key) or "") or (
        ("fg:" + skill["_coarse_hash"]) if skill.get("_coarse_hash") else "")


def diversify(skills, k, sig_key="signature", per_sig=1, backfill=False):
    """Pick up to k skills covering DISTINCT transformations, preserving input
    order (i.e. relevance rank). At most `per_sig` skills per signature.

    `backfill` is OFF by default and that is deliberate: if only 2 distinct
    disconnections are relevant, injecting 2 beats padding to 5 with three
    paraphrases. Padding is exactly the failure this module exists to prevent --
    the 2026-07-19 run spent all 5 slots on one idea and the model received one
    suggestion repeated five times. Fewer, distinct skills is the goal, not a
    full block.

    Skills with no signature ("" -- unparseable template) are never deduped
    against each other, since we cannot prove they are the same transformation.

    This is the runtime invariant that makes injection robust to store quality:
    however redundant the store becomes, the k slots always spend on k
    different ideas."""
    if k <= 0:
        return list(skills)
    picked, seen, spill = [], {}, []
    for s in skills:
        sig = dedup_key(s, sig_key)
        if sig and seen.get(sig, 0) >= per_sig:
            spill.append(s)
            continue
        seen[sig] = seen.get(sig, 0) + 1
        picked.append(s)
        if len(picked) >= k:
            return picked
    if backfill:
        for s in spill:
            if len(picked) >= k:
                break
            picked.append(s)
    return picked
