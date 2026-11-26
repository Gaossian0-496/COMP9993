#!/usr/bin/env python3
"""Per-molecule skill mining (arm `mine`, 2026-11-22).

TWO THINGS LIVE HERE.

1. `how_is_this_made(smiles)` -- the callable tool. For the molecule under
   expansion it reports, for each feature it carries, how many recorded
   reactions involve that feature and what those reactions carried on the
   PRECURSOR side, enriched over background. It suggests no disconnection,
   runs no checker and exposes no purchasability, so like the disconnection
   tool it adds chemistry and not an oracle verdict.

2. `derivation_prompt(...)` -- the end-of-molecule skill writer. Called ONCE
   per finished molecule, with the whole search in hand, under hard limits
   (see ADMISSION) so the store cannot grow without bound.

DESIGN RULES, each one learned by getting it wrong first:
  * rank, NEVER filter. A support floor deleted the glycosidic bond (7
    reactions), the silyl ether (0) and the trityl (40) from a nucleoside and
    left "the methoxy comes from an alkyl halide". Low support IS the signal.
  * percentages need support; mentions do not. Quote a precursor share only
    with >= MIN_FOR_PERCENTS behind it; always name the feature regardless.
  * enrichment, not raw frequency. Unenriched, every profile reports that the
    precursor contains an aromatic bond -- true of 94.6% of all precursor sets.
  * skip the self-match. "the precursor to a trityl group is a trityl group"
    is the group arriving intact, which the caller already knows.
  * a ring system that nothing closes is only interesting when it is too big
    to be a catalogue item; nothing builds a CF3 either, and that is normal.
"""
import json, os, collections
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(HERE, "skill_index")
N_REACTIONS = 40008                 # the corpus the feature index was built from
MIN_FOR_PERCENTS = 200              # below this: name the feature, quote no shares
MIN_MENTION = 30                    # a count this low is noise, not rarity. NOT the support floor the
                                    # docstring forbids: that floor was applied to CREATED-only counts
                                    # (glycosidic 7, silyl 0, trityl 40). On the involved-counts index this
                                    # one is inert -- the rarest indexed feature is phosphonate at 100, and
                                    # the three motifs named above are 199 / 303 / 189. It exists only so a
                                    # feature added later with almost no support cannot speak.
MIN_ENRICH = 2.0                    # a precursor share must be this far over background
RARE = 0.02                         # <2% of reactions -> tag it rare, sort it first
COMMON = 0.10                       # >10% -> tag it common, sort it last
BIG_RING = 12                       # a ring system of >= this many atoms cannot be bought

PT = Chem.GetPeriodicTable()

FEATURES = {
    # linkages
    "amide C(=O)N": "[CX3](=O)[NX3]", "ester C(=O)O": "[CX3](=O)[OX2][#6]",
    "carbamate N-C(=O)-O": "[NX3][CX3](=O)[OX2][#6]", "urea N-C(=O)-N": "[NX3][CX3](=O)[NX3]",
    "sulfonamide S(=O)(=O)N": "[SX4](=O)(=O)[NX3]", "biaryl c-c": "[c;R][c;R]",
    "aryl ether c-O-C": "[c][OX2][#6]", "aryl amine c-N": "[c][NX3]",
    "glycosidic N-C(O)": "[#7;R][CX4;R][OX2;R]", "ether C-O-C": "[CX4][OX2][CX4]",
    "C-N sp3": "[CX4][NX3]",
    # functional groups
    "ketone": "[#6][CX3](=O)[#6]", "aldehyde": "[CX3H1](=O)[#6]",
    "carboxylic acid": "[CX3](=O)[OX2H1]", "primary alcohol": "[CX4H2][OX2H1]",
    "secondary alcohol": "[CX4H1][OX2H1]", "primary amine": "[NX3H2][CX4]",
    "secondary amine": "[NX3H1]([CX4])[CX4]", "oxime": "[CX3]=[NX2][OX2H1]",
    "imine": "[CX3]=[NX2][#6]", "nitrile": "[NX1]#[CX2]", "nitro": "[NX3](=O)=O",
    "aryl halide": "[c][F,Cl,Br,I]", "alkyl halide": "[CX4][Cl,Br,I]",
    "boronic acid/ester": "[#5]([OX2])[OX2]", "alkene": "[CX3]=[CX3]", "alkyne": "[CX2]#[CX2]",
    # protecting groups
    "PG silyl ether": "[OX2][Si]([#6])([#6])[#6]",
    "PG trityl/benzhydryl": "[OX2][CX4](c1ccccc1)c1ccccc1",
    "PG Cbz": "[NX3]C(=O)OCc1ccccc1", "PG Boc": "[NX3]C(=O)OC([CH3])([CH3])[CH3]",
    "PG benzyl ether": "[OX2][CH2]c1ccccc1",
    # ring context and the groups the first pass was blind to
    "amine on ring CH": "[NX3;!$(N[CX3]=[OX1,SX1,NX2])][CX4;R;H1]",
    "azide": "[NX2]=[NX2+]=[NX1-]", "sulfonate ester": "[SX4](=O)(=O)[OX2][#6]",
    "sulfone": "[#6][SX4](=O)(=O)[#6]", "thioether": "[#6][SX2][#6]",
    "trifluoromethyl": "[CX4](F)(F)F", "aryl fluoride": "[c][F]",
    "tertiary amine": "[NX3]([CX4])([CX4])[CX4]", "guanidine": "[NX3][CX3](=[NX2])[NX3]",
    "phosphonate/phosphate": "[PX4](=O)([OX2])[OX2]", "epoxide": "[CX4]1[OX2][CX4]1",
    "lactam": "[NX3;R][CX3;R](=O)", "lactone": "[OX2;R][CX3;R](=O)",
    "hydrazide/hydrazone": "[NX3][NX3,NX2]", "quaternary carbon": "[CX4]([#6])([#6])([#6])[#6]",
    "aryl nitro": "[c][NX3](=O)=O",
}
QUERIES = {k: Chem.MolFromSmarts(v) for k, v in FEATURES.items()}

_IDX = {}


def _load():
    if _IDX:
        return _IDX
    with open(os.path.join(INDEX_DIR, "feature_involved.json")) as fh:
        inv = json.load(fh)
    with open(os.path.join(INDEX_DIR, "feature_involved_prec.json")) as fh:
        prof = json.load(fh)
    with open(os.path.join(INDEX_DIR, "ringform_index.json")) as fh:
        ring = json.load(fh)
    # `.get(f, 0)`, not `inv[f]`: the two index files are written separately, and a key present in the
    # precursor profile but missing from the support index would raise KeyError here. The caller wraps
    # everything in a blanket except, so that would not crash the run -- it would silently turn the tool
    # into "how_is_this_made failed: KeyError" for every call of the whole run (code review 2026-11-22).
    tot = sum(inv.get(f, 0) for f in prof)
    bg = collections.Counter()
    for f, d in prof.items():
        for g, c in d.items():
            bg[g] += c
    _IDX.update(inv=inv, prof=prof, ring=ring, P={g: c / tot for g, c in bg.items()})
    return _IDX


def feats(m):
    return {k for k, q in QUERIES.items() if q is not None and m.HasSubstructMatch(q)}


def ring_systems(m):
    ri = m.GetRingInfo(); rings = [set(r) for r in ri.AtomRings()]; sysl = []
    for r in rings:
        hit = [s for s in sysl if s & r]
        if hit:
            hit[0] |= r
            for o in hit[1:]:
                hit[0] |= o; sysl.remove(o)
        else:
            sysl.append(set(r))
    out = []
    for s in sysl:
        het = collections.Counter()
        for i in s:
            n = m.GetAtomWithIdx(i).GetAtomicNum()
            if n and n != 6:
                het[n] += 1
        out.append((sum(1 for r in rings if r <= s), tuple(sorted(het.items())), len(s)))
    return out


def _rlabel(k):
    h = ", ".join(f"{PT.GetElementSymbol(z)}{c}" for z, c in k[1]) or "carbocycle"
    return f"{k[0]} fused ring(s), {k[2]} atoms, {h}"


def how_is_this_made(smiles, top=4):
    """-> JSON string. Never raises."""
    try:
        ix = _load()
        if not smiles:                                  # MolFromSmiles("") returns an EMPTY mol, not None,
            return json.dumps({"error": "no molecule given"})   # which would read as "nothing distinctive"
        m = Chem.MolFromSmiles(smiles)
        if m is None or m.GetNumAtoms() == 0:
            return json.dumps({"error": "could not read the molecule"})
        out = []
        for k in ring_systems(m):                       # coverage, big systems only
            if k[2] >= BIG_RING and len(ix["ring"].get(str(k), [])) == 0:
                out.append({"motif": _rlabel(k),
                            "finding": "no recorded reaction closes this ring system; it has to be "
                                       "built up a ring at a time rather than cut across"})
        rows = []
        for f in feats(m):
            n = ix["inv"].get(f, 0)
            if n < MIN_MENTION:
                continue
            best = []
            for g, c in ix["prof"].get(f, {}).items():
                if g == f or c < 10 or ix["P"].get(g, 0) <= 0:
                    continue                            # skip the self-match
                best.append((c / n / ix["P"][g], g))
            best.sort(reverse=True)
            if not best or best[0][0] < MIN_ENRICH:
                continue
            share = n / N_REACTIONS
            rows.append((share, n, f, best[:2]))
        for share, n, f, best in sorted(rows)[:top]:    # RANK by rarity, drop nothing
            tag = "rare" if share < RARE else ("common" if share > COMMON else "typical")
            out.append({"motif": f, "how_often": f"{100*share:.1f}% of recorded reactions ({tag})",
                        "precursors_usually_carry": [f"{g} ({L:.1f}x more often than chance)"
                                                     for L, g in best]})
        if not out:
            return json.dumps({"note": "nothing distinctive found for this molecule"})
        return json.dumps({"molecule": smiles, "findings": out,
                           "note": "this says how these motifs are MADE, not which bond to break; "
                                   "a rare motif is where the unusual chemistry is"})
    except Exception as e:                              # a tool must never kill an expansion
        return json.dumps({"error": f"how_is_this_made failed: {type(e).__name__}"})


# ---------------------------------------------------------------- derivation
# ADMISSION. The store must not grow without bound, so a molecule may contribute
# AT MOST ONE skill and only when all of these hold:
#   * the molecule finished (solved or exhausted) -- never mid-search
#   * at least one motif came back tagged `rare` (< 2% of reactions)
#   * the accepted step that acted on that motif was NOT one the disconnection
#     tool had offered -- otherwise the tool already supplies it next time
#   * no stored skill's gate already matches this molecule
# and the gate is the SMARTS of the motif that fired, never written by hand.
MAX_SKILLS_PER_RUN = 40

DERIVATION_RULES = """Write ONE skill, or write NOTHING.

Write nothing unless the route turned on something a competent chemist would
not have guessed from the molecule alone. Routine chemistry is not a skill:
amide from an acid, ester from an alcohol, Suzuki from a halide and a boronate,
Boc from an amine. If that is all that happened, answer exactly: NO SKILL.

If you do write one, obey every rule:
1. The skill is about CHEMISTRY. Never mention this search, the budget, how
   many calls it took, the benchmark, the reaction corpus, or any number taken
   from a run.
2. `why` must argue from mechanism: how the bond or ring is MADE in the forward
   direction, and therefore what it traces back to. Not "this worked".
3. `applies` describes the structural condition in words. It must be true of
   molecules other than this one.
4. `avoid` must name a real case where the advice is WRONG. A skill with no
   honest exception is overfitted to its molecule; if you cannot think of one,
   answer NO SKILL.
5. Do not invent chemistry. Every transformation you name must appear in the
   route or in the evidence given to you.
6. At most 120 words in `why`, at most 60 in each of `applies` and `avoid`.
7. `motif` must be copied EXACTLY from one of the motif names listed in the corpus evidence below -- the one
   your skill is actually about. The gate that decides when this skill is shown is built from that motif, so a
   motif that does not match your advice makes the skill fire on the wrong molecules.

Answer with one JSON object in the template, or the two words NO SKILL."""

SKILL_TEMPLATE = {
    "motif": "<copy one motif name exactly from the evidence>",
    "id": "<short_machine_name>",
    "name": "<one line, written as advice>",
    "applies": "<the structural condition, in words>",
    "avoid": "<where this advice is wrong>",
    "why": "<the mechanism: how it is made forwards, so what it traces back to>",
    "examples": ["<the worked case, chemistry only>"],
}


def derivation_prompt(target, solved, route_steps, rejected, findings, tool_offered, reasoning):
    """Everything the writer sees. The gate is NOT asked for -- it is attached in code."""
    return (
        f"{DERIVATION_RULES}\n\n"
        f"TEMPLATE (fill exactly these fields):\n{json.dumps(SKILL_TEMPLATE, indent=1)}\n\n"
        f"TARGET: {target}\n"
        f"OUTCOME: {'a route was found' if solved else 'no route was found'}\n\n"
        f"THE ROUTE (or how far it got):\n{route_steps}\n\n"
        f"STEPS THAT WERE REJECTED, and why:\n{rejected}\n\n"
        f"WHAT THE CORPUS SAYS ABOUT THIS MOLECULE'S MOTIFS:\n{findings}\n\n"
        f"DISCONNECTIONS THE TOOL HAD ALREADY OFFERED (do not write a skill about these):\n"
        f"{tool_offered}\n\n"
        f"THE PLANNER'S OWN REASONING ALONG THE WAY:\n{reasoning}\n")


def attach_gate(skill, motif):
    """The gate is the SMARTS of the motif that fired -- as general as the chemistry,
    never hand-narrowed. Breadth is corrected later from the measured hit rate."""
    sma = FEATURES.get(motif)
    if not sma:
        return None
    skill = dict(skill)
    skill["gate"] = [sma]
    skill["derived_from_motif"] = motif
    skill["stats"] = {"fired": 0, "followed": 0, "accepted": 0}
    skill["status"] = "provisional"
    return skill
