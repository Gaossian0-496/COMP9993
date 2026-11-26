"""Functional-group applicability matcher for skill retrieval (Run S).

Replaces the scaffold-biased ECFP4 prefilter. Instead of "is this skill's
example product structurally similar to the target?" it asks the retrosynthesis-
relevant question: "does the target carry the reactive functional groups this
skill operates on?" -- aggregated over ALL of a skill's examples, not just the
first one.

Chemistry-semantic, RDKit-only, offline, interpretable. No neural embedder.

Signal per skill = TF-IDF-weighted set of functional groups seen across all its
example target_smiles. Distinctive groups (azide, sulfonate, ...) dominate;
ubiquitous ones (aromatic ring, ether) are down-weighted, so a skill is retrieved
for *what it disconnects*, not for sharing a benzene ring with the target.
"""
import math
from rdkit import Chem, RDLogger
from rdkit.Chem import Fragments
RDLogger.DisableLog('rdApp.*')

# The 85 RDKit fr_* detectors are a ready-made functional-group vocabulary.
_FG_FUNCS = [(name, getattr(Fragments, name))
             for name in dir(Fragments) if name.startswith("fr_")]


def fg_set(smiles):
    """Set of functional-group names present in a molecule (>=1 match)."""
    m = Chem.MolFromSmiles(smiles or "")
    if m is None:
        return frozenset()
    out = set()
    for name, fn in _FG_FUNCS:
        try:
            if fn(m) > 0:
                out.add(name)
        except Exception:
            pass
    return frozenset(out)


def skill_fg_profile(skill):
    """FG *frequency* across ALL of a skill's example products: {fg: fraction of
    examples containing it}. Frequency (not a bare union) is what distinguishes a
    skill that is ABOUT azides (azide in most examples) from one that merely saw
    an azide once among thousands -- the union saturates, the frequency doesn't.
    Returns (freq_dict, n_examples_parsed)."""
    from collections import Counter
    cnt, n = Counter(), 0
    for ex in (skill.get("examples") or []):
        smi = (ex.get("target_smiles") or "").strip()
        if not smi:
            continue
        s = fg_set(smi)
        if s:
            for fg in s:
                cnt[fg] += 1
            n += 1
    if n == 0:
        return {}, 0
    return {fg: c / n for fg, c in cnt.items()}, n


_RETRON_Q_CACHE = {}   # template-LHS string -> [compiled SMARTS query mols]


def _retron_queries(template):
    """Compiled SMARTS query mols for a retro template's PRODUCT side (the LHS
    before '>>'), cached. Returns [] if the template is missing/unparseable."""
    if not template or '>>' not in template:
        return []
    lhs = template.split('>>', 1)[0]
    cached = _RETRON_Q_CACHE.get(lhs)
    if cached is not None:
        return cached
    qs = []
    for part in lhs.split('.'):        # product side is normally one component
        try:
            q = Chem.MolFromSmarts(part)
        except Exception:
            q = None
        if q is not None:
            qs.append(q)
    _RETRON_Q_CACHE[lhs] = qs
    return qs


def retron_matches(fragment_mol_or_smiles, template):
    """True iff the skill's template PRODUCT pattern matches the fragment as a
    substructure -- i.e. the disconnection has a valid retron in THIS fragment.

    This is the discriminator FG-overlap lacks: an alkene present in the target
    makes a Sonogashira skill FG-relevant, but the Sonogashira product pattern
    (an aryl/vinyl-alkynyl bond) has no substructure match, so it is rejected.
    Necessary-not-sufficient: a passing skill has a real retron here; it may still
    not be the best cut, but an impossible cut can never pass."""
    qs = _retron_queries(template)
    if not qs:
        return False
    m = fragment_mol_or_smiles
    if isinstance(m, str):
        m = Chem.MolFromSmiles(m or '')
    if m is None:
        return False
    return any(m.HasSubstructMatch(q) for q in qs)


def retron_specificity(fragment_mol_or_smiles, template):
    """Heavy-atom size of the LARGEST matching product pattern -- how much of the
    local environment the template constrains. A 2-atom `[C:1]-[N:2]` matches any
    amine (coincidental, low specificity); a 7-atom pattern pinning the carbonyl
    and both flanking groups is a real, specific retron. Returns 0 if none match."""
    m = fragment_mol_or_smiles
    if isinstance(m, str):
        m = Chem.MolFromSmiles(m or '')
    if m is None:
        return 0
    best = 0
    for q in _retron_queries(template):
        if m.HasSubstructMatch(q):
            best = max(best, q.GetNumAtoms())
    return best


class FGMatcher:
    """Precompute per-skill FG profiles + corpus IDF, then rank by weighted
    functional-group overlap with the target."""

    def __init__(self, skills):
        self.skills = skills
        self.profiles = [skill_fg_profile(s)[0] for s in skills]   # {fg: freq}
        # IDF over the skill corpus: rare FGs are informative, common ones aren't.
        df = {}
        for prof in self.profiles:
            for fg in prof:
                df[fg] = df.get(fg, 0) + 1
        n = max(len(self.profiles), 1)
        self.idf = {fg: math.log((1 + n) / (1 + c)) + 1.0 for fg, c in df.items()}

    def score(self, target_fgs):
        """Score = sum over FGs SHARED with the target of idf(fg) * skill's
        frequency of that fg, normalised by the target's total idf weight. So a
        skill scores high only for the target's *distinctive* groups (high idf,
        e.g. azide) AND only if that group is characteristic of the skill (high
        frequency across its examples) -- not merely present once."""
        tw = sum(self.idf.get(fg, 1.0) for fg in target_fgs) or 1.0
        out = []
        for prof in self.profiles:
            s = 0.0
            for fg in target_fgs:
                if fg in prof:
                    s += self.idf.get(fg, 1.0) * prof[fg]
            out.append(s / tw)
        return out

    def rank(self, target_smiles, top_n):
        tfgs = fg_set(target_smiles)
        scores = self.score(tfgs)
        order = sorted(range(len(self.skills)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in order[:top_n]], tfgs
