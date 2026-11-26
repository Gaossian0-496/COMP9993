"""Switch choice for AOT (config `aot.switch_choice`, off by default = upstream behaviour).

Upstream, when step 1 of an answer has no applicable USPTO template (or repeats a reaction already
used on that molecule), the checker silently takes the most similar template that applies
(BaseOptimizer.blurry_search) and overwrites the model's reactants with its outcome
(fix_reaction_error). In AOT v3 that wrote 48% of the steps in solved routes, 62% of them near-copies,
and 80% of the steps above the molecules the search got stuck on (docs/experiment_history.md).

With switch choice the checker does not switch silently. It collects up to `max_options` templates
from the same candidate list whose outcome is a NEAR MISS of the model's own reactants (user decision
2026-10-15: "different" outcomes are never offered) and the model is asked, in a free follow-up call,
to pick one or NONE. Later prompts keep the history: the "tried previously" list says which
reactions came from the database (and what the model had proposed), and each expansion prompt shows
the route from the target with who wrote each step.

Pure helpers (RDKit only); the checker side is BaseOptimizer.switch_options / _step1_substitute and
the AOT side AOTOptimizer._switch_choose.
"""
import re

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

CLOSE = ("main", "isomer", "close")        # labels offered to the model; "different" never is
REASON_WORDS = {
    "no_template": "no known reaction (USPTO template) makes this molecule from these reactants",
    "not_applicable": "a known reaction type matches, but it does not apply to this molecule",
    "already_used": "this reaction is already in the search for this molecule",
    "different_outcome": "the known reaction type of your step applies, but it gives different reactants from yours",
}
LABEL_WORDS = {"main": "keeps your main precursor", "isomer": "main precursor is an isomer of yours",
               "close": "main precursor is similar to yours"}
SRC_OWN, SRC_CHOSEN, SRC_TEMPLATE = "own", "chosen", "template"
SRC_SALVAGE = "salvage"   # AOT salvage re-attached a later step (shown to the model like SRC_TEMPLATE)


def _mol(s):
    return Chem.MolFromSmiles(s) if isinstance(s, str) and s else None


def _canon(s, iso=False):
    m = _mol(s)
    return Chem.MolToSmiles(m, isomericSmiles=iso) if m is not None else None


def _main(smiles_list):
    """Largest valid molecule (heavy atoms) of a reactant list, or None."""
    mols = [(m.GetNumHeavyAtoms(), s, m) for s in smiles_list for m in [_mol(s)] if m is not None]
    return max(mols, key=lambda t: t[0]) if mols else None


def closeness(option, mine):
    """(label, tanimoto) of a template outcome `option` vs the model's reactants `mine`, with
    analysis/swap_audit.label's rules for one proposal:
      main     the option's largest reactant is one of the model's reactants (stereo ignored)
      isomer   the two largest reactants have the same formula (regio / ring-size near miss)
      close    Morgan-2 Tanimoto of the two largest reactants >= 0.6
      different otherwise (or nothing to compare)."""
    om, qm = _main(option or []), _main(mine or [])
    if om is None or qm is None:
        return "different", 0.0
    if _canon(om[1]) in {_canon(x) for x in mine if _mol(x) is not None}:
        return "main", 1.0
    fo, fq = (AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in (om[2], qm[2]))
    sim = round(DataStructs.TanimotoSimilarity(fo, fq), 3)
    if rdMolDescriptors.CalcMolFormula(om[2]) == rdMolDescriptors.CalcMolFormula(qm[2]):
        return "isomer", sim
    return ("close" if sim >= 0.6 else "different"), sim


def path_lines(steps):
    """Route from the target down to a molecule, one line per step with who wrote it.
    steps: [{product, reactants, src, orig}] target first."""
    out = []
    for n, s in enumerate(steps, 1):
        if s.get("src") == SRC_CHOSEN:
            who = f"known database reaction you chose instead of your proposal {s.get('orig')}"
        elif s.get("src") in (SRC_TEMPLATE, SRC_SALVAGE):
            who = "supplied by the reaction database, not written by you"
        else:
            who = "your reaction"
        out.append(f"  {n}. {s['product']}>>{'.'.join(s['reactants'])}   [{who}]")
    return out


def route_block(steps):
    """Text appended to an expansion prompt ("" when the molecule is the target)."""
    if not steps:
        return ""
    return ("\nRoute so far, from the final target down to this molecule (for context; database reactions "
            "are marked):\n" + "\n".join(path_lines(steps)) + "\n")


def annotate_tried(reaction, rec):
    """One line of the 'tried previously' list, marking reactions the database supplied."""
    if not rec or rec.get("src") == SRC_OWN:
        return reaction
    if rec.get("src") == SRC_CHOSEN:
        return f"{reaction}   (known database reaction you chose instead of your proposal {rec.get('orig')})"
    return f"{reaction}   (supplied by the reaction database, not written by you)"


def choice_prompt(target, node, qwen_reaction, rationale, reason, options, steps):
    """The follow-up question: pick one near-miss database reaction or NONE."""
    lines = [f"Final target of the synthesis: {target}"]
    if steps:
        lines += ["Route so far, from the final target down to the molecule below:"] + path_lines(steps)
    lines += [f"Molecule to make: {node}",
              "Your proposed first step for this molecule was:",
              f"  {qwen_reaction}"]
    if rationale:
        lines.append(f"  Your reasoning: \"{rationale[:600]}\"")
    lines += [f"The checker cannot accept it as written: {REASON_WORDS.get(reason, 'the checker rejected it')}.",
              "The reaction database has these known reactions for this molecule that are close to your proposal:"]
    for n, o in enumerate(options, 1):
        lines.append(f"  {n}. {node}>>{'.'.join(o['reactants'])}   ({LABEL_WORDS.get(o['label'], o['label'])})")
    lines += ["",
              "Would you like to use one of these known reactions as this step instead of yours? Pick the one "
              "you would actually run, or NONE if none of them is a step you would take (your proposal is then "
              "dropped and you will be asked again later).",
              "Answer with one line 'CHOICE: <number>' or 'CHOICE: NONE', then one sentence explaining why."]
    return "\n".join(lines)


def parse_choice(text, n):
    """1..n, "NONE", or None (unreadable / out of range). The last CHOICE line wins."""
    hits = re.findall(r"CHOICE\s*[:=]\s*\**\s*(NONE|\d+)", text or "", re.I)
    if not hits:
        return None
    h = hits[-1].upper()
    if h == "NONE":
        return "NONE"
    k = int(h)
    return k if 1 <= k <= n else None
