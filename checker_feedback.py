"""Checker feedback for a rejected AOT expansion (config `aot.checker_feedback`, default off).

Upstream tells the search only "valid / invalid". On AOT v3, 24% of answers were
lost to output format and another 24% to an unreadable SMILES in step 1, which
makes the checker drop the whole step before its own template repair can run
(utils.verify_reaction_step: any None reactant -> reaction_valid False). This
module turns such a rejection into a specific, actionable message for one
follow-up call: what broke, why (RDKit's reason), a possibly intended spelling,
and the closest valid disconnection the checker itself found.

Pure functions only (RDKit, no oracle data); aot_search wires them in.
"""
import ast
import re

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

LIST_FIELDS = ("Molecule set", "Product", "Reaction", "Reactants", "Updated molecule set")

# Reagents the model writes as formulas or abbreviations instead of SMILES (most
# common in AOT v3 first: NH3 134, H2O 46, Br2 43, H2 37, NaN3 22, ...).
FORMULA_SMILES = {
    "NH3": "N", "H2O": "O", "Br2": "BrBr", "H2": "[H][H]", "NaN3": "[N-]=[N+]=[N-].[Na+]",
    "Cl2": "ClCl", "I2": "II", "F2": "FF", "LiOH": "[Li+].[OH-]", "NaOH": "[Na+].[OH-]",
    "KOH": "[K+].[OH-]", "SOCl2": "O=S(Cl)Cl", "HCl": "Cl", "HBr": "Br", "HI": "I", "HF": "F",
    "Li": "[Li]", "Na": "[Na]", "K": "[K]", "Mg": "[Mg]", "Zn": "[Zn]", "Pd": "[Pd]",
    "Cu": "[Cu]", "Fe": "[Fe]", "PBr3": "BrP(Br)Br", "PCl3": "ClP(Cl)Cl",
    "PCl5": "ClP(Cl)(Cl)(Cl)Cl", "POCl3": "O=P(Cl)(Cl)Cl", "HCN": "C#N",
    "NaCN": "[C-]#N.[Na+]", "KCN": "[C-]#N.[K+]", "CuCN": "[C-]#N.[Cu+]",
    "B2pin2": "CC1(C)OB(B2OC(C)(C)C(C)(C)O2)OC1(C)C", "NaBH4": "[BH4-].[Na+]",
    "LiAlH4": "[AlH4-].[Li+]", "NaH": "[H-].[Na+]", "H2O2": "OO", "CO2": "O=C=O",
    "CO": "[C-]#[O+]", "SO3": "O=S(=O)=O", "N2H4": "NN", "NH2OH": "NO", "HNO3": "O=[N+]([O-])O",
    "H2SO4": "O=S(=O)(O)O", "NaNO2": "O=N[O-].[Na+]", "K2CO3": "O=C([O-])[O-].[K+].[K+]",
    "Na2CO3": "O=C([O-])[O-].[Na+].[Na+]", "Cs2CO3": "O=C([O-])[O-].[Cs+].[Cs+]",
    "NaHCO3": "O=C([O-])O.[Na+]", "MeOH": "CO", "EtOH": "CCO", "MeI": "CI", "EtI": "CCI",
    "TFA": "O=C(O)C(F)(F)F", "AcOH": "CC(=O)O", "Ac2O": "CC(=O)OC(C)=O", "AcCl": "CC(=O)Cl",
    "Boc2O": "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C", "MsCl": "CS(=O)(=O)Cl",
    "TsCl": "Cc1ccc(S(=O)(=O)Cl)cc1", "TMSCl": "C[Si](C)(C)Cl", "TBSCl": "CC(C)(C)[Si](C)(C)Cl",
    "NBS": "O=C1CCC(=O)N1Br", "NCS": "O=C1CCC(=O)N1Cl", "NIS": "O=C1CCC(=O)N1I",
    "DMF": "CN(C)C=O", "DMSO": "CS(C)=O", "THF": "C1CCOC1", "Et3N": "CCN(CC)CC",
    "TEA": "CCN(CC)CC", "DIPEA": "CCN(C(C)C)C(C)C", "Py": "c1ccncc1", "pyridine": "c1ccncc1",
    "KSCN": "N#C[S-].[K+]", "NaSCN": "N#C[S-].[Na+]", "LDA": "CC(C)[N-]C(C)C.[Li+]",
    "n-BuLi": "[Li]CCCC", "BuLi": "[Li]CCCC", "MeLi": "[Li]C", "MeMgBr": "C[Mg]Br",
    "NaOMe": "C[O-].[Na+]", "NaOEt": "CC[O-].[Na+]", "KOtBu": "CC(C)(C)[O-].[K+]",
}
_ABBREV = re.compile(r"(?<![a-z])(Ph|Bn|Bz|Boc|Cbz|Fmoc|Ts|Tf|OTf|Ms|TBS|TBDMS|TIPS|TMS|PMB|THP|"
                     r"Piv|Ac|Me|Et|Pr|iPr|Bu|tBu|pin|Bpin)(?![a-z])")

REAGENT_NOTE = ("Reagents, solvents, catalysts and bases are not reactants: list only the molecules "
                "whose atoms end up in the product.")

# Formula-style hydrogens inside SMILES, e.g. C#CC(C)(OH)C or OCH2C (outside [...] only).
_H_GROUPS = [(r"\(OH\)", "(O)"), (r"CH3", "C"), (r"CH2", "C"), (r"NH2", "N"), (r"OH", "O"),
             (r"CH", "C"), (r"NH", "N"), (r"SH", "S")]


def _outside_brackets(smi, fn):
    """Apply fn to the parts of smi outside [...] atoms."""
    parts = re.split(r"(\[[^\]]*\])", smi)
    return "".join(p if p.startswith("[") else fn(p) for p in parts)


def _valid(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    return Chem.MolToSmiles(m) if m is not None else None


def _syntax_hint(smi):
    """Concrete syntax problems RDKit's bare parse failure does not name."""
    hints = []
    o, c = smi.count("("), smi.count(")")
    if o != c:
        hints.append(f"unbalanced parentheses ({o} '(' vs {c} ')')")
    if smi.count("[") != smi.count("]"):
        hints.append(f"unbalanced square brackets ({smi.count('[')} '[' vs {smi.count(']')} ']')")
    body = re.sub(r"\[[^\]]*\]", "", smi)
    digits = re.findall(r"%\d\d|\d", body)
    open_rings = [d for d in sorted(set(digits)) if digits.count(d) % 2]
    if open_rings:
        hints.append("ring-bond number(s) " + ", ".join(open_rings) + " opened but never closed")
    ab = sorted(set(_ABBREV.findall(body)))
    if ab:
        hints.append("abbreviation(s) " + ", ".join(ab) + " are not SMILES; write the group's atoms out "
                     "(e.g. Ph = c1ccccc1, Me = C, Ts = S(=O)(=O)c1ccc(C)cc1)")
    if "H" in body:          # a bare H is only legal inside [...] in SMILES
        hints.append("hydrogens written as in a formula (OH, CH2, NH2); in SMILES hydrogens are "
                     "implicit, e.g. C(O) not C(OH), OCC not OCH2C")
    return hints


def _kekule_suggestion(smi):
    """Try the usual aromatic fix: one unkekulizable aromatic N written as [nH]."""
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return None
    for a in m.GetAtoms():
        if a.GetIsAromatic() and a.GetSymbol() == "N" and a.GetNumExplicitHs() == 0 and a.GetDegree() == 2:
            mm = Chem.RWMol(m)
            mm.GetAtomWithIdx(a.GetIdx()).SetNumExplicitHs(1)
            try:
                Chem.SanitizeMol(mm)
                return Chem.MolToSmiles(mm)
            except Exception:
                continue
    return None


def _atom_desc(m, idx):
    try:
        a = m.GetAtomWithIdx(idx)
    except Exception:
        return f"atom number {idx + 1} in writing order"
    nb = ", ".join(sorted(n.GetSymbol() for n in a.GetNeighbors()))
    return (f"atom number {idx + 1} in writing order ({a.GetSymbol()}"
            f"{', aromatic' if a.GetIsAromatic() else ''}, bonded to {nb or 'nothing'})")


def smiles_problem(smi):
    """(reason, suggestion) for an invalid SMILES; (None, None) if it is valid.
    `suggestion` is a parseable SMILES the model may have meant, or None."""
    if _valid(smi):
        return None, None
    if not isinstance(smi, str) or not smi.strip():
        return "empty molecule", None
    s = smi.strip()
    if s in FORMULA_SMILES:
        return (f"'{s}' is a chemical formula or abbreviation, not SMILES. {REAGENT_NOTE}",
                FORMULA_SMILES[s])
    m = Chem.MolFromSmiles(s, sanitize=False)
    if m is None:
        hints = _syntax_hint(s)
        reason = "SMILES syntax error" + (": " + "; ".join(hints) if hints else
                                           " (check ring-bond numbers, brackets and parentheses)")
        sug = None
        cand = _outside_brackets(s, lambda p: _apply_h_groups(p))
        if cand != s:
            sug = _valid(cand)
        return reason, sug
    msgs = []
    for p in Chem.DetectChemistryProblems(m):
        t = p.GetType()
        if t == "AtomValenceException":
            msgs.append(f"{_atom_desc(m, p.GetAtomIdx())} has more bonds than its valence allows")
        elif t == "KekulizeException":
            atoms = list(p.GetAtomIndices())
            msgs.append("the aromatic (lowercase) atoms number " + ", ".join(str(i + 1) for i in atoms) +
                        " in writing order" +
                        " cannot form a valid aromatic ring: check the ring size, that aromatic carbons "
                        "carry at most one substituent, and pyrrole-type nitrogens written as [nH]")
        elif t == "AtomKekulizeException":
            msgs.append(f"{_atom_desc(m, p.GetAtomIdx())} is marked aromatic but is not in a ring")
        else:
            msgs.append(p.Message())
    reason = "; ".join(msgs) or "RDKit sanitization failed (chemistry is inconsistent)"
    try:
        sug = _kekule_suggestion(s) if any("aromatic" in x for x in msgs) else None
    except Exception:
        sug = None
    if sug is None:
        cand = _outside_brackets(s, _apply_h_groups)
        sug = _valid(cand) if cand != s else None
    return reason, sug


def _apply_h_groups(part):
    for pat, rep in _H_GROUPS:
        part = re.sub(pat, rep, part)
    return part


def _as_list(v):
    try:
        x = ast.literal_eval(v) if isinstance(v, str) else v
    except Exception:
        return None
    return x if isinstance(x, list) else None


def diagnose_format(body, step_example="['CCO', 'N']"):
    """Problems that kept a <ROUTE> block from being read. `body` None = no block at all."""
    if body is None:
        return ["The answer has no <ROUTE>...</ROUTE> block. Give the route once, inside "
                "<ROUTE></ROUTE>, and keep the reasoning short so the answer is not cut off."]
    try:
        route = ast.literal_eval(body)
    except SyntaxError as e:
        line = (e.text or "").strip()[:160]
        if line and not re.match(r"[\[\]{}'\"]", line):
            # the failing line is prose, not part of a list of dicts
            return [f"The <ROUTE> block contains text that is not part of the route list (line "
                    f"{e.lineno}: `{line}`). Put only the Python list of step dictionaries inside "
                    f"<ROUTE></ROUTE>; keep the reasoning outside it."]
        return [f"The <ROUTE> block is not a valid Python list of dicts ({e.msg}, line {e.lineno}: "
                f"`{line}`). Every value must be a quoted string, and molecule lists must be "
                f"quoted lists of quoted SMILES, e.g. 'Reactants': \"{step_example}\"."]
    except Exception as e:
        return [f"The <ROUTE> block could not be read ({type(e).__name__}: {str(e)[:120]}). "
                f"Molecule lists must be quoted lists of quoted SMILES, e.g. 'Reactants': \"{step_example}\"."]
    if not isinstance(route, list) or not route or not all(isinstance(s, dict) for s in route):
        return ["The <ROUTE> block must be a non-empty Python list of step dictionaries."]
    out, bad, first = [], [], None
    kinds = {}
    for i, step in enumerate(route, 1):
        for f in LIST_FIELDS:
            v = step.get(f)
            if f not in step:
                out.append(f"Step {i} is missing the field '{f}'.")
            elif not isinstance(v, str):
                kinds.setdefault("raw", []).append(f"step {i} '{f}'")
            elif _as_list(v) is None:
                bad.append(f"step {i} '{f}'")
                first = first or (i, f, v[:120])
            else:
                items = _as_list(v)
                if not all(isinstance(x, str) for x in items):
                    kinds.setdefault("nested", []).append(f"step {i} '{f}'")
                elif not items and f != "Updated molecule set":
                    kinds.setdefault("empty", []).append(f"step {i} '{f}'")
                elif f == "Reaction" and items and ">>" not in items[0]:
                    kinds.setdefault("arrow", []).append(f"step {i}")
    msg = {"raw": "are written as bare Python lists; each must be a STRING holding the list, "
                  f"e.g. 'Reactants': \"{step_example}\" (note the double quotes around the list)",
           "nested": "contain nested lists or non-string items; each must be a flat list of SMILES strings",
           "empty": "are empty lists",
           "arrow": "have a 'Reaction' without '>>'; write it as \"['Product>>Reactant1.Reactant2']\""}
    for k, where in kinds.items():
        out.append(f"{', '.join(where[:12])}{', ...' if len(where) > 12 else ''}: {msg[k]}.")
    if bad:
        # one line for all of them: the cause is the same everywhere
        out.append(f"{len(bad)} field(s) are not quoted lists of quoted SMILES ({', '.join(bad[:12])}"
                   f"{', ...' if len(bad) > 12 else ''}); e.g. step {first[0]} '{first[1]}' is "
                   f"{first[2]!r}. Write each like \"{step_example}\": each molecule in single "
                   f"quotes, the whole list in double quotes.")
    return out


def _reaction_parts(step):
    """(product molecules, reactant molecules) written in step['Reaction'], or (None, None)."""
    r = _as_list(step.get("Reaction")) or []
    if not r or not isinstance(r[0], str) or ">>" not in r[0]:
        return None, None
    p, q = r[0].split(">>", 1)
    return [x for x in p.split(".") if x], [x for x in q.split(".") if x]


def step1_smiles_problems(step):
    """Invalid SMILES in the model's first step, one problem line per distinct molecule, plus
    a 'Reaction' whose reactants differ from the 'Reactants' field (the checker looks the
    template up from 'Reaction' but verifies it against 'Reactants')."""
    out, seen = [], set()
    rp, rr = _reaction_parts(step)
    fields = [(f, _as_list(step.get(f)) or []) for f in ("Product", "Reactants", "Updated molecule set")]
    fields.append(("Reaction", (rp or []) + (rr or [])))
    for f, mols in fields:
        for smi in mols:
            if not isinstance(smi, str) or smi in seen:
                continue
            seen.add(smi)
            reason, sug = smiles_problem(smi)
            if reason is None:
                continue
            line = f"Step 1 '{f}': \"{smi}\" is not a valid SMILES: {reason.rstrip('.')}."
            if sug:
                line += f" Possibly intended: \"{sug}\"."
            out.append(line)
    reac = [x for x in (_as_list(step.get("Reactants")) or []) if isinstance(x, str)]
    if rr is not None and reac and not out:
        canon_set = lambda xs: {_valid(x) or x for x in xs}
        if canon_set(rr) != canon_set(reac):
            out.append(f"Step 1 'Reaction' has reactants {'.'.join(rr)} but 'Reactants' lists {reac}; "
                       f"they must be the same molecules.")
    return out


def build_repair_prompt(question, molecule, body, problems, closest=None):
    """The original expansion prompt + the rejected route + what to fix."""
    lines = [f"{i}. {p}" for i, p in enumerate(problems, 1)]
    txt = ("\n\nCHECKER FEEDBACK: your previous answer for the target molecule " + molecule +
           " was rejected by the route checker, so none of it was used. Problems found:\n" +
           "\n".join(lines) + "\n")
    if closest:
        txt += ("For reference, known USPTO reactions that do apply to " + molecule +
                " give (product>>reactants):\n" + "\n".join(f"- {c}" for c in closest) + "\n"
                "You may use one of these or correct your own disconnection.\n")
    if body is not None:
        txt += "Your previous route was:\n<ROUTE>\n" + body.strip()[:6000] + "\n</ROUTE>\n"
    txt += ("Fix every problem above and return the complete corrected route for " + molecule +
            " inside <ROUTE></ROUTE>, in the required format.")
    return question + txt
