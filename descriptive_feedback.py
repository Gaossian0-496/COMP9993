"""Step-localised oracle feedback (COMP9991, 2026-08-03).

Port of the TriFusion `_describe_oracle_failure` idea into the main pipeline, as a
single-variable arm on top of baseline A (no skill injection), so it compares
directly against the existing baseline results.

WHY. The paper's `get_feedback` is already molecule-specific -- it names the bad
reaction and the unpurchasable molecule. What it does NOT say is WHERE in the
route the failure is. `route.py:update_evaluation` walks the per-step evaluation
list and keeps only the FIRST invalid step's data, so by the time the prompt is
built the step index and the validated prefix have both been discarded. The model
is told "this reaction is not in USPTO" with no indication that steps 1-2 were
fine and only step 3 broke -- so it tends to rewrite the whole route rather than
repair the failing step.

WHY THIS ARM AND NOT A TRIFUSION RERUN. Run X v7 added this mechanism but wired it
only into the LLM3 fusion path, which fires rarely: measured over v7's logs, just
47 of 2005 prompts (2.3%) ever carried the descriptive text, while LLM1/LLM2 kept
a hardcoded generic string. v7's null result is therefore uninformative about the
mechanism -- the treatment was barely delivered. Here it goes on the modification
path, which is essentially every call.

Adds three things over the paper's feedback:
  1. step localisation      -- "Step 3 of your 5-step route failed"
  2. validated-prefix guard -- "Steps 1-2 validated; leave them unchanged"
  3. an actionable directive for the purchasability case, naming the molecules
     that still need disconnecting

Gated by top-level config `descriptive_feedback: true`; default off, so every
existing arm is byte-for-byte unchanged.
"""


def _ordinal(n):
    return "%d%s" % (n, {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th"))


def step_locator(full_evaluation):
    """'Step 3 of your 5-step route failed. Steps 1-2 validated...' or ''.

    `full_evaluation` is the per-step list of (step_id, is_valid, data) that
    route.py receives but discards down to the single failing entry.
    """
    if not full_evaluation:
        return ""
    total = len(full_evaluation)
    bad = None
    for i, entry in enumerate(full_evaluation):
        try:
            _, is_valid, _ = entry
        except Exception:
            continue
        if not is_valid:
            bad = i
            break
    if bad is None:
        return ""
    if total == 1:
        return "\nThis failure is in the only step of your route.\n"
    msg = ("\nThe failure is in the %s step of your %d-step route." % (_ordinal(bad + 1), total))
    if bad > 0:
        subj = ("The first %d steps of your route have" % bad) if bad > 1 \
            else "The first step of your route has"
        obj = "them" if bad > 1 else "it"
        msg += (" %s already validated successfully -- leave %s unchanged and "
                "repair only the failing step and anything after it." % (subj, obj))
    else:
        msg += " No earlier step validated, so the route needs rebuilding from the target."
    return msg + "\n"


def purchasability_directive(evaluation, inventory):
    """Actionable instruction for the 'all steps valid, leaves not buyable' case."""
    try:
        if not evaluation.get("check_availability"):
            return ""
        bad_ids = evaluation.get("unavailable_mol_id") or []
        upd = evaluation.get("updated_molecule_set") or []
        bad = [upd[i] for i in bad_ids if i < len(upd)]
        if not bad:
            return ""
        names = ", ".join("`%s`" % s for s in bad[:6])
        return ("\nEvery reaction in your route validated, so the chemistry is sound. "
                "What remains is that these molecules cannot be purchased: %s. "
                "Add one or more further steps that disconnect *these specific "
                "molecules* into commercially available fragments. Do not modify the "
                "steps that already validated.\n" % names)
    except Exception:
        return ""


def nearest_precedents(evaluation, ctx, k=3):
    """The PRIMARY lever. For the dominant failure -- 78% of baseline A feedback is
    'reaction not in USPTO' -- the paper's remedy is "make sure all the molecules in
    the reaction are in SMILES format", which is simply the wrong advice: the SMILES
    are fine, the TEMPLATE isn't in the 270,794-entry set. And the model cannot check
    membership, so it has nothing to act on. (Same root cause as the Run S azide
    diagnostic, where the model hallucinated reaction_key 99999999999 while believing
    it was using USPTO chemistry.)

    So instead of only saying which reaction is absent, show the nearest reactions
    that ARE present. Reuses `_nearest_reaction_templates` from agentic_tools -- the
    difference-fingerprint search already built for the tool arm.
    """
    try:
        if evaluation.get("reaction_existance") is not False:
            return ""
        reaction = evaluation.get("reaction")
        if not reaction or ctx is None:
            return ""
        from agentic_tools import _nearest_reaction_templates
        hits = _nearest_reaction_templates(reaction, ctx, k=k)
        if not hits:
            return ""
        lines = ["\nYour reaction is not in the USPTO set, and you cannot verify "
                 "membership yourself -- so here are the closest reactions that ARE "
                 "in it, for this transformation:"]
        for h in hits:
            lines.append("  - %s   (similarity %.2f)" % (h["template"], h["similarity"]))
        lines.append("Rewrite the failing step using one of these known "
                     "disconnections, or a different transformation you are "
                     "confident appears in USPTO. Do not simply re-word the same "
                     "reaction -- it will fail again.\n")
        return "\n".join(lines)
    except Exception:
        return ""


def enhance(feedback, evaluation, full_evaluation, inventory, ctx=None):
    """Append nearest-precedent list + step localisation + purchasability directive."""
    try:
        extra = (nearest_precedents(evaluation, ctx)
                 + step_locator(full_evaluation)
                 + purchasability_directive(evaluation, inventory))
    except Exception:
        return feedback          # never let the enhancement break the prompt
    return feedback + extra if extra else feedback
