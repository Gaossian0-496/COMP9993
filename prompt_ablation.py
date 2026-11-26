"""Prompt ablation transforms (COMP9991 ablation study, 2026-08-03).

Wang et al. (ICML 2025) Table 3 ablates the two SMALL explainability sections --
`<EXPLANATION>` (worth 11.5 pts on USPTO-Easy) and the per-step `'Rational'`
(2.5 pts) -- but never ablates `<PLAN>`, which is by far the largest: median
6,258 tokens, ~85% of output, ~60% of the total token bill. These transforms
complete that row and test whether the block's *length* is load-bearing or only
its presence.

Why a post-assembly transform rather than editing the templates: `<PLAN>` appears
in THREE files with separate copies of the instruction text --
`modification_prompt.py` (20), `run.py` (3, the initialization path) and
`optimizer.py` (2). Editing one path would half-apply the ablation and silently
invalidate the result. Rewriting the fully-assembled prompt catches every path,
and `_assert_absent` fails loudly if any phrasing slips through.

Config (top level, all default off so every existing arm is unchanged):

    prompt_ablation:
      no_plan: false          # remove <PLAN> entirely
      no_explanation: false   # remove <EXPLANATION> entirely
      plan_cap_words: 0       # keep <PLAN> but cap it (0 = uncapped)
"""

import re

# The <ROUTE> block and its five parsed keys are NEVER touched: run.py does
# ast.literal_eval on the raw block and reads 'Molecule set', 'Reaction',
# 'Product', 'Reactants', 'Updated molecule set' by name.
_PROTECTED = ("<ROUTE>", "Molecule set", "Reaction", "Product",
              "Reactants", "Updated molecule set")

# Worked-example lines, e.g. "<PLAN>: Analyze the target molecule ... </PLAN>"
_EX_LINE = "(?:^[ \t]*<{tag}>:.*?</{tag}>[ \t]*\r?\n?)"
# Numbered requirement referencing the tag, e.g. "7. In the <PLAN>, you should ..."
_RULE = r"(?:^[ \t]*\d+\.[^\n]*<{tag}>[^\n]*\r?\n?)"
# Instruction sentences that mention the tag, up to the sentence-ending period.
_SENTENCE = r"(?:[^.\n]*<{tag}>[^.]*\.[ \t]*)"


def _strip(text, tag):
    for pat in (_EX_LINE, _RULE, _SENTENCE):
        text = re.sub(pat.format(tag=tag), "", text, flags=re.M | re.S)
    # bare residual mentions of the tag pair
    text = text.replace("<%s></%s>" % (tag, tag), "")
    return text


def _assert_absent(text, tag, where):
    if "<%s>" % tag in text or "</%s>" % tag in text:
        raise RuntimeError(
            "prompt_ablation: <%s> survived stripping in %s -- the ablation would "
            "be only half-applied, which silently invalidates the arm. Add the "
            "surviving phrasing to prompt_ablation.py." % (tag, where))


_CAP = (
    "\n\nIMPORTANT -- keep the <PLAN> under {n} words. State only the "
    "disconnection strategy: which bond to break and what the precursors are. "
    "Do not restate or re-derive the target's structure, do not enumerate "
    "functional groups, and do not revise or correct yourself mid-plan. The "
    "<ROUTE> block is unchanged and must still follow the required format.\n"
)


def apply(question, cfg, where="modification"):
    """Rewrite an assembled prompt per the `prompt_ablation` config block.

    `question` may be a str or a list of content items (multimodal); returns the
    same shape. A no-op when the config block is absent or all-default.
    """
    if not cfg:
        return question
    no_plan = bool(cfg.get("no_plan", False))
    no_expl = bool(cfg.get("no_explanation", False))
    cap = int(cfg.get("plan_cap_words", 0) or 0)
    if not (no_plan or no_expl or cap):
        return question

    def one(text):
        if not isinstance(text, str):
            return text
        keep = [p for p in _PROTECTED if p in text]
        if no_plan:
            text = _strip(text, "PLAN")
            _assert_absent(text, "PLAN", where)
        if no_expl:
            text = _strip(text, "EXPLANATION")
            _assert_absent(text, "EXPLANATION", where)
        # Only append the cap to prompts that actually ask for a <PLAN>. query_LLM
        # is shared with the skill-rank / skill-update / guide-update calls, whose
        # prompts have no PLAN section -- appending there would be noise.
        if cap and not no_plan and "<PLAN>" in text:
            text = text + _CAP.format(n=cap)
        # the parsed schema must survive untouched
        for p in keep:
            if p not in text:
                raise RuntimeError(
                    "prompt_ablation: stripping removed protected token %r in %s "
                    "-- this would break route parsing." % (p, where))
        return text

    if isinstance(question, list):
        return [dict(it, text=one(it["text"])) if isinstance(it, dict) and "text" in it
                else it for it in question]
    return one(question)
