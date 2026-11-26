"""Arm reopen (2026-11-18): going back to the target for a NEW opening.

THE GAP THIS ANSWERS, measured on hard100_ctrl_v2 and hard100_cutselR (docs/experiment_history.md,
2026-11-18). The AOT* search seeds the root with `num_initial_routes` independent route calls -- in
practice 3 calls yielding 3-5 distinct first moves -- and after that the target molecule is just another
node competing on selection score, which it loses essentially forever. On the targets that ran the budget
out, EVERY root expansion happened inside the first 3-7 of 250 calls and the opening was then never
revisited: median 243 further calls spent digging under openings that had already failed. Two targets of
~20 were exceptions, so the search can come back; it almost never does.

WHY IT IS WORTH A SHOT WHERE THE FRONTIER ADVISORS WERE NOT. Arms `ask` and `ask2` re-ranked molecules the
search had already found and changed nothing (0W/0L against cutselR, twice). This is a decision the search
STRUCTURALLY cannot make, not one it was already making adequately.

WHY THE RISK IS SMALL. Solves are early: on cutselR the median solve takes 6 LLM calls, p90 is 31, and only
4 of 96 needed more than 100. A target still unsolved at call 100 is unlikely to come good unaided. And a
reopen here ADDS a branch -- the tree is kept and the new opening competes alongside the old ones -- so a
late solve in progress is not thrown away. The cost is the calls the new opening consumes.

TWO VARIANTS, so that "does reopening help at all" is separable from "does a model pick the moment":
  * `decide: counter` -- reopen as soon as the threshold is reached. No LLM call, no judgement.
  * `decide: llm`     -- at the same threshold, ask the model whether to reopen; it may decline, and is
                        asked again at the next threshold. The advisor call is NOT charged to the budget
                        (as with every other advisor in this codebase); the root expansion it triggers is.
The counter arm is the control for the llm arm: same trigger, same action, the only difference being who
decides. If the counter does nothing, no judgement about WHEN was ever going to rescue it.
"""
import json

MOVES = ("REOPEN", "STAY")


def question(target, calls, budget, reopens, openings, tried, deepest, unsolved_open):
    """The prompt for `decide: llm`. Every number in it is something the search measured.

    openings: the first-step reactions already attached to the target, as "target>>a.b" strings.
    tried:    [{molecule, outcome}] for the last few expansions anywhere in the tree.
    deepest:  the depth the search has reached.
    unsolved_open: how many open molecules are still waiting for work.
    """
    left = max(int(budget) - int(calls), 0)
    L = [f"You are advising on a retrosynthesis of: {target}",
         "",
         "HOW THIS SEARCH WORKS. It began by asking for a few different opening moves -- different ways to "
         "break this target apart -- and has since been working downwards from them, one accepted "
         "disconnection at a time. It is finished when every loose end is a molecule that can be bought.",
         "",
         f"BUDGET: {calls} of {budget} calls used, {left} left. The target is NOT yet solved.",
         f"For scale: a target that gets solved usually takes about 6 calls, and 9 in 10 are done inside 31.",
         "",
         f"THE OPENINGS ALREADY TRIED ({len(openings)}):"]
    L += [f"  - {o}" for o in openings] or ["  (none recorded)"]
    L += ["",
          f"WHAT THE SEARCH HAS DONE WITH THEM: it has reached depth {deepest}, and {unsolved_open} "
          f"molecules are still open and unsolved. None of these openings has produced a complete route."]
    if tried:
        L += ["", "THE LAST FEW EXPANSIONS (oldest first):"]
        L += [f"  {t['molecule']}  ->  {t['outcome']}" for t in tried]
    L += ["",
          "THE DECISION. You may send the search back to the target for a NEW opening -- a different first "
          "move, avoiding the ones listed above. Nothing is thrown away: the existing branches stay and "
          "keep competing. The cost is the calls the new opening consumes, out of the "
          f"{left} remaining.",
          "",
          "Say REOPEN if the openings above look like the wrong way into this molecule, so that more work "
          "underneath them is unlikely to pay off. Say STAY if one of them is a sound strategy that simply "
          "has not been finished yet, or if too little budget is left for a new direction to get anywhere.",
          "",
          "Answer in exactly this form and nothing else:",
          "  MOVE: REOPEN | STAY",
          "  WHY: <one sentence>"]
    if reopens:
        L += ["", f"NOTE: you have already reopened this target {reopens} time(s)."]
    return "\n".join(L)


def parse(answer):
    """-> (move, why, status). An unreadable answer is STAY: the default must be to leave the search alone."""
    move, why = None, ""
    for raw in str(answer or "").splitlines():
        s = raw.strip().lstrip("*#- ").strip()
        up = s.upper()
        if up.startswith("WHY:"):
            why = s.split(":", 1)[1].strip()
        elif up.startswith("MOVE:"):
            v = s.split(":", 1)[1].strip().strip("*`() ").upper()
            if v.startswith("REOPEN"):
                move = "REOPEN"
            elif v.startswith("STAY"):
                move = "STAY"
    if move is None:
        return "STAY", why, "no_move"
    return move, why, "ok"


def result(move, why):
    return json.dumps({"reopen": move, "reason": why})


def choose_question(target, target_atoms, calls, budget, candidates, skills=""):
    """Arm reopenP (2026-11-19, user: "ask for 3 more different molecules to start with and ask the
    advisor to choose the best one using some sort of skill heuristic").

    The reopen already asks for fresh openings; this asks the advisor WHICH of them to pursue. Each
    candidate is described by the facts the situational skills are defined over -- the share of the
    target's mass its largest piece still carries, how many pieces it makes, how many of them can be
    bought, its accepted-split count and its stereocentre load -- so the heuristic in `skills` and the
    rows below are in the same units and can actually be compared.

    candidates: [{tag, pieces, largest_share, n_pieces, buyable, splits, stereo}]
    """
    left = max(int(budget) - int(calls), 0)
    L = [f"You are choosing the opening move for a retrosynthesis of: {target}"
         + (f"  ({target_atoms} heavy atoms)" if target_atoms else ""),
         "",
         "The earlier openings for this target did not lead to a route, so fresh ones were generated. "
         "Each row below is ONE candidate first move: the target broken into these pieces. The search "
         "will carry on from whichever you pick, so pick the one whose pieces look most likely to reach "
         "molecules that can simply be bought.",
         "",
         f"BUDGET: {calls} of {budget} used, {left} left."]
    if skills:
        L += ["", skills]
    L += ["", "THE CANDIDATE OPENINGS:"]
    for c in candidates:
        L.append(f"  ({c['tag']}) {' + '.join(c.get('pieces') or [])}")
        bits = []
        if c.get("largest_share") is not None:
            bits.append(f"largest piece keeps {round(100 * float(c['largest_share']))}% of the target")
        if c.get("n_pieces"):
            bits.append(f"{c['n_pieces']} pieces")
        if c.get("buyable"):
            bits.append(f"{c['buyable']} already purchasable")
        bits.append("splits not checked" if c.get("splits") is None
                    else f"{c['splits']} accepted splits on the largest piece")
        if c.get("stereo") is not None:
            bits.append(f"{c['stereo']} stereocentres in the largest piece")
        L.append("        " + ", ".join(bits))
    L += ["",
          "A smaller largest piece means the opening has genuinely taken the target apart; a largest "
          "piece near 100% means the move achieved almost nothing, whatever else it did. Pieces that can "
          "already be bought are finished work.",
          "",
          "Answer in exactly this form and nothing else:",
          "  PICK: <tag>",
          "  WHY: <one sentence>"]
    return "\n".join(L)


def parse_choice(answer, valid_tags=()):
    """-> (tag, why, status). None = no usable pick, and the caller then leaves the search's own order
    alone rather than guessing one."""
    tag, why = None, ""
    for raw in str(answer or "").splitlines():
        s = raw.strip().lstrip("*#- ").strip()
        up = s.upper()
        if up.startswith("WHY:"):
            why = s.split(":", 1)[1].strip()
        elif up.startswith("PICK:"):
            v = s.split(":", 1)[1].strip().strip("*`() ")
            tag = v.split()[0] if v.split() else None
    if not tag:
        return None, why, "no_pick"
    if valid_tags and tag not in valid_tags:
        return None, why, "bad_tag"
    return tag, why, "ok"
