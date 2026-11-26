"""Arm joint (try 11, 2026-10-20): Qwen and Intern design a route together, as peers, from the SAME prompt.

Tries 7-10 were all ASYMMETRIC: Intern saw a curated item ({target, node, path, attempts}) plus a facts list and
ADVISED Qwen, which then accepted or refused. The measured result (docs/experiment_history.md, 2026-10-20) is that
Qwen fully complied with 3 of 949 course-changes, and 131 of 135 FOLLOW tags were on advice that had already
conceded to Qwen's own plan. So the asymmetry itself is the thing left to test.

Here (user, 2026-10-20: "I don't want to choose either or from qwen and intern, I want them to jointly discuss a
new path which can be different to either proposed"; "give intern EXACTLY what qwen has in terms of context and
info"):

  1. BOTH models get the byte-identical prompt -- the route question `_llm_route` built for Qwen (RAG examples,
     format requirements, task description, repair hint, tried-reactions note), minus the joint block itself.
     Intern has never seen the RAG examples or the tried list before.
  2. BOTH propose a route independently. Neither sees the other's first.
  3. BOTH are shown BOTH proposals, UNLABELLED by author (so neither can defer to "the advisor" or dismiss "the
     other model"), and are told in terms that the agreed path may be a third one that is neither of theirs.
  4. Up to `rounds` further turns each, stopping as soon as both say AGREED.
  5. Qwen writes the final route in the required format (Intern failed the format preflight 12/24 vs Qwen 23/24,
     so letting Intern emit would lose answers to format rather than chemistry).

What is FORWARDED between them is the route block plus a short rationale, never the raw answer: measured on
strat28_ctrl, the <ROUTE> block is 7% of a median answer (569 tokens) and the private <PLAN> block is 6,715.
Forwarding raw answers would overflow Intern's 49,152-token context in 2 rounds; forwarding routes leaves room for
many. Each model still THINKS at its full cap -- we cap what crosses, not what is produced.

Rounds are deliberately generous and then discouraged (user: "give it max rounds but discourage it"): every turn
must open with AGREED or CONTINUE, and CONTINUE is only legitimate for a specific chemical change. The discussion
also stops when the next turn would not fit the advisor's context (status "context_full"), rather than letting
ask_advisor_fn shrink a late turn into an unusable cut.

Pure helpers: the model calls are passed in as functions, so every path is testable without a server.
"""
import re

# Advisor arms are not part of this distribution; these imports are allowed to fail.
try:
    import advisor_plan as ap
    ADVISORS_AVAILABLE = True
except ImportError:
    ap = None
    ADVISORS_AVAILABLE = False


# A turn that crosses between the models: the route plus a short reason. The <PLAN> block never crosses.
RATIONALE_WORDS = 150
# The stance is read as a word ON A LINE OF ITS OWN, anywhere in the turn, NOT from the first 200 characters.
# Measured on the aborted first run (job 9573619, 2026-10-21): Intern-S2 reasons before it answers, so its stance
# word sat at character 1,263 / 1,395 / 2,772 and 220 of its 237 turns parsed as no stance -- mutual AGREED could
# never fire and 8 of 9 molecules ended "context_full". A bare \b(AGREED|CONTINUE)\b anywhere is NOT safe: the
# models restate the instruction ("I need to output \"AGREED\" or \"CONTINUE\" at the very beginning"), which the
# old window read as a stance 7 times. The route block and the WHY tail are removed first for the same reason
# ("aligns with the explicitly agreed Plan of Attack"). Last match wins: it is the conclusion, not the preamble.
STANCE_RE = re.compile(r"(?im)^\W{0,4}(AGREED|CONTINUE)\b")
_ROUTE_BLOCK = re.compile(r"(?s)<ROUTE>.*?(?:</ROUTE>|$)")
_WHY_TAIL = re.compile(r"(?ims)^\W{0,4}WHY\s*:.*\Z")


def read_stance(text):
    """AGREED / CONTINUE / None from a turn, ignoring the route block and the WHY rationale."""
    found = STANCE_RE.findall(_WHY_TAIL.sub("", _ROUTE_BLOCK.sub("", text or "")))
    return found[-1].upper() if found else None

PEER = ("You and another model of comparable ability are designing one retrosynthesis route together. You are "
        "peers: neither of you is in charge, and neither proposal has priority. The route you settle on does NOT "
        "have to be either of the two first proposals -- if a third route is better, propose it.")

OPEN_ASK = ("Propose the route you think is best, in the required format (a <ROUTE> block). Then, after the route, "
            f"add one line 'WHY: ' and at most {RATIONALE_WORDS} words saying why this route rather than the "
            "obvious alternative.")


def _rationale(text):
    """The WHY line of an answer (what crosses to the other model), without the private planning prose."""
    if not text:
        return None
    m = re.search(r"WHY\s*:?\s*(.+?)(?:\n\s*\n|<ROUTE>|$)", text, re.S | re.I)
    if not m:
        return None
    words = " ".join(m.group(1).split())
    return words[:1200] or None


def parse_turn(text, finish=None, extract=None):
    """A model's turn -> {route, why, stance, usable}. `extract` = run.extract_route_block (injected).

    usable needs a route block; a turn cut at its cap is never usable (as advisor_plan.parse_plan).
    stance is AGREED / CONTINUE from read_stance (a line of its own, anywhere), else None -- a model that
    reasons before it answers still gets its stance read, and an unreadable stance only costs a round."""
    if finish == "length" or not text:
        return {"route": None, "why": None, "stance": None, "usable": False}
    route = extract(text) if extract else None
    return {"route": route, "why": _rationale(text), "stance": read_stance(text),
            "usable": bool(route)}


def proposals_block(routes, whys):
    """Both proposals, UNLABELLED and in a fixed given order (the caller shuffles), so neither model can tell
    which is its own by position alone. Identical proposals are shown once."""
    seen, items = set(), []
    for r, w in zip(routes, whys):
        key = " ".join((r or "").split())
        if not r or key in seen:
            continue
        seen.add(key)
        items.append((r, w))
    lines = ["", f"THE {'TWO' if len(items) > 1 else 'ONE'} PROPOSALS ON THE TABLE (unlabelled; one may be your "
                 "own):" if items else ""]
    for n, (r, w) in enumerate(items, 1):
        lines.append(f"\nProposal {n}:\n<ROUTE>\n{r}\n</ROUTE>")
        if w:
            lines.append(f"Its reason: {w}")
    return "\n".join(lines) + "\n", len(items)


def converge_ask(rounds_left):
    """The instruction for every turn after the proposals. Max rounds, actively discouraged."""
    return ("\nDecide, with the other model, which single route to use. It may be either proposal, a modification "
            "of one, or a different route altogether -- say so plainly if neither is right.\n"
            "Put ONE word on a line by itself -- AGREED or CONTINUE -- before your <ROUTE> block. If you reason "
            "first, that is fine: put the word on its own line after the reasoning and before the route. Do not "
            "write either word anywhere else.\n"
            "  AGREED  -- you accept the route now on the table; give it in a <ROUTE> block so there is no doubt "
            "which one you mean.\n"
            "  CONTINUE -- only if there is a SPECIFIC chemical change you want: name the step and what is wrong "
            "with it, and give your revised route in a <ROUTE> block.\n"
            f"Agree as soon as the differences left would not change which reactions get tried. You have at most "
            f"{rounds_left} more turn(s), but using them all is a poor outcome: converge early.\n"
            f"After the route add one line 'WHY: ' and at most {RATIONALE_WORDS} words.\n")


def open_messages(question, system):
    """Turn 1, identical for both models (the caller passes the same `question` to each)."""
    return [{"role": "system", "content": system},
            {"role": "user", "content": question + "\n" + PEER + "\n" + OPEN_ASK + "\n"}]


def fits(msgs, add_chars, context, estimate, reserve):
    """Would a turn still leave `reserve` tokens to generate in the advisor's context?"""
    return estimate(msgs + [{"role": "user", "content": "x" * max(0, int(add_chars))}]) + reserve <= int(context)


def discuss(question, ask_qwen, ask_advisor, extract, rounds=10, system="", shuffle=None,
            context=49152, estimate=None, reserve=6144, keep_going=None):
    """Both propose from `question`, then converge. -> (result, trace).

    result = {route, status, rounds, differs_from_both, proposals, usable}
      status: "agreed" (both said AGREED) | "standoff" (rounds ran out) | "context_full" (no room for another
              turn) | "out_of_time" (keep_going() went false: the molecule's time cap) | "one_sided" (only one
              model ever gave a usable route; that route stands) | "no_proposal"
      route:  the agreed route block, or the last usable one; None if neither model produced one.
      differs_from_both: the agreed route is neither model's FIRST proposal (the point of the arm).

    ask_qwen(messages) / ask_advisor(messages) -> {text, finish, tokens, latency_s, error}. Never raises:
    a model that fails drops out and the other's route stands."""
    trace = []
    est = estimate or (lambda m: sum(len(x["content"]) for x in m) // 3)
    qmsgs, amsgs = open_messages(question, system), open_messages(question, system)

    turns = {}
    for who, ask, msgs in (("qwen", ask_qwen, qmsgs), ("advisor", ask_advisor, amsgs)):
        r = ask(msgs)
        t = parse_turn(r["text"], r["finish"], extract)
        trace.append({"stage": f"joint_open_{who}", "prompt": msgs[-1]["content"], "r": r,
                      "parsed": {"usable": t["usable"], "has_why": bool(t["why"]), "round": 0}})
        turns[who] = t

    first = {w: turns[w]["route"] for w in turns}
    usable = [w for w in ("qwen", "advisor") if turns[w]["usable"]]
    if not usable:
        return {"route": None, "status": "no_proposal", "rounds": 0, "differs_from_both": False,
                "proposals": 0, "usable": False}, trace
    if len(usable) == 1:
        w = usable[0]
        return {"route": turns[w]["route"], "status": "one_sided", "rounds": 0, "differs_from_both": False,
                "proposals": 1, "usable": True}, trace

    order = shuffle([("qwen", turns["qwen"]), ("advisor", turns["advisor"])]) if shuffle else \
        [("qwen", turns["qwen"]), ("advisor", turns["advisor"])]
    block, n_shown = proposals_block([t["route"] for _, t in order], [t["why"] for _, t in order])
    for msgs in (qmsgs, amsgs):
        msgs.append({"role": "assistant", "content": "(proposal submitted)"})

    current, status, k = turns["qwen"]["route"], "standoff", 0
    for k in range(1, int(rounds) + 1):
        ask_text = block + converge_ask(int(rounds) - k)
        if keep_going is not None and not keep_going():
            status = "out_of_time"     # the molecule's wall-clock cap: stop between rounds, keep what we have
            k -= 1
            break
        if not fits(amsgs, len(ask_text), context, est, reserve):
            status = "context_full"
            k -= 1
            break
        said = {}
        for who, ask, msgs in (("qwen", ask_qwen, qmsgs), ("advisor", ask_advisor, amsgs)):
            msgs.append({"role": "user", "content": ask_text})
            r = ask(msgs)
            t = parse_turn(r["text"], r["finish"], extract)
            trace.append({"stage": f"joint_turn_{who}", "prompt": ask_text, "r": r,
                          "parsed": {"usable": t["usable"], "stance": t["stance"], "round": k}})
            if t["usable"]:
                current = t["route"]
            msgs.append({"role": "assistant", "content": (t["route"] or "")[:4000]})
            said[who] = t
        if all(said[w]["stance"] == "AGREED" for w in said):
            status = "agreed"
            break
        block, n_shown = proposals_block([said["qwen"]["route"] or first["qwen"],
                                          said["advisor"]["route"] or first["advisor"]],
                                         [said["qwen"]["why"], said["advisor"]["why"]])

    norm = lambda s: " ".join((s or "").split())
    differs = bool(current) and norm(current) not in {norm(first["qwen"]), norm(first["advisor"])}
    return {"route": current, "status": status, "rounds": k, "differs_from_both": differs,
            "proposals": n_shown, "usable": bool(current)}, trace


def final_ask(route):
    """What Qwen is asked to emit after the discussion: the agreed route, in the required format."""
    return ("\nYou and the other model agreed on this route:\n"
            f"<ROUTE>\n{route}\n</ROUTE>\n"
            "Write it out now as your answer in the required format. Keep the agreed chemistry: change a step only "
            "if it cannot be written validly, and say so in your explanation if you do.\n")
