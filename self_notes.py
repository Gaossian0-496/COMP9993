"""Arm notes (try 12, 2026-10-26): the planner writes notes to ITSELF that persist for one target's search.

User request (2026-10-26): "Can we try a run where we give the model ability to create notes for itself" /
"that persists throughout the length of the whole molecule." Decisions taken the same day: ONE note per target
(shared by every expansion in that target's search, not keyed per intermediate), the four headings below, and
**no cap** on its length.

The gap this attacks: today every route call starts cold. The model gets the RAG examples, the tried-reactions
list (fix C) and the tree's facts, but never its own reasoning from an earlier call -- the <PLAN> block it writes
each time (median 6,715 tokens, measured on strat28_ctrl) is thrown away as soon as the route is parsed. Over a
250-call budget on one target it re-derives its approach from scratch every call.

Why this is not Run S again: skill memory was cross-molecule content injected from elsewhere, and the diagnosed
mechanism for its net-negative result was that injection steers the population away from productive free
exploration. Tries 7-11 were a second model's opinion, which failed because it was aimed at the wrong molecules
(advice at median depth 7-8, ~1.4% of it on the target). A self-note is the model's own words about the target in
front of it, and nothing in the history tests it.

Cost: the note is written as an extra block in the SAME completion as the route, so the arm spends **no extra
oracle calls** and stays matched against `strat28_ctrl` at 250. (This is what made arm joint hard to read: it was
matched on budget but used two models on 2 GPUs.)

Parsing follows the lesson from the joint arm (job 9573619, 2026-10-21): never assume where in a model's output
something appears, and strip the route block first so route content can never be read as a note. Headings are
matched line-anchored anywhere in the block -- that form parses at 93-100% on this stack (tries 9 and 10), where
the free-form stance word the joint arm relied on parsed at 7%.

NO CAP, but a guard: the note is in the prompt on EVERY route call, so a note that outgrows the context window
does not degrade the arm, it kills the call. There is no instructed word limit and nothing the model writes is
silently discarded; instead, when the note approaches the window the prompt asks it to shorten the note itself,
and only if it exceeds the hard limit is the shown copy truncated (and said to be truncated). Both events are
counted (`notes_shrink_asked`, `notes_truncated`) so the run reports whether the guard ever fired.

Pure helpers: no model calls here, so every path is testable without a server.
"""
import re

NOTE_FIELDS = ("STRATEGY", "DEAD ENDS", "WHAT WORKED", "NEXT")

_ROUTE_BLOCK = re.compile(r"(?s)<ROUTE>.*?(?:</ROUTE>|$)")
_NOTES_BLOCK = re.compile(r"(?s)<NOTES>(.*?)</NOTES>")
_HEADING = re.compile(r"(?im)^\W{0,4}(STRATEGY|DEAD\s*ENDS|WHAT\s*WORKED|NEXT)\s*:")

# chars -> tokens, as joint_path.discuss's default estimate
def estimate(text):
    return len(text or "") // 3


def _canon_field(name):
    return re.sub(r"\s+", " ", (name or "").strip().upper())


def parse_notes(text):
    """The <NOTES> block of a route answer -> {fields, text, usable}.

    usable is False when there is no complete block (no `</NOTES>` = the answer was cut at its cap, as
    advisor_plan.parse_plan treats finish "length") or when no heading parses. An unusable answer leaves the
    previous note standing: a bad write never destroys what the model already had.
    The route block is removed first, so a route that happens to contain the word NEXT is never read as a note."""
    stripped = _ROUTE_BLOCK.sub("", text or "")
    m = _NOTES_BLOCK.search(stripped)
    if not m:
        return {"fields": {}, "text": None, "usable": False}
    body = m.group(1)
    heads = [(h.start(), h.end(), _canon_field(h.group(1))) for h in _HEADING.finditer(body)]
    fields = {}
    for j, (_s, e, name) in enumerate(heads):
        end = heads[j + 1][0] if j + 1 < len(heads) else len(body)
        val = body[e:end].strip(" *_\n\t")
        if name not in fields and val:
            fields[name] = val
    if not fields:
        return {"fields": {}, "text": None, "usable": False}
    return {"fields": fields, "text": render_note(fields), "usable": True}


def render_note(fields):
    """The note as it is stored and shown back: the headings that have content, in a fixed order."""
    return "\n".join(f"{k}: {fields[k]}" for k in NOTE_FIELDS if fields.get(k))


def ask_block(max_words=0):
    """What the route prompt asks for. max_words 0 = no instructed limit (the user's decision, 26 Oct)."""
    limit = ("" if not max_words else
             f" Keep the whole block under {max_words} words.")
    return ("\nAFTER the route block, write your own notes for this target in a <NOTES> block. These notes are "
            "yours: nobody else reads them, and you will be shown them again on every later call you make while "
            "working on this target, so write what you would want to know if you came to this target fresh.\n"
            "<NOTES>\n"
            "STRATEGY: the overall approach you are taking to this target, and why.\n"
            "DEAD ENDS: what you have tried that did not work, and what went wrong.\n"
            "WHAT WORKED: steps that were accepted, and what made them work.\n"
            "NEXT: what you intend to try next.\n"
            "</NOTES>\n"
            "Rewrite the whole block each time, keeping what is still true and dropping what is not." + limit +
            " If nothing has changed, write the block out again unchanged. The notes never replace the route: "
            "always give the route block as well.\n")


def render_block(note, shrink=False, truncated=False):
    """The note as it appears in the next route prompt. Empty string when there is no note and no guard to
    report. When the note had to be cut the instruction to shorten it is emitted EVEN IF nothing could be
    shown (free room <= 0): dropping the note silently would leave the model no way to recover the space."""
    if not note and not truncated:
        return ""
    head = "\nYOUR NOTES ON THIS TARGET (written by you on an earlier call; only you see them):\n"
    tail = ""
    if truncated:
        tail = ("\n[Your notes have grown too long to show in full here" +
                (" and have been cut." if note else ", so they are not shown at all this time.") +
                " Rewrite them much shorter in your <NOTES> block this time, keeping only what still "
                "matters.]\n")
    elif shrink:
        tail = ("\n[These notes are getting long enough to crowd out the rest of this prompt. Rewrite them "
                "shorter in your <NOTES> block this time, keeping only what still matters.]\n")
    return (head + note + "\n" + tail) if note else tail


def guard(note, prompt_tokens, context, reserve, est=None):
    """How much of the note fits. -> (shown_note, shrink, truncated).

    `reserve` is everything that must stay free: the generation cap plus slack. No instructed cap means the
    model may write as much as it likes, so this is what stops a long note killing the route call instead of
    merely crowding it. shrink fires at half the free room, truncation only at the whole of it."""
    est = est or estimate
    if not note:
        return None, False, False
    free = int(context) - int(prompt_tokens) - int(reserve)
    if free <= 0:
        return None, False, True              # nothing fits: ask for a shorter note, show none
    n = est(note)
    if n <= free // 2:
        return note, False, False
    if n <= free:
        return note, True, False
    return note[:max(0, free * 3)], False, True
