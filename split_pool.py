"""Run the split search in separate PROCESSES so it escapes the GIL (2026-11-11, config `aot.split_pool`).

WHY. Job 9717221 timed the split search (`disconnections.search` + the route checker) 1 thread vs 28 threads in
ONE process, on the same molecules:

    1 thread :  3 items,   319.8 s wall, 106.6 s per item
    28 threads: 84 items, 11320.9 s wall, 3702.7 s per item   -> 34.7x the latency
    throughput: 134.8 s/item at 28 threads vs 106.6 s/item at 1 -> no parallel gain AT ALL

The work is `rdkit` / `rdchiral` / pure-Python template matching, so it holds the GIL and our 28 worker threads
run it one at a time (the job used 1.5 of 4 CPUs, so the limit is the lock, not the CPU request). That is the
standing suspect for `cutsel` being the only strat28 arm to hit the 24 h per-molecule cap (3 of 28, at 15 / 67 /
83 of 250 calls -- see docs/experiment_history.md, 8 Nov).

HOW. A fixed set of fork()ed child processes, each with its own `Pipe` to the parent:

  * forked ONCE from the main thread before any worker thread starts (`start()` refuses otherwise), so no lock
    can be held by another thread at fork time -- the classic fork-with-threads deadlock;
  * the children inherit the ~25 GB of reaction data through copy-on-write, so there is no second copy and the
    optimizer is never pickled (it could not be);
  * a calling thread takes an idle child, sends one molecule's work, and waits in `conn.poll()`, which is a
    syscall that RELEASES the GIL -- so the other 27 threads keep running while the CPU work happens elsewhere;
  * nothing is ever re-forked. A child that dies is dropped, not replaced (a replacement would have to fork
    from the parent while it IS multi-threaded). With every child gone every call MISSes at once, without
    waiting `wait_s`.

RESULTS. A pooled verdict equals the in-process one only because the pooled configs set `prefer_model_reactants`
and leave `aot.switch_choice` off: the checker ranks a template's outcomes with `rank_reactants`, which reads
`visited_molecules` / `dead_molecules`, and a child holds only the fork-time copy of those. With
`prefer_model_reactants` the model's own outcome is moved to the front whenever the template produces it, so the
ranking can no longer decide "accepted as written". `aot_search.split_pool_settings` refuses any other combination.

SAFETY. Off by default. `MISS` comes back on every failure path -- pool not started, no idle child, broken
pipe, child exception, timeout, unpicklable result -- and the caller then does the work in-process exactly as
before. So, under the condition above, the pool can change *when* a result arrives, never *what* it is, and it
can never cost an expansion its LLM call. The child only reads shared data and returns plain dicts; any state it touches dies with it
(`advisor_live.dry_accepts` already restores the three optimizer fields it uses, which is what makes a verdict
computed in a child equal to one computed here).
"""
import atexit
import gc
import multiprocessing
import os
import queue
import signal
import threading
import time

MISS = object()                 # nothing came back: the caller must do the work in-process

_IDLE = queue.Queue()           # child handles that are free (a queue = backpressure without a spin loop)
_CHILDREN = []                  # every child ever forked, for stop()
_OPT = None                     # the optimizer the children inherit; set before the fork, read in the child
_TIMEOUT = 900.0                # seconds to wait for one molecule's answer
_WAIT = 0.0                     # seconds to wait for an idle child before giving up (0 = don't)
_STARTED = False
_LOCK = threading.Lock()
STATS = {}                      # ok / miss_* / error counts, printed by the caller's fix-effect counters


def _bump(k, n=1):
    with _LOCK:
        STATS[k] = STATS.get(k, 0) + n


# --------------------------------------------------------------------------- the child side
def _child_loop(conn):
    """One child: answer requests until the parent closes the pipe. Never exits on a bad request."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)        # the parent owns Ctrl-C / PBS signals
    gc.enable()                                         # new objects here are ours; the inherited ones are frozen
    import disconnections as dc
    from utils import sanitize_smiles
    while True:
        try:
            req = conn.recv()
        except (EOFError, OSError, KeyboardInterrupt):
            return
        if req is None:
            return
        try:
            kind, args = req
            if kind == "best_steps":
                out = dc.prefetch(_OPT, args[0], sanitize_smiles, args[1], args[2], args[3], args[4])
            elif kind == "template_split_exists":
                st = {}
                out = {"verdict": dc.template_split_exists(_OPT, args[0], sanitize_smiles, st)}
            else:
                out = ("__error__", f"unknown request {kind!r}")
            conn.send(out)
        except BaseException as e:                       # never let one molecule kill the child
            try:
                conn.send(("__error__", f"{type(e).__name__}: {e}"[:300]))
            except Exception:
                return


class _Child:
    __slots__ = ("proc", "conn", "n")

    def __init__(self, proc, conn):
        self.proc, self.conn, self.n = proc, conn, 0

    def alive(self):
        return self.proc.is_alive()

    def kill(self):
        for f in (self.conn.close, self.proc.terminate):
            try:
                f()
            except Exception:
                pass


# --------------------------------------------------------------------------- the parent side
def start(opt, processes=4, timeout_s=900, wait_s=0.0):
    """Fork `processes` children that share `opt`'s data. MUST be called from the main thread before any worker
    thread starts; returns False (and leaves the pool off) if that is not the case or the fork fails."""
    global _OPT, _TIMEOUT, _WAIT, _STARTED
    with _LOCK:
        if _STARTED:
            return bool(_CHILDREN)
        if threading.active_count() > 1:
            names = sorted(t.name for t in threading.enumerate() if t is not threading.current_thread())
            print(f"[split_pool] refusing to fork: {threading.active_count()} threads are already running "
                  f"(other threads: {names}; a lock held by another thread at fork time deadlocks the child)")
            return False
        if int(processes) < 1:
            return False
        _OPT, _TIMEOUT, _WAIT, _STARTED = opt, float(timeout_s), float(wait_s), True
        try:
            ctx = multiprocessing.get_context("fork")
        except Exception as e:
            print(f"[split_pool] no fork start method: {e}")
            return False
        try:
            gc.collect()
            gc.freeze()          # the inherited data is never collected, so a child's GC cannot write to its pages
        except Exception:
            pass
        for i in range(int(processes)):
            try:
                mine, theirs = ctx.Pipe(duplex=True)
                p = ctx.Process(target=_child_loop, args=(theirs,), name=f"split{i}", daemon=True)
                p.start()
                theirs.close()   # the parent keeps only its own end, so a dead child gives EOF instead of hanging
                c = _Child(p, mine)
                _CHILDREN.append(c)
                _IDLE.put(c)
            except Exception as e:
                print(f"[split_pool] fork {i} failed: {e}")
                break
        print(f"[split_pool] {len(_CHILDREN)} worker processes, timeout {_TIMEOUT:.0f}s "
              f"(pid {os.getpid()}); the split search now runs outside the GIL")
        atexit.register(stop)
        return bool(_CHILDREN)


def enabled():
    """True while at least one child is still alive in the pool (dead ones are removed as they are found)."""
    return bool(_CHILDREN)


def _drop(c, why):
    """A child that failed is killed AND removed, so `enabled()` and the waiters see the pool shrink. Without the
    removal a pool whose children had all died (OOM kill, timeouts) still looked enabled, and every caller waited
    the full `wait_s` for an idle child that could never come (review, 12 Nov)."""
    _bump(why)
    c.kill()
    with _LOCK:
        if c in _CHILDREN:
            _CHILDREN.remove(c)
        left = len(_CHILDREN)
    if left == 0:
        print("[split_pool] every worker process is gone; the split search now runs in-process (the slow path)")


def _take():
    """An idle child, or None. With `wait_s` > 0 the wait is sliced so a caller stops waiting as soon as the pool
    has no live child left, instead of sitting out the whole `wait_s`."""
    if _WAIT <= 0:
        try:
            return _IDLE.get_nowait()
        except queue.Empty:
            return None
    deadline = time.time() + _WAIT
    while True:
        left = deadline - time.time()
        if left <= 0 or not _CHILDREN:
            return None
        try:
            return _IDLE.get(timeout=min(1.0, left))
        except queue.Empty:
            continue


def stop():
    """Ask every child to exit, then make sure it has."""
    with _LOCK:
        kids, _CHILDREN[:] = list(_CHILDREN), []
    for c in kids:
        try:
            c.conn.send(None)
        except Exception:
            pass
    deadline = time.time() + 5
    for c in kids:
        try:
            c.proc.join(max(0.1, deadline - time.time()))
        except Exception:
            pass
        if c.alive():
            c.kill()
    while True:                                  # drop the idle handles so a late call MISSes
        try:
            _IDLE.get_nowait()
        except queue.Empty:
            break


def _ask(kind, args):
    """One request on one child. -> the child's payload, or MISS (the caller then computes in-process)."""
    if not _CHILDREN:
        return MISS
    c = _take()
    if c is None:
        # every child occupied (do it here rather than queue behind them), or none left alive at all
        _bump("miss_busy" if _CHILDREN else "miss_none")
        return MISS
    if not c.alive():
        _drop(c, "miss_dead")
        return MISS
    try:
        c.conn.send((kind, args))
        if not c.conn.poll(_TIMEOUT):            # RELEASES THE GIL: this is the whole point of the pool
            print(f"[split_pool] {kind} timed out after {_TIMEOUT:.0f}s; dropping that worker")
            _drop(c, "miss_timeout")
            return MISS
        out = c.conn.recv()
    except BaseException as e:
        print(f"[split_pool] {kind} lost its worker ({type(e).__name__}: {e}); dropping it")
        _drop(c, "miss_broken")
        return MISS
    c.n += 1
    _IDLE.put(c)                                 # only a healthy child goes back in
    if isinstance(out, tuple) and len(out) == 2 and out[0] == "__error__":
        _bump("miss_error")
        print(f"[split_pool] {kind} failed in the worker: {out[1]}")
        return MISS
    _bump("ok")
    return out


def best_steps(smiles, max_piece, show, check_max, buyable_max=0):
    """`disconnections.best_steps` in a child, plus the caches it filled (see `disconnections.prefetch`).
    -> {"steps", "checked", "n", "cands", "verdicts"} or MISS."""
    out = _ask("best_steps", (smiles, float(max_piece), int(show), int(check_max), int(buyable_max)))
    if out is MISS:
        return MISS
    if not (isinstance(out, dict) and {"steps", "checked", "n", "cands", "verdicts"} <= set(out)):
        _bump("miss_shape")
        return MISS
    return out


def template_split_exists(smiles):
    """`disconnections.template_split_exists` in a child. -> True / False / None, or MISS."""
    out = _ask("template_split_exists", (smiles,))
    if out is MISS:
        return MISS
    if not isinstance(out, dict) or "verdict" not in out:
        _bump("miss_shape")
        return MISS
    v = out["verdict"]
    return v if v is None or isinstance(v, bool) else MISS
