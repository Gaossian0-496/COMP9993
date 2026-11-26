"""AND-OR tree for the AOT* search mode (config `search: aot`).

Port of the released AOT* implementation (github.com/ShawnKS/AOTstar, MIT;
released AOT* code, aotcore/tree_nodes.py
and aotcore/llm_tree_optimizer.py). Pure data structure + tree algorithms, no LLM,
no chemistry validation -- those live in aot_search.py.

  OR node  = a molecule. Solved if purchasable, or if ANY child reaction is solved.
  AND node = a reaction. Solved only if ALL of its reactant OR nodes are solved.

Molecules are identified by canonical SMILES, so two spellings of one molecule are
one node. Deliberate difference from the release: an identical reaction (same
product, same reactant set) proposed twice is merged into one AND node instead of
creating a duplicate.
"""
import math


class ORNode:
    __slots__ = ("smiles", "is_solved", "purchasable", "child_reactions",
                 "parent_reactions", "solving_and_node")

    def __init__(self, smiles, purchasable):
        self.smiles = smiles
        self.purchasable = purchasable
        self.is_solved = purchasable
        self.child_reactions = []
        self.parent_reactions = []
        self.solving_and_node = None


class ANDNode:
    __slots__ = ("key", "product", "reactants", "parent_molecule", "child_molecules",
                 "visit_count", "total_value", "average_value", "feasibility_score",
                 "depth", "reaction")

    def __init__(self, key, parent_molecule, depth, reaction=None):
        self.key = key
        self.product = parent_molecule.smiles
        self.parent_molecule = parent_molecule
        self.child_molecules = []
        self.reactants = []
        self.visit_count = 0
        self.total_value = 0.0
        self.average_value = 0.0
        self.feasibility_score = 0.0
        self.depth = depth
        self.reaction = reaction

    def is_fully_solved(self):
        return all(c.is_solved for c in self.child_molecules)


class AndOrTree:
    """Tree state for one molecule. `canon(smi)` -> canonical SMILES or None;
    `purchasable(canon_smi)` -> bool. Both injected so the tree is testable
    without RDKit / the 23M inventory."""

    def __init__(self, target, canon, purchasable, c_param=0.5, unsolved_penalty=0.3,
                 max_depth=20, max_reexpansions=5, backprop="average", depth_penalty=0.0,
                 propagate_failure=False, propagate_from="proven"):
        self.canon = canon
        self.purchasable = purchasable
        # arm reopenG (2026-11-20): a reopened opening's guaranteed selections. Empty/0 = released select().
        self.protect_keys = set()
        self.protect_left = 0
        self.c_param = c_param
        self.unsolved_penalty = unsolved_penalty
        self.depth_penalty = depth_penalty
        self.max_depth = max_depth
        self.max_reexpansions = max_reexpansions
        # arm advprune (2026-11-01): AND-node key -> divisor on its selection score. A mark NEVER
        # removes a candidate (16 Oct: acting on "go back" would have cut the found route on 7-11 of
        # 18 solves), so a marked branch is tried later and still taken when nothing else is left.
        self.deprioritised = {}
        self.backprop = backprop
        self.molecules = {}          # canonical smiles -> ORNode
        self.reactions = {}          # key -> ANDNode
        self.leaf_and_nodes = set()  # frontier (selectable AND nodes)
        self.dead = set()            # molecules whose expansion repeatedly failed
        # termination fix (2026-11-08, off by default): molecules NO checker template decomposes (also put in
        # `dead`), and whether failure is passed up the tree (failure_state / closed_by_failure)
        self.proven_dead = set()
        self.judged_failed = set()   # arm steer REMOVE (2026-11-08): reaction keys an advisor closed for good
        self.propagate_failure = propagate_failure
        self.propagate_from = propagate_from
        self.last_term = None
        self.root = self.get_or_create(target)
        if self.root is None:
            raise ValueError(f"invalid target SMILES: {target}")

    # ------------------------------------------------------------ pickling
    # canon/purchasable are injected callables (the purchasable check is bound to
    # the optimizer, which holds ~25 GB of shared data) -- never pickle them; the
    # owner re-attaches them after a checkpoint restore.
    def __getstate__(self):
        state = dict(self.__dict__)
        state["canon"] = None
        state["purchasable"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # checkpoints written before the termination fix (2026-11-08) lack these; the 12 Oct pistachio_hard_aot3
        # checkpoints also predate arm bfd (depth_penalty, 20 Oct) and advprune (deprioritised, 1 Nov) -- without
        # these two A_aot3fin died on its first select (13 Nov). Defaults = the constructor's (arms off).
        # ... and checkpoints written before arm reopenG (20 Nov) lack the protect state; a tree restored
        # without it would raise in in_protected/protect. Defaults = the constructor's (arm off).
        for k, v in (("proven_dead", set()), ("propagate_failure", False), ("propagate_from", "proven"),
                     ("last_term", None), ("judged_failed", set()), ("depth_penalty", 0.0), ("deprioritised", {}),
                     ("protect_keys", set()), ("protect_left", 0)):
            self.__dict__.setdefault(k, v)

    # ------------------------------------------------------------ nodes
    def get_or_create(self, smiles):
        c = self.canon(smiles)
        if c is None:
            return None
        node = self.molecules.get(c)
        if node is None:
            node = ORNode(c, bool(self.purchasable(c)))
            self.molecules[c] = node
        return node

    def find(self, smiles):
        c = self.canon(smiles)
        return self.molecules.get(c) if c else None

    def add_reaction(self, product_node, reactant_smiles, depth, reaction=None):
        """Attach product -> reactants. Returns (ANDNode, created?) or (None, False)
        if a reactant is invalid or the reaction is a no-op / self-cycle."""
        kids = [self.get_or_create(s) for s in reactant_smiles]
        if not kids or any(k is None for k in kids):
            return None, False
        kid_smiles = sorted({k.smiles for k in kids})
        if product_node.smiles in kid_smiles:
            return None, False                       # product among its own reactants
        key = product_node.smiles + ">>" + ".".join(kid_smiles)
        existing = self.reactions.get(key)
        if existing is not None:
            return existing, False
        a = ANDNode(key, product_node, depth, reaction)
        seen = set()
        for k in kids:
            if k.smiles in seen:
                continue
            seen.add(k.smiles)
            a.child_molecules.append(k)
            a.reactants.append(k.smiles)
            k.parent_reactions.append(a)
        product_node.child_reactions.append(a)
        self.reactions[key] = a
        if not product_node.is_solved:
            self.leaf_and_nodes.add(a)
        return a, True

    # ------------------------------------------------------------ solved status
    def update_solved(self):
        """Propagate solved status to a fixed point. Returns newly solved molecules."""
        newly = []
        changed = True
        while changed:
            changed = False
            for node in list(self.molecules.values()):
                if node.is_solved:
                    continue
                for a in node.child_reactions:
                    if a.is_fully_solved():
                        node.is_solved = True
                        node.solving_and_node = a
                        newly.append(node)
                        changed = True
                        break
        for node in newly:
            self._retire_subtree_of_solved(node)
        return newly

    def _retire_subtree_of_solved(self, node):
        """A solved molecule needs no more work: drop its reactions (and, through
        them, descendants no longer needed by any live unsolved reaction) from the
        frontier. Mirrors _cleanup_newly_solved_nodes/_cleanup_subtree."""
        visited = set()

        def retire(a):
            if id(a) in visited:
                return
            visited.add(id(a))
            self.leaf_and_nodes.discard(a)
            for child in a.child_molecules:
                live = [p for p in child.parent_reactions
                        if p in self.leaf_and_nodes and not p.is_fully_solved()]
                if not live:
                    for ca in child.child_reactions:
                        retire(ca)

        for a in node.child_reactions:
            retire(a)

    # ------------------------------------------------------------ selection
    def ucb(self, a):
        if a.visit_count == 0:
            return float("inf")
        siblings = sum(s.visit_count for s in a.parent_molecule.child_reactions)
        return a.average_value + self.c_param * math.sqrt(
            math.log(max(siblings, 1)) / max(a.visit_count, 1))

    def path_burden(self, a):
        """Unsolved molecules that must still be made on the cheapest path from this
        reaction to the root (released `_path_unsolved_burden`)."""
        burden = sum(1 for c in a.child_molecules if not c.is_solved)
        cur = a.parent_molecule
        seen = set()
        while cur is not None and cur is not self.root and id(cur) not in seen:
            seen.add(id(cur))
            if not cur.parent_reactions:
                break
            best, best_parent = float("inf"), None
            for p in cur.parent_reactions:
                extra = sum(1 for c in p.child_molecules if c is not cur and not c.is_solved)
                if extra < best:
                    best, best_parent = extra, p
            if best_parent is None:
                break
            burden += best
            cur = best_parent.parent_molecule
        return burden

    def has_expandable_reactants(self, a):
        for c in a.child_molecules:
            if not c.is_solved and c.smiles in self.dead:
                return False          # AND needs ALL children: a dead one dooms it
        return any(not c.is_solved for c in a.child_molecules)

    def in_protected(self, a):
        """Arm reopenG: is this frontier reaction inside the protected opening's subtree? Walks up
        parent_molecule/parent_reactions to the root, so descendants attached AFTER the reopen count too."""
        if not self.protect_keys:
            return False
        seen = set()
        stack = [a]
        while stack:
            n = stack.pop()
            if n.key in self.protect_keys:
                return True
            if n.key in seen:
                continue
            seen.add(n.key)
            stack.extend(n.parent_molecule.parent_reactions)
        return False

    def protect(self, keys, budget):
        """Guarantee the next `budget` selections come from inside these reactions' subtree."""
        self.protect_keys = set(keys or ())
        self.protect_left = int(budget or 0) if self.protect_keys else 0

    def select(self, exclude=None):
        """Flat penalized UCB over frontier reactions (released
        `_selection_phase_flat_ucb`): ucb / (1 + burden * unsolved_penalty).

        With `depth_penalty` > 0 (arm bfd, off by default) the score is divided by
        (1 + depth_penalty * depth) as well, so shallow frontier reactions are
        preferred over deep ones at equal UCB: breadth over depth.

        Arm reopenG (2026-11-20): while a protected quota remains, the candidate set is RESTRICTED to the
        reopened opening's subtree. Measured reason: in reopenC/L/P the new opening was created and then
        ignored -- only 1-2% of the ~150 expansions after a reopen touched depth <= 1, because a branch with
        no visits loses the burden-divided UCB to branches carrying 100 calls of accumulated value. The
        option was generated and never tried, so reopening was never actually tested. If the protected
        subtree has nothing expandable, the quota is dropped rather than stalling the search."""
        cands = [a for a in self.leaf_and_nodes
                 if a.depth < self.max_depth and not a.is_fully_solved()
                 and self.has_expandable_reactants(a)]
        if getattr(self, "protect_left", 0) > 0:
            inside = [a for a in cands if self.in_protected(a)]
            if inside:
                cands = inside
                self.protect_left -= 1
            else:
                self.protect_keys, self.protect_left = set(), 0   # nothing left to try in there
        if exclude:
            # arm steer (2026-11-08): ELSEWHERE = the best reaction outside the current branch
            cands = [a for a in cands if a.key not in exclude]
        if self.propagate_failure:
            # termination fix: drop reactions that failed, or that only failed branches need
            closed = self.closed_by_failure()
            n_open = len(cands)
            cands = [a for a in cands if a.key not in closed]
            self.last_term = {"term_open": n_open, "term_closed": n_open - len(cands),
                              "term_proven_dead": len(self.proven_dead)}
        if not cands:
            return None
        return max(cands, key=lambda a: (self.selection_score(a), -a.depth, a.key))

    # ------------------------------------------------------------ failure (termination fix, 2026-11-08)
    def failure_state(self):
        """Molecules and reactions that can no longer be solved under the search's own rules: a least fixed
        point, recomputed from scratch on every call, so a reaction attached later (route mapping can add one
        beyond max_reexpansions) revives its molecule. Standard AND-OR failure propagation (AO*, proof-number
        search); the released AOT* only blocks the reaction directly above a dead molecule.
          seeds   : proven_dead (no checker template applies). With propagate_from="all", every dead molecule
                    too (an expansion whose LLM tries all failed: not proof, so not the default)
          reaction: failed if ANY unsolved reactant failed (AND)
          molecule: failed if it can get no new reaction (>= max_reexpansions, or dead), has at least one
                    reaction, and ALL of them failed (OR)
        Solved molecules never fail. -> (failed molecule smiles, failed reaction keys)."""
        seeds = set(self.proven_dead) | (set(self.dead) if self.propagate_from == "all" else set())
        fm = {s for s in seeds if s in self.molecules and not self.molecules[s].is_solved}
        fr = {k for k in self.judged_failed if k in self.reactions}   # arm steer REMOVE: permanent, not proof
        changed = True
        while changed:
            changed = False
            for a in self.reactions.values():
                if a.key not in fr and any(c.smiles in fm for c in a.child_molecules if not c.is_solved):
                    fr.add(a.key)
                    changed = True
            for o in self.molecules.values():
                if (o.smiles not in fm and not o.is_solved and o.child_reactions
                        and (len(o.child_reactions) >= self.max_reexpansions or o.smiles in self.dead)
                        and all(x.key in fr for x in o.child_reactions)):
                    fm.add(o.smiles)
                    changed = True
        return fm, fr

    def _reachable_reactions(self, fm, fr):
        """Reaction keys reachable from the target through unsolved molecules not in fm and reactions not
        in fr (and not already fully solved)."""
        live, seen = set(), set()
        stack = [] if self.root.is_solved else [self.root]
        while stack:
            o = stack.pop()
            if id(o) in seen:
                continue
            seen.add(id(o))
            for a in o.child_reactions:
                if a.key in fr or a.is_fully_solved():
                    continue
                live.add(a.key)
                for c in a.child_molecules:
                    if not c.is_solved and c.smiles not in fm:
                        stack.append(c)
        return live

    def closed_by_failure(self):
        """Reaction keys that failed, or that are reachable from the target only through failed reactions or
        molecules. Defined as a difference against the same traversal with nothing failed, so with no
        failures nothing is closed and selection is exactly the release's."""
        fm, fr = self.failure_state()
        if not fm and not fr:
            return set()
        return (self._reachable_reactions(set(), set()) - self._reachable_reactions(fm, fr)) | fr

    def selection_score(self, a, marked=True):
        score = self.ucb(a) / (1.0 + self.path_burden(a) * self.unsolved_penalty)
        if self.depth_penalty:
            score /= 1.0 + self.depth_penalty * a.depth
        if marked and self.deprioritised:
            f = self.deprioritised.get(a.key)
            if f:
                score /= float(f)              # arm advprune: a soft, reversible de-prioritisation
        return score

    def expansion_target(self, a):
        """Unsolved reactant most needing work: never-expanded first, then least
        visited; skips dead ones and ones at max_reexpansions."""
        cands = [c for c in a.child_molecules
                 if not c.is_solved and c.smiles not in self.dead
                 and len(c.child_reactions) < self.max_reexpansions]
        if not cands:
            return None

        def prio(c):
            if not c.child_reactions:
                return (0, 0, c.smiles)
            return (1, sum(x.visit_count for x in c.child_reactions), c.smiles)
        return min(cands, key=prio)

    def selectable(self, a):
        """Would select() consider this reaction at all? (Same filter, without the failure-closure step.)"""
        return (a in self.leaf_and_nodes and a.depth < self.max_depth and not a.is_fully_solved()
                and self.has_expandable_reactants(a))

    def branch_keys(self, a):
        """Arm steer: reaction `a` and the reactions below it that are reachable ONLY through `a`.

        Molecules are merged by canonical SMILES, so the graph has cycles and shared nodes: a plain walk down from
        `a` can come back up to the root and swallow unrelated branches (review, 8 Nov: it made ELSEWHERE vanish).
        So: walk down from `a`, but stop at any molecule that is also reachable from the root without passing
        through `a`. Those molecules, and everything under them, belong to another branch too."""
        outside = self._reachable_molecules(skip=a)
        keys, seen, stack = {a.key}, set(), list(a.child_molecules)
        while stack:
            o = stack.pop()
            if id(o) in seen or o.smiles in outside:
                continue
            seen.add(id(o))
            for x in o.child_reactions:
                keys.add(x.key)
                stack.extend(x.child_molecules)
        return keys

    def subtree_keys(self, o):
        """Arm steer: every reaction below molecule `o`, to any depth (no stopping at shared molecules)."""
        keys, seen, stack = set(), set(), [o]
        while stack:
            m = stack.pop()
            if id(m) in seen:
                continue
            seen.add(id(m))
            for x in m.child_reactions:
                keys.add(x.key)
                stack.extend(x.child_molecules)
        return keys

    def _reachable_molecules(self, skip=None):
        """Molecule smiles reachable from the root through unsolved molecules, never using reaction `skip`."""
        out, stack = set(), [self.root]
        while stack:
            o = stack.pop()
            if o.smiles in out:
                continue
            out.add(o.smiles)
            for x in o.child_reactions:
                if x is skip:
                    continue
                for ch in x.child_molecules:
                    if not ch.is_solved:
                        stack.append(ch)
        return out

    def closed_by_failure_if(self, a):
        """What REMOVE on reaction `a` would really close: `a` plus the reactions that nothing else still needs.
        Used for the number quoted to the advisor, so it is never larger than the truth."""
        fm, fr = self.failure_state()
        return (self._reachable_reactions(fm, fr) - self._reachable_reactions(fm, fr | {a.key})) | {a.key}

    def retire(self, a):
        self.leaf_and_nodes.discard(a)

    # ------------------------------------------------------------ values
    def init_stats(self, a, score):
        a.visit_count = 1
        a.total_value = score
        a.average_value = score
        a.feasibility_score = score

    def backpropagate(self, a, reward):
        if self.backprop == "andor":
            self._backprop_andor(a)
        else:
            self._backprop_average(a, reward, set())

    def _backprop_average(self, a, reward, visited):
        if id(a) in visited:
            return
        visited.add(id(a))
        cur = a.parent_molecule
        if cur is None or cur is self.root:
            return
        for p in cur.parent_reactions:
            p.visit_count += 1
            p.total_value += reward
            p.average_value = p.total_value / p.visit_count
            self._backprop_average(p, reward, visited)

    def _backprop_andor(self, a):
        visited = set()

        def or_value(o):
            if o.is_solved:
                return 1.0
            if not o.child_reactions:
                return 0.5
            vals = [x.average_value for x in o.child_reactions
                    if self.has_expandable_reactants(x) or x.is_fully_solved()]
            return max(vals) if vals else 0.0

        def up(o):
            if id(o) in visited or o is self.root:
                return
            visited.add(id(o))
            for p in o.parent_reactions:
                vals = [or_value(c) for c in p.child_molecules]
                v = min(vals) if vals else 0.0
                p.visit_count += 1
                p.average_value = v
                p.total_value = v * p.visit_count
                up(p.parent_molecule)

        up(a.parent_molecule)

    # ------------------------------------------------------------ output
    def solution(self):
        """Solved synthesis tree from the root (None if unsolved)."""
        if not self.root.is_solved:
            return None

        def build(o, seen):
            if o.purchasable:
                return {"type": "building_block", "molecule": o.smiles}
            a = o.solving_and_node
            if a is None or o.smiles in seen:
                return {"type": "unresolved", "molecule": o.smiles}
            return {"type": "reaction", "molecule": o.smiles, "reaction": a.reaction,
                    "reactants": [build(c, seen | {o.smiles}) for c in a.child_molecules]}
        return build(self.root, frozenset())

    def solution_steps(self):
        """Solution as a flat list of (product, [reactants]) steps, root first."""
        out = []

        def walk(o, seen):
            a = o.solving_and_node
            if o.purchasable or a is None or o.smiles in seen:
                return
            out.append((o.smiles, list(a.reactants)))
            for c in a.child_molecules:
                walk(c, seen | {o.smiles})
        if self.root.is_solved:
            walk(self.root, frozenset())
        return out

    def stats(self):
        out = {"molecules": len(self.molecules), "reactions": len(self.reactions),
               "frontier": len(self.leaf_and_nodes),
               "solved_molecules": sum(1 for m in self.molecules.values() if m.is_solved),
               "dead": len(self.dead)}
        if self.propagate_failure or self.proven_dead:     # termination fix only: keys absent otherwise
            fm, fr = self.failure_state()
            closed = self.closed_by_failure() if self.propagate_failure else set()
            out.update({"proven_dead": len(self.proven_dead), "failed_molecules": len(fm),
                        "failed_reactions": len(fr),
                        "frontier_closed_by_failure": len(closed & {a.key for a in self.leaf_and_nodes})})
        return out
