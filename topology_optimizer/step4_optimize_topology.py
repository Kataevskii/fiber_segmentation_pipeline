"""
step4_optimize_topology.py
===========================
Global fiber topology optimizer.

Problem formulation
-------------------
We have a set of skeleton fragments F = {f_1, ..., f_N}, each with two
endpoints (head, tail). We want to assign each fragment into a "fiber chain"
that:
  - Is a simple path (no branching, no V-shapes) -- degree constraint
  - Maximises fiber length
  - Has smooth direction (no sharp bends)
  - Has consistent curvature (gradual curves are fine)

This is equivalent to a **minimum-weight perfect matching on an endpoint
compatibility graph**, which is NP-hard in general but tractable here because:
  1. The graph is very sparse (only nearby endpoints can bridge)
  2. Fibers never split (= each endpoint connects to at most ONE other endpoint)
  3. We accept near-optimal greedy solutions

Algorithm
---------
1. Gather all bridge candidates (from step 2) and all isolated fragments.
2. Score each bridge with the composite cost from cost_functions.
3. Sort candidates by score (lower cost = better).
4. Greedily commit bridges in order:
      - A bridge (ep_i, ep_j) is accepted if:
          * neither ep_i nor ep_j has already been connected (degree constraint)
          * the resulting chain does not form a cycle
5. Unconnected endpoints remain as standalone stubs.
6. Return the final assignment: each fragment -> chain ID.

This greedy approach is O(B log B) in the number of bridge candidates and
empirically achieves near-optimal solutions for sparse geometric graphs.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from .step2_bridge_gaps import BridgeCandidate
    from .step3_build_fragment_graph import FiberFragment
    from .cost_functions import merged_fiber_cost, segment_self_cost, bridge_cost as bc_func
except ImportError:
    from step2_bridge_gaps import BridgeCandidate
    from step3_build_fragment_graph import FiberFragment
    from cost_functions import merged_fiber_cost, segment_self_cost, bridge_cost as bc_func


# ---------------------------------------------------------------------------
# Union-Find for cycle detection
# ---------------------------------------------------------------------------

class UnionFind:
    """Path-compressed union-find with rank."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False   # would form a cycle
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

@dataclass
class FiberChain:
    """A connected chain of fiber fragments (= one resolved fiber)."""
    chain_id: int
    fragment_ids: list[int]          # ordered fragment IDs
    bridge_coords: list[tuple]       # list of (coord_a, coord_b) for each bridge
    total_length: int                # sum of fragment lengths (no gap voxels)
    total_cost: float


def optimize_topology(
    fragments: list[FiberFragment],
    candidates: list[BridgeCandidate],
    max_commit_cost: float = 35.0,
    verbose: bool = True,
) -> list[FiberChain]:
    """
    Greedy minimum-cost topology optimizer.

    Parameters
    ----------
    fragments       : list[FiberFragment] -- from step 3
    candidates      : list[BridgeCandidate] -- from step 2, sorted best-first
    max_commit_cost : float -- bridges with cost exceeding this (e.g. sharp turns, large gaps)
                      are penalized out and not committed (default: 35.0)
    verbose         : bool

    Returns
    -------
    chains : list[FiberChain] -- one chain per resolved fiber
    """
    N = len(fragments)
    if N == 0:
        return []

    # Map frag_id -> index
    frag_id_to_idx = {f.frag_id: i for i, f in enumerate(fragments)}

    # ------------------------------------------------------------------ #
    # Pre-score all candidates using bridge_cost with turn penalties     #
    # ------------------------------------------------------------------ #
    scored: list[tuple[float, BridgeCandidate]] = []
    for cand in candidates:
        idx_a = frag_id_to_idx.get(cand.seg_id_a)
        idx_b = frag_id_to_idx.get(cand.seg_id_b)
        if idx_a is None or idx_b is None:
            continue
        fa = fragments[idx_a]
        fb = fragments[idx_b]

        cost = bc_func(
            cand.ori_a, cand.ori_b,
            cand.gap_distance,
            cand.path_ori_samples,
            forward_align_a=cand.forward_align_a,
            forward_align_b=cand.forward_align_b,
        )
        scored.append((cost, cand))

    # Sort ascending by cost (lower = commit first)
    scored.sort(key=lambda x: x[0])

    if verbose:
        print(f"  [optimizer] {N} fragments, {len(scored)} scored bridge candidates.", flush=True)

    # ------------------------------------------------------------------ #
    # Greedy assignment with degree constraint and cycle avoidance        #
    # ------------------------------------------------------------------ #
    uf = UnionFind(N)

    # Each endpoint can connect at most once: track per fragment which endpoints
    # are "used". A fragment has two endpoints: 'head' (idx 0) and 'tail' (idx -1).
    head_used = [False] * N   # fragment i's head endpoint is used
    tail_used = [False] * N   # fragment i's tail endpoint is used

    accepted_bridges: list[tuple[BridgeCandidate, int, int]] = []

    # We need to track which endpoint of each fragment a candidate touches.
    # We use Euclidean distance to decide: is cand.coord_a closer to the head
    # or tail of fragment fa?
    def closest_endpoint(frag: FiberFragment, coord: np.ndarray) -> str:
        """Return 'head' or 'tail' -- whichever is closer to coord."""
        d_head = float(np.linalg.norm(frag.head_coord - coord))
        d_tail = float(np.linalg.norm(frag.tail_coord - coord))
        return 'head' if d_head <= d_tail else 'tail'

    for cost, cand in scored:
        if max_commit_cost is not None and cost > max_commit_cost:
            continue

        idx_a = frag_id_to_idx.get(cand.seg_id_a)
        idx_b = frag_id_to_idx.get(cand.seg_id_b)
        if idx_a is None or idx_b is None:
            continue

        fa = fragments[idx_a]
        fb = fragments[idx_b]

        ep_a = closest_endpoint(fa, cand.coord_a)
        ep_b = closest_endpoint(fb, cand.coord_b)

        # Check degree constraint
        ep_a_used = head_used[idx_a] if ep_a == 'head' else tail_used[idx_a]
        ep_b_used = head_used[idx_b] if ep_b == 'head' else tail_used[idx_b]
        if ep_a_used or ep_b_used:
            continue

        # Check cycle constraint
        if not uf.union(idx_a, idx_b):
            continue   # would form a cycle

        # Accept!
        if ep_a == 'head':
            head_used[idx_a] = True
        else:
            tail_used[idx_a] = True

        if ep_b == 'head':
            head_used[idx_b] = True
        else:
            tail_used[idx_b] = True

        accepted_bridges.append((cand, idx_a, idx_b))

    if verbose:
        print(f"  [optimizer] Accepted {len(accepted_bridges)} bridges.", flush=True)

    # ------------------------------------------------------------------ #
    # Build chains from union-find groups                                 #
    # ------------------------------------------------------------------ #
    from collections import defaultdict
    group_to_frags: dict[int, list[int]] = defaultdict(list)
    for i in range(N):
        group_to_frags[uf.find(i)].append(i)

    # For each group, collect which bridges connect them
    bridge_by_pair: dict[tuple, BridgeCandidate] = {}
    for cand, idx_a, idx_b in accepted_bridges:
        bridge_by_pair[(idx_a, idx_b)] = cand
        bridge_by_pair[(idx_b, idx_a)] = cand

    chains: list[FiberChain] = []
    chain_id = 1

    for root, frag_indices in group_to_frags.items():
        if not frag_indices:
            continue

        total_len = sum(fragments[i].length for i in frag_indices)
        frag_id_list = [fragments[i].frag_id for i in frag_indices]

        # Collect bridges for this chain
        bridges_in_chain = []
        for (ia, ib), cand in bridge_by_pair.items():
            if ia in frag_indices and ib in frag_indices and (ia, ib) in bridge_by_pair:
                bridges_in_chain.append((cand.coord_a, cand.coord_b))

        # Cost estimate for this chain
        cost_val = sum(segment_self_cost(fragments[i].coords) for i in frag_indices)

        chain = FiberChain(
            chain_id=chain_id,
            fragment_ids=frag_id_list,
            bridge_coords=bridges_in_chain,
            total_length=total_len,
            total_cost=cost_val,
        )
        chains.append(chain)
        chain_id += 1

    # Sort chains by total length descending (longest first)
    chains.sort(key=lambda c: c.total_length, reverse=True)

    if verbose:
        lengths = [c.total_length for c in chains]
        print(f"  [optimizer] Resolved into {len(chains)} fiber chains.", flush=True)
        if lengths:
            print(f"             Length: min={min(lengths)}, "
                  f"mean={np.mean(lengths):.0f}, max={max(lengths)}", flush=True)

    return chains
