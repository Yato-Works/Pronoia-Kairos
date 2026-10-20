"""Exact-path KV trie store for M4 (PREREG_M4_REALWORLD §8 addendum 4).

Real prefix caching with MC0-exact semantics: a request may only reuse the
longest resident *exact* token path. Each node stores the KV span of exactly
one runtime key (1:1, mirror-gated); when the mirror evicts a key the node
and any now-unreachable descendants are dropped from the real store while
the mirror keeps sim-style accounting -> real stored tokens <= mirror
accounting <= budget (conservative, §8-4).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

Path = Tuple[int, ...]


class TrieNode:
    __slots__ = ("path", "owner_key", "start", "end", "layers")

    def __init__(self, path: Path, owner_key: str, start: int, end: int,
                 layers: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.path = path
        self.owner_key = owner_key
        self.start = start
        self.end = end
        self.layers = layers


class SegmentTrieStore:
    """KV spans keyed by exact token paths; eviction driven by the mirror."""

    def __init__(self, budget_tokens: int):
        self.budget = budget_tokens
        self.nodes: Dict[Path, TrieNode] = {}
        self.owner_path: Dict[str, Path] = {}
        self.usage = 0

    # -- queries -----------------------------------------------------------

    def walk(self, token_path: Sequence[int],
             boundaries: Sequence[Tuple[int, str]]) -> Tuple[int, Optional[TrieNode]]:
        """Deepest resident exact path along the request's segment boundaries."""
        hit_end, hit_node = 0, None
        for end_pos, _owner in boundaries:
            node = self.nodes.get(tuple(token_path[:end_pos]))
            if node is not None:
                hit_end, hit_node = end_pos, node
        return hit_end, hit_node

    def _chain(self, node: TrieNode) -> List[TrieNode]:
        chain = [node]
        while chain[0].start > 0:
            parent = self.nodes.get(chain[0].path[: chain[0].start])
            if parent is None or parent.end != chain[0].start:
                break  # invariant violation guard: stop, stay conservative
            chain.insert(0, parent)
        return chain

    def cache_up_to(self, token_path: Sequence[int], hit_end: int
                    ) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Rebuild per-layer (key, value) lists covering positions [0, hit_end)."""
        if hit_end <= 0:
            return None
        node = self.nodes.get(tuple(token_path[:hit_end]))
        if node is None:
            return None
        chain = self._chain(node)
        n_layers = len(chain[0].layers)
        keys = [torch.cat([n.layers[i][0] for n in chain], dim=2)
                for i in range(n_layers)]
        values = [torch.cat([n.layers[i][1] for n in chain], dim=2)
                  for i in range(n_layers)]
        return keys, values

    # -- writes ------------------------------------------------------------

    def materialize(self, token_path: Sequence[int],
                    computed: List[Tuple[torch.Tensor, torch.Tensor]],
                    boundaries: Sequence[Tuple[int, str]],
                    gates: Dict[str, bool], hit_end: int) -> List[Tuple[str, int]]:
        """Store new segment nodes from a just-finished prefill.

        computed: per-layer (k, v) for the FULL current sequence [0, L].
        gates: owner_key -> mirror says NOT resident (only then may a node be
        created; §8-4 1:1 rule). Materialization stops at the first blocked
        gate so every stored span is exactly one mirror segment.
        """
        stored: List[Tuple[str, int]] = []
        prev = hit_end
        for end_pos, owner in boundaries:
            if end_pos <= hit_end:
                continue
            if not gates.get(owner, False):
                break
            path = tuple(token_path[:end_pos])
            if path in self.nodes:
                break
            layers = [(k[:, :, prev:end_pos, :].contiguous(),
                       v[:, :, prev:end_pos, :].contiguous())
                      for k, v in computed]
            self.nodes[path] = TrieNode(path, owner, prev, end_pos, layers)
            self.owner_path[owner] = path
            self.usage += end_pos - prev
            stored.append((owner, end_pos - prev))
            prev = end_pos
        return stored

    def evict_keys(self, keys: Sequence[str]) -> int:
        """Mirror-driven eviction: drop owner nodes + unreachable descendants."""
        freed = 0
        for key in keys:
            path = self.owner_path.pop(key, None)
            if path is None:
                continue
            doomed = [p for p in self.nodes
                      if p == path or p[: len(path)] == path]
            for p in doomed:
                node = self.nodes.pop(p)
                freed += node.end - node.start
                self.usage -= node.end - node.start
                if node.owner_key != key:
                    self.owner_path.pop(node.owner_key, None)
        return freed

    # -- introspection -----------------------------------------------------

    def owners(self) -> List[str]:
        return sorted(self.owner_path)

    def check_accounting(self) -> bool:
        total = sum(n.end - n.start for n in self.nodes.values())
        return total == self.usage and self.usage <= self.budget
