# M3: semantic demand shuffle — pi 構築 + 介入点の回帰テスト (PREREG §6)
"""PREREG_M3_SHUFFLE.md §6 のテスト要件:

  pi 構築の決定性 / 双射 (MC1) / 多集合保存 (MC2) / stream hash 決定性 (MC3) /
  signal_perm=None の現行同一性 (True condition) / PYTHONHASHSEED 非依存。

MC4 (seed12 一致) と MC5 (repeat 一致) は compare/m3_shuffle_runner.py が
384 実行時に検査し、m3_shuffle_run_summary.json に保存する。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from compare.full_runtime_experiment import build_family, ladder
from compare.m3_shuffle_runner import (build_signal_perm, stream_hash,
                                       stream_keys)
from runtime.adaptive_runtime import AdaptiveRuntime, RuntimeConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# MC1: pi construction — determinism, bijection, salt separation
# ---------------------------------------------------------------------------

def test_perm_deterministic_regardless_of_input_order():
    """sorted(U) 前提なら入力順・集合列挙順に依存しない (hash非依存の根拠)。"""
    keys = [f"k{i}" for i in range(50)]
    p1 = build_signal_perm(keys, "W7_agent", 3)
    p2 = build_signal_perm(list(reversed(keys)), "W7_agent", 3)
    p3 = build_signal_perm(sorted(set(keys), reverse=True), "W7_agent", 3)
    assert p1 == p2 == p3


def test_perm_bijection_on_stream_universe():
    stream, _ = build_family("W7_agent", 5)
    universe = stream_keys(stream)
    perm = build_signal_perm(universe, "W7_agent", 5)
    assert sorted(perm.keys()) == universe
    assert sorted(perm.values()) == universe
    assert len(set(perm.values())) == len(universe)
    assert all(v in perm for v in perm.values())


def test_perm_salt_separates_family_and_seed():
    """同じ U でも family/seed で別の pi (導出系の分離 = PREREG §2-2)。"""
    stream, _ = build_family("W7_agent", 5)
    universe = stream_keys(stream)
    a = build_signal_perm(universe, "W7_agent", 5)
    b = build_signal_perm(universe, "W7_agent", 6)
    c = build_signal_perm(universe, "W8_RAG", 5)
    assert a != b and a != c


def test_perm_small_universe_exclusion():
    """|U| < 2 は除外 (shuffle 不能) — 空 pi、ランナーが記録 (§2 除外条件)。"""
    assert build_signal_perm(["a"], "X", 1) == {}
    assert build_signal_perm([], "X", 1) == {}
    p = build_signal_perm(["a", "b"], "X", 1)
    assert sorted(p) == ["a", "b"]
    assert sorted(p.values()) == ["a", "b"]


# ---------------------------------------------------------------------------
# MC3: stream hash determinism
# ---------------------------------------------------------------------------

def test_stream_hash_deterministic_and_seed_sensitive():
    s1, _ = build_family("W7_agent", 9)
    s2, _ = build_family("W7_agent", 9)
    s3, _ = build_family("W7_agent", 10)
    assert stream_hash(s1) == stream_hash(s2)
    assert stream_hash(s1) != stream_hash(s3)


# ---------------------------------------------------------------------------
# PYTHONHASHSEED independence of the permutation (M1 lesson, subprocess)
# ---------------------------------------------------------------------------

_PERM_RUNNER = r"""
import json
from compare.m3_shuffle_runner import build_signal_perm, stream_keys
from compare.full_runtime_experiment import build_family
s, _ = build_family("W2_short_chat_high_reuse", 7)
print("RESULT" + json.dumps(
    build_signal_perm(stream_keys(s), "W2_short_chat_high_reuse", 7),
    sort_keys=True))
"""


def _perm_for_hashseed(hashseed: str) -> dict:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", _PERM_RUNNER], cwd=ROOT,
                          env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = [ln for ln in proc.stdout.splitlines()
             if ln.startswith("RESULT")]
    assert lines, proc.stdout[-2000:]
    return json.loads(lines[0][len("RESULT"):])


@pytest.mark.parametrize("pair", [("0", "6"), ("1", "42")])
def test_perm_independent_of_hashseed(pair):
    """PYTHONHASHSEED が違っても pi は完全一致 (sha256 + sorted の帰結)。"""
    assert _perm_for_hashseed(pair[0]) == _perm_for_hashseed(pair[1])


# ---------------------------------------------------------------------------
# Intervention semantics: signal_perm=None identity / pi re-routing
# ---------------------------------------------------------------------------

COUNTS = {"a": 1, "b": 4, "c": 2}


def _rt_with(counts, perm=None) -> AdaptiveRuntime:
    rt = AdaptiveRuntime(RuntimeConfig(name="m3_t", signal_perm=perm))
    rt.access_count.update(counts)
    rt._sig_cache = {}
    rt._sig_cache_max = 0
    return rt


def test_signal_perm_default_none_in_ladder():
    _, spec = build_family("W2_short_chat_high_reuse", 1)
    for cfg in ladder(spec):
        assert cfg.signal_perm is None


def test_signal_none_is_identity():
    rt = _rt_with(COUNTS)
    assert rt._signal("b") == 1.0
    assert rt._signal("a") == 0.25
    assert rt._signal("unknown") == 0.0
    assert rt._signals() == {"a": 0.25, "b": 1.0, "c": 0.5}


def test_identity_perm_equals_none():
    ident = {k: k for k in COUNTS}
    rt_n = _rt_with(COUNTS)
    rt_i = _rt_with(COUNTS, ident)
    for k in list(COUNTS) + ["unknown"]:
        assert rt_i._signal(k) == rt_n._signal(k)
    assert rt_i._signals() == rt_n._signals()


def test_perm_reroutes_all_signal_reads():
    """経路1-3: _signal(k) == raw(pi(k))、_signals() = {k: raw(pi(k))}、
    key 集合は True と同一 (§2 適用仕様)。"""
    perm = {"a": "b", "b": "c", "c": "a"}
    rt = _rt_with(COUNTS, perm)
    assert rt._signal("a") == 4 / 4       # raw(pi(a)) = raw(b)
    assert rt._signal("b") == 2 / 4       # raw(pi(b)) = raw(c)
    assert rt._signal("c") == 1 / 4       # raw(pi(c)) = raw(a)
    sigs = rt._signals()
    assert set(sigs) == set(COUNTS)       # key set unchanged
    assert sigs == {"a": 1.0, "b": 0.5, "c": 0.25}
    assert sorted(sigs.values()) == sorted([0.25, 0.5, 1.0])  # multiset


def test_perm_key_used_by_min_seen_path():
    """経路4: min_seen は _perm_key(target) 経由 (コード上の参照を回帰)。
    min_seen 判定は _serve_one 内にあり、クラス全体ソースで検査する。"""
    import inspect
    src = inspect.getsource(AdaptiveRuntime)
    assert "self.access_count.get(self._perm_key(target), 0)" in src
    assert "self.access_count.get(target, 0)" not in src


def test_multiset_preserved_over_universe_including_unseen():
    """MC2: U 全体で evidence 多集合は pi 下で不変 (正規化値も含む)。
    アクセス済/未エビデンス境界をまたぐ pi でも構造的に同一。"""
    perm = {"a": "d", "b": "c", "c": "b", "d": "a"}
    rt = _rt_with(COUNTS, perm)
    universe = sorted(set(COUNTS) | {"d"})
    lhs = sorted(rt.access_count.get(perm[k], 0) for k in universe)
    rhs = sorted(rt.access_count.get(k, 0) for k in universe)
    assert lhs == rhs == [0, 1, 2, 4]
    mx = max(rt.access_count.values())
    assert (sorted(rt.access_count.get(perm[k], 0) / mx for k in universe)
            == sorted(rt.access_count.get(k, 0) / mx for k in universe))
    # _signals() の key 集合はアクセス済み集合のまま (仕様 §2-step4)
    assert set(rt._signals()) == set(COUNTS)


def test_record_access_truth_unaffected_by_perm():
    """PREREG §2「固定」: 記録は常に真値 — perm の有無で増え方が変わらない。"""
    rt_t = _rt_with({}, None)
    rt_s = _rt_with({}, {"a": "b", "b": "c", "c": "a"})
    for rt in (rt_t, rt_s):
        rt._record_access(["a", "b", "a"])
    assert rt_t.access_count == {"a": 2, "b": 1}
    assert rt_s.access_count == {"a": 2, "b": 1}


def test_shuffled_config_runs_end_to_end():
    """perm を付けた F_full が通常どおり execute できる (smoke)。"""
    stream, spec = build_family("W2_short_chat_high_reuse", 1)
    universe = stream_keys(stream)
    perm = build_signal_perm(universe, "W2_short_chat_high_reuse", 1)
    f_cfg = next(c for c in ladder(spec) if c.name == "F_full")
    res = AdaptiveRuntime(replace(f_cfg, signal_perm=perm)).serve(
        list(stream))
    assert res.n_requests == len(stream)
    assert res.goodput >= 0.0
