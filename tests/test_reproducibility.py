# reproducibility: PYTHONHASHSEED 非依存の回帰テスト (2026-10-19 バグ検出)
"""_pf_live が set のままだと hit 承認ループの同点決着が hash順依存に。

事前登録 PREREG_STRONG_REEVAL.md §3 の保守性チェック (seed12_merged と
1e-6 一致) が W7 で不一致を検出して発覚。同一コード・同一入力でも
プロセス毎に goodput/p値が揺れていた (W8 seed1 で F-E が 0.0 / 4.02e-05)。

修正: runtime/adaptive_runtime.py `_pf_live` を set → 挿入順維持 dict。
本テストは異なる PYTHONHASHSEED のサブプロセスで同一 stream を走らせ、
全 config の (goodput, evictions, prefetch_hits, prefetch_waste) が
完全一致することを回帰確認する。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_RUNNER = r"""
import json
from compare.full_runtime_experiment import build_family, ladder
from runtime.adaptive_runtime import AdaptiveRuntime
stream, spec = build_family("W8_RAG", 1)
out = []
for cfg in ladder(spec):
    r = AdaptiveRuntime(cfg).serve(list(stream))
    out.append([cfg.name, r.goodput, r.evictions,
                r.prefetch_hits, r.prefetch_waste])
print("RESULT" + json.dumps(out))
"""


def _run(hashseed: str) -> list:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", _RUNNER], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = [ln for ln in proc.stdout.splitlines()
             if ln.startswith("RESULT")]
    assert lines, proc.stdout[-2000:]
    return json.loads(lines[0][len("RESULT"):])


@pytest.mark.parametrize("pair", [("0", "6"), ("1", "42")])
def test_hashseed_does_not_change_runtime_results(pair):
    """PYTHONHASHSEED が違っても全 config の結果が bit 単位で一致する。

    pair ("0","6") は修正前 W8 で実際に乖離した組み合わせ。
    """
    a, b = _run(pair[0]), _run(pair[1])
    assert a == b, f"hashseed-dependent results: {pair}: {a} vs {b}"
