"""harness-engineering 组件单测 (stdlib unittest, 无网/无 key/无 GPU)。

运行: python -m unittest discover -s src/tests   (在专题根目录下)
或:   python src/tests/test_all.py
"""
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

import provider as P
import compaction as C
import otel_trace as T
import long_horizon as L
import harness_eval as E


class TestProvider(unittest.TestCase):
    def test_mock_default_deterministic(self):
        prov = P.MockProvider()
        msgs = [{"role": "user", "content": "用 echo 工具"}]
        a = list(prov.stream(msgs, tools=[{"name": "echo"}]))
        prov2 = P.MockProvider()
        b = list(prov2.stream(msgs, tools=[{"name": "echo"}]))
        self.assertEqual([(c.kind, c.text, c.tool_name) for c in a],
                         [(c.kind, c.text, c.tool_name) for c in b])
        self.assertTrue(any(c.kind == "tool_call" for c in a))
        self.assertTrue(any(c.kind == "done" for c in a))

    def test_script_advances(self):
        prov = P.MockProvider(script=[P.Turn(text="一"), P.Turn(text="二", stop=True)])
        t1 = "".join(c.text for c in prov.stream([], None) if c.kind == "text")
        t2 = "".join(c.text for c in prov.stream([], None) if c.kind == "text")
        self.assertIn("一", t1)
        self.assertIn("二", t2)


class TestCompaction(unittest.TestCase):
    def _big_history(self):
        msgs = [{"role": "system", "content": "系统提示", "pinned": True}]
        for i in range(20):
            msgs.append({"role": "user", "content": "x" * 600, "kind": "tool_result"})
        return msgs

    def test_reduces_under_budget(self):
        msgs = self._big_history()
        before = C.total_tokens(msgs)
        comp = C.Compactor(max_tokens=before // 4)
        new, events = comp.compact(msgs)
        self.assertLessEqual(C.total_tokens(new), before // 4)
        self.assertTrue(events)

    def test_system_pinned_survives(self):
        msgs = self._big_history()
        comp = C.Compactor(max_tokens=50)
        new, _ = comp.compact(msgs)
        self.assertTrue(any(m.get("role") == "system" and m.get("kind") != "summary" for m in new))

    def test_noop_when_under_budget(self):
        msgs = [{"role": "user", "content": "短"}]
        comp = C.Compactor(max_tokens=10_000)
        new, events = comp.compact(msgs)
        self.assertEqual(events, [])
        self.assertEqual(new, msgs)

    def test_stages_escalate(self):
        msgs = self._big_history()
        comp = C.Compactor(max_tokens=80)
        _, events = comp.compact(msgs)
        stages = {e.stage for e in events}
        self.assertTrue(max(stages) >= 2)  # 轻手段不够, 升级到更重阶段


class TestTracer(unittest.TestCase):
    def test_nesting_and_stats(self):
        tr = T.Tracer()
        with tr.span("window", "window"):
            with tr.span("reason", "reasoning"):
                with tr.span("tool:echo", "tool"):
                    pass
        d = tr.to_dict()
        self.assertEqual(d[0]["kind"], "window")
        self.assertEqual(d[0]["children"][0]["children"][0]["kind"], "tool")
        stats = tr.stats()
        self.assertEqual(stats["tool"]["count"], 1)
        self.assertIn("◆", tr.render())


class TestLongHorizon(unittest.TestCase):
    def test_hook_rescues_early_stop(self):
        with tempfile.TemporaryDirectory() as d:
            prov, goal, tools, store, gm = L.demo_setup(d, total_steps=6, early_stop_at=2)
            res = L.run_long_horizon(prov, goal, tools, store, gm, hook=True, max_windows=6)
            self.assertTrue(res.success)
            self.assertGreaterEqual(res.n_windows, 2)          # 至少换了一次窗口
            self.assertTrue(any(w.stop_intercepted for w in res.windows))

    def test_no_hook_fails_long_task(self):
        with tempfile.TemporaryDirectory() as d:
            prov, goal, tools, store, gm = L.demo_setup(d, total_steps=6, early_stop_at=2)
            res = L.run_long_horizon(prov, goal, tools, store, gm, hook=False, max_windows=6)
            self.assertFalse(res.success)
            self.assertTrue(res.aborted_early)

    def test_state_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as d:
            prov, goal, tools, store, gm = L.demo_setup(d, total_steps=4, early_stop_at=2)
            L.run_long_horizon(prov, goal, tools, store, gm, hook=True)
            self.assertTrue((Path(d) / "state.json").exists())
            self.assertGreaterEqual(store.load()["progress"], 4)


class TestHarnessEval(unittest.TestCase):
    def test_configs_differ(self):
        with tempfile.TemporaryDirectory() as d:
            rows = E.evaluate(E.default_configs(), d, total_steps=6, early_stop_at=2)
            by = {r["harness"]: r for r in rows}
            # 无 hook 的 naive 失败; 有 hook 的成功
            self.assertFalse(by["A_naive"]["success"])
            self.assertTrue(by["B_hook_only"]["success"])
            self.assertTrue(by["C_hook_compaction"]["success"])
            # compaction 降低上下文成本
            self.assertLess(by["C_hook_compaction"]["context_tokens"],
                            by["B_hook_only"]["context_tokens"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
