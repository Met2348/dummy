"""A mini agent harness, built from scratch (stdlib-only, deterministic).

The harness is the runtime engine that turns a raw model into an agent: the
agentic loop, tool dispatch, context management, permissions, memory, tracing.
Each submodule is one concern; `loop.py` ties them together; `mini_harness.py`
assembles a usable Harness.
"""
