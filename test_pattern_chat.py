#!/usr/bin/env python3
"""
Headless tests for the Ask-Claude-about-a-pattern chat (no network).

- build_pattern_context: for real patterns, includes each one's Lua and what the
  engine drew on an example serial.
- PatternChatSession: prepends the context to the first turn, accumulates history,
  and returns the transport's text — all via an injected fake transport, so no
  API is ever called.

Run:  ./venv/bin/python test_pattern_chat.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pattern_engine_v3 import PatternEngineV3
import ai_pattern_chat


def _patterns_with_examples(engine, n=2):
    picked = []
    for name, info in engine.lua_patterns.items():
        if info.examples:
            picked.append(name)
        if len(picked) >= n:
            break
    return picked


def test_build_context():
    engine = PatternEngineV3()
    names = _patterns_with_examples(engine, 2)
    assert len(names) == 2, "need two patterns with examples to test"
    ctx = ai_pattern_chat.build_pattern_context(engine, names)

    assert "function match" in ctx, "context should include the Lua source"
    assert "MATCHED" in ctx, "context should describe what the engine drew on an example"
    for name in names:
        assert name in ctx, f"context should name pattern {name}"
        sample = engine.lua_patterns[name].examples[0]
        assert sample in ctx, f"context should quote the example serial {sample}"
    # Lua is capped
    assert len(ctx) < ai_pattern_chat.MAX_LUA * 3 + 5000
    print(f"  build_context: {len(names)} patterns, {len(ctx)} chars, Lua + drawings present  OK")


def test_session_no_network():
    calls = {"n": 0, "last_messages": None, "last_system": None}

    def fake_transport(system, messages, model, max_tokens):
        calls["n"] += 1
        calls["last_messages"] = messages
        calls["last_system"] = system
        return f"reply #{calls['n']}"

    s = ai_pattern_chat.PatternChatSession(
        provider="anthropic", api_key="test-key", model="claude-opus-5",
        transport=fake_transport)

    r1 = s.ask("What does this do?", context="CONTEXT-BLOCK about the pattern")
    assert r1.success and r1.text == "reply #1", r1
    # context is prepended to the first user message
    assert "CONTEXT-BLOCK" in calls["last_messages"][-1]["content"]
    assert "What does this do?" in calls["last_messages"][-1]["content"]
    assert calls["last_system"] == ai_pattern_chat.SYSTEM
    assert len(s.history) == 2  # user + assistant committed

    r2 = s.ask("And compared to a radar?")  # no new context this turn
    assert r2.success and r2.text == "reply #2"
    # second turn sends the running history + the new question, without re-sending context
    assert "CONTEXT-BLOCK" not in calls["last_messages"][-1]["content"]
    assert len(calls["last_messages"]) == 3  # u, a, u
    assert len(s.history) == 4

    # empty input and missing key are handled without a transport call
    assert not ai_pattern_chat.PatternChatSession("anthropic", "", "m").ask("hi").success
    assert not s.ask("   ").success
    print(f"  session: {calls['n']} transport calls, history bounded, no network  OK")


def test_resolve_provider():
    class _AI:
        provider = "openai"; anthropic_api_key = ""; openai_api_key = "k-openai"
        anthropic_model = "claude-opus-5"; openai_model = "gpt-5"

    class _S:
        ai = _AI()

    prov, key, model = ai_pattern_chat.resolve_provider(_S())
    assert (prov, key, model) == ("openai", "k-openai", "gpt-5"), (prov, key, model)
    assert ai_pattern_chat.is_configured(_S())
    # falls back to whichever provider has a key even if provider mismatches
    _AI.provider = "anthropic"; _AI.anthropic_api_key = ""
    assert ai_pattern_chat.resolve_provider(_S())[0] == "openai"
    _AI.openai_api_key = ""
    assert not ai_pattern_chat.is_configured(_S())
    print("  resolve_provider: picks a configured provider, else none  OK")


def main():
    print("test_pattern_chat:")
    test_build_context()
    test_session_no_network()
    test_resolve_provider()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
