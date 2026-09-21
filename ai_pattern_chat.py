"""
Shared prompt + context for the "Ask Claude about this pattern" chat.

Importable WITHOUT Qt (the GUI dialog in gui/pattern_chat_dialog.py drives it).
Ported from the web tool's api/app/ask.py: the two web system prompts (SYSTEM =
analyze/compare, WRITER = write/fix scripts) are MERGED here into one, because the
desktop chat is a single unified conversation that does both. The per-pattern
context builder mirrors ask.py._prompt: each pattern's header facts, its Lua
source (capped), and what the engine drew on one example serial.

The engine already ran the patterns, so "what matched, matched" — the model is
told never to argue with that. No corpus / market / web lookups exist here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

MAX_LUA = 6000        # characters of each pattern's Lua source (mirrors ask.py)
MAX_TOKENS = 2000     # response token cap (mirrors ask.py)
MAX_HISTORY = 12      # keep the last N turns so a long chat stays bounded

# One merged system prompt: it both ANSWERS questions about patterns (analyze /
# compare, in a collector's terms) and, when asked, WRITES or FIXES a pattern
# script. Guardrails from both web prompts are kept.
SYSTEM = """You are the pattern assistant inside Dollar Detective, a desktop tool for a small group
of US banknote collectors. A serial number is checked against a library of "fancy serial"
patterns (a radar reads the same backwards, a ladder counts up, and so on). Each pattern is a
small Lua script that decides whether a serial matches and says which digits to draw boxes, Xs,
group boxes and arcs on; the drawing is most of the value to these collectors. The person
talking to you is a collector, not a programmer.

You are given one or more patterns' Lua, plus what each drew on one example serial. Have a
normal back-and-forth conversation. You do two kinds of things in the same chat:

ANSWERING / COMPARING
- Say plainly what a pattern does, or what the difference between two patterns is, in
  collector's terms ("this one also accepts a serial where only the middle pair is off"),
  not in code terms.
- Quote a concrete serial only when you were given it, and say which pattern matches it.
- If a pattern looks wrong or buggy, say so and what the fix would be.
- If asked which of two to keep, recommend one and why (clearer rule, better drawing, wider
  or narrower coverage); if it's a judgement call, say what it hinges on.

WRITING / FIXING A PATTERN
- When the person asks you to write a new pattern or change one, reply with the COMPLETE Lua
  script in ONE ```lua fenced block, then one short plain paragraph on what you changed and
  what to look for in the drawing. Use this header format exactly:
  --[[
  Pattern: UPPER_SNAKE_NAME
  DisplayName: Short Name
  Description: one line, plain English
  Tier: 1-10 (1 is rarest)
  Examples: ["00012345", "01234555"]
  --]]
  function match(ctx) ... end
- ctx.digits is the 8 digits as a string; ctx.digit_list is those digits as numbers,
  1-indexed in Lua; ctx.full_serial includes the letters. Drawing positions are 0-7 from the
  left (mixing these up with Lua's 1-based indexes is the usual bug); anything outside 0-7 is
  dropped. highlights style is box, x or boxed_x; connectors style is arc, dashed, line, arrow
  or bracket; group_boxes take from/to and an optional thickness. Colors are names that signal
  DISTINCTION, not meaning — one name per group of digits that belong together (blue, orange,
  magenta, red, purple, hotpink, black; gray or charcoal for digits that don't count). The
  sandbox gives string, table, math, pairs, ipairs, tonumber, tostring, type, pcall, error —
  no io/os/require/load/debug, and keep loops bounded by the 8 digits. Always include the
  Examples line. Never shadow Lua built-ins (pairs, type, string, ...) with variable names.

RULES THAT ALWAYS APPLY
- The engine already ran these patterns: what it says matched, matched. Never tell the person
  a pattern shouldn't match a serial the engine matched, or that some other pattern must be
  responsible for a drawing — work out why the rule accepts it. If your reading of the Lua
  disagrees with the engine, you have misread the Lua.
- Only quote serials you were given. You cannot run the patterns, so never invent an example
  or reason one out; if you have no example, describe the rule instead.
- Any price or odds you see comes from a pattern's own header — a figure someone typed in, not
  a market. Quote it as what the library claims, not what a note is worth.
- You cannot look anything up: no web, no eBay, no price guide, no current market. If asked
  what something sells for or what collectors call it elsewhere, say plainly you can't check
  and that anything you said would be from memory and possibly wrong. Do not dress a guess up
  as a recollection.
- Keep answers to a few short paragraphs of plain sentences (no headings, no bullet lists),
  except the required ```lua block when you are writing a script.
"""


def _describe_drawing(drew: dict) -> str:
    """Compact description of what a pattern drew (from get_digit_highlights)."""
    if not drew:
        return "{}"
    boxes = []
    for ph in drew.get("highlights", []):
        for h in ph.get("highlights", []):
            boxes.append({"pos": ph.get("position"),
                          "digit": ph.get("digit"),
                          "color": h.get("color"),
                          "style": h.get("style", "box")})
    slim = {
        "highlights": boxes,
        "connectors": drew.get("connectors", []),
        "group_boxes": drew.get("group_boxes", []),
    }
    return json.dumps(slim)


def build_pattern_context(engine, pattern_names) -> str:
    """Evidence block for one or more patterns: header facts, the Lua source
    (capped at MAX_LUA), and what the engine drew on one example serial from the
    pattern's own Examples. Mirrors the web tool's ask.py._prompt."""
    parts = []
    for name in pattern_names:
        info = engine.get_pattern_info(name)
        if not info:
            continue
        examples = info.get("examples") or []
        sample = examples[0] if examples else None
        block = [
            f"--- {info.get('display_name') or name} "
            f"(library {info.get('library')}, id {name}, tier {info.get('tier')}, "
            f"odds {info.get('odds') or 'n/a'})",
            f"Description: {info.get('description', '')}",
        ]
        if sample:
            drew = engine.get_digit_highlights(sample, [name])
            also = [n for n in engine.classify_simple(sample) if n != name]
            block.append(
                f"On the example serial {sample}, the engine RAN this pattern: it MATCHED "
                f"and drew {_describe_drawing(drew)}.")
            if also:
                more = " …" if len(also) > 8 else ""
                block.append(f"Other patterns that also matched {sample}: "
                             f"{', '.join(also[:8])}{more}.")
        else:
            block.append("This pattern has no example serial in its header.")
        block.append("Lua source:")
        block.append((info.get("script") or "")[:MAX_LUA])
        parts.append("\n".join(block))
    return "\n\n".join(parts)


@dataclass
class ChatResult:
    success: bool
    text: str = ""
    error: str = ""


@dataclass
class PatternChatSession:
    """Multi-turn chat with the configured provider about one or more patterns.

    The API call is isolated in ``_call`` and can be overridden (``transport``)
    for tests so nothing hits the network. ``history`` holds the running
    conversation as [{"role": "user"|"assistant", "content": str}] and is capped
    at MAX_HISTORY turns.
    """
    provider: str
    api_key: str
    model: str
    system: str = SYSTEM
    history: list = field(default_factory=list)
    # Optional override for tests: (system, messages, model, max_tokens) -> text
    transport: Optional[Callable[[str, list, str, int], str]] = None

    def ask(self, user_text: str, context: str = "") -> ChatResult:
        """Send one turn. ``context`` (the pattern evidence block, or an updated
        one when the compare-to set changes) is prepended to this message only;
        it stays in history so later turns still see it without re-sending."""
        if not self.api_key:
            return ChatResult(False, error="No API key configured.")
        if not (user_text or "").strip():
            return ChatResult(False, error="Type a question first.")

        content = (f"{context.strip()}\n\n{user_text.strip()}"
                   if context and context.strip() else user_text.strip())
        messages = self._trimmed_history() + [{"role": "user", "content": content}]
        try:
            text = self._call(messages)
        except Exception as e:  # surfaced to the user, never crashes the dialog
            return ChatResult(False, error=str(e))
        if not (text or "").strip():
            return ChatResult(False, error="The model sent back an empty answer.")
        # Commit both turns to history only on success.
        self.history.append({"role": "user", "content": content})
        self.history.append({"role": "assistant", "content": text.strip()})
        return ChatResult(True, text=text.strip())

    def _trimmed_history(self) -> list:
        if len(self.history) <= MAX_HISTORY * 2:
            return list(self.history)
        return self.history[-MAX_HISTORY * 2:]

    def _call(self, messages: list) -> str:
        if self.transport is not None:
            return self.transport(self.system, messages, self.model, MAX_TOKENS)
        if self.provider == "anthropic":
            return self._call_anthropic(messages)
        if self.provider == "openai":
            return self._call_openai(messages)
        raise RuntimeError(f"Unknown provider: {self.provider}")

    def _call_anthropic(self, messages: list) -> str:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("The 'anthropic' package isn't installed.\n"
                               "Run: pip install anthropic") from e
        client = anthropic.Anthropic(api_key=self.api_key)
        try:
            resp = client.messages.create(
                model=self.model, max_tokens=MAX_TOKENS,
                system=self.system, messages=messages)
        except anthropic.AuthenticationError as e:
            raise RuntimeError("Invalid Anthropic API key.") from e
        except anthropic.RateLimitError as e:
            raise RuntimeError("Rate limit exceeded. Try again in a moment.") from e
        return "\n\n".join(b.text for b in resp.content
                           if getattr(b, "type", "") == "text").strip()

    def _call_openai(self, messages: list) -> str:
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("The 'openai' package isn't installed.\n"
                               "Run: pip install openai") from e
        client = openai.OpenAI(api_key=self.api_key)
        try:
            resp = client.chat.completions.create(
                model=self.model, max_tokens=MAX_TOKENS,
                messages=[{"role": "system", "content": self.system}] + messages)
        except openai.AuthenticationError as e:
            raise RuntimeError("Invalid OpenAI API key.") from e
        except openai.RateLimitError as e:
            raise RuntimeError("Rate limit exceeded. Try again in a moment.") from e
        return (resp.choices[0].message.content or "").strip()


def resolve_provider(settings) -> tuple[str, str, str]:
    """(provider, api_key, model) from AISettings — the configured provider if it
    has a key, else whichever provider does. ('', '', '') if none is configured."""
    ai = settings.ai
    order = []
    if ai.provider:
        order.append(ai.provider)
    order += [p for p in ("anthropic", "openai") if p not in order]
    for provider in order:
        key = ai.anthropic_api_key if provider == "anthropic" else ai.openai_api_key
        if key:
            model = ai.anthropic_model if provider == "anthropic" else ai.openai_model
            return provider, key, model
    return "", "", ""


def is_configured(settings) -> bool:
    return bool(resolve_provider(settings)[1])
