"""Shared LLM plumbing for the pattern examples.

Backend selection is automatic:

    live Claude   if the `anthropic` SDK is importable AND credentials resolve
    mock          otherwise (deterministic, offline, free)

Override with `AGENTIC_LLM`:

    AGENTIC_LLM=auto     (default) prefer live, fall back to mock
    AGENTIC_LLM=claude   require live; fail loudly rather than fall back
    AGENTIC_LLM=mock     force the offline mock even if a key is present

Credentials are *detected* the way the SDK resolves them, but detection never
blocks a live call: with AGENTIC_LLM=claude the SDK is treated as authoritative
and gets to try. Detection only decides the default.

    ANTHROPIC_MODEL / AGENTIC_EFFORT   optional overrides for the live backend

Nothing in the pattern files knows which backend is active, and that is the
point: these patterns are about *who decides the control flow*, not about which
model answers. Swapping the backend must not change the orchestration code.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

# Claude Opus 5 is the current top-tier model. Thinking is on by default on it,
# so there is no `thinking` parameter to configure here.
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")

# Non-streaming requests should stay under ~16k output tokens or they risk
# hitting the SDK's HTTP timeout. Anything larger belongs on `.stream()`.
NON_STREAMING_MAX_TOKENS = 16_000

# Optional: low | medium | high | xhigh | max. Unset means the API default (high).
EFFORT = os.environ.get("AGENTIC_EFFORT", "").strip().lower() or None

_MODE = os.environ.get("AGENTIC_LLM", "auto").strip().lower()
_LIVE_ALIASES = {"claude", "real", "anthropic", "live"}


class ScriptMiss(RuntimeError):
    """The mock backend had no scripted reply for the prompt it was given."""


# ---------------------------------------------------------------------------
# Credential detection. Mirrors the SDK's own resolution order:
#   ANTHROPIC_API_KEY -> ANTHROPIC_AUTH_TOKEN -> `ant auth login` profile
#   -> workload identity federation -> default profile on disk
# An unset ANTHROPIC_API_KEY does NOT mean there are no credentials.
# ---------------------------------------------------------------------------


def _config_dir() -> Path:
    override = os.environ.get("ANTHROPIC_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", "~")).expanduser() / "Anthropic"
    return Path("~/.config/anthropic").expanduser()


def _wif_configured() -> bool:
    required = (
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_ORGANIZATION_ID",
        "ANTHROPIC_SERVICE_ACCOUNT_ID",
    )
    token = os.environ.get("ANTHROPIC_IDENTITY_TOKEN_FILE") or os.environ.get(
        "ANTHROPIC_IDENTITY_TOKEN"
    )
    return bool(token) and all(os.environ.get(k) for k in required)


def detect_credentials() -> tuple[bool, str]:
    """Return (found, human-readable source)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True, "ANTHROPIC_API_KEY"
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True, "ANTHROPIC_AUTH_TOKEN"
    if _wif_configured():
        return True, "workload identity federation"

    profiles = sorted((_config_dir() / "credentials").glob("*.json"))
    if profiles:
        wanted = os.environ.get("ANTHROPIC_PROFILE")
        names = [p.stem for p in profiles]
        if wanted and wanted not in names:
            return False, f"ANTHROPIC_PROFILE={wanted} has no stored credential"
        return True, f"ant auth profile ({wanted or names[0]})"

    return False, "no Anthropic credentials found"


def _credential_warnings() -> list[str]:
    """Misconfigurations that authenticate as the wrong thing, or not at all."""
    problems = []
    if "ANTHROPIC_API_KEY" in os.environ and not os.environ["ANTHROPIC_API_KEY"]:
        problems.append(
            "ANTHROPIC_API_KEY is set but empty. It still wins the credential "
            "precedence slot and authenticates with an empty key - unset it "
            "entirely rather than blanking it."
        )
    if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        problems.append(
            "Both ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are set. The SDK "
            "sends both headers and the API rejects the request - unset one."
        )
    return problems


def _sdk_available() -> bool:
    return importlib.util.find_spec("anthropic") is not None


def _resolve_backend() -> tuple[str, str]:
    """Decide the backend once, at import. Returns (kind, reason)."""
    if _MODE == "mock":
        return "mock", "forced by AGENTIC_LLM=mock"

    if _MODE in _LIVE_ALIASES:
        if not _sdk_available():
            raise RuntimeError(
                "AGENTIC_LLM=claude needs the SDK. Run: pip install anthropic"
            )
        found, source = detect_credentials()
        # Do not block: the SDK may resolve a credential this check cannot see.
        return "claude", source if found else f"{source} - letting the SDK try anyway"

    if _MODE != "auto":
        raise RuntimeError(
            f"Unknown AGENTIC_LLM={_MODE!r}. Use 'auto', 'claude', or 'mock'."
        )

    if not _sdk_available():
        found, source = detect_credentials()
        if found:
            # Most likely confusion: a key is set but nothing can use it.
            return "mock", (
                f"credentials found ({source}) but the anthropic SDK is not "
                "installed - pip install anthropic to go live"
            )
        return "mock", "anthropic SDK not installed (pip install anthropic)"
    found, source = detect_credentials()
    if not found:
        return "mock", source
    return "claude", source


BACKEND_KIND, BACKEND_REASON = _resolve_backend()

for _problem in _credential_warnings():
    print(f"[llm] warning: {_problem}", file=sys.stderr)


def is_live() -> bool:
    return BACKEND_KIND == "claude"


def backend_label() -> str:
    if is_live():
        via = f"via {BACKEND_REASON}"
        if os.environ.get("ANTHROPIC_BASE_URL"):
            via += f", base_url={os.environ['ANTHROPIC_BASE_URL']}"
        effort = f", effort={EFFORT}" if EFFORT else ""
        return f"{DEFAULT_MODEL} live ({via}{effort})"
    return f"mock, offline and deterministic ({BACKEND_REASON})"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _normalise(script: Iterable[tuple[Sequence[str] | str, str]]):
    rules = []
    for keys, reply in script:
        if isinstance(keys, str):
            keys = (keys,)
        rules.append((tuple(keys), reply))
    return rules


class MockBackend:
    """A deterministic stand-in for a model call.

    A *script* is a list of ``(keywords, reply)`` rules. The first rule whose
    keywords all match the prompt wins. A keyword prefixed with ``!`` must
    *not* appear in the prompt, which is how a scripted agent can answer
    differently on its first turn than on its later ones.
    """

    label = "mock"

    def __init__(self, name: str, script: Iterable[tuple[Sequence[str] | str, str]] = ()):
        self.name = name
        self.calls = 0
        self._rules = _normalise(script)

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        haystack = f"{system or ''}\n{prompt}".lower()
        for keys, reply in self._rules:
            if all(self._matches(key, haystack) for key in keys):
                return reply.strip()
        raise ScriptMiss(
            f"{self.name}: no scripted reply matched this prompt. Either add a "
            f"rule to its script or run against a live model (export "
            f"ANTHROPIC_API_KEY, or AGENTIC_LLM=claude).\n"
            f"--- prompt ---\n{prompt[:600]}"
        )

    @staticmethod
    def _matches(key: str, haystack: str) -> bool:
        if key.startswith("!"):
            return key[1:].lower() not in haystack
        return key.lower() in haystack


class ClaudeBackend:
    """Real Claude calls. One ``generate()`` is one Messages API request."""

    label = "claude"

    def __init__(
        self,
        name: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = NON_STREAMING_MAX_TOKENS,
        **_ignored: Any,
    ):
        import anthropic  # presence already checked in _resolve_backend

        self.name = name
        self.calls = 0
        self._model = model
        self._max_tokens = max_tokens
        # Credentials resolve from ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
        # `ant auth login` profile - nothing to pass in here.
        self._client = anthropic.Anthropic()

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            # Opus 5's safety classifiers can decline a request outright. With
            # `fallbacks` the API re-runs it server-side on Anthropic's
            # recommended substitute instead of handing back an empty response.
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
        if system:
            kwargs["system"] = system
        if EFFORT:
            kwargs["output_config"] = {"effort": EFFORT}

        try:
            response = self._client.beta.messages.create(**kwargs)
        except TypeError:
            # Installed SDK predates the `fallbacks` parameter. Drop it rather
            # than fail: a refusal then surfaces below instead of being retried.
            for key in ("betas", "fallbacks"):
                kwargs.pop(key, None)
            response = self._client.beta.messages.create(**kwargs)

        # Always check stop_reason before reading content: on a refusal the
        # content list can be empty and `content[0]` would blow up.
        if response.stop_reason == "refusal":
            raise RuntimeError(f"{self.name}: the model declined this request")

        return "".join(b.text for b in response.content if b.type == "text").strip()


def make_backend(name: str, script: Iterable[tuple[Sequence[str] | str, str]] = ()):
    return ClaudeBackend(name) if is_live() else MockBackend(name, script)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """A named LLM caller: a role, a system prompt, and a way to ask questions.

    Deliberately thin. Everything that makes a pattern a pattern lives in the
    orchestration code of each example, not in here.
    """

    def __init__(
        self,
        name: str,
        system: str | None = None,
        script: Iterable[tuple[Sequence[str] | str, str]] = (),
    ):
        self.name = name
        self.system = system
        self.backend = make_backend(name, script)

    @property
    def calls(self) -> int:
        return self.backend.calls

    def ask(self, prompt: str) -> str:
        return self.backend.generate(prompt, self.system)

    def ask_json(self, prompt: str) -> Any:
        """Ask for JSON and parse it, tolerating fences and surrounding prose.

        A live model occasionally adds a preamble, so one corrective retry is
        cheaper than failing the whole run. Against the real API you would
        usually reach for structured outputs instead
        (``output_config={"format": {"type": "json_schema", ...}}``), which
        removes the guesswork entirely - kept out of here so the examples read
        the same under both backends.
        """
        raw = self.ask(prompt)
        try:
            return extract_json(raw, who=self.name)
        except ValueError:
            if not is_live():
                raise
            raw = self.ask(
                prompt
                + "\n\nYour previous reply was not valid JSON. Reply with the "
                "JSON object only - no prose, no code fence."
            )
            return extract_json(raw, who=self.name)


def extract_json(raw: str, who: str = "response") -> Any:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"{who}: could not parse JSON from:\n{raw[:400]}")
