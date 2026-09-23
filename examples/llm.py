"""
Real Claude call will happen if ANTHROPIC_API_KEY is set, otherwise the "stub" passed at the call
site. So every example in this folder runs with no key and no cost.
"""

import os

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
API_KEY = os.getenv("ANTHROPIC_API_KEY", None)


def ask(prompt: str, system: str | None = None, stub: str = "") -> str:
    """One prompt in, one string out."""
    if not API_KEY:
        return stub

    # Not a fan of inline imports, but keeping it here so that
    # the examples can also run offline without installing the package.
    import anthropic
    anthropic_client = anthropic.Anthropic(api_key=API_KEY)

    kwargs = {
        "model": MODEL,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    reply = anthropic_client.messages.create(**kwargs)
    return "".join(b.text for b in reply.content if b.type == "text").strip()
