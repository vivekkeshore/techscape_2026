"""Pattern 5: Collaboration. Each peer does its part and names the next peer."""

import json

from llm import ask

PEERS = {
    "copywriter": "You write short social copy.",
    "art_director": "You turn copy into an image direction.",
    "compliance": "You check claims and reject unsupported ones.",
}

# dummy output if no anthropic api key is provided.
STUBS = [
    '{"work": "Headline: One cup. Every refill.", "to": "art_director"}',
    '{"work": "Overhead shot of one cup, faint coffee rings, morning light.",'
    ' "to": "compliance"}',
    '{"work": "REJECTED: plastic-free is unsupported, the lid gasket is silicone.",'
    ' "to": "copywriter"}',
    '{"work": "Headline: Your 4th coffee of the day, same cup.", "to": "art_director"}',
    '{"work": "2x2 grid: the same cup at four times of one workday.",'
    ' "to": "compliance"}',
    '{"work": "APPROVED. Every claim is substantiated.", "to": "done"}',
]

thread, peer = [], "copywriter"

for handoff in range(8):  # the cycle guard belongs in code
    reply = json.loads(
        ask(
            "Campaign: spring launch for a refillable coffee cup.\n"
            f"Thread so far: {thread or 'nothing yet'}\n\n"
            f"Do your part, then pick who goes next ({', '.join(PEERS)}, or done).\n"
            'JSON only: {"work": str, "to": str}',
            system=PEERS[peer],
            stub=STUBS[min(handoff, len(STUBS) - 1)],
        )
    )
    print(f"{handoff + 1}. {peer:<13} {reply['work']}\n   -> {reply['to']}")
    thread.append(f"{peer}: {reply['work']}")

    if reply["to"] == "done":
        break
    peer = reply["to"]  # note: a peer can hand work BACKWARDS
