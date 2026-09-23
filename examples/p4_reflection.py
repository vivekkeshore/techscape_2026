"""Pattern 4: Reflection. A critic decides when the work is good enough."""

from llm import ask

BRIEF = "Meta description for a rechargeable trail headlamp. Max 155 characters."
LIMIT = 155

# dummy output if no anthropic api key is provided.
DRAFTS = [
    "Discover the best-in-class TrailLight 2, a revolutionary rechargeable trail "
    "headlamp built for serious night runners, with a huge 40-hour burn time and "
    "an ultra-bright beam. Buy yours today.",
    "TrailLight 2: a rechargeable trail headlamp with 40-hour burn time and a "
    "400-lumen beam for night runs. See sizes and pricing.",
]
CRITIQUES = ["FAIL: 'best-in-class' is filler and the CTA is generic.", "PASS"]


def lint(text):  # never pay a model to count characters
    return [f"{len(text)} characters, limit is {LIMIT}"] if len(text) > LIMIT else []


issues = []
for round_no in range(3):  # the ceiling is code, the critic decides
    i = min(round_no, len(DRAFTS) - 1)

    draft = ask(
        f"{BRIEF}\nFix these issues: {issues or 'none'}",
        system="Write the description only.",
        stub=DRAFTS[i],
    )
    verdict = ask(
        f"Brief: {BRIEF}\nDraft: {draft}\n\nReply PASS, or FAIL: <reason>.",
        system="You are a hostile reviewer. Find what is wrong.",
        stub=CRITIQUES[i],
    )

    issues = lint(draft) + ([verdict] if verdict.startswith("FAIL") else [])
    print(f"\nround {round_no + 1} ({len(draft)} chars): {draft}")
    print(f"  -> {issues or 'APPROVED'}")

    if not issues:
        break
