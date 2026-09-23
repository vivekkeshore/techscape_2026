"""Pattern 2: Voting. Independent reviewers, tallied by code."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from llm import ask

CLAUSE = "Vendor's liability is unlimited. The contract auto-renews for 24 months."

# Stub decisions, one per lens/role, used when no API key is set.
LENSES = {
    "liability": "BLOCK",
    "privacy": "ACCEPT",
    "finance": "NEGOTIATE",
    "commercial": "NEGOTIATE",
    "regulatory": "ACCEPT",
}  # dummy output if no anthropic api key is provided.


def review(lens):
    return ask(
        f"Clause: {CLAUSE}\n\nReply with one word: BLOCK, NEGOTIATE or ACCEPT.",
        system=f"You review contracts through one lens only: {lens}.",
        stub=LENSES[lens],
    ).upper()


with ThreadPoolExecutor() as pool:  # independent calls, multiprocessing can be used as well.
    votes = list(pool.map(review, LENSES))

tally = Counter(votes)
for lens, vote in zip(LENSES, votes):
    print(f"{lens:<12} {vote}")

top, count = tally.most_common(1)[0]
decision = top if count >= 3 else "ESCALATE"  # the policy is code, not a prompt
print(f"\ntally={dict(tally)} -> {decision}")
