"""Pattern 3: ReAct. The model picks the next tool call every turn."""

import re

from llm import ask

STOCK = {"SKU-4417": 6}

TOOLS = {
    "check_stock": lambda sku: f"{sku}: {STOCK.get(sku, 0)} units on hand",
    "reorder": lambda sku, qty: f"created a PO for {qty} units of {sku}",
}

SYSTEM = """Work in a loop. Each turn emit exactly one line, either:
Action: tool_name(arg, ...)
Done: <your answer>

Tools: check_stock(sku), reorder(sku, qty)"""

STUBS = [
    "Action: check_stock(SKU-4417)",
    "Action: reorder(SKU-4417, 120)",
    "Done: Only 6 units were left before a promo, so I ordered 120 more.",
]  # dummy output if no anthropic api key is provided.

transcript = "Task: SKU-4417 may stock out before the weekend promo. Fix it."

for step in range(6):  # the step cap belongs in code, not the prompt
    stub = STUBS[step] if step < len(STUBS) else "Done: out of steps"
    reply = ask(transcript, system=SYSTEM, stub=stub)
    print(reply)

    if reply.startswith("Done:"):
        break

    name, raw = re.match(r"Action:\s*(\w+)\((.*)\)", reply).groups()
    result = TOOLS[name](*[a.strip() for a in raw.split(",") if a.strip()])
    print(f"Observation: {result}\n")
    transcript += f"\n{reply}\nObservation: {result}"
