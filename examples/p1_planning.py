"""Pattern 1: Planning. The model plans once and the code executes the plan."""

import json

from llm import ask

GOAL = "Ship a REST API for expense reports"

# dummy output if no anthropic api key is provided.
PLAN_STUB = '{"steps": ["design the data model", "write the endpoints", "add tests"]}'


def make_plan(goal):
    reply = ask(
        f"Break this into 3 ordered steps. JSON only: {{'steps': [str]}}\n\nGoal: {goal}",
        system="You are a staff engineer.",
        stub=PLAN_STUB,
    )
    return json.loads(reply)["steps"]


def build(step, done):
    return ask(
        f"Already done: {done or 'nothing'}\nNow do exactly this step: {step}",
        system="You are an engineer. Do one step. Output the artifact only.",
        stub=f"<artifact for: {step}>",
    )


steps = make_plan(GOAL) # one call decides the shape of the work
print("PLAN:", steps)

done = []
for i, step in enumerate(steps, 1):  # this loop is Python code. The model has no say.
    print(f"\n[{i}/{len(steps)}] {step}\n    {build(step, done)}")
    done.append(step)
