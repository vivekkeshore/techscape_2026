"""Pattern 6 - Supervisor. Isolated workers report; the manager synthesises."""

import json

from llm import ask

TICKET = (
    "Since the SSO cutover we were charged twice in March, and our security team "
    "sees logins from IP ranges nobody recognises. Renewal is in three weeks."
)

WORKERS = {
    "billing": "You own invoices and refunds.",
    "platform": "You own SSO and provisioning.",
}

BOSS = "You triage enterprise support tickets."

# dummy output if no anthropic api key is provided.
PLAN_STUB = (
    '{"assign": [{"worker": "billing", "task": "Is the duplicate charge real?"},'
    '{"worker": "platform", "task": "Did the SSO cutover break anything?"}]}'
)
FINAL_STUB = (
    '{"answer": "One misconfigured cutover created a duplicate org record. That '
    'single fault caused both the double invoice and the unfamiliar logins, which '
    'came from our own provisioning range."}'
)
REPORTS = {
    "billing": "Two charges four days apart. A duplicate org entity was billed twice.",
    "platform": "SCIM still pointed at the old directory and created a second org.",
}

plan = json.loads(
    ask(
        f"Ticket: {TICKET}\n\nAssign work to {list(WORKERS)}.\n"
        'JSON only: {"assign": [{"worker": str, "task": str}]}',
        system=BOSS,
        stub=PLAN_STUB,
    )
)

reports = {}
for job in plan["assign"]:
    reports[job["worker"]] = ask(
        f"Ticket: {TICKET}\n\nYour task: {job['task']}",   # ENTIRE context
        system=WORKERS[job["worker"]],
        stub=REPORTS[job["worker"]],
    )
    print(f"{job['worker']:<10} {reports[job['worker']]}")

answer = json.loads(
    ask(
        f"Ticket: {TICKET}\nWorker reports: {reports}\n\n"
        'Write the answer to the customer. JSON only: {"answer": str}',
        system=BOSS,
        stub=FINAL_STUB,
    )
)["answer"]

print(f"\nRESOLUTION: {answer}")
