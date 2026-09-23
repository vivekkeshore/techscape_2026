"""Pattern 3b - ReAct against a REAL environment

The other examples simulate their world: `INVENTORY` is a dict, every tool
returns a canned string, so the agent can only ever "discover" what was planted
for it. This one is different.

    REAL       The repository under test. It is written to a temp directory and
               it genuinely contains a bug. `run_tests` really shells out to
               unittest. `read_file`, `grep` and `apply_fix` really touch disk.
               The fix is really applied and the suite is really re-run.
    THE MODEL  A live Claude call if credentials are available, otherwise a
               recorded session (REPLAY below) so the example still runs free
               and offline. Only the agent's *decisions* are ever replayed.

Live is the better demonstration - the model finds the bug with no script, and
the path can differ between runs. The replay exists so the example works with no
key at all, and it comes with a guard: a recording is only valid against the
state it was recorded on, so `assert_replay_premise()` re-runs the suite before
the loop and aborts if the fixture no longer fails the way it did. That guard is
not decoration - without it, a pre-fixed fixture makes the first matching rule
the *closing* one, and the agent reports a fix it never made. A replayed session
will narrate work it did not do unless something checks its premise.

Why a failing test is the right ReAct example
    The traceback is an assertion failure, not an exception, so it names the
    test - not the bug. Two modules are innocent and one is guilty, and the only
    way to tell is to go and look. No plan written in advance survives that,
    which is exactly when a Thought -> Action -> Observation loop earns its cost.
    Note also that two of the three tests pass: that is why this bug would have
    shipped.

Run it:
    python 03b_react_real_tools.py          # sandbox is deleted afterwards
    KEEP_SANDBOX=1 python 03b_react_real_tools.py   # keep it to poke around
    STEP=1 python 03b_react_real_tools.py   # on stage: press Enter between steps
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import console as ui
from llm import Agent, backend_label, extract_json, is_live

MAX_STEPS = 10
TIMEOUT_SECONDS = 60
STEP_THROUGH = bool(os.environ.get("STEP"))

TASK = (
    "The checkout test suite has a failing test. Find the cause, fix it, and "
    "prove the suite passes. Do not change the tests."
)

# ---------------------------------------------------------------------------
# The fixture repository. A real, small, buggy package.
#
# The bug: flat coupons are subtracted AFTER GST is added, so the customer is
# charged tax on money they never paid. Percentage coupons take a different
# branch and are correct - which is why only one of the three tests fails.
# ---------------------------------------------------------------------------

FIXTURE: dict[str, str] = {
    "shop/__init__.py": "",
    "shop/pricing.py": '''"""Line pricing and coupon arithmetic."""

from decimal import Decimal, ROUND_HALF_UP


def money(value) -> Decimal:
    """Round to paise, half-up, the way an invoice does."""
    return Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def line_subtotal(item: dict) -> Decimal:
    return money(Decimal(str(item["price"])) * item["qty"])


def apply_coupon(amount: Decimal, coupon: dict) -> Decimal:
    """Subtract a coupon from an amount.

    Order-agnostic on purpose: this function does not know whether tax has been
    added yet. The caller decides when to apply it.
    """
    kind = coupon["type"]
    if kind == "flat":
        return money(max(Decimal("0"), amount - Decimal(str(coupon["value"]))))
    if kind == "percent":
        return money(amount * (Decimal("1") - Decimal(str(coupon["value"])) / 100))
    raise ValueError(f"unknown coupon type {kind!r}")
''',
    "shop/tax.py": '''"""GST."""

from decimal import Decimal

from .pricing import money

GST_RATE = Decimal("0.18")


def add_gst(amount: Decimal) -> Decimal:
    return money(amount * (Decimal("1") + GST_RATE))
''',
    "shop/cart.py": '''"""Checkout totals."""

from decimal import Decimal

from .pricing import apply_coupon, line_subtotal
from .tax import add_gst


def checkout_total(items: list[dict], coupon: dict | None = None) -> Decimal:
    """Total payable for a cart, in rupees."""
    subtotal = sum((line_subtotal(item) for item in items), Decimal("0"))

    if coupon is not None and coupon["type"] == "percent":
        return add_gst(apply_coupon(subtotal, coupon))

    taxed = add_gst(subtotal)
    if coupon is not None:
        taxed = apply_coupon(taxed, coupon)
    return taxed
''',
    "tests/__init__.py": "",
    "tests/test_checkout.py": '''"""Checkout total expectations, signed off by finance."""

import unittest
from decimal import Decimal

from shop.cart import checkout_total

ITEMS = [{"sku": "TL-2", "price": "1000.00", "qty": 1}]


class CheckoutTotals(unittest.TestCase):
    def test_no_coupon_adds_gst(self):
        # 1000 + 18% = 1180
        self.assertEqual(checkout_total(ITEMS), Decimal("1180.00"))

    def test_percent_coupon_applies_before_gst(self):
        # (1000 - 10%) + 18% = 1062
        coupon = {"type": "percent", "value": 10}
        self.assertEqual(checkout_total(ITEMS, coupon), Decimal("1062.00"))

    def test_flat_coupon_applies_before_gst(self):
        # (1000 - 100) + 18% = 1062. GST is owed on the discounted amount only.
        coupon = {"type": "flat", "value": 100}
        self.assertEqual(checkout_total(ITEMS, coupon), Decimal("1062.00"))


if __name__ == "__main__":
    unittest.main()
''',
}

SANDBOX = Path(tempfile.mkdtemp(prefix="react_checkout_"))


def build_sandbox() -> None:
    for relative, content in FIXTURE.items():
        path = SANDBOX / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


# ---------------------------------------------------------------------------
# Real tools. Every one of these does actual work.
# ---------------------------------------------------------------------------


def _resolve(relative: str) -> Path:
    """Confine every path to the sandbox. Tool inputs are untrusted."""
    target = (SANDBOX / relative).resolve()
    if not target.is_relative_to(SANDBOX.resolve()):
        raise ValueError(f"path escapes the sandbox: {relative}")
    return target


def run_tests() -> str:
    """run_tests() - run the suite and report the outcome."""
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
        cwd=SANDBOX,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )
    output = (proc.stdout + proc.stderr).strip()
    passed = len(re.findall(r"\.\.\. ok\b", output))
    failed = len(re.findall(r"\.\.\. (FAIL|ERROR)\b", output))
    tail = "\n".join(output.splitlines()[-24:])
    return f"summary: {passed} passed, {failed} failed\n{tail}"


def list_files() -> str:
    """list_files() - every Python file in the project."""
    files = sorted(
        str(p.relative_to(SANDBOX)) for p in SANDBOX.rglob("*.py") if p.is_file()
    )
    return "\n".join(files)


def read_file(path: str, start: int = 1, end: int = 0) -> str:
    """read_file(path, start, end) - file contents with line numbers."""
    lines = _resolve(path).read_text().splitlines()
    last = len(lines) if not end else min(int(end), len(lines))
    first = max(1, int(start))
    return "\n".join(f"{n:>4} | {lines[n - 1]}" for n in range(first, last + 1))


def grep(pattern: str) -> str:
    """grep(pattern) - regex search across the project's Python files."""
    matcher = re.compile(pattern)
    hits = []
    for file in sorted(SANDBOX.rglob("*.py")):
        for number, line in enumerate(file.read_text().splitlines(), 1):
            if matcher.search(line):
                hits.append(f"{file.relative_to(SANDBOX)}:{number}: {line.strip()}")
    return "\n".join(hits) if hits else f"no matches for {pattern!r}"


def apply_fix(path: str, old: str, new: str) -> str:
    """apply_fix(path, old, new) - exact string replacement; must be unique."""
    target = _resolve(path)
    source = target.read_text()
    occurrences = source.count(old)
    if occurrences != 1:
        return (
            f"refused: found {occurrences} occurrences of that snippet in {path}. "
            "An edit has to match exactly once."
        )
    target.write_text(source.replace(old, new, 1))
    return f"replaced 1 occurrence in {path}. The file now reads:\n{new}"


TOOLS = {
    fn.__name__: fn
    for fn in (run_tests, list_files, read_file, grep, apply_fix)
}

EXPECTED_START = "summary: 2 passed, 1 failed"


def assert_replay_premise() -> None:
    """Check the fixture still fails the way this example assumes.

    Under a live model this is a plain fixture sanity check. Under the replay it
    is load-bearing: without it, a pre-fixed fixture makes the first rule that
    matches the *closing* one, and the agent cheerfully reports a fix it never
    made. Verified, not assumed - that behaviour is what this check exists to
    stop, and it is worth knowing that a replayed run will narrate work it did
    not do unless something checks its premise.
    """
    observed = run_tests().splitlines()[0]
    if observed == EXPECTED_START:
        return
    if is_live():
        raise SystemExit(
            f"\nFixture is not broken as expected: got {observed!r}, wanted "
            f"{EXPECTED_START!r}. The example has nothing for the agent to find."
        )
    raise SystemExit(
        f"\nStale replay. The recorded session starts from a red suite "
        f"({EXPECTED_START!r}) but this run reports {observed!r}.\n"
        "The fixture changed, so the recorded decisions no longer describe "
        "this repository. Re-run against a live model (export ANTHROPIC_API_KEY) "
        "rather than trusting the transcript."
    )

TOOL_SPEC = "\n".join(f"- {fn.__doc__}" for fn in TOOLS.values())

SYSTEM = f"""You are a debugging agent working in a small Python repository.

Each turn emit exactly one of:

Thought: <one line>
Action: {{"tool": "<name>", "args": {{...}}}}

...or, once the suite is green:

Thought: <one line>
Final Answer: <cause, fix, and evidence it passes>

Tools:
{TOOL_SPEC}

Never edit the tests. Never guess a tool result - it comes back to you."""

# ---------------------------------------------------------------------------
# The recorded session. Each rule is keyed on REAL tool output, and rules are
# checked in order, so the later-state rules come first.
# ---------------------------------------------------------------------------

REPLAY = [
    (
        "summary: 3 passed, 0 failed",
        """
Thought: All three tests pass now, so the fix is proven rather than asserted.
Final Answer: Cause - shop/cart.py added GST first and only then subtracted a
flat coupon, so the customer was taxed on the 100 rupees they never paid: 1000
-> 1180 -> 1080, against an expected 1062. Percentage coupons took a separate,
correct branch above it, which is why only one of the three tests failed and why
this shipped. Fix - discount the subtotal first, then tax it once, for both
coupon kinds. Evidence - the suite went from 2 passed / 1 failed to 3 passed / 0
failed. Note for review: the `coupon["type"] == "percent"` branch is now
redundant since the general path handles it, but removing it is a refactor
beyond this fix.
""",
    ),
    (
        "replaced 1 occurrence",
        """
Thought: The edit is in. Re-run the suite - a fix I have not seen pass is not a fix.
Action: {"tool": "run_tests", "args": {}}
""",
    ),
    # Raw string: the \n escapes below must reach json.loads intact.
    (
        "order-agnostic",
        r"""
Thought: Confirmed - apply_coupon is order-agnostic, so the bug is the call
order in cart.py, not the coupon maths. Discount first, then tax once.
Action: {"tool": "apply_fix", "args": {"path": "shop/cart.py", "old": "    taxed = add_gst(subtotal)\n    if coupon is not None:\n        taxed = apply_coupon(taxed, coupon)\n    return taxed", "new": "    if coupon is not None:\n        subtotal = apply_coupon(subtotal, coupon)\n    return add_gst(subtotal)"}}
""",
    ),
    (
        "taxed = apply_coupon",
        """
Thought: There it is - the percent branch discounts before add_gst, but the path
below taxes the subtotal first and subtracts the flat coupon afterwards. Before
I edit, check whether apply_coupon itself expects a pre-tax amount.
Action: {"tool": "read_file", "args": {"path": "shop/pricing.py"}}
""",
    ),
    (
        "shop/tax.py",
        """
Thought: Three modules. The total is assembled in cart.py, so the ordering
mistake is most likely there rather than in pricing or tax.
Action: {"tool": "read_file", "args": {"path": "shop/cart.py"}}
""",
    ),
    (
        "summary: 2 passed, 1 failed",
        """
Thought: An assertion failure, so the traceback names the test, not the bug -
expected 1062.00, got 1080.00, a difference of 18.00 which is exactly 18% of the
100 rupee coupon. The other two tests pass. I need to see the code layout.
Action: {"tool": "list_files", "args": {}}
""",
    ),
    (
        "!summary:",
        """
Thought: Start from the actual failure rather than from a guess about it.
Action: {"tool": "run_tests", "args": {}}
""",
    ),
]

agent = Agent(name="debug-agent", system=SYSTEM, script=REPLAY)


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------


def parse(reply: str) -> tuple[str, dict | None, str, str]:
    """Return (thought, action, final_answer, parse_error).

    A live model will occasionally deviate from the format. That is a normal
    turn to recover from, not a crash: the caller feeds the error back as an
    observation and the agent tries again.
    """
    thought = _first(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer):|\Z)", reply)
    final = _first(r"Final Answer:\s*(.+)", reply)
    if final:
        return thought, None, final, ""
    if "Action:" not in reply:
        return thought, None, "", (
            "error: your reply contained neither an Action nor a Final Answer. "
            'Emit exactly: Action: {"tool": "<name>", "args": {...}}'
        )
    _, _, after = reply.partition("Action:")
    try:
        action = extract_json(after, who="action")
    except ValueError:
        return thought, None, "", (
            "error: your Action was not valid JSON. Emit exactly: "
            'Action: {"tool": "<name>", "args": {...}}'
        )
    if not isinstance(action, dict) or "tool" not in action:
        return thought, None, "", (
            'error: the Action JSON needs a "tool" key naming one of: '
            f"{', '.join(TOOLS)}"
        )
    return thought, action, "", ""


def _first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.S)
    return " ".join(match.group(1).split()) if match else ""


def invoke(action: dict) -> str:
    name = action.get("tool", "")
    fn = TOOLS.get(name)
    if fn is None:
        return f"error: no tool named {name!r}. Available: {', '.join(TOOLS)}"
    try:
        return fn(**action.get("args", {}))
    except Exception as exc:  # a real tool can really fail; report it, don't crash
        return f"error: {name} raised {type(exc).__name__}: {exc}"


def describe(action: dict) -> str:
    args = action.get("args", {})
    shown = ", ".join(
        f"{k}={v!r}" if len(str(v)) < 40 else f"{k}=<{len(str(v))} chars>"
        for k, v in args.items()
    )
    return f"{action.get('tool')}({shown})"


def pause() -> None:
    """On stage, wait for Enter so the room can read each step."""
    if STEP_THROUGH:
        input("\n    [Enter] next step ")


def run() -> None:
    build_sandbox()
    ui.title(
        "PATTERN 3b - ReAct AGAINST A REAL REPOSITORY",
        f"tools: real (disk + unittest)   |   "
        f"decisions: {'the model' if is_live() else 'replayed'}   |   "
        f"llm: {backend_label()}",
    )
    ui.note(f"Task: {TASK}")
    ui.note(f"Sandbox: {SANDBOX}")

    transcript = f"Task: {TASK}\n"

    try:
        # A replay that no longer matches its environment must fail loudly.
        assert_replay_premise()
        pause()

        for step in range(1, MAX_STEPS + 1):
            reply = agent.ask(transcript + "\nWhat is your next step?")
            thought, action, final, parse_error = parse(reply)

            ui.phase(f"STEP {step}")
            ui.kv("Thought", thought, indent=4, pad=11)

            if parse_error:
                # Malformed turn: hand the error back and let the agent retry.
                ui.kv("Loop", parse_error, indent=4, pad=11)
                transcript += f"\n{reply.strip()}\nObservation: {parse_error}\n"
                continue

            if action is None:
                ui.kv("Answer", final, indent=4, pad=11)
                pause()
                verify = run_tests()
                ui.phase("INDEPENDENT VERIFICATION (not the agent's word for it)")
                ui.body(verify.splitlines()[0], indent=4)

                if is_live():
                    closing = (
                        f"Closed in {step} steps, {agent.calls} model calls.\n\n"
                        "Fully autonomous: the model chose every tool call, and the "
                        "tools really ran unittest, really read disk, and really "
                        "edited shop/cart.py. The verification line above re-ran the "
                        "suite after the loop exited - the one number here that does "
                        "not depend on the agent's own account of its work.\n\n"
                        "Re-run it and the path may differ. That variance is what an "
                        "autonomous pattern buys and costs."
                    )
                else:
                    closing = (
                        f"Closed in {step} steps.\n\n"
                        "Everything the agent saw was real: the traceback came out of "
                        "a real unittest run, the file contents off disk, and the fix "
                        "was really written to shop/cart.py - the verification above "
                        "re-ran the suite after the loop exited.\n\n"
                        "Only the choice of which tool to call next was replayed, and "
                        "assert_replay_premise() checked before the loop that this "
                        "repository still fails the way the recording assumed. Drop "
                        "that guard and a pre-fixed fixture makes the agent claim a "
                        "fix it never applied - which is the real lesson about "
                        "replayed runs, and about believing an agent's own summary.\n\n"
                        "Export ANTHROPIC_API_KEY to watch a live model find this "
                        "itself, with no script."
                    )
                ui.result(closing)
                return

            observation = invoke(action)
            ui.kv("Action", describe(action), indent=4, pad=11)
            ui.body("Observation:", indent=4)
            ui.body(observation, indent=6)

            transcript += (
                f"\nThought: {thought}\nAction: {json.dumps(action)}\n"
                f"Observation: {observation}\n"
            )
            pause()

        ui.result(
            f"Hit the {MAX_STEPS}-step ceiling with the suite still red. Hand the "
            "transcript to a human rather than raising the cap."
        )
    finally:
        if os.environ.get("KEEP_SANDBOX"):
            print(f"\nSandbox kept at: {SANDBOX}")
        else:
            shutil.rmtree(SANDBOX, ignore_errors=True)


if __name__ == "__main__":
    run()
