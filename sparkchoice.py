"""
SparkChoice — smallest system that chooses the best next use of AI
and expresses it as a concrete artifact-producing action.

The loop:
  1. Takes a goal + current state (what exists so far)
  2. Generates candidate next-actions (each produces a tangible artifact)
  3. Scores each on: dependency (does it unblock others?), risk (does it
     reduce uncertainty?), readiness (can we do it right now?), impact
  4. Emits the most prudent choice — not the flashiest, the one that
     makes everything else easier or possible
"""

from __future__ import annotations
import json
import os
import sys
from dataclasses import dataclass, field, asdict

import anthropic


# ── Domain ──────────────────────────────────────────────────────────────

@dataclass
class Scores:
    """Why this action was chosen over alternatives."""
    unblocks: int      # 1-5: how many downstream actions does this enable?
    reduces_risk: int  # 1-5: does this retire uncertainty or validate assumptions?
    readiness: int     # 1-5: do we have everything needed to do this now?
    impact: int        # 1-5: how much does this move the goal forward?

    @property
    def prudence(self) -> float:
        """Weighted score favoring actions that are ready and unblocking."""
        return (
            self.unblocks * 3
            + self.reduces_risk * 2
            + self.readiness * 2
            + self.impact * 1
        )


@dataclass
class Action:
    """A concrete, artifact-producing thing AI should do next."""
    verb: str           # e.g. "write", "analyze", "generate", "refactor"
    artifact: str       # what gets produced — a file, dataset, report, etc.
    description: str    # one-sentence plain-English spec
    rationale: str      # why this is the prudent next step
    scores: Scores
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Turn this action into an executable prompt for an AI agent."""
        parts = [
            f"# Task: {self.verb} → {self.artifact}",
            f"\n{self.description}",
        ]
        if self.inputs:
            parts.append("\n## Inputs\n" + "\n".join(f"- {i}" for i in self.inputs))
        if self.outputs:
            parts.append("\n## Expected outputs\n" + "\n".join(f"- {o}" for o in self.outputs))
        return "\n".join(parts)


# ── Core ────────────────────────────────────────────────────────────────

SYSTEM = """\
You are SparkChoice, a planning engine. Given a goal and current state,
you propose 3-5 candidate actions an AI could take next, then pick the
most PRUDENT one — not the most ambitious, the one that is the wisest
next step given the current situation.

Each action MUST produce a concrete artifact (file, document, dataset,
code module, analysis, etc.) — no vague "think about" or "explore" steps.

Score each candidate on four dimensions (1-5):
- unblocks: How many downstream actions does completing this enable?
- reduces_risk: Does this retire uncertainty, validate assumptions, or
  prevent wasted work?
- readiness: Do we have everything needed to execute this right now?
- impact: How much does this move the overall goal forward?

The prudent choice is typically the one that is ready NOW, unblocks the
most future work, and reduces the biggest current risk — even if another
candidate has higher raw impact.

Respond with a JSON object:
{
  "candidates": [
    {
      "verb": "...", "artifact": "...", "description": "...",
      "rationale": "...",
      "scores": {"unblocks": N, "reduces_risk": N, "readiness": N, "impact": N},
      "inputs": ["..."], "outputs": ["..."]
    }
  ],
  "chosen": 0,
  "reasoning": "one sentence on why this candidate wins"
}
"chosen" is the zero-based index of the best candidate.
No markdown fences. No commentary outside the JSON."""


def choose(goal: str, state: str = "", model: str = "claude-sonnet-4-6") -> tuple[Action, list[Action], str]:
    """Return (chosen_action, all_candidates, reasoning)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    user_msg = f"## Goal\n{goal}"
    if state:
        user_msg += f"\n\n## Current state\n{state}"

    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    payload = json.loads(resp.content[0].text)

    def _to_action(d: dict) -> Action:
        s = d.pop("scores")
        return Action(scores=Scores(**s), **d)

    candidates = [_to_action(c) for c in payload["candidates"]]
    chosen = candidates[payload["chosen"]]
    return chosen, candidates, payload["reasoning"]


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python sparkchoice.py '<goal>' ['<current state>']")
        sys.exit(1)

    goal = sys.argv[1]
    state = sys.argv[2] if len(sys.argv) > 2 else ""
    chosen, candidates, reasoning = choose(goal, state)

    print("=== Candidates ===\n")
    for i, c in enumerate(candidates):
        marker = " ← CHOSEN" if c is chosen else ""
        print(f"  [{i}] {c.verb} → {c.artifact}{marker}")
        print(f"      unblocks={c.scores.unblocks} risk={c.scores.reduces_risk} "
              f"ready={c.scores.readiness} impact={c.scores.impact} "
              f"prudence={c.scores.prudence}")

    print(f"\n=== Why ===\n{reasoning}")
    print(f"\n=== Chosen action ===\n")
    print(json.dumps(asdict(chosen), indent=2))
    print("\n--- Executable prompt ---\n")
    print(chosen.to_prompt())


if __name__ == "__main__":
    main()
