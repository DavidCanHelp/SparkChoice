"""
SparkChoice — smallest system that chooses the best next use of AI
and expresses it as a concrete artifact-producing action.

The loop:
  1. Takes a goal + current state (what exists so far)
  2. Generates candidate next-actions (each produces a tangible artifact)
  3. Scores each on: unblocks, reduces_risk, readiness, impact
  4. Selects a scoring strategy appropriate to the situation
  5. Ranks candidates using the chosen strategy
  6. Emits the most prudent choice
"""

from __future__ import annotations
import datetime
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from functools import reduce

import anthropic


# ── Domain ──────────────────────────────────────────────────────────────

@dataclass
class Scores:
    """Raw dimension scores for a candidate action."""
    unblocks: int      # 1-5: how many downstream actions does this enable?
    reduces_risk: int  # 1-5: does this retire uncertainty or validate assumptions?
    readiness: int     # 1-5: do we have everything needed to do this now?
    impact: int        # 1-5: how much does this move the goal forward?

    @property
    def prudence(self) -> float:
        """Legacy weighted score. Use a Strategy for smarter ranking."""
        return (
            self.unblocks * 3
            + self.reduces_risk * 2
            + self.readiness * 2
            + self.impact * 1
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return dimensions as (unblocks, reduces_risk, readiness, impact)."""
        return (self.unblocks, self.reduces_risk, self.readiness, self.impact)


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


# ── Strategies ──────────────────────────────────────────────────────────

class Strategy(ABC):
    """Base class for candidate ranking strategies."""
    name: str = ""
    description: str = ""

    @abstractmethod
    def rank(self, candidates: list[Action]) -> list[Action]:
        """Return candidates sorted best-first."""
        ...

    def __repr__(self):
        return f"<Strategy: {self.name}>"


class WeightedSum(Strategy):
    """Original linear weighted sum. Fast, simple, good default."""
    name = "weighted_sum"
    description = (
        "Linear weighted sum (unblocks×3, risk×2, readiness×2, impact×1). "
        "Best when candidates are clearly differentiated."
    )

    def __init__(self, weights: tuple[int, ...] = (3, 2, 2, 1)):
        self.weights = weights

    def rank(self, candidates: list[Action]) -> list[Action]:
        def score(a: Action) -> float:
            return sum(d * w for d, w in zip(a.scores.as_tuple(), self.weights))
        return sorted(candidates, key=score, reverse=True)


class GeometricMean(Strategy):
    """Geometric mean punishes any weak dimension severely."""
    name = "geometric_mean"
    description = (
        "Geometric mean across all dimensions. Rewards balance, "
        "punishes any weakness. Best when you can't afford a blind spot."
    )

    def rank(self, candidates: list[Action]) -> list[Action]:
        def score(a: Action) -> float:
            dims = a.scores.as_tuple()
            product = reduce(lambda x, y: x * y, dims)
            return product ** (1.0 / len(dims))
        return sorted(candidates, key=score, reverse=True)


class EliminationGates(Strategy):
    """Filter out candidates below thresholds, then delegate ranking."""
    name = "elimination_gates"
    description = (
        "Eliminates candidates below minimum thresholds, then ranks "
        "survivors with a delegate strategy. Best when some options "
        "are clearly not viable."
    )

    def __init__(
        self,
        min_readiness: int = 3,
        min_any: int = 2,
        then: Strategy | None = None,
    ):
        self.min_readiness = min_readiness
        self.min_any = min_any
        self._then = then or WeightedSum()

    def rank(self, candidates: list[Action]) -> list[Action]:
        survivors = [
            c for c in candidates
            if c.scores.readiness >= self.min_readiness
            and all(d >= self.min_any for d in c.scores.as_tuple())
        ]
        # If gates kill everything, fall back to all candidates
        if not survivors:
            survivors = candidates
        return self._then.rank(survivors)


class ParetoThenRank(Strategy):
    """Keep only non-dominated candidates, then delegate ranking."""
    name = "pareto"
    description = (
        "Pareto ranking: keeps only non-dominated candidates, then breaks "
        "ties with a delegate strategy. Best when several candidates "
        "are competitive."
    )

    def __init__(self, then: Strategy | None = None):
        self._then = then or WeightedSum()

    def rank(self, candidates: list[Action]) -> list[Action]:
        def dominates(a: Action, b: Action) -> bool:
            a_dims = a.scores.as_tuple()
            b_dims = b.scores.as_tuple()
            at_least = all(ai >= bi for ai, bi in zip(a_dims, b_dims))
            strict = any(ai > bi for ai, bi in zip(a_dims, b_dims))
            return at_least and strict

        non_dominated = [
            c for c in candidates
            if not any(
                dominates(other, c)
                for other in candidates if other is not c
            )
        ]
        if not non_dominated:
            non_dominated = candidates
        return self._then.rank(non_dominated)


# Keep old name as alias for backward compatibility
ParetoThenWeighted = ParetoThenRank


class PhaseAdaptive(Strategy):
    """Shifts weights based on project phase."""
    name = "phase_adaptive"
    description = (
        "Adjusts scoring weights based on project phase. "
        "Best when the goal clearly implies greenfield, building, "
        "polishing, or firefighting."
    )

    PHASE_WEIGHTS: dict[str, tuple[int, ...]] = {
        "greenfield":   (2, 3, 2, 1),  # risk reduction dominates
        "building":     (3, 2, 2, 1),  # unblocking matters most
        "polishing":    (1, 2, 2, 3),  # impact rises
        "firefighting": (1, 3, 3, 1),  # risk + readiness, move NOW
    }

    def __init__(self, phase: str = "building"):
        if phase not in self.PHASE_WEIGHTS:
            phase = "building"
        self.phase = phase
        self.weights = self.PHASE_WEIGHTS[phase]

    def rank(self, candidates: list[Action]) -> list[Action]:
        return WeightedSum(weights=self.weights).rank(candidates)


# ── Strategy Registry ───────────────────────────────────────────────────

STRATEGIES: dict[str, type[Strategy]] = {
    "weighted_sum": WeightedSum,
    "geometric_mean": GeometricMean,
    "elimination_gates": EliminationGates,
    "pareto": ParetoThenRank,
    "phase_adaptive": PhaseAdaptive,
}


def get_strategy(name: str, **kwargs) -> Strategy:
    """Instantiate a strategy by name."""
    cls = STRATEGIES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy: {name}. "
            f"Available: {list(STRATEGIES.keys())}"
        )
    return cls(**kwargs)


# ── Core ────────────────────────────────────────────────────────────────

SYSTEM = """\
You are SparkChoice, a planning engine. Given a goal and current state,
you perform three distinct steps in order:

STEP 1 — GENERATE CANDIDATES
Propose 3-5 candidate actions an AI could take next. Each action MUST
produce a concrete artifact (file, document, dataset, code module,
analysis, etc.) — no vague "think about" or "explore" steps.

STEP 2 — SCORE INDEPENDENTLY
Score each candidate on four dimensions (1-5). Assign scores based only
on each candidate's intrinsic properties relative to the goal and current
state. Do not shape scores to favor any particular ranking strategy.

- unblocks: How many downstream actions does completing this enable?
- reduces_risk: Does this retire uncertainty, validate assumptions, or
  prevent wasted work?
- readiness: Do we have everything needed to execute this right now?
- impact: How much does this move the overall goal forward?

STEP 3 — RECOMMEND STRATEGY
After scoring, look at the resulting score landscape and recommend the
most appropriate ranking strategy. Strategy recommendation happens after
scoring; treat scoring and strategy selection as distinct judgments.

- "weighted_sum": Linear weighted sum. Good default for clear-cut decisions.
- "geometric_mean": Geometric mean. Use when balance matters and no
  dimension should be weak.
- "elimination_gates": Filter out unready or weak candidates first.
  Use when some options are clearly not viable yet.
- "pareto": Keep only non-dominated candidates, then pick. Use when
  several candidates are competitive and you need nuance.
- "phase_adaptive": Shift weights based on project phase. Use when
  the goal/state clearly implies greenfield, building, polishing,
  or firefighting. Include a "phase" field if you choose this.

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
  "strategy": "strategy_name",
  "phase": "optional — only if strategy is phase_adaptive",
  "reasoning": "why this strategy and this winner fit the situation"
}
No "chosen" field — the strategy determines the winner.
No markdown fences. No commentary outside the JSON."""


def choose(
    goal: str,
    state: str = "",
    model: str = "claude-sonnet-4-6",
    strategy: str | None = None,
) -> tuple[Action, list[Action], str, Strategy]:
    """Return (chosen_action, ranked_candidates, reasoning, strategy_used).

    If `strategy` is provided, it overrides Claude's recommendation.
    Candidates are returned in ranked order (best first).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    user_msg = f"## Goal\n{goal}"
    if state:
        user_msg += f"\n\n## Current state\n{state}"

    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    payload = json.loads(resp.content[0].text)

    def _to_action(d: dict) -> Action:
        d = dict(d)  # shallow copy to avoid mutating payload
        s = d.pop("scores")
        return Action(scores=Scores(**s), **d)

    candidates = [_to_action(c) for c in payload["candidates"]]

    # Strategy selection: explicit override > Claude's recommendation
    strategy_name = strategy or payload.get("strategy", "weighted_sum")
    strategy_kwargs: dict = {}
    if strategy_name == "phase_adaptive" and "phase" in payload:
        strategy_kwargs["phase"] = payload["phase"]

    strat = get_strategy(strategy_name, **strategy_kwargs)
    ranked = strat.rank(candidates)
    chosen = ranked[0]

    reasoning = payload.get("reasoning", "")
    return chosen, ranked, reasoning, strat


# ── Decision Logging ───────────────────────────────────────────────────

def append_decision_log(
    path: str,
    goal: str,
    state: str,
    model: str,
    strategy: Strategy,
    reasoning: str,
    chosen: Action,
    candidates: list[Action],
) -> None:
    """Append a single JSONL entry to the decision log file."""
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "goal": goal,
        "state": state,
        "model": model,
        "strategy": strategy.name,
        "reasoning": reasoning,
        "chosen": asdict(chosen),
        "candidates": [asdict(c) for c in candidates],
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python sparkchoice.py '<goal>' ['<current state>'] [--strategy NAME] [--log FILE]")
        sys.exit(1)

    # Parse args
    args = sys.argv[1:]
    strategy_override = None
    log_path = None

    if "--strategy" in args:
        idx = args.index("--strategy")
        if idx + 1 < len(args):
            strategy_override = args[idx + 1]
            args = args[:idx] + args[idx + 2:]

    if "--log" in args:
        idx = args.index("--log")
        if idx + 1 < len(args):
            log_path = args[idx + 1]
            args = args[:idx] + args[idx + 2:]

    goal = args[0]
    state = args[1] if len(args) > 1 else ""
    model = "claude-sonnet-4-6"
    chosen, ranked, reasoning, strat = choose(
        goal, state, model=model, strategy=strategy_override
    )

    if log_path:
        append_decision_log(
            log_path, goal, state, model, strat, reasoning, chosen, ranked,
        )

    print(f"=== Strategy: {strat.name} ===")
    print(f"    {strat.description}\n")
    print("=== Candidates ===\n")
    for i, c in enumerate(ranked):
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
