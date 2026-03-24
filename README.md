# SparkChoice

The smallest system that chooses the best next use of AI and expresses it as a concrete, artifact-producing action.

## How It Works

Given a goal and current state, SparkChoice uses Claude to generate 3-5 candidate next-actions, scores each on four dimensions, then selects the most appropriate **scoring strategy** for the situation and ranks candidates accordingly.

The system doesn't pretend one formula works for everything. Different situations call for different ways of deciding.

### Scoring Dimensions

| Dimension | Question |
|-----------|----------|
| **Unblocks** | How many downstream actions does this enable? |
| **Reduces Risk** | Does this retire uncertainty or validate assumptions? |
| **Readiness** | Do we have everything needed to do this now? |
| **Impact** | How much does this move the goal forward? |

### Scoring Strategies

| Strategy | When to use |
|----------|-------------|
| **weighted_sum** | Default. Linear weighted sum. Good when candidates are clearly differentiated. |
| **geometric_mean** | When balance matters. A single weak dimension craters the score. |
| **elimination_gates** | When some options are clearly not viable. Filters first, then ranks survivors. |
| **pareto** | When several candidates are competitive. Keeps only non-dominated options. |
| **phase_adaptive** | When the goal implies a clear project phase (greenfield, building, polishing, firefighting). |

Claude recommends a strategy per query based on the goal, state, and candidates. You can override it.

Every action must produce a tangible artifact — a file, document, dataset, code module, analysis, etc. No vague "think about" or "explore" steps.

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your-key-here
```

## Usage

```bash
# Let Claude pick the strategy
python sparkchoice.py 'Build a SaaS analytics dashboard'
python sparkchoice.py 'Build a SaaS analytics dashboard' 'Have a landing page and auth system'

# Override the strategy
python sparkchoice.py 'Fix the production outage' --strategy elimination_gates
python sparkchoice.py 'Polish the MVP for launch' --strategy phase_adaptive
```

## Programmatic Usage

```python
from sparkchoice import choose, get_strategy, STRATEGIES

# Let Claude recommend the strategy
chosen, candidates, reasoning, strat = choose("Build a SaaS analytics dashboard")

print(f"Strategy: {strat.name}")
print(f"{chosen.verb} → {chosen.artifact}")
print(chosen.to_prompt())  # executable prompt for an AI agent

# Override with a specific strategy
chosen, candidates, reasoning, strat = choose(
    "Fix the production outage",
    strategy="elimination_gates",
)

# Use strategies directly for local ranking
from sparkchoice import GeometricMean
ranked = GeometricMean().rank(candidates)
```

## Tests

```bash
pytest test_sparkchoice.py
```

### Judgment Integrity

SparkChoice includes adversarial test fixtures (`TestScoreIndependence`) designed to detect unwanted coupling between scoring and strategy selection. The risk: a model that recommends a strategy might also shape candidate scores to validate that recommendation, collapsing meaningful disagreement between strategies.

The tests verify that strategy disagreement is preserved in edge cases rather than erased by score shaping. Representative cases:

- **Spike vs balanced** — weighted_sum and geometric_mean must pick different winners; if they agree, scores may have been smoothed to erase the tradeoff
- **Phase flip** — identical scores produce different winners under different project phases, proving the strategy is applied after scoring
- **Geometric mean catastrophe** — a single weak dimension craters geometric_mean but not weighted_sum; a coupled model would avoid assigning low scores
- **Disagreement matrix** — four candidates where at least three strategies each pick a different winner; the strongest test that strategies operate independently over honest scores

## License

[MIT](LICENSE) — David Liedle <david.liedle@protonmail.com>
