# SparkChoice

The smallest system that chooses the best next use of AI and expresses it as a concrete, artifact-producing action.

## How It Works

Given a goal and current state, SparkChoice uses Claude to generate 3-5 candidate next-actions, scores each on four dimensions, and picks the most **prudent** one — not the most ambitious, but the one that makes everything else easier or possible.

### Scoring Dimensions

| Dimension | Weight | Question |
|-----------|--------|----------|
| **Unblocks** | 3x | How many downstream actions does this enable? |
| **Reduces Risk** | 2x | Does this retire uncertainty or validate assumptions? |
| **Readiness** | 2x | Do we have everything needed to do this now? |
| **Impact** | 1x | How much does this move the goal forward? |

Every action must produce a tangible artifact — a file, document, dataset, code module, analysis, etc. No vague "think about" or "explore" steps.

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your-key-here
```

## Usage

```bash
python sparkchoice.py 'Build a SaaS analytics dashboard'
python sparkchoice.py 'Build a SaaS analytics dashboard' 'Have a landing page and auth system'
```

## Programmatic Usage

```python
from sparkchoice import choose

chosen, candidates, reasoning = choose("Build a SaaS analytics dashboard")

print(chosen.verb, "→", chosen.artifact)
print(chosen.scores.prudence)
print(chosen.to_prompt())  # executable prompt for an AI agent
```

## Tests

```bash
pytest test_sparkchoice.py
```

## License

[MIT](LICENSE) — David Liedle <david.liedle@protonmail.com>
