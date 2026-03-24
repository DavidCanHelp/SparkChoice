# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

SparkChoice is a single-file AI planning engine that takes a goal + current state, generates candidate next-actions via the Claude API, scores them on four dimensions, selects a scoring strategy appropriate to the situation, and emits the best action as an executable prompt.

## Commands

```bash
# Run the tool
python sparkchoice.py '<goal>' ['<current state>'] [--strategy NAME]

# Run tests
pytest test_sparkchoice.py

# Run a single test
pytest test_sparkchoice.py::TestWeightedSum::test_default_weights_match_prudence
```

## Architecture

Single module (`sparkchoice.py`) with four layers:

- **Domain**: `Scores` (4-dimension raw scores with `as_tuple()` and legacy `prudence` property) and `Action` (artifact-producing step with `to_prompt()` for downstream AI agents)
- **Strategies**: Abstract `Strategy` base class with five implementations:
  - `WeightedSum` — linear weighted sum with configurable weights (default: unblocks×3, risk×2, readiness×2, impact×1)
  - `GeometricMean` — geometric mean across dimensions; punishes any weakness
  - `EliminationGates` — filters below thresholds (min_readiness=3, min_any=2), then ranks survivors; falls back to full list if gates kill everything
  - `ParetoThenWeighted` — removes dominated candidates, then ranks non-dominated set by weighted sum
  - `PhaseAdaptive` — shifts weight profiles by project phase (greenfield, building, polishing, firefighting)
- **Registry**: `STRATEGIES` dict mapping name → class; `get_strategy(name, **kwargs)` for instantiation
- **Core**: `choose(goal, state, model, strategy)` — calls Claude API, parses candidates, selects strategy (explicit override > Claude's recommendation > default weighted_sum), ranks, returns `(chosen, candidates, reasoning, strategy)`
- **CLI**: `main()` — argv parser with `--strategy` flag; prints strategy info, ranked candidates, reasoning, and executable prompt

The API contract: Claude returns a JSON object with `candidates` array, `strategy` name, optional `phase`, and `reasoning` string. Each candidate has `verb`, `artifact`, `description`, `rationale`, `scores`, `inputs`, `outputs`. There is no `chosen` field — the strategy determines the winner.

## Environment

- Requires `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` env var
- Default model: `claude-sonnet-4-6`
- Dependencies: `anthropic`, `pytest` (for tests)

## Design Principles

- Different situations call for different ways of deciding — no single formula is universal
- Strategies are pure Python ranking over Claude-provided scores; no extra API calls
- Every action must produce a concrete artifact
- Strategy selection is a first-class concern, not an afterthought
