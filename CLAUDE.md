# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

SparkChoice is a single-file AI planning engine that takes a goal + current state, generates candidate next-actions via the Claude API, scores them on a "prudence" heuristic, and emits the best one as an executable prompt. It uses the Anthropic Python SDK (`anthropic`).

## Commands

```bash
# Run the tool
python sparkchoice.py '<goal>' ['<current state>']

# Run tests
pytest test_sparkchoice.py

# Run a single test
pytest test_sparkchoice.py::TestScores::test_prudence_weights
```

## Architecture

Single module (`sparkchoice.py`) with three layers:

- **Domain**: `Scores` (4-dimension scoring with weighted `prudence` property: unblocks×3, risk×2, readiness×2, impact×1) and `Action` (artifact-producing step with `to_prompt()` for downstream AI agents)
- **Core**: `choose(goal, state, model)` — calls Claude API with a JSON-only system prompt, parses response into Actions, returns `(chosen, candidates, reasoning)`
- **CLI**: `main()` — argv wrapper that prints candidates, reasoning, and the chosen action's executable prompt

The API contract: Claude returns a JSON object with `candidates` array, `chosen` index, and `reasoning` string. Each candidate has `verb`, `artifact`, `description`, `rationale`, `scores`, `inputs`, `outputs`.

## Environment

- Requires `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` env var
- Default model: `claude-sonnet-4-6`
- Dependencies: `anthropic`, `pytest` (for tests)
