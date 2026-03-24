"""Tests for SparkChoice — strategies, scoring, parsing, CLI."""

import json
import subprocess
import sys
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from sparkchoice import (
    Action, Scores, Strategy,
    WeightedSum, GeometricMean, EliminationGates,
    ParetoThenWeighted, PhaseAdaptive,
    STRATEGIES, get_strategy, choose,
)


# ── Helpers ─────────────────────────────────────────────────────────────

def make_action(verb="write", artifact="spec.md", **score_overrides):
    defaults = dict(unblocks=3, reduces_risk=3, readiness=3, impact=3)
    defaults.update(score_overrides)
    return Action(
        verb=verb,
        artifact=artifact,
        description=f"{verb} the {artifact}.",
        rationale="Test action.",
        scores=Scores(**defaults),
        inputs=[],
        outputs=[artifact],
    )


# ── Scores ──────────────────────────────────────────────────────────────

class TestScores:
    def test_prudence_weights(self):
        s = Scores(unblocks=5, reduces_risk=5, readiness=5, impact=5)
        assert s.prudence == 40

    def test_prudence_favors_unblocks(self):
        high_unblock = Scores(unblocks=5, reduces_risk=1, readiness=1, impact=1)
        high_impact = Scores(unblocks=1, reduces_risk=1, readiness=1, impact=5)
        assert high_unblock.prudence > high_impact.prudence

    def test_prudence_readiness_beats_impact(self):
        ready = Scores(unblocks=3, reduces_risk=3, readiness=5, impact=1)
        impactful = Scores(unblocks=3, reduces_risk=3, readiness=1, impact=5)
        assert ready.prudence > impactful.prudence

    def test_prudence_minimum(self):
        s = Scores(unblocks=1, reduces_risk=1, readiness=1, impact=1)
        assert s.prudence == 8

    def test_as_tuple(self):
        s = Scores(unblocks=1, reduces_risk=2, readiness=3, impact=4)
        assert s.as_tuple() == (1, 2, 3, 4)


# ── Action ──────────────────────────────────────────────────────────────

class TestAction:
    def test_to_prompt_contains_verb_and_artifact(self):
        a = make_action(inputs=["req.txt"])
        prompt = a.to_prompt()
        assert "write" in prompt
        assert "spec.md" in prompt

    def test_to_prompt_includes_inputs_and_outputs(self):
        a = make_action(inputs=["req.txt"], artifact="spec.md")
        prompt = a.to_prompt()
        assert "req.txt" in prompt
        assert "spec.md" in prompt

    def test_to_prompt_omits_empty_sections(self):
        a = make_action()
        a.outputs = []
        prompt = a.to_prompt()
        assert "## Inputs" not in prompt
        assert "## Expected outputs" not in prompt

    def test_asdict_roundtrip(self):
        a = make_action(unblocks=4)
        d = asdict(a)
        assert d["verb"] == "write"
        assert d["scores"]["unblocks"] == 4


# ── Strategy: WeightedSum ───────────────────────────────────────────────

class TestWeightedSum:
    def test_default_weights_match_prudence(self):
        """With default weights, WeightedSum ranking agrees with prudence."""
        a = make_action(verb="a", unblocks=5, reduces_risk=1, readiness=1, impact=1)
        b = make_action(verb="b", unblocks=1, reduces_risk=1, readiness=1, impact=5)
        ranked = WeightedSum().rank([b, a])
        assert ranked[0].verb == "a"

    def test_custom_weights(self):
        """Custom weights can flip the ranking."""
        a = make_action(verb="a", unblocks=5, reduces_risk=1, readiness=1, impact=1)
        b = make_action(verb="b", unblocks=1, reduces_risk=1, readiness=1, impact=5)
        # impact×10 overwhelms unblocks×1
        ranked = WeightedSum(weights=(1, 1, 1, 10)).rank([a, b])
        assert ranked[0].verb == "b"


# ── Strategy: GeometricMean ─────────────────────────────────────────────

class TestGeometricMean:
    def test_balanced_beats_spikey(self):
        """A balanced candidate beats one with a low dimension."""
        balanced = make_action(verb="bal", unblocks=4, reduces_risk=4, readiness=4, impact=4)
        spikey = make_action(verb="spk", unblocks=5, reduces_risk=5, readiness=1, impact=5)
        ranked = GeometricMean().rank([spikey, balanced])
        assert ranked[0].verb == "bal"

    def test_all_ones_still_ranks(self):
        a = make_action(verb="a", unblocks=1, reduces_risk=1, readiness=1, impact=1)
        b = make_action(verb="b", unblocks=2, reduces_risk=2, readiness=2, impact=2)
        ranked = GeometricMean().rank([a, b])
        assert ranked[0].verb == "b"


# ── Strategy: EliminationGates ──────────────────────────────────────────

class TestEliminationGates:
    def test_eliminates_low_readiness(self):
        ready = make_action(verb="rdy", unblocks=3, reduces_risk=3, readiness=4, impact=3)
        unready = make_action(verb="not", unblocks=5, reduces_risk=5, readiness=1, impact=5)
        ranked = EliminationGates().rank([unready, ready])
        assert ranked[0].verb == "rdy"

    def test_eliminates_below_min_any(self):
        ok = make_action(verb="ok", unblocks=3, reduces_risk=3, readiness=3, impact=3)
        weak = make_action(verb="weak", unblocks=5, reduces_risk=1, readiness=5, impact=5)
        ranked = EliminationGates().rank([weak, ok])
        assert ranked[0].verb == "ok"

    def test_fallback_when_all_eliminated(self):
        """If gates kill everything, fall back to full list."""
        a = make_action(verb="a", unblocks=1, reduces_risk=1, readiness=1, impact=1)
        b = make_action(verb="b", unblocks=1, reduces_risk=1, readiness=2, impact=1)
        ranked = EliminationGates().rank([a, b])
        # Both fail gates, so fallback to WeightedSum on all candidates
        assert len(ranked) == 2


# ── Strategy: ParetoThenWeighted ────────────────────────────────────────

class TestPareto:
    def test_dominated_candidate_removed(self):
        dominant = make_action(verb="dom", unblocks=5, reduces_risk=5, readiness=5, impact=5)
        dominated = make_action(verb="sub", unblocks=3, reduces_risk=3, readiness=3, impact=3)
        ranked = ParetoThenWeighted().rank([dominated, dominant])
        assert ranked[0].verb == "dom"
        # dominated should be excluded from ranking
        assert len(ranked) == 1

    def test_non_dominated_both_kept(self):
        a = make_action(verb="a", unblocks=5, reduces_risk=1, readiness=3, impact=3)
        b = make_action(verb="b", unblocks=1, reduces_risk=5, readiness=3, impact=3)
        ranked = ParetoThenWeighted().rank([a, b])
        assert len(ranked) == 2


# ── Strategy: PhaseAdaptive ─────────────────────────────────────────────

class TestPhaseAdaptive:
    def test_firefighting_favors_readiness(self):
        ready = make_action(verb="rdy", unblocks=1, reduces_risk=3, readiness=5, impact=1)
        blocker = make_action(verb="blk", unblocks=5, reduces_risk=1, readiness=1, impact=1)
        ranked = PhaseAdaptive(phase="firefighting").rank([blocker, ready])
        assert ranked[0].verb == "rdy"

    def test_polishing_favors_impact(self):
        impactful = make_action(verb="imp", unblocks=1, reduces_risk=2, readiness=3, impact=5)
        unblocking = make_action(verb="unb", unblocks=5, reduces_risk=2, readiness=3, impact=1)
        ranked = PhaseAdaptive(phase="polishing").rank([unblocking, impactful])
        assert ranked[0].verb == "imp"

    def test_invalid_phase_defaults_to_building(self):
        strat = PhaseAdaptive(phase="nonsense")
        assert strat.phase == "building"


# ── Strategy Registry ───────────────────────────────────────────────────

class TestRegistry:
    def test_all_strategies_registered(self):
        assert set(STRATEGIES.keys()) == {
            "weighted_sum", "geometric_mean",
            "elimination_gates", "pareto", "phase_adaptive",
        }

    def test_get_strategy_returns_instance(self):
        strat = get_strategy("geometric_mean")
        assert isinstance(strat, GeometricMean)

    def test_get_strategy_passes_kwargs(self):
        strat = get_strategy("phase_adaptive", phase="firefighting")
        assert strat.phase == "firefighting"

    def test_get_strategy_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("magic_8_ball")


# ── choose() with mocked API ───────────────────────────────────────────

MOCK_API_RESPONSE = {
    "candidates": [
        {
            "verb": "write",
            "artifact": "schema.sql",
            "description": "Write the DB schema.",
            "rationale": "Foundation for everything.",
            "scores": {"unblocks": 5, "reduces_risk": 4, "readiness": 5, "impact": 4},
            "inputs": ["requirements"],
            "outputs": ["schema.sql"],
        },
        {
            "verb": "build",
            "artifact": "api.py",
            "description": "Build the API.",
            "rationale": "Connects frontend to backend.",
            "scores": {"unblocks": 3, "reduces_risk": 2, "readiness": 3, "impact": 4},
            "inputs": ["schema.sql"],
            "outputs": ["api.py"],
        },
    ],
    "strategy": "pareto",
    "reasoning": "Schema is ready now and unblocks the API. Pareto confirms it dominates.",
}


class TestChoose:
    def _mock_response(self):
        mock_resp = MagicMock()
        mock_block = MagicMock()
        mock_block.text = json.dumps(MOCK_API_RESPONSE)
        mock_resp.content = [mock_block]
        return mock_resp

    @patch("sparkchoice.anthropic.Anthropic")
    def test_returns_chosen_action(self, mock_cls):
        mock_cls.return_value.messages.create.return_value = self._mock_response()
        chosen, candidates, reasoning, strat = choose("build something")

        assert chosen.verb == "write"
        assert chosen.artifact == "schema.sql"
        assert chosen.scores.unblocks == 5

    @patch("sparkchoice.anthropic.Anthropic")
    def test_returns_all_candidates(self, mock_cls):
        mock_cls.return_value.messages.create.return_value = self._mock_response()
        chosen, candidates, reasoning, strat = choose("build something")

        assert len(candidates) == 2
        assert candidates[1].verb == "build"

    @patch("sparkchoice.anthropic.Anthropic")
    def test_returns_reasoning(self, mock_cls):
        mock_cls.return_value.messages.create.return_value = self._mock_response()
        _, _, reasoning, _ = choose("build something")

        assert "Schema" in reasoning

    @patch("sparkchoice.anthropic.Anthropic")
    def test_uses_claude_recommended_strategy(self, mock_cls):
        mock_cls.return_value.messages.create.return_value = self._mock_response()
        _, _, _, strat = choose("build something")

        assert strat.name == "pareto"

    @patch("sparkchoice.anthropic.Anthropic")
    def test_strategy_override(self, mock_cls):
        mock_cls.return_value.messages.create.return_value = self._mock_response()
        _, _, _, strat = choose("build something", strategy="geometric_mean")

        assert strat.name == "geometric_mean"

    @patch("sparkchoice.anthropic.Anthropic")
    def test_state_included_in_prompt(self, mock_cls):
        mock_cls.return_value.messages.create.return_value = self._mock_response()
        choose("build something", state="have a landing page")

        call_kwargs = mock_cls.return_value.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "landing page" in user_content


# ── CLI ─────────────────────────────────────────────────────────────────

class TestCLI:
    def test_no_args_prints_usage(self):
        result = subprocess.run(
            [sys.executable, "sparkchoice.py"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "Usage" in result.stderr or "Usage" in result.stdout
