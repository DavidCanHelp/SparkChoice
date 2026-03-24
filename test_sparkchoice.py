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

def make_action(verb="write", artifact="spec.md", inputs=None, **score_overrides):
    defaults = dict(unblocks=3, reduces_risk=3, readiness=3, impact=3)
    defaults.update(score_overrides)
    return Action(
        verb=verb,
        artifact=artifact,
        description=f"{verb} the {artifact}.",
        rationale="Test action.",
        scores=Scores(**defaults),
        inputs=inputs or [],
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


# ── Score Independence (adversarial fixtures) ─────────────────────────
#
# These fixtures simulate score profiles that a model might produce if it
# were shaping scores to validate a preferred strategy rather than scoring
# independently. Each test runs ALL strategies over the SAME scores and
# checks that the strategies genuinely disagree where they should — proving
# the ranking is downstream of the scores, not baked into them.


class TestScoreIndependence:
    """Adversarial cases where score-strategy coupling is most likely to
    break down. If scores were manufactured to favor a strategy, these
    fixtures would expose it by showing that alternative strategies
    produce different — and defensible — rankings."""

    def test_uniform_scores_all_strategies_agree(self):
        """When all candidates score identically, every strategy should
        return the same set (order may vary but no candidate is preferred).
        A coupled model might inflate one candidate to create a 'winner'."""
        a = make_action(verb="a", unblocks=3, reduces_risk=3, readiness=3, impact=3)
        b = make_action(verb="b", unblocks=3, reduces_risk=3, readiness=3, impact=3)
        c = make_action(verb="c", unblocks=3, reduces_risk=3, readiness=3, impact=3)
        pool = [a, b, c]

        for name, cls in STRATEGIES.items():
            kwargs = {"phase": "building"} if name == "phase_adaptive" else {}
            ranked = cls(**kwargs).rank(pool)
            assert len(ranked) == 3, f"{name} dropped a candidate on uniform scores"

    def test_spike_vs_balanced_strategies_disagree(self):
        """A high-spike candidate should win weighted_sum but lose to a
        balanced candidate under geometric_mean. If scores were shaped to
        make one strategy 'right', this disagreement would vanish."""
        spike = make_action(verb="spike", unblocks=5, reduces_risk=5, readiness=1, impact=5)
        balanced = make_action(verb="balanced", unblocks=4, reduces_risk=4, readiness=4, impact=4)

        ws_winner = WeightedSum().rank([spike, balanced])[0]
        gm_winner = GeometricMean().rank([spike, balanced])[0]

        assert ws_winner.verb == "spike", "weighted_sum should favor raw total"
        assert gm_winner.verb == "balanced", "geometric_mean should punish readiness=1"

    def test_unready_high_scorer_gates_vs_weighted(self):
        """A candidate that dominates on 3 of 4 dimensions but has
        readiness=1 should win weighted_sum but be eliminated by gates.
        Coupling would tempt the model to bump readiness to 3."""
        powerhouse = make_action(verb="power", unblocks=5, reduces_risk=5, readiness=1, impact=5)
        modest = make_action(verb="modest", unblocks=3, reduces_risk=3, readiness=4, impact=3)

        ws_winner = WeightedSum().rank([powerhouse, modest])[0]
        eg_winner = EliminationGates().rank([powerhouse, modest])[0]

        assert ws_winner.verb == "power", "weighted_sum ignores readiness threshold"
        assert eg_winner.verb == "modest", "gates should eliminate readiness=1"

    def test_pareto_preserves_tradeoff_weighted_sum_breaks_tie(self):
        """Two non-dominated candidates with opposite strengths. Pareto
        keeps both; weighted_sum must pick one. A coupled model might
        suppress one dimension to create artificial domination."""
        a = make_action(verb="unblocker", unblocks=5, reduces_risk=1, readiness=3, impact=3)
        b = make_action(verb="derisker", unblocks=1, reduces_risk=5, readiness=3, impact=3)

        pareto_ranked = ParetoThenWeighted().rank([a, b])
        assert len(pareto_ranked) == 2, "neither dominates the other — both must survive"

        ws_winner = WeightedSum().rank([a, b])[0]
        assert ws_winner.verb == "unblocker", "default weights favor unblocks×3"

    def test_phase_flip_same_scores_different_winners(self):
        """Identical scores, but switching phase flips the winner. This
        proves the strategy is applied after scoring — if scores were
        baked for one phase, the other phase would pick wrong."""
        ready_impactful = make_action(
            verb="polish", unblocks=1, reduces_risk=2, readiness=5, impact=5,
        )
        risky_unblocker = make_action(
            verb="explore", unblocks=5, reduces_risk=5, readiness=1, impact=1,
        )

        ff_winner = PhaseAdaptive(phase="firefighting").rank(
            [ready_impactful, risky_unblocker]
        )[0]
        gf_winner = PhaseAdaptive(phase="greenfield").rank(
            [ready_impactful, risky_unblocker]
        )[0]

        assert ff_winner.verb == "polish", "firefighting: readiness+risk dominate"
        assert gf_winner.verb == "explore", "greenfield: risk reduction dominates"

    def test_one_point_gap_all_strategies_sensitive(self):
        """When candidates differ by just 1 point on a single dimension,
        strategies should still produce consistent rankings — not random
        ones. A coupled model might exaggerate gaps to create clarity."""
        a = make_action(verb="a", unblocks=4, reduces_risk=3, readiness=3, impact=3)
        b = make_action(verb="b", unblocks=3, reduces_risk=3, readiness=3, impact=3)

        # a beats b on unblocks by 1 point, equal everywhere else
        # Every strategy should rank a >= b
        for name, cls in STRATEGIES.items():
            kwargs = {"phase": "building"} if name == "phase_adaptive" else {}
            ranked = cls(**kwargs).rank([b, a])
            assert ranked[0].verb == "a", (
                f"{name} failed: 1-point unblocks advantage should still win"
            )

    def test_elimination_gates_fallback_preserves_all(self):
        """When all candidates fail gates, the fallback must return
        everyone — not silently drop the 'inconvenient' ones. A model
        that baked in elimination might produce scores where exactly
        one candidate barely passes."""
        a = make_action(verb="a", unblocks=1, reduces_risk=1, readiness=1, impact=1)
        b = make_action(verb="b", unblocks=1, reduces_risk=1, readiness=2, impact=1)
        c = make_action(verb="c", unblocks=2, reduces_risk=1, readiness=1, impact=1)

        ranked = EliminationGates().rank([a, b, c])
        assert len(ranked) == 3, "all fail gates, so all must be in fallback"

    def test_geometric_mean_zero_dimension_catastrophe(self):
        """If any dimension could hit 0 (or effective 0 via score=1),
        geometric mean craters it. A coupled model might avoid giving
        any candidate a 1 when recommending geometric_mean.
        hole: gm((5,5,5,1)) = 3.34, balanced: gm((4,4,4,4)) = 4.0"""
        strong_with_hole = make_action(
            verb="hole", unblocks=5, reduces_risk=5, readiness=5, impact=1,
        )
        balanced = make_action(
            verb="balanced", unblocks=4, reduces_risk=4, readiness=4, impact=4,
        )

        gm_winner = GeometricMean().rank([strong_with_hole, balanced])[0]
        ws_winner = WeightedSum().rank([strong_with_hole, balanced])[0]

        assert ws_winner.verb == "hole", "weighted_sum: high total wins"
        assert gm_winner.verb == "balanced", "geometric_mean: impact=1 craters the score"

    def test_pareto_three_way_no_domination(self):
        """Three candidates where none dominates another — a rock-paper-
        scissors pattern. Pareto must keep all three. A coupled model
        might tweak one candidate to be dominated and simplify the choice."""
        rock = make_action(verb="rock", unblocks=5, reduces_risk=1, readiness=3, impact=3)
        paper = make_action(verb="paper", unblocks=3, reduces_risk=5, readiness=1, impact=3)
        scissors = make_action(verb="scissors", unblocks=3, reduces_risk=3, readiness=5, impact=1)

        ranked = ParetoThenWeighted().rank([rock, paper, scissors])
        assert len(ranked) == 3, "rock-paper-scissors: no candidate is dominated"

    def test_strategy_disagreement_matrix(self):
        """The ultimate coupling detector: a 4-candidate fixture designed
        so that at least 3 different strategies produce different winners.
        If scores were shaped for one strategy, this would collapse."""
        # High total, weak readiness — weighted_sum's darling
        # ws: 5*3+5*2+2*2+5*1 = 34, gm: (5*5*2*5)^0.25 = 3.98
        a = make_action(verb="a", unblocks=5, reduces_risk=5, readiness=2, impact=5)
        # Perfectly balanced — geometric_mean's darling
        # ws: 4*3+4*2+4*2+4*1 = 32, gm: (4*4*4*4)^0.25 = 4.0
        b = make_action(verb="b", unblocks=4, reduces_risk=4, readiness=4, impact=4)
        # High readiness + risk — firefighting phase's darling
        # ff: 1*1+5*3+5*3+2*1 = 33, beats b's 32
        c = make_action(verb="c", unblocks=1, reduces_risk=5, readiness=5, impact=2)
        # Below gates on risk — should be eliminated by gates
        d = make_action(verb="d", unblocks=4, reduces_risk=1, readiness=4, impact=4)

        pool = [a, b, c, d]

        ws_winner = WeightedSum().rank(pool)[0]
        gm_winner = GeometricMean().rank(pool)[0]
        eg_winner = EliminationGates().rank(pool)[0]
        ff_winner = PhaseAdaptive(phase="firefighting").rank(pool)[0]

        # Weighted sum: a has highest raw weighted total
        assert ws_winner.verb == "a"
        # Geometric mean: b is most balanced (a has readiness=1)
        assert gm_winner.verb == "b"
        # Elimination gates: a fails readiness gate, d fails risk gate
        assert eg_winner.verb == "b"
        # Firefighting: c dominates on readiness+risk which firefighting weights heavily
        assert ff_winner.verb == "c"

        # At least 3 distinct winners across strategies
        winners = {ws_winner.verb, gm_winner.verb, ff_winner.verb}
        assert len(winners) >= 3, (
            f"Only {len(winners)} distinct winners — strategies may not be independent"
        )


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
