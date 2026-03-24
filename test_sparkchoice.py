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
    ParetoThenRank, ParetoThenWeighted, PhaseAdaptive,
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
        assert len(ranked) == 2

    def test_custom_delegate(self):
        """EliminationGates can delegate to a non-default strategy."""
        ready = make_action(verb="rdy", unblocks=3, reduces_risk=3, readiness=4, impact=3)
        balanced = make_action(verb="bal", unblocks=4, reduces_risk=4, readiness=4, impact=4)
        # Default delegate (WeightedSum) and GeometricMean agree here,
        # but we verify the plumbing works
        strat = EliminationGates(then=GeometricMean())
        ranked = strat.rank([ready, balanced])
        assert ranked[0].verb == "bal"  # geometric_mean favors balance


# ── Strategy: ParetoThenRank ───────────────────────────────────────────

class TestPareto:
    def test_dominated_candidate_removed(self):
        dominant = make_action(verb="dom", unblocks=5, reduces_risk=5, readiness=5, impact=5)
        dominated = make_action(verb="sub", unblocks=3, reduces_risk=3, readiness=3, impact=3)
        ranked = ParetoThenRank().rank([dominated, dominant])
        assert ranked[0].verb == "dom"
        assert len(ranked) == 1

    def test_non_dominated_both_kept(self):
        a = make_action(verb="a", unblocks=5, reduces_risk=1, readiness=3, impact=3)
        b = make_action(verb="b", unblocks=1, reduces_risk=5, readiness=3, impact=3)
        ranked = ParetoThenRank().rank([a, b])
        assert len(ranked) == 2

    def test_custom_delegate(self):
        """ParetoThenRank can delegate to a non-default strategy."""
        a = make_action(verb="a", unblocks=5, reduces_risk=1, readiness=3, impact=3)
        b = make_action(verb="b", unblocks=3, reduces_risk=3, readiness=3, impact=3)
        # Neither dominates the other. GeometricMean should prefer b (balanced).
        strat = ParetoThenRank(then=GeometricMean())
        ranked = strat.rank([a, b])
        assert ranked[0].verb == "b"

    def test_backward_compat_alias(self):
        """ParetoThenWeighted is an alias for ParetoThenRank."""
        assert ParetoThenWeighted is ParetoThenRank


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


# ── Strategy Composition ────────────────────────────────────────────────

class TestStrategyComposition:
    def test_gates_then_geometric(self):
        """EliminationGates filtering + GeometricMean ranking."""
        # Unready powerhouse: eliminated by gates
        a = make_action(verb="a", unblocks=5, reduces_risk=5, readiness=1, impact=5)
        # Balanced and ready
        b = make_action(verb="b", unblocks=4, reduces_risk=4, readiness=4, impact=4)
        # Ready but spikey
        c = make_action(verb="c", unblocks=5, reduces_risk=2, readiness=4, impact=5)

        strat = EliminationGates(then=GeometricMean())
        ranked = strat.rank([a, b, c])

        # a eliminated (readiness=1). Among b and c, geometric_mean
        # prefers b (balanced 4s) over c (has a 2).
        assert ranked[0].verb == "b"
        assert len(ranked) == 2  # a was filtered out

    def test_pareto_then_geometric(self):
        """Pareto filtering + GeometricMean for survivors."""
        dominant = make_action(verb="dom", unblocks=5, reduces_risk=5, readiness=5, impact=5)
        balanced = make_action(verb="bal", unblocks=4, reduces_risk=4, readiness=4, impact=4)
        tradeoff = make_action(verb="trd", unblocks=5, reduces_risk=1, readiness=5, impact=5)

        strat = ParetoThenRank(then=GeometricMean())
        ranked = strat.rank([tradeoff, balanced, dominant])

        # dominant dominates balanced. tradeoff is not dominated by
        # dominant (reduces_risk: 1 < 5, but pareto needs all >=).
        # Wait — dominant has 5,5,5,5 and tradeoff has 5,1,5,5.
        # dominant >= tradeoff on all dims and > on reduces_risk.
        # So tradeoff IS dominated. Only dominant survives.
        assert ranked[0].verb == "dom"
        assert len(ranked) == 1

    def test_gates_then_weighted_with_custom_weights(self):
        """Compose gates with a custom-weighted sum."""
        # Both pass gates, but custom weights favor impact
        a = make_action(verb="a", unblocks=5, reduces_risk=3, readiness=3, impact=2)
        b = make_action(verb="b", unblocks=2, reduces_risk=3, readiness=3, impact=5)

        strat = EliminationGates(then=WeightedSum(weights=(1, 1, 1, 10)))
        ranked = strat.rank([a, b])
        assert ranked[0].verb == "b"  # impact×10 wins


# ── Score Independence (adversarial fixtures) ───────────────────────
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

        pareto_ranked = ParetoThenRank().rank([a, b])
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

        for name, cls in STRATEGIES.items():
            kwargs = {"phase": "building"} if name == "phase_adaptive" else {}
            ranked = cls(**kwargs).rank([b, a])
            assert ranked[0].verb == "a", (
                f"{name} failed: 1-point unblocks advantage should still win"
            )

    def test_elimination_gates_fallback_preserves_all(self):
        """When all candidates fail gates, the fallback must return
        everyone — not silently drop the 'inconvenient' ones."""
        a = make_action(verb="a", unblocks=1, reduces_risk=1, readiness=1, impact=1)
        b = make_action(verb="b", unblocks=1, reduces_risk=1, readiness=2, impact=1)
        c = make_action(verb="c", unblocks=2, reduces_risk=1, readiness=1, impact=1)

        ranked = EliminationGates().rank([a, b, c])
        assert len(ranked) == 3, "all fail gates, so all must be in fallback"

    def test_geometric_mean_zero_dimension_catastrophe(self):
        """If any dimension could hit 0 (or effective 0 via score=1),
        geometric mean craters it."""
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
        """Three candidates where none dominates another — rock-paper-
        scissors pattern. Pareto must keep all three."""
        rock = make_action(verb="rock", unblocks=5, reduces_risk=1, readiness=3, impact=3)
        paper = make_action(verb="paper", unblocks=3, reduces_risk=5, readiness=1, impact=3)
        scissors = make_action(verb="scissors", unblocks=3, reduces_risk=3, readiness=5, impact=1)

        ranked = ParetoThenRank().rank([rock, paper, scissors])
        assert len(ranked) == 3, "rock-paper-scissors: no candidate is dominated"

    def test_strategy_disagreement_matrix(self):
        """The ultimate coupling detector: a 4-candidate fixture designed
        so that at least 3 different strategies produce different winners."""
        a = make_action(verb="a", unblocks=5, reduces_risk=5, readiness=2, impact=5)
        b = make_action(verb="b", unblocks=4, reduces_risk=4, readiness=4, impact=4)
        c = make_action(verb="c", unblocks=1, reduces_risk=5, readiness=5, impact=2)
        d = make_action(verb="d", unblocks=4, reduces_risk=1, readiness=4, impact=4)

        pool = [a, b, c, d]

        ws_winner = WeightedSum().rank(pool)[0]
        gm_winner = GeometricMean().rank(pool)[0]
        eg_winner = EliminationGates().rank(pool)[0]
        ff_winner = PhaseAdaptive(phase="firefighting").rank(pool)[0]

        assert ws_winner.verb == "a"
        assert gm_winner.verb == "b"
        assert eg_winner.verb == "b"
        assert ff_winner.verb == "c"

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
        chosen, ranked, reasoning, strat = choose("build something")

        assert chosen.verb == "write"
        assert chosen.artifact == "schema.sql"
        assert chosen.scores.unblocks == 5

    @patch("sparkchoice.anthropic.Anthropic")
    def test_returns_ranked_candidates(self, mock_cls):
        """Candidates are returned in ranked order, not original order."""
        mock_cls.return_value.messages.create.return_value = self._mock_response()
        chosen, ranked, reasoning, strat = choose("build something")

        assert len(ranked) == 2
        # First in ranked list should be the chosen one
        assert ranked[0] is chosen

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

    @patch("sparkchoice.anthropic.Anthropic")
    def test_payload_not_mutated(self, mock_cls):
        """_to_action should not mutate the original API response."""
        mock_cls.return_value.messages.create.return_value = self._mock_response()
        choose("build something")

        # If _to_action mutated the payload, a second call would fail
        # because "scores" would be missing from candidates
        mock_cls.return_value.messages.create.return_value = self._mock_response()
        chosen, _, _, _ = choose("build something")
        assert chosen.scores.unblocks == 5


# ── CLI ─────────────────────────────────────────────────────────────────

class TestCLI:
    def test_no_args_prints_usage(self):
        result = subprocess.run(
            [sys.executable, "sparkchoice.py"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "Usage" in result.stderr or "Usage" in result.stdout
