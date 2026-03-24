"""Tests for SparkChoice — scoring logic, parsing, Action output, CLI."""

import json
import subprocess
import sys
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from sparkchoice import Action, Scores, choose


# ── Scores ──────────────────────────────────────────────────────────────

class TestScores:
    def test_prudence_weights(self):
        s = Scores(unblocks=5, reduces_risk=5, readiness=5, impact=5)
        # 5*3 + 5*2 + 5*2 + 5*1 = 15+10+10+5 = 40
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
        # 1*3 + 1*2 + 1*2 + 1*1 = 8
        assert s.prudence == 8


# ── Action ──────────────────────────────────────────────────────────────

class TestAction:
    def make_action(self, **overrides):
        defaults = dict(
            verb="write",
            artifact="spec.md",
            description="Write a spec.",
            rationale="Unblocks everything.",
            scores=Scores(unblocks=4, reduces_risk=3, readiness=5, impact=3),
            inputs=["requirements.txt"],
            outputs=["spec.md"],
        )
        defaults.update(overrides)
        return Action(**defaults)

    def test_to_prompt_contains_verb_and_artifact(self):
        a = self.make_action()
        prompt = a.to_prompt()
        assert "write" in prompt
        assert "spec.md" in prompt

    def test_to_prompt_includes_inputs_and_outputs(self):
        a = self.make_action()
        prompt = a.to_prompt()
        assert "requirements.txt" in prompt
        assert "spec.md" in prompt

    def test_to_prompt_omits_empty_sections(self):
        a = self.make_action(inputs=[], outputs=[])
        prompt = a.to_prompt()
        assert "## Inputs" not in prompt
        assert "## Expected outputs" not in prompt

    def test_asdict_roundtrip(self):
        a = self.make_action()
        d = asdict(a)
        assert d["verb"] == "write"
        assert d["scores"]["unblocks"] == 4


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
    "chosen": 0,
    "reasoning": "Schema is ready now and unblocks the API.",
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
        chosen, candidates, reasoning = choose("build something")

        assert chosen.verb == "write"
        assert chosen.artifact == "schema.sql"
        assert chosen.scores.unblocks == 5

    @patch("sparkchoice.anthropic.Anthropic")
    def test_returns_all_candidates(self, mock_cls):
        mock_cls.return_value.messages.create.return_value = self._mock_response()
        chosen, candidates, reasoning = choose("build something")

        assert len(candidates) == 2
        assert candidates[1].verb == "build"

    @patch("sparkchoice.anthropic.Anthropic")
    def test_returns_reasoning(self, mock_cls):
        mock_cls.return_value.messages.create.return_value = self._mock_response()
        _, _, reasoning = choose("build something")

        assert "Schema" in reasoning

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
