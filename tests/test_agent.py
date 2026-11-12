import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
from agent import JDInfo, Match, Review, Tailored

RESUME = "Python developer. Built REST APIs with Flask at Acme Corp for 3 years. Used MySQL daily. " * 3
JD = "We need a Python engineer with Flask and Docker experience to build APIs. " * 3


def test_route_approved_goes_to_save():
    assert agent.route_after_review({"review": {"approved": True}, "attempts": 1}) == "save"


def test_route_rejected_retries():
    assert agent.route_after_review({"review": {"approved": False}, "attempts": 1}) == "tailor"


def test_route_stops_at_max_attempts():
    state = {"review": {"approved": False}, "attempts": agent.MAX_ATTEMPTS}
    assert agent.route_after_review(state) == "save"


@pytest.mark.parametrize("jd,resume", [("", RESUME), (JD, ""), ("short", RESUME), (JD, "short"),
                                       ("x" * 30000, RESUME)])
def test_validate_rejects_bad_input(jd, resume):
    with pytest.raises(ValueError):
        agent.validate_inputs(jd, resume)


def test_validate_accepts_good_input():
    jd, resume = agent.validate_inputs("  " + JD + "  ", RESUME)
    assert jd == JD.strip()


class FakeLLM:
    """Stands in for Gemini. The reviewer rejects the first draft, then approves."""
    def __init__(self):
        self.reviews = 0
        self.tailor_prompts = []

    def with_structured_output(self, schema):
        outer = self

        class Runner:
            def invoke(self, prompt):
                if schema is JDInfo:
                    return JDInfo(role="Engineer", company="Initech", required_skills=["Python", "Docker"],
                                  keywords=["APIs"])
                if schema is Match:
                    return Match(score=70, matched_skills=["Python"], missing_skills=["Docker"])
                if schema is Tailored:
                    outer.tailor_prompts.append(prompt)
                    return Tailored(bullets=["Built APIs with Flask"], cover_letter="Hello")
                outer.reviews += 1
                if outer.reviews == 1:
                    return Review(approved=False, fabricated_claims=["Docker"], feedback="Remove Docker")
                return Review(approved=True, fabricated_claims=[], feedback="ok")
        return Runner()


def test_self_correction_loop(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(agent, "get_llm", lambda: fake)
    state = agent.run_pipeline(JD, RESUME, persist=False)
    assert state["attempts"] == 2
    assert state["review"]["approved"] is True
    assert state["first_pass_approved"] is False
    # The second tailor prompt must contain the fabricated claim to remove.
    assert "Docker" in fake.tailor_prompts[1] and "NOT in the resume" in fake.tailor_prompts[1]
    assert "Docker" in fake.tailor_prompts[0]  # missing skills are passed on the first pass too


def test_llm_failure_raises_pipeline_error(monkeypatch):
    class Broken:
        def with_structured_output(self, schema):
            raise RuntimeError("quota exceeded")
    monkeypatch.setattr(agent, "get_llm", lambda: Broken())
    monkeypatch.setattr(agent, "RETRY_BASE_DELAY", 0)
    with pytest.raises(agent.PipelineError):
        agent.run_pipeline(JD, RESUME, persist=False)


def test_db_failure_does_not_lose_results(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: FakeLLM())
    def boom(*a, **k):
        raise RuntimeError("mysql down")
    monkeypatch.setattr(agent, "save_application", boom)
    state = agent.run_pipeline(JD, RESUME, persist=True)
    assert state["tailored"]["bullets"] and "mysql down" in state["save_error"]
