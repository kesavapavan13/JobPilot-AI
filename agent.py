"""JobPilot AI: LangGraph multi-agent pipeline.

Analyzer -> Matcher -> Tailor -> Reviewer -> (retry Tailor on fabricated claims, max 3) -> Tracker
"""
import logging
import os
import time
from typing import Iterator, List, Tuple, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from db import content_hash, save_application

load_dotenv()
log = logging.getLogger("jobpilot")

MAX_ATTEMPTS = 3        # Tailor/Reviewer rounds before giving up
LLM_RETRIES = 3         # retries per LLM call (rate limits, malformed output)
RETRY_BASE_DELAY = 2.0  # seconds; doubles each retry
MIN_JD_CHARS = 80
MIN_RESUME_CHARS = 150
MAX_INPUT_CHARS = 20_000


class PipelineError(RuntimeError):
    """Raised when an agent fails after all retries."""


# ---------- LLM ----------
def get_llm():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise PipelineError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return ChatGoogleGenerativeAI(
        model=os.getenv("MODEL", "gemini-3.5-flash-lite"),
        temperature=0,
        google_api_key=key,
    )


def _invoke(schema, prompt: str):
    """Call the LLM with structured output, retrying on errors or empty results."""
    last_err = None
    for attempt in range(LLM_RETRIES):
        try:
            result = get_llm().with_structured_output(schema).invoke(prompt)
            if result is None:
                raise ValueError("model returned no structured output")
            return result
        except PipelineError:
            raise
        except Exception as e:  # network, quota, parsing: all retryable
            last_err = e
            log.warning("%s call failed (try %d/%d): %s", schema.__name__, attempt + 1, LLM_RETRIES, e)
            if attempt < LLM_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
    raise PipelineError(f"{schema.__name__} step failed after {LLM_RETRIES} tries: {last_err}") from last_err


# ---------- Structured output schemas ----------
class JDInfo(BaseModel):
    role: str
    company: str = Field(description="Company name, or 'Unknown' if not stated")
    required_skills: List[str]
    keywords: List[str]


class Match(BaseModel):
    score: int = Field(ge=0, le=100)
    matched_skills: List[str]
    missing_skills: List[str]


class Tailored(BaseModel):
    bullets: List[str] = Field(description="Rewritten resume bullets, 6-8 items")
    cover_letter: str = Field(description="Cover letter under 300 words")


class Review(BaseModel):
    approved: bool
    fabricated_claims: List[str]
    feedback: str


# ---------- Graph state ----------
class State(TypedDict, total=False):
    jd: str
    resume: str
    jd_info: dict
    match: dict
    tailored: dict
    review: dict
    attempts: int
    first_pass_approved: bool
    saved_id: int
    save_error: str


def validate_inputs(jd: str, resume: str) -> Tuple[str, str]:
    """Trim and sanity-check inputs before spending tokens."""
    jd, resume = (jd or "").strip(), (resume or "").strip()
    if len(jd) < MIN_JD_CHARS:
        raise ValueError(f"Job description is too short (need at least {MIN_JD_CHARS} characters).")
    if len(resume) < MIN_RESUME_CHARS:
        raise ValueError(f"Resume is too short (need at least {MIN_RESUME_CHARS} characters).")
    if len(jd) > MAX_INPUT_CHARS or len(resume) > MAX_INPUT_CHARS:
        raise ValueError(f"Inputs are limited to {MAX_INPUT_CHARS:,} characters each.")
    return jd, resume


# ---------- Agents (nodes) ----------
def analyzer(state: State):
    info = _invoke(
        JDInfo,
        f"Extract the role, company, required skills and ATS keywords:\n\n{state['jd']}",
    )
    return {"jd_info": info.model_dump(), "attempts": 0}


def matcher(state: State):
    m = _invoke(
        Match,
        "Score (0-100) how well this resume fits the required skills. "
        "Only count skills actually evidenced in the resume.\n\n"
        f"Required skills: {state['jd_info']['required_skills']}\n\nResume:\n{state['resume']}",
    )
    return {"match": m.model_dump()}


def tailor(state: State):
    review = state.get("review") or {}
    claims = review.get("fabricated_claims") or []
    feedback = review.get("feedback") or "None"
    if claims:
        feedback += f"\nThese claims are NOT in the resume. Remove them: {claims}"
    match = state["match"]
    out = _invoke(
        Tailored,
        f"Rewrite the resume bullets and write a cover letter for the role "
        f"'{state['jd_info']['role']}' at '{state['jd_info']['company']}'.\n"
        f"Use keywords where truthful: {state['jd_info']['keywords']}.\n"
        f"Skills the resume evidences (emphasize these): {match['matched_skills']}\n"
        f"Skills the resume does NOT evidence (never claim these): {match['missing_skills']}\n"
        "STRICT RULE: use only facts present in the resume. Never invent skills, tools, "
        "employers, or numbers.\n"
        f"Reviewer feedback to fix: {feedback}\n\nResume:\n{state['resume']}",
    )
    return {"tailored": out.model_dump(), "attempts": state.get("attempts", 0) + 1}


def reviewer(state: State):
    t = state["tailored"]
    r = _invoke(
        Review,
        "You are a strict fact-checker. Compare the ORIGINAL resume to the NEW content. "
        "List any skill, tool, employer, or number in the new content that is not supported "
        "by the original. Approve only if there are none.\n\n"
        f"ORIGINAL:\n{state['resume']}\n\nNEW BULLETS:\n{t['bullets']}\n\nNEW COVER LETTER:\n{t['cover_letter']}",
    )
    update = {"review": r.model_dump()}
    if state.get("attempts") == 1:
        update["first_pass_approved"] = bool(r.approved)
    return update


def route_after_review(state: State):
    if state["review"]["approved"] or state["attempts"] >= MAX_ATTEMPTS:
        return "save"
    return "tailor"  # self-correction loop


def make_save_node(persist: bool):
    def save(state: State):
        if not persist:
            return {}
        try:
            app_id = save_application(
                state["jd_info"]["company"],
                state["jd_info"]["role"],
                state["match"]["score"],
                state["attempts"],
                state["review"]["approved"],
                bullets=state["tailored"]["bullets"],
                cover_letter=state["tailored"]["cover_letter"],
                missing_skills=state["match"]["missing_skills"],
                content_hash_value=content_hash(state["jd"], state["resume"]),
            )
            return {"saved_id": app_id}
        except Exception as e:  # never lose the generated content because the DB is down
            log.error("Could not save application: %s", e)
            return {"save_error": str(e)}
    return save


def build_graph(persist: bool = True):
    """Compile the pipeline. Pass persist=False to skip the MySQL tracker (used by eval/tests)."""
    g = StateGraph(State)
    nodes = [("analyzer", analyzer), ("matcher", matcher), ("tailor", tailor),
             ("reviewer", reviewer), ("save", make_save_node(persist))]
    for name, fn in nodes:
        g.add_node(name, fn)
    g.add_edge(START, "analyzer")
    g.add_edge("analyzer", "matcher")
    g.add_edge("matcher", "tailor")
    g.add_edge("tailor", "reviewer")
    g.add_conditional_edges("reviewer", route_after_review, {"tailor": "tailor", "save": "save"})
    g.add_edge("save", END)
    return g.compile()


def stream_pipeline(jd: str, resume: str, persist: bool = True) -> Iterator[Tuple[str, dict, State]]:
    """Run the graph, yielding (node_name, node_update, merged_state) after each node finishes."""
    jd, resume = validate_inputs(jd, resume)
    state: State = {"jd": jd, "resume": resume}
    for chunk in build_graph(persist).stream(state, stream_mode="updates"):
        for node, update in chunk.items():
            state.update(update or {})
            log.info("node=%s done (attempts=%s)", node, state.get("attempts"))
            yield node, update or {}, state


def run_pipeline(jd: str, resume: str, persist: bool = True) -> State:
    final = None
    for _, _, final in stream_pipeline(jd, resume, persist):
        pass
    return final
