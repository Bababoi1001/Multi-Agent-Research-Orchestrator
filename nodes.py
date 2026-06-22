"""The agent functions — one per graph node.

Each function takes the current state and returns a partial update.
LangGraph merges that update back into the running state.
"""

import logging
from typing import List

from langchain_ollama import ChatOllama
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from memory import query_past_findings, store_findings
from state import ResearchState
from tools import search_web

logger = logging.getLogger(__name__)

MAX_REVISIONS = 2

# --- Local testing via Ollama ---
# Using one pulled model (llama3.2:3b) for both roles to keep this within
# 8GB VRAM. Swap back to ChatAnthropic (claude-haiku-4-5 / claude-sonnet-4-6)
# for production-quality runs — small local models are noticeably less
# reliable with structured output (SubQuestions / CriticReview below).
fast_llm = ChatOllama(model="llama3.2:3b", temperature=0)
quality_llm = ChatOllama(model="llama3.2:3b", temperature=0.3)


class SubQuestions(BaseModel):
    sub_questions: List[str] = Field(
        description="3-4 focused research questions that together cover the topic"
    )


class CriticReview(BaseModel):
    approved: bool = Field(description="True if the draft fully answers the topic")
    feedback: str = Field(
        description="Specific, actionable feedback if not approved; empty if approved"
    )


def memory_lookup_node(state: ResearchState) -> dict:
    logger.info("[memory_lookup] checking past findings for topic: %r", state["topic"])
    findings = query_past_findings(state["topic"])
    logger.info("[memory_lookup] found %d relevant past finding(s)", len(findings))
    return {"past_findings": findings}


def supervisor_node(state: ResearchState) -> dict:
    logger.info("[supervisor] decomposing topic into sub-questions...")
    context = "\n".join(state.get("past_findings", [])) or "None"
    prompt = (
        f"Topic: {state['topic']}\n\n"
        f"Relevant findings from past research (if any):\n{context}\n\n"
        "Break this topic into 3-4 focused sub-questions that together give a "
        "complete picture. Don't repeat ground already covered above."
    )
    structured_llm = fast_llm.with_structured_output(SubQuestions)
    result = structured_llm.invoke(prompt)
    logger.info(
        "[supervisor] produced %d sub-question(s): %s",
        len(result.sub_questions),
        result.sub_questions,
    )
    return {"sub_questions": result.sub_questions, "plan_approved": False}


def human_approval_node(state: ResearchState) -> dict:
    """Pauses the graph so a human can approve or edit the plan before
    any search calls (and their cost) happen.
    """
    logger.info("[human_approval] pausing graph for plan approval...")
    decision = interrupt(
        {
            "message": "Approve this research plan, or edit the sub-questions.",
            "sub_questions": state["sub_questions"],
        }
    )
    edited = decision.get("sub_questions", state["sub_questions"])
    logger.info("[human_approval] resumed with %d sub-question(s)", len(edited))
    return {"sub_questions": edited, "plan_approved": True}


def researcher_node(state: dict) -> dict:
    """Runs once per sub-question — spawned in parallel via Send.

    `state` here is the small payload Send hands it: {"sub_question": ...},
    not the full ResearchState. See graph.py for the fan-out wiring.
    """
    sub_question = state["sub_question"]
    logger.info("[researcher] searching for: %r", sub_question)
    sources = search_web(sub_question)
    logger.info("[researcher] found %d source(s) for: %r", len(sources), sub_question)
    return {"sources": sources}


def writer_node(state: ResearchState) -> dict:
    sources_text = "\n\n".join(
        f"[{s['sub_question']}] {s['summary']} (source: {s['url']})"
        for s in state["sources"]
    )
    feedback = state.get("feedback")
    if feedback:
        logger.info("[writer] revising draft based on critic feedback...")
        prompt = (
            f"Topic: {state['topic']}\n\nPrevious draft:\n{state['draft']}\n\n"
            f"Critic feedback to address:\n{feedback}\n\n"
            f"Sources:\n{sources_text}\n\nRevise the report to address the feedback."
        )
    else:
        logger.info("[writer] drafting initial report from %d source(s)...", len(state["sources"]))
        prompt = (
            f"Topic: {state['topic']}\n\nSources:\n{sources_text}\n\n"
            "Write a clear, well-organized research report answering the topic, "
            "citing sources inline by URL."
        )
    response = quality_llm.invoke(prompt)
    revision_count = state.get("revision_count", 0)
    new_revision_count = revision_count + 1 if feedback else revision_count
    logger.info(
        "[writer] draft ready (%d chars, revision_count=%d)",
        len(response.content),
        new_revision_count,
    )
    return {
        "draft": response.content,
        "revision_count": new_revision_count,
    }


def critic_node(state: ResearchState) -> dict:
    logger.info("[critic] reviewing draft...")
    structured_llm = quality_llm.with_structured_output(CriticReview)
    prompt = (
        f"Topic: {state['topic']}\n\nDraft report:\n{state['draft']}\n\n"
        "Does this draft fully and accurately answer the topic? Approve it, "
        "or give specific, actionable feedback for revision."
    )
    review = structured_llm.invoke(prompt)
    if review.approved:
        logger.info("[critic] draft approved")
    else:
        logger.info("[critic] draft needs revision: %s", review.feedback)
    return {"feedback": None if review.approved else review.feedback}


def finalize_node(state: ResearchState) -> dict:
    logger.info("[finalize] storing %d finding(s) to memory and finalizing report", len(state["sources"]))
    for source in state["sources"]:
        store_findings(state["topic"], source["sub_question"], source["summary"])
    return {"final_report": state["draft"]}