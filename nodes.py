"""The agent functions — one per graph node.

Each function takes the current state and returns a partial update.
LangGraph merges that update back into the running state.
"""

from typing import List

from langchain_anthropic import ChatAnthropic
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from memory import query_past_findings, store_findings
from state import ResearchState
from tools import search_web

MAX_REVISIONS = 2

fast_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
quality_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.3)


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
    findings = query_past_findings(state["topic"])
    return {"past_findings": findings}


def supervisor_node(state: ResearchState) -> dict:
    context = "\n".join(state.get("past_findings", [])) or "None"
    prompt = (
        f"Topic: {state['topic']}\n\n"
        f"Relevant findings from past research (if any):\n{context}\n\n"
        "Break this topic into 3-4 focused sub-questions that together give a "
        "complete picture. Don't repeat ground already covered above."
    )
    structured_llm = fast_llm.with_structured_output(SubQuestions)
    result = structured_llm.invoke(prompt)
    return {"sub_questions": result.sub_questions, "plan_approved": False}


def human_approval_node(state: ResearchState) -> dict:
    """Pauses the graph so a human can approve or edit the plan before
    any search calls (and their cost) happen.
    """
    decision = interrupt(
        {
            "message": "Approve this research plan, or edit the sub-questions.",
            "sub_questions": state["sub_questions"],
        }
    )
    edited = decision.get("sub_questions", state["sub_questions"])
    return {"sub_questions": edited, "plan_approved": True}


def researcher_node(state: dict) -> dict:
    """Runs once per sub-question — spawned in parallel via Send.

    `state` here is the small payload Send hands it: {"sub_question": ...},
    not the full ResearchState. See graph.py for the fan-out wiring.
    """
    sub_question = state["sub_question"]
    return {"sources": search_web(sub_question)}


def writer_node(state: ResearchState) -> dict:
    sources_text = "\n\n".join(
        f"[{s['sub_question']}] {s['summary']} (source: {s['url']})"
        for s in state["sources"]
    )
    feedback = state.get("feedback")
    if feedback:
        prompt = (
            f"Topic: {state['topic']}\n\nPrevious draft:\n{state['draft']}\n\n"
            f"Critic feedback to address:\n{feedback}\n\n"
            f"Sources:\n{sources_text}\n\nRevise the report to address the feedback."
        )
    else:
        prompt = (
            f"Topic: {state['topic']}\n\nSources:\n{sources_text}\n\n"
            "Write a clear, well-organized research report answering the topic, "
            "citing sources inline by URL."
        )
    response = quality_llm.invoke(prompt)
    revision_count = state.get("revision_count", 0)
    return {
        "draft": response.content,
        "revision_count": revision_count + 1 if feedback else revision_count,
    }


def critic_node(state: ResearchState) -> dict:
    structured_llm = quality_llm.with_structured_output(CriticReview)
    prompt = (
        f"Topic: {state['topic']}\n\nDraft report:\n{state['draft']}\n\n"
        "Does this draft fully and accurately answer the topic? Approve it, "
        "or give specific, actionable feedback for revision."
    )
    review = structured_llm.invoke(prompt)
    return {"feedback": None if review.approved else review.feedback}


def finalize_node(state: ResearchState) -> dict:
    for source in state["sources"]:
        store_findings(state["topic"], source["sub_question"], source["summary"])
    return {"final_report": state["draft"]}