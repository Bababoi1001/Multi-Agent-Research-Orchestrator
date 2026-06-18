"""Shared state schema for the multi-agent research assistant graph.

This TypedDict flows through every node in the graph. Each node reads
the fields it needs and returns a partial update; LangGraph merges
updates back into the running state.
"""

import operator
from typing import TypedDict, List, Optional, Annotated


class Source(TypedDict):
    """A single piece of evidence gathered by a researcher."""
    sub_question: str
    url: str
    summary: str


class ResearchState(TypedDict):
    # --- input ---
    topic: str

    # --- memory (cross-session) ---
    past_findings: List[str]          # relevant notes retrieved from the vector store

    # --- supervisor output ---
    sub_questions: List[str]          # decomposed research questions

    # --- human-in-the-loop ---
    plan_approved: bool               # set True once the user approves/edits the plan

    # --- researcher output ---
    sources: Annotated[List[Source], operator.add]  # merged across parallel researchers

    # --- writer / critic loop ---
    draft: Optional[str]
    feedback: Optional[str]           # critic's notes if the draft needs work
    revision_count: int               # capped to stop infinite loops

    # --- final output ---
    final_report: Optional[str]