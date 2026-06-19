"""Builds and compiles the LangGraph pipeline.

memory lookup -> supervisor -> human approval
  -> parallel researchers (fan-out/fan-in) -> writer -> critic
  -> (back to writer, or finalize) -> end
"""

import os

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from nodes import (
    memory_lookup_node,
    supervisor_node,
    human_approval_node,
    researcher_node,
    writer_node,
    critic_node,
    finalize_node,
    MAX_REVISIONS,
)
from state import ResearchState


def fan_out_researchers(state: ResearchState):
    """One Send per sub-question — these run in parallel."""
    return [Send("researcher", {"sub_question": q}) for q in state["sub_questions"]]


def review_decision(state: ResearchState):
    """Critic's verdict decides: back to the writer, or done."""
    no_feedback = state.get("feedback") is None
    out_of_revisions = state.get("revision_count", 0) >= MAX_REVISIONS
    return "finalize" if (no_feedback or out_of_revisions) else "writer"


def build_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("memory_lookup", memory_lookup_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "memory_lookup")
    builder.add_edge("memory_lookup", "supervisor")
    builder.add_edge("supervisor", "human_approval")
    builder.add_conditional_edges("human_approval", fan_out_researchers, ["researcher"])
    builder.add_edge("researcher", "writer")  # fan-in: waits for every researcher
    builder.add_edge("writer", "critic")
    builder.add_conditional_edges(
        "critic", review_decision, {"finalize": "finalize", "writer": "writer"}
    )
    builder.add_edge("finalize", END)

    # A checkpointer is required for interrupt()/resume to work at all —
    # it's what lets the graph pause at human_approval and pick back up later.
    pool = ConnectionPool(
        conninfo=os.environ["POSTGRES_URI"],
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()  # creates the checkpoint tables if they don't exist yet
    return builder.compile(checkpointer=checkpointer)