"""Entry point.

Usage:
    python main.py "your research topic here"
"""

import sys
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from graph import build_graph

load_dotenv()


def run(topic: str) -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state = {
        "topic": topic,
        "past_findings": [],
        "sub_questions": [],
        "plan_approved": False,
        "sources": [],
        "draft": None,
        "feedback": None,
        "revision_count": 0,
        "final_report": None,
    }

    result = graph.invoke(initial_state, config=config)

    # The graph pauses here — human_approval_node called interrupt().
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\nProposed research plan:")
        for i, q in enumerate(payload["sub_questions"], 1):
            print(f"  {i}. {q}")

        choice = input(
            "\nPress Enter to approve, or type replacement questions "
            "separated by ';': "
        ).strip()

        if choice:
            edited = [q.strip() for q in choice.split(";") if q.strip()]
            resume_value = {"sub_questions": edited}
        else:
            resume_value = {"sub_questions": payload["sub_questions"]}

        result = graph.invoke(Command(resume=resume_value), config=config)

    print("\n--- Final report ---\n")
    print(result["final_report"])


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or input("Research topic: ")
    run(topic)