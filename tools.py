"""Web search tool used by the researcher nodes.

Wraps langchain-tavily so a researcher node can pass in one
sub-question and get back a list of Source dicts, ready to drop
straight into state["sources"].
"""

from typing import List

from langchain_tavily import TavilySearch

from state import Source

_search = TavilySearch(max_results=4, topic="general")


def search_web(sub_question: str) -> List[Source]:
    """Run a web search for one sub-question, return Source dicts."""
    raw = _search.invoke({"query": sub_question})
    results = raw.get("results", []) if isinstance(raw, dict) else raw

    sources: List[Source] = []
    for r in results:
        sources.append(
            {
                "sub_question": sub_question,
                "url": r.get("url", ""),
                # keep each summary bounded so it doesn't blow up the
                # writer's input token count later
                "summary": r.get("content", "")[:1000],
            }
        )
    return sources