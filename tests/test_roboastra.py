from unittest.mock import patch

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables.graph import Graph
from langchain_openai import ChatOpenAI

from roboastra.roboastra import build_roboastra


def test_graph_runs_with_and_without_tools():
    answer = AIMessage(content="7")
    tool_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "call_specialist",
            "args": {"specialist_name": "call_human"},
            "id": "call_1",
        }],
    )
    final_answer = AIMessage(content="Waiting for a human.")
    with (
        patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key"}),
        patch.object(Graph, "draw_mermaid_png"),
        patch.object(ChatOpenAI, "invoke", side_effect=[
            answer, tool_call, final_answer,
        ]) as invoke,
    ):
        graph = build_roboastra()
        question = HumanMessage(content="Add 3 and 4.")
        result = graph.invoke({"messages": [question]})
        assert result["messages"] == [question, answer]
        assert result["llm_calls"] == 1

        request = HumanMessage(content="Ask a human for help.")
        result = graph.invoke({"messages": [request]})
        tool_result = ToolMessage(
            content="Has informed human, awaiting response", tool_call_id="call_1"
        )
        assert result["messages"] == [request, tool_call, tool_result, final_answer]
        assert result["llm_calls"] == 2
        assert invoke.call_count == 3
        assert invoke.call_args.args[0][-1] == tool_result


if __name__ == "__main__":
    test_graph_runs_with_and_without_tools()
    print("Graph regression check passed.")
