from functools import partial

from roboastra.util.message import MessagesState
from langchain.messages import HumanMessage
from roboastra.agent.hri import HRI
from langgraph.graph import StateGraph, START, END

from roboastra.util.node import llm_call, tool_node, should_continue



def build_roboastra(save_agent_graph_path = None):
    hri = HRI()
    agent = hri.build_agent()

    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", partial(llm_call, agent))
    agent_builder.add_node("tool_node", partial(tool_node, hri.tools_by_name))

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile the agent
    roboastra = agent_builder.compile()

    # Show the agent
    roboastra.get_graph(xray=True).draw_mermaid_png(output_file_path = save_agent_graph_path)

    return roboastra



def run():
    roboastra = build_roboastra(save_agent_graph_path="roboastra_graph.png")
    messages = [HumanMessage(content="Do insert_plug, if insert_plug fails, call clear_workspace. If clear_workspace fails, call final_failure_recovery. If all else fails, call call_human.")]
    messages = roboastra.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
