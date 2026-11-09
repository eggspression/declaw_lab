
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END


# 1. Define shared state
class AgentState(TypedDict):
    messages: Annotated[list[str], add]
    llm_calls: int


# 2. Define nodes
def model_node(state: AgentState):
    print("MODEL received:", state)

    return {
        "messages": ["Assistant requests add(3, 4)"],
        "llm_calls": state["llm_calls"] + 1
    }


def tool_node(state: AgentState):
    print("TOOL received:", state)

    return {
        "messages": ["Calculator result: 7"]
    }


# 3. Build the graph
builder = StateGraph(AgentState)

builder.add_node("model", model_node)
builder.add_node("tool", tool_node)

builder.add_edge(START, "model")
builder.add_edge("model", "tool")
builder.add_edge("tool", END)

graph = builder.compile()


# 4. Execute
result = graph.invoke({
    "messages": ["User: Calculate 3 + 4"],
    "llm_calls": 0
})

print("FINAL:", result)
