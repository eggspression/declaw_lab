
from langchain_core.messages.ai import AIMessage
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.messages import AnyMessage, SystemMessage, AIMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
import operator
# model = ChatOllama(
#     model="qwen3.5:9b-mlx",
#     base_url="http://localhost:11434",
#     reasoning=False,
#     temperature=0.3,
#     num_ctx=8192,
#     num_predict=2048,
# )

model = init_chat_model(
    "halogen-qwen3.8-flash-next",
    model_provider="openai",
    base_url="http://10.147.17.164:8731/v1",
    api_key = "not-needed",
    temperature=0.3,
)

@tool(parse_docstring=True) #Decorator
def multiply(a: int, b: int) -> int:
    """
    Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name : tool for tool in tools}
model_with_tools = model.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def llm_call(state: MessagesState):
    """
    LLM decides whether to call a tool or not
    """
    request = []
    request.append(SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."))
    request.extend(state["messages"])
    response = model_with_tools.invoke(request)
    return {
        "messages": [response],
        "llm_calls": state['llm_calls'] + 1
    }

def tool_node(state: MessagesState): 
    """
    Perform the tool call
    """
    result = []
    tool_calls = state["messages"][-1].tool_calls
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState):
    """
    Decide whether to continue the loop or not
    """
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return END

agent_builder = StateGraph(MessagesState)
# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

messages = [HumanMessage(content="Add 3 and 4.")]
result = agent.invoke({"messages": messages, "llm_calls":0})
for m in result["messages"]:
    m.pretty_print()