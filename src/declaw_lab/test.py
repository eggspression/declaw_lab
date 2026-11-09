
from langchain_core.messages.ai import AIMessage
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.messages import AnyMessage, SystemMessage, AIMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
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

response = model.invoke("P = NP")
print(response)