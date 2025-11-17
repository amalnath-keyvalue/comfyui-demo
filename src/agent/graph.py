from typing import TypedDict

from langchain_core.messages import AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from .comfyui_client import ComfyUIClient


class AgentState(TypedDict):
    messages: list
    next_action: str
    image_prompt: str
    image_data: bytes


def classify_intent(state: AgentState) -> AgentState:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    last_message = state["messages"][-1]

    system_prompt = SystemMessage(
        content="""You are a classifier. Determine if the user wants to generate an image.
Respond with EXACTLY one word: 'generate' if they want to generate an image, or 'chat' if they want to chat normally.

Examples:
- "create an image of a sunset" -> generate
- "make a picture of a cat" -> generate  
- "what is the weather?" -> chat
- "hello how are you?" -> chat"""
    )

    response = llm.invoke([system_prompt, last_message])
    intent = response.content.strip().lower()

    if "generate" in intent:
        state["next_action"] = "generate"
    else:
        state["next_action"] = "chat"

    return state


def extract_image_prompt(
    state: AgentState,
) -> AgentState:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
    )

    last_message = state["messages"][-1]

    system_prompt = SystemMessage(
        content="""Extract a detailed image generation prompt from the user's message.
Make it descriptive and suitable for stable diffusion. Return only the prompt, nothing else."""
    )

    response = llm.invoke([system_prompt, last_message])
    state["image_prompt"] = response.content.strip()

    return state


def generate_image(
    state: AgentState,
) -> AgentState:
    client = ComfyUIClient()

    def progress_callback(message: str):
        print(f"  → {message}")

    try:
        print("\nGenerating image...")
        image_data = client.generate_simple_text_image(
            state["image_prompt"], progress_callback
        )
        state["image_data"] = image_data

        response_msg = AIMessage(
            content=f"Generated image with prompt: {state['image_prompt']}"
        )
        state["messages"].append(response_msg)
    except Exception as e:
        error_msg = AIMessage(content=f"Failed to generate image: {str(e)}")
        state["messages"].append(error_msg)

    return state


def chat_response(
    state: AgentState,
) -> AgentState:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))

    return state


def route_action(
    state: AgentState,
) -> str:
    if state["next_action"] == "generate":
        return "extract_prompt"
    else:
        return "chat"


def create_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("classify", classify_intent)
    workflow.add_node("extract_prompt", extract_image_prompt)
    workflow.add_node("generate", generate_image)
    workflow.add_node("chat", chat_response)

    workflow.set_entry_point("classify")

    workflow.add_conditional_edges(
        "classify", route_action, {"extract_prompt": "extract_prompt", "chat": "chat"}
    )

    workflow.add_edge("extract_prompt", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("chat", END)

    return workflow.compile()
