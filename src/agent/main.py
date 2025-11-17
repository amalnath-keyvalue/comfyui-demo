from io import BytesIO

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from PIL import Image

from .comfyui_manager import ComfyUIManager
from .graph import create_agent

load_dotenv()


def run_agent(
    user_input: str,
):
    agent = create_agent()

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "next_action": "",
        "image_prompt": "",
        "image_data": b"",
    }

    result = agent.invoke(initial_state)

    if result["image_data"]:
        print(f"\n{result['messages'][-1].content}")
        image = Image.open(BytesIO(result["image_data"]))
        print("Opening image...")
        image.show()
    else:
        print(f"\n{result['messages'][-1].content}")


def main():
    manager = ComfyUIManager()
    if not manager.is_running():
        if not manager.start_server():
            print("Error: Could not start ComfyUI server")
            return

    print("ComfyUI Agent - Type 'exit' to quit")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            run_agent(user_input)
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
