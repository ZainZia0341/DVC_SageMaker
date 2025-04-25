from ..Graph_State import State
from langgraph.types import Command, interrupt
from langchain_core.messages import ToolMessage

def human_file_selection_node(state: State):
    print("------------------- human file selection node -------------------")
    print(state)
    print("human_file_selection_node => available messages:", state["messages"])
    human_response = interrupt({"data": ""})
    print("Resumed with:", human_response.get("data"))

    # Store user feedback in files

    return Command(update= {"files": (ToolMessage(human_response, tool_call_id="tool_id_for_gemini"))})
