# my_graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import the node functions
import os
from dotenv import load_dotenv
from .nodes.filter import filter_node
from .nodes.tags import tags_generation_node
from .nodes.search import file_search_node
from .nodes.human_select import human_file_selection_node
from .nodes.s3_fetch import get_file_from_s3_node
from .nodes.llm_response import llm_response_generation_node
import ast

from .Graph_State import State
from langgraph.types import Command

load_dotenv()

graph_builder = StateGraph(State)
memory = MemorySaver()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# LangSmith for Error Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

graph_builder.add_node("filter_node", filter_node)
graph_builder.add_node("tags_generation_node", tags_generation_node)
graph_builder.add_node("file_search_node", file_search_node)
graph_builder.add_node("human_file_selection_node", human_file_selection_node)
graph_builder.add_node("get_file_from_s3_node", get_file_from_s3_node)
graph_builder.add_node("llm_response_generation_node", llm_response_generation_node)

graph_builder.add_edge(START, "filter_node")
graph_builder.add_edge("filter_node", "tags_generation_node")
graph_builder.add_edge("tags_generation_node", "file_search_node")
graph_builder.add_edge("file_search_node", "human_file_selection_node")
graph_builder.add_edge("human_file_selection_node", "get_file_from_s3_node")
graph_builder.add_edge("get_file_from_s3_node", "llm_response_generation_node")
graph_builder.add_edge("llm_response_generation_node", END)

graph = graph_builder.compile(checkpointer=memory)


#######################################################
# 5) Helper Functions to Run/Resume Graph
#######################################################
def run_graph(initial_state: dict, config: dict):
    """
    Stream the graph from an initial state until it finishes or interrupts.
    Returns a tuple: (final_events_list, paused_bool, interrupt_data).
    The final_events_list will contain one aggregated event that merges
    the human message, tags, and files from all intermediate events.
    """
    events_list = []
    paused = False
    interrupt_data = None

    # Collect all events from the graph
    for event in graph.stream(initial_state, config, stream_mode="values"):
        print("------------------------ stream of chat endpoint before interrupt -------------------------")
        print(event)
        events_list.append(event)
        if "interrupt" in event:
            paused = True
            interrupt_data = event["interrupt"]
            break

    # Aggregate events into one final event.
    aggregated_event = {"messages": [], "tags": [], "files": []}

    for event in events_list:
        # Merge messages: for this example, we assume that the human message is the one you want.
        if "messages" in event and event["messages"]:
            for message in event["messages"]:
                # Only consider human messages (or you could change the criteria as needed)
                if message.type == "human" and message.content not in aggregated_event["messages"]:
                    aggregated_event["messages"].append(message.content)

        # Merge tags: they could come as a list or a comma-separated string.
        if "tags" in event and event["tags"]:
            if isinstance(event["tags"], str):
                # If it's a non-empty string, split by comma (if applicable)
                tag_list = [tag.strip() for tag in event["tags"].split(",") if tag.strip()]
            elif isinstance(event["tags"], list):
                tag_list = event["tags"]
            else:
                tag_list = []
            for tag in tag_list:
                if tag not in aggregated_event["tags"]:
                    aggregated_event["tags"].append(tag)

        # Merge files: files may come as a comma-separated string or a list.
        if "files" in event and event["files"]:
            if isinstance(event["files"], str):
                file_list = [f.strip() for f in event["files"].split(",") if f.strip()]
            elif isinstance(event["files"], list):
                file_list = event["files"]
            else:
                file_list = []
            for f in file_list:
                if f not in aggregated_event["files"]:
                    aggregated_event["files"].append(f)

    # Optionally, if only one human message is desired, you could choose the first one:
    if aggregated_event["messages"]:
        aggregated_event["messages"] = [aggregated_event["messages"][0]]
    
    # Return the aggregated event as a single-element list.
    final_events = [aggregated_event]
    print(f"Aggregated final events: {final_events}")
    return final_events, paused, interrupt_data

def resume_graph(cmd: Command, config: dict):
    """
    Stream the graph for resuming from an interrupt and aggregate all events.
    Returns a single aggregated event with the following keys:
      - question: the original human message,
      - answer: the latest AI response,
      - tags: a unique list of tags,
      - files: the list of file names as provided by the ToolMessage in the state.
    """
    events_list = []
    for event in graph.stream(cmd, config, stream_mode="values"):
        print("------------------------ stream of chat endpoint after interrupt -------------------------")
        print(event)
        events_list.append(event)

    # Initialize aggregated event with separate keys for question, answer, tags, and files.
    aggregated_event = {
        "question": "",
        "answer": "",
        "tags": [],
        "files": []  # final files will come from the ToolMessage if present
    }

    # Process each collected event.
    for event in events_list:
        # Process messages: store the first human message as question and update answer with each AI message.
        if "messages" in event and event["messages"]:
            for msg in event["messages"]:
                if msg.type == "human" and not aggregated_event["question"]:
                    aggregated_event["question"] = msg.content
                elif msg.type == "ai":
                    aggregated_event["answer"] = msg.content

        # Process tags by merging unique items.
        if "tags" in event and event["tags"]:
            if isinstance(event["tags"], str):
                tag_list = [tag.strip() for tag in event["tags"].split(",") if tag.strip()]
            elif isinstance(event["tags"], list):
                tag_list = event["tags"]
            else:
                tag_list = []
            for tag in tag_list:
                if tag not in aggregated_event["tags"]:
                    aggregated_event["tags"].append(tag)

        # Process files:
        # First, check if the files field is a ToolMessage with type 'tool'.
        if "files" in event and event["files"]:
            files_obj = event["files"]
            # If it's a ToolMessage (detected by having an attribute 'type' equal to "tool")
            if hasattr(files_obj, "type") and files_obj.type == "tool":
                try:
                    data_dict = ast.literal_eval(files_obj.content)
                    if isinstance(data_dict, dict) and "data" in data_dict:
                        file_list = data_dict["data"]
                        if not isinstance(file_list, list):
                            file_list = [file_list]
                        # Override aggregated files with the selected files from the ToolMessage.
                        aggregated_event["files"] = file_list
                except Exception as e:
                    print("Error parsing files from ToolMessage content:", e)
            else:
                # Fallback: if not a ToolMessage, try to merge as before.
                if isinstance(files_obj, str):
                    file_list = [f.strip() for f in files_obj.split(",") if f.strip()]
                    for f in file_list:
                        if f not in aggregated_event["files"]:
                            aggregated_event["files"].append(f)
                elif isinstance(files_obj, list):
                    for f in files_obj:
                        if f not in aggregated_event["files"]:
                            aggregated_event["files"].append(f)

    # In case no question was found, set it to an empty string.
    if not aggregated_event["question"]:
        aggregated_event["question"] = ""
    
    # Wrap the aggregated event in a list.
    final_events = [aggregated_event]
    print("Aggregated resume events:", final_events)
    return final_events



# #######################################################
# # 6) (Optional) Local Test
# #######################################################
# if __name__ == "__main__":
#     initial_state = {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Hi! I'd like info about my financial transactions. Please find relevant documents."
#             }
#         ],
#         "tags": "",
#         "files": ""
#     }
#     config = {"configurable": {"thread_id": "demo-1"}}

#     events, paused, interrupt_data = run_graph(initial_state, config)
#     print("Events so far:", events)
#     if paused:
#         print("Interrupted with data:", interrupt_data)
#         # Resume example
#         cmd = Command(resume={"data": "Sure, proceed with Document1.txt"})
#         resume_events = resume_graph(cmd, config)
#         print("Resume events:", resume_events)
