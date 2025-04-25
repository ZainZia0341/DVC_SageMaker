from typing import Annotated, List
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

#######################################################
# 1) Define your custom State
#######################################################
class State(TypedDict):
    messages: Annotated[list, add_messages]
    tags: List[str]  
    files: str
    match_threshold: int

graph_builder = StateGraph(State)