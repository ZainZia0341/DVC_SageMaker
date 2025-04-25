from ..Graph_State import State
from langchain_core.messages import RemoveMessage

def filter_node(state: State):
    filtered_msgs = [RemoveMessage(id=m.id) for m in state["messages"][:-10]]
    
    print("------------------- filter_node -------------------")
    print(state)
    print("Messages to remove:", filtered_msgs)
    
    return {"messages": filtered_msgs}
