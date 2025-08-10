from langgraph.graph import MessagesState

# This is the state that will be passed between nodes in the graph.
# It is a TypedDict with a single key, "messages", which is a list of messages.
# We are simply re-exporting it here for clarity and to fit our project structure.
__all__ = ["MessagesState"]
