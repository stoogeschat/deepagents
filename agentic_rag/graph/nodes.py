from langchain.chat_models import init_chat_model

from agentic_rag.core.state import MessagesState
from agentic_rag.config import (
    RESPONSE_MODEL,
    REWRITE_PROMPT,
    GENERATE_PROMPT,
)


class GraphNodes:
    """
    A class that encapsulates all the node logic for the agentic RAG graph.
    """

    def __init__(self, retriever_tool):
        """
        Initializes the GraphNodes class with the retriever tool.

        Args:
            retriever_tool: The retriever tool to be used by the nodes.
        """
        self.retriever_tool = retriever_tool
        self.response_model = init_chat_model(RESPONSE_MODEL, temperature=0)

    def generate_query_or_respond(self, state: MessagesState):
        """
        Call the model to generate a response based on the current state.
        Given the question, it will decide to retrieve using the retriever tool,
        or simply respond to the user.
        """
        response = self.response_model.bind_tools([self.retriever_tool]).invoke(
            state["messages"]
        )
        return {"messages": [response]}

    def rewrite_question(self, state: MessagesState):
        """
        Rewrite the original user question to be more specific.
        """
        messages = state["messages"]
        question = messages[0].content
        prompt = REWRITE_PROMPT.format(question=question)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [{"role": "user", "content": response.content}]}

    def generate_answer(self, state: MessagesState):
        """
        Generate an answer using the retrieved context.
        """
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}
