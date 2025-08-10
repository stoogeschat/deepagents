from langchain_openai import ChatOpenAI

from agentic_rag.core.state import MessagesState
from agentic_rag.config import (
    CHAT_MODEL_NAME,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    REWRITE_PROMPT,
    GENERATE_PROMPT,
)


class GraphNodes:
    """
    A class that encapsulates all the node logic for the agentic RAG graph.
    """

    def __init__(self, retriever_tool):
        """
        Initializes the GraphNodes class with the retriever tool and local models.
        """
        self.retriever_tool = retriever_tool
        self.response_model = ChatOpenAI(
            model=CHAT_MODEL_NAME,
            openai_api_base=OPENAI_API_BASE,
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
        )

    def generate_query_or_respond(self, state: MessagesState):
        """
        Call the model to generate a response based on the current state.
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
