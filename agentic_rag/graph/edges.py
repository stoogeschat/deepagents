from typing import Literal
from langchain_openai import ChatOpenAI

from agentic_rag.core.state import MessagesState
from agentic_rag.core.models import GradeDocuments
from agentic_rag.config import (
    CHAT_MODEL_NAME,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    GRADE_PROMPT,
)

# Initialize the grader model once and reuse it
grader_model = ChatOpenAI(
    model=CHAT_MODEL_NAME,
    openai_api_base=OPENAI_API_BASE,
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """
    Determine whether the retrieved documents are relevant to the question.
    """
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)

    # Use structured output to ensure the response is in the correct format
    structured_grader = grader_model.with_structured_output(GradeDocuments)
    response = structured_grader.invoke([{"role": "user", "content": prompt}])

    score = response.binary_score

    if score.lower() == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"
