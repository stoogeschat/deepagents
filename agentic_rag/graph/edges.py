from typing import Literal
from langchain.chat_models import init_chat_model

from agentic_rag.core.state import MessagesState
from agentic_rag.core.models import GradeDocuments
from agentic_rag.config import GRADER_MODEL, GRADE_PROMPT

# Initialize the grader model once and reuse it
grader_model = init_chat_model(GRADER_MODEL, temperature=0)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """
    Determine whether the retrieved documents are relevant to the question.

    Args:
        state: The current graph state.

    Returns:
        A string indicating the next node to call.
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
