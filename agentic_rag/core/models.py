from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """
    A Pydantic model for grading documents based on their relevance to a question.
    The grade is a binary score.
    """

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )
