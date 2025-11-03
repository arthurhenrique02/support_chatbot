from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="The user's message to the chatbot.",
        example="What`s time in brazil?",
    )
