from fastapi import APIRouter

from core.services.bot.agents import chat_agent
from models.chat import ChatRequest

blueprint_name = "chat"

router = APIRouter(
    prefix=f"/{blueprint_name}",
    tags=[blueprint_name],
    responses={404: {"description": "Not found"}},
)


@router.post("/chat")
async def chat_endpoint(data: ChatRequest):
    response = chat_agent.invoke({"messages": data.message})
    return {"response": response}
