from fastapi import APIRouter

from core.services.bot.hugging_face import ChatBot
from models.chat import ChatRequest

blueprint_name = "chat"

router = APIRouter(
    prefix=f"/{blueprint_name}",
    tags=[blueprint_name],
    responses={404: {"description": "Not found"}},
)


bot = ChatBot(
    name="SupportChatBot",
    definition_prompt="You are a helpful customer support assistant.",
    model="meta-llama/Llama-3.2-1B",
)


@router.post("/chat")
async def chat_endpoint(data: ChatRequest):
    response = bot.task(data.message)
    return {"response": response}
