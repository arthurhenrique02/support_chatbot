import os

from langchain.agents import create_agent

from core.services.bot.hugging_face import ChatBot

bot = ChatBot(
    name="SupportChatBot",
    definition_prompt="You are a helpful customer support assistant.",
    model=os.getenv("CHAT_BOT_MODEL"),
    setup_kwargs={
        "temperature": 0,
        "max_new_tokens": 500,
        "task": "text-generation",
    },
)

chat_agent = create_agent(
    model=bot.llm, system_prompt=bot.definition_prompt, tools=[bot.task]
)
