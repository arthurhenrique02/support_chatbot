import os

from langchain.agents import create_agent
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

chat_agent = create_agent(
    model=ChatHuggingFace(
        llm=HuggingFacePipeline.from_model_id(
            name="Chat Support Agent",
            model_id=os.getenv("CHAT_BOT_MODEL"),
            device_map="cuda:0",
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 512,
                "return_full_text": False,
                "temperature": 0.2,
            },
        ),
    ),
    system_prompt="You are a helpful customer support agent, responsible to assist users with their inquiries regarding our products and services. Provide clear and concise answers, and ensure a positive user experience.",
)
