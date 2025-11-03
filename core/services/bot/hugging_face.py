import os

from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer

from core.services.bot.generics import GenericBot


class ChatBot(GenericBot):
    def __init__(self, name: str, definition_prompt: str, model: str, setup_kwargs={}):
        super().__init__(name, definition_prompt, model)
        self.setup_llm(**setup_kwargs)

    def setup_llm(
        self,
        **kwargs,
    ) -> None:
        temperature = None
        max_new_tokens = None
        task = None

        if not (temperature := kwargs.get("temperature")):
            temperature = 1.0
        if not (max_new_tokens := kwargs.get("max_new_tokens")):
            max_new_tokens = 1000
        if not (task := kwargs.get("task")):
            task = "text-generation"

        model = HuggingFacePipeline.from_model_id(
            name=self.name,
            model_id=self.model,
            device_map="cuda:0",
            task=task,
            pipeline_kwargs={
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
        )

        self.llm = ChatHuggingFace(
            llm=model,
            tokenizer=AutoTokenizer.from_pretrained(os.getenv("TOKENIZER_MODEL")),
        )

    @tool
    def task(self, prompt: str) -> str:
        """
        From a given prompt, return the answer as a text prompt. Cleaned up with just the repsonde.
        """
        response = self.llm.invoke(prompt)
        return response | StrOutputParser()
