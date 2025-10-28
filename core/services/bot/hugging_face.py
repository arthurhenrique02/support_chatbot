from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from core.services.bot.generics import GenericBot


class ChatBot(GenericBot):
    def __init__(self, name: str, definition_prompt: str, model: str):
        super().__init__(name, definition_prompt, model)
        self.setup_llm()

    def setup_llm(
        self,
        *args,
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

        self.llm = ChatHuggingFace(llm=model)

    def task(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response
