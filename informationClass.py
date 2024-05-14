from typing import Dict, Any
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

from dotenv import load_dotenv


class InformationProcessor:
    def __init__(self, information: str, model_name: str, temperature: int) -> None:
        self.information: str = information
        self.model_name: str = model_name
        self.temperature: int = temperature
        self.chain: LLMChain | None = None
        load_dotenv()

    def create_prompt_template(self) -> PromptTemplate:
        summary_template: str = """
      given the information {information} about a person I want you to create:
      1. A short summary
      2. two interesting facts about them
      """
        return PromptTemplate(input_variables=["information"], template=summary_template)

    def create_llm_chain(self, prompt_template: PromptTemplate) -> None:
        llm: ChatOpenAI = ChatOpenAI(
            temperature=self.temperature, model_name=self.model_name)
        self.chain = LLMChain(llm=llm, prompt=prompt_template)

    def process_information(self) -> Any:
        if self.chain is None:
            raise ValueError("LLMChain has not been created.")
        res: Any = self.chain.invoke(
            input={"information": self.information})
        return res
