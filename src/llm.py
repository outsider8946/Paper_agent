from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from utils.templates import SYSTEM_BASIC_TEMPLATE, ANSWERER_TEMPLATE
from langchain_core.output_parsers.string import StrOutputParser

class LLMWorker():
    def __init__(self, model_name):
        self.llm = ChatOllama(model=model_name)
        
    def answer_by_context(self, query: str, context: str):
        prompt = ChatPromptTemplate([
            ('system', SYSTEM_BASIC_TEMPLATE),
            ('human', ANSWERER_TEMPLATE)
        ])
        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({'query':query, 'context': context})