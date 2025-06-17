from omegaconf import DictConfig
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from utils.templates import SYSTEM_RAG_TEMPLATE
from langchain_core.output_parsers.string import StrOutputParser

class LLMWorker():
    def __init__(self, config: DictConfig):
        self.llm = ChatOllama(model=config.llm.model_name,
                              temperature=config.llm.temperature,
                              top_k=config.llm.top_k,
                              top_p=config.llm.top_p,
                              repeat_penalty=config.llm.repeat_penalty)
        
    def answer_by_context(self, query: str, context: str):
        prompt = ChatPromptTemplate([
            ('system', SYSTEM_RAG_TEMPLATE),
            ('human', '{query}')
        ])
        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({'query':query, 'context':context})