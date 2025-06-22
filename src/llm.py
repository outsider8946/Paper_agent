import os
from pydantic import SecretStr
from omegaconf import DictConfig
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from utils.templates import SYSTEM_RAG_TEMPLATE, SYSTEM_REPHRASE_TEMPLATE, SYSTEM_RERANKING_TEMPLATE
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.string import StrOutputParser

load_dotenv()

class LLMOllama(ChatOllama):
    def __init__(self, config: DictConfig):
        super().__init__(model=config.llm.model_name,
                        temperature=config.llm.temperature,
                        top_k=config.llm.top_k,
                        top_p=config.llm.top_p,
                        repeat_penalty=config.llm.repeat_penalty)

class LLMOpenRouter(ChatOpenAI):
        def __init__(self, config: DictConfig):
            super().__init__(base_url="https://openrouter.ai/api/v1", 
                            api_key=SecretStr(os.environ.get("OPENROUTER_API_KEY") or ""),
                            model=config.llm.model_name,
                            temperature=config.llm.temperature,
                            top_p=config.llm.top_p,
                            presence_penalty=config.llm.repeat_penalty)

class LLMWorker():
    def __init__(self, config: DictConfig):
        if config.llm.local:
            self.llm = LLMOllama(config)
        else:
            self.llm = LLMOpenRouter(config)
        
        self.history = []
    
    def _run_llm(self, system_prompt: str, **input_params):
        prompt = ChatPromptTemplate([
            ('system', system_prompt),
            MessagesPlaceholder('history', optional=True),
            ('human', '{query}')
        ])
        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke(input_params)
    
    def answer_by_context(self, query: str, context: str):
        answer = self._run_llm(SYSTEM_RAG_TEMPLATE, **{'query':query,'history': self.history ,'context':context})
        self.history.extend([('human', query), ('ai', answer)])
        return answer
    
    def rephrase_query(self, query: str):
        return self._run_llm(SYSTEM_REPHRASE_TEMPLATE, **{'query':query})
    
    def reranking(self, query: str, context: str):
        return self._run_llm(SYSTEM_RERANKING_TEMPLATE, **{'query':query,'context':context})
