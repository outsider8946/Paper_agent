import os
import logging
from pathlib import Path
from omegaconf import DictConfig
from langchain_core.prompts import ChatPromptTemplate
from utils.templates import SYSTEM_RAG_TEMPLATE
from langchain_core.output_parsers.string import StrOutputParser
from db import DBWorker
from llm import LLMOpenRouter

class RAG():
    def __init__(self, path2pdf: str, config: DictConfig):
        self._ocr(path2pdf=path2pdf)
        mmd_content = self._get_content()
        self.debug = config.rag.debug
        self.db_worker = DBWorker(mmd_content=mmd_content, config=config)
        self.llm = LLMOpenRouter(model_name=config.llm.model_name)
    
    def __call__(self, query):
        documents = self.db_worker.search(query)
        context = ''

        for doc in documents:
            context += f'{doc.page_content}\n\n'
        
        if self.debug:
            logging.info(f'Answering by context.\n\nquery:{query}\n\ncontext:{context}')
        
        prompt = ChatPromptTemplate([
            ('system', SYSTEM_RAG_TEMPLATE),
            ('human', '{query}')
        ])
        chain = prompt | self.llm | StrOutputParser()
            
        return chain.invoke({'query':query, 'context':context})
    
    def _get_content(self):
        content = None

        with open(self.path2mmd, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content

    def _ocr(self, path2pdf: str, path2output: str = 'output'):
        name = Path(path2pdf).name.replace('.pdf', '')
        self.path2mmd = f'{path2output}/{name}.mmd'

        if os.path.exists(self.path2mmd):
            return 
        
        logging.info('OCR is started')
        os.system(f'nougat {path2pdf} -o {path2output} -m 0.1.0-base --no-skipping')
        logging.info('OCR is done')