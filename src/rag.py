import os
import logging
from pathlib import Path
from omegaconf import DictConfig
from db import DBWorker
from llm import LLMWorker

class RAG():
    def __init__(self, path2pdf: str, config: DictConfig):
        self._ocr(path2pdf=path2pdf)
        mmd_content = self._get_content()
        self.debug = config.rag.debug
        self.db_worker = DBWorker(mmd_content=mmd_content, config=config)
        self.llm_worker= LLMWorker(config=config)

        if self.debug:
            logging.info(f'LLM set is local: {config.llm.local}')
    
    def __call__(self, query):
        if self.debug:
            logging.info('Rephrasing query...')

        rephrase_query = self.llm_worker.rephrase_query(query)
        text_context, equation_context, table_context = self.db_worker.search(rephrase_query)
        
        if self.debug:
            logging.info(f'First staged context:\n{text_context}')
            logging.info('Reranking context...')
        
        reranking_context = self.llm_worker.reranking(query=rephrase_query, context=text_context)
        full_context = f'{reranking_context}\n\n{equation_context}\n\n{table_context}'

        if self.debug:
            logging.info(f'Old query: {query} ---- Rephrased query: {rephrase_query}')
            logging.info(f'Answering by context.\n\nquery:{rephrase_query}\n\ncontext:{full_context}')
        
        return self.llm_worker.answer_by_context(query=rephrase_query, context=full_context)
        
    
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