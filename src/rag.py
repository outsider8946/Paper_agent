import os
import logging
from pathlib import Path
from db import DBWorker
from llm import LLMWorker

class RAG():
    def __init__(self, path2pdf: str, model_name: str, debug: bool = False):
        self._ocr(path2pdf=path2pdf)
        mmd_content = self._get_content()
        self.debug = debug
        self.db_worker = DBWorker(mmd_content=mmd_content, debug=debug)
        self.llm_worker = LLMWorker(model_name)
    
    def __call__(self, query):
        documents = self.db_worker.search(query, k=2)
        context = ''

        for doc in documents:
            context += f'{doc.page_content}\n\n'
        
        if self.debug:
            logging.info(f'Answering by context.\n\nquery:{query}\n\ncontext:{context}')
            
        return self.llm_worker.answer_by_context(query=query, context=context)
    
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