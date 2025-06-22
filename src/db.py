import logging
from uuid import uuid4
from omegaconf import DictConfig
from typing import Optional
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from utils.extract_utils import extract_text, extract_tables, extract_equations

class DBWorker():
    def __init__(self, config: DictConfig, mmd_content: Optional[str] = None):
        self.model_name = config.embeddings.model_name
        
        self.k_text = config.rag.k_text
        self.k_eqation = config.rag.k_equation
        self.k_table = config.rag.k_table
        self.debug = config.rag.debug

        self.embeddings = self._get_embeddings()
        self.db = self._create_db()

        if not Path('chroma').exists():
            collection = self.db._client.get_collection('paper_collection')
            ids = collection.get(include=[])['ids']
            if ids:
                collection.delete(ids=ids)
            self._fill_db(mmd_content)
    
    def search(self, query: str):
        logging.info(f'Similirarty search for query: {query}')
        documents_texts = self.db.similarity_search(query=query, k=self.k_text, filter={'source':'text'})
        documents_equations = self.db.similarity_search(query=query, k=self.k_eqation, filter={'source':'equation'})
        documents_tables = self.db.similarity_search(query=query, k=self.k_table, filter={'source':'table'})
        
        text_context = ''
        equation_context = ''
        table_context = ''

        for i, doc in enumerate(documents_texts):
            text_context += f'{i+1}. {doc.page_content}\n'
        
        for doc in documents_equations:
            equation_context += f'{doc.page_content}\n'
        
        for doc in documents_tables:
            table_context += f'{doc.page_content}\n'

        return text_context, equation_context, table_context

    def _get_embeddings(self):
        return OllamaEmbeddings(
            model=self.model_name,
            num_gpu=1,
            )
    
    def _create_db(self):
        return Chroma(
            collection_name="paper_collection",
            embedding_function=self.embeddings,
            persist_directory="chroma",
        )
            
    def _fill_db(self, mmd_content: str):
        tables = extract_tables(mmd_content)
        equations = extract_equations(mmd_content)
        texts = extract_text(mmd_content)
        documents = tables + equations + texts
        logging.info(f'Documents to add: {len(documents)}')
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.db.add_documents(documents=documents, ids=uuids)