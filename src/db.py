import logging
from uuid import uuid4
from typing import Optional, Dict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils.extract_utils import extract_text, extract_tables, extract_equations

class DBWorker():
    def __init__(self, mmd_content: Optional[str] = None, debug: bool = False):
        self.embeddings = self._get_embeddings()
        self.db = self._create_db()
        self.debug = debug

        if mmd_content is not None:
            collection = self.db._client.get_collection('paper_collection')
            ids = collection.get(include=[])['ids']
            if ids:
                collection.delete(ids=ids)
            self._fill_db(mmd_content)
    
    def search(self, query: str, k: int = 5, filter: Optional[Dict] = None):
        logging.info(f'Similirarty search for query: {query}')
        return self.db.similarity_search(query=query, k=k, filter=filter)

    def _get_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-en-v1.5',
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True},
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