import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

def extract_tables(mmd_content: str) -> List[Document]:
    table_blocks = re.findall(r"\\begin\{table\}.*?\\end\{table\}", mmd_content, flags=re.DOTALL)
    tables_docs = [Document(page_content=item, metadata={'source':'table'}, id=i) for i, item in enumerate(set(table_blocks))]
    return tables_docs

def extract_equations(mmd_content: str) -> List[Document]:
    equations = re.findall(r'\\(?:\([^)]*\)|\[[^]]*\])', mmd_content)
    equations_docs = [Document(page_content=item, metadata={'source':'equation'}, id=i) for i, item in enumerate(set(equations))]
    return equations_docs

def extract_text(mmd_content: str) -> List[Document]:
    unique_headers = set(re.findall(r'#+', mmd_content))
    headers_to_split_on = [(header, f'Header {i}') for i, header in enumerate(unique_headers)]
    documents = MarkdownHeaderTextSplitter(headers_to_split_on).split_text(mmd_content)
    updated_documents = []

    for i, doc in enumerate(documents):
        headers = ', '.join(list(doc.metadata.values()))
        metadata = {'source': 'text', 'headers': headers}
        updated_documents.append(Document(page_content=doc.page_content, metadata=metadata, id=i))

    return updated_documents

def clean_data(mmd_content: str):
    pass