from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List
from langchain_core.documents import Document

import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_pdf_file():
    loader = DirectoryLoader(
        DATA_DIR,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    document = loader.load()
    return document

extract_data = load_pdf_file()


def filter_to_minimal_docs(docs: List[Document]) ->List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata = {"source":src}
            )
        )
    return minimal_docs
minimal_docs = filter_to_minimal_docs(extract_data)

def text_split(minimal_docs):
    text_split = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50)
    text_chunks = text_split.split_documents(minimal_docs)
    return text_chunks
text_chunks = text_split(minimal_docs)