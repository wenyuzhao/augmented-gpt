from io import BytesIO
from pathlib import Path
import chromadb
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.base import BaseGPTIndex
import os
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.chroma import ChromaReader
from llama_index.core import SimpleDirectoryReader, Document


class KnowledgeBase:
    def __init__(self, id: str | Path):
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY must be set to enable the knowledge base")

        # Create or load chroma client
        if isinstance(id, str):
            db_dir = Path.cwd() / ".cache" / "knowledge-base" / id
            self.__db_dir = db_dir
        else:
            db_dir = id
            self.__db_dir = db_dir
        chroma_client = chromadb.PersistentClient(path=str(db_dir / "vector_store"))
        vector_store = ChromaVectorStore(
            chroma_collection=chroma_client.get_or_create_collection("vector_store")
        )
        self.__index = VectorStoreIndex.from_vector_store(vector_store)
        self.__query_engine = CitationQueryEngine.from_args(
            self.__index, similarity_top_k=16, citation_chunk_size=1024
        )

    @staticmethod
    def is_file_supported(file_ext: str) -> bool:
        SUPPORTED_EXTS = [
            "csv",
            "docx",
            "epub",
            "hwp",
            "ipynb",
            "jpeg",
            "jpg",
            "mbox",
            "md",
            "mp3",
            "mp4",
            "pdf",
            "png",
            "ppt",
            "pptm",
            "pptx",
            # common text files
            "txt",
            "log",
            "tex",
        ]
        return file_ext.lower().strip(".") in SUPPORTED_EXTS

    async def query(self, query: str) -> str:
        """Query the knowledge base"""
        response = await self.__query_engine.aquery(query)
        formatted_response = response.response + "\n\n\nSOURCES:\n\n"
        for i, node in enumerate(response.source_nodes):
            file_name = node.node.metadata.get("file_name")
            index_str = f"[{i}] " if file_name is None else f"[{i}] {file_name}"
            formatted_response += f"{index_str}\n\n{node.get_text()}\n"
        return formatted_response

    def reset(self):
        # self.__index = VectorStoreIndex.from_vector_store(self.__index.vector_store)
        pass

    def load_documents_in_folder(self, folder: Path):
        """Load documents from a folder"""
        if not folder.exists():
            raise ValueError("Folder does not exist")
        folder_is_empty = len(list(folder.glob("*"))) == 0
        if folder_is_empty:
            return
        docs = SimpleDirectoryReader(input_dir=str(folder), recursive=True).load_data()
        for d in docs:
            self.__index.insert(d)

    def add_document(self, doc: BytesIO):
        """Add documents to the knowledge base. Documents are a dictionary of document ID to base64 url"""

        (self.__db_dir / "temp").mkdir(parents=True, exist_ok=True)

        if doc.name is None or doc.name == "":
            raise ValueError("Document name must be provided")

        with open(self.__db_dir / "temp" / doc.name, "wb") as f:
            f.write(doc.read())
            f.flush()
        docs = SimpleDirectoryReader(
            input_files=[str(self.__db_dir / "temp" / doc.name)]
        ).load_data()
        for d in docs:
            self.__index.insert(d)

    def add_documents(self, docs: list[BytesIO]):
        for doc in docs:
            self.add_document(doc)
