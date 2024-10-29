from io import BytesIO
from pathlib import Path
import shelve
import chromadb
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core import VectorStoreIndex
import os
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core import QueryBundle
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from filelock import FileLock


class FilteredRetriever(BaseRetriever):
    def __init__(self, vector_index: VectorStoreIndex, file: str | None) -> None:
        super().__init__()
        self.vector_index = vector_index
        self.file = file
        self.global_index: VectorStoreIndex | None = None

    def _get_retriever(self) -> tuple[BaseRetriever, BaseRetriever | None]:
        if self.file:
            filters = MetadataFilters(
                filters=[
                    ExactMatchFilter(key="file_name", value=self.file),
                ]
            )
        else:
            filters = None
        retriever = VectorIndexRetriever(
            index=self.vector_index, similarity_top_k=16, filters=filters
        )
        global_retriever = None
        if self.global_index is not None:
            global_retriever = VectorIndexRetriever(
                index=self.global_index, similarity_top_k=16, filters=None
            )
        return retriever, global_retriever

    async def _aretrieve(self, query_bundle: QueryBundle):
        retriever, global_retriever = self._get_retriever()
        local_nodes = await retriever.aretrieve(query_bundle)
        if global_retriever:
            global_nodes = await global_retriever.aretrieve(query_bundle)
        else:
            global_nodes = []
        # print([n.node.metadata.get("file_name") for n in vector_nodes])
        return sorted(
            [*local_nodes, *global_nodes], key=lambda x: x.score, reverse=True
        )

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        retriever, global_retriever = self._get_retriever()
        vector_nodes = retriever.retrieve(query_bundle)
        if global_retriever:
            global_nodes = global_retriever.retrieve(query_bundle)
        else:
            global_nodes = []
        return sorted(
            [*vector_nodes, *global_nodes], key=lambda x: x.score, reverse=True
        )


class KnowledgeBase:
    def __init__(self, id: str | Path):
        """Initialize an empty knowledge base. If the given ID or save path already exists, it will be loaded."""

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
        self.__retriever = FilteredRetriever(vector_index=self.__index, file=None)
        self.__query_engine = CitationQueryEngine.from_args(
            self.__index,
            similarity_top_k=16,
            citation_chunk_size=1024,
            retriever=self.__retriever,
        )

    def set_global_index(self, index: VectorStoreIndex | None):
        self.__retriever.global_index = index

    def load_global_docs_and_persist(
        self, docs_path: Path, index_path: Path
    ) -> list[str]:
        """Create a vector store from the global documents. This also caches indices."""
        indexed_files_path = index_path / "indexed_files"
        vector_store_path = index_path / "vector_store"
        lock_path = index_path / "lock"
        # Collect all the files
        files = {
            f.name: f
            for f in docs_path.iterdir()
            if f.is_file() and KnowledgeBase.is_file_supported(f.suffix)
        }
        (index_path).mkdir(parents=True, exist_ok=True)
        all_files = list(files.keys())
        del_files: set[str] = set()
        with FileLock(lock_path):
            with shelve.open(indexed_files_path) as g:
                for f in g:
                    if f not in files:
                        del_files.add(f)
                    elif g[f] >= files[f].stat().st_mtime:  # not modified
                        del files[f]
                    else:  # file is modified
                        del_files.add(f)
            # Load vector store
            chroma_client = chromadb.PersistentClient(path=str(vector_store_path))
            vector_store = ChromaVectorStore(
                chroma_collection=chroma_client.get_or_create_collection("vector_store")
            )
            index = VectorStoreIndex.from_vector_store(vector_store)
            # Remove files
            if len(del_files) > 0:
                nodes = index.vector_store.get_nodes(None)
                del_files_set = set(del_files)
                nodes_to_remove = [
                    n.node_id
                    for n in nodes
                    if n.metadata.get("file_name") in del_files_set
                ]
                index.delete_nodes(nodes_to_remove)
            # Index new files
            if len(files) > 0:
                print(f"Indexing {len(files)} files")
                docs = SimpleDirectoryReader(
                    input_files=[str(f) for f in files.values()],
                    exclude_hidden=False,
                ).load_data()
                for d in docs:
                    index.insert(d)
            # Update the global files
            with shelve.open(indexed_files_path) as g:
                for f in del_files:
                    del g[f]
                for f in files:
                    g[f] = files[f].stat().st_mtime
            # Set global index
            if len(all_files) > 0:
                self.set_global_index(index)
            else:
                self.set_global_index(None)
            return all_files

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

    async def query(self, query: str, file: str | None) -> str:
        """Query the knowledge base"""
        self.__retriever.file = file
        response = await self.__query_engine.aquery(query)
        formatted_response = response.response + "\n\n\nSOURCES:\n\n"
        for i, node in enumerate(response.source_nodes):
            file_name = node.node.metadata.get("file_name")
            index_str = f"[{i}] " if file_name is None else f"[{i}] {file_name}"
            formatted_response += f"{index_str}\n\n{node.get_text()}\n"
        # print(formatted_response)
        return formatted_response

    def load_documents_in_folder(self, folder: Path) -> list[str]:
        """Load documents from a folder"""
        if not folder.exists():
            raise ValueError("Folder does not exist")
        folder_is_empty = len(list(folder.glob("*"))) == 0
        if folder_is_empty:
            return []
        docs = SimpleDirectoryReader(
            input_dir=str(folder), exclude_hidden=False
        ).load_data()
        names = set()
        for d in docs:
            if file_name := d.metadata.get("file_name"):
                names.add(file_name)
            self.__index.insert(d)
        return list(names)

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
