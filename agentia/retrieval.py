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
import logging

TOP_K = 16


class VectorStore:
    def __init__(self, persist_path: Path, docs: Path | None = None):
        self.persist_path = persist_path
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_path / "vector_store"))
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.client.get_or_create_collection("vector_store")
        )
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        self.initial_files: list[str] | None = None
        docs = docs or self.persist_path / "docs"
        docs.mkdir(parents=True, exist_ok=True)
        if docs:
            files = self.__load_from_source(docs)
            self.initial_files = files

    def __load_from_source(self, source: Path) -> list[str]:
        self.persist_path.mkdir(parents=True, exist_ok=True)
        indexed_files_path = self.persist_path / "indexed_files"
        lock_path = self.persist_path / "lock"
        # Collect all the files
        files = {
            f.name: f
            for f in source.iterdir()
            if f.is_file() and KnowledgeBase.is_file_supported(f.suffix)
        }
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
            # Remove files
            if len(del_files) > 0:
                nodes = self.index.vector_store.get_nodes(None)
                del_files_set = set(del_files)
                nodes_to_remove = [
                    n.node_id
                    for n in nodes
                    if n.metadata.get("file_name") in del_files_set
                ]
                self.index.delete_nodes(nodes_to_remove)
            # Index new files
            if len(files) > 0:
                logging.info(f"Indexing {len(files)} files")
                docs = SimpleDirectoryReader(
                    input_files=[str(f) for f in files.values()],
                    exclude_hidden=False,
                ).load_data()
                for d in docs:
                    self.index.insert(d)
            # Update the global files
            with shelve.open(indexed_files_path) as g:
                for f in del_files:
                    del g[f]
                for f in files:
                    g[f] = files[f].stat().st_mtime
            self.index = self.index
            return all_files


class MultiRetriever(BaseRetriever):
    def __init__(self, knowledge_base: "KnowledgeBase", file: str | None) -> None:
        super().__init__()
        self.knowledge_base = knowledge_base
        self.file = file

    def _get_retrievers(self) -> list[BaseRetriever]:
        if self.file:
            filters = MetadataFilters(
                filters=[
                    ExactMatchFilter(key="file_name", value=self.file),
                ]
            )
        else:
            filters = None
        retrievers = []
        for store in self.knowledge_base.vector_stores.values():
            retriever = VectorIndexRetriever(
                index=store.index, similarity_top_k=TOP_K, filters=filters
            )
            retrievers.append(retriever)
        return retrievers

    def _sort_nodes(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        nodes.sort(key=lambda x: x.score or 0, reverse=True)
        return nodes

    async def _aretrieve(self, query_bundle: QueryBundle):
        retrievers = self._get_retrievers()
        nodes = []
        for retriever in retrievers:
            nodes.extend(await retriever.aretrieve(query_bundle))
        return self._sort_nodes(nodes)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        retrievers = self._get_retrievers()
        nodes = []
        for retriever in retrievers:
            nodes.extend(retriever.retrieve(query_bundle))
        return self._sort_nodes(nodes)


class KnowledgeBase:
    def __init__(self, id: str | Path, docs: Path | None = None):
        """Initialize an empty knowledge base. If the given ID or save path already exists, it will be loaded."""

        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY must be set to enable the knowledge base")

        if isinstance(id, str):
            from agentia.agent import _get_global_cache_dir

            persist_dir = _get_global_cache_dir() / "knowledge-base" / id
        else:
            persist_dir = id
        persist_dir.mkdir(parents=True, exist_ok=True)
        if docs:
            docs.mkdir(parents=True, exist_ok=True)
        self.vector_stores = {
            "default": VectorStore(persist_path=persist_dir, docs=docs)
        }
        self.__persist_dir = persist_dir
        self.__retriever = MultiRetriever(knowledge_base=self, file=None)
        self.__query_engine = CitationQueryEngine.from_args(
            self.vector_stores["default"].index,
            similarity_top_k=TOP_K,
            citation_chunk_size=1024,
            retriever=self.__retriever,
        )

    def add_vector_store(self, name: str, store: VectorStore):
        if name in self.vector_stores:
            raise ValueError(f"Vector store with name {name} already exists")
        self.vector_stores[name] = store

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
            assert isinstance(node, NodeWithScore)
            file_name = node.node.metadata.get("file_name")
            index_str = f"[{i}] " if file_name is None else f"[{i}] {file_name}"
            formatted_response += f"{index_str}\n\n{node.get_text()}\n"
        # print(formatted_response)
        return formatted_response

    def add_document(self, doc: BytesIO):
        """Add documents to the knowledge base. Documents are a dictionary of document ID to base64 url"""
        if doc.name is None or doc.name == "":
            raise ValueError("Document name must be provided")
        assert isinstance(doc.name, str)
        # write the file to disk
        temp_file = self.__persist_dir / "temp" / doc.name
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.__persist_dir / "temp" / doc.name, "wb") as f:
            f.write(doc.read())
            f.flush()
        # index the file
        docs = SimpleDirectoryReader(
            input_files=[str(temp_file)], exclude_hidden=True
        ).load_data()
        store = self.vector_stores["default"]
        for d in docs:
            store.index.insert(d)
        # remove the temp file
        temp_file.unlink()

    def add_documents(self, docs: list[BytesIO]):
        for doc in docs:
            self.add_document(doc)
