from pathlib import Path
import shelve
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader
from filelock import FileLock
import logging


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


class VectorStore:
    def __init__(self, persist_path: Path, docs: Path | None = None):
        """
        Initialize a vector store. If the given path already exists, it will be loaded.
        The contents of the `docs` directory will be indexed. Any docs that are not in the directory will be removed from the index.

        :param persist_path: Base path to store the vector store
        :param docs: Optional path to the directory containing the documents. Default to <persist_path>/docs`
        """
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
            files = self.__update_from_source(docs)
            self.initial_files = files

    def __update_from_source(self, source: Path) -> list[str]:
        self.persist_path.mkdir(parents=True, exist_ok=True)
        indexed_files_path = self.persist_path / "indexed_files"
        lock_path = self.persist_path / "lock"
        # Collect all the files
        files = {
            f.name: f
            for f in source.iterdir()
            if f.is_file() and is_file_supported(f.suffix)
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
