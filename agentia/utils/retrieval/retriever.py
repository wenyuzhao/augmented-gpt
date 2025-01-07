from llama_index.core import QueryBundle
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from .vector_store import VectorStore

TOP_K = 16


class MultiRetriever(BaseRetriever):
    def __init__(self, vector_stores: list[VectorStore], file: str | None) -> None:
        super().__init__()
        self.vector_stores = vector_stores
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
        for store in self.vector_stores:
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
