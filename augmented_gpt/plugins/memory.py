from ..decorators import *
from typing import Any
from ..message import Message
import json
import pandas as pd
from . import Plugin
import os
import numpy as np
import datetime
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore


class MemoryPlugin(Plugin):
    def __init__(self, data_file: str = "_memory-with-embedding.csv"):
        self.data_file = data_file

    def __load_data(self) -> pd.DataFrame:
        return (
            pd.read_csv(self.data_file)  # type: ignore
            if os.path.isfile(self.data_file) and os.path.getsize(self.data_file) > 0
            else pd.DataFrame(columns=["timestamp", "text", "embedding"])
        )

    def __save_new_data(self, text: str):
        df: Any = self.__load_data()
        embedding = self.__get_embedding(text, model="text-embedding-ada-002")
        df.loc[len(df.index)] = [datetime.datetime.now().isoformat(), text, embedding]
        df.to_csv(self.data_file, index=False)

    def __get_embedding(self, text: str, model: str = "text-embedding-ada-002"):
        # text = text.replace("\n", " ")
        return (
            self.gpt.client.embeddings.create(input=[text], model=model)
            .data[0]
            .embedding
        )

    def clear_memory(self):
        f = open(self.data_file, "w")
        f.close()

    @function
    def search_from_memory(
        self,
        query: str = param("The query of the search"),
    ):
        """Search content from the long term memory, by using the given query. The memory includes all the prior chat history and prior knowledge.
        The search results are in descending order in importance. Always respect to the top results.
        If you're unsure about some past events or some knowledge, always check the memory first as you may forgot it.
        """
        embedding = self.__get_embedding(query)
        df: Any = self.__load_data()
        df["embedding"] = df.embedding.apply(eval).apply(np.array)

        def compute_cosine_similarity(x: Any) -> float:
            return cosine_similarity([x], [embedding])[0][0]  # type: ignore

        df["similarity"] = df.embedding.apply(compute_cosine_similarity)
        data = df.sort_values("similarity", ascending=False)
        results = ""
        n = 0
        for _index, row in data.iterrows():
            results += "TIMESTAMP: " + row["timestamp"] + "\n"
            results += "CONTENT:\n" + row["text"] + "\n\n---\n\n"
            if n >= 2:
                break
            n += 1
        return results

    @function
    def remember(
        self,
        content: str = param("The content to remember in the memory"),
    ):
        """Actively remember some information as part of the long-term memory, if it is important."""
        self.__save_new_data(content)
        return "done."

    def on_new_chat_message(self, msg: Message):
        msg_s = json.dumps(msg.to_json())
        self.__save_new_data(msg_s)
