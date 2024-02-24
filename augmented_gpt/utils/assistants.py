from dataclasses import dataclass
import openai
from datetime import datetime, UTC
import os


@dataclass
class AssistantInfo:
    id: str
    name: str | None
    description: str | None
    created_at: datetime


class AssistantManager:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)

    def list(self) -> list[AssistantInfo]:
        results: list[AssistantInfo] = []
        assistants = self.client.beta.assistants.list(limit=100)
        while len(assistants.data) > 0:
            for a in assistants.data:
                results.append(
                    AssistantInfo(
                        id=a.id,
                        name=a.name,
                        description=a.description,
                        created_at=datetime.fromtimestamp(a.created_at, UTC),
                    )
                )
            assistants = self.client.beta.assistants.list(
                limit=100, after=results[-1].id
            )
        return results

    def delete(self, id: str):
        self.client.beta.assistants.delete(assistant_id=id)

    def delete_all(self):
        assistants = self.list()
        for a in assistants:
            self.delete(a.id)
        print(f"Deleted {len(assistants)} assistants.")
        # Delete assistant files
        files_to_delete: list[str] = []
        files = self.client.files.list()
        for f in files.data:
            if "assistant" in f.purpose:
                files_to_delete.append(f.id)
        for f in files_to_delete:
            self.client.files.delete(file_id=f)
        print(f"Deleted {len(files_to_delete)} assistant files.")
