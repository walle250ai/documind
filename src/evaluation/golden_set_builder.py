import json
import random
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class GoldenQA(BaseModel):
    question: str
    ground_truth: str
    reference_context: str
    source: str


class GoldenSetBuilder:
    def __init__(self):
        pass

    def generate_from_chunks(
        self,
        chunks: List[Document],
        n_questions: int = 30,
        llm_model: str = "gpt-4o-mini"
    ) -> List[GoldenQA]:
        sampled_chunks = random.sample(chunks, min(n_questions, len(chunks)))
        llm = ChatOpenAI(model=llm_model)
        golden_qas = []

        for chunk in sampled_chunks:
            prompt = f"""Based on the following text, generate one factual question and its answer.
The question should be specific and answerable from the text only.
Text: {chunk.page_content}
Respond in JSON: {{"question": "...", "answer": "...", "context": "..."}}"""

            response = llm.invoke(prompt)
            try:
                parsed = json.loads(response.content)
                golden_qas.append(GoldenQA(
                    question=parsed["question"],
                    ground_truth=parsed["answer"],
                    reference_context=chunk.page_content,
                    source=chunk.metadata.get("source", "")
                ))
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                continue

        return golden_qas

    def save(self, golden_set: List[GoldenQA], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([qa.model_dump() for qa in golden_set], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> List[GoldenQA]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [GoldenQA(**item) for item in data]
