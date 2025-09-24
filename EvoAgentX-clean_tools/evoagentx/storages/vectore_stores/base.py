from enum import Enum
from abc import ABC, abstractmethod

from llama_index.core.vector_stores.types import BasePydanticVectorStore


class VectorStoreType(str, Enum):
    FAISS = "faiss"


class VectorStoreBase(ABC):
    """Base interface for vector stores."""

    @abstractmethod
    def get_vector_store(self) -> BasePydanticVectorStore:
        """Return the LlamaIndex-compatible vector store."""
        pass