from typing import Union, Literal

import faiss
from llama_index.vector_stores.faiss import FaissMapVectorStore

from .base import VectorStoreBase


class FaissVectorStoreWrapper(VectorStoreBase):
    """Wrapper for FAISS vector store."""
    
    def __init__(self, dimensions: int = 1536, 
                 metrics: Union[Literal["flat_l2", "ivf_flat"]] = "flat_l2", **kwargs):
        self.dimensions = dimensions
        self.metrics = metrics
        self.faiss_index = self._create_index()
        self.vector_store = FaissMapVectorStore(faiss_index=faiss.IndexIDMap2(self.faiss_index))

    def _create_index(self) -> faiss.Index:
        if self.metrics == "flat_l2":
            return faiss.IndexFlatL2(self.dimensions)
        elif self.metrics == "ivf_flat":
            quantizer = faiss.IndexFlatL2(self.dimensions)
            return faiss.IndexIVFFlat(quantizer, self.dimensions, 100)
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.metrics}")
        
    def get_vector_store(self) -> FaissMapVectorStore:
        """Return the FAISS vector store."""
        return self.vector_store
    
    