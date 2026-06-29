"""
base_cold_storage.py
────────────────────
Abstract interface for cold storage backends.

This is the key modularity boundary in Vector-ARC. By programming against
this interface, the rest of the pipeline (rag_coordinator) never needs to
change when you swap cold storage strategies.

Planned implementations:
  ✅ BM25ColdStorage  — keyword search, zero vector overhead (current)
  🔲 SQLColdStorage   — full-text SQL search, easy to scale
  🔲 VectorColdStorage — semantic search, higher cost but best recall

To add a new backend:
  1. Create a class that inherits from BaseColdStorage
  2. Implement search() and get_corpus_size()
  3. Pass your class to AdaptiveRAGSystem(cold_storage=YourBackend())
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseColdStorage(ABC):
    """
    Abstract base class for cold storage backends.

    Cold storage is the fallback retrieval layer used on a cache miss.
    It is intentionally decoupled from the Vector-ARC cache logic so
    we can benchmark BM25, SQL, and vector-DB alternatives independently.
    """

    @abstractmethod
    def search(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Find the most relevant document for the given natural-language query.

        Args:
            query: The raw user query string.

        Returns:
            (doc_id, document_text) if a relevant document is found.
            None if the corpus is empty or no match is found.
        """
        pass

    @abstractmethod
    def get_corpus_size(self) -> int:
        """Returns the total number of documents indexed in this backend."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(corpus_size={self.get_corpus_size()})"
