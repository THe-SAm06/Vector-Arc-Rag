import logging
import duckdb
from typing import Optional, Tuple
from src.base_cold_storage import BaseColdStorage

logger = logging.getLogger(__name__)

class DuckDBColdStorage(BaseColdStorage):
    """
    DuckDB-backed cold storage for out-of-core sparse text retrieval.
    Performs BM25 search via DuckDB's FTS extension.
    """
    def __init__(self, db_path: str = "data/cold_storage.duckdb", json_path: str = "data/scifact_corpus_full.json"):
        import os
        self.db_path = db_path
        
        # If DB doesn't exist, build it from the JSON
        if not os.path.exists(db_path):
            logger.info(f"Database {db_path} not found. Building it from {json_path}...")
            self._build_db(db_path, json_path)
            
        self.con = duckdb.connect(db_path, read_only=True)
        
        # Verify FTS index exists
        try:
            count = self.con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            logger.info(f"DuckDBColdStorage connected to {db_path} with {count} documents.")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB cold storage: {e}")
            raise

    def _build_db(self, db_path: str, json_path: str):
        import json
        con = duckdb.connect(db_path)
        con.execute("""
            CREATE TABLE documents (
                _id VARCHAR PRIMARY KEY,
                title VARCHAR,
                text VARCHAR
            )
        """)
        
        logger.info(f"Reading {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
            
        logger.info("Inserting documents into DuckDB...")
        # For simplicity, we just use the dictionary values. The dataset was pre-merged as Title - Text.
        # But we need an _id. We can use the dictionary keys.
        for doc_id, text_content in corpus.items():
            # In our corpus, the text is already merged. We just store it in 'text' and leave title empty.
            safe_text = text_content.replace("'", "''")
            con.execute(f"INSERT INTO documents VALUES ('{doc_id}', '', '{safe_text}')")
            
        logger.info("Building Full-Text Search (FTS) index...")
        con.execute("PRAGMA create_fts_index('documents', '_id', 'title', 'text')")
        con.close()
        logger.info("Database build complete.")

    def get_corpus_size(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def search(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Perform a BM25 keyword search using DuckDB FTS.
        Returns (doc_id, text) of the best matching document.
        """
        # Escape single quotes in query for SQL
        safe_query = query.replace("'", "''")
        
        # DuckDB FTS search query
        sql = f"""
            SELECT _id, text, fts_main_documents.match_bm25(_id, '{safe_query}') as score
            FROM documents
            WHERE score > 0
            ORDER BY score DESC
            LIMIT 5
        """
        
        try:
            results = self.con.execute(sql).fetchall()
            if results:
                combined_text = "\n\n---\n\n".join([row[1] for row in results])
                top_id = results[0][0]
                top_score = results[0][2]
                logger.debug(f"DuckDBColdStorage hit | top _id={top_id} score={top_score:.4f}, retrieved {len(results)} docs")
                return str(top_id), combined_text
            return None
        except Exception as e:
            logger.error(f"DuckDB search failed: {e}")
            return None
