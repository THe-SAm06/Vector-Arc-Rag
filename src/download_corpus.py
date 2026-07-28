import os
import duckdb
from datasets import load_dataset

def main():
    print("Downloading TREC-COVID corpus from HuggingFace (super fast)...")
    dataset = load_dataset("BeIR/trec-covid", "corpus", split="corpus")
    
    db_path = "data/cold_storage.duckdb"
    print(f"Ingesting into DuckDB at {db_path}...")
    
    os.makedirs("data", exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
        
    con = duckdb.connect(db_path)
    
    con.execute("""
        CREATE TABLE documents (
            _id VARCHAR PRIMARY KEY,
            title VARCHAR,
            text VARCHAR
        )
    """)
    
    print("Streaming documents into DuckDB...")
    # Convert HF dataset to Pandas DataFrame and ingest in chunks
    df = dataset.to_pandas()
    
    con.execute("""
        INSERT INTO documents 
        SELECT _id, title, text FROM df
    """)
    
    print("Building Full-Text Search (FTS) index...")
    con.execute("PRAGMA create_fts_index('documents', '_id', 'title', 'text')")
    
    count = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"Successfully ingested {count} documents into DuckDB cold storage!")

if __name__ == "__main__":
    main()
