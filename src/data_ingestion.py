import json
import os
import logging

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Please run: pip install datasets")

logging.basicConfig(level=logging.INFO, format='%(message)s')

class BenchmarkDataIngestor:
    """
    Pulls the SciFact benchmark dataset to provide undeniable proof 
    that Vector-ARC outperforms the baseline on standard academic data.
    """
    def __init__(self, target_path: str = "data/scifact_corpus.json", limit: int = 500):
        self.target_path = target_path
        self.limit = limit

    def execute(self):
        if os.path.exists(self.target_path):
            logging.info(f"✅ Data already exists at {self.target_path}")
            return

        logging.info(f"📥 Downloading {self.limit} documents from the SciFact RAG benchmark...")
        
        # Load the corpus split of the SciFact dataset
        dataset = load_dataset("mteb/scifact", "corpus", split="corpus")
        
        corpus_dict = {}
        for i, row in enumerate(dataset):
            if i >= self.limit:
                break
            # Merge title and text for a dense, high-quality RAG context
            corpus_dict[str(row['_id'])] = f"{row['title']} - {row['text']}"
            
        os.makedirs(os.path.dirname(self.target_path), exist_ok=True)
        with open(self.target_path, "w", encoding="utf-8") as f:
            json.dump(corpus_dict, f, indent=4)
            
        logging.info(f"🚀 Successfully generated {self.target_path} for benchmarking.")

if __name__ == "__main__":
    ingestor = BenchmarkDataIngestor()
    ingestor.execute()