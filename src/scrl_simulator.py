import numpy as np
import faiss
import time
import os
import argparse
from tqdm import tqdm
from src.vector_arc_cache import VectorARC

def load_data(stream_path):
    print(f"Loading data from {stream_path}...")
    query_embs = np.load(os.path.join(stream_path, "query_embeddings.npy"))
    doc_embs = np.load(os.path.join(stream_path, "doc_embeddings.npy"))
    doc_ids = np.load(os.path.join(stream_path, "doc_ids.npy"))
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embs)
    faiss.normalize_L2(doc_embs)
    
    return query_embs, doc_embs, doc_ids

class SCRLSimulator:
    def __init__(self, doc_embs, doc_ids, capacity=1000, threshold=0.85):
        self.doc_embs = doc_embs
        self.doc_ids = doc_ids
        self.threshold = threshold
        
        print("Building Cold FAISS index...")
        self.cold_index = faiss.IndexFlatIP(doc_embs.shape[1])
        self.cold_index.add(doc_embs)
        
        self.hot_index = faiss.IndexIDMap(faiss.IndexFlatIP(doc_embs.shape[1]))
        self.cache = VectorARC(capacity=capacity)
        
        self.stats = {"hits": 0, "misses": 0, "cold_lookups": 0, "evictions": 0}

    def process_query(self, query_emb):
        hit = False
        if self.hot_index.ntotal > 0:
            q = query_emb.reshape(1, -1)
            hot_sims, hot_idx = self.hot_index.search(q, 1)
            if hot_sims[0][0] >= self.threshold:
                hit = True
                self.stats["hits"] += 1
                doc_id = self.doc_ids[hot_idx[0][0]]
                self.cache.get(str(doc_id))
                return True
                
        if not hit:
            self.stats["misses"] += 1
            self.stats["cold_lookups"] += 1
            q = query_emb.reshape(1, -1)
            cold_sims, cold_idx = self.cold_index.search(q, 1)
            best_idx = cold_idx[0][0]
            best_doc_id = str(self.doc_ids[best_idx])
            best_doc_emb = self.doc_embs[best_idx]
            
            evicted = self.cache.put(best_doc_id, {"vector": best_doc_emb})
            self.hot_index.add_with_ids(best_doc_emb.reshape(1, -1), np.array([best_idx]))
            
            if evicted:
                for ev_key in evicted:
                    ev_idx = np.where(self.doc_ids == ev_key)[0][0]
                    self.hot_index.remove_ids(np.array([ev_idx]))
                    self.stats["evictions"] += 1
            
            return False

def run_simulation(stream_path, capacity=1000, threshold=0.85):
    query_embs, doc_embs, doc_ids = load_data(stream_path)
    
    import src.vector_arc_cache
    src.vector_arc_cache._EMBEDDING_DIM = doc_embs.shape[1]
    
    sim = SCRLSimulator(doc_embs, doc_ids, capacity=capacity, threshold=threshold)
    
    print(f"Running simulation on {len(query_embs)} queries with cache capacity {capacity}...")
    start = time.time()
    
    for i in tqdm(range(len(query_embs))):
        sim.process_query(query_embs[i])
        
    elapsed = time.time() - start
    
    print("\n=== Simulation Results ===")
    print(f"Total Queries: {len(query_embs)}")
    print(f"Cache Hits:    {sim.stats['hits']} ({sim.stats['hits']/len(query_embs)*100:.2f}%)")
    print(f"Cache Misses:  {sim.stats['misses']} ({sim.stats['misses']/len(query_embs)*100:.2f}%)")
    print(f"Cold Lookups:  {sim.stats['cold_lookups']}")
    print(f"Evictions:     {sim.stats['evictions']}")
    print(f"Time Elapsed:  {elapsed:.2f}s ({(elapsed/len(query_embs))*1000:.2f} ms/query)")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", type=str, default="/mnt/c/Users/thesa/OneDrive/Desktop/ML Projects/scrl/scrl/datasets/streams/abrupt")
    parser.add_argument("--capacity", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    
    run_simulation(args.stream, args.capacity, args.threshold)
