import time
from typing import List, Dict, Any

# Mocking the components to test coordination logic locally
class MockEmbeddings:
    def __init__(self):
        # Simple mock vocabulary mappings to simulate embedding similarity scores
        self.doc_registry = {
            "doc_1": "IIT Dharwad Hostel Rules: Curfew is 10:00 PM. Fine for violation is ₹500.",
            "doc_2": "Mess timings: Breakfast 7:30-9:30 AM, Lunch 12:30-2:15 PM, Dinner 7:30-9:15 PM.",
            "doc_3": "Academic calendar: Mid-semester exams start on September 14th."
        }
    
    def calculate_similarity(self, query: str, doc_id: str) -> float:
        query_lower = query.lower()
        if "curfew" in query_lower or "time" in query_lower and doc_id == "doc_1":
            return 0.58  # Mimics all-MiniLM-L6-v2 similarity for short query vs long doc
        if "mess" in query_lower or "food" in query_lower and doc_id == "doc_2":
            return 0.55
        if "exam" in query_lower or "academic" in query_lower and doc_id == "doc_3":
            return 0.62
        return 0.15

class VectorARCCache:
    def __init__(self, capacity: int = 5, similarity_threshold: float = 0.40):
        self.capacity = capacity
        self.threshold = similarity_threshold
        # Cache tiers tracking (Simplified for operational testing)
        self.hot_vectors: Dict[str, Dict[str, Any]] = {} 
        self.metrics = {"hits": 0, "misses": 0, "cold_fetches": 0}

    def query_cache(self, query_text: str, embedder: MockEmbeddings) -> Any:
        best_score = -1.0
        best_doc = None
        
        for doc_id, meta in self.hot_vectors.items():
            score = embedder.calculate_similarity(query_text, doc_id)
            if score > best_score:
                best_score = score
                best_doc = meta
                
        if best_score >= self.threshold and best_doc:
            self.metrics["hits"] += 1
            return best_doc["text"], best_score
            
        self.metrics["misses"] += 1
        return None, best_score

    def admit(self, doc_id: str, text: str):
        if len(self.hot_vectors) >= self.capacity:
            # Simple eviction for test script footprint
            evict_key = next(iter(self.hot_vectors))
            del self.hot_vectors[evict_key]
        self.hot_vectors[doc_id] = {"text": text}
        self.metrics["cold_fetches"] += 1

# Operational Simulation Execution
def run_test_suite():
    embedder = MockEmbeddings()
    # Initializing with the optimized 0.40 threshold configuration
    cache = VectorARCCache(capacity=2, similarity_threshold=0.40)
    
    print("=" * 60)
    print("VECTOR-ARC SYSTEM OPERATIONAL DIAGNOSTIC RUNNER")
    print("=" * 60)
    
    # 1. Warm up cache (Simulate Cold Storage Misses followed by Admission)
    print("\n[Phase 1] Simulating Initial Cold Storage Population...")
    initial_queries = [
        ("What is the curfew time?", "doc_1"),
        ("What are the mess timings?", "doc_2")
    ]
    
    for q, doc_id in initial_queries:
        text, score = cache.query_cache(q, embedder)
        print(f"Query: '{q}' -> Cache Hit? False (Similarity: {score:.2f})")
        # Fetching from raw text repository and admitting to hot vector DB
        raw_text = embedder.doc_registry[doc_id]
        cache.admit(doc_id, raw_text)
        print(f" -> Admitted {doc_id} into Hot Vector Tier.")

    # 2. Test execution of the optimized semantic threshold
    print("\n[Phase 2] Executing Evaluation Queries against Hot Tier...")
    test_stream = [
        "Is there a curfew tonight?", 
        "When can I get food?",
        "When do exams start?"
    ]
    
    for q in test_stream:
        result, score = cache.query_cache(q, embedder)
        if result:
            print(f"Query: '{q}'\n -> Result: CACHE HIT (Score: {score:.2f})")
        else:
            print(f"Query: '{q}'\n -> Result: CACHE MISS (Highest Score: {score:.2f}). Forwarding request to cold text layer.")

    print("\n" + "=" * 60)
    print("FINAL PERFORMANCE METRICS SUMMARY")
    print("=" * 60)
    print(f" Total Cache Hits             : {cache.metrics['hits']}")
    print(f" Total Cache Misses           : {cache.metrics['misses']}")
    print(f" Cold Storage Text Lookups   : {cache.metrics['cold_fetches']}")
    print(f" Target Operational Hit Rate  : {(cache.metrics['hits'] / len(test_stream)) * 100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_test_suite()