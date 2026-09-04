"""
cache.py -- Part 4 Task 16: in-memory response cache keyed by normalized
query text, for the grounded-generation step.
"""
import re
import time


def normalize_query(query: str) -> str:
    q = query.strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^\w\s]", "", q)
    return q


class ResponseCache:
    def __init__(self):
        self._store = {}
        self.hits = 0
        self.misses = 0

    def get(self, query: str):
        key = normalize_query(query)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def set(self, query: str, value):
        key = normalize_query(query)
        self._store[key] = value

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "entries": len(self._store),
        }


if __name__ == "__main__":
    import time as _t

    cache = ResponseCache()
    call_counter = {"n": 0}

    def expensive_grounded_generation(query: str):
        """Stands in for rag_core.grounded_answer(): simulates real work with
        a sleep and increments a call counter so the cache-hit demo has
        concrete before/after evidence."""
        call_counter["n"] += 1
        _t.sleep(0.05)
        return f"[answer for: {query}]"

    def answer_with_cache(query: str):
        cached = cache.get(query)
        if cached is not None:
            return cached, True
        result = expensive_grounded_generation(query)
        cache.set(query, result)
        return result, False

    q = "What documents do I need for KYC?"
    t0 = time.perf_counter()
    r1, hit1 = answer_with_cache(q)
    t1 = time.perf_counter()
    r2, hit2 = answer_with_cache("  What DOCUMENTS do I need for KYC?  ")  # same query, different casing/whitespace
    t2 = time.perf_counter()

    print(f"1st call: cache_hit={hit1} latency={t1-t0:.4f}s calls_to_generation={call_counter['n']}")
    print(f"2nd call (same query, normalized): cache_hit={hit2} latency={t2-t1:.4f}s calls_to_generation={call_counter['n']}")
    print("Cache stats:", cache.stats())
    assert hit2 is True and call_counter["n"] == 1, "expected a cache hit that avoided a 2nd generation call"
    print("VERIFIED: repeated query produced a cache hit and avoided a redundant generation call.")
