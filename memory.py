class MemoryStore:
    def __init__(self):
        self.data = []
        self.summary = ""   # NEW: compressed memory

    def add(self, user, assistant):
        self.data.append({
            "user": user,
            "assistant": assistant
        })

    def search(self, query):
        # naive retrieval first (we improve later)
        results = self.data[-5:]
        return "\n".join([str(r) for r in results])