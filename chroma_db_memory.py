import os
import chromadb
import uuid
from sentence_transformers import SentenceTransformer

# Minimum character length a summary must reach before we trust it
# and commit the purge of the source memory entries.
_MIN_SUMMARY_LENGTH = 80


class MemoryStore:
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name="agent_memory")

        # Persistent summary retrieval on initialization
        if os.path.exists("summary.txt"):
            with open("summary.txt", "r", encoding="utf-8") as f:
                self.summary = f.read().strip()
        else:
            self.summary = ""

    def add(self, text: str, metadata: dict = None):
        """Store a memory chunk with strict metadata enforcement."""
        embedding = self.encoder.encode(text).tolist()

        if not metadata:
            metadata = {"source": "agent_interaction"}

        self.collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    def search(self, query: str, k: int = 3) -> str:
        if self.collection.count() == 0:
            return ""

        query_embedding = self.encoder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        if results and results["documents"] and results["documents"][0]:
            return "\n".join(results["documents"][0])
        return ""

    def compact_memory(self, llm_call):
        """
        Compresses raw memory into a core summary using Incremental Batching
        to prevent token overflows.
        """
        all_data = self.collection.get()
        all_docs = all_data["documents"]
        all_ids = all_data["ids"]

        # Trigger compaction only if we have accumulated enough logs
        if len(all_docs) < 8:
            return

        # Incremental sliding window — only process the oldest 5 items at a time.
        BATCH_SIZE = 5
        docs_to_compress = all_docs[:BATCH_SIZE]
        ids_to_compress = all_ids[:BATCH_SIZE]

        history_to_compress = (
            f"Existing Summary:\n{self.summary}\n\nNew Interactions:\n"
            + "\n".join(docs_to_compress)
        )

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a memory compression system for an AI. "
                    "Your job is to merge the existing summary with the new interactions "
                    "into a single, consolidated profile. Extract key facts, user preferences, "
                    "and important historical events.\n"
                    "CRITICAL: Keep the final consolidated summary under 400 words. "
                    "Remove duplicates or conversational noise."
                ),
            },
            {"role": "user", "content": history_to_compress},
        ]

        try:
            updated_summary = llm_call(prompt)

            # ── VALIDATION GATE ──────────────────────────────────────────────
            # Only commit the new summary and purge source entries if the LLM
            # returned something substantive. A blank string, a one-word reply,
            # or a suspiciously short response means the compaction call failed
            # silently; in that case we keep the original memories intact.
            if not updated_summary or len(updated_summary.strip()) < _MIN_SUMMARY_LENGTH:
                print(
                    f"⚠️ Compaction aborted: summary too short "
                    f"({len(updated_summary.strip()) if updated_summary else 0} chars). "
                    "Original memories preserved."
                )
                return
            # ─────────────────────────────────────────────────────────────────

            self.summary = updated_summary.strip()

            # Write the validated summary to disk
            with open("summary.txt", "w", encoding="utf-8") as f:
                f.write(self.summary)

            # Only now — after a successful write — purge the processed batch
            if ids_to_compress:
                self.collection.delete(ids=ids_to_compress)
                print(f"✅ Successfully compacted {len(ids_to_compress)} memory elements.")

        except Exception as e:
            print(f"⚠️ Memory Compaction Failed Silently to preserve app runtime: {e}")
