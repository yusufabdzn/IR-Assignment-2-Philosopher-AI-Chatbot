from llm import call_llm

# ── Tier-1: Hard conversational fillers — never need a search ────────────────
_CONVERSATIONAL_FILLERS = frozenset([
    "hello", "hi", "hey", "thanks", "thank you", "good morning",
    "good evening", "good night", "quit", "bye", "goodbye",
])

# ── Tier-2: Strong factual signals — always search, no LLM call needed ───────
_ALWAYS_SEARCH_KEYWORDS = [
    "latest", "news", "current", "update", "today", "tonight",
    "price", "weather", "stock", "score", "who is", "what is",
    "where is", "when is", "how much", "2024", "2025", "2026",
]

# ── Tier-3: Strong abstract/philosophical signals — never need a search ───────
_NEVER_SEARCH_KEYWORDS = [
    "meaning of", "philosophy of", "what do you think", "how do i feel",
    "should i", "help me understand", "explain", "define",
]


def should_search(query: str) -> bool:
    """
    Tiered semantic router. Reduces round-trip latency by resolving the
    majority of queries with cheap local checks before falling back to an
    upstream LLM classification call.

    Tier 1 — Conversational filler  → False  (no API call)
    Tier 2 — Strong factual signal  → True   (no API call)
    Tier 3 — Strong abstract signal → False  (no API call)
    Tier 4 — Ambiguous              → LLM classification call
    Tier 5 — LLM call failed        → keyword fallback (same as Tier 2)

    Fix applied:
    Previously every non-filler message triggered an LLM round-trip just
    to classify intent, doubling perceived latency. The two keyword tiers
    resolve the common cases instantly; the LLM is only consulted for
    genuinely ambiguous queries where it adds real value.
    """
    lowered = query.lower().strip()

    # Tier 1: conversational filler — cheap set lookup
    if any(filler in lowered for filler in _CONVERSATIONAL_FILLERS):
        return False

    # Tier 2: strong factual signals — search immediately
    if any(kw in lowered for kw in _ALWAYS_SEARCH_KEYWORDS):
        return True

    # Tier 3: strong abstract/philosophical signals — skip search
    if any(kw in lowered for kw in _NEVER_SEARCH_KEYWORDS):
        return False

    # Tier 4: ambiguous — pay the LLM latency cost only here
    routing_prompt = [
        {
            "role": "system",
            "content": (
                "You are an expert intent-routing mechanism for a search-augmented AI agent.\n"
                "Your sole job is to determine if the user's query requires real-time, historical, "
                "or external factual data that an offline model would not inherently know with 100% precision.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "- Respond with exactly one word: 'SEARCH' if it needs web context retrieval.\n"
                "- Respond with exactly one word: 'NONE' if it is a general question, abstract reasoning, "
                "philosophical thought, or standard conversational interaction.\n"
                "- Do not include any punctuation, conversational filler, or explanations."
            ),
        },
        {"role": "user", "content": f"Query to evaluate: {query}"},
    ]

    try:
        decision = call_llm(routing_prompt).strip().upper()
        return "SEARCH" in decision

    except Exception as e:
        # Tier 5: deterministic fallback if the routing call fails or times out
        print(f"⚠️ Routing LLM call failed, using keyword fallback: {e}")
        return any(kw in lowered for kw in _ALWAYS_SEARCH_KEYWORDS)
