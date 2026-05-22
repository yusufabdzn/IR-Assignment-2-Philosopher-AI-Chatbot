import os
import re
import random  # Used to inject randomized semantic angles to prevent repetitive outputs
from llm import call_llm
from chroma_db_memory import MemoryStore
from toolbox import ToolBox
from tools.routing import should_search
from tools.web_search import web_search


# ── Crisis keyword detection ──────────────────────────────────────────────────
# Compiled once at import time for efficiency.
# Uses word boundaries (\b) so "die" no longer matches "diet", "died" (in a
# historical context), "diehard", etc.
_CRISIS_PATTERN = re.compile(
    r"\b("
    + "|".join(
        re.escape(kw)
        for kw in [
            "suicide",
            "kill myself",
            "end my life",
            "self-harm",
            "self harm",
            "hopeless",
            "giving up",
            "want to die",
            "going to die",
        ]
    )
    + r")\b",
    re.IGNORECASE,
)
# ─────────────────────────────────────────────────────────────────────────────


class Agent:
    def __init__(self, api_key=None):
        self.memory = MemoryStore()
        self.chat_history = []
        self.api_key = api_key or os.getenv("BERGET_API_KEY")
        self.tools = ToolBox(self.api_key) if self.api_key else None

    def chat(self, user_message, philosopher_mode=False):
        # 1. Retrieve vector database history (IR Step)
        relevant_memory = self.memory.search(user_message)

        # 2. DETECT SEVERE DISTRESS / CRISIS INTENT
        # Fix: word-boundary regex replaces the naive substring check that
        # falsely triggered on words like "diet", "diehard", or historical uses
        # of "died".
        is_crisis = bool(_CRISIS_PATTERN.search(user_message))

        # 3. Context Retrieval Track
        # Fix: The "-term" blacklist syntax (e.g. "-yoga") is a Google DSL
        # feature that Tavily silently ignores, so it was polluting every query
        # string with literal garbage text. Blocklist filtering is now handled
        # inside web_search.py at the result level, where it actually works.
        if is_crisis:
            safe_crisis_queries = [
                "stoic philosophy quotes on enduring hardship and finding hope",
                "Islamic and Islamicate philosophy quotes on hope, resilience, and the value of life in hard times",
                "Quranic quotes on hope, resilience, and the value of life in hard times",
                "Hadith quotes on hope, resilience, and the value of life in hard times",
                "historical quotes about finding hope in dark times and choosing to live",
                "philosophical quotes about overcoming severe hardship and building inner strength",
                "stoic wisdom on enduring emotional pain and deep resilience",
                "existential philosophy quotes on finding meaning in life and enduring suffering",
                "profound historical wisdom and quotes on perseverance through trials and staying strong",
                "inspiring historical figures quotes on finding light in despair and moving forward",
            ]
            selected_search = random.choice(safe_crisis_queries)
            web_context = web_search(selected_search, self.api_key)

        elif philosopher_mode:
            perspectives = [
                "stoicism",
                "Islamic philosophy",
                "stoic and grit",
                "Quranic wisdom",
                "Hadith wisdom",
                "Go rin no sho",
                "Dao de jing",
                "existential and purpose",
                "eastern philosophy",
                "ethics and morality",
                "Japanese philosophy",
                "Chinese philosophy",
                "psychology",
            ]
            selected_angle = random.choice(perspectives)
            search_query = f"historical quotes about {user_message} focus on {selected_angle}"
            web_context = web_search(search_query, self.api_key)

        elif should_search(user_message):
            web_context = web_search(user_message, self.api_key)

        else:
            web_context = ""

        # 4. Dynamic System Instructions
        if is_crisis:
            system_instruction = """You are operating in a COMPASSIONATE AND RESILIENT PHILOSOPHER MODE.
            The user is expressing profound distress. Your absolute objective is to provide comfort, hope, and practical philosophical advice on choosing to live, finding internal power, and enduring hardship.

            STRICT LAYOUT AND STRUCTURAL CONSTRAINTS:
            1. MAIN QUOTE: Read the provided Web Context and extract exactly ONE primary historical quote at the very top that encourages resilience, hope, or finding meaning.
               Format: "Quote" - Author
            
            2. MAIN AUTHOR DETAILS: Immediately below the main quote, provide a brief biographical context sentence enclosed in parentheses.
               Format: (Author was a...)
               
            3. DISCUSSION & ADVICE BLOCK: Below the biography, write a highly supportive, fluid advice essay. 
               - Give actionable advice on finding internal strength and choosing to move forward. Avoid abstract or academic definitions of death.
               - Create elegant text transitions so the insights flow smoothly without sudden jumps.
            
            4. SUPPORTING EVIDENCE WITH FULL BIOGRAPHIES: Inside your discussion, you MUST introduce at least one or two supporting quotes from DIFFERENT authors found in your context or knowledge base to reinforce your point.
               - CRITICAL MANDATE: Every supporting author cited must be provided with short historical background context inside parentheses immediately after their quote or name.
               Format inside text: "Supporting Quote" - Author (A short phrase describing what they are/were known for).
               Example: "All mankind is from Adam and Eve, a white has no superiority over black nor a black has any superiority over white except by piety and good action." - Prophet Muhammad (Prophet of Islam).
            
            5. SAFETY BAN: Never romanticize, normalize, or analyze the desire to die. Keep the tone completely focused on action, advice, and survival.
            
            Do not include conversational filler or intros. Start immediately with the main quote.
            """
        elif philosopher_mode:
            system_instruction = """You are operating strictly in ADVANCED ANALYTICAL PHILOSOPHER MODE.
            Your absolute objective is to select and break down historical wisdom regarding the user's topic.
            
            STRICT LAYOUT AND STRUCTURAL CONSTRAINTS:
            1. MAIN QUOTE: You must output exactly ONE primary historical quote at the very top.
               Format: "Quote" - Author
            
            2. MAIN AUTHOR DETAILS: Immediately below the main quote, provide a brief biographical context sentence enclosed in parentheses.
               Format: (Author was a...)
               
            3. DISCUSSION BLOCK: Below the biography, write a deep, cohesive philosophical essay.
               - Craft eloquent transition sentences so the discussion text bridges ideas and authors seamlessly without jumpy or abrupt cuts.
            
            4. SUPPORTING EVIDENCE WITH FULL BIOGRAPHIES: Within your discussion, you must incorporate supporting quotes from DIFFERENT authors.
               - CRITICAL MANDATE: Every supporting author cited must be provided with short historical background context inside parentheses immediately after their quote or name.
               Format inside text: "Supporting Quote" - Author (A short phrase describing what they are/were known for).
               Example: "All mankind is from Adam and Eve, a white has no superiority over black nor a black has any superiority over white except by piety and good action." - Prophet Muhammad (Prophet of Islam).
            
            5. NAMING EXCEPTION: If you cite or attribute quotes to the historical figure Muhammad, you must strictly write the attribution as "- Prophet Muhammad" (Prophet Muhammad was...). Never shorten it to just "- Muhammad".
            
            Do not include introductions or conversational pleasantries. Start immediately with the layout.
            """
        else:
            system_instruction = "You are a helpful and wise AI Assistant."

        # SAFETY VALVE 1: Strict Character Caps on Ingested Text Frameworks
        if web_context and len(web_context) > 6000:
            web_context = web_context[:6000] + "\n... [Web Context Truncated for Length] ..."
        if relevant_memory and len(relevant_memory) > 2000:
            relevant_memory = relevant_memory[:2000] + "\n... [Memory Truncated for Length] ..."

        # 5. Prompt Assembly with Integrated Global Content Constraints
        system_content = f"""{system_instruction}
                
CRITICAL NEGATIVE CONTENT CONSTRAINTS:
- Absolute Prohibition: Do NOT reference, mention, quote, or incorporate any philosophies, practices, ideologies, or movements related to: yoga, zionism, nazism, or associated geopolitical/religious extremes.
- If the web context or your internal knowledge base contains these topics, filter them out completely and replace them with alternative universal historical perspectives.

Core Memory Summary:
{self.memory.summary}

Relevant Past Experiences:
{relevant_memory}

Current Web Context:
{web_context}
"""

        # SAFETY VALVE 2: Dynamic Short-Term History Sliding Window Budget
        MAX_CHAR_BUDGET = 22000
        active_history = []
        total_chars = len(system_content) + len(user_message)

        for msg in reversed(self.chat_history):
            msg_len = len(msg["content"])
            if total_chars + msg_len < MAX_CHAR_BUDGET:
                active_history.insert(0, msg)
                total_chars += msg_len
            else:
                break

        messages = (
            [{"role": "system", "content": system_content}]
            + active_history
            + [{"role": "user", "content": user_message}]
        )

        # 7. Execute LLM Call
        response = call_llm(messages)

        # 8. Short-Term Memory Append
        self.chat_history.append({"role": "user", "content": user_message})
        self.chat_history.append({"role": "assistant", "content": response})
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        # 9. Long-Term Vector DB Append
        self.memory.add(
            f"User: {user_message}\nAssistant: {response}",
            metadata={"session_track": "philosopher_chat"},
        )
        self.memory.compact_memory(call_llm)

        return response
