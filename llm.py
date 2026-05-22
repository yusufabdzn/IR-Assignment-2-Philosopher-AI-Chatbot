import os
import requests
from dotenv import load_dotenv

load_dotenv()

URL = "https://api.berget.ai/v1/chat/completions"
MODEL = "meta-llama/llama-3.1-8b-instruct"


def call_llm(messages):
    api_key = os.getenv("BERGET_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "BERGET_API_KEY is not set. Add it to your .env file or environment."
        )

    try:
        response = requests.post(
            URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000,
            },
            timeout=30,
        )

        if response.status_code != 200:
            print(f"\n❌ BERGET API ERROR DETECTED (Status Code: {response.status_code})")
            print(f"❌ Error Context text: {response.text}\n")
            raise RuntimeError(f"API Error {response.status_code}: {response.text}")

        data = response.json()

        if "choices" not in data:
            print(f"\n❌ MALFORMED JSON RESPONSE FROM API: {data}\n")
            raise KeyError("The key 'choices' was missing from the returned API JSON structure.")

        return data["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        print(f"\n❌ NETWORK CONNECTION ERROR: Connection to Berget server failed.\n")
        raise RuntimeError(f"Network error calling LLM provider: {e}")
