import json
import re
from ollama import Client
from config import OLLAMA_HOST, OLLAMA_API_KEY, OLLAMA_MODEL

_headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}
client = Client(host=OLLAMA_HOST, headers=_headers, timeout=120)

def chat(prompt: str, schema=None, num_predict=None, temperature=None) -> str:
    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if num_predict is not None:
        options["num_predict"] = num_predict
    resp = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        options=options or None,
    )
    return resp["message"]["content"]

def parse_json_object(text: str) -> dict:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in model output: {text[:80]!r}")
    return json.loads(text[start:end + 1])

def chat_json(prompt: str, example: str, schema=None, num_predict: int = 600) -> dict:
    full_prompt = (f"{prompt}\n\nRespond with ONLY one JSON object - no explanation, no markdown, "
                   f"no code fences - in exactly this shape:\n{example}")
    raw = chat(full_prompt, schema=schema, temperature=0, num_predict=num_predict)
    return parse_json_object(raw)
