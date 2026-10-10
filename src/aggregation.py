from pydantic import BaseModel
import re
import ollama
from config import OLLAMA_MODEL
from src.llm import chat_json

AGGREGATE_KEYWORDS = {
    "sum": ["total", "sum", "combined", "altogether"],
    "average": ["average", "mean", "on average"],
    "max": ["highest", "maximum", "largest", "greatest", "peak"],
    "min": ["lowest", "minimum", "smallest", "least"],
    "count": ["how many", "number of"],
}

class NumericItem(BaseModel):
    label: str
    value: float

class NumericExtraction(BaseModel):
    items: list[NumericItem]

def detect_aggregate_intent(question: str):
    q = question.lower()
    for op, phrases in AGGREGATE_KEYWORDS.items():
        if any(p in q for p in phrases):
            return op
    return None

def extract_question_years(question: str):
    return [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", question)]

def regex_pair_values_years(text: str):
    """Deterministic parser for the common 'value, year, value, year...'
    pattern produced when Docling flattens a table into plain text.
    No LLM involved - this is exact, not a guess."""
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    pairs = []
    i = 0
    value_re = re.compile(r"^-?\d+\.\d+%?$")
    year_re = re.compile(r"^(19|20)\d{2}$")
    while i < len(tokens) - 1:
        t1, t2 = tokens[i], tokens[i + 1]
        if value_re.match(t1) and year_re.match(t2):
            pairs.append((float(t1.rstrip("%")), int(t2)))
            i += 2
            continue
        if year_re.match(t1) and value_re.match(t2):
            pairs.append((float(t2.rstrip("%")), int(t1)))
            i += 2
            continue
        i += 1
    return pairs

def extract_numbers_smart(context: str, question: str) -> NumericExtraction:
    """Only ever extract numbers from chunks that are topically relevant to
    the question. This applies to both the regex path and the LLM fallback -
    a numerically clean but unrelated chunk (e.g. revenue growth figures
    showing up for a market-cap question) must never be used just because
    it happens to contain parseable numbers."""
    question_years = extract_question_years(question)
    question_keywords = [w for w in re.findall(r"[a-zA-Z]{4,}", question.lower())
                          if w not in {"what", "which", "among", "across", "fiscal", "highest",
                                        "lowest", "average", "total", "year", "years", "growth"}]

    relevant_chunks = [
        chunk for chunk in context.split("---")
        if any(kw in chunk.lower() for kw in question_keywords)
    ]

    if not relevant_chunks:
        return NumericExtraction(items=[])

    relevant_pairs = []
    for chunk in relevant_chunks:
        relevant_pairs.extend(regex_pair_values_years(chunk))

    if question_years:
        matched = {y: v for v, y in relevant_pairs if y in question_years}
        if len(matched) >= 2:
            items = [NumericItem(label=f"Fiscal {y}", value=v) for y, v in sorted(matched.items())]
            return NumericExtraction(items=items)

    if relevant_pairs:
        seen_years = {}
        for v, y in relevant_pairs:
            seen_years.setdefault(y, v)
        items = [NumericItem(label=f"Fiscal {y}", value=v) for y, v in sorted(seen_years.items())]
        if items:
            return NumericExtraction(items=items)

    relevant_context = "\n\n---\n\n".join(relevant_chunks)
    return extract_numbers_llm(relevant_context, question)

NUMERIC_EXAMPLE = '{"items": [{"label": "<what the number is>", "value": 0.0}]}'

def extract_numbers_llm(context: str, question: str, retries: int = 2) -> NumericExtraction:
    prompt = f"""Extract every numeric figure from the context below that is relevant to
answering the question. Do NOT calculate anything. Just list each relevant
number with a short label describing what it is.

Context:
{context[:3000]}

Question: {question}

If no relevant numbers exist, return an empty list."""

    for attempt in range(retries + 1):
        try:
            data = chat_json(prompt, NUMERIC_EXAMPLE,
                             schema=NumericExtraction.model_json_schema(), num_predict=600)
            items = []
            for it in data.get("items") or []:
                try:
                    items.append(NumericItem(label=str(it["label"]),
                                             value=float(str(it["value"]).replace(",", ""))))
                except Exception:
                    continue
            return NumericExtraction(items=items)
        except Exception as e:
            if attempt == retries:
                print(f"Numeric extraction failed: {e}")
                return NumericExtraction(items=[])

def compute_aggregate(items: list[NumericItem], operation: str):
    values = [i.value for i in items]
    if not values:
        return None
    if operation == "sum":
        return sum(values)
    if operation == "average":
        return sum(values) / len(values)
    if operation == "max":
        return max(values)
    if operation == "min":
        return min(values)
    if operation == "count":
        return len(values)
    return None
