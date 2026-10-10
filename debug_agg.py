from src.retrieval import hybrid_retrieve
from src.aggregation import extract_numbers, compute_aggregate

question = "What was the average revenue growth over fiscal 2024, 2025, and 2026?"
chunks = hybrid_retrieve(question)

print("RETRIEVED CHUNKS:")
for i, c in enumerate(chunks):
    print(f"--- chunk {i} ---\n{c[:200]}\n")

context = "\n\n---\n\n".join(chunks)
extraction = extract_numbers(context, question)
print("EXTRACTED ITEMS:", extraction)
print("COMPUTED AVERAGE:", compute_aggregate(extraction.items, "average"))
