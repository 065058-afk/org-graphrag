from sentence_transformers import CrossEncoder
from src.extraction import extract
from src.entity_resolution import embedder
from src.graph_loader import driver
from src.aggregation import detect_aggregate_intent, extract_numbers_smart, compute_aggregate
from src.llm import chat

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_retrieve(question: str, top_k: int = 5, hops: int = 1):
    q_emb = embedder.encode(question).tolist()

    with driver.session() as session:
        vector_hits = session.run("""
            CALL db.index.vector.queryNodes('chunk_embeddings', $k, $emb)
            YIELD node, score RETURN node.id AS id, node.text AS text, score AS vscore
        """, k=20, emb=q_emb).data()

        q_entities = [e.name.lower().strip() for e in extract(question, "query").entities]
        graph_hits = []
        if q_entities:
            graph_hits = session.run(f"""
                MATCH (e:Entity)-[:RELATION*1..{hops}]-(other:Entity)<-[:MENTIONS]-(c:Chunk)
                WHERE e.name IN $names
                RETURN DISTINCT c.id AS id, c.text AS text, 0.5 AS vscore
            """, names=q_entities).data()

    merged = {}
    for h in vector_hits + graph_hits:
        if h["id"] not in merged:
            merged[h["id"]] = h
    if not merged:
        return []

    items = list(merged.values())
    rerank_scores = reranker.predict([[question, h["text"]] for h in items])

    combined = []
    for h, rscore in zip(items, rerank_scores):
        blended = 0.6 * h["vscore"] + 0.4 * (rscore / 10)
        combined.append((h["text"], blended))

    combined.sort(key=lambda x: -x[1])
    return [text for text, _ in combined[:top_k]]

def answer_with_sources(question: str):
    chunks = hybrid_retrieve(question)
    if not chunks:
        return "I don't have information on that in the ingested documents.", []
    context = "\n\n---\n\n".join(chunks)

    operation = detect_aggregate_intent(question)
    if operation:
        extraction = extract_numbers_smart(context, question)
        result = compute_aggregate(extraction.items, operation)
        if result is not None:
            breakdown = "\n".join(f"- {i.label}: {i.value}" for i in extraction.items)
            prompt = f"""The question asks for a {operation} calculation. The actual computed
{operation} is {result}.

The exact figures used in this calculation were:
{breakdown}

Question: {question}

Write a short answer. State the computed {operation} ({result}) as the answer.
Then list the exact figures above that were used, using their exact labels.
Do not describe or rename these figures differently than how they are labeled above.
Do not recalculate."""
            return chat(prompt), chunks

    prompt = f"""Answer the question using only the context below. Cite which section the info came from.

Context:
{context}

Question: {question}"""
    return chat(prompt), chunks

def answer_question(question: str):
    return answer_with_sources(question)[0]