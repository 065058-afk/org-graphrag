from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from src.extraction import extract
from src.entity_resolution import EntityResolver, embedder, normalize
from src.graph_loader import load_chunk, load_chunk_graph, existing_entities
from src.baseline_rag import add_document_to_baseline

_splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
_resolver = None

def _get_resolver():
    global _resolver
    if _resolver is None:
        _resolver = EntityResolver()
        _resolver.preload(existing_entities())
    return _resolver

def ingest_text(text: str, filename: str):
    baseline_added = add_document_to_baseline(text, filename)
    resolver = _get_resolver()
    chunks = [n.get_content() for n in _splitter.get_nodes_from_documents([Document(text=text)])]

    for i, chunk in enumerate(chunks):
        chunk_id = f"{filename}_{i}"
        load_chunk(chunk_id, chunk, embedder.encode(chunk).tolist(), dept="uploaded")

        extraction = extract(chunk, chunk_id)
        entities, name_map = [], {}
        for ent in extraction.entities:
            canon = resolver.resolve(ent.name, ent.type)
            if canon:
                name_map[normalize(ent.name)] = canon
                entities.append((canon, resolver.types[canon]))

        relations = []
        for r in extraction.relations:
            s, t = name_map.get(normalize(r.source)), name_map.get(normalize(r.target))
            if s and t and s != t:
                relations.append((s, r.relation, t))

        load_chunk_graph(chunk_id, list(set(entities)), list(set(relations)))

    return {"graphrag_chunks_added": len(chunks), "baseline_chunks_added": baseline_added}