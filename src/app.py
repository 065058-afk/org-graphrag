from fastapi import FastAPI, UploadFile, File
from src.baseline_rag import build_baseline_index
from src.retrieval import answer_question, hybrid_retrieve
from src.extraction import extract
from src.entity_resolution import EntityResolver, embedder, normalize
from src.graph_loader import load_chunk, load_chunk_graph, existing_entities
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from docling.document_converter import DocumentConverter
import tempfile

app = FastAPI()
converter = DocumentConverter()
splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)

_resolver = None

def get_resolver():
    global _resolver
    if _resolver is None:
        _resolver = EntityResolver()
        _resolver.preload(existing_entities())
    return _resolver

@app.get("/query_baseline")
def query_baseline(q: str):
    index, _ = build_baseline_index()
    qe = index.as_query_engine(similarity_top_k=5)
    return {"answer": str(qe.query(q))}

@app.get("/query")
def query_graphrag(q: str):
    return {"answer": answer_question(q), "chunks_used": hybrid_retrieve(q)}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    resolver = get_resolver()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = converter.convert(tmp_path)
    text = result.document.export_to_markdown()
    chunks = [n.get_content() for n in splitter.get_nodes_from_documents([Document(text=text)])]

    for i, chunk in enumerate(chunks):
        chunk_id = f"{file.filename}_{i}"
        load_chunk(chunk_id, chunk, embedder.encode(chunk).tolist(), dept="uploaded")

        extraction = extract(chunk, chunk_id)
        entities = []
        name_map = {}
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

    return {"status": "ingested", "chunks_added": len(chunks)}