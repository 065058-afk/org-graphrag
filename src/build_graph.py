import csv
import os
import time
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from src.extraction import extract
from src.entity_resolution import EntityResolver, embedder, normalize
from src.graph_loader import (init_vector_index, load_chunk, load_chunk_graph,
                              done_chunk_ids, existing_entities)

LOG = "eval/graph_build_log.csv"

def build_graph(limit=None):
    init_vector_index()
    docs = SimpleDirectoryReader("data/parsed", filename_as_id=True).load_data()
    nodes = SentenceSplitter(chunk_size=500, chunk_overlap=50).get_nodes_from_documents(docs)
    if limit:
        nodes = nodes[:limit]

    resolver = EntityResolver()
    resolver.preload(existing_entities())
    done = done_chunk_ids()

    new_log = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as lf:
        log = csv.writer(lf)
        if new_log:
            log.writerow(["chunk_id", "seconds", "entities", "relations"])

        for i, node in enumerate(nodes):
            chunk_id = f"chunk_{i}"
            if chunk_id in done:
                continue
            start = time.time()
            text = node.get_content()
            dept = node.metadata.get("file_name", "unknown").replace(".md", "")
            load_chunk(chunk_id, text, embedder.encode(text).tolist(), dept)

            result = extract(text, chunk_id)
            name_map, entities = {}, []
            for ent in result.entities:
                canon = resolver.resolve(ent.name, ent.type)
                if canon:
                    name_map[normalize(ent.name)] = canon
                    entities.append((canon, resolver.types[canon]))

            relations = []
            for r in result.relations:
                s, t = name_map.get(normalize(r.source)), name_map.get(normalize(r.target))
                if s and t and s != t:
                    relations.append((s, r.relation, t))

            entities, relations = list(set(entities)), list(set(relations))
            load_chunk_graph(chunk_id, entities, relations)

            elapsed = round(time.time() - start, 1)
            log.writerow([chunk_id, elapsed, len(entities), len(relations)])
            lf.flush()
            print(f"[{i + 1}/{len(nodes)}] {elapsed}s - {len(entities)} entities, "
                  f"{len(relations)} relations", flush=True)

if __name__ == "__main__":
    build_graph()