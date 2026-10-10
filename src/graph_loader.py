from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def init_vector_index():
    with driver.session() as s:
        s.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        s.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        s.run("""
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
        """)

def load_chunk(chunk_id, text, embedding, dept):
    with driver.session() as s:
        s.run("""
            MERGE (c:Chunk {id: $id})
            SET c.text = $text, c.embedding = $emb, c.department = $dept
        """, id=chunk_id, text=text, emb=embedding, dept=dept)

def load_chunk_graph(chunk_id, entities, relations):
    with driver.session() as s:
        for name, etype in entities:
            s.run("""
                MERGE (e:Entity {name: $name}) ON CREATE SET e.type = $type
                WITH e MATCH (c:Chunk {id: $cid}) MERGE (c)-[:MENTIONS]->(e)
            """, name=name, type=etype, cid=chunk_id)
        for src, rel, tgt in relations:
            s.run("""
                MATCH (a:Entity {name: $src}), (b:Entity {name: $tgt})
                MERGE (a)-[r:RELATION {type: $rel}]->(b)
                ON CREATE SET r.chunks = [$cid]
                ON MATCH SET r.chunks = CASE WHEN $cid IN r.chunks
                                             THEN r.chunks ELSE r.chunks + $cid END
            """, src=src, tgt=tgt, rel=rel, cid=chunk_id)
        s.run("MATCH (c:Chunk {id: $cid}) SET c.extracted = true", cid=chunk_id)

def done_chunk_ids():
    with driver.session() as s:
        rows = s.run("MATCH (c:Chunk) WHERE c.extracted = true RETURN c.id AS id").data()
    return {r["id"] for r in rows}

def existing_entities():
    with driver.session() as s:
        rows = s.run("MATCH (e:Entity) RETURN e.name AS name, e.type AS type").data()
    return [(r["name"], r["type"]) for r in rows]
