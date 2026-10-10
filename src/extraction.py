from typing import Literal, get_args
from pydantic import BaseModel
from src.llm import chat_json

EntityType = Literal["Person", "BusinessSegment", "Metric", "Initiative", "Risk", "Department"]
RelationType = Literal["LEADS", "REPORTS_TO", "AFFECTS", "INVESTS_IN", "PART_OF", "MEASURED_BY"]
ENTITY_TYPES = set(get_args(EntityType))
RELATION_TYPES = set(get_args(RelationType))

class Entity(BaseModel):
    name: str
    type: EntityType

class Relation(BaseModel):
    source: str
    relation: RelationType
    target: str

class ExtractionResult(BaseModel):
    entities: list[Entity]
    relations: list[Relation]

EXAMPLE = ('{"entities": [{"name": "<name>", "type": "<one entity type>"}], '
           '"relations": [{"source": "<entity name>", "relation": "<one relation type>", '
           '"target": "<entity name>"}]}')

def _build(data: dict) -> ExtractionResult:
    entities, relations = [], []
    for e in data.get("entities") or []:
        if isinstance(e, dict) and e.get("name") and e.get("type") in ENTITY_TYPES:
            entities.append(Entity(name=str(e["name"]), type=e["type"]))
    for r in data.get("relations") or []:
        if (isinstance(r, dict) and r.get("source") and r.get("target")
                and r.get("relation") in RELATION_TYPES):
            relations.append(Relation(source=str(r["source"]), relation=r["relation"],
                                      target=str(r["target"])))
    return ExtractionResult(entities=entities, relations=relations)

def extract(chunk_text: str, chunk_id: str = "", retries: int = 2) -> ExtractionResult:
    prompt = f"""Extract entities and relations from this annual report excerpt.
Entity types (use exactly these): Person, BusinessSegment, Metric, Initiative, Risk, Department.
Relation types (use exactly these): LEADS, REPORTS_TO, AFFECTS, INVESTS_IN, PART_OF, MEASURED_BY.
Only use names that appear in the text. If nothing relevant, return empty lists.

Text: {chunk_text[:2000]}"""

    for attempt in range(retries + 1):
        try:
            data = chat_json(prompt, EXAMPLE, schema=ExtractionResult.model_json_schema(),
                             num_predict=600)
            return _build(data)
        except Exception as e:
            if attempt == retries:
                print(f"Extraction failed for {chunk_id}: {e}")
                return ExtractionResult(entities=[], relations=[])
