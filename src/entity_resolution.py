import torch
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def normalize(name: str) -> str:
    return " ".join(name.lower().strip().split())

class EntityResolver:
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.names = []
        self.types = {}
        self.embs = None

    def preload(self, pairs):
        if not pairs:
            return
        self.names = [p[0] for p in pairs]
        self.types = dict(pairs)
        self.embs = embedder.encode(self.names, convert_to_tensor=True)

    def resolve(self, name: str, etype: str):
        norm = normalize(name)
        if not norm:
            return None
        if norm in self.types:
            return norm
        emb = embedder.encode(norm, convert_to_tensor=True)
        if self.embs is not None:
            sims = util.cos_sim(emb, self.embs)[0]
            best = int(sims.argmax())
            if float(sims[best]) > self.threshold and self.types[self.names[best]] == etype:
                return self.names[best]
        self.names.append(norm)
        self.types[norm] = etype
        row = emb.unsqueeze(0)
        self.embs = row if self.embs is None else torch.cat([self.embs, row])
        return norm