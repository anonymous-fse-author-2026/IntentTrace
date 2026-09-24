import os
from collections import defaultdict, deque

import numpy as np
from sentence_transformers import SentenceTransformer

from build_process_model import build_process_model


class Embedder:
    _model_cache = {}

    @classmethod
    def _load(cls, model_name):
        if model_name in cls._model_cache:
            return cls._model_cache[model_name]

        base_dir = os.path.dirname(os.path.abspath(__file__))
        vendored = os.path.join(base_dir, "cache", model_name.replace("/", "_"))
        candidates = [vendored, None] if os.path.isdir(vendored) else [None]

        last_exc = None
        for cache_folder in candidates:
            try:
                model = SentenceTransformer(
                    model_name,
                    trust_remote_code=True,
                    cache_folder=cache_folder,
                    local_files_only=True,
                )
                break
            except Exception as exc:
                last_exc = exc
        else:
            raise RuntimeError(
                f"could not load {model_name} from the vendored cache "
                f"({vendored}) or the default HuggingFace cache. Pre-download "
                f"it once with network access, then re-run offline."
            ) from last_exc

        cls._model_cache[model_name] = model
        return model

    def __init__(self, model_name="Alibaba-NLP/gte-base-en-v1.5"):
        self.model = self._load(model_name)
        self._cache = {}

    def encode(self, texts):
        to_encode, idxs = [], []
        for i, t in enumerate(texts):
            if t not in self._cache:
                idxs.append(i)
                to_encode.append(t)
        if to_encode:
            embs = self.model.encode(to_encode, normalize_embeddings=True)
            for txt, emb in zip(to_encode, embs):
                self._cache[txt] = emb
        return np.stack([self._cache[t] for t in texts])

    def similarity(self, a, b):
        return self.model.similarity(a, b)


def merge_nodes(graph, node_data):
    rev = defaultdict(list)
    for u, outs in graph.items():
        for v, lab in outs:
            rev[v].append((u, lab))

    in_deg = {n: len(rev.get(n, [])) for n in node_data}
    out_deg = {n: len(graph.get(n, [])) for n in node_data}

    linear = {n for n in node_data if in_deg[n] <= 1 and out_deg[n] <= 1}

    visited = set()
    chains = []

    for n in node_data:
        if n not in linear or n in visited:
            continue

        cur = n
        while True:
            preds = [p for (p, _) in rev.get(cur, []) if p in linear]
            if len(preds) == 1:
                cur = preds[0]
            else:
                break

        chain = []
        while cur in linear and cur not in visited:
            visited.add(cur)
            chain.append(cur)
            succs = [c for (c, _) in graph.get(cur, []) if c in linear]
            if len(succs) == 1:
                cur = succs[0]
            else:
                break

        chains.append(chain)

    for n in node_data:
        if n not in visited:
            chains.append([n])

    old2new = {}
    new_node_data = {}
    for chain in chains:
        if len(chain) == 1:
            nid = chain[0]
            new_node_data[nid] = node_data[nid].copy()
            new_node_data[nid]["original_ids"] = [nid]
        else:
            nid = "_".join(chain)
            name_parts = []
            for i, node in enumerate(chain):
                name_parts.append(node_data[node]["name"])
                if i < len(chain) - 1:
                    next_node = chain[i + 1]
                    for tgt, lab in graph[node]:
                        if tgt == next_node and lab:
                            name_parts.append(f"{lab}")
                            break
            merged_name = " / ".join(name_parts)
            merged_type = node_data[chain[-1]]["type"]
            new_node_data[nid] = {
                "name": merged_name,
                "type": merged_type,
                "original_ids": chain
            }
        for old in chain:
            old2new[old] = nid

    new_graph = defaultdict(list)
    for u, outs in graph.items():
        u2 = old2new[u]
        for v, lab in outs:
            v2 = old2new[v]
            if u2 != v2:
                new_graph[u2].append((v2, lab))

    for u, outs in new_graph.items():
        seen = set()
        unique = []
        for v, lab in outs:
            key = (v, lab)
            if key not in seen:
                seen.add(key)
                unique.append((v, lab))
        new_graph[u] = unique

    return new_graph, new_node_data


def evaluate_threshold_at_end_one_to_many(csv1, csv2, threshold):
    g1, nd1, r1, _ = build_process_model(csv1)
    g2, nd2, r2, _ = build_process_model(csv2)
    g1, nd1 = merge_nodes(g1, nd1)
    g2, nd2 = merge_nodes(g2, nd2)

    children1 = {v for outs in g1.values() for (v, _) in outs}
    roots1 = [n for n in nd1 if n not in children1]
    r1 = roots1[0] if roots1 else None
    children2 = {v for outs in g2.values() for (v, _) in outs}
    roots2 = [n for n in nd2 if n not in children2]
    r2 = roots2[0] if roots2 else None

    emb = Embedder()
    all_texts = set(nd1[n]["name"] for n in nd1) | set(nd2[n]["name"] for n in nd2)
    for outs in (*g1.values(), *g2.values()):
        all_texts |= {lab for (_, lab) in outs}
    all_texts = list(all_texts)
    text2emb = dict(zip(all_texts, emb.encode(all_texts)))

    def sim(a, b):
        return float(emb.similarity(text2emb[a], text2emb[b]))

    all_matches = []
    visited = set()
    queue = deque()

    if r1 is not None and r2 is not None:
        root_sim = sim(nd1[r1]["name"], nd2[r2]["name"])
        queue.append((r1, r2, root_sim))

    while queue:
        u1, u2, score = queue.popleft()
        if (u1, u2) in visited:
            continue
        visited.add((u1, u2))
        all_matches.append((u1, u2, score))

        outs1 = g1.get(u1, [])
        outs2 = g2.get(u2, [])

        for (v1, lab1) in outs1:
            if outs2:
                best_j = None
                best_score = -float("inf")
                for j, (v2, lab2) in enumerate(outs2):
                    s_name = sim(nd1[v1]["name"], nd2[v2]["name"])
                    if lab1 == "" and lab2 == "":
                        combined = s_name
                    else:
                        s_lab = sim(lab1, lab2)
                        combined = (s_lab + s_name) / 2
                    if combined > best_score:
                        best_score = combined
                        best_j = j
                if best_j is not None:
                    v2, _ = outs2[best_j]
                    queue.append((v1, v2, best_score))

    sem_scores = [score for (_, _, score) in all_matches if score >= threshold]
    matched1 = {u1 for (u1, _, score) in all_matches if score >= threshold}
    matched2 = {u2 for (_, u2, score) in all_matches if score >= threshold}

    unmatched1 = [{"id": n, "name": nd1[n]["name"]} for n in nd1 if n not in matched1]
    unmatched2 = [{"id": n, "name": nd2[n]["name"]} for n in nd2 if n not in matched2]

    accept1 = len(matched1)
    accept2 = len(matched2)
    cov1 = accept1 / len(nd1) if len(nd1) > 0 else 0
    cov2 = accept2 / len(nd2) if len(nd2) > 0 else 0
    coverage = 0.5 * (cov1 + cov2)
    agg_sem = np.mean(sem_scores) if sem_scores else 0
    final_score = (coverage + agg_sem) / 2

    return {
        "coverage_1_to_2": cov1,
        "coverage_2_to_1": cov2,
        "accept1": accept1,
        "accept2": accept2,
        "total1": len(nd1),
        "total2": len(nd2),
        "coverage": coverage,
        "aggregated_semantic": agg_sem,
        "final_score": final_score,
        "sem_scores": str(sem_scores),
        "unmatched_nodes_chart1": unmatched1,
        "unmatched_nodes_chart2": unmatched2
    }
