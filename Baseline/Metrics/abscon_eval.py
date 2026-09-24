from typing import Callable, Dict, List, Tuple

import tiktoken

from build_process_model import build_process_model

_enc = tiktoken.get_encoding("o200k_base")


def token_similarity(s1: str, s2: str) -> float:
    a = set(_enc.encode(s1.lower()))
    b = set(_enc.encode(s2.lower()))
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def soft_cardinality(relation_set, similarity_func: Callable[[str, str], float]) -> float:
    result = 0
    for relation in relation_set:
        similarity = sum(similarity_func(relation, other) for other in relation_set)
        result += 1 / similarity
    return result


def soft_metrics(predicted_relations, gt_relations, similarity_func) -> dict:
    predicted_cardinality = soft_cardinality(predicted_relations, similarity_func)
    gt_cardinality = soft_cardinality(gt_relations, similarity_func)
    union_cardinality = soft_cardinality(
        list(predicted_relations) + list(gt_relations), similarity_func
    )
    intersect_cardinality = predicted_cardinality + gt_cardinality - union_cardinality

    precision = (
        intersect_cardinality / predicted_cardinality if predicted_cardinality else 0
    )
    recall = intersect_cardinality / gt_cardinality if gt_cardinality else 0
    return {"precision": precision, "recall": recall}


def graph_to_relation_set(graph: Dict[str, List[Tuple[str, str]]],
                          node_data: Dict[str, dict]) -> set:
    results = []
    for src, outs in graph.items():
        source_label = node_data.get(src, {}).get("name", str(src))
        for tgt, edge_label in outs:
            target_label = node_data.get(tgt, {}).get("name", str(tgt))
            if edge_label:
                results.append(f"{source_label} {edge_label} {target_label}")
            else:
                results.append(f"{source_label} {target_label}")
    return set(results)


def relation_set_from_csv(csv_text: str) -> set:
    graph, node_data, _root, _edges = build_process_model(csv_text)
    return graph_to_relation_set(graph, node_data)


def abscon_scores(gt_csv_text: str, gen_csv_text: str) -> dict:
    try:
        gt_relations = relation_set_from_csv(gt_csv_text)
        predicted = relation_set_from_csv(gen_csv_text)
        m = soft_metrics(predicted, gt_relations, token_similarity)
        return {"recall": m["recall"], "precision": m["precision"]}
    except Exception:
        return {"recall": 0.0, "precision": 0.0}
