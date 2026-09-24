import csv
from collections import defaultdict


def build_process_model(csv_text):
    reader = csv.DictReader(line for line in csv_text.strip().splitlines() if not line.startswith("#"))
    graph, node_data, children = defaultdict(list), {}, set()
    for row in reader:
        nid = row["id"].strip()
        node_data[nid] = {
            "name": row["name"].strip(),
            "type": row["type"].strip()
        }
        for p in row["parent"].split(","):
            p = p.strip()
            if not p:
                continue
            graph[p].append((nid, row["transition_label"].strip()))
            children.add(nid)
    roots = list(set(node_data) - children)
    root = roots[0] if roots else None
    total_edges = sum(len(v) for v in graph.values())
    return graph, node_data, root, total_edges
