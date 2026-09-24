from __future__ import annotations

from typing import Any, Dict, List, Tuple

FLOW_KINDS = {"init", "end", "branch", "converge", "action"}
CONTENT_KINDS = {"action", "label", "guard"}


def is_content_kind(kind: Any) -> bool:
    return str(kind or "").strip().lower() in CONTENT_KINDS


class ParsedModel:
    def __init__(
        self,
        nodes: List[Dict[str, Any]],
        precedences: List[Dict[str, str]],
        guards: List[Dict[str, str]],
        labels: List[Dict[str, str]],
        variables: List[Dict[str, Any]],
        actors: List[Dict[str, Any]],
    ) -> None:
        self.nodes = nodes
        self.precedences = precedences
        self.guards = guards
        self.labels = labels
        self.variables = variables
        self.actors = actors

    def node_index(self) -> Dict[str, Dict[str, Any]]:
        return {str(n["id"]): n for n in self.nodes if n.get("id")}


def _norm(value: Any) -> str:
    return str(value if value is not None else "").strip()


def build_parsed_model(final_output: Dict[str, Any]) -> ParsedModel:
    final_output = final_output or {}

    nodes: List[Dict[str, Any]] = []
    for elem in final_output.get("elements", []) or []:
        if not isinstance(elem, dict):
            continue
        node_id, kind = _norm(elem.get("id")), _norm(elem.get("type")).lower()
        if not node_id or kind not in FLOW_KINDS:
            continue
        nodes.append({
            "id": node_id,
            "kind": kind,
            "content": _norm(elem.get("content")),
            "gateway": _norm(elem.get("gateway")).upper(),
        })

    precedences: List[Dict[str, str]] = []
    guards: List[Dict[str, str]] = []
    labels: List[Dict[str, str]] = []
    for rel in final_output.get("relationships", []) or []:
        if not isinstance(rel, dict):
            continue
        rtype = _norm(rel.get("type")).lower()
        src, dst = _norm(rel.get("A")), _norm(rel.get("B"))
        if not src or not dst:
            continue
        if rtype == "precedence":
            precedences.append({"from": src, "to": dst})
        elif rtype == "guard":
            guards.append({"from": src, "to": dst, "condition": _norm(rel.get("condition"))})
        elif rtype == "label":
            labels.append({"from": src, "to": dst, "text": _norm(rel.get("str"))})

    variables: List[Dict[str, Any]] = []
    for var in final_output.get("variable_definitions", []) or []:
        if not isinstance(var, dict):
            continue
        var_id, name = _norm(var.get("X")), _norm(var.get("name"))
        if not var_id and not name:
            continue
        rng = var.get("R_X")
        variables.append({
            "id": var_id,
            "name": name,
            "range": [str(v) for v in rng] if isinstance(rng, list) else [],
        })

    actors: List[Dict[str, Any]] = []
    for actor in final_output.get("actors", []) or []:
        if not isinstance(actor, dict):
            continue
        assigned = actor.get("assigned_nodes") or []
        actors.append({
            "id": _norm(actor.get("id")),
            "name": _norm(actor.get("name")),
            "assigned_nodes": [str(a) for a in assigned] if isinstance(assigned, list) else [],
        })

    return ParsedModel(nodes, precedences, guards, labels, variables, actors)


def mapping_targets(model: ParsedModel) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []

    for node in model.nodes:
        kind = node["kind"]
        targets.append({
            "ref": f"{kind}:{node['id']}",
            "kind": kind,
            "node_id": node["id"],
            "gateway": node.get("gateway", "") if kind == "branch" else "",
            "content": node.get("content", "") if kind == "action" else "",
            "from": None,
            "to": None,
        })

    for kind, items, content_key in (
        ("label", model.labels, "text"),
        ("guard", model.guards, "condition"),
    ):
        for item in items:
            targets.append({
                "ref": f"{kind}:{item['from']}->{item['to']}",
                "kind": kind,
                "node_id": None,
                "gateway": "",
                "content": item.get(content_key, ""),
                "from": item["from"],
                "to": item["to"],
            })

    return targets


def variable_payload(model: ParsedModel) -> List[Dict[str, Any]]:
    return [
        {
            "ref": f"variable:{v['id'] or v['name']}",
            "id": v["id"],
            "name": v["name"],
            "range": "{" + ", ".join(v["range"]) + "}" if v["range"] else "{}",
            "range_values": v["range"],
        }
        for v in model.variables
    ]


def actor_payload(model: ParsedModel) -> List[Dict[str, Any]]:
    return [
        {
            "ref": f"actor:{a['id'] or a['name']}",
            "id": a["id"],
            "name": a["name"],
            "assigned_nodes": a["assigned_nodes"],
        }
        for a in model.actors
    ]


def node_catalog(model: ParsedModel) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for node in model.nodes:
        entry: Dict[str, Any] = {"id": node["id"], "kind": node["kind"]}
        if node["kind"] == "branch" and node.get("gateway"):
            entry["gateway"] = node["gateway"]
        if node.get("content"):
            entry["content"] = node["content"]
        catalog.append(entry)
    return catalog


def trace_graph(
    model: ParsedModel,
) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    node_ids = [n["id"] for n in model.nodes]
    node_set = set(node_ids)
    kinds = {n["id"]: n["kind"] for n in model.nodes}
    adjacency: Dict[str, List[str]] = {nid: [] for nid in node_ids}

    edges = (
        [(r["from"], r["to"]) for r in model.precedences]
        + [(g["from"], g["to"]) for g in model.guards]
        + [(lab["from"], lab["to"]) for lab in model.labels]
    )
    for src, dst in edges:
        if src in node_set and dst in node_set and dst not in adjacency[src]:
            adjacency[src].append(dst)

    return node_ids, kinds, adjacency
