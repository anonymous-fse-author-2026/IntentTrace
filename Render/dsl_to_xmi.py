from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from IntentTrace.grammar import parse_process
from Render.csv_to_plantuml import _infer_converge_gateways

XMI_NS = "http://www.omg.org/spec/XMI/20131001"
UML_NS = "http://www.eclipse.org/uml2/5.0.0/UML"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

XMI_ID = f"{{{XMI_NS}}}id"
XMI_TYPE = f"{{{XMI_NS}}}type"
XMI_VERSION = f"{{{XMI_NS}}}version"

GATEWAY_SPLIT = {"XOR": "DecisionNode", "OR": "ForkNode", "AND": "ForkNode"}
GATEWAY_JOIN = {"XOR": "MergeNode", "OR": "JoinNode", "AND": "JoinNode"}

_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _norm(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _clean_name(text: Any) -> str:
    s = _norm(text).replace("\n", " ").replace("\r", " ").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


class _Ids:
    def __init__(self) -> None:
        self._used: set[str] = set()

    def make(self, prefix: str, *parts: str) -> str:
        raw = "_".join(_ID_SAFE_RE.sub("_", _norm(p)) for p in parts if _norm(p))
        base = f"_{prefix}_{raw}" if raw else f"_{prefix}"
        candidate = base
        n = 2
        while candidate in self._used:
            candidate = f"{base}_{n}"
            n += 1
        self._used.add(candidate)
        return candidate


def _load_model(text: str) -> Dict[str, Any]:
    stripped = (text or "").lstrip()
    if stripped.startswith("{"):
        data = json.loads(stripped)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object containing 'elements'")
        return data
    return parse_process(text or "").model


def _collect_nodes(model: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    nodes: List[Dict[str, str]] = []
    index: Dict[str, Dict[str, str]] = {}
    for elem in model.get("elements") or []:
        if not isinstance(elem, dict):
            continue
        nid = _norm(elem.get("id"))
        kind = _norm(elem.get("type")).lower()
        if not nid or nid in index or kind not in {"init", "end", "action", "branch", "converge"}:
            continue
        node = {
            "id": nid,
            "kind": kind,
            "content": _clean_name(elem.get("content")),
            "gateway": _norm(elem.get("gateway")).upper(),
        }
        nodes.append(node)
        index[nid] = node
    return nodes, index


def _collect_edges(
    model: Dict[str, Any], index: Dict[str, Dict[str, str]]
) -> List[Dict[str, str]]:
    order: List[Tuple[str, str]] = []
    merged: Dict[Tuple[str, str], Dict[str, str]] = {}

    for rel in model.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        rtype = _norm(rel.get("type")).lower()
        src = _norm(rel.get("A"))
        dst = _norm(rel.get("B"))
        if rtype not in {"precedence", "guard", "label"}:
            continue
        if src not in index or dst not in index:
            continue

        key = (src, dst)
        if key not in merged:
            merged[key] = {"source": src, "target": dst, "guard": "", "name": ""}
            order.append(key)
        entry = merged[key]

        if rtype == "guard" and not entry["guard"]:
            entry["guard"] = _norm(rel.get("condition"))
        elif rtype == "label" and not entry["name"]:
            entry["name"] = _clean_name(rel.get("str"))

    return [merged[k] for k in order]


def _uml_class(node: Dict[str, str], converge_gateways: Dict[str, str]) -> str:
    kind = node["kind"]
    if kind == "init":
        return "InitialNode"
    if kind == "end":
        return "ActivityFinalNode"
    if kind == "action":
        return "OpaqueAction"
    if kind == "branch":
        return GATEWAY_SPLIT.get(node["gateway"] or "XOR", "DecisionNode")
    return GATEWAY_JOIN.get(converge_gateways.get(node["id"], "AND"), "JoinNode")


def model_to_xmi(model: Dict[str, Any], name: str = "Process") -> str:
    nodes, index = _collect_nodes(model)
    if not nodes:
        raise ValueError("no process-model nodes were found")
    edges = _collect_edges(model, index)

    converge_gateways = _infer_converge_gateways(
        model.get("elements") or [], model.get("relationships") or []
    )

    ids = _Ids()
    ET.register_namespace("xmi", XMI_NS)
    ET.register_namespace("uml", UML_NS)
    ET.register_namespace("xsi", XSI_NS)

    model_id = ids.make("model", name)
    root = ET.Element(f"{{{UML_NS}}}Model", {XMI_VERSION: "2.0", XMI_ID: model_id, "name": name})

    activity_id = ids.make("activity", name)
    activity = ET.SubElement(
        root, "packagedElement",
        {XMI_TYPE: "uml:Activity", XMI_ID: activity_id, "name": name},
    )

    node_ids: Dict[str, str] = {}
    for node in nodes:
        uml_class = _uml_class(node, converge_gateways)
        xmi_id = ids.make("node", node["id"])
        node_ids[node["id"]] = xmi_id

        attrs = {XMI_TYPE: f"uml:{uml_class}", XMI_ID: xmi_id}
        if node["kind"] == "action" and node["content"]:
            attrs["name"] = node["content"]
        element = ET.SubElement(activity, "node", attrs)

        if uml_class == "JoinNode" and converge_gateways.get(node["id"], "AND") == "OR":
            ET.SubElement(
                element, "joinSpec",
                {XMI_TYPE: "uml:LiteralString",
                 XMI_ID: ids.make("joinspec", node["id"]),
                 "value": "or"},
            )

    for edge in edges:
        attrs = {
            XMI_TYPE: "uml:ControlFlow",
            XMI_ID: ids.make("edge", edge["source"], edge["target"]),
            "source": node_ids[edge["source"]],
            "target": node_ids[edge["target"]],
        }
        if edge["name"]:
            attrs["name"] = edge["name"]
        element = ET.SubElement(activity, "edge", attrs)

        if edge["guard"]:
            guard = ET.SubElement(
                element, "guard",
                {XMI_TYPE: "uml:OpaqueExpression",
                 XMI_ID: ids.make("guard", edge["source"], edge["target"])},
            )
            ET.SubElement(guard, "body").text = edge["guard"]

    for var in model.get("variable_definitions") or []:
        if not isinstance(var, dict):
            continue
        var_id = _norm(var.get("X"))
        var_name = _clean_name(var.get("name"))
        if not var_id and not var_name:
            continue
        element = ET.SubElement(
            activity, "variable",
            {XMI_TYPE: "uml:Variable",
             XMI_ID: ids.make("var", var_id or var_name),
             "name": var_id or var_name},
        )
        rng = var.get("R_X")
        domain = [str(v) for v in rng] if isinstance(rng, list) else []
        comment = ET.SubElement(
            element, "ownedComment",
            {XMI_TYPE: "uml:Comment", XMI_ID: ids.make("varcomment", var_id or var_name)},
        )
        label = var_name or var_id
        ET.SubElement(comment, "body").text = (
            f"{label} in {{" + ", ".join(domain) + "}"
        )

    for actor in model.get("actors") or []:
        if not isinstance(actor, dict):
            continue
        actor_id = _norm(actor.get("id"))
        actor_name = _clean_name(actor.get("name"))
        if not actor_id and not actor_name:
            continue
        assigned = actor.get("assigned_nodes")
        refs = [
            node_ids[_norm(a)] for a in assigned
            if isinstance(assigned, list) and _norm(a) in node_ids
        ]
        attrs = {
            XMI_TYPE: "uml:ActivityPartition",
            XMI_ID: ids.make("partition", actor_id or actor_name),
            "name": actor_name or actor_id,
        }
        if refs:
            attrs["node"] = " ".join(refs)
        ET.SubElement(activity, "group", attrs)

    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def dsl_to_xmi(text: str, name: str = "Process") -> str:
    return model_to_xmi(_load_model(text), name=name)


def _convert_file(in_path: Path, out_path: Path) -> None:
    xmi = dsl_to_xmi(in_path.read_text(encoding="utf-8"), name=in_path.stem or "Process")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xmi, encoding="utf-8")
    print(f"XMI written: {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert a process model (DSL or model JSON) to UML 2.5 XMI."
    )
    ap.add_argument("--input", "-i", type=Path, required=True,
                    help="Input DSL/JSON file, or a directory of such files.")
    ap.add_argument("--output", "-o", type=Path, required=True,
                    help="Output .xmi file, or directory when --input is a directory.")
    args = ap.parse_args(argv)

    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        for f in sorted(args.input.iterdir()):
            if not f.is_file() or f.suffix.lower() not in (".txt", ".json"):
                continue
            try:
                _convert_file(f, args.output / (f.stem + ".xmi"))
            except (ValueError, json.JSONDecodeError) as exc:
                print(f"skipped {f}: {exc}", file=sys.stderr)
    else:
        _convert_file(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
