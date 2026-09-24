import argparse
import csv
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_ACTOR_ID = "a_default"
DEFAULT_ACTOR_NAME = "Default"


def parse_csv(csv_text: str):
    data_lines = [
        ln for ln in csv_text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    reader = csv.DictReader(data_lines)

    nodes: Dict[str, Dict[str, str]] = {}
    edges: List[Tuple[str, str, str]] = []

    for row in reader:
        nid = (row.get("id") or "").strip()
        if not nid:
            continue
        ntype = (row.get("type") or "").strip().lower()
        name = (row.get("name") or "").strip()

        if nid not in nodes:
            nodes[nid] = {"name": name, "type": ntype}

        label = (row.get("transition_label") or "").strip()
        parents = [p.strip() for p in (row.get("parent") or "").split(",") if p.strip()]
        for p in parents:
            edges.append((p, nid, label))

    seen = set()
    deduped: List[Tuple[str, str, str]] = []
    for e in edges:
        if e not in seen:
            seen.add(e)
            deduped.append(e)
    return nodes, deduped


_ID_SAFE = re.compile(r"[^A-Za-z0-9_]")


class IdAllocator:
    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._map: Dict[str, str] = {}

    def new(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{prefix}{self._counters[prefix]}"

    def for_original(self, original_id: str, prefix: str) -> str:
        if original_id not in self._map:
            self._map[original_id] = self.new(prefix)
        return self._map[original_id]


def q(value: str) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\n", " ").replace("\r", " ").strip()

    s = s.replace('"', "'")
    return f'"{s}"'


class Converter:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

        self.children: Dict[str, List[str]] = defaultdict(list)
        self.parents: Dict[str, List[str]] = defaultdict(list)

        self.edge_label: Dict[Tuple[str, str], str] = {}
        for p, c, lbl in edges:
            if c not in self.children[p]:
                self.children[p].append(c)
            if p not in self.parents[c]:
                self.parents[c].append(p)
            if lbl:
                self.edge_label[(p, c)] = lbl

        self.ids = IdAllocator()

        self.node_decls: List[str] = []
        self.var_decls: List[str] = []
        self.precedences: List[Tuple[str, str]] = []
        self.guards: List[Tuple[str, str, str]] = []
        self.labels: List[Tuple[str, str, str]] = []

        self.all_node_ids: List[str] = []

        self.id_map: Dict[str, dict] = {}

        self.entry: Dict[str, str] = {}
        self.exit: Dict[str, str] = {}

    def _add_node(self, decl: str, node_id: str):
        self.node_decls.append(decl)
        self.all_node_ids.append(node_id)

    def _precede(self, a: str, b: str):
        self.precedences.append((a, b))

    def _ref_only_id(self, original_id: str) -> str:
        if not hasattr(self, "_dangling"):
            self._dangling: Dict[str, str] = {}
        if original_id not in self._dangling:
            ref_id = self.ids.for_original(original_id, "x")
            self._dangling[original_id] = ref_id
            self.id_map[ref_id] = {
                "original_id": original_id, "type": "undeclared",
                "name": "", "role": "dangling_reference",
            }
        return self._dangling[original_id]

    def _resolve_exit(self, original_id: str) -> str:
        if original_id in self.exit:
            return self.exit[original_id]
        return self._ref_only_id(original_id)

    def _resolve_entry(self, original_id: str) -> str:
        if original_id in self.entry:
            return self.entry[original_id]
        return self._ref_only_id(original_id)

    def convert(self) -> str:
        self._expand_nodes()
        self._wire_edges()
        return self._render()

    def _expand_nodes(self):
        for nid, node in self.nodes.items():
            ntype = node["type"]
            name = node["name"]

            if ntype == "start":
                init_id = self.ids.for_original(nid, "i")
                self._add_node(f"init({init_id})", init_id)

                act_id = self.ids.new("n")
                self._add_node(f"action({act_id}, {q(name or 'start')})", act_id)
                self._precede(init_id, act_id)
                self.entry[nid] = init_id
                self.exit[nid] = act_id
                self.id_map[init_id] = {
                    "original_id": nid, "type": "init", "name": name,
                    "role": "start",
                }
                self.id_map[act_id] = {
                    "original_id": nid, "type": "action",
                    "name": name or "start", "role": "start_action",
                }

            elif ntype == "end":

                act_id = self.ids.new("n")
                self._add_node(f"action({act_id}, {q(name or 'end')})", act_id)
                end_id = self.ids.for_original(nid, "e")
                self._add_node(f"end({end_id})", end_id)
                self._precede(act_id, end_id)
                self.entry[nid] = act_id
                self.exit[nid] = end_id
                self.id_map[act_id] = {
                    "original_id": nid, "type": "action",
                    "name": name or "end", "role": "end_action",
                }
                self.id_map[end_id] = {
                    "original_id": nid, "type": "end", "name": name,
                    "role": "end",
                }

            elif ntype == "condition":

                decision_text = name.strip()
                has_text = bool(decision_text) and decision_text.lower() != "condition"

                branch_id = self.ids.for_original(nid, "b")

                if has_text:
                    act_id = self.ids.new("n")
                    self._add_node(f"action({act_id}, {q(decision_text)})", act_id)
                    self._add_node(f"branch({branch_id}, XOR)", branch_id)
                    self._precede(act_id, branch_id)
                    self.id_map[act_id] = {
                        "original_id": nid, "type": "action",
                        "name": decision_text, "role": "decision_action",
                    }
                    self.entry[nid] = act_id
                else:

                    self._add_node(f"branch({branch_id}, XOR)", branch_id)
                    self.entry[nid] = branch_id

                guard_labels = []
                for c in self.children.get(nid, []):
                    lbl = self.edge_label.get((nid, c), "")
                    if lbl and lbl not in guard_labels:
                        guard_labels.append(lbl)
                var_id = self.ids.new("v")
                rng = "{" + ", ".join(q(g) for g in guard_labels) + "}" if guard_labels else "{}"
                var_desc = decision_text if has_text else (name or "decision")
                self.var_decls.append(
                    f"defines({var_id}, {q(var_desc)}, {rng})"
                )

                self._cond_var = getattr(self, "_cond_var", {})
                self._cond_var[nid] = var_id
                self.id_map[var_id] = {
                    "original_id": nid, "type": "variable",
                    "name": name, "role": "decision_variable",
                }
                self.id_map[branch_id] = {
                    "original_id": nid, "type": "branch", "gateway": "XOR",
                    "name": name, "role": "decision_branch",
                }

                self.exit[nid] = branch_id

            elif ntype == "external_artifact":

                act_id = self.ids.for_original(nid, "EA")
                self._add_node(f"action({act_id}, {q(name)})", act_id)
                self.entry[nid] = act_id
                self.exit[nid] = act_id
                self.id_map[act_id] = {
                    "original_id": nid, "type": "action", "name": name,
                    "role": "external_artifact",
                }

            else:
                act_id = self.ids.for_original(nid, "n")
                self._add_node(f"action({act_id}, {q(name)})", act_id)
                self.entry[nid] = act_id
                self.exit[nid] = act_id
                self.id_map[act_id] = {
                    "original_id": nid, "type": "action", "name": name,
                    "role": "entity",
                }

    def _wire_edges(self):

        back_edges = self._find_back_edges()

        converge_for: Dict[str, str] = {}
        for nid, node in self.nodes.items():
            forward_preds = [
                p for p in self.parents.get(nid, [])
                if (p, nid) not in back_edges
            ]
            if len(forward_preds) > 1:
                conv_id = self.ids.new("c")
                self._add_node(f"converge({conv_id})", conv_id)
                self._precede(conv_id, self.entry[nid])
                converge_for[nid] = conv_id
                self.id_map[conv_id] = {
                    "original_id": nid, "type": "converge",
                    "name": node.get("name", ""), "role": "merge_before_node",
                }

        def is_artifact(node_id: str) -> bool:
            return self.nodes.get(node_id, {}).get("type") == "external_artifact"

        and_branch_for: Dict[str, str] = {}
        for nid, node in self.nodes.items():
            if node["type"] == "condition":
                continue
            real_succs = [c for c in self.children.get(nid, []) if not is_artifact(c)]
            if len(real_succs) > 1:
                and_id = self.ids.new("b")
                self._add_node(f"branch({and_id}, AND)", and_id)
                self._precede(self.exit[nid], and_id)
                and_branch_for[nid] = and_id
                self.id_map[and_id] = {
                    "original_id": nid, "type": "branch", "gateway": "AND",
                    "name": node.get("name", ""), "role": "parallel_split",
                }

        for parent, child, lbl in self._unique_edges():
            parent_node = self.nodes.get(parent, {})

            if parent in and_branch_for and not is_artifact(child):
                src = and_branch_for[parent]
            else:
                src = self._resolve_exit(parent)

            if child in converge_for and (parent, child) not in back_edges:
                dst = converge_for[child]
            else:
                dst = self._resolve_entry(child)

            self._precede(src, dst)

            if parent_node.get("type") == "condition" and lbl:

                var_id = getattr(self, "_cond_var", {}).get(parent)
                if var_id:
                    phi = f"{var_id} = '{lbl}'"
                    self.guards.append((src, dst, phi))
                else:
                    self.labels.append((src, dst, lbl))
            elif lbl:

                self.labels.append((src, dst, lbl))

    def _unique_edges(self):
        seen = set()
        out = []
        for p, c, lbl in self.edges:
            key = (p, c)
            if key in seen:
                continue
            seen.add(key)
            out.append((p, c, self.edge_label.get((p, c), "")))
        return out

    def _find_back_edges(self) -> set:

        adj: Dict[str, List[str]] = defaultdict(list)
        for p, c, _ in self._unique_edges():
            adj[p].append(c)

        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n: WHITE for n in self.nodes}
        back_edges: set = set()

        start_nodes = [n for n, d in self.nodes.items() if d["type"] == "start"]
        roots = start_nodes + [n for n in self.nodes if n not in start_nodes]

        for root in roots:
            if color.get(root, BLACK) != WHITE:
                continue

            stack: List[Tuple[str, int]] = [(root, 0)]
            color[root] = GRAY
            while stack:
                node, idx = stack[-1]
                neighbors = adj.get(node, [])
                if idx < len(neighbors):
                    stack[-1] = (node, idx + 1)
                    nxt = neighbors[idx]
                    c = color.get(nxt, BLACK)
                    if c == WHITE:
                        color[nxt] = GRAY
                        stack.append((nxt, 0))
                    elif c == GRAY:

                        back_edges.add((node, nxt))

                else:
                    color[node] = BLACK
                    stack.pop()
        return back_edges

    def _render(self) -> str:
        lines: List[str] = []

        lines.extend(self.node_decls)
        lines.extend(self.var_decls)

        actor_nodes = ", ".join(self.all_node_ids)
        lines.append(
            f"actor({DEFAULT_ACTOR_ID}, {q(DEFAULT_ACTOR_NAME)}, [{actor_nodes}])"
        )

        lines.append("")

        for a, b in self.precedences:
            lines.append(f"precedence({a}, {b})")
        for a, b, phi in self.guards:
            lines.append(f"guard({a}, {b}, {phi})")
        for a, b, s in self.labels:
            lines.append(f"label({a}, {b}, {q(s)})")

        return "\n".join(lines) + "\n"


def csv_to_dsl(csv_text: str):
    nodes, edges = parse_csv(csv_text)
    conv = Converter(nodes, edges)
    dsl = conv.convert()
    return dsl, conv.id_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FlowChart CSV to the LADEX process-model DSL."
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Path to a single input CSV (.txt) file or a directory.")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output DSL file (for a single input) or output "
                             "directory (when --input is a directory).")
    return parser.parse_args()


def convert_file(in_path: Path, out_path: Path):
    csv_text = in_path.read_text(encoding="utf-8")
    dsl, id_map = csv_to_dsl(csv_text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dsl, encoding="utf-8")

    map_path = out_path.with_suffix(out_path.suffix + ".map.json")
    map_path.write_text(json.dumps(id_map, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        for f in sorted(args.input.glob("*.txt")):
            convert_file(f, args.output / f.name)
            print(f"DSL written: {args.output / f.name}")
    else:
        convert_file(args.input, args.output)
        print(f"DSL written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
