from __future__ import annotations

import csv
import io
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from IntentTrace.grammar import parse_process


_GUARD_VALUE_RE = re.compile(r"""=\s*['"]?(?P<val>.*?)['"]?\s*$""")


def _guard_label(condition: str) -> str:
    if not condition:
        return ""
    m = _GUARD_VALUE_RE.search(condition.strip())
    if m:
        return m.group("val").strip()
    return condition.strip()


def _clean_name(text: str) -> str:
    s = (text or "").replace("\n", " ").replace("\r", " ").strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        s = s[1:-1].strip()
    return s


class _DslGraph:

    def __init__(self, model: Dict):

        self.nodes: Dict[str, Dict[str, str]] = {}
        for elem in model.get("elements", []):
            nid = str(elem.get("id", "")).strip()
            if not nid:
                continue
            self.nodes[nid] = {
                "kind": str(elem.get("type", "")).strip().lower(),
                "content": _clean_name(elem.get("content", "")),
                "gateway": str(elem.get("gateway", "")).strip().upper(),
            }

        self.succ: Dict[str, List[str]] = defaultdict(list)
        self.pred: Dict[str, List[str]] = defaultdict(list)
        self.edge_label: Dict[Tuple[str, str], str] = {}

        for rel in model.get("relationships", []):
            rtype = str(rel.get("type", "")).strip().lower()
            a = str(rel.get("A", "")).strip()
            b = str(rel.get("B", "")).strip()
            if rtype == "precedence":
                if a and b:
                    self._add_edge(a, b)
            elif rtype == "guard":
                if a and b:
                    self._add_edge(a, b)
                    self.edge_label[(a, b)] = _guard_label(rel.get("condition", ""))
            elif rtype == "label":
                if a and b:
                    self._add_edge(a, b)
                    lbl = _clean_name(rel.get("str", ""))
                    if lbl:
                        self.edge_label[(a, b)] = lbl

    def _add_edge(self, a: str, b: str) -> None:
        if b not in self.succ[a]:
            self.succ[a].append(b)
        if a not in self.pred[b]:
            self.pred[b].append(a)

    def kind(self, nid: str) -> str:
        return self.nodes.get(nid, {}).get("kind", "")


def _semantic_type(dg: _DslGraph, nid: str) -> str:
    node = dg.nodes[nid]
    kind = node["kind"]

    if kind == "action":

        for s in dg.succ.get(nid, []):
            if dg.kind(s) == "branch" and dg.nodes[s]["gateway"] == "XOR":
                return "condition"
        return "entity"
    if kind == "branch":

        return "condition"
    return "entity"


def dsl_to_csv(dsl_text: str) -> str:
    report = parse_process(dsl_text or "")
    dg = _DslGraph(report.model)

    csv_type: Dict[str, str] = {}
    csv_name: Dict[str, str] = {}

    entry: Dict[str, Optional[str]] = {}
    exit_: Dict[str, Optional[str]] = {}

    branch_owner: Dict[str, str] = {}
    action_branch: Dict[str, str] = {}
    for nid, node in dg.nodes.items():
        if node["kind"] == "branch" and node["gateway"] == "XOR":
            preds = dg.pred.get(nid, [])
            act_preds = [p for p in preds if dg.kind(p) == "action"]

            if len(act_preds) == 1 and len(dg.succ.get(act_preds[0], [])) == 1:
                owner = act_preds[0]
                branch_owner[nid] = owner
                action_branch[owner] = nid

    for nid, node in dg.nodes.items():
        kind = node["kind"]

        if kind == "init":

            entry[nid] = None
            exit_[nid] = None
        elif kind == "converge":
            entry[nid] = None
            exit_[nid] = None
        elif kind == "branch" and node["gateway"] != "XOR":

            entry[nid] = None
            exit_[nid] = None
        elif kind == "branch" and node["gateway"] == "XOR":
            if nid in branch_owner:

                owner = branch_owner[nid]
                entry[nid] = owner
                exit_[nid] = owner
            else:

                csv_type[nid] = "condition"
                csv_name[nid] = node["content"] or "condition"
                entry[nid] = nid
                exit_[nid] = nid
        elif kind == "action":
            if nid in action_branch:

                csv_type[nid] = "condition"
                csv_name[nid] = node["content"]
                entry[nid] = nid
                exit_[nid] = nid
            else:
                csv_type[nid] = _semantic_type(dg, nid)
                csv_name[nid] = node["content"]
                entry[nid] = nid
                exit_[nid] = nid
        elif kind == "end":
            csv_type[nid] = "end"
            csv_name[nid] = node["content"] or "end"
            entry[nid] = nid
            exit_[nid] = nid
        else:

            csv_type[nid] = "entity"
            csv_name[nid] = node.get("content", "")
            entry[nid] = nid
            exit_[nid] = nid

    for nid, node in dg.nodes.items():
        if node["kind"] == "init":
            for s in dg.succ.get(nid, []):
                real = _resolve_exit(dg, s, exit_, branch_owner)
                if real in csv_type and csv_type[real] != "condition":
                    csv_type[real] = "start"

    dropped_ends: set = set()
    for nid, node in dg.nodes.items():
        if node["kind"] == "end":
            preds = [p for p in dg.pred.get(nid, [])]
            real_preds = [
                _resolve_exit(dg, p, exit_, branch_owner) for p in preds
            ]
            real_preds = [p for p in real_preds if p in csv_type]

            if len(real_preds) == 1 and csv_type[real_preds[0]] in ("entity", "action"):
                csv_type[real_preds[0]] = "end"
                csv_name[real_preds[0]] = csv_name[real_preds[0]] or node["content"] or "end"
                dropped_ends.add(nid)

                entry[nid] = real_preds[0]
                exit_[nid] = real_preds[0]
                csv_type.pop(nid, None)
                csv_name.pop(nid, None)

    parents: Dict[str, List[str]] = defaultdict(list)
    trans_label: Dict[Tuple[str, str], str] = {}

    for a in list(dg.nodes.keys()):
        for b in dg.succ.get(a, []):
            src = _resolve_exit(dg, a, exit_, branch_owner)

            lbl = dg.edge_label.get((a, b), "")
            _emit_edges(dg, src, b, lbl, csv_type, entry, exit_, branch_owner,
                        parents, trans_label, set())

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["id", "name", "type", "parent", "transition_label"])

    for nid in csv_type:
        name = csv_name.get(nid, "")
        ntype = csv_type[nid]
        node_parents = parents.get(nid, [])
        if not node_parents:
            writer.writerow([nid, name, ntype, "", ""])
        else:
            for p in node_parents:
                lbl = trans_label.get((p, nid), "")
                writer.writerow([nid, name, ntype, p, lbl])

    return out.getvalue()


def _resolve_exit(dg: _DslGraph, nid: str, exit_: Dict[str, Optional[str]],
                  branch_owner: Dict[str, str]) -> Optional[str]:
    ex = exit_.get(nid, nid)
    if ex is not None:
        return ex

    seen = set()
    stack = list(dg.pred.get(nid, []))
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        pex = exit_.get(p, p)
        if pex is not None:
            return pex
        stack.extend(dg.pred.get(p, []))
    return None


def _emit_edges(dg: _DslGraph, src: Optional[str], b: str, lbl: str,
                csv_type: Dict[str, str], entry: Dict[str, Optional[str]],
                exit_: Dict[str, Optional[str]], branch_owner: Dict[str, str],
                parents: Dict[str, List[str]],
                trans_label: Dict[Tuple[str, str], str],
                visited_transparent: set) -> None:
    if src is None:
        return

    b_entry = entry.get(b, b)

    if b_entry is None:

        if b in visited_transparent:
            return
        visited_transparent = visited_transparent | {b}
        for c in dg.succ.get(b, []):
            next_lbl = dg.edge_label.get((b, c), "") or lbl
            _emit_edges(dg, src, c, next_lbl, csv_type, entry, exit_,
                        branch_owner, parents, trans_label, visited_transparent)
        return

    if b_entry == src:

        return

    if b_entry not in csv_type:

        return

    if src not in parents[b_entry]:
        parents[b_entry].append(src)
    if lbl and (src, b_entry) not in trans_label:
        trans_label[(src, b_entry)] = lbl


def _main(argv: List[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Convert LADEX DSL to FlowChart CSV.")
    ap.add_argument("--input", "-i", type=Path, required=True,
                    help="Input DSL (.txt) file or directory of DSL files.")
    ap.add_argument("--output", "-o", type=Path, required=True,
                    help="Output CSV file, or directory when --input is a dir.")
    args = ap.parse_args(argv)

    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        for f in sorted(args.input.glob("*.txt")):
            csv_text = dsl_to_csv(f.read_text(encoding="utf-8"))
            (args.output / f.name).write_text(csv_text, encoding="utf-8")
            print(f"CSV written: {args.output / f.name}")
    else:
        csv_text = dsl_to_csv(args.input.read_text(encoding="utf-8"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(csv_text, encoding="utf-8")
        print(f"CSV written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
