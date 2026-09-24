import csv
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

FLOW_LAYOUT_VERTICAL = "verticalflow"
FLOW_LAYOUT_HORIZONTAL = "horizontalflow"
FLOW_LAYOUT_CHOICES = {FLOW_LAYOUT_VERTICAL, FLOW_LAYOUT_HORIZONTAL}

FLOWCHART_HEADER_TEMPLATE = """## FlowChart Generator
# label: %name%
# style: whiteSpace=wrap;html=1;rounded=1;fillColor=#ffffff;strokeColor=#000000;
# namespace: csvimport-
# connect: {\"from\":\"parent\", \"to\":\"id\", \"fromlabel\":\"transition_label\", \"invert\":true, \"style\":\"endArrow=blockThin;endFill=1;fontSize=11;curved=1;\"}
# labels: {\"label1\" : \"%transition_label%\"}
# ignore: transition_label,parent
# styles: {\"init\":\"ellipse;fillColor=strokeColor;html=1;verticalAlign=middle;fontColor=#FFFFFF;\", \"end\":\"ellipse;html=1;shape=endState;fillColor=strokeColor;verticalAlign=middle;fontColor=#FFFFFF;\", \"action\":\"rounded=1;whiteSpace=wrap;html=1;absoluteArcSize=1;arcSize=14;strokeWidth=2;align=center;verticalAlign=middle;\", \"branchAND\":\"html=1;points=[];perimeter=orthogonalPerimeter;fillColor=strokeColor;verticalAlign=middle;fixedSize=1;height=5;width=100;rotation=0;\", \"convergeAND\":\"html=1;points=[];perimeter=orthogonalPerimeter;fillColor=strokeColor;verticalAlign=middle;fixedSize=1;height=5;width=100;\", \"branchXOR\":\"rhombus;verticalAlign=middle;spacingRight=35;spacingLeft=35;\", \"branchOR\":\"rhombus;verticalAlign=middle;spacingRight=35;spacingLeft=35;\", \"convergeXOR\":\"rhombus;verticalAlign=middle;spacingRight=35;spacingLeft=35;\", \"convergeOR\":\"rhombus;verticalAlign=middle;spacingRight=35;spacingLeft=35;\"}
# stylename: type
# width: @w
# height: @h
# padding: 0
# nodespacing: 40
# levelspacing: 100
# edgespacing: 40
# layout: __FLOW_LAYOUT__
# identity: id
id,name,type,parent,transition_label,w,h
"""


def _quote(value: str) -> str:
    escaped = str(value).replace('"', '""')
    return f'"{escaped}"'


def _edge_key(a: str, b: str) -> Tuple[str, str]:
    return a.strip(), b.strip()


def _collect_edges(relationships: List[Dict[str, Any]]) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], str], List[Tuple[str, str]]]:
    parents_by_child: Dict[str, List[str]] = {}
    transition_by_edge: Dict[Tuple[str, str], str] = {}
    precedence_edges: List[Tuple[str, str]] = []

    for rel in relationships:
        if rel.get("type") != "precedence":
            continue
        a = str(rel.get("A", "")).strip()
        b = str(rel.get("B", "")).strip()
        if not a or not b:
            continue
        parents_by_child.setdefault(b, []).append(a)
        precedence_edges.append((a, b))
        transition_by_edge.setdefault((a, b), "")

    for rel in relationships:
        rel_type = rel.get("type")
        if rel_type not in {"guard", "label"}:
            continue

        a = str(rel.get("A", "")).strip()
        b = str(rel.get("B", "")).strip()
        if not a or not b:
            continue

        key = _edge_key(a, b)
        existing = transition_by_edge.get(key, "").strip()

        if rel_type == "guard":
            condition = str(rel.get("condition", "")).strip()
            guard_text = f"[{condition}]" if condition else ""
            if not guard_text:
                continue
            transition_by_edge[key] = f"{existing} {guard_text}".strip() if existing else guard_text
        else:
            label_text = str(rel.get("str", "")).strip()
            if not label_text:
                continue
            transition_by_edge[key] = f"{existing} {label_text}".strip() if existing else label_text

    return parents_by_child, transition_by_edge, precedence_edges


def _node_name(node: Dict[str, Any]) -> str:
    node_type = str(node.get("type", "")).strip().lower()
    if node_type != "action":
        return ""

    content = node.get("content", "")
    if content is None:
        return ""
    return str(content).strip()


def _synthetic_type(node: Dict[str, Any], converge_gateway_map: Optional[Dict[str, str]] = None) -> str:
    raw_type = str(node.get("type", "")).strip().lower()
    if raw_type == "branch":
        gateway = str(node.get("gateway", "")).strip().upper()
        if gateway in {"AND", "OR", "XOR"}:
            return f"branch{gateway}"
        return raw_type
    if raw_type == "converge":
        node_id = str(node.get("id", "")).strip()
        if converge_gateway_map and node_id in converge_gateway_map:
            inferred = converge_gateway_map[node_id]
            if inferred in {"AND", "OR", "XOR"}:
                return f"converge{inferred}"

        return "convergeAND"
    return raw_type


def _infer_converge_gateways(
    elements: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> Dict[str, str]:
    node_type: Dict[str, str] = {}
    node_gateway: Dict[str, str] = {}
    converge_ids: List[str] = []
    for e in elements:
        nid = str(e.get("id", "")).strip()
        if not nid:
            continue
        t = str(e.get("type", "")).strip().lower()
        node_type[nid] = t
        if t == "branch":
            gw = str(e.get("gateway", "")).strip().upper()
            if gw in {"AND", "OR", "XOR"}:
                node_gateway[nid] = gw
        if t == "converge":
            converge_ids.append(nid)

    in_edges: Dict[str, List[str]] = defaultdict(list)
    for rel in relationships:
        if rel.get("type") != "precedence":
            continue
        src = str(rel.get("A", "")).strip()
        dst = str(rel.get("B", "")).strip()
        if not src or not dst:
            continue
        in_edges[dst].append(src)

    inferred: Dict[str, str] = {}
    for cid in converge_ids:

        seen = {cid}
        queue: List[str] = list(in_edges.get(cid, []))
        while queue:
            n = queue.pop(0)
            if n in seen:
                continue
            seen.add(n)
            if node_type.get(n) == "branch" and n in node_gateway:
                inferred[cid] = node_gateway[n]
                break
            for parent in in_edges.get(n, []):
                if parent not in seen:
                    queue.append(parent)
    return inferred


def _normalize_flow_layout(flow_layout: str) -> str:
    layout = str(flow_layout).strip().lower()
    if layout not in FLOW_LAYOUT_CHOICES:
        raise ValueError(
            f"Invalid flow layout: {flow_layout}. Choose one of: {FLOW_LAYOUT_VERTICAL}, {FLOW_LAYOUT_HORIZONTAL}."
        )
    return layout


def _build_header(flow_layout: str) -> str:
    return FLOWCHART_HEADER_TEMPLATE.replace("__FLOW_LAYOUT__", flow_layout)


def _node_dimensions(node_type: str, flow_layout: str) -> Tuple[str, str]:
    if node_type in {"branchAND", "convergeAND"}:
        if flow_layout == FLOW_LAYOUT_HORIZONTAL:
            return "5", "auto"
        return "auto", "5"
    if node_type in {"init", "end"}:
        return "30", "30"
    if node_type in {"branchXOR", "branchOR", "convergeXOR", "convergeOR"}:
        return "40", "40"
    return "auto", "auto"


def _size_cell(value: str) -> str:

    raw = str(value).strip()
    if raw.isdigit():
        return raw
    return _quote(raw)


def csv_to_plantuml_final(csv_text: str) -> str:
    lines = [line for line in csv_text.strip().split("\n") if line.strip() and not line.startswith("#")]
    reader = csv.DictReader(lines)

    nodes: Dict[str, Dict[str, str]] = {}
    edges: List[Dict[str, str]] = []

    for row in reader:
        nid = row["id"]
        if nid not in nodes:
            nodes[nid] = row

        parents = [p.strip() for p in row.get("parent", "").split(",") if p.strip()]
        label = row.get("transition_label", "").strip().strip('"')

        for p in parents:
            edges.append({"source": p, "target": nid, "label": label})

    children_of: Dict[str, List[str]] = defaultdict(list)
    edge_labels: Dict[Tuple[str, str], str] = {}

    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        children_of[src].append(tgt)
        if edge["label"]:
            edge_labels[(src, tgt)] = edge["label"]

    def edge_label(src: str, tgt: str) -> str:
        return edge_labels.get((src, tgt), "")

    back_edges: set = set()
    init_for_loops = next(
        (nid for nid, n in nodes.items() if n["type"].lower() == "init"),
        None,
    )
    if init_for_loops is not None:
        on_stack: set = set()
        done: set = set()

        def _classify(node_id: str) -> None:
            on_stack.add(node_id)
            for child in children_of.get(node_id, []):
                if child in on_stack:
                    back_edges.add((node_id, child))
                elif child not in done:
                    _classify(child)
            on_stack.discard(node_id)
            done.add(node_id)

        _classify(init_for_loops)

    loop_tails_of: Dict[str, List[str]] = defaultdict(list)
    for tail, head in back_edges:
        loop_tails_of[head].append(tail)

    def forward_children(src: str) -> List[str]:
        return [c for c in children_of.get(src, []) if (src, c) not in back_edges]

    actors_meta = _parse_actor_metadata(csv_text)
    node_to_actor = _node_to_actor_map(actors_meta)

    variables_meta = _parse_variable_metadata(csv_text)
    var_id_to_name: Dict[str, str] = {
        v["id"]: v.get("name", "") or "" for v in variables_meta if v.get("id")
    }
    var_id_to_range: Dict[str, str] = {
        v["id"]: v.get("range", "") or "" for v in variables_meta if v.get("id")
    }
    known_var_ids = set(var_id_to_name.keys())

    output = ["@startuml", "!theme plain"]

    declared_lane_ids: List[str] = []
    for actor in actors_meta:
        aid = str(actor.get("id") or "").strip()
        if not aid or aid in declared_lane_ids:
            continue
        name = str(actor.get("name") or "").strip()
        declared_lane_ids.append(aid)
        if name:
            output.append(f"|{aid}| {name}")
        else:
            output.append(f"|{aid}|")

    init_node_id = next(
        (nid for nid, n in nodes.items() if n["type"].lower() == "init"),
        None,
    )
    initial_lane: Optional[str] = None
    if init_node_id is not None:
        initial_lane = node_to_actor.get(init_node_id)
    if not initial_lane and declared_lane_ids:
        initial_lane = declared_lane_ids[0]
    if initial_lane:
        output.append(f"|{initial_lane}|")
    output.append("start")

    visited = set()
    current_lane: List[Optional[str]] = [initial_lane]

    def switch_lane(node_id: str) -> None:
        lane = node_to_actor.get(node_id)
        if not lane:
            return
        if lane == current_lane[0]:
            return
        output.append(f"|{lane}|")
        current_lane[0] = lane

    def emit_branch_object_signal(branch_node_id: str) -> None:
        if not known_var_ids:
            return

        refs: List[str] = []
        seen: set = set()
        for child in forward_children(branch_node_id):
            for vid in _extract_variable_ids(edge_label(branch_node_id, child), known_var_ids):
                if vid not in seen:
                    refs.append(vid)
                    seen.add(vid)
        if not refs:
            return
        bindings = []
        for vid in refs:
            name = var_id_to_name.get(vid, "").strip()
            range_text = var_id_to_range.get(vid, "").strip()
            label = f"{vid}: {name}" if name else vid
            if range_text:
                label = f"{label} ({range_text})"
            bindings.append(label)

        body = "\n".join(bindings)
        output.append(f":{body}; <<objectSignal>>")

    def get_merge_node(start_node_id: str):
        children = forward_children(start_node_id)
        if len(children) < 2:
            return None

        reachable_sets = []
        for child in children:
            r = set()
            q = [child]
            while q:
                c = q.pop(0)
                if c not in r:
                    r.add(c)
                    for cc in forward_children(c):
                        if cc not in r:
                            q.append(cc)
            reachable_sets.append(r)

        intersection = reachable_sets[0]
        for r in reachable_sets[1:]:
            intersection = intersection.intersection(r)

        if not intersection:
            return None

        q = list(children)
        v = set(children)
        while q:
            c = q.pop(0)
            if c in intersection:
                return c
            for cc in forward_children(c):
                if cc not in v:
                    v.add(cc)
                    q.append(cc)
        return None

    _in_loop_render: Dict[str, bool] = {}

    def _find_loop_branch(head_id: str):

        loops_back: set = set()

        for nid in nodes:
            seen: set = set()
            stack = [nid]
            while stack:
                c = stack.pop()
                if c in seen:
                    continue
                seen.add(c)
                for nxt in forward_children(c):
                    if nxt == head_id:
                        loops_back.add(nid)
                    if nxt not in seen:
                        stack.append(nxt)

        node_b = head_id
        guard = 0
        while guard < len(nodes) + 1:
            guard += 1
            kids = forward_children(node_b)
            ntype = nodes[node_b]["type"]
            if ntype in ("branchXOR", "branchOR") and len(kids) >= 2:
                loop_child = next((k for k in kids if k in loops_back or k in loop_tails_of.get(head_id, [])), None)
                exit_child = next((k for k in kids if k != loop_child), None)
                if loop_child is not None and exit_child is not None:
                    return node_b, loop_child, exit_child
                return None
            if len(kids) == 1:
                node_b = kids[0]
                continue
            return None
        return None

    def _render_loop(head_id: str, stop_node: str = None, incoming_label: str = None) -> None:
        info = _find_loop_branch(head_id)
        if info is None:

            _in_loop_render[head_id] = True
            process_node(head_id, stop_node=stop_node, incoming_label=incoming_label)
            return

        branch_id, loop_child, exit_child = info

        if incoming_label:
            output.append(f"-> {incoming_label};")

        _in_loop_render[head_id] = True
        node_b = head_id
        first_body = True
        while node_b != branch_id:
            switch_lane(node_b)
            visited.add(node_b)
            nm = nodes[node_b]["name"].strip('"')
            if first_body:
                output.append(f"repeat :{nm};")
                first_body = False
            else:
                output.append(f":{nm};")
            kids = forward_children(node_b)
            node_b = kids[0]

        if first_body:
            output.append("repeat")
        visited.add(branch_id)
        emit_branch_object_signal(branch_id)

        loop_guard = edge_label(branch_id, loop_child).strip()
        exit_guard = edge_label(branch_id, exit_child).strip()

        backward_actions: List[str] = []
        c = loop_child
        chain_guard = 0
        while c is not None and c != head_id and chain_guard < len(nodes) + 1:
            chain_guard += 1
            visited.add(c)
            nm = nodes[c]["name"].strip('"')
            if nm:
                backward_actions.append(nm)
            nxt = forward_children(c)
            c = nxt[0] if len(nxt) == 1 else None
        if backward_actions:
            output.append(f"backward:{'; '.join(backward_actions)};")

        cond = f"() is ({loop_guard})" if loop_guard else "()"
        tail = f" not ({exit_guard})" if exit_guard else ""
        output.append(f"repeat while {cond}{tail}")

        process_node(exit_child, stop_node=stop_node)

    def _branch_reaches(start: str, target: str) -> bool:
        if target is None:
            return False
        seen: set = set()
        stack = [start]
        while stack:
            c = stack.pop()
            if c == target:
                return True
            if c in seen:
                continue
            seen.add(c)
            stack.extend(forward_children(c))
        return False

    def _branch_reaches_end(start: str) -> bool:
        seen: set = set()
        stack = [start]
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            if nodes.get(c, {}).get("type", "").lower() == "end":
                return True
            stack.extend(forward_children(c))
        return False

    def _emit_split_branch(child_id: str, merge_node, incoming_label: str = None) -> None:
        if child_id == merge_node:
            if incoming_label:
                output.append(f"-> {incoming_label};")
            return
        process_node(child_id, stop_node=merge_node, incoming_label=incoming_label)
        rejoins = merge_node is not None and _branch_reaches(child_id, merge_node)
        if not rejoins and not _branch_reaches_end(child_id):
            output.append("kill")

    def process_node(node_id: str, stop_node: str = None, incoming_label: str = None) -> None:
        if node_id == stop_node or node_id in visited:
            return

        if node_id in loop_tails_of and not _in_loop_render.get(node_id):
            _render_loop(node_id, stop_node=stop_node, incoming_label=incoming_label)
            return

        visited.add(node_id)
        node = nodes[node_id]
        name = node["name"].strip('"')

        ntype = node["type"]
        children = forward_children(node_id)

        if ntype not in ("convergeAND", "convergeXOR", "convergeOR", "init"):
            switch_lane(node_id)

        if incoming_label and ntype not in ("convergeAND", "convergeXOR", "convergeOR", "init"):
            output.append(f"-> {incoming_label};")

        if ntype == "init":
            if children:
                child_id = children[0]
                lbl = edge_label(node_id, child_id)
                if child_id == stop_node:
                    if lbl:
                        output.append(f"-> {lbl};")
                else:
                    process_node(child_id, stop_node=stop_node, incoming_label=lbl)

        elif ntype == "branchXOR":
            merge_node = get_merge_node(node_id)
            branch_stop = merge_node if merge_node is not None else stop_node

            cond_text = f"{name}?" if name else ""

            emit_branch_object_signal(node_id)

            if len(children) > 2:
                output.append(f"switch ({cond_text})")
                for i, child in enumerate(children):
                    lbl = edge_label(node_id, child).strip()
                    if not lbl:
                        lbl = "else" if i == len(children) - 1 else f"case {i + 1}"

                    output.append(f"case ({lbl})")
                    if child != branch_stop:
                        process_node(child, stop_node=branch_stop)
                output.append("endswitch")

            elif len(children) > 0:

                branch_lane = current_lane[0]

                first_child = children[0]
                output.append(f"if ({cond_text}) then ({edge_label(node_id, first_child) or ' '})")
                if first_child == branch_stop:

                    output.append("->;")
                else:
                    process_node(first_child, stop_node=branch_stop)

                if len(children) > 1:

                    if branch_lane and current_lane[0] != branch_lane:
                        output.append(f"|{branch_lane}|")
                        current_lane[0] = branch_lane

                    last_child = children[-1]
                    output.append(f"else ({edge_label(node_id, last_child) or ' '})")
                    if last_child == branch_stop:

                        output.append("->;")
                    else:
                        process_node(last_child, stop_node=branch_stop)

                if branch_lane and current_lane[0] != branch_lane:
                    output.append(f"|{branch_lane}|")
                    current_lane[0] = branch_lane

                output.append("endif")

            if merge_node:
                process_node(merge_node, stop_node=stop_node)

        elif ntype == "end":
            if name:
                output.append(f":{name};")
            output.append("stop")

        elif ntype == "branchAND":
            join_node = get_merge_node(node_id)
            if len(children) > 1:
                output.append("split")
                for i, child_id in enumerate(children):
                    lbl = edge_label(node_id, child_id)
                    _emit_split_branch(child_id, join_node, incoming_label=lbl)
                    if i < len(children) - 1:
                        output.append("split again")
                output.append("end split")
            elif len(children) == 1:
                child_id = children[0]
                lbl = edge_label(node_id, child_id)
                if child_id == join_node:
                    if lbl:
                        output.append(f"-> {lbl};")
                else:
                    process_node(child_id, stop_node=join_node, incoming_label=lbl)

            if join_node:
                process_node(join_node, stop_node=stop_node)

        elif ntype == "branchOR":

            join_node = get_merge_node(node_id)
            branch_stop = join_node if join_node is not None else stop_node
            branch_lane = current_lane[0]

            emit_branch_object_signal(node_id)

            if len(children) > 1:
                output.append("fork")
                for i, child_id in enumerate(children):
                    lbl = edge_label(node_id, child_id) or " "
                    output.append(f"if () then ({lbl})")
                    if child_id != branch_stop:
                        process_node(child_id, stop_node=branch_stop)

                    if branch_lane and current_lane[0] != branch_lane:
                        output.append(f"|{branch_lane}|")
                        current_lane[0] = branch_lane
                    output.append("endif")
                    if i < len(children) - 1:
                        output.append("fork again")
                output.append("end fork")
            elif len(children) == 1:
                child_id = children[0]
                lbl = edge_label(node_id, child_id)
                if child_id == branch_stop:
                    if lbl:
                        output.append(f"-> {lbl};")
                else:
                    process_node(child_id, stop_node=branch_stop, incoming_label=lbl)

            if join_node:
                process_node(join_node, stop_node=stop_node)

        elif ntype in ("convergeAND", "convergeXOR", "convergeOR"):
            if children:
                child_id = children[0]
                lbl = edge_label(node_id, child_id)
                if child_id == stop_node:
                    if lbl:
                        output.append(f"-> {lbl};")
                else:
                    process_node(child_id, stop_node=stop_node, incoming_label=lbl)

        else:
            node_text = name if name else ""

            color = "#LightGray:" if str(node_id).startswith("EA") else ":"
            output.append(f"{color}{node_text};")

            if not children:
                return
            elif len(children) > 1:
                merge_node = get_merge_node(node_id)
                output.append("split")
                for i, child_id in enumerate(children):
                    lbl = edge_label(node_id, child_id)
                    _emit_split_branch(child_id, merge_node, incoming_label=lbl)
                    if i < len(children) - 1:
                        output.append("split again")
                output.append("end split")

                if merge_node:
                    process_node(merge_node, stop_node=stop_node)
            else:
                child_id = children[0]
                lbl = edge_label(node_id, child_id)
                if child_id == stop_node:
                    if lbl:
                        output.append(f"-> {lbl};")
                else:
                    process_node(child_id, stop_node=stop_node, incoming_label=lbl)

    start_node = next((node_id for node_id, n in nodes.items() if n["type"].lower() == "init"), None)
    if start_node:
        process_node(start_node)

    output.append("@enduml")
    return "\n".join(output)


def _build_actor_metadata_block(actors: List[Dict[str, Any]]) -> str:
    if not actors:
        return ""
    lines: List[str] = []
    for actor in actors:
        aid = str(actor.get("id", "")).strip()
        if not aid:
            continue

        name = str(actor.get("name", "") or "").strip().replace(":", " ")
        assigned = [str(n).strip() for n in (actor.get("assigned_nodes") or []) if str(n).strip()]
        lines.append(f"# actor:{aid}:{name}:{','.join(assigned)}")
    return "\n".join(lines)


def _parse_actor_metadata(csv_text: str) -> List[Dict[str, Any]]:
    actors: List[Dict[str, Any]] = []
    for raw in csv_text.splitlines():
        line = raw.strip()
        if not line.startswith("# actor:"):
            continue
        payload = line[len("# actor:") :]
        parts = payload.split(":", 2)
        if len(parts) < 2:
            continue
        aid = parts[0].strip()
        name = parts[1].strip() if len(parts) >= 2 else ""
        node_field = parts[2].strip() if len(parts) >= 3 else ""
        assigned = [tok.strip() for tok in node_field.split(",") if tok.strip()]
        actors.append({"id": aid, "name": name, "assigned_nodes": assigned})
    return actors


def _node_to_actor_map(actors: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for actor in actors:
        aid = str(actor.get("id") or "").strip()
        if not aid:
            continue
        for nid in actor.get("assigned_nodes") or []:
            nid = str(nid).strip()
            if not nid or nid in mapping:
                continue
            mapping[nid] = aid
    return mapping


def _format_variable_range(r_x: Any) -> str:
    if not isinstance(r_x, list) or not r_x:
        return ""
    return "{" + ", ".join(str(v).strip() for v in r_x if str(v).strip()) + "}"


def _build_variable_metadata_block(variable_definitions: List[Dict[str, Any]]) -> str:
    if not variable_definitions:
        return ""
    lines: List[str] = []
    for vdef in variable_definitions:
        vid = str(vdef.get("X", "") or "").strip()
        if not vid:
            continue

        name = str(vdef.get("name", "") or "").strip().replace(":", " ").replace("|", " ")
        range_text = _format_variable_range(vdef.get("R_X"))
        lines.append(f"# variable:{vid}:{name}|{range_text}")
    return "\n".join(lines)


def _parse_variable_metadata(csv_text: str) -> List[Dict[str, str]]:
    variables: List[Dict[str, str]] = []
    for raw in csv_text.splitlines():
        line = raw.strip()
        if not line.startswith("# variable:"):
            continue
        payload = line[len("# variable:") :]
        parts = payload.split(":", 1)
        if not parts or not parts[0].strip():
            continue
        vid = parts[0].strip()
        name_and_range = parts[1] if len(parts) >= 2 else ""
        if "|" in name_and_range:
            name, range_text = name_and_range.split("|", 1)
        else:
            name, range_text = name_and_range, ""
        variables.append(
            {"id": vid, "name": name.strip(), "range": range_text.strip()}
        )
    return variables


_GUARD_SEGMENT = re.compile(r'\[([^\[\]]*)\]')
_VAR_TOKEN = re.compile(r'\b([A-Za-z][A-Za-z0-9_]*)\b')


def _extract_variable_ids(text: str, known_var_ids: set) -> List[str]:
    if not text:
        return []
    seen: List[str] = []
    seen_set: set = set()
    for segment in _GUARD_SEGMENT.findall(text):
        for tok in _VAR_TOKEN.findall(segment):
            if tok in known_var_ids and tok not in seen_set:
                seen.append(tok)
                seen_set.add(tok)
    return seen


def json_to_csv(data: Dict[str, Any], flow_layout: str = FLOW_LAYOUT_HORIZONTAL) -> str:
    elements = data.get("elements", [])
    relationships = data.get("relationships", [])
    actors = data.get("actors", []) if isinstance(data.get("actors", []), list) else []
    variable_definitions = data.get("variable_definitions", [])
    if not isinstance(variable_definitions, list):
        variable_definitions = []
    layout = _normalize_flow_layout(flow_layout)

    if not isinstance(elements, list):
        raise ValueError("Invalid JSON: 'elements' must be a list.")
    if not isinstance(relationships, list):
        raise ValueError("Invalid JSON: 'relationships' must be a list.")

    parents_by_child, transition_by_edge, _ = _collect_edges(relationships)
    converge_gateway_map = _infer_converge_gateways(elements, relationships)

    lines: List[str] = [_build_header(layout).rstrip("\n")]

    metadata_blocks: List[str] = []
    actor_block = _build_actor_metadata_block(actors)
    if actor_block:
        metadata_blocks.append(actor_block)
    variable_block = _build_variable_metadata_block(variable_definitions)
    if variable_block:
        metadata_blocks.append(variable_block)
    if metadata_blocks:
        lines.insert(1, "\n".join(metadata_blocks))

    for node in elements:
        node_id = str(node.get("id", "")).strip()
        node_type = _synthetic_type(node, converge_gateway_map)
        if not node_id or not node_type:
            continue

        name = _node_name(node)
        w, h = _node_dimensions(node_type, layout)

        parents = parents_by_child.get(node_id, [])
        if not parents:
            line = ",".join([
                node_id,
                _quote(name),
                _quote(node_type),
                _quote(""),
                _quote(""),
                _size_cell(w),
                _size_cell(h),
            ])
            lines.append(line)
            continue

        parent_labels = [transition_by_edge.get((p, node_id), "").strip() for p in parents]
        any_edge_has_label = any(parent_labels)

        if len(parents) > 1 and any_edge_has_label:
            for parent, transition in zip(parents, parent_labels):
                line = ",".join([
                    node_id,
                    _quote(name),
                    _quote(node_type),
                    _quote(parent),
                    _quote(transition),
                    _size_cell(w),
                    _size_cell(h),
                ])
                lines.append(line)
        else:
            parent_value = ",".join(parents)
            transition_value = ""
            if len(parents) == 1:
                transition_value = parent_labels[0]
            line = ",".join([
                node_id,
                _quote(name),
                _quote(node_type),
                _quote(parent_value),
                _quote(transition_value),
                _size_cell(w),
                _size_cell(h),
            ])
            lines.append(line)

    return "\n".join(lines) + "\n"
