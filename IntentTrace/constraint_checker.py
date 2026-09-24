from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
import re

from .condition import (
    is_valid as condition_is_valid,
    parse_condition,
    vars_of,
    ConditionParseError,
)
from . import smt


class ConstraintChecking:

    NODE_TYPES = {"init", "end", "action", "branch", "converge"}
    VALID_GATEWAYS = {"AND", "OR", "XOR"}
    GUARD_GATEWAYS = {"XOR", "OR"}
    ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

    def __init__(
        self,
        diagram_json: Dict[str, Any],
    ):
        self.diagram_json = diagram_json
        self.elements: List[Dict[str, Any]] = list(diagram_json.get("elements", []))
        self.variable_definitions: List[Dict[str, Any]] = list(diagram_json.get("variable_definitions", []))
        self.relationships: List[Dict[str, Any]] = list(diagram_json.get("relationships", []))


        self.actors: List[Dict[str, Any]] = list(diagram_json.get("actors", []))

        self.element_ids = [str(e.get("id", "")).strip() for e in self.elements if e.get("id") is not None]
        self.known_nodes: Set[str] = set(self.element_ids)
        self.node_type: Dict[str, str] = {
            str(e.get("id", "")).strip(): str(e.get("type", "")).strip().lower() for e in self.elements if e.get("id")
        }


        self.node_gateway: Dict[str, str] = {
            str(e.get("id", "")).strip(): str(e.get("gateway", "")).strip().upper()
            for e in self.elements if e.get("id")
        }
        self.node_content: Dict[str, str] = {
            str(e.get("id", "")).strip(): str(e.get("content", "") if e.get("content") is not None else "")
            for e in self.elements
            if e.get("id")
        }

        self.precedences = [r for r in self.relationships if str(r.get("type", "")).lower() == "precedence"]
        self.guards = [r for r in self.relationships if str(r.get("type", "")).lower() == "guard"]
        self.labels = [r for r in self.relationships if str(r.get("type", "")).lower() == "label"]

        self.out_edges, self.in_edges = self._build_graph_index(self.precedences)
        self._prepare_diagram_data()

    def _build_graph_index(
        self, precedences: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        out_map: Dict[str, List[str]] = {n: [] for n in self.known_nodes}
        in_map: Dict[str, List[str]] = {n: [] for n in self.known_nodes}

        for rel in precedences:
            a = str(rel.get("A", "")).strip()
            b = str(rel.get("B", "")).strip()
            if a in out_map and b in in_map:
                out_map[a].append(b)
                in_map[b].append(a)

        return out_map, in_map

    def _prepare_diagram_data(self) -> None:
        self.init_nodes = sorted([nid for nid, t in self.node_type.items() if t == "init"])
        self.end_nodes = sorted([nid for nid, t in self.node_type.items() if t == "end"])
        self.action_nodes = sorted([nid for nid, t in self.node_type.items() if t == "action"])
        self.branch_nodes = sorted([nid for nid, t in self.node_type.items() if t == "branch"])
        self.converge_nodes = sorted([nid for nid, t in self.node_type.items() if t == "converge"])
        self.xor_or_branch_nodes = sorted(
            [n for n in self.branch_nodes if self.node_gateway.get(n) in self.GUARD_GATEWAYS]
        )

        self.precedence_pairs = {(str(r.get("A", "")).strip(), str(r.get("B", "")).strip()) for r in self.precedences}


        self.variable_ids: Set[str] = set()
        for d in self.variable_definitions:
            x = str(d.get("X", "")).strip()
            if x:
                self.variable_ids.add(x)

        self.actor_ids: Set[str] = set()

        self.actor_assignments: Dict[str, List[str]] = {}


        self.node_to_actors: Dict[str, List[str]] = {nid: [] for nid in self.known_nodes}
        for actor in self.actors:
            aid = str(actor.get("id", "")).strip()
            if aid:
                self.actor_ids.add(aid)
            assigned = [str(n).strip() for n in (actor.get("assigned_nodes") or []) if str(n).strip()]
            self.actor_assignments[aid] = assigned
            for nid in assigned:
                self.node_to_actors.setdefault(nid, []).append(aid)


        self.parsed_defines: List[Dict[str, Any]] = []
        for d in self.variable_definitions:
            x = str(d.get("X", "")).strip()
            r_x = d.get("R_X", [])
            values = list(r_x) if isinstance(r_x, list) else []
            self.parsed_defines.append({"X": x, "R_X": values})

        self.var_ranges: Dict[str, List[Any]] = {}
        for d in self.parsed_defines:
            if (
                d["X"]
                and self.ID_PATTERN.match(d["X"])
                and d["X"] not in self.known_nodes
                and self._is_valid_range_definition(d["R_X"])
            ):
                self.var_ranges[d["X"]] = d["R_X"]


        self.guards_by_edge: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.parsed_guards: List[Dict[str, Any]] = []
        for g in self.guards:
            a = str(g.get("A", "")).strip()
            b = str(g.get("B", "")).strip()
            condition_text = str(g.get("condition", "")).strip()
            ast = g.get("condition_ast")
            if ast is None and condition_text:
                try:
                    ast = parse_condition(condition_text)
                except ConditionParseError:
                    ast = None
            parsed = {
                "A": a,
                "B": b,
                "condition": condition_text,
                "condition_ast": ast,
            }
            self.parsed_guards.append(parsed)
            self.guards_by_edge[(a, b)].append(parsed)


        self.decision_var_map: Dict[str, Set[str]] = defaultdict(set)
        for bnode in self.xor_or_branch_nodes:
            outgoing = self.out_edges.get(bnode, [])
            for b in outgoing:
                for g in self.guards_by_edge.get((bnode, b), []):
                    ast = g.get("condition_ast")
                    if ast is None:
                        continue
                    self.decision_var_map[bnode].update(vars_of(ast))

    @staticmethod
    def _parse_literal(value: Any) -> Any:
        if isinstance(value, (int, float, bool)) or value is None:
            return value

        s = str(value).strip()
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
            s = s[1:-1]

        lower = s.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False

        if re.fullmatch(r"-?\d+", s):
            try:
                return int(s)
            except ValueError:
                pass

        if re.fullmatch(r"-?\d+\.\d+", s):
            try:
                return float(s)
            except ValueError:
                pass

        return s

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _normalize_range_token(token: Any) -> str:
        return str(token).strip()

    def _token_to_numeric_interval(self, token: Any) -> Optional[Tuple[Optional[float], bool, Optional[float], bool]]:
        s = self._normalize_range_token(token)

        if re.fullmatch(r"\[\s*-\s*\]", s):
            return (None, False, None, False)

        closed = re.fullmatch(r"\[\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*\]", s)
        if closed:
            lo = float(closed.group(1))
            hi = float(closed.group(2))
            if lo > hi:
                return None
            return (lo, True, hi, True)

        lower_open = re.fullmatch(r"\[\s*(-?\d+(?:\.\d+)?)\s*-\s*\]", s)
        if lower_open:
            lo = float(lower_open.group(1))
            return (lo, True, None, False)

        upper_open = re.fullmatch(r"\[\s*-\s*(-?\d+(?:\.\d+)?)\s*\]", s)
        if upper_open:
            hi = float(upper_open.group(1))
            return (None, False, hi, True)

        return None

    def _is_valid_range_definition(self, range_values: List[Any]) -> bool:
        if not range_values:
            return False
        if len(range_values) >= 2:
            return True
        return self._token_to_numeric_interval(range_values[0]) is not None

    def _build_numeric_domain(self, range_values: List[Any]) -> Optional[List[Tuple[Optional[float], bool, Optional[float], bool]]]:
        intervals: List[Tuple[Optional[float], bool, Optional[float], bool]] = []
        for raw in range_values:
            interval = self._token_to_numeric_interval(raw)
            if interval is not None:
                intervals.append(interval)
                continue

            parsed = self._parse_literal(raw)
            if not self._is_number(parsed):
                return None
            v = float(parsed)
            intervals.append((v, True, v, True))

        return intervals


    def _compute_reachable(self, starts: List[str], adjacency: Dict[str, List[str]]) -> Set[str]:
        reachable: Set[str] = set()
        queue = deque(starts)
        while queue:
            node = queue.popleft()
            if node in reachable:
                continue
            reachable.add(node)
            for nxt in adjacency.get(node, []):
                if nxt not in reachable:
                    queue.append(nxt)
        return reachable


    def validate_structural(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        def put(stc: str, ok: bool, violations: Optional[List[Any]] = None, details: Optional[Any] = None) -> None:
            results[stc] = bool(ok)
            if violations is not None:
                results[f"{stc}_violations"] = violations
            if details is not None:
                results[f"{stc}_details"] = details


        duplicate_ids = sorted({n for n in self.element_ids if self.element_ids.count(n) > 1})
        empty_ids = sorted({str(e.get("id", "")).strip() for e in self.elements if not str(e.get("id", "")).strip()})
        e_v_overlap = sorted(self.known_nodes & self.variable_ids)
        e_a_overlap = sorted(self.known_nodes & self.actor_ids)
        v_a_overlap = sorted(self.variable_ids & self.actor_ids)
        stc1_violations: List[Any] = []
        stc1_violations.extend(duplicate_ids)
        stc1_violations.extend(empty_ids)
        if e_v_overlap:
            stc1_violations.append({"E_intersect_V": e_v_overlap})
        if e_a_overlap:
            stc1_violations.append({"E_intersect_A": e_a_overlap})
        if v_a_overlap:
            stc1_violations.append({"V_intersect_A": v_a_overlap})
        put(
            "STC1",
            len(self.elements) > 0
            and not duplicate_ids
            and not empty_ids
            and not e_v_overlap
            and not e_a_overlap
            and not v_a_overlap,
            stc1_violations,
        )


        invalid_types: List[Dict[str, Any]] = []
        for nid in self.known_nodes:
            t = self.node_type.get(nid, "")
            if t not in self.NODE_TYPES:
                invalid_types.append({"id": nid, "type": t})
                continue
            if t == "branch":
                gateway = self.node_gateway.get(nid, "")
                if gateway not in self.VALID_GATEWAYS:
                    invalid_types.append({"id": nid, "type": t, "gateway": gateway, "issue": "invalid-gateway"})
            elif t == "converge":
                gateway = self.node_gateway.get(nid, "")
                if gateway:
                    invalid_types.append(
                        {"id": nid, "type": t, "gateway": gateway, "issue": "converge-must-not-have-gateway"}
                    )
        put("STC2", len(invalid_types) == 0, invalid_types)


        stc3_violations = self.init_nodes if len(self.init_nodes) != 1 else []
        put("STC3", len(stc3_violations) == 0, stc3_violations)


        stc4_violations = [nid for nid in self.init_nodes if len(self.in_edges.get(nid, [])) != 0]
        put("STC4", len(stc4_violations) == 0, stc4_violations)


        stc5_violations = ["missing-end-node"] if len(self.end_nodes) < 1 else []
        put("STC5", len(stc5_violations) == 0, stc5_violations)


        stc6_violations = [nid for nid in self.end_nodes if len(self.out_edges.get(nid, [])) != 0]
        put("STC6", len(stc6_violations) == 0, stc6_violations)


        stc7_violations: List[Dict[str, Any]] = []

        for nid in self.action_nodes:
            if not self.node_content.get(nid, "").strip():
                stc7_violations.append(
                    {
                        "node": nid,
                        "content": self.node_content.get(nid, ""),
                        "issue": "action-content-must-be-non-empty",
                    }
                )

        for actor in self.actors:
            aid = str(actor.get("id", "")).strip()
            name = str(actor.get("name", "") if actor.get("name") is not None else "").strip()
            if not name:
                stc7_violations.append(
                    {
                        "actor": aid,
                        "name": name,
                        "issue": "actor-name-must-be-non-empty",
                    }
                )


        for vdef in self.variable_definitions:
            vid = str(vdef.get("X", "")).strip()
            vname = str(vdef.get("name", "") if vdef.get("name") is not None else "").strip()
            if not vname:
                stc7_violations.append(
                    {
                        "variable": vid,
                        "name": vname,
                        "issue": "variable-name-must-be-non-empty",
                    }
                )
        put("STC7", len(stc7_violations) == 0, stc7_violations)


        stc8_violations = [
            {"node": nid, "in": len(self.in_edges.get(nid, [])), "out": len(self.out_edges.get(nid, []))}
            for nid in self.action_nodes
            if not (len(self.in_edges.get(nid, [])) == 1 and len(self.out_edges.get(nid, [])) == 1)
        ]
        put("STC8", len(stc8_violations) == 0, stc8_violations)


        stc9_violations = [
            {"node": nid, "in": len(self.in_edges.get(nid, [])), "out": len(self.out_edges.get(nid, []))}
            for nid in self.branch_nodes
            if not (len(self.in_edges.get(nid, [])) == 1 and len(self.out_edges.get(nid, [])) >= 2)
        ]
        put("STC9", len(stc9_violations) == 0, stc9_violations)


        stc10_violations = [
            {"node": nid, "in": len(self.in_edges.get(nid, [])), "out": len(self.out_edges.get(nid, []))}
            for nid in self.converge_nodes
            if not (len(self.in_edges.get(nid, [])) >= 2 and len(self.out_edges.get(nid, [])) == 1)
        ]
        put("STC10", len(stc10_violations) == 0, stc10_violations)


        stc11_violations = []
        for d in self.parsed_defines:
            numeric_domain = self._build_numeric_domain(d["R_X"])
            is_valid_count = False
            if numeric_domain is not None:
                has_interval = False
                points = set()
                for lo, _, hi, _ in numeric_domain:
                    if lo is None or hi is None or lo != hi:
                        has_interval = True
                        break
                    if lo is not None and lo == hi:
                        points.add(lo)
                if has_interval or len(points) >= 2:
                    is_valid_count = True
            else:
                if len(set(str(v).strip() for v in d["R_X"])) >= 2:
                    is_valid_count = True

            if not is_valid_count:
                stc11_violations.append({**d, "issue": "range-must-contain-at-least-two-distinct-values"})
        put("STC11", len(stc11_violations) == 0, stc11_violations)


        stc12_violations = []
        for rel in self.precedences:
            a = str(rel.get("A", "")).strip()
            b = str(rel.get("B", "")).strip()
            if a not in self.known_nodes or b not in self.known_nodes:
                stc12_violations.append({"A": a, "B": b, "issue": "unknown-node"})
        put("STC12", len(stc12_violations) == 0, stc12_violations)


        stc13_violations = []
        for g in self.parsed_guards:
            if (g["A"], g["B"]) not in self.precedence_pairs:
                stc13_violations.append(
                    {
                        "A": g["A"],
                        "B": g["B"],
                        "condition": g["condition"],
                        "issue": "guard-without-precedence",
                    }
                )
        put("STC13", len(stc13_violations) == 0, stc13_violations)


        stc14_violations = []
        for g in self.parsed_guards:
            if g.get("condition_ast") is None or not condition_is_valid(g.get("condition", "")):
                stc14_violations.append(
                    {
                        "A": g["A"],
                        "B": g["B"],
                        "condition": g["condition"],
                        "issue": "invalid-condition-expression",
                    }
                )
        put("STC14", len(stc14_violations) == 0, stc14_violations)


        stc15_violations = []
        for g in self.parsed_guards:
            ast = g.get("condition_ast")
            if ast is None:
                continue
            referenced = vars_of(ast)
            for v in sorted(referenced):
                if v not in self.var_ranges:
                    stc15_violations.append(
                        {
                            "A": g["A"],
                            "B": g["B"],
                            "condition": g["condition"],
                            "variable": v,
                            "issue": "undefined-variable",
                        }
                    )
        put("STC15", len(stc15_violations) == 0, stc15_violations)


        stc16_violations = []
        for bnode in self.xor_or_branch_nodes:
            for b in self.out_edges.get(bnode, []):
                guards = self.guards_by_edge.get((bnode, b), [])
                if len(guards) != 1:
                    stc16_violations.append(
                        {
                            "branch": bnode,
                            "edge": [bnode, b],
                            "guard_count": len(guards),
                            "issue": "missing-or-multiple-guards-for-xor-or-or-branch",
                        }
                    )
        put("STC16", len(stc16_violations) == 0, stc16_violations)


        stc17_violations = []
        for g in self.parsed_guards:
            origin = g["A"]
            if origin not in self.xor_or_branch_nodes:
                stc17_violations.append(
                    {
                        "A": g["A"],
                        "B": g["B"],
                        "condition": g["condition"],
                        "issue": "guard-does-not-originate-from-xor-or-or-branch",
                    }
                )
        put("STC17", len(stc17_violations) == 0, stc17_violations)


        stc18_violations = []
        for l in self.labels:
            a = str(l.get("A", "")).strip()
            b = str(l.get("B", "")).strip()
            if (a, b) not in self.precedence_pairs:
                stc18_violations.append({"A": a, "B": b, "issue": "label-without-precedence"})
        put("STC18", len(stc18_violations) == 0, stc18_violations)


        if len(self.init_nodes) == 1:
            reachable_from_init = self._compute_reachable([self.init_nodes[0]], self.out_edges)
            stc19_violations = sorted([n for n in self.known_nodes if n not in self.init_nodes and n not in reachable_from_init])
            put("STC19", len(stc19_violations) == 0, stc19_violations)
        else:
            put("STC19", False, sorted(list(self.known_nodes)), {"issue": "requires-exactly-one-init"})


        if self.end_nodes:
            can_reach_end = self._compute_reachable(self.end_nodes, self.in_edges)
            stc20_violations = sorted([n for n in self.known_nodes if n not in self.end_nodes and n not in can_reach_end])
            put("STC20", len(stc20_violations) == 0, stc20_violations)
        else:
            put("STC20", False, sorted(list(self.known_nodes)), {"issue": "requires-at-least-one-end"})


        stc21_violations: List[Dict[str, Any]] = []
        for actor in self.actors:
            aid = str(actor.get("id", "")).strip()
            assigned = self.actor_assignments.get(aid, [])
            if not assigned:
                stc21_violations.append(
                    {
                        "actor": aid,
                        "issue": "actor-assigned-nodes-must-be-non-empty",
                    }
                )
        put("STC21", len(stc21_violations) == 0, stc21_violations)


        stc22_violations: List[Dict[str, Any]] = []
        for nid in sorted(self.known_nodes):
            owners = self.node_to_actors.get(nid, [])
            if len(owners) == 0:
                stc22_violations.append(
                    {
                        "node": nid,
                        "issue": "node-not-assigned-to-any-actor",
                    }
                )
            elif len(owners) > 1:
                stc22_violations.append(
                    {
                        "node": nid,
                        "actors": owners,
                        "issue": "node-assigned-to-multiple-actors",
                    }
                )


        for aid, assigned in self.actor_assignments.items():
            for nid in assigned:
                if nid not in self.known_nodes:
                    stc22_violations.append(
                        {
                            "actor": aid,
                            "node": nid,
                            "issue": "actor-references-unknown-node",
                        }
                    )
        put("STC22", len(stc22_violations) == 0, stc22_violations)

        stc_keys = [k for k in results if re.fullmatch(r"STC\d+", k)]
        results["all_stc_passed"] = all(bool(results[k]) for k in stc_keys)
        results["stc_count"] = len(stc_keys)

        return results

    def validate_semantic(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {"logical_sanity_validation_engine": "smt-z3"}

        def put(lsc: str, ok: bool, violations: Optional[List[Any]] = None, details: Optional[Any] = None) -> None:
            results[lsc] = bool(ok)
            if violations is not None:
                results[f"{lsc}_violations"] = violations
            if details is not None:
                results[f"{lsc}_details"] = details

        self._validate_semantic_smt(put)

        lsc_keys = [k for k in results if re.fullmatch(r"LSC\d+", k)]
        results["all_lsc_passed"] = all(bool(results[k]) for k in lsc_keys)
        results["lsc_count"] = len(lsc_keys)

        return results


    def _smt_failure(self, asts: List[Any], exc: Exception) -> Dict[str, Any]:
        untyped: List[str] = []
        for ast in asts:
            if ast is None:
                continue
            for name in sorted(vars_of(ast)):
                if name not in self.var_ranges and name not in untyped:
                    untyped.append(name)
        if untyped:
            return {
                "issue": "guard-variable-has-no-declared-range",
                "variables": untyped,
            }
        return {"issue": "smt-error", "detail": str(exc)}

    def _validate_semantic_smt(self, put) -> None:


        lsc1_violations: List[Dict[str, Any]] = []
        for g in self.parsed_guards:
            ast = g.get("condition_ast")
            if ast is None:
                continue
            try:
                ok, _witness = smt.check_guard_satisfiable(
                    ast, self.var_ranges
                )
            except Exception as exc:
                lsc1_violations.append(
                    {
                        "A": g["A"],
                        "B": g["B"],
                        "condition": g["condition"],
                        **self._smt_failure([ast], exc),
                    }
                )
                continue
            if not ok:
                lsc1_violations.append(
                    {
                        "A": g["A"],
                        "B": g["B"],
                        "condition": g["condition"],
                        "issue": "guard-unsatisfiable-under-declared-ranges",
                    }
                )
        put("LSC1", len(lsc1_violations) == 0, lsc1_violations)


        lsc2_violations: List[Dict[str, Any]] = []
        lsc3_violations: List[Dict[str, Any]] = []

        for bnode in self.xor_or_branch_nodes:
            gateway = self.node_gateway.get(bnode)
            decision_guards: List[Dict[str, Any]] = []
            for b in self.out_edges.get(bnode, []):
                for g in self.guards_by_edge.get((bnode, b), []):
                    if g.get("condition_ast") is not None:
                        decision_guards.append(
                            {
                                "edge": (bnode, b),
                                "ast": g["condition_ast"],
                                "condition": g["condition"],
                            }
                        )

            if not decision_guards:
                continue


            if gateway == "XOR" and len(decision_guards) >= 2:
                try:
                    ok, witness = smt.check_xor_mutual_exclusion(
                        decision_guards, self.var_ranges
                    )
                except Exception as exc:
                    lsc2_violations.append(
                        {
                            "branch": bnode,
                            **self._smt_failure([x["ast"] for x in decision_guards], exc),
                        }
                    )
                else:
                    if not ok and witness is not None:
                        matched_edges = witness.pop("_matched_edges", [])
                        lsc2_violations.append(
                            {
                                "branch": bnode,
                                "state": witness,
                                "matched_edges": matched_edges,
                                "issue": "overlapping-guards-in-xor-branch",
                            }
                        )


            try:
                ok, witness = smt.check_xor_or_coverage(
                    decision_guards, self.var_ranges
                )
            except Exception as exc:
                lsc3_violations.append(
                    {
                        "branch": bnode,
                        **self._smt_failure([x["ast"] for x in decision_guards], exc),
                    }
                )
            else:
                if not ok and witness is not None:
                    lsc3_violations.append(
                        {
                            "branch": bnode,
                            "state": witness,
                            "issue": "state-not-covered-by-any-guard",
                        }
                    )

        put("LSC2", len(lsc2_violations) == 0, lsc2_violations)
        put("LSC3", len(lsc3_violations) == 0, lsc3_violations)

    def validate(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        stc_results = self.validate_structural()
        lsc_results = self.validate_semantic()

        results.update(stc_results)
        results.update(lsc_results)

        results["all_constraints_passed"] = stc_results.get("all_stc_passed", False) and lsc_results.get("all_lsc_passed", False)
        results["constraint_count"] = stc_results.get("stc_count", 0) + lsc_results.get("lsc_count", 0)

        return results
