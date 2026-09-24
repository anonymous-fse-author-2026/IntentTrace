from typing import Any, Dict, Iterable, List, Optional, Tuple

import z3

from .condition import vars_of


def _classify_range(range_values: List[Any]) -> str:
    if not range_values:
        return "string"


    for raw in range_values:
        s = str(raw).strip()
        if s.startswith("[") and s.endswith("]"):
            return "numeric"


    numeric_count = 0
    bool_count = 0
    for raw in range_values:
        s = str(raw).strip().lower()
        if s in {"true", "false"}:
            bool_count += 1
            continue
        try:
            float(s)
            numeric_count += 1
        except (TypeError, ValueError):
            pass

    if bool_count == len(range_values):
        return "boolean"
    if numeric_count == len(range_values):
        return "numeric"
    return "string"


def _interval_assertion(z3_var: Any, raw_token: str) -> Optional[Any]:
    import re
    s = str(raw_token).strip()
    numeric = r"-?\d+(?:\.\d+)?"

    if re.fullmatch(r"\[\s*-\s*\]", s):
        return None

    closed = re.fullmatch(rf"\[\s*({numeric})\s*-\s*({numeric})\s*\]", s)
    if closed:
        lo = float(closed.group(1))
        hi = float(closed.group(2))
        return z3.And(z3_var >= lo, z3_var <= hi)

    lower_open = re.fullmatch(rf"\[\s*({numeric})\s*-\s*\]", s)
    if lower_open:
        return z3_var >= float(lower_open.group(1))

    upper_open = re.fullmatch(rf"\[\s*-\s*({numeric})\s*\]", s)
    if upper_open:
        return z3_var <= float(upper_open.group(1))

    return None


def build_z3_env(
    var_ranges: Dict[str, List[Any]],
    needed_vars: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Any], List[Any]]:

    if needed_vars is None:
        needed_vars = list(var_ranges.keys())

    env: Dict[str, Any] = {}
    asserts: List[Any] = []

    for v in needed_vars:
        range_values = var_ranges.get(v)
        if range_values is None:

            env[v] = z3.Real(v)
            continue

        kind = _classify_range(range_values)

        if kind == "boolean":
            env[v] = z3.Bool(v)
            allowed = []
            for raw in range_values:
                s = str(raw).strip().lower()
                if s == "true":
                    allowed.append(env[v] == True)
                elif s == "false":
                    allowed.append(env[v] == False)
            if allowed:
                asserts.append(z3.Or(*allowed) if len(allowed) > 1 else allowed[0])

        elif kind == "numeric":
            env[v] = z3.Real(v)
            disjuncts: List[Any] = []
            for raw in range_values:
                s = str(raw).strip()
                if s.startswith("[") and s.endswith("]"):
                    interval = _interval_assertion(env[v], s)
                    if interval is None:

                        disjuncts = []
                        break
                    disjuncts.append(interval)
                else:
                    try:
                        disjuncts.append(env[v] == float(s))
                    except (TypeError, ValueError):
                        pass
            if disjuncts:
                asserts.append(z3.Or(*disjuncts) if len(disjuncts) > 1 else disjuncts[0])

        else:
            env[v] = z3.String(v)
            cleaned: List[str] = []
            for raw in range_values:
                s = str(raw).strip()
                if s.startswith('"') and s.endswith('"') and len(s) >= 2:
                    s = s[1:-1]
                elif s.startswith("'") and s.endswith("'") and len(s) >= 2:
                    s = s[1:-1]
                cleaned.append(s)
            if cleaned:
                disjuncts = [env[v] == z3.StringVal(c) for c in cleaned]
                asserts.append(z3.Or(*disjuncts) if len(disjuncts) > 1 else disjuncts[0])

    return env, asserts


def ast_to_z3(ast: Tuple[Any, ...], env: Dict[str, Any]) -> Any:

    head = ast[0]

    if head == "VAL":
        v = ast[1]
        if isinstance(v, bool):
            return z3.BoolVal(v)
        if isinstance(v, (int, float)):
            return z3.RealVal(v)
        return z3.StringVal(str(v))

    if head == "VAR":
        if ast[1] not in env:

            env[ast[1]] = z3.Real(ast[1])
        return env[ast[1]]

    if head == "NOT":
        return z3.Not(_to_bool(ast_to_z3(ast[1], env)))

    if head == "AND":
        return z3.And(_to_bool(ast_to_z3(ast[1], env)), _to_bool(ast_to_z3(ast[2], env)))

    if head == "OR":
        return z3.Or(_to_bool(ast_to_z3(ast[1], env)), _to_bool(ast_to_z3(ast[2], env)))

    if head == "REL":
        op = ast[1]
        left = ast_to_z3(ast[2], env)
        right = ast_to_z3(ast[3], env)


        left, right = _reconcile_sorts(left, right)
        if op == "=":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        raise ValueError(f"Unknown relational op: {op}")

    if head == "ADD":
        op = ast[1]
        left = ast_to_z3(ast[2], env)
        right = ast_to_z3(ast[3], env)
        return left + right if op == "+" else left - right

    if head == "MUL":
        op = ast[1]
        left = ast_to_z3(ast[2], env)
        right = ast_to_z3(ast[3], env)
        return left * right if op == "*" else left / right

    raise ValueError(f"Unknown AST node: {head}")


def _sort_of(expr: Any) -> Any:
    try:
        return expr.sort()
    except Exception:
        return None


def _reconcile_sorts(left: Any, right: Any) -> Tuple[Any, Any]:
    ls, rs = _sort_of(left), _sort_of(right)
    if ls is None or rs is None or ls == rs:
        return left, right

    str_sort = z3.StringSort()
    real_sort = z3.RealSort()
    int_sort = z3.IntSort()
    bool_sort = z3.BoolSort()

    def is_num_sort(s):
        return s in (real_sort, int_sort)


    def as_string_const(expr):
        try:
            if z3.is_string_value(expr):
                return expr.as_string()
        except Exception:
            pass
        return None

    def as_num_const(expr):
        try:
            if z3.is_rational_value(expr) or z3.is_int_value(expr):
                return expr
        except Exception:
            pass
        return None

    def as_bool_literal(expr):
        sc = as_string_const(expr)
        if sc is not None:
            low = sc.strip().lower()
            if low in ("true", "1"):
                return True
            if low in ("false", "0"):
                return False
            return None
        nc = as_num_const(expr)
        if nc is not None:
            try:
                f = float(nc.as_decimal(6).rstrip("?")) if hasattr(nc, "as_decimal") else None
            except Exception:
                f = None
            if f is None:
                try:
                    f = float(str(nc))
                except (TypeError, ValueError):
                    f = None
            if f == 1.0:
                return True
            if f == 0.0:
                return False
        return None


    if ls == bool_sort and rs != bool_sort:
        bv = as_bool_literal(right)
        if bv is not None:
            return left, z3.BoolVal(bv)
    if rs == bool_sort and ls != bool_sort:
        bv = as_bool_literal(left)
        if bv is not None:
            return z3.BoolVal(bv), right


    if ls == str_sort and is_num_sort(rs) and as_num_const(right) is not None:
        return left, z3.StringVal(str(right))
    if rs == str_sort and is_num_sort(ls) and as_num_const(left) is not None:
        return z3.StringVal(str(left)), right


    if is_num_sort(ls) and rs == str_sort:
        sc = as_string_const(right)
        if sc is not None:
            try:
                return left, z3.RealVal(float(sc))
            except (TypeError, ValueError):
                return left, right
    if is_num_sort(rs) and ls == str_sort:
        sc = as_string_const(left)
        if sc is not None:
            try:
                return z3.RealVal(float(sc)), right
            except (TypeError, ValueError):
                return left, right

    return left, right


def _to_bool(expr: Any) -> Any:
    try:
        sort = expr.sort()
    except Exception:
        return expr
    if sort == z3.BoolSort():
        return expr
    return expr == z3.BoolVal(True)


def _extract_state(model: Any, env: Dict[str, Any]) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    for name, z3_var in env.items():
        try:
            value = model.evaluate(z3_var, model_completion=True)
        except Exception:
            continue
        try:
            sort = z3_var.sort()
        except Exception:
            sort = None

        if sort == z3.BoolSort():
            state[name] = bool(z3.is_true(value))
        elif sort == z3.StringSort():
            try:
                state[name] = value.as_string()
            except Exception:
                state[name] = str(value)
        else:
            try:
                if value.is_int():
                    state[name] = value.as_long()
                else:

                    num = value.numerator_as_long()
                    den = value.denominator_as_long()
                    state[name] = num / den
            except Exception:
                try:
                    state[name] = float(value.as_decimal(6).rstrip("?"))
                except Exception:
                    state[name] = str(value)
    return state


def check_guard_satisfiable(
    ast: Tuple[Any, ...],
    var_ranges: Dict[str, List[Any]],
) -> Tuple[bool, Optional[Dict[str, Any]]]:

    referenced = vars_of(ast)
    env, range_asserts = build_z3_env(var_ranges, needed_vars=referenced)

    solver = z3.Solver()
    for a in range_asserts:
        solver.add(a)
    solver.add(_to_bool(ast_to_z3(ast, env)))

    if solver.check() == z3.sat:
        return True, _extract_state(solver.model(), env)
    return False, None


def check_xor_mutual_exclusion(
    guards: List[Dict[str, Any]],
    var_ranges: Dict[str, List[Any]],
) -> Tuple[bool, Optional[Dict[str, Any]]]:

    if len(guards) < 2:
        return True, None


    all_vars = set()
    for g in guards:
        all_vars.update(vars_of(g["ast"]))

    env, range_asserts = build_z3_env(var_ranges, needed_vars=all_vars)

    for i in range(len(guards)):
        for j in range(i + 1, len(guards)):
            solver = z3.Solver()
            for a in range_asserts:
                solver.add(a)
            solver.add(_to_bool(ast_to_z3(guards[i]["ast"], env)))
            solver.add(_to_bool(ast_to_z3(guards[j]["ast"], env)))

            if solver.check() == z3.sat:
                state = _extract_state(solver.model(), env)
                state["_matched_edges"] = [
                    list(guards[i].get("edge", ())),
                    list(guards[j].get("edge", ())),
                ]
                return False, state

    return True, None


def check_xor_or_coverage(
    guards: List[Dict[str, Any]],
    var_ranges: Dict[str, List[Any]],
) -> Tuple[bool, Optional[Dict[str, Any]]]:

    if not guards:
        return True, None

    all_vars = set()
    for g in guards:
        all_vars.update(vars_of(g["ast"]))

    env, range_asserts = build_z3_env(var_ranges, needed_vars=all_vars)

    solver = z3.Solver()
    for a in range_asserts:
        solver.add(a)

    disjuncts = [_to_bool(ast_to_z3(g["ast"], env)) for g in guards]
    if not disjuncts:
        return True, None

    union = disjuncts[0] if len(disjuncts) == 1 else z3.Or(*disjuncts)
    solver.add(z3.Not(union))

    if solver.check() == z3.sat:
        return False, _extract_state(solver.model(), env)
    return True, None
