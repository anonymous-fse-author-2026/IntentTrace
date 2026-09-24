"""Formal grammar parser for the process-model input language.

The grammar (EBNF, braces are literal terminals):

    <PROCESS>     ::= <DECL_BLOCK> <REL_BLOCK>
    <DECL_BLOCK>  ::= ( <NODE_DECL> | <VAR_DECL> | <ACTOR_DECL> )+
    <NODE_DECL>   ::= ( "init" | "end" | "converge" ) "(" <ID> ")"
                    | "branch" "(" <ID> "," ( "AND" | "OR" | "XOR" ) ")"
                    | "action" "(" <ID> "," <STR> ")"
    <VAR_DECL>    ::= "defines" "(" <VAR> "," <STR> "," <RANGE> ")"
    <ACTOR_DECL>  ::= "actor" "(" <ID> "," <STR> "," "[" <ID> ( "," <ID> )* "]" ")"
    <REL_BLOCK>   ::= ( <PRECEDENCE> | <GUARD> | <LABEL> )+
    <PRECEDENCE>  ::= "precedence" "(" <ID> "," <ID> ")"
    <GUARD>       ::= "guard" "(" <ID> "," <ID> "," <CONDITION> ")"
    <LABEL>       ::= "label" "(" <ID> "," <ID> "," <STR> ")"
    <RANGE>       ::= "{" ( <VAL> ( "," <VAL> )* | "[" <NUM>? "-" <NUM>? "]" )? "}"
    <CONDITION>   ::= (delegated to IntentTrace.condition)
    <ID>/<VAR>    ::= <LETTER> ( <LETTER> | <DIGIT> | "_" )*
    <STR>         ::= '"' <CHAR>* '"' | "'" <CHAR>* "'"
    <VAL>         ::= <STR> | <BOOL> | <NUM>
    <BOOL>        ::= "true" | "false"
    <NUM>         ::= "-"? <DIGIT>+ ( "." <DIGIT>+ )?

The parser produces the neutral model dict (the only representation used from
this point on):

    {
      "elements":             [ {id, type, content?, gateway?}, ... ],
      "variable_definitions": [ {X, name, R_X}, ... ],
      "actors":               [ {id, name, assigned_nodes}, ... ],
      "relationships":        [ precedence / guard / label dicts ],
    }

Plus a list of ``rejected`` statements (text + reason) so the caller can report
what was dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .condition import parse_condition, split_guard_args, ConditionParseError


class GrammarError(ValueError):
    pass


_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


@dataclass
class Token:
    kind: str

    value: Any


_KEYWORDS = {
    "init", "end", "converge", "branch", "action",
    "defines", "actor", "precedence", "guard", "label",
    "AND", "OR", "XOR", "true", "false",
}

_SINGLE = {
    "(": "LPAREN",
    ")": "RPAREN",
    "[": "LBRACK",
    "]": "RBRACK",
    "{": "LBRACE",
    "}": "RBRACE",
    ",": "COMMA",
}


def _tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in _SINGLE:
            tokens.append(Token(_SINGLE[ch], ch))
            i += 1
            continue

        if ch in ("'", '"'):
            quote = ch
            i += 1
            buf: List[str] = []
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    buf.append(text[i + 1])
                    i += 2
                    continue
                buf.append(text[i])
                i += 1
            if i >= n:
                raise GrammarError("unterminated string literal")
            i += 1
            tokens.append(Token("STRING", "".join(buf)))
            continue


        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            m = re.compile(r"\d+(?:\.\d+)?").match(text, i)
            raw = m.group(0)
            tokens.append(Token("NUMBER", float(raw) if "." in raw else int(raw)))
            i = m.end()
            continue

        if ch == "-":
            tokens.append(Token("DASH", "-"))
            i += 1
            continue

        m = _ID_RE.match(text, i)
        if m:
            raw = m.group(0)
            if raw in _KEYWORDS:
                tokens.append(Token("KEYWORD", raw))
            else:
                tokens.append(Token("IDENT", raw))
            i = m.end()
            continue
        raise GrammarError(f"unexpected character {ch!r}")
    return tokens


_STATEMENT_HEAD = re.compile(r"[A-Za-z][A-Za-z0-9_]*\s*\(")


def _strip_decorations(text: str) -> str:
    text = re.sub(r"```[a-zA-Z0-9_-]*|```", "", text)
    kept: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("===") and stripped.endswith("==="):
            continue
        kept.append(line)
    return "\n".join(kept)


def _split_statements(text: str) -> List[str]:
    statements: List[str] = []
    buf: List[str] = []
    depth = 0
    in_string: Optional[str] = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)
        if in_string is not None:
            if ch == "\\" and i + 1 < n:
                buf.append(text[i + 1])
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue
        if ch in ("'", '"'):
            in_string = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                statements.append("".join(buf).strip())
                buf = []
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return [s for s in statements if s]


class _StatementParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self, offset: int = 0) -> Optional[Token]:
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else None

    def _next(self) -> Token:
        tok = self._peek()
        if tok is None:
            raise GrammarError("unexpected end of statement")
        self.pos += 1
        return tok

    def _expect(self, kind: str, value: Any = None) -> Token:
        tok = self._peek()
        if tok is None or tok.kind != kind or (value is not None and tok.value != value):
            want = value if value is not None else kind
            got = tok.value if tok else "end-of-input"
            raise GrammarError(f"expected {want!r}, found {got!r}")
        return self._next()

    def _at_end(self) -> bool:
        return self.pos >= len(self.tokens)


    def _id(self) -> str:
        return str(self._expect("IDENT").value)

    def _string(self) -> str:
        return str(self._expect("STRING").value)


    def parse_statement(self) -> Dict[str, Any]:
        head = self._peek()
        if head is None or head.kind != "KEYWORD":
            raise GrammarError("statement must begin with a predicate name")
        name = head.value

        dispatch = {
            "init": self._simple_node,
            "end": self._simple_node,
            "converge": self._simple_node,
            "branch": self._branch_node,
            "action": self._action_node,
            "defines": self._var_decl,
            "actor": self._actor_decl,
            "precedence": self._precedence,
            "label": self._label,
        }
        if name not in dispatch:
            raise GrammarError(f"unknown predicate {name!r}")
        result = dispatch[name]()
        if not self._at_end():
            extra = self._peek()
            raise GrammarError(f"trailing tokens after statement: {extra.value!r}")
        return result


    def _simple_node(self) -> Dict[str, Any]:
        kind = self._expect("KEYWORD").value
        self._expect("LPAREN")
        node_id = self._id()
        self._expect("RPAREN")
        return {"_category": "node", "id": node_id, "type": kind, "content": ""}

    def _branch_node(self) -> Dict[str, Any]:
        self._expect("KEYWORD", "branch")
        self._expect("LPAREN")
        node_id = self._id()
        self._expect("COMMA")


        tok = self._peek()
        if tok is not None and tok.kind == "KEYWORD" and tok.value in ("AND", "OR", "XOR"):
            gate = self._next().value
        elif tok is not None and tok.kind == "STRING" and tok.value in ("AND", "OR", "XOR"):
            gate = self._next().value
        else:
            found = tok.value if tok else "end-of-input"
            raise GrammarError(f"branch gateway must be AND|OR|XOR, found {found!r}")
        self._expect("RPAREN")
        return {"_category": "node", "id": node_id, "type": "branch", "content": "", "gateway": gate}

    def _action_node(self) -> Dict[str, Any]:
        self._expect("KEYWORD", "action")
        self._expect("LPAREN")
        node_id = self._id()
        self._expect("COMMA")
        content = self._string()
        self._expect("RPAREN")
        return {"_category": "node", "id": node_id, "type": "action", "content": content}


    def _var_decl(self) -> Dict[str, Any]:
        self._expect("KEYWORD", "defines")
        self._expect("LPAREN")
        var_id = self._id()
        self._expect("COMMA")
        name = self._string()
        self._expect("COMMA")
        rng = self._range()
        self._expect("RPAREN")
        return {"_category": "variable", "X": var_id, "name": name, "R_X": rng}

    def _range(self) -> List[str]:
        self._expect("LBRACE")

        if self._peek() and self._peek().kind == "RBRACE":
            self._next()
            return []

        if self._peek() and self._peek().kind == "LBRACK":
            interval = self._interval()
            self._expect("RBRACE")
            return [interval]

        values = [self._value()]
        while self._peek() and self._peek().kind == "COMMA":
            self._next()
            values.append(self._value())
        self._expect("RBRACE")
        return values

    def _interval(self) -> str:
        self._expect("LBRACK")


        parts: List[str] = []
        while True:
            tok = self._peek()
            if tok is None:
                raise GrammarError("unterminated interval")
            if tok.kind == "RBRACK":
                break
            if tok.kind == "NUMBER":
                parts.append(_fmt_num(self._next().value))
            elif tok.kind == "DASH":
                self._next()
                parts.append("-")
            else:
                raise GrammarError(f"unexpected token in interval: {tok.value!r}")
        self._expect("RBRACK")
        raw = "".join(parts)
        canon = _canonicalize_interval(raw)
        if canon is None:
            raise GrammarError(f"malformed interval [{raw}]")
        return canon

    def _value(self) -> str:
        tok = self._peek()
        if tok is None:
            raise GrammarError("expected a value")
        if tok.kind == "STRING":
            return str(self._next().value)
        if tok.kind == "NUMBER":
            return _fmt_num(self._next().value)
        if tok.kind == "DASH":
            nxt = self._peek(1)
            if nxt is not None and nxt.kind == "NUMBER":
                self._next()
                return "-" + _fmt_num(self._next().value)
            raise GrammarError("expected a number after '-'")
        if tok.kind == "KEYWORD" and tok.value in ("true", "false"):
            return str(self._next().value)
        raise GrammarError(f"expected a string, number, or boolean value, found {tok.value!r}")


    def _actor_decl(self) -> Dict[str, Any]:
        self._expect("KEYWORD", "actor")
        self._expect("LPAREN")
        actor_id = self._id()
        self._expect("COMMA")
        name = self._string()
        self._expect("COMMA")
        self._expect("LBRACK")
        assigned = [self._id()]
        while self._peek() and self._peek().kind == "COMMA":
            self._next()
            assigned.append(self._id())
        self._expect("RBRACK")
        self._expect("RPAREN")
        return {"_category": "actor", "id": actor_id, "name": name, "assigned_nodes": assigned}


    def _precedence(self) -> Dict[str, Any]:
        self._expect("KEYWORD", "precedence")
        self._expect("LPAREN")
        a = self._id()
        self._expect("COMMA")
        b = self._id()
        self._expect("RPAREN")
        return {"_category": "relationship", "type": "precedence", "A": a, "B": b}

    def _label(self) -> Dict[str, Any]:
        self._expect("KEYWORD", "label")
        self._expect("LPAREN")
        a = self._id()
        self._expect("COMMA")
        b = self._id()
        self._expect("COMMA")
        text = self._string()
        self._expect("RPAREN")
        return {"_category": "relationship", "type": "label", "A": a, "B": b, "str": text}


def _fmt_num(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


_NUMERIC = r"-?\d+(?:\.\d+)?"


def _canonicalize_interval(raw: str) -> Optional[str]:
    s = raw.strip()
    if re.fullmatch(r"-", s):
        return "[-]"
    m = re.fullmatch(rf"({_NUMERIC})-({_NUMERIC})", s)
    if m:
        return f"[{m.group(1)}-{m.group(2)}]"
    m = re.fullmatch(rf"({_NUMERIC})-", s)
    if m:
        return f"[{m.group(1)}-]"
    m = re.fullmatch(rf"-({_NUMERIC})", s)
    if m:
        return f"[-{m.group(1)}]"
    return None


_FULL_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def _require_id(value: str) -> str:
    token = value.strip()
    if not _FULL_ID_RE.match(token):
        raise GrammarError(f"invalid identifier {value!r}")
    return token


def _parse_guard_statement(statement: str) -> Dict[str, Any]:
    head = re.match(r"\s*guard\s*\((.*)\)\s*\Z", statement, re.DOTALL)
    if not head:
        raise GrammarError("guard(...) is malformed")
    split = split_guard_args(head.group(1))
    if split is None:
        raise GrammarError(
            "guard(...) must take exactly three top-level arguments: A, B, <condition>"
        )
    a_raw, b_raw, cond_text = split
    a = _require_id(a_raw)
    b = _require_id(b_raw)
    if not cond_text.strip():
        raise GrammarError("guard(...) condition must not be empty")
    try:
        ast = parse_condition(cond_text)
    except ConditionParseError as exc:
        raise GrammarError(f"invalid guard condition: {exc}")
    return {
        "_category": "relationship",
        "type": "guard",
        "A": a,
        "B": b,
        "condition": cond_text.strip(),
        "condition_ast": ast,
    }


@dataclass
class ParseReport:

    model: Dict[str, Any] = field(default_factory=dict)
    accepted: List[str] = field(default_factory=list)
    rejected: List[Dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected


def parse_process(text: str) -> ParseReport:
    model: Dict[str, Any] = {
        "elements": [],
        "variable_definitions": [],
        "actors": [],
        "relationships": [],
    }
    accepted: List[str] = []
    rejected: List[Dict[str, str]] = []

    cleaned = _strip_decorations(text or "")
    for statement in _split_statements(cleaned):
        try:


            if re.match(r"\s*guard\s*\(", statement):
                parsed = _parse_guard_statement(statement)
            else:
                tokens = _tokenize(statement)
                if not tokens:
                    continue
                parsed = _StatementParser(tokens).parse_statement()
        except GrammarError as exc:
            rejected.append({"statement": statement, "reason": str(exc)})
            continue

        category = parsed.pop("_category")
        if category == "node":
            element = {"id": parsed["id"], "type": parsed["type"]}
            if parsed["type"] == "action":
                element["content"] = parsed.get("content", "")
            if parsed["type"] == "branch":
                element["gateway"] = parsed.get("gateway", "")
            model["elements"].append(element)
        elif category == "variable":
            model["variable_definitions"].append(
                {"X": parsed["X"], "name": parsed["name"], "R_X": parsed["R_X"]}
            )
        elif category == "actor":
            model["actors"].append(
                {
                    "id": parsed["id"],
                    "name": parsed["name"],
                    "assigned_nodes": parsed["assigned_nodes"],
                }
            )
        elif category == "relationship":
            model["relationships"].append({k: v for k, v in parsed.items()})

        accepted.append(statement)

    return ParseReport(model=model, accepted=accepted, rejected=rejected)


def format_rejected(report: ParseReport) -> str:
    if not report.rejected:
        return "All statements conform to the grammar."
    lines = [f"Rejected statements ({len(report.rejected)}):"]
    for item in report.rejected:
        lines.append(f"  - {item['statement']}")
        lines.append(f"      reason: {item['reason']}")
    return "\n".join(lines)
