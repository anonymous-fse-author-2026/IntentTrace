from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

REL_OPS = {"=", "!=", "<", "<=", ">", ">="}
ADD_OPS = {"+", "-"}
MUL_OPS = {"*", "/"}
KEYWORDS = {"AND", "OR", "NOT", "TRUE", "FALSE"}


class ConditionParseError(ValueError):
    pass


@dataclass
class Token:
    kind: str
    value: Any


def _parse_number(raw: str) -> Any:
    return float(raw) if "." in raw else int(raw)


def _tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    i, n = 0, len(text)

    while i < n:
        ch = text[i]

        if ch.isspace():
            i += 1
            continue

        if ch == "(":
            tokens.append(Token("LPAREN", "("))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token("RPAREN", ")"))
            i += 1
            continue

        two = text[i:i + 2]
        if two in {"<=", ">=", "!="}:
            tokens.append(Token("OP", two))
            i += 2
            continue
        if two == "==":
            tokens.append(Token("OP", "="))
            i += 2
            continue

        if ch in {"=", "<", ">"}:
            tokens.append(Token("OP", ch))
            i += 1
            continue

        if ch in {"+", "-", "*", "/"}:
            if ch == "-" and (
                not tokens
                or tokens[-1].kind in {"OP", "LPAREN"}
                or (tokens[-1].kind == "KEYWORD" and tokens[-1].value in {"AND", "OR", "NOT"})
            ):
                j = i + 1
                if j < n and (text[j].isdigit() or text[j] == "."):
                    start, i = i, j
                    while i < n and (text[i].isdigit() or text[i] == "."):
                        i += 1
                    tokens.append(Token("NUMBER", _parse_number(text[start:i])))
                    continue
            tokens.append(Token("OP", ch))
            i += 1
            continue

        if ch in {"'", '"'}:
            quote, i = ch, i + 1
            buf: List[str] = []
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    buf.append(text[i + 1])
                    i += 2
                    continue
                buf.append(text[i])
                i += 1
            if i >= n:
                raise ConditionParseError(f"Unterminated string literal in: {text!r}")
            i += 1
            tokens.append(Token("STRING", "".join(buf)))
            continue

        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            start = i
            while i < n and (text[i].isdigit() or text[i] == "."):
                i += 1
            tokens.append(Token("NUMBER", _parse_number(text[start:i])))
            continue

        if ch.isalpha() or ch == "_":
            start = i
            while i < n and (text[i].isalnum() or text[i] == "_"):
                i += 1
            raw = text[start:i]
            upper = raw.upper()
            if upper in {"TRUE", "FALSE"}:
                tokens.append(Token("BOOL", upper == "TRUE"))
            elif upper in KEYWORDS:
                tokens.append(Token("KEYWORD", upper))
            else:
                tokens.append(Token("IDENT", raw))
            continue

        raise ConditionParseError(f"Unexpected character {ch!r} at position {i} in: {text!r}")

    return tokens


class _Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _accept(self, kind: str, value: Any = None) -> Optional[Token]:
        tok = self._peek()
        if tok is None or tok.kind != kind:
            return None
        if value is not None and tok.value != value:
            return None
        self.pos += 1
        return tok

    def parse(self) -> Tuple[Any, ...]:
        node = self._condition()
        if self.pos != len(self.tokens):
            raise ConditionParseError(f"Unexpected token after expression: {self.tokens[self.pos]}")
        return node

    def _binary(self, sub, keyword: str):
        node = sub()
        while self._accept("KEYWORD", keyword):
            node = (keyword, node, sub())
        return node

    def _condition(self):
        return self._binary(self._b_term, "OR")

    def _b_term(self):
        return self._binary(self._b_factor, "AND")

    def _b_factor(self):
        if self._accept("KEYWORD", "NOT"):
            return ("NOT", self._b_factor())
        return self._rel_condition()

    def _rel_condition(self):
        left = self._arith(self._a_term, ADD_OPS, "ADD")
        tok = self._peek()
        if tok and tok.kind == "OP" and tok.value in REL_OPS:
            self.pos += 1
            return ("REL", tok.value, left, self._arith(self._a_term, ADD_OPS, "ADD"))
        return left

    def _arith(self, sub, ops: set, label: str):
        node = sub()
        while True:
            tok = self._peek()
            if tok and tok.kind == "OP" and tok.value in ops:
                self.pos += 1
                node = (label, tok.value, node, sub())
                continue
            return node

    def _a_term(self):
        return self._arith(self._a_factor, MUL_OPS, "MUL")

    def _a_factor(self):
        if self._accept("LPAREN"):
            inner = self._condition()
            if not self._accept("RPAREN"):
                raise ConditionParseError("Missing closing parenthesis")
            return inner
        tok = self._peek()
        if tok is None:
            raise ConditionParseError("Unexpected end of expression")
        self.pos += 1
        if tok.kind == "NUMBER" or tok.kind == "STRING":
            return ("VAL", tok.value)
        if tok.kind == "BOOL":
            return ("VAL", bool(tok.value))
        if tok.kind == "IDENT":
            return ("VAR", tok.value)
        raise ConditionParseError(f"Unexpected token: {tok}")


def parse_condition(text: str) -> Tuple[Any, ...]:
    tokens = _tokenize(text)
    if not tokens:
        raise ConditionParseError("Empty condition")
    ast = _Parser(tokens).parse()

    if isinstance(ast, tuple) and len(ast) == 2 and ast[0] == "VAL" and isinstance(ast[1], str):
        inner = ast[1].strip()
        if inner and inner != text.strip():
            try:
                inner_ast = _Parser(_tokenize(inner)).parse()
            except ConditionParseError:
                return ast
            if inner_ast[0] in {"REL", "AND", "OR", "NOT", "VAR", "ADD", "MUL"}:
                return inner_ast

    return ast


def is_valid(text: str) -> bool:
    try:
        parse_condition(text)
        return True
    except ConditionParseError:
        return False


def vars_of(ast: Tuple[Any, ...]) -> Set[str]:
    out: Set[str] = set()
    _collect_vars(ast, out)
    return out


def _collect_vars(ast: Tuple[Any, ...], out: Set[str]) -> None:
    if not isinstance(ast, tuple):
        return
    head = ast[0]
    if head == "VAR":
        out.add(ast[1])
    elif head == "NOT":
        _collect_vars(ast[1], out)
    elif head in {"AND", "OR"}:
        _collect_vars(ast[1], out)
        _collect_vars(ast[2], out)
    elif head in {"REL", "ADD", "MUL"}:
        _collect_vars(ast[2], out)
        _collect_vars(ast[3], out)


def split_guard_args(inside: str) -> Optional[Tuple[str, str, str]]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    in_str: Optional[str] = None
    i = 0

    while i < len(inside):
        ch = inside[i]
        if in_str:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(inside):
                buf.append(inside[i + 1])
                i += 2
                continue
            if ch == in_str:
                in_str = None
        elif ch in {"'", '"'}:
            in_str = ch
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0 and len(parts) < 2:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1

    parts.append("".join(buf).strip())
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else None
