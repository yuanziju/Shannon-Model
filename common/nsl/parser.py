"""Formal parser: Pratt (precedence-climbing) parser + SymPy/Lean4 interop (T2.5.3).

Tokenizes and parses symbolic expressions into :class:`ASTNode` trees and
converts bidirectionally between AST, SymPy expressions, and Lean 4 strings.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .grammar import ASTNode, NSLGrammar, SymbolType

try:
    import sympy
    _HAS_SYMPY = True
except ImportError:  # sympy optional at import time
    sympy = None
    _HAS_SYMPY = False


def _require_sympy():
    if not _HAS_SYMPY:
        raise RuntimeError("sympy is required for SymPy interop")


class FormalParser:
    """Pratt (precedence-climbing) parser with SymPy/Lean4 interop.

    Parses symbolic expressions into :class:`ASTNode` trees and converts
    bidirectionally between AST, SymPy expressions, and Lean 4 strings.
    """

    TOKEN_RE = re.compile(
        r"""
        \s*(?:
            (?P<number>\d+\.\d+|\d+)                       |
            (?P<ident>[A-Za-z_][A-Za-z0-9_?!]*)            |
            (?P<op><=|>=|==|!=|=>|&&|\|\||[-+*/^<>(){}\[\],:])
        )
        """,
        re.VERBOSE,
    )

    def __init__(self, grammar: Optional[NSLGrammar] = None):
        self.grammar = grammar or NSLGrammar()
        self._tokens: List[Tuple[str, str]] = []
        self._pos = 0

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    def tokenize(self, text: str) -> List[Tuple[str, str]]:
        tokens: List[Tuple[str, str]] = []
        pos = 0
        text = text.strip()
        while pos < len(text):
            m = self.TOKEN_RE.match(text, pos)
            if not m or m.end() == pos:
                if text[pos].isspace():
                    pos += 1
                    continue
                raise SyntaxError(f"Unexpected character {text[pos]!r} at {pos}")
            pos = m.end()
            kind = m.lastgroup
            value = m.group(kind)
            # Word operators (and / or / not / ...) are lexed as idents;
            # reclassify them as operator tokens.
            if kind == "ident" and self.grammar.is_operator(value):
                kind = "op"
            tokens.append((kind, value))
        tokens.append(("eof", ""))
        return tokens

    # ------------------------------------------------------------------
    # Pratt parsing
    # ------------------------------------------------------------------
    def _min_rbp(self) -> int:
        """Right-binding-power below every operator precedence (top-level call)."""
        precs = [info[1] for info in self.grammar.operators.values()]
        return (min(precs) - 1) if precs else -1

    def parse(self, text: str) -> ASTNode:
        self._tokens = self.tokenize(text)
        self._pos = 0
        node = self._parse_expression(self._min_rbp())
        if self._peek()[0] != "eof":
            raise SyntaxError(f"Unexpected trailing token {self._peek()}")
        return node

    def _peek(self) -> Tuple[str, str]:
        return self._tokens[self._pos]

    def _advance(self) -> Tuple[str, str]:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _lbp(self, token: Tuple[str, str]):
        """Left binding power of an infix operator token (None if not infix)."""
        kind, value = token
        if kind == "op" and self.grammar.is_operator(value):
            return self.grammar.precedence(value)
        return None

    def _parse_expression(self, rbp: int) -> ASTNode:
        left = self._parse_prefix()
        while True:
            tok = self._peek()
            lbp = self._lbp(tok)
            # Binding-power convention: continue only while the next infix
            # operator binds tighter than the current right-binding-power.
            if lbp is None or lbp <= rbp:
                break
            op = tok[1]
            self._advance()
            # Left-assoc recurses with rbp=lbp (equal precedence binds left);
            # right-assoc with rbp=lbp-1 (equal precedence binds right).
            if self.grammar.associativity(op) == "right":
                next_rbp = lbp - 1
            else:
                next_rbp = lbp
            rhs = self._parse_expression(next_rbp)
            left = ASTNode(SymbolType.OPERATOR, op, children=[left, rhs], arity=2)
        return left

    def _parse_prefix(self) -> ASTNode:
        kind, value = self._peek()
        if kind == "number":
            self._advance()
            return ASTNode(SymbolType.NUMBER, value)
        if kind == "ident":
            self._advance()
            # Function call: ident '(' args ')'
            if self._peek() == ("op", "("):
                self._advance()
                args: List[ASTNode] = []
                if self._peek() != ("op", ")"):
                    args.append(self._parse_expression(0))
                    while self._peek() == ("op", ","):
                        self._advance()
                        args.append(self._parse_expression(0))
                if self._peek() != ("op", ")"):
                    raise SyntaxError("Expected )")
                self._advance()
                return ASTNode(SymbolType.FUNCTION, value,
                               children=args, arity=len(args))
            return ASTNode(self.grammar.classify(value), value)
        if kind == "op" and value == "(":
            self._advance()
            node = self._parse_expression(0)
            if self._peek() != ("op", ")"):
                raise SyntaxError("Expected )")
            self._advance()
            return node
        if kind == "op" and value == "-":
            self._advance()
            operand = self._parse_prefix()
            return ASTNode(SymbolType.OPERATOR, "neg",
                           children=[operand], arity=1)
        if kind == "op" and value == "not":
            self._advance()
            operand = self._parse_prefix()
            return ASTNode(SymbolType.OPERATOR, "not",
                           children=[operand], arity=1)
        raise SyntaxError(f"Unexpected token {(kind, value)}")

    # ------------------------------------------------------------------
    # Unparse (AST -> string)
    # ------------------------------------------------------------------
    def unparse(self, node: ASTNode) -> str:
        if node.type in (SymbolType.NUMBER, SymbolType.VARIABLE,
                         SymbolType.CONSTANT, SymbolType.KEYWORD,
                         SymbolType.SYMBOL, SymbolType.LITERAL):
            return str(node.value)
        if node.type == SymbolType.OPERATOR:
            if node.value == "neg":
                return f"(-{self.unparse(node.children[0])})"
            if len(node.children) == 1:
                return f"({node.value} {self.unparse(node.children[0])})"
            if len(node.children) == 2:
                l, r = node.children
                return f"({self.unparse(l)} {node.value} {self.unparse(r)})"
            return str(node.value)
        if node.type == SymbolType.FUNCTION:
            args = ", ".join(self.unparse(c) for c in node.children)
            return f"{node.value}({args})"
        return str(node.value)

    # ------------------------------------------------------------------
    # SymPy interop
    # ------------------------------------------------------------------
    def to_sympy(self, node: ASTNode):
        _require_sympy()
        return self._ast_to_sympy(node)

    def _ast_to_sympy(self, node: ASTNode):
        op_map = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b,
            "^": lambda a, b: a ** b,
            "==": lambda a, b: sympy.Eq(a, b),
            "!=": lambda a, b: sympy.Ne(a, b),
            "<": lambda a, b: sympy.StrictLessThan(a, b),
            ">": lambda a, b: sympy.StrictLessThan(b, a),
            "<=": lambda a, b: sympy.Le(a, b),
            ">=": lambda a, b: sympy.Ge(a, b),
            "and": lambda a, b: sympy.And(a, b),
            "or": lambda a, b: sympy.Or(a, b),
            "not": lambda a: sympy.Not(a),
        }
        if node.type == SymbolType.NUMBER:
            v = node.value
            return sympy.Integer(int(v)) if "." not in v else sympy.Float(v)
        if node.type in (SymbolType.VARIABLE, SymbolType.CONSTANT):
            if node.value == "pi":
                return sympy.pi
            if node.value == "e":
                return sympy.E
            return sympy.Symbol(node.value)
        if node.type == SymbolType.OPERATOR:
            if node.value == "neg":
                return -self._ast_to_sympy(node.children[0])
            fn = op_map.get(node.value)
            if fn is None:
                raise ValueError(f"Unsupported operator for sympy: {node.value}")
            args = [self._ast_to_sympy(c) for c in node.children]
            return fn(*args)
        if node.type == SymbolType.FUNCTION:
            args = [self._ast_to_sympy(c) for c in node.children]
            sympy_fn = getattr(sympy, node.value, None)
            if sympy_fn is None:
                raise ValueError(f"Unknown sympy function: {node.value}")
            return sympy_fn(*args)
        raise ValueError(f"Cannot convert node type {node.type} to sympy")

    def from_sympy(self, expr) -> ASTNode:
        _require_sympy()
        return self._sympy_to_ast(expr)

    def _sympy_to_ast(self, expr) -> ASTNode:
        if expr.is_Number:
            return ASTNode(SymbolType.NUMBER, str(expr))
        if expr.is_Symbol:
            return ASTNode(SymbolType.VARIABLE, str(expr))
        if expr == sympy.pi:
            return ASTNode(SymbolType.CONSTANT, "pi")
        if expr == sympy.E:
            return ASTNode(SymbolType.CONSTANT, "e")

        rel_map = {
            sympy.Equality: "==", sympy.Unequality: "!=",
            sympy.StrictLessThan: "<", sympy.StrictGreaterThan: ">",
            sympy.LessThan: "<=", sympy.GreaterThan: ">=",
        }
        for rel, op in rel_map.items():
            if isinstance(expr, rel):
                return ASTNode(SymbolType.OPERATOR, op,
                               children=[self._sympy_to_ast(expr.lhs),
                                         self._sympy_to_ast(expr.rhs)],
                               arity=2)
        if isinstance(expr, sympy.And):
            return self._fold_logic("and", expr.args)
        if isinstance(expr, sympy.Or):
            return self._fold_logic("or", expr.args)
        if isinstance(expr, sympy.Not):
            return ASTNode(SymbolType.OPERATOR, "not",
                           children=[self._sympy_to_ast(expr.args[0])], arity=1)

        op_map = {sympy.Add: "+", sympy.Mul: "*", sympy.Pow: "^"}
        for cls, op in op_map.items():
            if isinstance(expr, cls):
                children = [self._sympy_to_ast(a) for a in expr.args]
                node = children[0]
                for c in children[1:]:
                    node = ASTNode(SymbolType.OPERATOR, op,
                                   children=[node, c], arity=2)
                return node

        if expr.func is not None and hasattr(expr, "args"):
            name = getattr(expr.func, "__name__", str(expr.func))
            children = [self._sympy_to_ast(a) for a in expr.args]
            return ASTNode(SymbolType.FUNCTION, name,
                           children=children, arity=len(children))
        return ASTNode(SymbolType.SYMBOL, str(expr))

    def _fold_logic(self, op: str, args) -> ASTNode:
        children = [self._sympy_to_ast(a) for a in args]
        node = children[0]
        for c in children[1:]:
            node = ASTNode(SymbolType.OPERATOR, op,
                           children=[node, c], arity=2)
        return node

    # ------------------------------------------------------------------
    # Lean 4 interop
    # ------------------------------------------------------------------
    LEAN_OP_MAP = {
        "+": "Nat.add",
        "*": "Nat.mul",
        "-": "Nat.sub",
        "/": "Nat.div",
        "^": "Nat.pow",
        "==": "Eq",
        "!=": "Ne",
        "<": "Nat.lt",
        ">": "Nat.gt",
        "<=": "Nat.le",
        ">=": "Nat.ge",
        "and": "And",
        "or": "Or",
    }

    def to_lean(self, node: ASTNode) -> str:
        if node.type in (SymbolType.NUMBER, SymbolType.VARIABLE,
                         SymbolType.CONSTANT, SymbolType.KEYWORD,
                         SymbolType.SYMBOL, SymbolType.LITERAL):
            return str(node.value)
        if node.type == SymbolType.OPERATOR:
            if node.value == "neg":
                return f"(Neg.neg {self.to_lean(node.children[0])})"
            lean_op = self.LEAN_OP_MAP.get(node.value, node.value)
            args = " ".join(self.to_lean(c) for c in node.children)
            return f"({lean_op} {args})"
        if node.type == SymbolType.FUNCTION:
            args = " ".join(self.to_lean(c) for c in node.children)
            return f"({node.value} {args})"
        return str(node.value)

    def from_lean(self, text: str) -> ASTNode:
        """Parse a subset of Lean 4 prefix syntax back into an AST."""
        self._ltokens = self._tokenize_lean(text)
        self._lpos = 0
        return self._parse_lean()

    def _tokenize_lean(self, text: str) -> List[Tuple[str, str]]:
        tokens: List[Tuple[str, str]] = []
        i, n = 0, len(text)
        while i < n:
            c = text[i]
            if c.isspace():
                i += 1
                continue
            if c in "()":
                tokens.append(("paren", c))
                i += 1
                continue
            if c.isdigit():
                j = i
                while j < n and (text[j].isdigit() or text[j] == "."):
                    j += 1
                tokens.append(("number", text[i:j]))
                i = j
                continue
            if c.isalpha() or c == "_":
                j = i
                while j < n and (text[j].isalnum() or text[j] in "_.'"):
                    j += 1
                tokens.append(("ident", text[i:j]))
                i = j
                continue
            j = i
            while j < n and not text[j].isspace() and text[j] not in "()":
                j += 1
            tokens.append(("op", text[i:j]))
            i = j
        tokens.append(("eof", ""))
        return tokens

    def _parse_lean(self) -> ASTNode:
        kind, value = self._ltokens[self._lpos]
        if kind == "number":
            self._lpos += 1
            return ASTNode(SymbolType.NUMBER, value)
        if kind == "ident":
            self._lpos += 1
            return ASTNode(self.grammar.classify(value), value)
        if kind == "paren" and value == "(":
            self._lpos += 1
            head = self._ltokens[self._lpos]
            self._lpos += 1
            args: List[ASTNode] = []
            while self._ltokens[self._lpos] != ("paren", ")"):
                if self._ltokens[self._lpos][0] == "eof":
                    raise SyntaxError("Unterminated Lean expression")
                args.append(self._parse_lean())
            self._lpos += 1  # consume ')'
            inv = {v: k for k, v in self.LEAN_OP_MAP.items()}
            if head[1] in inv:
                return ASTNode(SymbolType.OPERATOR, inv[head[1]],
                               children=args, arity=len(args))
            if head[1] == "Neg.neg":
                return ASTNode(SymbolType.OPERATOR, "neg",
                               children=args, arity=1)
            return ASTNode(SymbolType.FUNCTION, head[1],
                           children=args, arity=len(args))
        raise SyntaxError(f"Unexpected Lean token {(kind, value)}")
