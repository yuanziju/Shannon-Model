"""Neuro-Symbolic Language (NSL) grammar (T2.5.2).

Defines the symbolic vocabulary, AST node type, and grammar rules covering
math / logic / code symbols, with validation and JSON serialization
(neuro_grammar.json export).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


class SymbolType(str, Enum):
    """Symbolic node types for the Neuro-Symbolic Language."""

    NUMBER = "number"
    VARIABLE = "variable"
    CONSTANT = "constant"
    OPERATOR = "operator"
    FUNCTION = "function"
    SYMBOL = "symbol"            # generic symbolic atom
    KEYWORD = "keyword"
    LITERAL = "literal"
    LIST = "list"
    APPLICATION = "application"  # function application node
    BINDING = "binding"          # let / lambda binding
    TUPLE = "tuple"


@dataclass
class ASTNode:
    """A node in the symbolic AST.

    Attributes:
        type: the :class:`SymbolType` of this node.
        value: the symbol string (operator / function name / identifier / literal).
        children: ordered child AST nodes.
        arity: declared arity (for operators / functions).
        attrs: optional metadata (e.g. type annotation, source span).
    """

    type: SymbolType
    value: str
    children: List["ASTNode"] = field(default_factory=list)
    arity: int = 0
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "value": self.value,
            "arity": self.arity,
            "children": [c.to_dict() for c in self.children],
            "attrs": self.attrs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ASTNode":
        return cls(
            type=SymbolType(d["type"]),
            value=d["value"],
            arity=d.get("arity", 0),
            children=[cls.from_dict(c) for c in d.get("children", [])],
            attrs=d.get("attrs", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "ASTNode":
        return cls.from_dict(json.loads(text))

    def walk(self) -> Iterator["ASTNode"]:
        yield self
        for c in self.children:
            yield from c.walk()

    def __repr__(self) -> str:
        if self.children:
            return f"{self.value}({', '.join(map(repr, self.children))})"
        return str(self.value)


class NSLGrammar:
    """Neuro-Symbolic Language grammar.

    Holds the symbol vocabulary, operator precedence / arity, and validation
    rules covering math / logic / code symbols (T2.5.2). Supports JSON
    serialization for export (neuro_grammar.json).
    """

    DEFAULT_OPERATORS: Dict[str, tuple] = {
        # value -> (arity, precedence, associativity)
        "+": (2, 1, "left"),
        "-": (2, 1, "left"),
        "*": (2, 2, "left"),
        "/": (2, 2, "left"),
        "^": (2, 3, "right"),
        "==": (2, 0, "none"),
        "!=": (2, 0, "none"),
        "<": (2, 0, "left"),
        ">": (2, 0, "left"),
        "<=": (2, 0, "left"),
        ">=": (2, 0, "left"),
        "and": (2, -1, "left"),
        "or": (2, -2, "left"),
        "not": (1, 4, "right"),
        "=>": (2, -3, "right"),
        "neg": (1, 5, "right"),  # unary negation (surface syntax: prefix "-")
    }

    def __init__(self,
                 operators: Optional[Dict[str, tuple]] = None,
                 functions: Optional[set] = None,
                 constants: Optional[set] = None,
                 keywords: Optional[set] = None,
                 extra_symbols: Optional[set] = None):
        self.operators = dict(operators) if operators is not None \
            else dict(self.DEFAULT_OPERATORS)
        self.functions = set(functions or {
            "sin", "cos", "tan", "exp", "log", "sqrt", "abs",
            "min", "max", "sum", "len", "map", "filter", "reduce",
        })
        self.constants = set(constants or {"pi", "e", "True", "False", "nil"})
        self.keywords = set(keywords or {
            "let", "in", "fun", "if", "then", "else", "forall", "exists",
            "lambda", "def", "return", "match", "with",
        })
        self.extra_symbols = set(extra_symbols or set())
        self._rebuild_index()

    # ---- vocabulary indexing ----
    def _rebuild_index(self):
        self._symbol_index: Dict[str, int] = {}
        for i, s in enumerate(self.all_symbols()):
            self._symbol_index[s] = i + 1  # 0 reserved for unknown/pad

    def all_symbols(self) -> Iterator[str]:
        for op in self.operators:
            yield op
        for f in self.functions:
            yield f
        for c in self.constants:
            yield c
        for k in self.keywords:
            yield k
        for s in self.extra_symbols:
            yield s

    def vocab_size(self) -> int:
        return len(self._symbol_index) + 1  # +1 for unknown/pad id 0

    def symbol_id(self, symbol: str) -> int:
        return self._symbol_index.get(symbol, 0)

    def id_to_symbol(self, idx: int) -> str:
        if idx == 0:
            return ""
        symbols = list(self.all_symbols())
        if 1 <= idx <= len(symbols):
            return symbols[idx - 1]
        return ""

    # ---- predicates ----
    def is_operator(self, s: str) -> bool:
        return s in self.operators

    def is_function(self, s: str) -> bool:
        return s in self.functions

    def is_constant(self, s: str) -> bool:
        return s in self.constants

    def is_keyword(self, s: str) -> bool:
        return s in self.keywords

    def operator_info(self, op: str) -> Optional[tuple]:
        return self.operators.get(op)

    def precedence(self, op: str) -> Optional[int]:
        info = self.operators.get(op)
        return info[1] if info else None

    def arity(self, op: str) -> Optional[int]:
        info = self.operators.get(op)
        return info[0] if info else None

    def associativity(self, op: str) -> Optional[str]:
        info = self.operators.get(op)
        return info[2] if info else None

    def classify(self, token: str) -> SymbolType:
        if self.is_operator(token):
            return SymbolType.OPERATOR
        if self.is_function(token):
            return SymbolType.FUNCTION
        if self.is_constant(token):
            return SymbolType.CONSTANT
        if self.is_keyword(token):
            return SymbolType.KEYWORD
        try:
            float(token)
            return SymbolType.NUMBER
        except (ValueError, TypeError):
            pass
        if isinstance(token, str) and token[:1].isalpha() or token[:1] == "_":
            return SymbolType.VARIABLE
        return SymbolType.SYMBOL

    # ---- validation ----
    def validate(self, node: ASTNode) -> bool:
        """Recursively validate an AST against the grammar."""
        if not isinstance(node, ASTNode):
            return False
        try:
            SymbolType(node.type)
        except ValueError:
            return False
        if node.type == SymbolType.OPERATOR:
            info = self.operators.get(node.value)
            if info is None:
                return False
            if len(node.children) != info[0]:
                return False
        elif node.type == SymbolType.FUNCTION:
            if node.value not in self.functions:
                return False
        elif node.type == SymbolType.KEYWORD:
            if node.value not in self.keywords:
                return False
        for c in node.children:
            if not self.validate(c):
                return False
        return True

    # ---- serialization ----
    def to_json(self, path: Optional[str] = None) -> str:
        data = {
            "operators": {k: list(v) for k, v in self.operators.items()},
            "functions": sorted(self.functions),
            "constants": sorted(self.constants),
            "keywords": sorted(self.keywords),
            "extra_symbols": sorted(self.extra_symbols),
        }
        text = json.dumps(data, indent=2, ensure_ascii=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    @classmethod
    def from_json(cls, path_or_text: str) -> "NSLGrammar":
        text = path_or_text
        if not path_or_text.lstrip().startswith("{"):
            with open(path_or_text, "r", encoding="utf-8") as f:
                text = f.read()
        data = json.loads(text)
        ops = {k: tuple(v) for k, v in data.get("operators", {}).items()}
        return cls(
            operators=ops,
            functions=set(data.get("functions", [])),
            constants=set(data.get("constants", [])),
            keywords=set(data.get("keywords", [])),
            extra_symbols=set(data.get("extra_symbols", [])),
        )
