"""Neuro-Symbolic Language (NSL) module (T2.5.1-T2.5.6).

Public API:
    SymbolNeuralBridge - AST encoder + neural decoder + InfoNCE (nsl.bridge)
    NSLGrammar         - symbol types, validation, JSON serialization (nsl.grammar)
    FormalParser       - Pratt parser, SymPy/Lean4 interop (nsl.parser)
    NSLDecoder         - tree-structured AR decoder, cross-cycle state (nsl.decoder)
"""
from .grammar import NSLGrammar, ASTNode, SymbolType
from .bridge import SymbolNeuralBridge
from .parser import FormalParser
from .decoder import NSLDecoder

__all__ = [
    "SymbolNeuralBridge",
    "NSLGrammar",
    "FormalParser",
    "NSLDecoder",
    "ASTNode",
    "SymbolType",
]
