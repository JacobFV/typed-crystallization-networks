"""Typed program synthesis, exact execution, and synthetic curricula."""
from .types import Type, Value, BOOL, integer, floating, fixed, product, setof
from .graph import Program, Node, Candidate
__version__ = "0.1.0"
