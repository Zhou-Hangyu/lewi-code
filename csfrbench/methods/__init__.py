"""The methods of the benchmark: baselines, LeWi and the graph-informed references."""
from . import bimt, dense, fiedler, graph_oracle, l0, lewi, positions, rigl, weight_penalty  # noqa: F401  (registration)
from .base import METHODS, Method, get_method, register

__all__ = ["METHODS", "Method", "get_method", "register"]
