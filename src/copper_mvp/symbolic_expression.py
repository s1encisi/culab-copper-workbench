"""Numeric expression trees exported from a fixed, real-valued PySR operator set."""

import numpy as np

from copper_mvp.common import WorkbenchError


def encode_expression(expression, variables):
    import sympy as sp

    if isinstance(expression, sp.Symbol):
        if str(expression) not in variables:
            raise ValueError("Unexpected symbol in exported expression")
        return {"op": "feature", "index": variables.index(str(expression))}
    if expression.is_Number:
        value = float(expression)
        if not np.isfinite(value):
            raise ValueError("Non-finite constant in expression")
        return {"op": "constant", "value": value}
    operations = {sp.Add: "add", sp.Mul: "multiply", sp.Pow: "power", sp.Abs: "abs", sp.tanh: "tanh"}
    name = operations.get(expression.func)
    if name is None:
        raise ValueError("Unsupported exported expression operation")
    if name == "power":
        exponent = expression.args[1]
        if not exponent.is_Integer:
            raise ValueError("Only integer powers from multiplication and protected division are supported")
        finite_symbols = {s: sp.Symbol(str(s), real=True, finite=True) for s in expression.free_symbols}
        denominator = expression.args[0].xreplace(finite_symbols)
        if exponent < 0 and denominator.is_positive is not True:
            raise ValueError("An exported denominator must be positive on finite real inputs")
    return {"op": name, "args": [encode_expression(a, variables) for a in expression.args]}


def evaluate_expression(node, X):
    operation = node["op"]
    if operation == "constant":
        return np.full(len(X), node["value"], dtype=float)
    if operation == "feature":
        return X[:, node["index"]]
    values = [evaluate_expression(child, X) for child in node["args"]]
    if operation == "add":
        result = np.zeros(len(X))
        for value in values:
            result += value
        return result
    if operation == "multiply":
        result = np.ones(len(X))
        for value in values:
            result *= value
        return result
    if operation == "power":
        return np.power(values[0], values[1])
    if operation == "abs":
        return np.abs(values[0])
    if operation == "tanh":
        return np.tanh(values[0])
    raise WorkbenchError("未知表达式运算", "SYMBOLIC_EXPRESSION")


def expression_features(node):
    if node["op"] == "feature":
        return {node["index"]}
    return set().union(*(expression_features(a) for a in node.get("args", [])))


def structural_signature(node):
    if node["op"] == "constant":
        return ("constant",)
    if node["op"] == "feature":
        return ("feature", node["index"])
    return (node["op"], *(structural_signature(a) for a in node["args"]))
