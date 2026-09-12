import json
import numpy as np
import pytest
from copper_mvp.symbolic_expression import encode_expression,evaluate_expression
from copper_mvp.symbolic_models import SymbolicModel
from copper_mvp.common import WorkbenchError


def sympy_module():
    import sys
    from pathlib import Path
    import os
    directory=Path(os.environ["COPPER_SYMBOLIC_RUNTIME_DIR"])
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
    import sympy
    return sympy


def test_protected_division_is_finite_at_zero_and_matches_expression():
    sp=sympy_module();z0,z1=sp.symbols("z0 z1")
    expression=z0/(sp.Abs(z1)+1)
    tree=encode_expression(expression,["z0","z1"])
    X=np.array([[2.,0.],[2.,-10.],[-3.,10.]])
    expected=X[:,0]/(np.abs(X[:,1])+1)
    np.testing.assert_allclose(evaluate_expression(tree,X),expected)
    assert np.isfinite(evaluate_expression(tree,X)).all()


def test_unapproved_function_and_unprotected_denominator_are_rejected():
    sp=sympy_module();z0=sp.Symbol("z0")
    with pytest.raises(ValueError):encode_expression(sp.exp(z0),["z0"])
    with pytest.raises(ValueError):encode_expression(1/z0,["z0"])


def test_exported_polynomial_and_tanh_survive_json_roundtrip():
    sp=sympy_module();z0,z1=sp.symbols("z0 z1")
    expression=2*z0*z0+sp.tanh(z1)-sp.Rational(1,3)
    tree=json.loads(json.dumps(encode_expression(expression,["z0","z1"])))
    X=np.array([[1.,0.],[-2.,1.],[.1,-100.]])
    np.testing.assert_allclose(evaluate_expression(tree,X),2*X[:,0]**2+np.tanh(X[:,1])-1/3)


def test_unverified_sample_weights_are_not_accepted():
    with pytest.raises(WorkbenchError,match="权重"):
        SymbolicModel(17).fit(np.zeros((20,114)),np.zeros((20,2)),sample_weight=np.ones(20))
