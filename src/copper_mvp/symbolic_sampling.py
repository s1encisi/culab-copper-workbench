"""Run fixed-budget expression search and verify exported training losses."""
from pathlib import Path
from collections import Counter
import time,warnings
import numpy as np
from copper_mvp.common import write_json
from copper_mvp.symbolic_expression import encode_expression,evaluate_expression,expression_features,structural_signature


def fit_symbolic_models(X,y,settings,seed,directory,runtime_lock):
    import sympy as sp
    from pysr import PySRRegressor
    directory=Path(directory)
    variables=[f"z{i}" for i in range(X.shape[1])]
    results,notices=[],[]
    started=time.perf_counter()
    for target,name in enumerate(("cu","as")):
        write_json(directory/"state.json",{"status":"searching","target":name,"elapsed_seconds":time.perf_counter()-started})
        parameters=dict(settings)
        parameters["constraints"]={k:tuple(v) for k,v in parameters["constraints"].items()}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model=PySRRegressor(**parameters,random_state=(seed+target)%2**31,
                extra_sympy_mappings={"safe_div":lambda x,y:x/(sp.Abs(y)+1)},
                output_directory=str(directory/"search"),run_id=name)
            model.fit(X,y[:,target],variable_names=variables)
        notices.extend({"category":w.category.__name__,"message":str(w.message)[:400]} for w in captured)
        best=model.get_best()
        expression=best["sympy_format"]
        encoded=encode_expression(expression,variables)
        predicted=evaluate_expression(encoded,X)
        np.testing.assert_allclose(predicted,model.predict(X),rtol=1e-9,atol=1e-10)
        loss=float(np.mean((predicted-y[:,target])**2))
        np.testing.assert_allclose(loss,float(best["loss"]),rtol=1e-6,atol=1e-9)
        item={"target":name,"expression":str(expression),"julia_equation":str(best["equation"]),
              "ast":encoded,"complexity":int(best["complexity"]),
              "training_mse":loss,"native_training_loss":float(best["loss"]),
              "selected_features":sorted(expression_features(encoded)),
              "structure":structural_signature(encoded),
              "selection_policy":"training-only PySR best score within its loss threshold",
              "hall_of_fame_rows":len(model.equations_)}
        results.append(item)
        table=model.equations_.drop(columns=["lambda_format","sympy_format"],errors="ignore")
        table.to_csv(directory/(name+"_equations.csv"),index=False)
        write_json(directory/(name+"_equation.json"),item)
    counts=Counter((n["category"],n["message"]) for n in notices)
    return {"models":results,"runtime":runtime_lock,"settings":settings,"seed":seed,
            "fit_warnings":[{"category":k[0],"message":k[1],"count":n} for k,n in counts.items()],
            "elapsed_seconds":time.perf_counter()-started}
