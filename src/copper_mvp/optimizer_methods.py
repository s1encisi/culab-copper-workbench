"""G6c fixed pymoo methods, with explicit constraint and final-batch adapters.

MOEA/D and GDE3 adapters follow pymoo 0.6.2's neighborhood replacement
and parent/trial competition respectively. Raw F/G remain in the shared ledger.
"""

from __future__ import annotations

import sys
from importlib import metadata

import numpy as np
import pymoo
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.gde3 import GDE3, get_relation
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.omni import OmniOptimizer
from pymoo.algorithms.moo.rvea import RVEA, APDSurvival
from pymoo.core.population import Population
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from scipy.spatial.distance import cdist

from copper_mvp.common import PROJECT_ROOT, WorkbenchError

EXTENDED_OPTIMIZERS = ("NSGA-III", "MOEA-D", "RVEA", "AGE-MOEA", "C-TAEA", "GDE3", "Omni-Optimizer", "HypE")
METHOD_VERSION = "g6c.fixed.v1"


def optimizer_method_spec(name):
    descriptions = {
        "HypE": (
            "O15",
            "k-dependent hypervolume allocation for mating and split-front deletion",
            "adapted_feasibility_first",
        ),
        "NSGA-III": ("O01", "reference-direction survival after non-dominated sorting", "native_feasibility_first"),
        "MOEA-D": (
            "O03",
            "Tchebycheff decomposition and neighborhood replacement",
            "adapted_feasibility_first_neighbor_replacement",
        ),
        "RVEA": ("O06", "angle-penalized distance and adaptive reference vectors", "native_feasibility_first"),
        "AGE-MOEA": ("O07", "estimated front geometry and survival distance", "native_feasibility_first"),
        "C-TAEA": (
            "O08",
            "convergence and diversity archives with restricted mating",
            "native_two_archive_constraint_handling",
        ),
        "GDE3": (
            "O10",
            "differential evolution, parent-trial dominance and crowding truncation",
            "native_constraint_dominance",
        ),
        "Omni-Optimizer": (
            "O16",
            "loose dominance, objective/decision crowding and neighbor mating",
            "native_feasibility_first",
        ),
    }
    design_id, mechanism, constraint_handling = descriptions[name]
    parameters = {
        "population": 64,
        "objective_scaling": "fixed_shared_metric_scale",
        "variables": "continuous",
        "objectives": 2,
        "sample_reuse": "same_charged_initial_population",
        "gradient": False,
    }
    if name == "GDE3":
        parameters.update(
            variant="DE/rand/1/bin",
            CR=0.5,
            F=0.5,
            gamma=1e-4,
            repair="bounce-back",
            last_batch="sample_parent_indices_without_replacement_keep_unvisited_parents",
        )
    else:
        parameters.update(
            crossover="SBX",
            crossover_probability=0.9,
            crossover_eta=15,
            mutation="PM",
            mutation_eta=20,
            mutation_probability_per_variable="1/n_var",
        )
    if name in ("NSGA-III", "MOEA-D", "RVEA", "C-TAEA"):
        parameters.update(reference_directions="64_uniform_directions_in_two_objectives")
    if name == "MOEA-D":
        parameters.update(
            neighbors=20,
            neighbor_mating_probability=0.9,
            decomposition="Tchebicheff",
            ideal_point="feasible_observations_only_after_first_feasible_point",
        )
    if name == "RVEA":
        parameters.update(
            alpha=2.0,
            adaptation_frequency=0.1,
            progress="initial_generation_plus_ceil_search_budget_over_64",
            zero_objective_span="retain_previous_reference_vectors",
        )
    if name == "Omni-Optimizer":
        parameters.update(delta=0.001, objective_crowding=True, decision_crowding=True)
    if name == "HypE":
        parameters.update(
            hypervolume="exact_2D_cell_integral",
            reference=[1.1, 1.1],
            mating_k="population_size",
            survival_k="current_front_size_minus_slots",
            ties="seeded_random",
        )
    dependencies = {"pymoo": "0.6.2"}
    if name == "AGE-MOEA":
        dependencies.update(numba="0.61.2", llvmlite="0.44.0")
    return {
        "optimizer_id": name,
        "display_name": "MOEA/D" if name == "MOEA-D" else name,
        "design_id": design_id,
        "method_version": "g6e.hype.v1" if name == "HypE" else METHOD_VERSION,
        "mechanism": mechanism,
        "constraint_handling": constraint_handling,
        "parameters": parameters,
        "dependencies": dependencies,
    }


def require_optimizer_dependencies(name):
    if pymoo.__version__ != "0.6.2":
        raise WorkbenchError("扩展优化方法需要 pymoo 0.6.2", "OPTIMIZER_DEPENDENCY")
    if name == "AGE-MOEA":
        target = PROJECT_ROOT / "runs" / "dependencies" / "numba-0.61.2"
        if target.is_dir() and str(target) not in sys.path:
            sys.path.insert(0, str(target))
        for package, version in (("numba", "0.61.2"), ("llvmlite", "0.44.0")):
            try:
                actual = metadata.version(package)
            except metadata.PackageNotFoundError:
                raise WorkbenchError(
                    "AGE-MOEA 依赖未安装，请运行 scripts/setup_optimizer_methods.ps1", "OPTIMIZER_DEPENDENCY"
                ) from None
            if actual != version:
                raise WorkbenchError("扩展优化依赖版本不一致: " + package, "OPTIMIZER_DEPENDENCY")


class FeasibleMOEAD(MOEAD):
    """Keep native MOEA/D mating; compare actual CV before scalarized objectives."""

    def _setup(self, problem, **kwargs):
        self.pop_size = len(self.ref_dirs)
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind="quicksort")[
            :, : self.n_neighbors
        ]

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)
        feasible = self.pop.get("CV")[:, 0] <= 0
        self.feasible_ideal = self.pop[feasible].get("F").min(axis=0) if feasible.any() else None
        if self.feasible_ideal is not None:
            self.ideal = self.feasible_ideal.copy()

    def _replace(self, k, off):
        off_cv = float(off.CV[0])
        if off_cv <= 0:
            self.feasible_ideal = (
                off.F.copy() if self.feasible_ideal is None else np.minimum(self.feasible_ideal, off.F)
            )
        if self.feasible_ideal is not None:
            self.ideal = self.feasible_ideal.copy()
        neighbors = self.neighbors[k]
        parent_cv = self.pop[neighbors].get("CV")[:, 0]
        parent_f = self.decomposition.do(
            self.pop[neighbors].get("F"), weights=self.ref_dirs[neighbors], ideal_point=self.ideal
        )
        trial_f = self.decomposition.do(off.F[None, :], weights=self.ref_dirs[neighbors], ideal_point=self.ideal)
        better = (off_cv < parent_cv) | ((off_cv <= 0) & (parent_cv <= 0) & (trial_f < parent_f))
        self.pop[neighbors[better]] = off


class StableAPDSurvival(APDSurvival):
    """A zero objective span supplies no direction adaptation information."""

    def adapt(self):
        if self.nadir is not None and np.all(self.nadir - self.ideal > 0):
            super().adapt()


class BudgetedGDE3(GDE3):
    """A partial final generation retains unvisited parents and their indices."""

    def _infill(self):
        limit = self.n_offsprings
        self.n_offsprings = self.pop_size
        infills = super()._infill()
        indices = np.arange(len(self.pop))
        if limit < len(infills):
            indices = np.sort(self.random_state.choice(indices, size=limit, replace=False))
        self.trial_parent_indices = indices
        return infills[indices]

    def _advance(self, infills=None, **kwargs):
        selected = set(self.trial_parent_indices.tolist())
        survivors = [parent for i, parent in enumerate(self.pop) if i not in selected]
        for parent_index, off in zip(self.trial_parent_indices, infills, strict=True):
            parent = self.pop[parent_index]
            relation = get_relation(parent, off)
            if relation == 0:
                survivors.extend((parent, off))
            else:
                survivors.append(off if relation == -1 else parent)
        self.pop = self.survival.do(
            self.problem, Population.create(*survivors), n_survive=self.pop_size, random_state=self.random_state
        )


def make_extended_optimizer(name, population, n_var):
    require_optimizer_dependencies(name)
    u = np.linspace(0, 1, 64)
    directions = np.column_stack((u, 1 - u))
    common = {
        "sampling": population,
        "crossover": SBX(prob=0.9, eta=15),
        "mutation": PM(prob=1.0, prob_var=1 / n_var, eta=20),
    }
    if name == "HypE":
        from copper_mvp.hype import HypE

        return HypE(reference=[1.1, 1.1], pop_size=64, eliminate_duplicates=True, **common)
    if name == "NSGA-III":
        return NSGA3(ref_dirs=directions, pop_size=64, eliminate_duplicates=True, **common)
    if name == "MOEA-D":
        return FeasibleMOEAD(
            ref_dirs=directions, n_neighbors=20, prob_neighbor_mating=0.9, decomposition=Tchebicheff(), **common
        )
    if name == "RVEA":
        return RVEA(
            ref_dirs=directions,
            pop_size=64,
            alpha=2.0,
            adapt_freq=0.1,
            survival=StableAPDSurvival(directions, alpha=2.0),
            eliminate_duplicates=True,
            **common,
        )
    if name == "AGE-MOEA":
        from pymoo.algorithms.moo.age import AGEMOEA

        return AGEMOEA(pop_size=64, eliminate_duplicates=True, **common)
    if name == "C-TAEA":
        return CTAEA(ref_dirs=directions, eliminate_duplicates=True, **common)
    if name == "GDE3":
        return BudgetedGDE3(
            pop_size=64,
            sampling=population,
            variant="DE/rand/1/bin",
            CR=0.5,
            F=0.5,
            gamma=1e-4,
            de_repair="bounce-back",
        )
    if name == "Omni-Optimizer":
        return OmniOptimizer(
            pop_size=64, delta=0.001, obj_crowding=True, var_crowding=True, eliminate_duplicates=True, **common
        )
    raise WorkbenchError("没有该扩展优化器", "OPTIMIZER_NOT_FOUND")
