"""Optional Platypus methods using the shared raw evaluator and exact tail budget."""
from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
from importlib import metadata
import math
import random
import sys
from time import perf_counter

import numpy as np

from copper_mvp.common import PROJECT_ROOT, WorkbenchError

PLATYPUS_OPTIMIZERS = ("IBEA", "Epsilon-MOEA", "SMPSO", "PAES", "PESA-II", "MO-CMA-ES")
_ACTIVE_RANDOM = ContextVar("platypus_run_random", default=None)


class _EndBudget(Exception):
    pass


class _EndTime(Exception):
    pass


class _ScopedRandom:
    def __getattr__(self, name):
        active = _ACTIVE_RANDOM.get()
        if active is None:
            return getattr(random, name)
        rng, deadline = active
        if perf_counter() >= deadline:
            raise _EndTime()
        return getattr(rng, name)


_RANDOM_PROXY = _ScopedRandom()


def load_platypus():
    target = PROJECT_ROOT / "runs" / "dependencies" / "platypus-1.4.1"
    if target.is_dir() and str(target) not in sys.path:
        sys.path.insert(0, str(target))
    try:
        version = metadata.version("Platypus-Opt")
    except metadata.PackageNotFoundError:
        raise WorkbenchError("请运行 scripts/setup_platypus_methods.ps1 安装可选优化依赖", "OPTIMIZER_DEPENDENCY") from None
    if version != "1.4.1":
        raise WorkbenchError("Platypus 版本与固定协议不同", "OPTIMIZER_DEPENDENCY")
    import platypus
    # Only the package's module aliases change; Python's global RNG is untouched.
    for name, module in tuple(sys.modules.items()):
        if name.startswith("platypus.") and getattr(module, "random", None) is random:
            module.random = _RANDOM_PROXY
    return platypus


def platypus_method_spec(name):
    definitions = {
        "IBEA": ("O05", "binary hypervolume indicator fitness and iterative fitness removal", "adapted_constraint_first_fitness_comparator"),
        "Epsilon-MOEA": ("O09", "epsilon-box archive and population/archive mating", "native_constraint_dominance"),
        "SMPSO": ("O11", "constricted velocity, leader archive and polynomial perturbation", "native_constraint_dominance"),
        "PAES": ("O12", "single-parent mutation and adaptive-grid archive", "native_constraint_dominance"),
        "PESA-II": ("O13", "adaptive-grid region mating selection", "native_constraint_dominance"),
        "MO-CMA-ES": ("O14", "covariance adaptation with nondominated rank/crowding survival", "native_constraint_dominance"),
    }
    design, mechanism, constraints = definitions[name]
    parameters = {"initial_samples": 64, "objective_scaling": "fixed_shared_metric_scale", "decision_scaling": "unit_box",
                  "tail_budget": "evaluate_native_order_prefix_then_finish_without_incomplete_state_update", "randomness": "per_run_context_local_python_Random"}
    if name in ("IBEA", "Epsilon-MOEA", "PESA-II"):
        parameters.update(population=64, crossover="SBX", crossover_probability=0.9, crossover_eta=15,
                          mutation="PM", mutation_eta=20, mutation_probability_per_variable="1/n_var")
    if name == "IBEA":
        parameters.update(kappa=0.05, indicator_reference=2.0, indicator_objectives="shared_scaled_raw_objectives",
                          plateau_fitness="equal_fitness_when_all_pairwise_indicators_are_zero")
    if name == "Epsilon-MOEA":
        parameters.update(epsilons=[0.01, 0.01])
    if name == "SMPSO":
        parameters.update(swarm_size=64, leader_capacity=64, velocity="native_constriction_and_half_box_limit", mutation="PM", mutation_eta=20)
    if name in ("PAES", "PESA-II"):
        parameters.update(grid_divisions=8, archive_capacity=64)
    if name == "PAES":
        parameters.update(parents=1, initial_parent="seeded_choice_from_shared_nondominated_samples", archive_initialization="all_shared_initial_samples", mutation="PM", mutation_eta=20)
    if name == "MO-CMA-ES":
        parameters.update(offspring_size=64, sigma=0.2, indicator="crowding", covariance="full",
                          initialization="distribution_update_from_shared_population_centered_on_its_mean")
    return {"optimizer_id": name, "display_name": name, "design_id": design, "method_version": "g6d.fixed.v2",
            "mechanism": mechanism, "constraint_handling": constraints, "parameters": parameters,
            "dependencies": {"Platypus-Opt": "1.4.1"}, "variables": "continuous", "objectives": 2,
            "inequality_constraints": True, "package": "Platypus-Opt", "package_version": "1.4.1",
            "status": "registered", "execution_authorized": False}


def _native_components(p):
    class ConstraintFitness(p.Dominance):
        def compare(self, a, b):
            if a.constraint_violation != b.constraint_violation:
                return -1 if a.constraint_violation < b.constraint_violation else 1
            return (a.fitness > b.fitness) - (a.fitness < b.fitness)

    class FixedIndicator(p.HypervolumeFitnessEvaluator):
        # F has already been scaled by the common pilot, including plateau cases.
        def evaluate(self, solutions):
            if not solutions:
                return
            for solution in solutions:
                solution.normalized_objectives = list(solution.objectives)
            self.fitcomp = [[self.calculate_indicator(a, b) for b in solutions] for a in solutions]
            self.max_fitness = max(abs(v) for row in self.fitcomp for v in row) or 1.0
            for i, solution in enumerate(solutions):
                solution.fitness = sum(math.exp(-self.fitcomp[j][i] / self.max_fitness / self.kappa)
                                       for j in range(len(solutions)) if i != j)

    class SharedGenerator(p.Generator):
        def __init__(self, initial):
            self.initial = iter(initial)
        def generate(self, problem):
            return deepcopy(next(self.initial))

    class SharedCMAES(p.CMAES):
        def sample(self):
            if self.iteration == 0:
                self.xmean = np.mean([s.variables[:] for s in self.shared_initial], axis=0).tolist()
                self.iteration += 1
                return deepcopy(self.shared_initial)
            return super().sample()

    return ConstraintFitness, FixedIndicator, SharedGenerator, SharedCMAES


class PlatypusAdapter:
    """Small driver matching the existing setup/next interface; budgets stay external."""
    def __init__(self, name, population, scale):
        self.name, self.initial, self.scale = name, population, np.asarray(scale)
        self.n_offsprings = 64
        self.initialized = False
        self.stop_reason = None
        self.deadline = math.inf

    def setup(self, problem, termination=None, seed=None, verbose=False):
        p = load_platypus()
        self.rng = random.Random(seed)
        self.evaluator = problem.evaluator
        self.lower, self.span = problem.xl, problem.xu - problem.xl
        native_problem = p.Problem(problem.n_var, 2, problem.n_ieq_constr)
        native_problem.types[:] = p.Real(0, 1)
        native_problem.constraints[:] = "<=0"
        self.problem = native_problem
        initial = []
        for x, f, g in zip(self.initial.get("X"), self.initial.get("F"), self.initial.get("G")):
            solution = p.Solution(native_problem)
            solution.variables[:] = ((x - self.lower) / self.span).tolist()
            self._assign(solution, f, g)
            initial.append(solution)
        ConstraintFitness, FixedIndicator, SharedGenerator, SharedCMAES = _native_components(p)
        generator = SharedGenerator(initial)
        variator = p.GAOperator(p.SBX(probability=0.9, distribution_index=15), p.PM(probability=1 / problem.n_var, distribution_index=20))
        options = {"generator": generator}
        if self.name == "IBEA":
            self.native = p.IBEA(native_problem, population_size=64, variator=variator,
                                 fitness_evaluator=FixedIndicator(kappa=0.05, rho=2.0), fitness_comparator=ConstraintFitness(), **options)
        elif self.name == "Epsilon-MOEA":
            self.native = p.EpsMOEA(native_problem, epsilons=[0.01, 0.01], population_size=64, variator=variator, **options)
        elif self.name == "SMPSO":
            self.native = p.SMPSO(native_problem, swarm_size=64, leader_size=64, generator=generator,
                                  max_iterations=max(1, math.ceil(self.evaluator.budget / 64)),
                                  mutate=p.PM(probability=1 / problem.n_var, distribution_index=20))
        elif self.name == "PAES":
            parent = self.rng.choice(p.nondominated(initial))
            self.native = p.PAES(native_problem, divisions=8, capacity=64, generator=SharedGenerator([parent]),
                                 variator=p.PM(probability=1 / problem.n_var, distribution_index=20))
        elif self.name == "PESA-II":
            self.native = p.PESA2(native_problem, population_size=64, divisions=8, capacity=64, variator=variator, **options)
        elif self.name == "MO-CMA-ES":
            self.native = SharedCMAES(native_problem, offspring_size=64, sigma=0.2, diagonal_iterations=0,
                                      indicator="crowding", check_consistency=False)
            self.native.shared_initial = initial
        else:
            raise WorkbenchError("没有该 Platypus 方法", "OPTIMIZER_NOT_FOUND")
        owner = self
        class SharedEvaluator(p.Evaluator):
            def evaluate_all(self, jobs, **kwargs):
                jobs = list(jobs)
                if not jobs:
                    return []
                allowance = owner.n_offsprings - owner.step_used
                selected = jobs[:allowance]
                if selected:
                    unit = np.array([job.solution.variables[:] for job in selected])
                    F, G, _ = owner.evaluator.evaluate(owner.lower + unit * owner.span, "search")
                    owner.step_used += len(selected)
                    for job, f, g in zip(selected, F / owner.scale, G - 1e-8):
                        owner._assign(job.solution, f, g)
                if len(selected) != len(jobs):
                    raise _EndBudget()
                return selected
        self.native.evaluator = SharedEvaluator()
        native_evaluate_all = self.native.evaluate_all
        def charge_offspring(solutions):
            if self.initialized:
                for solution in solutions:
                    solution.evaluated = False
            return native_evaluate_all(solutions)
        self.native.evaluate_all = charge_offspring
        self.shared_initial = initial

    @staticmethod
    def _assign(solution, f, g):
        solution.objectives[:] = list(f)
        solution.constraints[:] = list(g)
        solution.constraint_violation = float(np.maximum(g, 0).sum())
        solution.feasible = solution.constraint_violation == 0
        solution.evaluated = True

    def has_next(self):
        return self.stop_reason is None

    def next(self):
        self.step_used = 0
        token = _ACTIVE_RANDOM.set((self.rng, self.deadline))
        try:
            self.native.step()
            if not self.initialized:
                if self.name == "PAES":
                    self.native.archive += self.shared_initial
                self.initialized = True
        except _EndBudget:
            self.stop_reason = "total_budget"
        except _EndTime:
            self.stop_reason = "time_budget"
        finally:
            _ACTIVE_RANDOM.reset(token)
