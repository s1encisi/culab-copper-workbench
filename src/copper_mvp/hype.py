"""Two-objective HypE using exact allocation of dominated objective-space cells.

Reference: Bader and Zitzler, HypE, Evolutionary Computation 19(1), 2011.
The allocation weights and split-front deletion follow the HypE algorithm,
with a shared fixed reference and feasibility-first adaptation for this project.
"""
from __future__ import annotations

import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.survival import Survival
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def allocation_weights(population_size, k):
    """HypE alpha[c] = product((k-j)/(N-j), j=1..c-1) / c."""
    alpha = np.zeros(population_size + 1)
    if population_size and k > 0:
        alpha[1] = 1.0
        for count in range(2, min(k, population_size) + 1):
            alpha[count] = alpha[count-1] * (k-count+1) / (population_size-count+1) * (count-1) / count
    return alpha


class ExactHypEGeometry:
    """A disjoint rectangle partition remains exact as points are removed."""
    def __init__(self, objectives, reference):
        self.F = np.asarray(objectives, dtype=float)
        self.reference = np.asarray(reference, dtype=float)
        if self.F.ndim != 2 or self.F.shape[1] != 2 or self.reference.shape != (2,):
            raise ValueError("Exact HypE requires a two-objective matrix and reference")
        eligible = self.F[np.all(self.F < self.reference, axis=1)]
        if not len(eligible):
            self.cover = np.empty((len(self.F), 0), dtype=bool)
            self.areas = np.empty(0)
        else:
            x = np.unique(np.r_[eligible[:, 0], self.reference[0]])
            y = np.unique(np.r_[eligible[:, 1], self.reference[1]])
            mx, my = np.meshgrid(x[:-1], y[:-1], indexing="ij")
            self.cover = (self.F[:, 0, None] <= mx.ravel()) & (self.F[:, 1, None] <= my.ravel())
            self.areas = (np.diff(x)[:, None] * np.diff(y)[None, :]).ravel()
        self.cover_float = self.cover.astype(float)
        self.counts = self.cover.sum(axis=0)
        self.active = np.ones(len(self.F), dtype=bool)

    def fitness(self, k):
        weights = allocation_weights(int(self.active.sum()), k)
        return self.cover_float @ (self.areas * weights[self.counts])

    def remove(self, index):
        self.active[index] = False
        self.counts -= self.cover[index]


def hype_tournament(pop, pairs, random_state=None, **kwargs):
    winners = []
    for a, b in pairs:
        cv_a, cv_b = float(pop[a].CV[0]), float(pop[b].CV[0])
        if cv_a != cv_b:
            winners.append(a if cv_a < cv_b else b)
        else:
            fa, fb = pop[a].get("hype_fitness"), pop[b].get("hype_fitness")
            winners.append(a if fa > fb else b if fb > fa else random_state.choice([a, b]))
    return np.asarray(winners)[:, None]


class HypESurvival(Survival):
    def __init__(self, reference):
        super().__init__(filter_infeasible=True)
        self.reference = np.asarray(reference, dtype=float)

    def _do(self, problem, pop, *args, n_survive=None, random_state=None, **kwargs):
        fronts = NonDominatedSorting().do(pop.get("F"), n_stop_if_ranked=n_survive)
        survivors = []
        for front in fronts:
            needed = n_survive - len(survivors)
            if len(front) <= needed:
                survivors.extend(front)
            else:
                geometry = ExactHypEGeometry(pop[front].get("F"), self.reference)
                while int(geometry.active.sum()) > needed:
                    remaining = np.flatnonzero(geometry.active)
                    scores = geometry.fitness(len(remaining) - needed)
                    least = scores[remaining].min()
                    tied = remaining[scores[remaining] == least]
                    geometry.remove(int(random_state.choice(tied)))
                survivors.extend(front[geometry.active])
                break
        return pop[survivors]


class HypE(GeneticAlgorithm):
    """Exact 2-D HypE fitness for mating and k-dependent split-front deletion."""
    def __init__(self, reference, **kwargs):
        self.reference = np.asarray(reference, dtype=float)
        super().__init__(selection=TournamentSelection(func_comp=hype_tournament),
                         survival=HypESurvival(reference), advance_after_initial_infill=True, **kwargs)

    def _refresh_fitness(self):
        values = np.zeros(len(self.pop))
        feasible = np.flatnonzero(self.pop.get("CV")[:, 0] <= 0)
        if len(feasible):
            geometry = ExactHypEGeometry(self.pop[feasible].get("F"), self.reference)
            values[feasible] = geometry.fitness(len(feasible))
        self.pop.set("hype_fitness", values)

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)
        self._refresh_fitness()

    def _advance(self, infills=None, **kwargs):
        super()._advance(infills, **kwargs)
        self._refresh_fitness()

    def _set_optimum(self, **kwargs):
        feasible = np.flatnonzero(self.pop.get("CV")[:, 0] <= 0)
        if len(feasible):
            front = NonDominatedSorting().do(self.pop[feasible].get("F"), only_non_dominated_front=True)
            self.opt = self.pop[feasible[front]]
        else:
            self.opt = self.pop[[int(np.argmin(self.pop.get("CV")[:, 0]))]]
