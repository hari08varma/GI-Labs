"""Value signal (plan §7.6): goal relevance x learning progress.

For each open learning target tau (a frame without a locked column):
  LP_tau = mean prediction error over the previous window - mean over the latest window
  (W = 5 events; Oudeyer's intrinsic motivation, Baranes & Oudeyer).
  value_tau = G_tau * (LP_tau + eps_explore), G = 1 for the current task's frames, else 0.3.
Prediction errors are prequential: each observation is predicted before it is read.
Noise has no learning progress, so it does not attract effort (the second noisy-TV guard).
Before two full windows exist, LP is estimated optimistically as half the mean error.
"""
from __future__ import annotations

import numpy as np


class ValueSignal:
    def __init__(self, hp):
        self.W = hp.lp_window
        self.eps = hp.eps_explore
        self.goal_off = hp.goal_off

    def progress(self, errors: list[int]) -> float:
        W = self.W
        if not errors:
            return 0.5
        if len(errors) < 2 * W:
            return 0.5 * float(np.mean(errors))
        return float(np.mean(errors[-2 * W:-W]) - np.mean(errors[-W:]))

    def value(self, cl, F: str) -> float:
        col = cl.active_col(F)
        if col is not None and col.state == "locked":
            return 0.0
        G = 1.0 if cl is cl.a.cur else self.goal_off
        base = G * (max(self.progress(cl.errors.get(F, [])), 0.0) + self.eps)
        return base * (1.0 + self.blocking(cl, F))

    def blocking(self, cl, F: str) -> int:
        """Goal relevance through dependence: how many other unexplained frames need the
        kind of primitive F's rule would provide. Needs come from experience: the library's
        schemas for a frame's type whose holes ask for a primitive kind this context lacks.
        Nothing about the curriculum is assumed; with an empty library this is 0."""
        a = cl.a
        if not a.hp.flags.priors and not a.hp.flags.library:
            return 0
        info = cl.ctx.frames[F]
        if info["arity"] == 1 and not info["bool"]:
            provides = {"U", "P", "C"}
        else:
            provides = {"R"} if info["bool"] else {"B"}
        P = cl.prims()
        have = {k for k in ("U", "P", "C", "B", "R") if P.kinds(k)}
        n = 0
        for G, gi in cl.ctx.frames.items():
            if G == F or not cl.ctx.words_of(G):
                continue
            c = cl.active_col(G)
            if c is not None and c.state == "locked":
                continue
            ot = "B" if gi["bool"] else "O"
            needs = set()
            for s in a.priors.library(gi["arity"], ot):
                for kinds in s.kinds.values():
                    needs |= (set(kinds) - {"E", "T"}) - have
            if needs & provides:
                n += 1
        return n
