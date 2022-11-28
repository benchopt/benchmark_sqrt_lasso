from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from skglm.prototype_pdcd.penalties import L1
    from skglm.prototype_pdcd.datafits import SqrtQuadratic
    from skglm.prototype_pdcd.algorithms import ChambollePock


class Solver(BaseSolver):
    name = "Chambolle-Pock"

    requirements = [
        'pip:git+https://github.com/Badr-MOUFAD/skglm.git@pdcd-algo'
    ]

    parameters = {
        'with_dual_init': [True, False]
    }

    def __init__(self, with_dual_init):
        self.with_dual_init = with_dual_init
        self.dual_init = None

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        self.penalty = L1(lmbd)
        self.datafit = SqrtQuadratic()

        if self.with_dual_init:
            self.dual_init = self.y / np.linalg.norm(y)

        self.solver = ChambollePock(tol=1e-9, dual_init=self.dual_init)

        # Cache Numba compilation
        self.run(5)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros([self.X.shape[1]])
        else:
            self.solver.max_iter = n_iter
            coef = self.solver.solve(self.X, self.y,
                                     self.datafit, self.penalty)[0]

            self.coef = coef.flatten()

    def get_result(self):
        return self.coef
