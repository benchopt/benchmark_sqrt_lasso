from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from skglm.prototype_pdcd.penalties import L1
    from skglm.prototype_pdcd.datafits import SqrtQuadratic
    from skglm.prototype_pdcd.algorithms import PDCD, PDCD_WS


class Solver(BaseSolver):
    name = "PDCD"

    requirements = [
        'pip:git+https://github.com/Badr-MOUFAD/skglm.git@pdcd-algo'
    ]

    parameters = {
        'with_ws': [True, False]
    }

    def __init__(self, with_ws):
        self.with_ws = with_ws

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        solver_class = PDCD_WS if self.with_ws else PDCD

        self.penalty = L1(lmbd)
        self.datafit = SqrtQuadratic()
        self.solver = solver_class(tol=1e-9)

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
