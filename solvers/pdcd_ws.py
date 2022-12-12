from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from skglm.penalties import L1
    from skglm.experimental.pdcd_ws import PDCD_WS
    from skglm.experimental.sqrt_lasso import SqrtQuadratic


class Solver(BaseSolver):
    name = "PDCD-WS"

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/scikit-learn-contrib/skglm.git@main'
    ]

    reference = [
        "Bertrand, Q. and Klopfenstein, Q. and Bannier, P.-A. and Gidel, G."
        "and Massias, M."
        "'Beyond L1: Faster and Better Sparse Models with skglm',"
        "NeurIPS, 2022, https://arxiv.org/abs/2204.07826"
    ]

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        self.penalty = L1(lmbd)
        self.datafit = SqrtQuadratic()

        self.solver = PDCD_WS()

        # Cache Numba compilation
        self.run(5)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros(self.X.shape[1])
        else:
            self.solver.max_iter = n_iter
            coef = self.solver.solve(self.X, self.y,
                                     self.datafit, self.penalty)[0]

            self.coef = coef.flatten()

    def get_result(self):
        return self.coef
