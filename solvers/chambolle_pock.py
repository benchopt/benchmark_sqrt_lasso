from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from benchmark_utils import prox_L1, prox_conjugate_L2


class Solver(BaseSolver):
    name = "Chambolle-Pock"

    requirements = ['numba']

    references = [
        'Antonin Chambolle, Thomas Pock'
        '"A first-order primal-dual algorithm for convex '
        'problems with applications to imaging", '
        'Journal of Mathematical Imaging and Vision, 40 (2011), pp 120-145. '
        'https://hal.archives-ouvertes.fr/hal-00490826/document'
    ]

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        # Cache Numba compilation
        self.run(2)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros(self.X.shape[1])
        else:
            self.solver.max_iter = n_iter
            coef = chambolle_pock(self.X, self.y, self.lmbd, n_iter)

            self.coef = coef.flatten()

    def get_result(self):
        return self.coef


@njit
def chambolle_pock(X, y, alpha=1., max_iter=1000):
    n_samples, n_features = X.shape

    # init steps
    L = np.linalg.norm(X, ord=2)
    dual_step = 0.99 / L
    primal_step = 0.99 / L

    # primal vars
    w = np.zeros(n_features)
    w_bar = np.zeros(n_features)

    # dual vars
    z = np.zeros(n_samples)

    for _ in range(max_iter):
        # dual update
        z[:] = prox_conjugate_L2(z + dual_step * X @ w_bar,
                                 dual_step, y)

        # primal update
        old_w = w.copy()
        w[:] = prox_L1(old_w - primal_step * X.T @ z,
                       primal_step, alpha)
        w_bar[:] = 2 * w - old_w

    return w
