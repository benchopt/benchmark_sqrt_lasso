from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from benchmark_utils import prox_L1, prox_conjugate_L2


class Solver(BaseSolver):
    name = "vu-condat-cd"

    requirements = ['numba']

    references = [
        'Olivier Fercoq and Pascal Bianchi, '
        '"A Coordinate-Descent Primal-Dual Algorithm with Large Step Size '
        'and Possibly Nonseparable Functions", SIAM Journal on Optimization, 2020, '
        'https://epubs.siam.org/doi/10.1137/18M1168480,'
        'code: https://github.com/Badr-MOUFAD/Fercoq-Bianchi-solver'
    ]

    parameters = {
        'random_cd': [False, True]
    }

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        # Cache Numba compilation
        self.run(2)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros(self.X.shape[1])
        else:
            self.solver.max_iter = n_iter
            coef = vu_condat_cd(self.X, self.y, self.lmbd,
                                n_iter, self.random_cd)

            self.coef = coef.flatten()

    def get_result(self):
        return self.coef


@njit
def vu_condat_cd(X, y, alpha=1., max_iter=1000, random_cd=False):
    n_samples, n_features = X.shape
    arranged_features = np.arange(n_features)

    # init steps
    # step sizes in fercoq package are:
    #   - sigma = max(1 / np.sqrt(n_features * norm(X, axis=0, ord=2)**2))
    #   - tau = 0.9 / (norm(A, axis=0, ord=2)**2 * sigma * n_features)
    dual_step = 1 / np.linalg.norm(X, ord=2)
    primal_steps = np.zeros(n_features)
    for j in arranged_features:
        primal_steps[j] = 1 / np.linalg.norm(X[:, j])

    # primal vars
    w = np.zeros(n_features)
    Xw = np.zeros(n_samples)

    # dual vars
    z = np.zeros(n_samples)
    z_bar = np.zeros(n_samples)

    for _ in range(max_iter):

        # CD strategy
        if random_cd:
            features = np.random.choice(n_features, n_features)
        else:
            features = arranged_features

        # one epoch
        for j in features:
            # update primal
            old_w_j = w[j]

            w[j] = prox_L1(old_w_j - primal_steps[j] * X[:, j] @ (2 * z_bar - z),
                           primal_steps[j], alpha)

            # keep Xw synchro with X @ w
            if old_w_j != w[j]:
                Xw += (w[j] - old_w_j) * X[:, j]

            # update dual
            z_bar[:] = prox_conjugate_L2(z + dual_step * Xw,
                                         dual_step, y)
            z += (z_bar - z) / n_features

    return w
