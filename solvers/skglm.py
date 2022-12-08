from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    import numpy as np
    from skglm.experimental import SqrtLasso
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = "skglm"

    install_cmd = 'conda'
    requirements = ['pip:skglm']

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        n_samples = self.X.shape[0]

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        self.sqrt_lasso = SqrtLasso(
            alpha=self.lmbd / np.sqrt(n_samples), tol=1e-9)

        # Cache Numba compilation
        self.run(5)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros([self.X.shape[1]])
        else:
            self.sqrt_lasso.max_iter = n_iter
            self.sqrt_lasso.fit(self.X, self.y)

            self.coef = self.sqrt_lasso.coef_.flatten()

    def get_result(self):
        return self.coef
