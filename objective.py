from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm


class Objective(BaseObjective):
    min_benchopt_version = "1.3"
    name = "Square root Lasso"

    parameters = {
        'reg': [0.5, 0.1, 0.05, 0.01],
    }

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.lmbd = self.reg * Objective._compute_alpha_max(X, y)

    def compute(self, beta):
        datafit_val = norm(self.y - self.X @ beta)
        penalty_val = self.lmbd * norm(beta, ord=1)

        return {
            'value': datafit_val + penalty_val,
            'support size': (beta != 0).sum()
        }

    def get_objective(self):
        return dict(X=self.X, y=self.y, lmbd=self.lmbd)

    @staticmethod
    def _compute_alpha_max(X, y):
        return norm(X.T @ y, ord=np.inf) / norm(y)

    def get_one_solution(self):
        n_features = self.X.shape[1]
        return np.zeros(n_features)
