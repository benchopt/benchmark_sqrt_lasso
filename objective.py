from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm


class Objective(BaseObjective):
    name = "Square root Lasso"

    # All parameters 'p' defined here are available as 'self.p'
    parameters = {
        'reg': [1e-1, 1e-2, 1e-3],
    }

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.lmbd = self.reg * Objective._compute_alpha_max(X, y)

    def compute(self, beta):
        datafit_val = norm(self.y - self.X @ beta)
        penalty_val = self.lmbd * norm(beta, ord=1)
        return datafit_val + penalty_val

    def to_dict(self):
        return dict(X=self.X, y=self.y, lmbd=self.lmbd)

    @staticmethod
    def _compute_alpha_max(X, y):
        return norm(X.T @ y, ord=np.inf) / norm(y)
