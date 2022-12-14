from benchopt import BaseSolver, safe_import_context


with safe_import_context() as safe_import_ctx:
    from skglm.experimental import SqrtLasso


class Solver(BaseSolver):
    name = "skglm-prox-newton"

    stopping_strategy = "iteration"

    install_cmd = "conda"
    requirements = [
        'pip:git+https://github.com/scikit-learn-contrib/skglm.git@main'
    ]

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbda = X, y, lmbd
        self.clf = SqrtLasso(alpha=lmbd, max_iter=2, verbose=0, tol=1e-10)
        self.run(2)

    @staticmethod
    def get_next(n_iter):
        return n_iter + 1

    def run(self, n_iter):
        if hasattr(self.clf, "solver_"):
            self.clf.solver_.max_iter = n_iter
        self.clf.fit(self.X, self.y)
        self.w = self.clf.coef_

    def get_result(self):
        return self.w
