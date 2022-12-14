from benchopt.base import Solver, safe_import_context


with safe_import_context() as safe_import_ctx:
    from skglm.experimental import SqrtLasso


class Solver():
    name = "skglm-prox-newton"
    install_cmd = "conda"
    stopping_strategy = "iteration"

    requirements = ['pip:git+https://github.com/scikit-learn-contrib/skglm']

    def set_objective(X, y, lmbd):
        self.X, self.y, self.lmbda = X, y, lmbd
        self.clf = SqrtLasso(alpha=lmbd, fit_intercept=False)
        self.run(2)

    @staticmethod
    def get_next(n_iter):
        return n_iter + 1

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)
        self.w = self.clf.coef_

    def get_results(self):
        return self.w
