import pytest

import numpy as np
from numpy.linalg import norm
from benchopt.datasets import make_correlated_data

from solvers.vu_condat_cd import vu_condat_cd
from solvers.chambolle_pock import chambolle_pock
from skglm.experimental import SqrtLasso


@pytest.mark.parametrize('solver', [chambolle_pock, vu_condat_cd,
                                    lambda *a: vu_condat_cd(*a, random_cd=True)])
def test_solver(solver):
    rho = 1e-1
    n_samples, n_features = 50, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=12365)

    alpha_max = norm(X.T @ y, ord=np.inf) / norm(y)
    alpha = rho * alpha_max

    w = solver(X, y, alpha)
    estimator = SqrtLasso(alpha, tol=1e-9).fit(X, y)

    np.testing.assert_allclose(w, estimator.coef_.flatten())
