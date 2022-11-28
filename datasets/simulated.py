from benchopt import BaseDataset
from benchopt.datasets import make_correlated_data


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_samples, n_features': [
            (500, 4000),
            (1000, 800),
        ]
    }

    def __init__(self, n_samples, n_features, random_state=27):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):
        X, y, _ = make_correlated_data(self.n_samples, self.n_features, rho=0.3,
                                       random_state=self.random_state)

        return dict(X=X, y=y)
