"""Models for the joint distribution of weekly calendar data.

```python
model = LatentCalendar(n_components=3, random_state=42)

X = df_wide.to_numpy()
model.fit(X)

X_latent = model.transform(X)
X_pred = model.predict(X)
```


"""

from packaging.version import Version

import numpy as np

from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation as BaseLDA


from conjugate.distributions import Dirichlet
from conjugate.models import multinomial_dirichlet


def joint_distribution(X_latent: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Marginalize out the components."""
    return X_latent @ components


class LatentCalendar(BaseLDA):
    """Model weekly calendar data as a mixture of multinomial distributions.

    Adapted from sklearn's [Latent Dirichlet Allocation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) model.

    Provides a `predict` method that returns the marginal probability of each time slot for a given row and
    a `transform` method that returns the latent representation of each row.

    """

    @property
    def normalized_components_(self) -> np.ndarray:
        """Components that each sum to 1."""
        return self.components_ / self.components_.sum(axis=1)[:, np.newaxis]

    def joint_distribution(self, X_latent: np.ndarray) -> np.ndarray:
        """Marginalize out the components."""
        return joint_distribution(
            X_latent=X_latent, components=self.normalized_components_
        )

    def predict(self, X: np.ndarray, y=None) -> np.ndarray:
        r"""Return the marginal probabilities for a given row.

        Marginalize out the loads via law of total probability

        $$P[time=t | Row=r] = \sum_{l=0}^{c} P[time=t | L=l, Row=r] * P[L=l | Row=r]$$

        """
        # (n, n_components)
        X_latent = self.transform(X)

        return self.joint_distribution(X_latent=X_latent)

    @property
    def component_distribution_(self) -> np.ndarray:
        """Population frequency of each component."""
        return self.components_.sum(axis=1) / self.components_.sum()


class DummyModel(LatentCalendar):
    """Return even probability of a latent.

    This can be used as the worse possible baseline.

    """

    def fit(self, X, y=None) -> "DummyModel":
        """All components are equal probabilty of every hour."""
        # Even probabilty for every thing
        self.n_components = 1
        TIME_SLOTS = X.shape[1]
        EVEN_PROBABILITY = 1 / TIME_SLOTS
        self.components_ = np.ones((self.n_components, TIME_SLOTS)) * EVEN_PROBABILITY

        return self

    def transform(self, X, y=None) -> np.ndarray:
        """Everyone has equal probability of being in each group."""
        nrows = len(X)

        return np.ones((nrows, self.n_components)) / self.n_components

    @classmethod
    def create(cls) -> "DummyModel":
        """Return a dummy model ready for transforming and predicting."""
        model = cls()
        model.fit(X=None)

        return model

    @classmethod
    def from_prior(cls, prior: np.ndarray) -> "DummyModel":
        """Return a dummy model from a prior."""
        model = cls()
        model.components_ = prior[np.newaxis, :]
        model.n_components = 1

        return model


class MarginalModel(LatentCalendar):
    def fit(self, X, y=None) -> "MarginalModel":
        """Just sum over all the rows."""
        self.n_components = 1
        # (1, n_times)
        self.components_ = X.sum(axis=0)[np.newaxis, :]

        return self

    def transform(self, X, y=None) -> np.ndarray:
        """There is only one component to be a part of."""
        nrows = len(X)

        # (nrows, 1)
        return np.repeat(1, nrows)[:, np.newaxis]


def constant_prior(X: np.ndarray, value: float = 1.0) -> np.ndarray:
    """Return the prior for each hour of the day.

    This is the average of all the rows.

    Args:
        X: (nrows, n_times)
    """
    TIME_SLOTS = X.shape[1]
    return np.repeat(value, TIME_SLOTS)


def hourly_prior(X: np.ndarray) -> np.ndarray:
    """Return the prior for each hour of the day.

    This is the average of all the rows.

    Args:
        X: (nrows, n_times)

    Returns:
        (n_times,)

    """
    return (X > 0).sum(axis=0) / len(X)


class ConjugateModel(BaseEstimator, TransformerMixin):
    """Conjugate model for the calendar joint distribution.

    This is a wrapper around the conjugate model for the multinomial
    distribution. It is a wrapper around the Dirichlet distribution.

    This doesn't use dimensionality reduction, but it does use the
    conjugate model.

    Args:
        a: (n_times,) prior for each hour of the day. If None, then
            the prior is the average of the data.

    """

    def __init__(self, a: np.ndarray | None = None) -> None:
        self.a = a

    def fit(self, X, y=None) -> "ConjugateModel":
        """Fit the conjugate model."""
        if self.a is None:
            self.a = hourly_prior(X)

        self.prior_ = Dirichlet(alpha=self.a)
        return self

    def transform(self, X, y=None) -> np.ndarray:
        return multinomial_dirichlet(x=X, prior=self.prior_).dist.mean()

    def predict(self, X, y=None) -> np.ndarray:
        return self.transform(X, y=y)


DOC_LINK_TEMPLATE = "https://williambdean.github.io/latent-calendar/modules/model/#latent_calendar.model.latent_calendar.{class_name}"


def url_param_generator_old(self, estimator):
    return {"class_name": estimator.__class__.__name__}


def url_param_generator_new(self):
    return {"class_name": self.__class__.__name__}


switch_version = Version("1.5.2")
current_version = Version(sklearn_version)
url_param_generator = (
    url_param_generator_new
    if current_version >= switch_version
    else url_param_generator_old
)


for klass in [LatentCalendar, DummyModel, MarginalModel, ConjugateModel]:
    klass._doc_link_module = "latent_calendar"
    klass._doc_link_template = DOC_LINK_TEMPLATE
    klass._doc_link_url_param_generator = url_param_generator
