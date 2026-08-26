"""Models for the joint distribution of weekly calendar data.

```python
model = LatentCalendar(n_components=3, random_state=42)

X = df_wide.to_numpy()
model.fit(X)

X_latent = model.transform(X)
X_pred = model.predict(X)
```


"""

import numpy as np
import pandas as pd
from conjugate.distributions import Dirichlet
from conjugate.models import multinomial_dirichlet
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation as BaseLDA
from sklearn.decomposition._lda import _dirichlet_expectation_2d


def joint_distribution(X_latent: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Marginalize out the components."""
    return X_latent @ components


class LatentCalendar(BaseLDA):
    """Model weekly calendar data as a mixture of multinomial distributions.

    Adapted from sklearn's [Latent Dirichlet Allocation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) model.

    Provides a `predict` method that returns the marginal probability of each time slot for a given row and
    a `transform` method that returns the latent representation of each row.

    Args:
        init: initial component matrix of shape (n_components, n_features).
            If None, components are initialized randomly via Gamma distribution.
        init_weights: initial component weights of shape (n_components,).
            Scales each component row. If None, weights are derived from
            component row sums.
        n_components: number of topics (default 10).
        doc_topic_prior: prior for document-topic distribution.
        topic_word_prior: prior for topic-word distribution.
        learning_method: 'batch' or 'online'.
        learning_decay: decay factor for online learning.
        learning_offset: control for earlier iterations.
        max_iter: maximum number of iterations.
        batch_size: number of documents per batch.
        evaluate_every: evaluate perplexity every N iterations.
        total_samples: total documents for online learning.
        perp_tol: perplexity tolerance for early stopping.
        mean_change_tol: mean change tolerance for early stopping.
        max_doc_update_iter: max E-step iterations.
        n_jobs: number of parallel jobs.
        verbose: verbosity level.
        random_state: random state for reproducibility.

    """

    def __init__(
        self,
        *,
        init=None,
        init_weights=None,
        n_components=10,
        doc_topic_prior=None,
        topic_word_prior=None,
        learning_method="batch",
        learning_decay=0.7,
        learning_offset=10.0,
        max_iter=10,
        batch_size=128,
        evaluate_every=-1,
        total_samples=1e6,
        perp_tol=0.1,
        mean_change_tol=1e-3,
        max_doc_update_iter=100,
        n_jobs=None,
        verbose=0,
        random_state=None,
    ):
        super().__init__(
            n_components=n_components,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
            learning_method=learning_method,
            learning_decay=learning_decay,
            learning_offset=learning_offset,
            max_iter=max_iter,
            batch_size=batch_size,
            evaluate_every=evaluate_every,
            total_samples=total_samples,
            perp_tol=perp_tol,
            mean_change_tol=mean_change_tol,
            max_doc_update_iter=max_doc_update_iter,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
        self.init = init
        self.init_weights = init_weights

    def _init_latent_vars(self, n_features, dtype=np.float64):
        super()._init_latent_vars(n_features, dtype=dtype)

        components = self.components_

        if self.init is not None:
            init = np.asarray(self.init, dtype=dtype)
            if init.shape != (self.n_components, n_features):
                raise ValueError(
                    f"init shape {init.shape} != expected "
                    f"({self.n_components}, {n_features})"
                )
            if np.any(init < 0):
                raise ValueError("init must contain non-negative values")
            components = init

        if self.init_weights is not None:
            weights = np.asarray(self.init_weights, dtype=dtype)
            if weights.shape != (self.n_components,):
                raise ValueError(
                    f"init_weights shape {weights.shape} != expected "
                    f"({self.n_components},)"
                )
            if np.any(weights < 0):
                raise ValueError("init_weights must contain non-negative values")
            components = components * weights[:, np.newaxis]

        self.components_ = components
        self.exp_dirichlet_component_ = np.exp(
            _dirichlet_expectation_2d(self.components_)
        )

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

    def create_sampler(
        self,
        random_state: int | None = None,
        concentration_scale: float = 1.0,
    ):
        """Create a sampler for generating synthetic calendar data.

        Args:
            random_state: seed for reproducibility
            concentration_scale: scale for Gamma-perturbing each user's Dirichlet
                concentration before sampling mixture weights. 1.0 (default) means
                no perturbation — each user draws from the fixed population prior.
                Values > 1.0 increase variance across users' mixture weights.

        Returns:
            LatentCalendarSampler bound to this fitted model

        Example:
            >>> model = LatentCalendar(n_components=5).fit(X)
            >>> sampler = model.create_sampler(random_state=42)
            >>> df_weights, df_events = sampler.sample(n_samples=[10, 5, 20])

        """
        from latent_calendar.generate import LatentCalendarSampler

        return LatentCalendarSampler(
            self, random_state=random_state, concentration_scale=concentration_scale
        )


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
    def from_prior(cls, prior: np.ndarray | pd.Series) -> "DummyModel":
        """Return a dummy model from a prior.

        Args:
            prior: prior probability weights over time slots. Can be a numpy
                array of shape (n_time_slots,) or a segment Series (e.g. from
                `create_box_segment`) with a FULL_VOCAB-compatible index.

        Returns:
            DummyModel with a single component defined by the prior.

        Example:
            Build a model that concentrates on weekday mornings:

            ```python
            from latent_calendar import DummyModel
            from latent_calendar.segments import create_box_segment

            mornings = create_box_segment(
                day_start=0, day_end=5, hour_start=7, hour_end=10,
                name="Weekday mornings",
            )
            model = DummyModel.from_prior(mornings)
            sampler = model.create_sampler(random_state=0)
            df_weights, df_events = sampler.sample(n_samples=[20, 30, 15])
            ```

        """
        if isinstance(prior, pd.Series):
            prior = prior.to_numpy()

        model = cls()
        model.components_ = prior[np.newaxis, :]
        model.n_components = 1

        return model

    @classmethod
    def from_segments(
        cls,
        df_segments: "pd.DataFrame",
        weights: "np.ndarray | list[float] | None" = None,
    ) -> "DummyModel":
        """Return a multi-component model where each segment is one component.

        Each row of `df_segments` becomes one component in the model. The
        population-level mixture over components is derived from
        `component_distribution_` — by default this weights components
        proportionally to the number of active slots in each segment. Pass
        explicit `weights` to override this.

        Args:
            df_segments: segments DataFrame in wide format, shape
                (n_segments, n_time_slots), e.g. from `stack_segments`.
            weights: optional 1-D array of length n_segments. Scales each
                component's contribution to the population prior. If None,
                weighting is proportional to active slot count per segment.

        Returns:
            DummyModel with one component per segment.

        Example:
            ```python
            from latent_calendar import DummyModel
            from latent_calendar.segments import create_box_segment, stack_segments

            mornings = create_box_segment(
                day_start=0, day_end=5, hour_start=7, hour_end=10, name="Mornings"
            )
            evenings = create_box_segment(
                day_start=0, day_end=5, hour_start=18, hour_end=22, name="Evenings"
            )
            df_segments = stack_segments([mornings, evenings])

            # Equal implicit weight (proportional to active slots)
            model = DummyModel.from_segments(df_segments)

            # Mornings 3x more likely than evenings
            model = DummyModel.from_segments(df_segments, weights=[3, 1])

            sampler = model.create_sampler(random_state=0)
            df_weights, df_events = sampler.sample(n_samples=[10, 20, 15])
            ```

        """
        components = df_segments.to_numpy().astype(float)

        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            components = components * weights[:, np.newaxis]

        model = cls()
        model.components_ = components
        model.n_components = len(df_segments)

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
