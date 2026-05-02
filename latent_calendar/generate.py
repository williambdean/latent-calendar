"""Generate some fake data for various purposes."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from latent_calendar.const import FULL_VOCAB


def wide_format_dataframe(
    n_rows: int,
    rate: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Generate some data from Poisson distribution.

    Args:
        n_rows: number of rows to generate
        rate: rate parameter for Poisson distribution
        random_state: random state for reproducibility

    Returns:
        DataFrame with columns from FULL_VOCAB and n_rows rows

    """
    if random_state is not None:
        np.random.seed(random_state)

    data = np.random.poisson(lam=rate, size=(n_rows, len(FULL_VOCAB)))

    return pd.DataFrame(data, columns=FULL_VOCAB)


def _sample_calendar(
    component_weights: np.ndarray,
    normalized_components: np.ndarray,
    n_samples: np.ndarray,
    rng: np.random.Generator,
    concentration_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample from an LDA-style generative model.

    Iterates over users in pure Python; the hot path per user (Dirichlet draw,
    component selection, inverse-CDF time slot sampling, and histogram
    accumulation) is handled by numpy and runs in C. Performance is acceptable
    for hundreds of users; for very large populations consider batching.

    For each user i:
        1. Optionally perturb concentration via Gamma draws (if concentration_scale != 1)
        2. Draw mixture weights from Dirichlet(component_weights[i])
        3. Draw component indices for all n_samples[i] events at once
        4. Gather component distributions and draw time slots via cumulative probs
        5. Aggregate time slot draws into a count vector

    Args:
        component_weights: Dirichlet concentration per user (n_users, n_components)
        normalized_components: probability over time slots per component
            (n_components, n_time_slots)
        n_samples: number of events per user (n_users,)
        rng: numpy random Generator
        concentration_scale: scale parameter for Gamma-perturbing the concentration
            before each user's Dirichlet draw. 1.0 (default) means no perturbation —
            each user draws directly from the fixed concentration. Values > 1.0
            increase variance across users' mixture weights.

    Returns:
        Tuple of:
            - mixture_weights: (n_users, n_components)
            - event_counts: (n_users, n_time_slots)

    """
    n_users, n_components = component_weights.shape
    n_time_slots = normalized_components.shape[1]

    mixture_weights = np.zeros((n_users, n_components))
    for i in range(n_users):
        alpha = component_weights[i]
        if concentration_scale != 1.0:
            alpha = rng.gamma(shape=alpha, scale=concentration_scale)
            alpha = np.clip(alpha, a_min=1e-8, a_max=None)
        mixture_weights[i] = rng.dirichlet(alpha)

    event_counts = np.zeros((n_users, n_time_slots), dtype=int)

    for i, n in enumerate(n_samples):
        if n == 0:
            continue
        component_indices = rng.choice(n_components, size=int(n), p=mixture_weights[i])
        probs = normalized_components[component_indices]
        cumprobs = probs.cumsum(axis=1)
        u = rng.random(size=(int(n), 1))
        time_slots = (u > cumprobs).sum(axis=1)
        np.add.at(event_counts[i], time_slots, 1)

    return mixture_weights, event_counts


class LatentCalendarSampler:
    """Sampler for generating synthetic calendar data from a fitted LatentCalendar model.

    Args:
        model: a fitted LatentCalendar model
        random_state: seed for reproducibility
        concentration_scale: scale for Gamma-perturbing each user's Dirichlet
            concentration before sampling mixture weights. 1.0 (default) means
            no perturbation. Values > 1.0 increase variance across users.

    Example:
        >>> model = LatentCalendar(n_components=5).fit(X)
        >>> sampler = model.create_sampler(random_state=42)
        >>> df_weights, df_events = sampler.sample(n_samples=[10, 5, 20])

    """

    def __init__(
        self,
        model,
        random_state: int | None = None,
        concentration_scale: float = 1.0,
    ) -> None:
        self.model = model
        self.random_state = random_state
        self.concentration_scale = concentration_scale
        self._rng = np.random.default_rng(random_state)

    def sample(
        self,
        n_samples: Union[int, list[int], np.ndarray],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Sample synthetic calendar events from the fitted model.

        Component mixture weights for each user are drawn from the population-level
        Dirichlet prior derived from the fitted model's component distribution.

        Args:
            n_samples: number of events per user. A single int produces one user
                with that many events. A list/array produces one user per element.

        Returns:
            Tuple of:
                - df_weights: mixture weight DataFrame (n_users, n_components)
                - df_events: event count DataFrame (n_users, n_time_slots)

        """
        if isinstance(n_samples, int):
            n_samples = [n_samples]

        n_samples = np.asarray(n_samples, dtype=int)
        n_users = len(n_samples)

        # Broadcast population-level concentration to (n_users, n_components)
        component_concentration = self.model.component_distribution_
        component_weights = np.broadcast_to(
            component_concentration, (n_users, len(component_concentration))
        ).copy()

        mixture_weights, event_counts = _sample_calendar(
            component_weights=component_weights,
            normalized_components=self.model.normalized_components_,
            n_samples=n_samples,
            rng=self._rng,
            concentration_scale=self.concentration_scale,
        )

        df_weights = pd.DataFrame(
            mixture_weights,
            columns=range(self.model.n_components),
        )
        columns = (
            self.model.feature_names_in_
            if hasattr(self.model, "feature_names_in_")
            else FULL_VOCAB
        )
        df_events = pd.DataFrame(event_counts, columns=columns)

        return df_weights, df_events

    def sample_events(self, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Sample events for a single user.

        Args:
            n: number of events to draw

        Returns:
            Tuple of:
                - df_weights: mixture weight DataFrame (1, n_components)
                - df_events: event count DataFrame (1, n_time_slots)

        """
        return self.sample(n_samples=n)


def sample_from_latent_calendar(
    model,
    n_samples: Union[int, list[int], np.ndarray],
    random_state: int | None = None,
    concentration_scale: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample synthetic calendar data from a fitted LatentCalendar model.

    Convenience wrapper around :class:`LatentCalendarSampler`.

    Args:
        model: fitted LatentCalendar model
        n_samples: number of events per user. A single int produces one user
            with that many events. A list/array produces one user per element.
        random_state: seed for reproducibility
        concentration_scale: scale for Gamma-perturbing each user's Dirichlet
            concentration before sampling mixture weights. 1.0 (default) means
            no perturbation. Values > 1.0 increase variance across users.

    Returns:
        Tuple of:
            - df_weights: mixture weight DataFrame (n_users, n_components)
            - df_events: event count DataFrame (n_users, n_time_slots)

    """
    return LatentCalendarSampler(
        model, random_state=random_state, concentration_scale=concentration_scale
    ).sample(n_samples)


def sample_from_lda(*args, **kwargs):
    """Removed. Use :func:`sample_from_latent_calendar` instead."""
    raise ImportError(
        "sample_from_lda has been removed. Use sample_from_latent_calendar instead."
    )
