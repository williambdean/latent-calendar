import numpy as np
import pytest

from latent_calendar.const import FULL_VOCAB
from latent_calendar.generate import (
    LatentCalendarSampler,
    sample_from_latent_calendar,
    wide_format_dataframe,
)
from latent_calendar.model.latent_calendar import LatentCalendar


N_COMPONENTS = 3
N_ROWS = 20


@pytest.fixture
def fitted_model() -> LatentCalendar:
    df = wide_format_dataframe(n_rows=N_ROWS, rate=1.0, random_state=0)
    model = LatentCalendar(n_components=N_COMPONENTS, random_state=0)
    model.fit(df)
    return model


def test_sampler_single_user(fitted_model) -> None:
    sampler = LatentCalendarSampler(fitted_model, random_state=42)
    df_weights, df_events = sampler.sample_events(n=10)

    assert df_weights.shape == (1, N_COMPONENTS)
    assert df_events.shape == (1, 168)
    assert df_events.values.sum() == 10


def test_sampler_multiple_users(fitted_model) -> None:
    n_samples = [10, 5, 20]
    sampler = LatentCalendarSampler(fitted_model, random_state=42)
    df_weights, df_events = sampler.sample(n_samples=n_samples)

    assert df_weights.shape == (3, N_COMPONENTS)
    assert df_events.shape == (3, 168)
    assert list(df_events.values.sum(axis=1)) == n_samples


def test_sampler_scalar_n_samples(fitted_model) -> None:
    sampler = LatentCalendarSampler(fitted_model, random_state=0)
    df_weights, df_events = sampler.sample(n_samples=7)

    assert df_weights.shape == (1, N_COMPONENTS)
    assert df_events.values.sum() == 7


def test_sampler_reproducible(fitted_model) -> None:
    s1 = LatentCalendarSampler(fitted_model, random_state=99)
    s2 = LatentCalendarSampler(fitted_model, random_state=99)
    _, df1 = s1.sample([10, 20])
    _, df2 = s2.sample([10, 20])
    np.testing.assert_array_equal(df1.values, df2.values)


def test_sampler_columns_match_vocab(fitted_model) -> None:
    sampler = LatentCalendarSampler(fitted_model, random_state=0)
    _, df_events = sampler.sample([5])
    assert list(df_events.columns) == list(FULL_VOCAB)


def test_create_sampler_method(fitted_model) -> None:
    sampler = fitted_model.create_sampler(random_state=42)
    assert isinstance(sampler, LatentCalendarSampler)
    df_weights, df_events = sampler.sample([10])
    assert df_weights.shape == (1, N_COMPONENTS)
    assert df_events.values.sum() == 10


def test_sample_from_latent_calendar(fitted_model) -> None:
    n_samples = [10, 20, 30]
    df_weights, df_events = sample_from_latent_calendar(
        fitted_model, n_samples=n_samples, random_state=42
    )
    assert df_weights.shape == (3, N_COMPONENTS)
    assert df_events.shape == (3, 168)
    assert list(df_events.values.sum(axis=1)) == n_samples
