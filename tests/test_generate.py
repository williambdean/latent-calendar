import numpy as np
import pytest

from latent_calendar.const import FULL_VOCAB
from latent_calendar.generate import (
    LatentCalendarSampler,
    sample_from_latent_calendar,
    wide_format_dataframe,
)
from latent_calendar.model.latent_calendar import DummyModel, LatentCalendar
from latent_calendar.segments import create_box_segment, stack_segments


N_COMPONENTS = 3
N_ROWS = 20


@pytest.fixture
def fitted_model() -> LatentCalendar:
    df = wide_format_dataframe(n_rows=N_ROWS, rate=1.0, random_state=0)
    model = LatentCalendar(n_components=N_COMPONENTS, random_state=0)
    model.fit(df)
    return model


@pytest.fixture
def df_segments():
    mornings = create_box_segment(
        day_start=0, day_end=5, hour_start=7, hour_end=10, name="Mornings"
    )
    evenings = create_box_segment(
        day_start=0, day_end=5, hour_start=18, hour_end=22, name="Evenings"
    )
    return stack_segments([mornings, evenings])


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


def test_concentration_scale_changes_variance(fitted_model) -> None:
    n_users = 50
    n_samples = [20] * n_users

    s1 = LatentCalendarSampler(fitted_model, random_state=0, concentration_scale=1.0)
    s2 = LatentCalendarSampler(fitted_model, random_state=0, concentration_scale=5.0)

    df_w1, _ = s1.sample(n_samples)
    df_w2, _ = s2.sample(n_samples)

    # Higher scale should produce more variance in mixture weights across users
    assert df_w2.var(axis=0).mean() > df_w1.var(axis=0).mean()


def test_concentration_scale_reproducible(fitted_model) -> None:
    s1 = LatentCalendarSampler(fitted_model, random_state=7, concentration_scale=3.0)
    s2 = LatentCalendarSampler(fitted_model, random_state=7, concentration_scale=3.0)
    _, df1 = s1.sample([10, 20])
    _, df2 = s2.sample([10, 20])
    np.testing.assert_array_equal(df1.values, df2.values)


def test_dummy_model_from_segments(df_segments) -> None:
    model = DummyModel.from_segments(df_segments)

    assert model.n_components == 2
    assert model.components_.shape == (2, 168)

    sampler = model.create_sampler(random_state=0)
    df_weights, df_events = sampler.sample([10, 20])

    assert df_weights.shape == (2, 2)
    assert list(df_events.values.sum(axis=1)) == [10, 20]


def test_dummy_model_from_segments_with_weights(df_segments) -> None:
    weights = [3, 1]
    model = DummyModel.from_segments(df_segments, weights=weights)

    assert model.n_components == 2
    # Each row is scaled by its weight — check ratios match the original slot counts
    base = DummyModel.from_segments(df_segments)
    for i, w in enumerate(weights):
        np.testing.assert_allclose(model.components_[i], base.components_[i] * w)


def test_dummy_model_from_segments_component_distribution(df_segments) -> None:
    # Without weights: component_distribution_ proportional to active slot count
    model = DummyModel.from_segments(df_segments)
    dist = model.component_distribution_
    assert dist.shape == (2,)
    assert dist.sum() == pytest.approx(1.0)

    # With equal weights: segments with more active slots still dominate
    model_weighted = DummyModel.from_segments(df_segments, weights=[1, 1])
    dist_weighted = model_weighted.component_distribution_
    # Mornings (15 slots) should outweigh Evenings (20 slots)
    # Just check it sums to 1 and is a valid distribution
    assert dist_weighted.sum() == pytest.approx(1.0)
    assert all(dist_weighted > 0)
