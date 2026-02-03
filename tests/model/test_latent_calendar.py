import pytest

import pandas as pd
import numpy as np

from conjugate.distributions import Dirichlet

from latent_calendar.generate import wide_format_dataframe

from latent_calendar.const import TIME_SLOTS
from latent_calendar.model.latent_calendar import (
    LatentCalendar,
    MarginalModel,
    ConjugateModel,
    DummyModel,
)


@pytest.fixture
def mock_data() -> pd.DataFrame:
    return wide_format_dataframe(10, rate=1.0, random_state=42)


@pytest.fixture
def mock_latent_calendar(mock_data) -> LatentCalendar:
    model = LatentCalendar()
    model.fit(mock_data.to_numpy())

    return model


def test_latent_calendar(mock_latent_calendar) -> None:
    nrows = 10
    X = np.ones((nrows, TIME_SLOTS))

    assert mock_latent_calendar.transform(X).shape == (
        nrows,
        mock_latent_calendar.n_components,
    )
    assert mock_latent_calendar.predict(X).shape == (nrows, TIME_SLOTS)
    assert mock_latent_calendar.component_distribution_.shape == (
        mock_latent_calendar.n_components,
    )


@pytest.fixture
def dummy_model() -> DummyModel:
    return DummyModel()


def test_default_probabilities(dummy_model) -> None:
    nrows = 10
    X = np.ones((nrows, TIME_SLOTS))
    dummy_model.fit(X)
    assert dummy_model.components_.shape == (dummy_model.n_components, TIME_SLOTS)
    np.testing.assert_allclose(dummy_model.normalized_components_.sum(axis=1), 1.0)

    assert dummy_model.transform(X).shape == (nrows, dummy_model.n_components)
    assert dummy_model.predict(X).shape == (nrows, TIME_SLOTS)


@pytest.fixture
def conjugate_model() -> ConjugateModel:
    return ConjugateModel()


def test_conjugate_model(conjugate_model) -> None:
    nrows = 10
    X = np.ones((nrows, TIME_SLOTS))
    conjugate_model.fit(X)

    assert isinstance(conjugate_model.prior_, Dirichlet)

    assert conjugate_model.transform(X).shape == (nrows, TIME_SLOTS)
    assert conjugate_model.predict(X).shape == (nrows, TIME_SLOTS)


@pytest.mark.parametrize(
    "estimator",
    [
        MarginalModel,
        ConjugateModel,
        LatentCalendar,
        DummyModel,
    ],
)
def test_sklearn_documentation(estimator) -> None:
    instance = estimator()

    assert "williambdean" in instance._get_doc_link()
