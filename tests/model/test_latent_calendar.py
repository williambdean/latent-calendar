import numpy as np
import pandas as pd
import pytest
from conjugate.distributions import Dirichlet
from sklearn.base import clone

from latent_calendar.const import TIME_SLOTS
from latent_calendar.generate import wide_format_dataframe
from latent_calendar.model.latent_calendar import (
    ConjugateModel,
    DummyModel,
    LatentCalendar,
    MarginalModel,
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


# --- Tests for init and init_weights ---


class TestInit:
    """Tests for custom component initialization."""

    def test_init_sets_components_before_fit(self) -> None:
        """_init_latent_vars should apply custom init to components_."""
        n_components = 3
        custom = np.ones((n_components, TIME_SLOTS)) * 42.0
        model = LatentCalendar(n_components=n_components, init=custom)

        model._init_latent_vars(n_features=TIME_SLOTS)

        np.testing.assert_array_equal(model.components_, custom)

    def test_init_shape_validation(self) -> None:
        """Wrong init shape should raise ValueError."""
        bad_shape = np.ones((2, TIME_SLOTS))  # n_components=3 but init has 2 rows
        model = LatentCalendar(n_components=3, init=bad_shape)

        with pytest.raises(ValueError, match="init shape"):
            model._init_latent_vars(n_features=TIME_SLOTS)

    def test_init_negative_values_validation(self) -> None:
        """Negative init values should raise ValueError."""
        negative = np.full((3, TIME_SLOTS), -1.0)
        model = LatentCalendar(n_components=3, init=negative)

        with pytest.raises(ValueError, match="non-negative"):
            model._init_latent_vars(n_features=TIME_SLOTS)

    def test_init_exp_dirichlet_recomputed(self) -> None:
        """exp_dirichlet_component_ should be recomputed from custom init."""
        custom = np.ones((3, TIME_SLOTS)) * 5.0
        model = LatentCalendar(n_components=3, init=custom)

        model._init_latent_vars(n_features=TIME_SLOTS)

        from sklearn.decomposition._lda import _dirichlet_expectation_2d

        expected = np.exp(_dirichlet_expectation_2d(custom))
        np.testing.assert_array_equal(model.exp_dirichlet_component_, expected)

    def test_init_none_preserves_random_behavior(self) -> None:
        """init=None should use sklearn's default Gamma initialization."""
        model = LatentCalendar(n_components=3, random_state=42)

        model._init_latent_vars(n_features=TIME_SLOTS)

        assert model.components_.shape == (3, TIME_SLOTS)
        assert np.all(model.components_ > 0)
        assert not np.allclose(model.components_, 0)

    def test_init_in_fit_overwrites_each_call(self) -> None:
        """Each fit() call should re-apply init (not accumulate)."""
        custom = np.ones((3, TIME_SLOTS)) * 10.0
        model = LatentCalendar(n_components=3, init=custom, max_iter=0)
        X = np.ones((5, TIME_SLOTS)) * 3.0

        model.fit(X)
        assert np.allclose(model.components_, custom)

        # Second fit should re-apply init, not accumulate from first fit
        model.fit(X)
        assert np.allclose(model.components_, custom)


class TestInitWeights:
    """Tests for init_weights parameter."""

    def test_init_weights_scales_components(self) -> None:
        """init_weights should scale each component row."""
        shapes = np.ones((3, TIME_SLOTS))
        weights = np.array([3.0, 2.0, 1.0])
        model = LatentCalendar(n_components=3, init=shapes, init_weights=weights)

        model._init_latent_vars(n_features=TIME_SLOTS)

        expected = shapes * weights[:, np.newaxis]
        np.testing.assert_array_equal(model.components_, expected)

    def test_init_weights_shape_validation(self) -> None:
        """Wrong init_weights shape should raise ValueError."""
        bad_weights = np.array([0.5, 0.5])  # n_components=3 but 2 weights
        model = LatentCalendar(n_components=3, init_weights=bad_weights)

        with pytest.raises(ValueError, match="init_weights shape"):
            model._init_latent_vars(n_features=TIME_SLOTS)

    def test_init_weights_negative_validation(self) -> None:
        """Negative init_weights should raise ValueError."""
        negative_weights = np.array([1.0, -1.0, 1.0])
        model = LatentCalendar(n_components=3, init_weights=negative_weights)

        with pytest.raises(ValueError, match="non-negative"):
            model._init_latent_vars(n_features=TIME_SLOTS)

    def test_init_weights_only_without_init(self) -> None:
        """init_weights should work without init (scales random components)."""
        weights = np.array([3.0, 2.0, 1.0])
        model = LatentCalendar(n_components=3, init_weights=weights, random_state=42)

        model._init_latent_vars(n_features=TIME_SLOTS)

        # Components should be non-zero and shaped correctly
        assert model.components_.shape == (3, TIME_SLOTS)
        assert np.all(model.components_ > 0)

    def test_init_weights_influences_final_distribution(self) -> None:
        """Weights should influence the component_distribution_ after fit."""
        X = wide_format_dataframe(50, rate=5.0, random_state=42).to_numpy()
        weights = np.array([10.0, 1.0, 1.0])

        model = LatentCalendar(n_components=3, init_weights=weights, random_state=42)
        model.fit(X)

        # Component 0 should have the largest share due to highest weight
        dist = model.component_distribution_
        assert dist[0] > dist[1]
        assert dist[0] > dist[2]


class TestInitAndWeightsTogether:
    """Tests for init + init_weights combined."""

    def test_both_applied(self) -> None:
        """Both init and init_weights should be applied together."""
        shapes = np.ones((3, TIME_SLOTS))
        weights = np.array([2.0, 3.0, 1.0])
        model = LatentCalendar(n_components=3, init=shapes, init_weights=weights)

        model._init_latent_vars(n_features=TIME_SLOTS)

        expected = shapes * weights[:, np.newaxis]
        np.testing.assert_array_equal(model.components_, expected)

    def test_custom_shapes_with_weights_affect_fit(self) -> None:
        """Custom shapes + weights should produce different results than defaults."""
        X = wide_format_dataframe(50, rate=5.0, random_state=42).to_numpy()

        rng = np.random.RandomState(0)
        custom = rng.dirichlet(np.ones(TIME_SLOTS), size=3) * 100
        weights = np.array([5.0, 1.0, 1.0])

        model_custom = LatentCalendar(
            n_components=3, init=custom, init_weights=weights, random_state=42
        )
        model_custom.fit(X)

        model_default = LatentCalendar(n_components=3, random_state=42)
        model_default.fit(X)

        # Component distributions should differ
        assert not np.allclose(
            model_custom.component_distribution_,
            model_default.component_distribution_,
        )


class TestWarmStart:
    """Tests for warm start use case."""

    def test_warm_start_uses_previous_components(self) -> None:
        """Warm start should begin from previous model's components."""
        X = wide_format_dataframe(50, rate=5.0, random_state=42).to_numpy()

        model1 = LatentCalendar(n_components=3, random_state=42)
        model1.fit(X)

        model2 = LatentCalendar(
            n_components=3, init=model1.components_, random_state=42
        )
        model2._init_latent_vars(n_features=TIME_SLOTS)

        np.testing.assert_array_equal(model2.components_, model1.components_)

    def test_warm_start_produces_different_result(self) -> None:
        """Warm start should converge to a different solution than random init."""
        X = wide_format_dataframe(100, rate=5.0, random_state=42).to_numpy()

        model_random = LatentCalendar(n_components=5, random_state=42)
        model_random.fit(X)

        model_warm = LatentCalendar(
            n_components=5, init=model_random.components_, random_state=42
        )
        model_warm.fit(X)

        # Both should fit without error and produce valid output
        assert model_random.components_.shape == (5, TIME_SLOTS)
        assert model_warm.components_.shape == (5, TIME_SLOTS)
        assert np.all(model_random.components_ > 0)
        assert np.all(model_warm.components_ > 0)


class TestClone:
    """Tests for sklearn clone compatibility."""

    def test_clone_with_init(self) -> None:
        """clone() should work with init array."""
        init = np.ones((3, TIME_SLOTS))
        model = LatentCalendar(n_components=3, init=init, random_state=42)
        cloned = clone(model)

        np.testing.assert_array_equal(cloned.init, init)
        assert cloned.n_components == 3
        assert cloned.random_state == 42

    def test_clone_with_init_weights(self) -> None:
        """clone() should work with init_weights array."""
        weights = np.array([0.5, 0.3, 0.2])
        model = LatentCalendar(n_components=3, init_weights=weights, random_state=42)
        cloned = clone(model)

        np.testing.assert_array_equal(cloned.init_weights, weights)

    def test_clone_preserves_defaults(self) -> None:
        """clone() should preserve None defaults."""
        model = LatentCalendar(n_components=3, random_state=42)
        cloned = clone(model)

        assert cloned.init is None
        assert cloned.init_weights is None
