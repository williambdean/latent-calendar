"""Generate some fake data for various purposes."""

import numpy as np
from numpy import typing as npt
import pandas as pd

from latent_calendar.const import FULL_VOCAB
from latent_calendar.model.latent_calendar import LatentCalendar

try:
    import pymc as pm
    from pytensor.tensor import TensorVariable
    from pytensor import tensor as pt
except ImportError:

    class TensorVariable:
        pass

    class MockModule:
        def __getattr__(self, name):
            msg = (
                "PyMC is not installed."
                " Please install it directly or"
                " with extra installs: pip install 'latent-calendar[pymc]'"
            )
            raise ImportError(msg)

    pm = MockModule()
    pt = MockModule()


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


def define_single_user_samples(
    travel_style,
    time_slot_styles: TensorVariable,
    n_samples: int,
) -> tuple[TensorVariable, TensorVariable]:
    travel_style_user = pm.Categorical.dist(p=travel_style, shape=n_samples)
    time_slots = pm.Multinomial.dist(p=time_slot_styles[travel_style_user], n=1)

    return travel_style_user, time_slots


N_SAMPLES = Union[npt.NDArray[np.int_], int]
SAMPLE_RESULT = Tuple[pd.DataFrame, pd.DataFrame]


def _sample_lda(
    travel_style: npt.NDArray[np.float_],
    time_slot_styles: npt.NDArray[np.float_],
    n_samples: N_SAMPLES,
    random_state: Optional[int] = None,
) -> SAMPLE_RESULT:
    rng = np.random.default_rng(random_state)

    user_travel_style_data = []
    user_time_slot_data = []

    if isinstance(n_samples, int):
        n_samples = [n_samples]

    for n in n_samples:
        _, user_time_slots = define_single_user_samples(
            travel_style, time_slot_styles, n_samples=int(n)
        )

        user_travel_style_samples, user_time_slot_samples = pm.draw(
            [travel_style, user_time_slots.sum(axis=0)], draws=1, random_seed=rng
        )

        user_travel_style_data.append(user_travel_style_samples)
        user_time_slot_data.append(user_time_slot_samples)

    df_user_travel_style = pd.DataFrame(user_travel_style_data)
    df_user_time_slots = pd.DataFrame(user_time_slot_data)

    return df_user_travel_style, df_user_time_slots


def sample_from_lda(
    components_prior: Union[np.ndarray, TensorVariable],
    components_time_slots_prior: Union[np.ndarray, TensorVariable],
    n_samples: N_SAMPLES,
    random_state: Optional[int] = None,
) -> SAMPLE_RESULT:
    """Sample from LDA model.

    Args:
        components_prior: prior probability of each component (n_components, )
        components_time_slots_prior: prior for time slots (n_components, n_time_slots)
        n_samples: number of samples for all users or for each user (n_user, )
        random_state: random state for sampling

    Returns:
        probability DataFrame (n_user, n_components) and event count DataFrame with (n_user, n_time_slots) with each row summing up to `n`

    """

    travel_style = pm.Dirichlet.dist(components_prior)
    time_slot_styles = pm.Dirichlet.dist(components_time_slots_prior)

    return _sample_lda(
        travel_style=travel_style,
        time_slot_styles=time_slot_styles,
        n_samples=n_samples,
        random_state=random_state,
    )


def sample_from_latent_calendar(
    latent_calendar: LatentCalendar,
    n_samples: Union,
    random_state: Optional[int] = None,
) -> SAMPLE_RESULT:
    """Sample from a latent calendar model.

    Args:
        latent_calendar: fitted latent calendar model
        n_samples: number of rows to sample
        random_state: random state for reproducibility

    Returns:
        probability DataFrame (n_user, n_components) and event count DataFrame with (n_user, n_time_slots) with each row summing up to `n`

    """
    # TODO: Figure out how to best recreate based on the population
    travel_style = pm.Dirichlet.dist(latent_calendar.component_distribution_)
    time_slot_styles = pm.Dirichlet.dist(latent_calendar.components_)
    return _sample_lda(
        travel_style=travel_style,
        time_slot_styles=time_slot_styles,
        n_samples=n_samples,
        random_state=random_state,
    )
