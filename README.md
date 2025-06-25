# Latent Calendar

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://github.com/williambdean/latent-calendar/actions/workflows/tests.yml/badge.svg)](https://github.com/williambdean/latent-calendar/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/latent-calendar.svg)](https://badge.fury.io/py/latent-calendar)
[![docs](https://github.com/williambdean/latent-calendar/actions/workflows/docs.yml/badge.svg)](https://williambdean.github.io/latent-calendar/)
[![codecov](https://codecov.io/gh/williambdean/latent-calendar/graph/badge.svg?token=WN7MMJPZ1S)](https://codecov.io/gh/williambdean/latent-calendar)

Analyze and model data on a weekly calendar

## Installation

Install from PyPI:

```bash
pip install latent-calendar
```

Or install directly from GitHub for the latest functionality.

## Features

- DataFrame agnostic transformations and aggregation powered by [`narwhals`](https://narwhals-dev.github.io/narwhals/)
- Integrated automatically into `pandas` and `polars` with [`cal` attribute on DataFrame and Series](https://williambdean.github.io/latent-calendar/modules/extensions)
- Compatible with [`scikit-learn` pipelines and transformers](https://williambdean.github.io/latent-calendar/examples/model/sklearn-compat)
- [Transform and visualize data on a weekly calendar](https://williambdean.github.io/latent-calendar/examples/cal-attribute)
- [Model weekly calendar data with a mixture of calendars](https://williambdean.github.io/latent-calendar/methodology)
- Create lower dimensional representations of calendar data

## Documentation

Find more examples and documentation [here](https://williambdean.github.io/latent-calendar/).
