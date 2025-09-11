import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Try out `latent-calendar`


    Use
    ```python
    import latent_calendar  # noqa: F401

    import polars as pl



    ```
    """
    )
    return


@app.cell
async def _():
    from io import BytesIO

    import marimo as mo
    import micropip

    await micropip.install(["latent-calendar"])

    import matplotlib.pyplot as plt

    import latent_calendar  # noqa: F401

    return BytesIO, mo, plt


@app.cell
def _(mo):
    file = mo.ui.file(kind="area")

    file
    return (file,)


@app.cell
def _(BytesIO, file):
    import narwhals as nw

    df = nw.read_csv(BytesIO(file.contents()), backend="pandas")
    return df, nw


@app.cell
def _(df, mo):
    id_col = mo.ui.multiselect(df.columns, label="ID column(s):")

    id_col
    return (id_col,)


@app.cell
def _(df, id_col, mo):
    timestamp_col = mo.ui.dropdown(
        set(df.columns) - set(id_col.value), label="Timestamp column:"
    )

    timestamp_col
    return (timestamp_col,)


@app.cell
def _(mo, timestamp_col):
    if not timestamp_col.value:
        mo.stop()

    seed = mo.ui.text(label="Random seed", value="42")
    nsamples = mo.ui.number(label=r"\# samples")
    max_cols = mo.ui.number(label=r"max \# rows", value=3)
    fig_height = mo.ui.slider(
        label="Figure height", start=5, stop=20, value=10, show_value=True
    )
    fig_width = mo.ui.slider(
        label="Figure width", start=10, stop=30, value=20, show_value=True
    )

    mo.vstack(
        [
            mo.md("**Plot configuration:**"),
            seed,
            nsamples,
            max_cols,
            fig_height,
            fig_width,
        ]
    )
    return fig_height, fig_width, max_cols, nsamples, seed


@app.cell
def _(df, id_col, mo, nsamples, nw, seed, timestamp_col):
    def widen(df):
        return (
            df.with_columns(
                nw.col(timestamp_col.value).str.to_datetime(),
            )
            .to_native()
            .cal.aggregate_events(id_col.value, timestamp_col.value)
            .cal.normalize("max")
        )

    try:
        df_widen = widen(df)
    except Exception as e:
        mo.output.replace(
            mo.md(f"""
            Error processing data:

            ````
            {e}
            ````
            """),
        )

    random_state = sum(map(ord, seed.value))

    if nsamples.value:
        df_widen = df_widen.sample(n=nsamples.value, random_state=random_state)
    return (df_widen,)


@app.cell
def _(df_widen, fig_height, fig_width, max_cols, plt):
    df_widen.cal.plot_by_row(max_cols=max_cols.value)
    fig = plt.gcf()
    fig.set_figwidth(fig_width.value)
    fig.set_figheight(fig_height.value)
    plt.tight_layout()

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
