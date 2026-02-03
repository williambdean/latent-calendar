import pandas as pd

import matplotlib.pyplot as plt

import latent_calendar


if __name__ == "__main__":
    # More information on the dataset:
    # https://posit-dev.github.io/great-tables/reference/data.pizzaplace.html#great_tables.data.pizzaplace
    file = "https://raw.githubusercontent.com/posit-dev/great-tables/main/great_tables/data/05-pizzaplace.csv"

    df = pd.read_csv(file)

    # Create a datetime column
    datetime_column = pd.to_datetime(df["date"].str.cat(df["time"], sep=" "))


    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
    fig.suptitle("Plotting datetime column from posit-dev/great-tables pizza place dataset")

    ax = axes[0]
    datetime_column.sample(n=100, random_state=0).cal.plot(ax=ax)
    ax.set_title("Continuous Series cal.plot()")

    ax = axes[1]
    (
        datetime_column
        .cal.aggregate_events(minutes=30)
        .cal.plot_row(ax=ax)
    )
    ax.set_title("Discretized Series cal.plot_row()")

    plt.show()
