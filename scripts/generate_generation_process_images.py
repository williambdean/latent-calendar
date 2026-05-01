"""Generate images for docs/examples/generation-process.md."""

import matplotlib.pyplot as plt

from latent_calendar import DummyModel
from latent_calendar.generate import LatentCalendarSampler
from latent_calendar.plot import plot_model_components
from latent_calendar.segments import create_box_segment, stack_segments

OUT = "docs/images"
DPI = 200
ROW_HEIGHT = 4.0  # inches per subplot row

# --- Segments used throughout ---
mornings = create_box_segment(
    day_start=0, day_end=5, hour_start=7, hour_end=10, name="Mornings"
)
evenings = create_box_segment(
    day_start=0, day_end=5, hour_start=18, hour_end=22, name="Evenings"
)
afternoons = create_box_segment(
    day_start=0, day_end=7, hour_start=12, hour_end=17, name="Afternoons"
)
weekends = create_box_segment(
    day_start=5, day_end=7, hour_start=12, hour_end=16, name="Weekends"
)
df_segments = stack_segments([mornings, afternoons, evenings, weekends])

# 1. Segment definitions — 2x2 grid (2 rows)
GRID_COLS = 2
n_segment_rows = -(-len(df_segments) // GRID_COLS)  # ceiling division
plt.rcParams["figure.figsize"] = (12, n_segment_rows * ROW_HEIGHT)
df_segments.cal.plot_by_row(max_cols=GRID_COLS)
plt.suptitle("Defined Segments", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/generation-segments.png", bbox_inches="tight", dpi=DPI)
plt.close()
print("saved generation-segments.png")

# 2. Model components from from_segments — 2x2 grid (2 rows)
model = DummyModel.from_segments(df_segments, weights=[3, 2, 1, 2])
n_component_rows = -(-model.n_components // GRID_COLS)  # ceiling division
plt.rcParams["figure.figsize"] = (12, n_component_rows * ROW_HEIGHT)
plot_model_components(model, max_cols=GRID_COLS)
plt.suptitle("Model Components (from segments)", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/generation-components.png", bbox_inches="tight", dpi=DPI)
plt.close()
print("saved generation-components.png")

# 3. Sampled events — 6 users, 2x3 grid (3 rows)
sampler = model.create_sampler(random_state=0)
_, df_events = sampler.sample(n_samples=[10, 20, 15, 25, 30, 18])
n_event_rows = -(-len(df_events) // GRID_COLS)  # ceiling division
plt.rcParams["figure.figsize"] = (12, n_event_rows * ROW_HEIGHT)
df_events.cal.plot_by_row(max_cols=GRID_COLS)
plt.suptitle("Sampled Events", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/generation-sampled-events.png", bbox_inches="tight", dpi=DPI)
plt.close()
print("saved generation-sampled-events.png")

# 4. Low vs high concentration_scale
fig, axes = plt.subplots(1, 2, figsize=(12, 3))

for ax, scale, title in [
    (axes[0], 1.0, "concentration_scale=1.0 (low variance)"),
    (axes[1], 5.0, "concentration_scale=5.0 (high variance)"),
]:
    s = LatentCalendarSampler(model, random_state=0, concentration_scale=scale)
    df_w, _ = s.sample([20] * 6)
    ax.bar(range(model.n_components), df_w.mean(), yerr=df_w.std(), capsize=4)
    ax.set_xticks(range(model.n_components))
    ax.set_xticklabels(["Mornings", "Afternoons", "Evenings", "Weekends"])
    ax.set_ylabel("Mean mixture weight")
    ax.set_ylim(0, 1)
    ax.set_title(title)

plt.tight_layout()
plt.savefig(f"{OUT}/generation-concentration-scale.png", bbox_inches="tight", dpi=DPI)
plt.close()
print("saved generation-concentration-scale.png")
