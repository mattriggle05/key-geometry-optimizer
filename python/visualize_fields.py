import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FIELDS_DIR = "data/fields"
with open(os.path.join(FIELDS_DIR, "fields_meta.json")) as f:
    meta = json.load(f)

N_R = meta["n_r"]
N_THETA = meta["n_theta"]
R_MAX = meta["r_max"]
finger_order = meta["finger_order"]
fingers = meta["finger_data"]

theta_values = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
r_values = np.linspace(0, R_MAX, N_R)
THETA, R = np.meshgrid(theta_values, r_values)

left_fingers = sorted([f for f in fingers if f["hand"] == "left"],  key=lambda f: finger_order.index(f["finger"]))
right_fingers = sorted([f for f in fingers if f["hand"] == "right"], key=lambda f: finger_order.index(f["finger"]))

fig = plt.figure(figsize=(20, 9))
gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.5, wspace=0.4)

def plot_field(ax, field, title):
    pc = ax.pcolormesh(
        THETA, R, field,
        cmap="RdYlGn_r", # green=easy, red=hard
        vmin=0.0, vmax=1.0,
        shading="auto"
    )

    ax.plot(0, 0, "ko", markersize=6, label="rest")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax.set_xticklabels(["", "", "", ""])
    ax.set_yticks([R_MAX * 0.33, R_MAX * 0.66, R_MAX])
    ax.set_yticklabels([f"{int(R_MAX*0.33)}", f"{int(R_MAX*0.66)}", f"{int(R_MAX)}mm"], fontsize=6)
    ax.set_title(f"{title}", fontsize=9, pad=8)

    return pc

last_pc = None
for col, finger_data in enumerate(left_fingers):
    field = np.load(os.path.join(FIELDS_DIR, finger_data["file"]))

    ax = fig.add_subplot(gs[0, col], projection="polar")
    last_pc = plot_field(
        ax, field,
        f"L {finger_data['finger']}"
    )

for col, finger_data in enumerate(right_fingers):
    field = np.load(os.path.join(FIELDS_DIR, finger_data["file"]))

    ax = fig.add_subplot(gs[1, col], projection="polar")
    last_pc = plot_field(
        ax, field,
        f"R {finger_data['finger']}"
    )

# shared colorbar
cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
cbar = fig.colorbar(last_pc, cax=cbar_ax)
cbar.set_label("Effort (0=easy, 1=hard)", fontsize=10)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

# second figure: all fingers overlaid in cartesian for spatial sanity check
fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.set_title("Finger Rest Positions (mm from origin)", fontsize=13)
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
ax2.set_aspect("equal")
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.axvline(0, color="gray", linewidth=0.5)
ax2.grid(True, alpha=0.3)

colors = {"left": "steelblue", "right": "tomato"}
for finger_data in fingers:
    color = colors[finger_data["hand"]]
    ax2.plot(finger_data["x"], finger_data["y"], "o", color=color, markersize=10)
    ax2.annotate(
        f"{finger_data['hand'][0]}_{finger_data['finger']}",
        xy=(finger_data["x"], finger_data["y"]),
        xytext=(4, 4), textcoords="offset points",
        fontsize=8, color=color
    )

KEY_DIAMETER_MM = meta.get("key_diameter_mm", 19.05)
for finger_data in fingers:
    circle = plt.Circle(
        (finger_data["x"], finger_data["y"]),
        KEY_DIAMETER_MM / 2,
        color=colors[finger_data["hand"]],
        fill=False, linestyle="--", alpha=0.4
    )
    ax2.add_patch(circle)

ax2.legend(
    handles=[
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="steelblue", label="left hand"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="tomato",    label="right hand"),
    ],
    loc="upper right"
)

plt.tight_layout()
plt.show()