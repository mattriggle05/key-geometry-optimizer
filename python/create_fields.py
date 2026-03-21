import numpy as np
import json
import os

N_R = 32
N_THETA = 64
R_MAX = 60
EFFORT_EXPONENT = 1.5
BIAS_STRENGTH = 0.5
INDEX_OFFSET = 20.0

OUTPUT_DIR = "data/fields"
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open("data/finger_positions.json") as f:
    hands = json.load(f)

offsets = {
    "left": (-INDEX_OFFSET, 0.0),
    "right": (INDEX_OFFSET, 0.0),
}

r_values = np.linspace(0, 1, N_R)
theta_values = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
THETA_GRID, R_GRID = np.meshgrid(theta_values, r_values)

THETA_RIGHT = 0.0
THETA_UP = np.pi / 2
THETA_LEFT = np.pi
THETA_DOWN = 3 * np.pi / 2
THETA_FORWARD = THETA_DOWN

def make_field(easy_directions, easy_weights, bias_strength=BIAS_STRENGTH):
    easy_weights = np.array(easy_weights, dtype=float)
    easy_weights /= easy_weights.sum()

    radial = R_GRID ** EFFORT_EXPONENT

    angular_ease = np.zeros_like(THETA_GRID)
    for angle, weight in zip(easy_directions, easy_weights):
        angular_ease += weight * (0.5 + 0.5 * np.cos(THETA_GRID - angle))

    angular_effort = 1.0 - angular_ease
    field = (1.0 - bias_strength) * radial + bias_strength * radial * (0.5 + 0.5 * angular_effort)
    field = field / field.max()
    return field.astype(np.float32)

def finger_field(hand_side, finger_name):
    inward = THETA_LEFT if hand_side == "right" else THETA_RIGHT
    outward = THETA_RIGHT if hand_side == "right" else THETA_LEFT

    if finger_name == "thumb":
        return make_field(
            easy_directions=[THETA_FORWARD, outward],
            easy_weights=[1.0],
            bias_strength=0.8
        )
    elif finger_name == "middle":
        return make_field(
            easy_directions=[THETA_FORWARD, inward],
            easy_weights=[0.6, 0.4],
        )
    elif finger_name == "index":
        return make_field(
            easy_directions=[inward],
            easy_weights=[1.0],
        )
    elif finger_name == "ring":
        return make_field(
            easy_directions=[outward],
            easy_weights=[0.6, 0.4],
        )
    elif finger_name == "pinky":
        return make_field(
            easy_directions=[outward],
            easy_weights=[0.6, 0.4],
        )


meta = {
    "n_r": N_R,
    "n_theta": N_THETA,
    "r_max": R_MAX,
    "index_offset": INDEX_OFFSET,
    "finger_order": ["pinky", "ring", "middle", "index", "thumb"],
    "finger_data": []
}

for hand_side, fingers in hands.items():
    ox, oy = offsets[hand_side]

    for finger in fingers:
        name = finger["finger"]

        field = finger_field(hand_side, name)
        filename = f"{hand_side}_{name}.npy"
        filepath = os.path.join(OUTPUT_DIR, filename)
        np.save(filepath, field)

        meta["finger_data"].append({
            "hand": hand_side,
            "finger": name,
            "x": finger["x"] + ox,
            "y": finger["y"] + oy,
            "file": filename
        })

meta_path = os.path.join(OUTPUT_DIR, "fields_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=4)