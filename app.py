import os
import math
import tempfile

import cv2
import numpy as np
import pandas as pd
import imageio
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Car RL Playground",
    page_icon="🏎️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRACK_PATH = "custom_track.png"
MODEL_PATH = "custom_model.zip"
SIZE = 400
CHUNK_SIZE = 5000
MAX_EVAL_STEPS = 2000

REWARD_DEFAULTS = {
    "crash_penalty": -15.0,
    "center_weight": 1.5,
    "speed_weight": 0.3,
    "revisit_penalty": 0.5,
}

TRACK_SOURCES = ["Draw", "Circle", "Oval", "Square", "Figure eight"]

# RGB render colors
COLOR_BG = (8, 9, 11)
COLOR_TRACK = (36, 40, 50)
COLOR_TRAIL = (95, 104, 120)
COLOR_CAR = (76, 141, 255)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
    --base: #08090B;
    --surface: #0F1114;
    --raised: #151820;
    --line: #1C1F26;
    --text: #E4E6EA;
    --muted: #7D8491;
    --accent: #4C8DFF;
    --accent-hover: #6AA0FF;
}

html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp { background: var(--base); color: var(--text); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display: none; }
.block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 4rem; }

h1 { font-weight: 600; font-size: 1.6rem; letter-spacing: -0.02em; padding: 0; }
h3 { font-weight: 600; font-size: 1rem; letter-spacing: -0.01em; padding: 0 0 0.25rem 0; }
div[data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
label p { font-size: 0.82rem; font-weight: 500; color: var(--muted) !important; }

/* Panels: only containers created with key="panel_*" */
div[class*="st-key-panel"] {
    background: var(--surface);
    border: 1px solid var(--line) !important;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
}

/* Tabs */
button[data-baseweb="tab"] { background: transparent; color: var(--muted); font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--text); }
div[data-baseweb="tab-highlight"] { background-color: var(--accent); height: 2px; }
div[data-baseweb="tab-border"] { background-color: var(--line); }

/* Inputs */
div[data-baseweb="input"], div[data-baseweb="base-input"] {
    background: var(--raised) !important;
    border-color: var(--line) !important;
    border-radius: 8px !important;
}
div[data-baseweb="input"]:focus-within { border-color: var(--accent) !important; }

/* Buttons */
div.stButton > button, div.stDownloadButton > button {
    border-radius: 8px;
    font-weight: 500;
    border: 1px solid var(--line);
    background: var(--raised);
    color: var(--text);
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    border-color: var(--muted); color: var(--text);
}
div.stButton > button[kind="primary"] {
    background: var(--accent); border-color: var(--accent); color: #fff;
    padding: 0.65rem 1.2rem;
}
div.stButton > button[kind="primary"]:hover {
    background: var(--accent-hover); border-color: var(--accent-hover);
}

/* Metrics */
div[data-testid="stMetric"] {
    background: var(--raised);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
}
div[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: 0.78rem; }
div[data-testid="stMetricValue"] {
    font-size: 1.25rem; font-weight: 600; font-variant-numeric: tabular-nums;
}

/* Progress */
div[data-testid="stProgress"] > div > div { background: var(--line); height: 6px; border-radius: 6px; }
div[data-testid="stProgress"] > div > div > div { background: var(--accent); border-radius: 6px; }

/* Alerts */
div[data-testid="stAlert"] {
    background: var(--raised); border: 1px solid var(--line); border-radius: 10px;
}

/* Media */
video, img { border-radius: 10px; }
iframe[title*="st_canvas"] { border: 1px solid var(--line); border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------
class TrackEnv(gym.Env):
    def __init__(self, track_path=TRACK_PATH):
        super().__init__()
        self.track_path = track_path
        self._load_track()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        self.visited_cells = set()
        self.car_length = 20
        self.car_width = 10
        self.acceleration_factor = 0.2
        self.braking_factor = 0.4
        self.friction_factor = 0.03
        self.max_speed = 5.0
        self.turn_rate = 0.08

    def _load_track(self):
        img = cv2.imread(self.track_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = preset_track("Circle")
            cv2.imwrite(self.track_path, img)
        _, self.track = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        self.height, self.width = self.track.shape

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_track()
        y_indices, x_indices = np.where(self.track == 1)
        if len(y_indices) == 0:
            self.car_x = float(self.width // 2)
            self.car_y = float(self.height // 2)
        else:
            self.car_x = float(x_indices[0])
            self.car_y = float(y_indices[0])
        self.car_angle = 0.0
        self.car_speed = 0.0
        self.visited_cells.clear()
        return self._get_obs(), {}

    def _get_radar_distance(self, angle_offset):
        ray_angle = self.car_angle + angle_offset
        dx, dy = math.cos(ray_angle), math.sin(ray_angle)
        for distance in range(1, 100):
            check_x = int(self.car_x + dx * distance)
            check_y = int(self.car_y + dy * distance)
            if (check_x < 0 or check_x >= self.width or
                    check_y < 0 or check_y >= self.height or
                    self.track[check_y, check_x] == 0):
                return distance / 100.0
        return 1.0

    def _get_obs(self):
        radar_angles = [-math.pi / 3, -math.pi / 6, 0, math.pi / 6, math.pi / 3]
        radars = [self._get_radar_distance(a) for a in radar_angles]
        return np.array(radars + [self.car_speed / self.max_speed], dtype=np.float32)


class CustomizableTrackEnv(TrackEnv):
    def __init__(self, track_path=TRACK_PATH, crash_p=-15.0, center_w=1.5, speed_w=0.3, revisit_p=0.5):
        super().__init__(track_path)
        self.crash_penalty = crash_p
        self.center_weight = center_w
        self.speed_weight = speed_w
        self.revisit_penalty = revisit_p

    def step(self, action):
        if action == 0:
            self.car_angle -= self.turn_rate
        elif action == 1:
            self.car_angle += self.turn_rate

        if action == 2:
            self.car_speed += self.acceleration_factor
        elif action == 3:
            self.car_speed -= self.braking_factor

        self.car_speed -= self.friction_factor
        self.car_speed = max(0.0, min(self.max_speed, self.car_speed))

        self.car_x += math.cos(self.car_angle) * self.car_speed
        self.car_y += math.sin(self.car_angle) * self.car_speed

        terminated = False
        reward = 0.1

        x_int, y_int = int(self.car_x), int(self.car_y)
        center_x, center_y = self.width / 2, self.height / 2
        distance_from_center = math.sqrt((x_int - center_x) ** 2 + (y_int - center_y) ** 2)
        max_possible_distance = math.sqrt(center_x ** 2 + center_y ** 2)
        normalized_distance = distance_from_center / max_possible_distance if max_possible_distance > 0 else 0

        reward += (1 - normalized_distance) * self.center_weight
        reward += self.car_speed * self.speed_weight

        if (x_int < 0 or x_int >= self.width or y_int < 0 or y_int >= self.height
                or self.track[y_int, x_int] == 0):
            terminated = True
            reward = self.crash_penalty
        else:
            current_cell = (x_int, y_int)
            if current_cell in self.visited_cells:
                reward -= self.revisit_penalty
            else:
                self.visited_cells.add(current_cell)

        return self._get_obs(), reward, terminated, False, {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def preset_track(name):
    img = np.zeros((SIZE, SIZE), dtype=np.uint8)
    c = SIZE // 2
    if name == "Circle":
        cv2.circle(img, (c, c), 130, 255, thickness=40)
    elif name == "Oval":
        cv2.ellipse(img, (c, c), (155, 100), 0, 0, 360, 255, thickness=40)
    elif name == "Square":
        cv2.rectangle(img, (75, 75), (SIZE - 75, SIZE - 75), 255, thickness=40)
    elif name == "Figure eight":
        cv2.circle(img, (c, 115), 80, 255, thickness=36)
        cv2.circle(img, (c, 285), 80, 255, thickness=36)
    return img


def canvas_to_track(canvas_result):
    if canvas_result is None or canvas_result.image_data is None:
        return None
    rgba = canvas_result.image_data.astype(np.uint8)
    if not np.any(rgba[:, :, :3] > 10):
        return None
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    if binary.shape != (SIZE, SIZE):
        binary = cv2.resize(binary, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return binary


def track_preview(binary):
    """Themed RGB preview with the start position marked."""
    img = np.empty((*binary.shape, 3), dtype=np.uint8)
    img[:] = COLOR_BG
    img[binary > 127] = COLOR_TRACK
    ys, xs = np.where(binary > 127)
    if len(ys):
        start = (int(xs[0]), int(ys[0]))
        cv2.circle(img, start, 7, COLOR_CAR, -1, cv2.LINE_AA)
        cv2.arrowedLine(img, start, (start[0] + 26, start[1]), COLOR_CAR, 2, cv2.LINE_AA, tipLength=0.4)
    return img


def parse_layers(text):
    try:
        layers = [int(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError:
        return None
    if not layers or any(n <= 0 for n in layers):
        return None
    return layers


def render_rollout(track, states, car_length, car_width):
    """Draw each recorded (x, y, angle) state as an RGB frame with a trail."""
    base = np.empty((*track.shape, 3), dtype=np.uint8)
    base[:] = COLOR_BG
    base[track == 1] = COLOR_TRACK

    hl, hw = car_length / 2, car_width / 2
    corners_rel = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])

    trail = base.copy()
    frames = []
    prev = None
    for x, y, angle in states:
        pt = (int(round(x)), int(round(y)))
        if prev is not None:
            cv2.line(trail, prev, pt, COLOR_TRAIL, 2, cv2.LINE_AA)
        prev = pt

        frame = trail.copy()
        c, s = math.cos(angle), math.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        corners = (corners_rel @ rot.T + np.array([x, y])).astype(np.int32)
        cv2.fillPoly(frame, [corners], COLOR_CAR, cv2.LINE_AA)
        frames.append(frame)
    return frames


def encode_video(frames, fps=30):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    try:
        imageio.mimsave(path, frames, fps=fps)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)


def reset_rewards():
    for key, value in REWARD_DEFAULTS.items():
        st.session_state[key] = value


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, value in REWARD_DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("history", [])
st.session_state.setdefault("best_video", None)
st.session_state.setdefault("total_steps", 0)
st.session_state.setdefault("run_note", "")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Car RL Playground")
st.caption("Pick or draw a track, tune the reward, and train a PPO agent to drive it.")
st.write("")

# ---------------------------------------------------------------------------
# Setup panels
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    with st.container(key="panel_track"):
        st.subheader("Track")
        source = st.radio("Source", TRACK_SOURCES, horizontal=True, label_visibility="collapsed")

        canvas_result = None
        if source == "Draw":
            brush_width = st.slider("Brush width", 10, 80, 30)
            canvas_result = st_canvas(
                fill_color="#FFFFFF",
                stroke_width=brush_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=SIZE,
                width=SIZE,
                drawing_mode="freedraw",
                return_image_data=True,
                update_streamlit=True,
                key="canvas",
            )
            st.caption("Use the canvas toolbar to undo or clear. "
                       "The car starts at the topmost point of the track, facing right.")
        else:
            st.image(track_preview(preset_track(source)), width=SIZE)
            st.caption("Blue marker shows where the car starts and which way it faces.")

with col_right:
    with st.container(key="panel_settings"):
        st.subheader("Settings")
        tab_train, tab_reward = st.tabs(["Training", "Rewards"])

        with tab_train:
            steps = st.slider("Total timesteps", 5000, 100000, 20000, 5000)
            lr = st.number_input("Learning rate", value=0.0003, min_value=0.000001,
                                 max_value=0.1, format="%.5f", step=0.0001)
            net_arch = st.text_input("Network layers", "64, 64",
                                     help="Comma-separated hidden layer sizes, e.g. 128, 64")

            has_existing_model = os.path.exists(MODEL_PATH)
            load_saved_weights = st.toggle(
                "Resume from saved weights",
                value=False,
                disabled=not has_existing_model,
                help="Continues training the last saved model. Network layers are ignored when resuming.",
            )
            if not has_existing_model:
                st.caption("No saved weights yet.")

        with tab_reward:
            st.slider("Crash penalty", -50.0, 0.0, key="crash_penalty",
                      help="Reward given when the car leaves the track. Ends the episode.")
            st.slider("Centerline reward", 0.0, 5.0, key="center_weight",
                      help="Bonus for staying near the center of the canvas.")
            st.slider("Speed reward", 0.0, 2.0, key="speed_weight",
                      help="Bonus proportional to current speed.")
            st.slider("Revisit penalty", 0.0, 2.0, key="revisit_penalty",
                      help="Penalty for driving over a pixel already visited this episode.")
            st.button("Reset to defaults", on_click=reset_rewards)

st.write("")
train_clicked = st.button("Train and simulate", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Validate and prepare a run
# ---------------------------------------------------------------------------
reward_cfg = dict(
    crash_p=st.session_state.crash_penalty,
    center_w=st.session_state.center_weight,
    speed_w=st.session_state.speed_weight,
    revisit_p=st.session_state.revisit_penalty,
)

if train_clicked:
    layers = parse_layers(net_arch)
    if layers is None and not (load_saved_weights and has_existing_model):
        st.error("Network layers should be positive whole numbers separated by commas, like 64, 64.")
        st.stop()

    notes = []
    track = canvas_to_track(canvas_result) if source == "Draw" else preset_track(source)
    if track is None:
        track = preset_track("Circle")
        notes.append("Canvas was empty, so the circle track was used.")
    cv2.imwrite(TRACK_PATH, track)

    st.session_state.history = []
    st.session_state.best_video = None
    st.session_state.total_steps = int(steps)

    custom_env = CustomizableTrackEnv(track_path=TRACK_PATH, **reward_cfg)
    if load_saved_weights and has_existing_model:
        lr_value = float(lr)
        ppo_model = PPO.load(
            MODEL_PATH, env=custom_env,
            custom_objects={"learning_rate": lr_value, "lr_schedule": lambda _: lr_value},
        )
        notes.insert(0, "Resumed from saved weights.")
    else:
        ppo_model = PPO(
            "MlpPolicy", custom_env, verbose=0, learning_rate=float(lr),
            policy_kwargs=dict(net_arch=dict(pi=layers, vf=layers)),
        )
        notes.insert(0, "Fresh training run.")
    st.session_state.run_note = " ".join(notes)

# ---------------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------------
history = st.session_state.history
show_results = train_clicked or bool(history)

if show_results:
    st.write("")
    with st.container(key="panel_results"):
        st.subheader("Training")
        if st.session_state.run_note:
            st.caption(st.session_state.run_note)

        progress_ph = st.empty()
        m1, m2, m3, m4 = st.columns(4)
        metric_phs = [m1.empty(), m2.empty(), m3.empty(), m4.empty()]

        st.write("")
        v_col, c_col = st.columns([1, 1.1], gap="large")
        with v_col:
            st.caption("Best run so far")
            video_ph = st.empty()
        with c_col:
            st.caption("Score per checkpoint")
            chart_ph = st.empty()

        footer = st.container()

    def draw_results(done_steps, refresh_video=True):
        total = st.session_state.total_steps
        hist = st.session_state.history
        last = hist[-1] if hist else None

        progress_ph.progress(min(1.0, done_steps / total) if total else 0.0)
        metric_phs[0].metric("Steps", f"{done_steps:,} / {total:,}")
        metric_phs[1].metric("Checkpoint score", f"{last['Score']:.1f}" if last else "–")
        metric_phs[2].metric("Best score", f"{last['Best']:.1f}" if last else "–")
        metric_phs[3].metric("Episode length", f"{last['Length']:,}" if last else "–")

        if not refresh_video:
            pass
        elif st.session_state.best_video:
            video_ph.video(st.session_state.best_video, format="video/mp4",
                           loop=True, autoplay=True, muted=True)
        else:
            video_ph.info("The first simulation appears after one checkpoint.")

        if hist:
            df = pd.DataFrame(hist).set_index("Steps")[["Score", "Best"]]
            chart_ph.line_chart(df, color=["#7D8491", "#4C8DFF"], height=300)
        else:
            chart_ph.info(f"A checkpoint is evaluated every {CHUNK_SIZE:,} steps.")

    if train_clicked:
        total_timesteps = st.session_state.total_steps
        current_steps = 0
        best_reward = -float("inf")
        draw_results(0)

        while current_steps < total_timesteps:
            steps_to_run = min(CHUNK_SIZE, total_timesteps - current_steps)
            ppo_model.learn(total_timesteps=steps_to_run, reset_num_timesteps=False)
            current_steps += steps_to_run

            # Evaluate this checkpoint, recording states only
            eval_env = CustomizableTrackEnv(track_path=TRACK_PATH, **reward_cfg)
            obs, _ = eval_env.reset()
            episode_reward = 0.0
            states = [(eval_env.car_x, eval_env.car_y, eval_env.car_angle)]
            for _ in range(MAX_EVAL_STEPS):
                action, _ = ppo_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                states.append((eval_env.car_x, eval_env.car_y, eval_env.car_angle))
                if terminated or truncated:
                    break

            # Render video only when this checkpoint is the new best
            new_best = episode_reward > best_reward
            if new_best:
                best_reward = episode_reward
                ppo_model.save(MODEL_PATH)
                frames = render_rollout(eval_env.track, states, eval_env.car_length, eval_env.car_width)
                st.session_state.best_video = encode_video(frames)

            st.session_state.history.append({
                "Steps": current_steps,
                "Score": round(float(episode_reward), 2),
                "Best": round(float(best_reward), 2),
                "Length": len(states) - 1,
            })
            draw_results(current_steps, refresh_video=new_best)

        st.toast(f"Training complete. Best score {best_reward:.1f}")
    else:
        draw_results(history[-1]["Steps"] if history else 0)

    if os.path.exists(MODEL_PATH):
        with footer:
            st.write("")
            with open(MODEL_PATH, "rb") as file:
                st.download_button(
                    label="Download weights (.zip)",
                    data=file,
                    file_name="custom_model.zip",
                    mime="application/zip",
                )
