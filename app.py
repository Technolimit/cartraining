import os
import math
import cv2
import numpy as np
import imageio
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# 1. MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Car RL Playground",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# THEME
# Base      #08090B   page
# Surface   #0F1114   panels
# Raised    #151820   inputs / hover
# Line      #1C1F26   borders
# Text      #E4E6EA   primary text
# Muted     #7D8491   secondary text
# Accent    #4C8DFF   primary action
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

/* Page */
.stApp { background: var(--base); color: var(--text); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display: none; }
.block-container { max-width: 1180px; padding-top: 3rem; padding-bottom: 4rem; }

/* Type */
h1 { font-weight: 600; font-size: 1.75rem; letter-spacing: -0.02em; padding: 0 0 0.25rem 0; }
h2, h3 { font-weight: 600; letter-spacing: -0.01em; color: var(--text); }
h3 { font-size: 1.05rem; padding-top: 0; }
p, label, span, li { color: var(--text); }
div[data-testid="stCaptionContainer"], .stCaption, small { color: var(--muted) !important; }
label p { font-size: 0.85rem; font-weight: 500; color: var(--muted) !important; }

/* Panels (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {
    border-color: var(--line);
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.5rem 0.75rem;
}

/* Tabs */
button[data-baseweb="tab"] { background: transparent; color: var(--muted); font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--text); }
div[data-baseweb="tab-highlight"] { background-color: var(--accent); height: 2px; }
div[data-baseweb="tab-border"] { background-color: var(--line); }

/* Inputs */
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="select"] > div {
    background: var(--raised) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
}
div[data-baseweb="input"]:focus-within { border-color: var(--accent) !important; }
input, textarea { color: var(--text) !important; }
button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] {
    background: var(--raised); color: var(--muted);
}

/* Sliders */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background: var(--line); }
div[data-testid="stSlider"] [role="slider"] {
    background: var(--accent); box-shadow: none; border: 2px solid var(--surface);
}
div[data-testid="stSliderThumbValue"], div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] {
    color: var(--muted);
}

/* Checkbox */
label[data-baseweb="checkbox"] span { color: var(--muted); }

/* Buttons */
div.stButton > button, div.stDownloadButton > button {
    border-radius: 8px;
    font-weight: 500;
    border: 1px solid var(--line);
    background: var(--raised);
    color: var(--text);
    transition: background 0.15s, border-color 0.15s;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    border-color: var(--muted); color: var(--text);
}
div.stButton > button[kind="primary"] {
    background: var(--accent); border-color: var(--accent); color: #fff;
    padding: 0.6rem 1.2rem;
}
div.stButton > button[kind="primary"]:hover {
    background: var(--accent-hover); border-color: var(--accent-hover);
}

/* Progress */
div[data-testid="stProgress"] > div > div { background: var(--line); height: 6px; border-radius: 6px; }
div[data-testid="stProgress"] > div > div > div { background: var(--accent); border-radius: 6px; }

/* Alerts */
div[data-testid="stAlert"] {
    background: var(--surface); border: 1px solid var(--line); border-radius: 10px; color: var(--text);
}

/* Media */
video, iframe { border-radius: 10px; }
iframe[title*="st_canvas"] { border: 1px solid var(--line); border-radius: 10px; background: #000; }

/* Dividers */
hr { border-color: var(--line); }
</style>
""",
    unsafe_allow_html=True,
)


# 2. ENVIRONMENT CLASSES
class TrackEnv(gym.Env):
    def __init__(self, track_path='custom_track.png'):
        super(TrackEnv, self).__init__()
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
            img = np.zeros((400, 400), dtype=np.uint8)
            cv2.circle(img, (200, 200), 120, 255, thickness=40)
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
        radar_angles = [-math.pi/3, -math.pi/6, 0, math.pi/6, math.pi/3]
        radars = [self._get_radar_distance(a) for a in radar_angles]
        return np.array(radars + [self.car_speed / self.max_speed], dtype=np.float32)


class CustomizableTrackEnv(TrackEnv):
    def __init__(self, track_path='custom_track.png', crash_p=-15.0, center_w=1.5, speed_w=0.3, revisit_p=0.5):
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
        distance_from_center = math.sqrt((x_int - center_x)**2 + (y_int - center_y)**2)
        max_possible_distance = math.sqrt(center_x**2 + center_y**2)
        normalized_distance = distance_from_center / max_possible_distance if max_possible_distance > 0 else 0

        reward += (1 - normalized_distance) * self.center_weight
        reward += self.car_speed * self.speed_weight

        if (x_int < 0 or x_int >= self.width or y_int < 0 or y_int >= self.height or self.track[y_int, x_int] == 0):
            terminated = True
            reward = self.crash_penalty
        else:
            current_cell = (x_int, y_int)
            if current_cell in self.visited_cells:
                reward -= self.revisit_penalty
            else:
                self.visited_cells.add(current_cell)

        return self._get_obs(), reward, terminated, False, {}


# 3. UI
st.title("Car RL Playground")
st.caption("Draw a track, tune the reward, and watch a PPO agent learn to drive it.")

st.write("")
col_left, col_right = st.columns([1.15, 1], gap="large")

with col_left:
    with st.container(border=True):
        st.subheader("Track")
        st.caption("Draw with the white brush. The car starts at the top-left of your path.")
        brush_width = st.slider("Brush width", 10, 80, 25)
        canvas_result = st_canvas(
            fill_color="#FFFFFF",
            stroke_width=brush_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=400,
            width=400,
            drawing_mode="freedraw",
            return_image_data=True,
            update_streamlit=True,
            key="canvas",
        )

with col_right:
    with st.container(border=True):
        tab_train, tab_reward = st.tabs(["Training", "Rewards"])

        with tab_train:
            steps = st.slider("Total timesteps", 5000, 100000, 20000, 5000)
            lr = st.number_input("Learning rate", value=0.0003, format="%.5f")
            net_arch = st.text_input("Network layers", "64, 64", help="Comma-separated layer sizes")

            has_existing_model = os.path.exists("custom_model.zip")
            load_saved_weights = st.checkbox(
                "Resume from saved weights",
                value=False,
                disabled=not has_existing_model,
            )

        with tab_reward:
            crash_penalty = st.slider("Crash penalty", -50.0, 0.0, -15.0)
            center_weight = st.slider("Centerline reward", 0.0, 5.0, 1.5)
            speed_weight = st.slider("Speed reward", 0.0, 2.0, 0.3)
            revisit_penalty = st.slider("Revisit penalty", 0.0, 2.0, 0.5)

st.write("")
train_clicked = st.button("Train and simulate", type="primary", use_container_width=True)

if train_clicked:
    # Track creation logic
    track_saved = False
    if canvas_result.image_data is not None:
        img_rgba = canvas_result.image_data.astype(np.uint8)
        if np.any(img_rgba[:, :, :3] > 10):
            img_gray = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)
            _, img_thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
            cv2.imwrite('custom_track.png', img_thresh)
            track_saved = True

    if not track_saved:
        img = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(img, (200, 200), 120, 255, thickness=40)
        cv2.imwrite('custom_track.png', img)

    custom_env = CustomizableTrackEnv(
        track_path='custom_track.png',
        crash_p=crash_penalty,
        center_w=center_weight,
        speed_w=speed_weight,
        revisit_p=revisit_penalty
    )

    layers = [int(x.strip()) for x in net_arch.split(',')]

    if load_saved_weights and has_existing_model:
        ppo_model = PPO.load("custom_model.zip", env=custom_env, learning_rate=float(lr))
        st.info("Loaded saved weights. Continuing training.")
    else:
        ppo_model = PPO(
            "MlpPolicy", custom_env, verbose=0, learning_rate=float(lr),
            policy_kwargs=dict(net_arch=dict(pi=layers, vf=layers))
        )
        st.info("Starting a fresh training run.")

    # Live progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    st.subheader("Best run so far")
    st.caption("Updates whenever a checkpoint beats the previous best score.")

    v_col1, v_col2, v_col3 = st.columns([1, 2, 1])
    with v_col2:
        live_video_placeholder = st.empty()

    total_timesteps = int(steps)
    chunk_size = 5000
    current_steps = 0
    best_reward = -float('inf')

    base_track_img = cv2.imread('custom_track.png')
    if base_track_img is not None and len(base_track_img.shape) == 2:
        base_track_img = cv2.cvtColor(base_track_img, cv2.COLOR_GRAY2BGR)

    while current_steps < total_timesteps:
        steps_to_run = min(chunk_size, total_timesteps - current_steps)
        ppo_model.learn(total_timesteps=steps_to_run, reset_num_timesteps=False)
        current_steps += steps_to_run

        # Evaluate current model checkpoint via rollout
        eval_env = CustomizableTrackEnv(
            track_path='custom_track.png',
            crash_p=crash_penalty,
            center_w=center_weight,
            speed_w=speed_weight,
            revisit_p=revisit_penalty
        )
        obs, _ = eval_env.reset()
        episode_reward = 0
        frames = []

        for _ in range(2000):
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward

            current_render_img = base_track_img.copy()
            car_len, car_wid = eval_env.car_length, eval_env.car_width
            half_len, half_wid = car_len / 2, car_wid / 2
            car_center_x, car_center_y = eval_env.car_x, eval_env.car_y
            car_angle_rad = eval_env.car_angle

            corners_rel = np.array([
                [-half_len, -half_wid],
                [ half_len, -half_wid],
                [ half_len,  half_wid],
                [-half_len,  half_wid]
            ])

            cos_angle = math.cos(car_angle_rad)
            sin_angle = math.sin(car_angle_rad)
            rotation_matrix = np.array([
                [cos_angle, -sin_angle],
                [sin_angle,  cos_angle]
            ])

            rotated_corners = np.dot(corners_rel, rotation_matrix.T)
            final_corners = (rotated_corners + np.array([car_center_x, car_center_y])).astype(int)

            cv2.fillPoly(current_render_img, [final_corners], (0, 0, 255))
            frames.append(current_render_img)

            if terminated or truncated:
                break

        # Update progress UI
        progress_bar.progress(min(1.0, current_steps / total_timesteps))
        status_text.caption(
            f"{current_steps:,} / {total_timesteps:,} steps  |  checkpoint score {episode_reward:.1f}"
        )

        # Save as best model and refresh preview using unique checkpoint filenames
        if episode_reward > best_reward:
            best_reward = episode_reward
            ppo_model.save("custom_model.zip")

            sim_filename = f"sim_step_{current_steps}.mp4"
            imageio.mimsave(sim_filename, frames, fps=20)

            with v_col2:
                live_video_placeholder.video(sim_filename)

    st.success(f"Training complete. Best score: {best_reward:.1f}")

if os.path.exists("custom_model.zip"):
    st.divider()
    with open("custom_model.zip", "rb") as file:
        st.download_button(
            label="Download weights (.zip)",
            data=file,
            file_name="custom_model.zip",
            mime="application/zip",
        )
