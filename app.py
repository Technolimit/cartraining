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
import shutil

# 1. MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(layout="wide")

# Force Dark UI Theme using CSS
st.markdown(
    """<style>
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}
div[data-testid="stExpander"] {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
}
div[data-testid="stSidebar"] {
    background-color: #161B22;
}
.card {
    background-color: #161B22;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #30363D;
    margin-bottom: 1rem;
}
</style>""",
    unsafe_allow_html=True
)

# 2. ORIGINAL ENVIRONMENT CLASS FROM NOTEBOOK
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

st.title("🏎️ Autonomous Car RL Playground (Streamlit Dark)")

# Main screen columns layout
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown(\"<div class='card'><h3>🎨 Draw Your Custom Track</h3><p>Use the white brush below to draw a path over the black track background.</p></div>\", unsafe_allow_html=True)
    
    # Brush settings
    brush_width = st.slider("Brush Width", 10, 80, 25)
    
    # Drawable Canvas integration
    canvas_result = st_canvas(
        fill_color="#FFFFFF",
        stroke_width=brush_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )

with col_right:
    st.markdown(\"<div class='card'><h3>⚙️ Model Settings</h3></div>\", unsafe_allow_html=True)
    steps = st.slider("Total Timesteps", 5000, 100000, 20000, 5000)
    lr = st.number_input("Learning Rate", value=0.0003, format="%.5f")
    net_arch = st.text_input("Policy Network Layers", "64, 64")

    has_existing_model = os.path.exists("custom_model.zip")
    load_saved_weights = st.checkbox("Resume training from previously saved model weights", value=False, disabled=not has_existing_model)

    st.markdown(\"<div class='card'><h3>🏆 Customize Reward Weights</h3></div>\", unsafe_allow_html=True)
    crash_penalty = st.slider("Crash Penalty (Negative)", -50.0, 0.0, -15.0)
    center_weight = st.slider("Centerline Follow Reward", 0.0, 5.0, 1.5)
    speed_weight = st.slider("Speed Incentive", 0.0, 2.0, 0.3)
    revisit_penalty = st.slider("Revisitation Penalty", 0.0, 2.0, 0.5)

if st.button("Train Model & Generate Simulation", type="primary"):
    with st.spinner("Running training configuration..."):
        # Track creation logic from canvas
        if canvas_result.image_data is not None:
            # Convert RGBA to Grayscale
            img_rgba = canvas_result.image_data.astype(np.uint8)
            img_gray = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)
            # Ensure clean binary image
            _, img_thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
            cv2.imwrite('custom_track.png', img_thresh)
        else:
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
            shutil.unpack_archive("custom_model.zip", "extracted_model", "zip")
            ppo_model = PPO.load("extracted_model/custom_model", env=custom_env, learning_rate=float(lr))
            st.info("Loaded existing model weights. Continuing training on new track...")
        else:
            ppo_model = PPO(
                "MlpPolicy", custom_env, verbose=0, learning_rate=float(lr),
                policy_kwargs=dict(net_arch=dict(pi=layers, vf=layers))
            )
            st.info("Starting fresh training session...")

        ppo_model.learn(total_timesteps=int(steps))

        # Save model inside a dedicated folder before archiving
        os.makedirs("model_dir", exist_ok=True)
        ppo_model.save("model_dir/custom_model")

        if os.path.exists("custom_model.zip"):
            os.remove("custom_model.zip")
        shutil.make_archive("custom_model", 'zip', "model_dir")

        # Clean up temporary directories
        shutil.rmtree("model_dir", ignore_errors=True)
        shutil.rmtree("extracted_model", ignore_errors=True)

        # ORIGINAL DETAILED SIMULATION & CAR ANGLE RENDERING PIPELINE
        obs, _ = custom_env.reset()
        base_track_img = cv2.imread('custom_track.png')
        if base_track_img is not None and len(base_track_img.shape) == 2:
            base_track_img = cv2.cvtColor(base_track_img, cv2.COLOR_GRAY2BGR)

        frames = []
        step_count = 0
        max_render_steps = 2000

        while step_count < max_render_steps:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = custom_env.step(action)

            current_render_img = base_track_img.copy()
            
            # Accurate corner-math from the notebook to render a properly rotated car
            car_len = custom_env.car_length
            car_wid = custom_env.car_width
            half_len = car_len / 2
            half_wid = car_wid / 2
            car_center_x, car_center_y = custom_env.car_x, custom_env.car_y
            car_angle_rad = custom_env.car_angle

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

            cv2.fillPoly(current_render_img, [final_corners], (0, 0, 255)) # Draw original blue/red car
            frames.append(current_render_img)

            step_count += 1
            if terminated or truncated:
                break

        imageio.mimsave("custom_simulation.mp4", frames, fps=20)
        st.success("Training Complete!")

# Persist display components layout
if os.path.exists("custom_simulation.mp4"):
    col_video, col_download = st.columns(2)
    with col_video:
        st.subheader("Simulation Run")
        st.video("custom_simulation.mp4")
    with col_download:
        st.subheader("Model Weights")
        if os.path.exists("custom_model.zip"):
            with open("custom_model.zip", "rb") as file:
                st.download_button(
                    label="Download Weights (ZIP)",
                    data=file,
                    file_name="custom_model.zip",
                    mime="application/zip"
                )
