import os
import math
import cv2
import numpy as np
import imageio
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import streamlit as st
import shutil

# 1. MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(layout="wide")

# Force Dark UI Theme using CSS
st.markdown(
    """<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div[data-testid="stExpander"] { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; }
    div[data-testid="stSidebar"] { background-color: #161B22; }
    </style>""", 
    unsafe_allow_html=True
)

# 2. FIXED ENVIRONMENT CLASS (Handles missing files dynamically)
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
            # Match x_indices with car_x (horizontal) and y_indices with car_y (vertical)
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

st.sidebar.header("1. Define Track")
uploaded_file = st.sidebar.file_uploader("Upload track layout (white track, black background)", type=["png", "jpg", "jpeg"])

st.sidebar.header("2. Model Settings")
steps = st.sidebar.slider("Total Timesteps", 5000, 100000, 20000, 5000)
lr = st.sidebar.number_input("Learning Rate", value=0.0003, format="%.5f")
net_arch = st.sidebar.text_input("Policy Network Layers", "64, 64")

has_existing_model = os.path.exists("custom_model.zip")
load_saved_weights = st.sidebar.checkbox("Resume training from previously saved model weights", value=False, disabled=not has_existing_model)

with st.sidebar.expander("🏆 Reward Tuning", expanded=True):
    crash_penalty = st.slider("Crash Penalty", -50.0, 0.0, -15.0)
    center_weight = st.slider("Centerline Follow Reward", 0.0, 5.0, 1.5)
    speed_weight = st.slider("Speed Incentive", 0.0, 2.0, 0.3)
    revisit_penalty = st.slider("Revisitation Penalty", 0.0, 2.0, 0.5)

if st.sidebar.button("Train Model & Generate Simulation", type="primary"):
    with st.spinner("Running training configuration..."):
        # Track creation logic
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (400, 400))
            cv2.imwrite('custom_track.png', img)
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

        # Simulation Generation Loop
        obs, _ = custom_env.reset()
        base_track_img = cv2.imread('custom_track.png')
        if base_track_img is not None and len(base_track_img.shape) == 2:
            base_track_img = cv2.cvtColor(base_track_img, cv2.COLOR_GRAY2BGR)
            
        frames = []
        for _ in range(500):
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, _, terminated, _, _ = custom_env.step(action)
            
            frame = base_track_img.copy()
            cv2.circle(frame, (int(custom_env.car_x), int(custom_env.car_y)), 6, (0, 0, 255), -1)
            frames.append(frame)
            if terminated:
                break
                
        imageio.mimsave("custom_simulation.mp4", frames, fps=20)
        st.success("Training Complete!")

# Persist display components layout 
if os.path.exists("custom_simulation.mp4"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Simulation Run")
        st.video("custom_simulation.mp4")
    with col2:
        st.subheader("Model Weights")
        if os.path.exists("custom_model.zip"):
            with open("custom_model.zip", "rb") as file:
                st.download_button(
                    label="Download Weights (ZIP)",
                    data=file,
                    file_name="custom_model.zip",
                    mime="application/zip"
                )
