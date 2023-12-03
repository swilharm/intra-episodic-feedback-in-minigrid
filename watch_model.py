import json
import re

import pygame
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import VecFrameStack, VecEnv

from envs.follower_env import FollowerEnv
from envs.shared_env import SharedEnv
from speaker.baseline_speaker import BaselineSpeaker
from speaker.heuristic_speaker import HeuristicSpeaker
from util.wrappers import apply_wrappers


class WatchModel:

    def __init__(
            self,
            model: OnPolicyAlgorithm,
            env: VecEnv,
    ) -> None:
        self.model = model
        self.env = env
        self.closed = False
        self.obs = None

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset()

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.closed = True
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Actions):
        self.obs, reward, done, _ = self.env.step(action)
        print(f"step={self.env.envs[0].step_count}, reward={reward[0]:.2f}")
        self.env.render()

    def reset(self):
        self.obs = self.env.reset()
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.closed = True
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return
        if key == "space":
            actions, _ = self.model.predict(self.obs, deterministic=True)
            print(f"Model decided action {actions[0]}: {Actions(actions[0]).name}")
            self.step(actions)
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step([action])
        else:
            print(key)


if __name__ == '__main__':
    env_config = 'data/fetch_12x12_5d_test.json'
    model_name = '20231029_161129_PPO_fs_fetch_12x12_5d_baseline'

    with open(env_config, 'r') as file:
        test_config = json.load(file)

    match = re.match(r"(?:.*?/)?([^_]+?)_(\d+)x(\d+)_(\d+)d_([^_]+)(?:.json)?", env_config.lower())
    if match:
        env_type = match[1]
        env_size = int(match[2])
    else:
        raise ValueError(f"Cannot parse env config {env_config}")

    match = re.match(r"(?:\d{8}_\d{6}_)([^_]+?)(?:_(fs))?_([^_]+?)_(\d+)x(\d+)_(\d+)d_([^_]+)", model_name.lower())
    if match:
        model_type = match[1]
        fs = match[2]
        speaker_type = match[7]
    else:
        raise ValueError(f"Cannot parse model name {model_name}")

    if speaker_type == "baseline":
        speaker = BaselineSpeaker
    elif speaker_type == "heuristic":
        speaker = HeuristicSpeaker
    else:
        raise ValueError(f"Speaker could not be determined from {speaker_type}.")

    env_kwargs = {
        "size": env_size,
        "speaker": speaker,
        "render_mode": "human",
        "tile_size": 64,
        "agent_view_size": 3,
    }

    test_kwargs = {**env_kwargs, "configs": test_config}
    env = make_vec_env(FollowerEnv, n_envs=1, seed=150494, wrapper_class=apply_wrappers, env_kwargs=test_kwargs)
    if fs:
        env = VecFrameStack(env, n_stack=3)

    if model_type == "ppo":
        model = PPO.load(f"../checkpoints/{model_name}/best_model.zip", env=None)
    elif model_type == "rppo":
        model = RecurrentPPO.load(f"../checkpoints/{model_name}/best_model.zip", env=None)
    else:
        raise ValueError(f"Model type could not be determined from {model_type}.")

    WatchModel(model, env).start()
