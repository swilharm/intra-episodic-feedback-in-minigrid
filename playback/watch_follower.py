import json
import warnings
from pathlib import Path

import pygame
from minigrid.core.actions import Actions
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import VecFrameStack, VecEnv

from envs.follower_env import FollowerEnv
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
        else:
            print(key)


if __name__ == '__main__':

    model_name = '20240429_213324_ppo_fs_follower_9x9_4d_heuristic'
    model_path = (Path('~') / "checkpoints" / model_name / "best_model").expanduser()
    model_type, frame_stacking, speaker = 'ppo', True, HeuristicSpeaker
    env = Path('data') / 'fetch_9x9_4d_test.json'
    env_size = 9

    test_config = json.loads(env.read_text("utf-8"))

    env_kwargs = {
        "configs": test_config,
        "size": env_size,
        "speaker": speaker,
        "render_mode": "human",
        "tile_size": 64,
        "highlight": True,
        "agent_pov": False,
        "agent_view_size": 3,
    }
    test_env = make_vec_env(FollowerEnv, n_envs=1, seed=150494, wrapper_class=apply_wrappers,
                            env_kwargs=env_kwargs)
    if frame_stacking:
        test_env = VecFrameStack(test_env, n_stack=3)

    if model_type == "ppo":
        model = PPO.load(model_path, env=None)
    elif model_type == "rppo":
        model = RecurrentPPO.load(model_path, env=None)
    else:
        raise ValueError(f"Model type could not be determined from {model_type}.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        WatchModel(model, test_env).start()
