import json
import re
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Type

import numpy as np
import stable_baselines3.common.utils
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack
from tqdm import tqdm

from envs.follower_env import FollowerEnv
from envs.shared_env import SharedEnv
from follower.heuristic_follower import HeuristicFollower
from speaker.baseline_speaker import BaselineSpeaker
from speaker.heuristic_speaker import HeuristicSpeaker
from speaker.speaker import Speaker
from util.callbacks import LogSuccessCallback
from util.wrappers import apply_wrappers


def eval_follower(env: Path, env_size: int, device: int = 0):
    test_config = json.loads(env.read_text("utf-8"))

    test_kwargs = {
        "size": env_size,
        "configs": test_config,
        "speaker": HeuristicSpeaker,
        "follower": HeuristicFollower,
        "render_mode": "rgb_array",
        "tile_size": 8,
        "highlight": False,
        "agent_pov": True,
        "agent_view_size": 3,
    }

    test_env = make_vec_env(SharedEnv, n_envs=1, seed=150494, wrapper_class=apply_wrappers, env_kwargs=test_kwargs)

    rewards = []
    episode_lengths = []
    successes = []

    for i in range(len(test_config)):
        test_env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < test_env.envs[0].max_steps:
            test_env.envs[0].make_speaker_act()
            _, reward, terminated, _, _ = test_env.envs[0].make_follower_act()
            steps += 1
        rewards.append(reward)
        episode_lengths.append(steps)
        successes.append(int(reward > 0))

    success_rate = np.mean(successes)
    mean_reward = np.mean(rewards)
    mean_epl = np.mean(episode_lengths)
    detailed_results = {'model': "heuristics", 'env': env_config.stem,
                        'mean_reward': mean_reward, 'mean_epl': mean_epl, 'success_rate': success_rate,
                        'rewards': rewards, 'episode_lengths': episode_lengths}

    return mean_reward, mean_epl, success_rate, detailed_results

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for env_config in Path('../data').glob('*test*.json'):
            mean_reward, mean_epl, success_rate, detailed_results = eval_follower(env_config, int(
                    re.search(r'(\d+)x\1', str(env_config))[1]), 0)
            print(f"model: heuristics, env: {env_config}")
            print(f"mean reward: {mean_reward:.4f}, mean_epl: {mean_epl:.4f}, success_rate: {success_rate:.4f}")
            print(f"----------------------------------------------------")
