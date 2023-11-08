import json
import re
from argparse import ArgumentParser

import numpy as np
import stable_baselines3.common.utils
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

from envs.custom_fetch import CustomFetchEnv
from speaker.baseline_speaker import BaselineSpeaker
from speaker.heuristic_speaker import HeuristicSpeaker
from util.callbacks import LogSuccessCallback
from util.wrappers import apply_wrappers


def eval_model(model_name: str, env_config: str, device: int = 0):
    with open(env_config, 'r') as file:
        test_config = json.load(file)

    match = re.match(r"(?:\d{8}_\d{6}_)([^_]+?)(?:_(fs))?_([^_]+?)_(\d+)x(\d+)_(\d+)d_([^_]+)", model_name.lower())
    if match:
        model_type = match[1]
        fs = match[2]
        env_name = match[3]
        env_size = int(match[4])
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
        "render_mode": "rgb_array",
        "tile_size": 8,
        "highlight": False,
        "agent_pov": True,
        "agent_view_size": 3,
    }

    test_kwargs = {**env_kwargs, "configs": test_config}
    test_env = make_vec_env(CustomFetchEnv, n_envs=1, seed=150494, wrapper_class=apply_wrappers, env_kwargs=test_kwargs)
    if fs:
        test_env = VecFrameStack(test_env, n_stack=3)

    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(0.5, device)

    if model_type == "ppo":
        model = PPO.load(f"../checkpoints/{model_name}/best_model.zip", env=None)
    elif model_type == "rppo":
        model = RecurrentPPO.load(f"../checkpoints/{model_name}/best_model.zip", env=None)
    else:
        raise ValueError(f"Model type could not be determined from {model_type}.")

    callback = LogSuccessCallback()
    callback.on_eval_start()
    stable_baselines3.common.utils.set_random_seed(150494)
    rewards, episode_lengths = evaluate_policy(model, test_env,
                                               n_eval_episodes=10,
                                               deterministic=True,
                                               callback=callback.on_eval_step,
                                               return_episode_rewards=True,
                                               )
    success_rate = callback.on_eval_end()

    mean_reward = np.mean(rewards)
    mean_epl = np.mean(episode_lengths)
    detailed_results = {'model': model_name, 'env': env_config[5:],
                        'mean_reward': mean_reward, 'mean_epl': mean_epl, 'success_rate': success_rate,
                        'rewards': rewards, 'episode_lengths': episode_lengths}

    return mean_reward, mean_epl, success_rate, detailed_results


if __name__ == '__main__':
    env_config = 'data/fetch_12x12_5d_test.json'
    model_name = '20231029_161220_PPO_fs_fetch_12x12_5d_heuristic'
    mean_reward, mean_epl, success_rate, detailed_results = eval_model(model_name, env_config, 0)
    print(f"model: {model_name}, env: {env_config}")
    print(f"mean reward: {mean_reward:.4f}, mean_epl: {mean_epl:.4f}, success_rate: {success_rate:.4f}")
    print(f"----------------------------------------------------")

    # detailed_result_list = []
    # for env_config in [
    #     'data/fetch_6x6_2d_test.json', 'data/fetch_12x12_5d_test.json', 'data/fetch_12x12_8d_test.json',
    # ]:
    #     for speaker in ['baseline', 'heuristic']:
    #         model_name = f"PPO_fetch_6x6_2d_{speaker}"
    #         mean_reward, mean_epl, success_rate, detailed_results = eval_model(model_name, env_config)
    #         detailed_result_list.append(detailed_results)
    #         print(f"model: {model_name}, env: {env_config}")
    #         print(f"mean reward: {mean_reward:.4f}, mean episode_length: {mean_epl:.4f}, success rate: {success_rate:.4f}")
    #         print(f"----------------------------------------------------")
    # with open('eval_results_2.json', 'w') as file:
    #     json.dump(detailed_result_list, file)
