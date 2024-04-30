import json
import re
import warnings
from pathlib import Path

import numpy as np
import stable_baselines3.common.utils
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

from envs.speaker_env import SpeakerEnv
from follower.heuristic_follower import HeuristicFollower
from util.callbacks import LogSuccessCallback
from util.wrappers import apply_wrappers


def eval_speaker(model_path: Path, model_type: str, frame_stacking: bool,
                 env: Path, env_size: int, device: int = 0):
    test_config = json.loads(env.read_text("utf-8"))

    env_kwargs = {
        "size": env_size,
        "follower": HeuristicFollower,
        "render_mode": "rgb_array",
        "tile_size": 8,
        "highlight": False,
        "agent_pov": True,
        "agent_view_size": 3,
    }

    test_kwargs = {**env_kwargs, "configs": test_config}
    test_env = make_vec_env(SpeakerEnv, n_envs=1, seed=150494, wrapper_class=apply_wrappers, env_kwargs=test_kwargs)
    if frame_stacking:
        test_env = VecFrameStack(test_env, n_stack=3)

    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if model_type == "ppo":
        model = PPO.load(model_path, env=None)
    elif model_type == "rppo":
        model = RecurrentPPO.load(model_path, env=None)
    else:
        raise ValueError(f"Model type could not be determined from {model_type}.")

    callback = LogSuccessCallback()
    callback.on_eval_start()
    stable_baselines3.common.utils.set_random_seed(150494)
    rewards, episode_lengths = evaluate_policy(model, test_env,
                                               n_eval_episodes=len(test_config),
                                               deterministic=True,
                                               callback=callback.on_eval_step,
                                               return_episode_rewards=True,
                                               )
    success_rate = callback.on_eval_end()

    mean_reward = np.mean(rewards)
    mean_epl = np.mean(episode_lengths)
    detailed_results = {'model': model_name, 'env': env_config.stem,
                        'mean_reward': mean_reward, 'mean_epl': mean_epl, 'success_rate': success_rate,
                        'rewards': rewards, 'episode_lengths': episode_lengths}

    return mean_reward, mean_epl, success_rate, detailed_results


models = [
    ('20240427_231410_ppo_speaker_9x9_4d', 'ppo', False, False),
    ('20240429_213310_ppo_speaker_partial_9x9_4d', 'ppo', True, False),
    ('20240429_213404_ppo_fs_speaker_9x9_4d', 'ppo', False, True),
    ('20240429_213421_ppo_fs_speaker_partial_9x9_4d', 'ppo', True, True)
]

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model_name, model_type, partial, frame_stacking in models:
            model_path = (Path('~') / "checkpoints" / model_name / "best_model").expanduser()
            for env_config in Path('data').glob('*test*.json'):
                mean_reward, mean_epl, success_rate, detailed_results = eval_speaker(
                    model_path, model_type, frame_stacking, env_config,
                    int(re.search(r'(\d+)x\1', str(env_config))[1]))
                print(f"model: {model_name}, env: {env_config}")
                print(f"mean reward: {mean_reward:.4f}, mean_epl: {mean_epl:.4f}, success_rate: {success_rate:.4f}")
                print(f"----------------------------------------------------")
