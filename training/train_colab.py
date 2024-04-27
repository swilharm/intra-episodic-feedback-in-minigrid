import json
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from envs.follower_env import FollowerEnv
from envs.shared_env import SharedEnv
from envs.speaker_env import SpeakerEnv
from feature_extraction.feature_extractor import FollowerFeaturesExtractor, SpeakerWithPartialFeaturesExtractor, \
    SpeakerWithoutPartialFeaturesExtractor
from follower.rl_follower import RLFollower
from speaker.baseline_speaker import BaselineSpeaker
from speaker.heuristic_speaker import HeuristicSpeaker
from speaker.rl_speaker import RLSpeaker
from util.callbacks import LogSuccessCallback
from util.wrappers import apply_wrappers

with open('data/fetch_9x9_4d_train.json', 'r') as file:
    train = json.load(file)

with open('data/fetch_9x9_4d_val.json', 'r') as file:
    val = json.load(file)

arg_parser = ArgumentParser()
arg_parser.add_argument('--model', '-m', required=True, choices=['ppo', 'rppo'])
arg_parser.add_argument('--frame-stacking', '-fs', action='store_true')
arg_parser.add_argument('--partial', '-p', action='store_true')
arg_parser.add_argument('--nsteps', '-n', required=False, default=5_000_000, type=int)
arg_parser.add_argument('--device', '-d', required=False, default=0, type=int)
args = arg_parser.parse_args()

name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
name += f"_{args.model}"
follower_name = name + f"_fs" if args.frame_stacking else name
speaker_name = name + f"_fs" if args.frame_stacking else name
follower_name += f"_colab_follower"
speaker_name += f"_colab_speaker"
speaker_name += f"_partial" if args.partial else ""
follower_name += f"_9x9_4d"
speaker_name += f"_9x9_4d"
follower_name = follower_name.lower()
speaker_name = speaker_name.lower()
print(follower_name)
print(speaker_name)

env_kwargs = {
    "size": 9,
    "render_mode": "rgb_array",
    "tile_size": 8,
    "highlight": False,
    "agent_pov": True,
    "agent_view_size": 3,
}

train_kwargs = {**env_kwargs, "configs": train}
follower_train_env = make_vec_env(FollowerEnv, n_envs=10, seed=150494, wrapper_class=apply_wrappers, env_kwargs=train_kwargs)
speaker_train_env = make_vec_env(SpeakerEnv, n_envs=10, seed=150494, wrapper_class=apply_wrappers, env_kwargs=train_kwargs)
val_kwargs = {**env_kwargs, "configs": val}
follower_val_env = make_vec_env(FollowerEnv, n_envs=1, seed=150494, wrapper_class=apply_wrappers, env_kwargs=val_kwargs)
speaker_val_env = make_vec_env(SpeakerEnv, n_envs=1, seed=150494, wrapper_class=apply_wrappers, env_kwargs=val_kwargs)

if args.frame_stacking:
    follower_train_env = VecFrameStack(follower_train_env, n_stack=3)
    follower_val_env = VecFrameStack(follower_val_env, n_stack=3)
    speaker_train_env = VecFrameStack(speaker_train_env, n_stack=3)
    speaker_val_env = VecFrameStack(speaker_val_env, n_stack=3)

# Make sure follower and speaker share the same underlying env
for i in range(follower_train_env.num_envs):
    speaker_train_env.unwrapped.envs[i].env = follower_train_env.unwrapped.envs[i].env
for i in range(follower_val_env.num_envs):
    speaker_val_env.unwrapped.envs[i].env = follower_val_env.unwrapped.envs[i].env

follower_policy_kwargs = dict(
    features_extractor_class=FollowerFeaturesExtractor,
    features_extractor_kwargs={
        "vision_dim": 128,
        "embedding_dim": 128,
        "language_dim": 128,
        "direction_dim": 128,
        "target_dim": 128,
        "film_dim": 128,
    },
)

speaker_policy_kwargs = dict(
    features_extractor_kwargs={
        "vision_dim": 128,
        "embedding_dim": 128,
        "language_dim": 128,
        "direction_dim": 128,
        "target_dim": 128,
        "film_dim": 128,
    },
)
if args.partial:
    speaker_policy_kwargs["features_extractor_class"] = SpeakerWithPartialFeaturesExtractor
else:
    speaker_policy_kwargs["features_extractor_class"] = SpeakerWithoutPartialFeaturesExtractor

# torch.set_num_threads(1)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    torch.cuda.set_device(args.device)

if sys.platform == 'linux':
    log_path = "/cache/tensorboard-logdir/"
else:
    log_path = f"{Path.home()}/logs/"

follower_model = PPO("MultiInputPolicy", follower_train_env, policy_kwargs=follower_policy_kwargs,
                     verbose=0, tensorboard_log=log_path, )
speaker_model = PPO("MultiInputPolicy", speaker_train_env, policy_kwargs=speaker_policy_kwargs,
                    verbose=0, tensorboard_log=log_path, )

for i in range(follower_train_env.num_envs):
    follower = RLFollower(follower_train_env.unwrapped.envs[i].unwrapped.env, follower_model)
    follower_train_env.unwrapped.envs[i].unwrapped.env.follower = follower
    speaker = RLSpeaker(speaker_train_env.unwrapped.envs[i].unwrapped.env, speaker_model)
    speaker_train_env.unwrapped.envs[i].unwrapped.env.speaker = speaker
for i in range(follower_val_env.num_envs):
    follower = RLFollower(follower_val_env.unwrapped.envs[i].unwrapped.env, follower_model)
    follower_val_env.unwrapped.envs[i].unwrapped.env.follower = follower
    speaker = RLSpeaker(speaker_val_env.unwrapped.envs[i].unwrapped.env, speaker_model)
    speaker_val_env.unwrapped.envs[i].unwrapped.env.speaker = speaker

log_callback = LogSuccessCallback()
follower_evaluation_callback = EvalCallback(eval_env=follower_val_env, n_eval_episodes=len(val), eval_freq=500_000,
                                            log_path=f"../evals/{follower_name}/",
                                            best_model_save_path=f"../checkpoints/{follower_name}/",
                                            )
speaker_evaluation_callback = EvalCallback(eval_env=speaker_val_env, n_eval_episodes=len(val), eval_freq=500_000,
                                           log_path=f"../evals/{speaker_name}/",
                                           best_model_save_path=f"../checkpoints/{speaker_name}/",
                                           )

for steps in range(0, args.nsteps, 10240):
    follower_model.learn(10240, tb_log_name=follower_name,
                         callback=[log_callback, follower_evaluation_callback], progress_bar=True)
    speaker_model.learn(10240, tb_log_name=speaker_name,
                        callback=[log_callback, speaker_evaluation_callback], progress_bar=True)

follower_model.save(f"../final/{follower_name}.zip")
speaker_model.save(f"../final/{speaker_name}.zip")
