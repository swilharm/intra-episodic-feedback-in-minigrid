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
from envs.speaker_env import SpeakerEnv
from feature_extraction.feature_extractor import FollowerFeaturesExtractor, SpeakerWithPartialFeaturesExtractor, \
    SpeakerWithoutPartialFeaturesExtractor
from follower.heuristic_follower import HeuristicFollower
from speaker.baseline_speaker import BaselineSpeaker
from speaker.heuristic_speaker import HeuristicSpeaker
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
name += f"_fs" if args.frame_stacking else ""
name += f"_speaker"
name += f"_partial" if args.partial else ""
name += f"_9x9_4d"
name = name.lower()
print(name)

env_kwargs = {
    "size": 9,
    'follower': HeuristicFollower,
    "render_mode": "rgb_array",
    "tile_size": 8,
    "highlight": False,
    "agent_pov": True,
    "agent_view_size": 3,
}

train_kwargs = {**env_kwargs, "configs": train}
train_env = make_vec_env(SpeakerEnv, n_envs=10, seed=150494, wrapper_class=apply_wrappers, env_kwargs=train_kwargs)
val_kwargs = {**env_kwargs, "configs": val}
val_env = make_vec_env(SpeakerEnv, n_envs=1, seed=150494, wrapper_class=apply_wrappers, env_kwargs=val_kwargs)

if args.frame_stacking:
    train_env = VecFrameStack(train_env, n_stack=3)
    val_env = VecFrameStack(val_env, n_stack=3)

policy_kwargs = dict(
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
    policy_kwargs["features_extractor_class"] = SpeakerWithPartialFeaturesExtractor
else:
    policy_kwargs["features_extractor_class"] = SpeakerWithoutPartialFeaturesExtractor


torch.set_num_threads(1)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    torch.cuda.set_device(args.device)

if sys.platform == 'linux':
    log_path = "/cache/tensorboard-logdir/"
else:
    log_path = f"{Path.home()}/logs/"

if args.model == 'ppo':
    model = PPO("MultiInputPolicy", train_env, policy_kwargs=policy_kwargs, verbose=0,
                tensorboard_log=log_path,
                )
else:
    policy_kwargs = {
        **policy_kwargs,
        "shared_lstm": True,
        "enable_critic_lstm": False,
    }
    model = RecurrentPPO("MultiInputLstmPolicy", train_env, policy_kwargs=policy_kwargs, verbose=0,
                         tensorboard_log=log_path,
                         )

log_callback = LogSuccessCallback()
evaluation_callback = EvalCallback(eval_env=val_env, n_eval_episodes=len(val), eval_freq=500_000,
                                   log_path=f"../evals/{name}/",
                                   best_model_save_path=f"../checkpoints/{name}/",
                                   )
try:
    model.learn(args.nsteps, tb_log_name=name, callback=[log_callback, evaluation_callback], progress_bar=True)
except KeyboardInterrupt as e:
    model.save(f"../checkpoints/{name}/interrupt.zip")
    raise e
model.save(f"../final/{name}.zip")
