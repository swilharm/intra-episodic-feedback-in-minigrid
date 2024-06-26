import json

from minigrid.manual_control import ManualControl

from envs.follower_env import FollowerEnv
from envs.shared_env import SharedEnv
from speaker.baseline_speaker import BaselineSpeaker
from speaker.heuristic_speaker import HeuristicSpeaker

with open('data/fetch_9x9_4d_test_ambig.json', 'r') as file:
    configs = json.load(file)

env: FollowerEnv = FollowerEnv(
    speaker=HeuristicSpeaker,
    configs=configs,
    size=9,
    render_mode="human",
    tile_size=64,
    # highlight=False,
    # agent_pov=True,
    agent_view_size=3,
)

ManualControl(env).start()
