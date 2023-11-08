from collections import defaultdict

import numpy as np
from gymnasium import ObservationWrapper, Env, Space
from gymnasium import spaces
from gymnasium.core import ObsType
from minigrid.wrappers import RGBImgPartialObsWrapper
from stable_baselines3.common.vec_env import VecFrameStack, StackedObservations

from envs.custom_env import CustomMiniGridEnv

# Word to index mapping
W2I = dict()
# Index to word mapping
I2W = dict()


def apply_wrappers(env) -> Env:
    """Applies all three wrappers in order and returns resulting environment"""
    global W2I, I2W
    env = RGBImgPartialObsWrapper(env, tile_size=8)
    env = TextWrapper(env)
    W2I = env.w2i
    I2W = {value: key for key, value in W2I.items()}
    env = DirectionWrapper(env)
    return env


class TextWrapper(ObservationWrapper):
    """Applies index function to mission and turns it into a Box observation space"""

    def __init__(self, env: Env):
        super().__init__(env)
        possible_feedback = env.unwrapped.speaker.list_possible_statements()
        self.vocab = sorted({token for feedback in possible_feedback for token in feedback.split()})
        self.max_len = len(max([feedback.split() for feedback in possible_feedback], key=lambda x: len(x)))

        new_space = spaces.Box(low=0, high=len(self.vocab) + 1, shape=(self.max_len,), dtype=np.int8)
        env.observation_space = spaces.Dict({**env.observation_space.spaces, "mission": new_space})

        self.w2i = {"<PAD>": 0}
        for token in self.vocab:
            self.w2i[token] = len(self.w2i)

    def observation(self, obs: Dict[ObsType]) -> Dict[ObsType]:
        mission = obs['mission']
        new_mission = []
        for token in mission.split():
            new_mission.append(self.w2i[token])
        while len(new_mission) < self.max_len:
            new_mission.append(0)

        return {**obs, "mission": new_mission}


class DirectionWrapper(ObservationWrapper):
    """Turns integer representation of direction into a Box observation space"""

    def __init__(self, env: Env):
        super().__init__(env)
        direction = env.observation_space.spaces['direction']
        new_space = spaces.Box(low=0, high=3, shape=(1,), dtype=np.int8)
        env.observation_space = spaces.Dict({**env.observation_space.spaces, "direction": new_space})

    def observation(self, obs: Dict[ObsType]) -> Dict[ObsType]:
        return {**obs, "direction": [obs['direction']]}
