from typing import Dict

import numpy as np
from gymnasium import ObservationWrapper, Env
from gymnasium import spaces
from gymnasium.core import ObsType
from minigrid.wrappers import RGBImgPartialObsWrapper, RGBImgObsWrapper

# Word to index mapping
W2I = dict()
# Index to word mapping
I2W = dict()


def apply_wrappers(env) -> Env:
    """Applies all wrappers in order and returns resulting environment"""
    global W2I, I2W
    env = VisionWrapper(env)
    env = LanguageWrapper(env)
    W2I = env.w2i
    I2W = {value: key for key, value in W2I.items()}
    env = DirectionWrapper(env)
    return env


class VisionWrapper(ObservationWrapper):
    """Turns image into pixel space. Returns partial view of agent and full view with and without highlights."""

    def __init__(self, env: Env):
        super().__init__(env)
        self.partial = RGBImgPartialObsWrapper(env)
        self.full_with_highlight = RGBImgObsWrapper(env)
        self.full_without_highlight = RGBImgObsWrapperNoHighlight(env)
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces,
             "image": self.partial.observation_space.spaces['image'],
             "overview": self.full_without_highlight.observation_space.spaces['image'],
             "overview_highlighted": self.full_with_highlight.observation_space.spaces['image']}
        )

    def observation(self, obs: Dict[str, ObsType]) -> Dict[str, ObsType]:
        partial = self.partial.observation(obs)['image']
        full_with_highlight = self.full_with_highlight.observation(obs)['image']
        full_without_highlight = self.full_without_highlight.observation(obs)['image']
        return {**obs,
                "image": partial,
                "overview": full_without_highlight,
                "overview_highlighted": full_with_highlight}


class RGBImgObsWrapperNoHighlight(RGBImgObsWrapper):

    def observation(self, obs: Dict[str, ObsType]) -> Dict[str, ObsType]:
        rgb_img = self.get_frame(highlight=False, tile_size=self.tile_size)
        return {**obs, "image": rgb_img}


class LanguageWrapper(ObservationWrapper):
    """Applies index function to mission and turns it into a Box observation space"""

    def __init__(self, env: Env):
        super().__init__(env)
        possible_feedback = env.unwrapped.observation_space.spaces['mission'].ordered_placeholders[0]
        self.vocab = sorted({token for feedback in possible_feedback for token in feedback.split()})
        self.max_len = len(max([feedback.split() for feedback in possible_feedback], key=lambda x: len(x)))

        new_space = spaces.Box(low=0, high=len(self.vocab) + 1, shape=(self.max_len,), dtype=np.int8)
        env.observation_space = spaces.Dict({**env.observation_space.spaces, "mission": new_space})

        self.w2i = {"<PAD>": 0}
        for token in self.vocab:
            self.w2i[token] = len(self.w2i)

    def observation(self, obs: Dict[str, ObsType]) -> Dict[str, ObsType]:
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
        new_space = spaces.Box(low=0, high=3, shape=(1,), dtype=np.int8)
        env.observation_space = spaces.Dict({**env.observation_space.spaces, "direction": new_space})

    def observation(self, obs: Dict[str, ObsType]) -> Dict[str, ObsType]:
        return {**obs, "direction": [obs['direction']]}
