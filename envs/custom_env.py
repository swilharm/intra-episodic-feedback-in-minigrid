from abc import ABC, abstractmethod
from typing import Iterable, Tuple, List, Type

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv

from speaker.speaker import Speaker
from speaker.baseline_speaker import BaselineSpeaker


class CustomMiniGridEnv(MiniGridEnv, ABC):
    """
    A custom environment derived from a MiniGridEnv.
    Adapted to allow a changing mission that is supplied by a speaker.
    """

    def __init__(
            self,
            size: int = 6,
            num_distractors: int = 2,
            holdouts: Iterable[Tuple[str, str]] = (),
            speaker: Type[Speaker] = BaselineSpeaker,
            configs: List[dict] = None,
            **kwargs
    ):
        """
        :param size: Map size
        :param num_distractors: Number of distractors
        :param holdouts: List of holdout objects to not include when randomly generating
        :param speaker: Speaker class
        :param configs: Optional configs that override the random choices.
        See child class _gen_grid() for details regarding what can be configured.
        :param kwargs: Passed on to super init
        """
        if 'max_steps' not in kwargs:
            kwargs['max_steps'] = 5 * size

        self.size = size
        self.num_distractors = num_distractors
        self.holdouts = holdouts
        self.speaker = speaker(self)
        self.configs = configs
        if configs is not None:
            self._configs_iter = iter(configs)

        self.target: WorldObj | None = None
        self.distractors: List[WorldObj] = []

        self.initial_mission = ""

        mission_space = MissionSpace(
            mission_func=lambda mission: mission,
            ordered_placeholders=[self.speaker.list_possible_statements()]
        )
        super().__init__(mission_space=mission_space,
                         grid_size=size,
                         see_through_walls=True,
                         **kwargs
                         )

    def initial_mission_func(self, *args) -> str:
        """This function matches the original Minigrid mission."""
        return ""

    def initial_mission_placeholders(self) -> List[List[str]]:
        """Returns a list of possible strings per placeholder."""
        return []

    @property
    def OBJ_TYPES(self):
        """Object types that can appear in this environment"""
        return []

    @property
    def OBJ_COLORS(self):
        """Object colors that can appear in this environment"""
        return []

    @abstractmethod
    def _gen_grid(self, width, height):
        raise NotImplementedError

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.mission = self.speaker.gen_feedback(self.size/4, self.size/3)
        obs['mission'] = self.mission
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.speaker.reset()
        self.mission = self.speaker.gen_feedback(self.size/4, self.size/3)
        obs['mission'] = self.mission
        if self.render_mode == "human":
            self.render()
        return obs, info
