from enum import IntEnum
from typing import Any, SupportsFloat

from gymnasium import Env, spaces
from gymnasium.core import RenderFrame, ObsType, ActType

from envs.shared_env import SharedEnv
from speaker.heuristic_speaker import HeuristicSpeaker


class SpeakerEnv(Env):

    def __init__(self, **kwargs):
        self.env = SharedEnv(**kwargs)
        all_possible_statements = HeuristicSpeaker(self.env).list_possible_statements() + ["_"]
        self.actions = IntEnum('SpeakerActions',
                               {statement: i for i, statement in enumerate(all_possible_statements)})
        self.action_space = spaces.Discrete(len(self.actions))

        self.speech_count = 0

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        statement = self.actions(action).name
        if statement == "_":
            statement = ""
        else:
            self.speech_count += 1
        self.env.act_as_speaker(statement)
        _, reward, terminated, truncated, info = self.env.make_follower_act()

        if reward > 0:
            reward = self._reward()

        return self.env.obs, reward, terminated, truncated, info

    def _reward(self):
        return 1 - 0.9 * ((self.step_count + self.speech_count) / 2 / self.max_steps)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset()
        self.env.make_follower_act()
        self.speech_count = 0
        return self.env.obs, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.env.render()

    def close(self):
        self.env.close()

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self.env, attr)
        return super().__getattr__(attr)
