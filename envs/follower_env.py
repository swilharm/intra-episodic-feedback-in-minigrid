from typing import Any, SupportsFloat

from gymnasium import Env
from gymnasium.core import RenderFrame, ObsType, ActType

from envs.shared_env import SharedEnv


class FollowerEnv(Env):

    def __init__(self, **kwargs):
        self.render_mode = kwargs.get("render_mode", "rgb_array")
        self.env = SharedEnv(**kwargs)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        _, reward, terminated, truncated, info = self.env.act_as_follower(action)
        self.env.make_speaker_act()
        return self.env.obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, ) \
            -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset()
        self.env.make_speaker_act()
        return self.env.obs, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.env.render()

    def close(self):
        self.env.close()

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self.env, attr)
        return super().__getattr__(attr)
