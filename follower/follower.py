from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, List

if TYPE_CHECKING:
    from envs.shared_env import SharedEnv


class Follower:

    def __init__(self, env: "SharedEnv"):
        self.env: "SharedEnv" = env
        self.plan = []
        pass

    @abstractmethod
    def predict(self) -> int:
        """
        Decides the action based on environment state.
        :return: action as int
        """
        raise NotImplementedError

    def reset(self):
        """Resets the follower for a new instance"""
        self.plan = []
