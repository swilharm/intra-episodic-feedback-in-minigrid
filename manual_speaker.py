import json

import pygame
from gymnasium import Env
from minigrid.core.actions import Actions

from envs.speaker_env import SpeakerEnv
from follower.heuristic_follower import HeuristicFollower


class ManualControl:
    def __init__(
            self,
            env: Env,
            seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        actions = [(i, b.name) for i, b in enumerate(self.env.actions)]
        for action in actions:
            print(action)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.closed = True
                    self.env.close()
                    break
                action_id = int(input())
                action = self.env.actions(action_id)
                self.step(action)

    def step(self, action: Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()
        print(self.env.target.color, self.env.target.type, self.env.target.cur_pos)

if __name__ == '__main__':

    with open('data/fetch_6x6_2d_test.json', 'r') as file:
        configs = json.load(file)

    env: SpeakerEnv = SpeakerEnv(
        follower=HeuristicFollower,
        configs=configs,
        size=6,
        render_mode="human",
        tile_size=64,
        # highlight=False,
        # agent_pov=True,
        agent_view_size=3,
    )

    ManualControl(env).start()
