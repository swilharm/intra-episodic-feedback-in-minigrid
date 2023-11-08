from typing import List

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.world_object import Key, Ball, Wall

from envs.custom_env import CustomMiniGridEnv


class CustomFetchEnv(CustomMiniGridEnv):

    def initial_mission_func(self, syntax, color, obj_type) -> str:
        return f"{syntax} {color} {obj_type}"

    def initial_mission_placeholders(self) -> List[List[str]]:
        return [self.MISSION_SYNTAX, COLOR_NAMES, self.OBJ_TYPES]

    @property
    def MISSION_SYNTAX(self):
        return ["get a", "go get a", "fetch a", "go fetch a", "you must fetch a"]

    @property
    def OBJ_COLORS(self):
        return COLOR_NAMES

    @property
    def OBJ_TYPES(self):
        return ["key", "ball"]

    def _gen_grid(self, width: int, height: int) -> None:
        """
        Populates the grids with a target, distractors and the agent.
        The number of distractors is defined in the init().
        Types, colors and positions are decided randomly unless specified in the config.
        Config can also partially specify (e.g. define color and type but not position) and rest is filled randomly.
        Objects are never placed touching each other, directly or diagonally to ensure a solvable grid.
        :param width: Width of the grid
        :param height: Height of the grid
        """
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        if self.configs is None:
            config = {}
        else:
            try:
                config = next(self._configs_iter)
            except StopIteration:
                self._configs_iter = iter(self.configs)
                config = next(self._configs_iter)

        # Place target
        if 'target' in config:
            if config['target']['type'] == 'key':
                obj = Key(config['target']['color'])
            elif config['target']['type'] == 'ball':
                obj = Ball(config['target']['color'])
            else:
                raise ValueError(f"Invalid type {config['target']['type']}")
            self.target = obj
            if 'pos' in config['target']:
                self.put_obj(obj, config['target']['pos'][0], config['target']['pos'][1])
            else:
                self.place_obj(obj)
        else:
            target_type = self._rand_elem(self.OBJ_TYPES)
            target_color = self._rand_elem(self.OBJ_COLORS)

            while (target_type, target_color) in self.holdouts:
                target_type = self._rand_elem(self.OBJ_TYPES)
                target_color = self._rand_elem(self.OBJ_COLORS)

            if target_type == "key":
                obj = Key(target_color)
            elif target_type == "ball":
                obj = Ball(target_color)
            else:
                raise ValueError(
                    f"{target_type} object type given. Object type can only be of values {self.OBJ_TYPES}.")
            self.target = obj
            self.place_obj(obj)

        # Place distractors
        self.distractors = []
        if 'distractors' in config:
            for distractor in config['distractors']:
                if distractor['type'] == 'key':
                    obj = Key(distractor['color'])
                elif distractor['type'] == 'ball':
                    obj = Ball(distractor['color'])
                else:
                    raise ValueError(f"Invalid type {distractor['type']}")
                self.distractors.append(obj)
                if 'pos' in distractor:
                    self.put_obj(obj, distractor['pos'][0], distractor['pos'][1])
                else:
                    self.place_obj(obj, reject_fn=self.reject_fn)
        else:
            while len(self.distractors) < self.num_distractors:
                distractor_type = self._rand_elem(self.OBJ_TYPES)
                distractor_color = self._rand_elem(self.OBJ_COLORS)

                if (distractor_type, distractor_color) in self.holdouts:
                    continue

                if distractor_type == "key":
                    obj = Key(distractor_color)
                elif distractor_type == "ball":
                    obj = Ball(distractor_color)
                else:
                    raise ValueError(
                        f"{distractor_type} object type given. Object type can only be of values key and ball.")

                self.distractors.append(obj)
                self.place_obj(obj, reject_fn=self.reject_fn)

        # Place agent
        if 'agent' in config:
            self.agent_pos = tuple(config['agent']['pos'])
            self.agent_dir = config['agent']['dir']
        else:
            # Randomize the player start position and orientation
            self.place_agent()

        # Generate the mission string
        instructions = self.initial_mission_placeholders()[0]
        self.initial_mission = f"{self._rand_elem(instructions)} {self.target.color} {self.target.type}"
        self.mission = ""

    def reject_fn(self, _, pos):
        """This function checks if new object touches previous object and rejects if true."""
        for x in range(pos[0] - 1, pos[0] + 2):
            for y in range(pos[1] - 1, pos[1] + 2):
                if self.grid.get(x, y) is not None \
                        and type(self.grid.get(x, y)) != Wall:
                    return True
        return False

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.carrying:
            if self.carrying == self.target:
                reward = self._reward()
                terminated = True
            else:
                reward = 0
                terminated = True

        return obs, reward, terminated, truncated, info
