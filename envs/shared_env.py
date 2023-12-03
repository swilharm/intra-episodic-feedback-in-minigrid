from abc import abstractmethod
from typing import Iterable, Tuple, Type, List, SupportsFloat, Any

from gymnasium.core import ActType, ObsType
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj, Key, Ball, Wall
from minigrid.minigrid_env import MiniGridEnv

from follower.follower import Follower
from speaker.speaker import Speaker


class SharedEnv(MiniGridEnv):

    def __init__(
            self,
            size: int = 6,
            num_distractors: int = 2,
            configs: List[dict] | None = None,
            holdouts: Iterable[Tuple[str, str]] = (),
            speaker: Type[Speaker] | None = None,
            follower: Type[Follower] | None = None,
            **kwargs
    ):
        """
                :param size: Map size
                :param num_distractors: Number of distractors
                :param holdouts: List of holdout objects to not include when randomly generating
                :param configs: Optional configs that override the random choices.
                :param speaker: Speaker class
                :param follower: Follower class
                See child class _gen_grid() for details regarding what can be configured.
                :param kwargs: Passed on to super init
                """

        self.size = size
        self.num_distractors = num_distractors
        self.holdouts = holdouts

        self.configs = configs
        if configs is not None:
            self._configs_iter = iter(configs)

        self.speaker = speaker(self) if speaker else None
        self.follower = follower(self) if follower else None

        self.target: WorldObj | None = None
        self.distractors: List[WorldObj] = []

        self.initial_mission = ""
        self.obs = None

        mission_space = MissionSpace(
            mission_func=lambda mission: mission,
            ordered_placeholders=[self.speaker.list_possible_statements()]
        )
        super().__init__(mission_space=mission_space,
                         grid_size=size,
                         max_steps=5 * size,
                         see_through_walls=True,
                         **kwargs
                         )

    def initial_mission_func(self, color, obj_type) -> str:
        return f"get the {color} {obj_type}"

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
        self.initial_mission = self.initial_mission_func(self.target.color, self.target.type)
        self.mission = ""

    def reject_fn(self, _, pos):
        """This function checks if new object touches previous object and rejects if true."""
        for x in range(pos[0] - 1, pos[0] + 2):
            for y in range(pos[1] - 1, pos[1] + 2):
                if self.grid.get(x, y) is not None \
                        and type(self.grid.get(x, y)) != Wall:
                    return True
        return False

    def make_follower_act(self) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.follower is not None, "Can not query follower if self.follower is None"
        action = self.follower.predict()
        return self._step(action)

    def make_speaker_act(self):
        assert self.speaker is not None, "Can not query speaker if self.speaker is None"
        statement = self.speaker.predict()
        self.mission = statement

    def act_as_follower(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return self._step(action)

    def act_as_speaker(self, statement):
        self.mission = statement

    def _step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.obs, reward, terminated, truncated, info = super().step(action)

        if self.carrying:
            if self.carrying == self.target:
                reward = self._reward()
                terminated = True
            else:
                reward = 0
                terminated = True

        return self.obs, reward, terminated, truncated, info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        raise UserWarning("Should not be accessing shared environment directly")

    def reset(self, *, seed=None, options=None):
        self.obs, info = super().reset(seed=seed, options=options)
        self.follower.reset() if self.follower else None
        self.speaker.reset() if self.speaker else None

        if self.render_mode == "human":
            self.render()
        return self.obs, info
