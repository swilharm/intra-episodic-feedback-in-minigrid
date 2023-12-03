import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, List

from minigrid.core.world_object import WorldObj

from util.util import distance, angle_between

if TYPE_CHECKING:
    from envs.shared_env import SharedEnv


class Speaker(ABC):
    """Abstract base class. Child classes have to implement gen_feedback and list_possible_statements"""

    def __init__(self, env: "SharedEnv"):
        self.env: "SharedEnv" = env
        self.steps_before_help: float = env.size / 4
        self.time_before_help: float = env.size / 3
        self.position_last_spoken: Tuple[int, int] = (-1, -1)
        self.timestep_last_spoken: int = -1

    @abstractmethod
    def predict(self) -> str:
        """
        Generates the feedback based on environment state.
        :return: Feedback as str
        """
        raise NotImplementedError

    def remember_pos_time(self):
        """Stores current time and position"""
        self.timestep_last_spoken = self.env.step_count
        self.position_last_spoken = self.env.agent_pos

    def initial_mission(self) -> str:
        """Initial mission as given by the environment"""
        self.remember_pos_time()
        return self.env.initial_mission

    def movement_feedback(self) -> str:
        """Feedback for movement.
        If the agent moved closer to the target, give affirmative feedback.
        If the agent has moved away from the target, give negative feedback.
        If agent has only turned, consider if turned direction is closer to target."""
        self.remember_pos_time()
        last_distance = distance(self.position_last_spoken, self.env.target.cur_pos)
        if self.position_last_spoken != self.env.agent_pos:
            # agent has moved
            current_distance = distance(self.env.agent_pos, self.env.target.cur_pos)
        else:
            # agent has not moved / only turned
            current_distance = distance(self.env.front_pos, self.env.target.cur_pos)
        if current_distance < last_distance:
            return f"yes this way"
        else:
            return f"not this way"

    def walking_direction_help(self) -> str:
        """Tells the agent whether to turn or to keep walking straight.
        Uses the angle to target to determine feedback."""
        self.remember_pos_time()
        agent_x, agent_y = self.env.agent_pos
        target_x, target_y = self.env.target.cur_pos
        target_vec = (target_x - agent_x, target_y - agent_y)
        facing_vec = self.env.dir_vec
        angle = angle_between(facing_vec, target_vec)
        if -135 < angle < -45:
            return "go left"
        elif -45 <= angle <= 45:
            return "go straight"
        elif 45 < angle < 135:
            return "go right"
        else:
            return "turn around"

    def object_pickup_feedback(self, obj_in_front: WorldObj) -> str:
        """Checks if object in front of the agent is the target.
        If yes, give affirmative feedback, else, give negative feedback.
        Includes a description of the object in front of the agent to facilitate learning types and colors."""
        self.remember_pos_time()
        if obj_in_front == self.env.target:
            return f"yes this {obj_in_front.color} {obj_in_front.type}"
        elif obj_in_front in self.env.distractors:
            return f"not this {obj_in_front.color} {obj_in_front.type}"
        raise ValueError(f"{obj_in_front} is neither target nor distractor."
                         f"This function was likely erroneously called when no object was in front.")

    def describe_object(self) -> str:
        """Describes color and type of target"""
        self.remember_pos_time()
        return f"the {self.env.target.color} {self.env.target.type}"

    @abstractmethod
    def list_possible_statements(self) -> List[str]:
        """A list of all statements the speaker can possibly utter.
        These make up the observation space for the mission observation."""
        raise NotImplementedError

    def possible_initial_missions(self) -> List[str]:
        """Generates all possible initial missions"""
        possible_statements = []
        for combination in itertools.product(self.env.OBJ_COLORS, self.env.OBJ_TYPES):
            possible_statements.append(self.env.initial_mission_func(*combination))
        return possible_statements

    def reset(self):
        """Resets the speaker for a new instance"""
        self.position_last_spoken: Tuple[int, int] = (-1, -1)
        self.timestep_last_spoken: int = -1
