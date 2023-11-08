from typing import List

from minigrid import Wall

from speaker.speaker import Speaker
from util.util import distance


class HeuristicSpeaker(Speaker):
    """The HeuristicSpeaker follows a predefined ruleset what to say when.
    This serves as the training partner when only training a follower"""

    def gen_feedback(self, steps_before_help: int, time_before_help: int) -> str:
        if self.timestep_last_spoken == -1:
            feedback = self.initial_mission()
        elif distance(self.position_last_spoken, self.env.agent_pos) >= steps_before_help:
            # move feedback
            obj_in_front = self.env.grid.get(*self.env.front_pos)
            if obj_in_front and not isinstance(obj_in_front, Wall):
                # looking at object
                feedback = self.object_pickup_feedback(obj_in_front)
            else:
                # not looking at object, give direction feedback (yes/no)
                feedback = self.movement_feedback()
        elif self.env.step_count - self.timestep_last_spoken >= time_before_help:
            # wait feedback
            obj_in_front = self.env.grid.get(*self.env.front_pos)
            if obj_in_front and not isinstance(obj_in_front, Wall):
                # looking at object
                feedback = self.object_pickup_feedback(obj_in_front)
            elif self.env.agent_sees(*self.env.target.cur_pos):
                # target in sight
                feedback = self.describe_object()
            else:
                # target not in sight, give direction help (left, right, forward, backward)
                feedback = self.walking_direction_help()
        elif self.env.mission == self.env.initial_mission:
            feedback = self.env.initial_mission
        else:
            feedback = ""
        return feedback

    def list_possible_statements(self) -> List[str]:
        possible_statements = self.possible_initial_missions()
        for obj_color in self.env.OBJ_COLORS:
            for obj_type in self.env.OBJ_TYPES:
                possible_statements.append(f"yes this {obj_color} {obj_type}")
                possible_statements.append(f"not this {obj_color} {obj_type}")
                possible_statements.append(f"the {obj_color} {obj_type}")
        possible_statements.append("yes this way")
        possible_statements.append("not this way")
        possible_statements.append("go left")
        possible_statements.append("go straight")
        possible_statements.append("go right")
        possible_statements.append("turn around")
        return possible_statements
