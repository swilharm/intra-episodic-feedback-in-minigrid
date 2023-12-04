import re

import numpy as np
from minigrid.core.actions import Actions

from follower.follower import Follower


class HeuristicFollower(Follower):
    """The HeuristicFollower follows a predefined ruleset how to act when.
    This serves as the training partner when only training a speaker"""

    def predict(self) -> int:
        mission = self.env.mission
        if mission == "":
            pass
        elif mission == "go left":
            self.plan = [Actions.left]
        elif mission == "go right":
            self.plan = [Actions.right]
        elif mission == "go straight":
            self.plan = [Actions.forward]
        elif mission == "turn around":
            self.plan = [Actions.right, Actions.right]
        elif mission == "yes this way":
            self.plan = [Actions.forward]
        elif mission == "not this way":
            self.plan = [Actions.right]
        elif mission.startswith("yes the"):
            self.plan = [Actions.pickup]
        elif mission.startswith("not the"):
            self.plan = [Actions.right]
        else:
            match = re.match(r".*?the (\w+) (\w+)$", mission)
            if match:
                color = match[1]
                type = match[2]
                for obj in [self.env.target] + self.env.distractors:
                    if color == obj.color and type == obj.type:
                        if self.env.agent_sees(*obj.cur_pos):
                            self.plan = self.path_to(*obj.cur_pos)
            else:
                raise ValueError(match)

        # If a plan was found, follow it
        if self.plan:
            return self.plan.pop(0)

        # If no actions are planned, default to going straight
        # or right if path blocked
        obj_in_front = self.env.grid.get(*self.env.front_pos)
        if not obj_in_front:
            return Actions.forward
        else:
            return Actions.right

    def path_to(self, x, y):
        (cur_x, cur_y), cur_dir = self.env.agent_pos, self.env.agent_dir

        # If object is grabbable, grab it
        if np.array_equal(self.env.front_pos, (x, y)):
            return [Actions.pickup]

        path = []
        rel = relative_vector(cur_x, cur_y, x, y, cur_dir)
        if rel[0] < 0:
            path.extend([Actions.left, Actions.forward, Actions.right])
        elif rel[0] > 0:
            path.extend([Actions.right, Actions.forward, Actions.left])
        path.extend([Actions.forward] * (rel[1] - 1))
        path.append(Actions.pickup)

        return path


def relative_vector(x1, y1, x2, y2, dir):
    # 0 swap
    # 1 flip x
    # 2 flip both, swap
    # 3 flip y
    dx, dy = (x2 - x1, y2 - y1)
    if dir == 0:
        return dy, dx
    elif dir == 1:
        return -dx, dy
    elif dir == 2:
        return -dy, -dx
    elif dir == 3:
        return dx, -dy
