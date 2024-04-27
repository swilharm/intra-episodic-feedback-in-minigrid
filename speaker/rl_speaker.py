from typing import List

from gymnasium import ObservationWrapper
from stable_baselines3 import PPO

from follower.follower import Follower
from speaker.heuristic_speaker import HeuristicSpeaker
from speaker.speaker import Speaker
from util.wrappers import apply_wrappers


class RLSpeaker(Speaker):

    def __init__(self, env, model):
        super().__init__(env)
        self.env = apply_wrappers(env)
        self.model = model
        self.idx2statement = {idx: statement for idx, statement in enumerate(self.list_possible_statements())}

    def predict(self) -> str:
        obs = self.unwrap(self.env, self.env.obs)
        prediction, _ = self.model.predict(obs)
        return self.idx2statement[prediction.item()]

    def unwrap(self, env, observation: dict) -> dict:
        if isinstance(env, ObservationWrapper):
            observation = self.unwrap(env.env, observation)
            return env.observation(observation)
        return observation


    def reset(self):
        pass

    def list_possible_statements(self) -> List[str]:
        possible_statements = self.possible_initial_missions()
        for obj_color in self.env.OBJ_COLORS:
            for obj_type in self.env.OBJ_TYPES:
                possible_statements.append(f"yes the {obj_color} {obj_type}")
                possible_statements.append(f"not the {obj_color} {obj_type}")
                possible_statements.append(f"the {obj_color} {obj_type}")
        possible_statements.append("yes this way")
        possible_statements.append("not this way")
        possible_statements.append("go left")
        possible_statements.append("go straight")
        possible_statements.append("go right")
        possible_statements.append("turn around")
        possible_statements.append("_")
        return possible_statements
