from follower.follower import Follower
from util.wrappers import apply_wrappers


class RLFollower(Follower):

    def __init__(self, env, model):
        super().__init__(env)
        self.env = apply_wrappers(env)
        self.model = model

    def predict(self) -> int:
        obs = self.env.observation(self.env.obs)
        return self.model.predict(obs)

    def reset(self):
        pass
