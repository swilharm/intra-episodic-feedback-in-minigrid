import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LogSuccessCallback(BaseCallback):
    """Logs mean episode length, mean reward and success rate per rollout to tensorboard"""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_rollout_start(self) -> None:
        self.n_envs = self.training_env.num_envs
        self.__rollout_reset()

    def on_eval_start(self, n_envs=1) -> None:
        """Evals do not take a callback class but only a step function.
        We therefore have to manually call this to initialize the counters"""
        self.n_envs = n_envs
        self.__rollout_reset()

    def __rollout_reset(self):
        self.rewards = []
        self.ep_lens = []
        self.successful_episodes = 0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        self.__update_counters(self.locals)
        return True

    def on_eval_step(self, _locals, _globals):
        self.__update_counters(_locals)

    def __update_counters(self, _locals):
        for i in range(self.n_envs):
            if _locals['dones'][i]:
                self.total_episodes += 1
                if _locals['rewards'][i] > 0:
                    self.successful_episodes += 1
                self.rewards.append(_locals['infos'][i]['episode']['r'])
                self.ep_lens.append(_locals['infos'][i]['episode']['l'])

    def _on_rollout_end(self):
        success_rate = self.__calculate_success_rate()
        self.logger.record("rollout_only/ep_len_mean", np.mean(self.ep_lens))
        self.logger.record("rollout_only/ep_rew_mean", np.mean(self.rewards))
        self.logger.record("rollout_only/ep_success_rate", success_rate)
        self.logger.record("rollout/ep_success_rate", success_rate)

    def on_eval_end(self):
        """Evals do not take a callback class but only a step function.
        We therefore have to manually call this to retrieve the success rate"""
        success_rate = self.__calculate_success_rate()
        return success_rate

    def __calculate_success_rate(self):
        if self.total_episodes == 0:
            return 0
        return self.successful_episodes / self.total_episodes
