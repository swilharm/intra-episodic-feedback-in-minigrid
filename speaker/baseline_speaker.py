from typing import List

from speaker.speaker import Speaker


class BaselineSpeaker(Speaker):
    """The BaselineSpeaker always returns the initial mission statement"""

    def gen_feedback(self, steps_before_help: int, time_before_help: int) -> str:
        """
        :param steps_before_help: ignored for baseline
        :param time_before_help: ignored for baseline
        :return: initial mission statement
        """
        self.remember_pos_time()
        return self.env.initial_mission

    def list_possible_statements(self) -> List[str]:
        """Only the possible initial possions are possible statements."""
        return self.possible_initial_missions()
