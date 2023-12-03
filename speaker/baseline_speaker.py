from typing import List

from speaker.speaker import Speaker


class BaselineSpeaker(Speaker):
    """The BaselineSpeaker always returns the initial mission statement"""

    def predict(self) -> str:
        """
        :return: initial mission statement
        """
        self.remember_pos_time()
        return self.env.initial_mission

    def list_possible_statements(self) -> List[str]:
        """Only the possible initial possions are possible statements."""
        return self.possible_initial_missions()
