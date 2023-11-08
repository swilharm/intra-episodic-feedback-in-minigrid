import math
from typing import Tuple

import numpy as np


def distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Returns the euclidian distance between two (x,y) tuples"""
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def angle_between(v1: Tuple[int, int], v2: Tuple[int, int]) -> float:
    """Returns the angle between two vectors"""
    angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
    return np.degrees(angle)
