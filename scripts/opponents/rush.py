# Two baseline attackers and no defender: tests whether a team leaves its food exposed.
from contest.baseline_team import OffensiveReflexAgent


def create_team(first_index, second_index, is_red, num_training=0):
    return [OffensiveReflexAgent(first_index), OffensiveReflexAgent(second_index)]
