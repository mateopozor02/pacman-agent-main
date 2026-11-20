# Two baseline defenders and no attacker: tests breaking through a guarded border.
from contest.baseline_team import DefensiveReflexAgent


def create_team(first_index, second_index, is_red, num_training=0):
    return [DefensiveReflexAgent(first_index), DefensiveReflexAgent(second_index)]
