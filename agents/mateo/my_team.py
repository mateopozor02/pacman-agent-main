# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# A* team: each turn an agent picks a goal (a set of cells) with simple rules,
# then plans a path to it with A* over the maze grid. Enemy ghosts are not
# simulated; they add a danger cost to the cells around them, so the planner
# routes around them instead of through them.

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='AStarOffensiveAgent', second='AStarDefensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

MOVES = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

# Draw the planned path in the GUI (only works when graphics are on)
DEBUG = False


class AStarAgent(CaptureAgent):
    """
    Base class: maze bookkeeping plus the A* planner shared by both agents.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.walls = None
        self.home_border = []

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)  # precomputes maze distances
        self.start = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()

        # Last column on our side; reaching any of these cells banks carried food
        width, height = self.walls.width, self.walls.height
        border_x = width // 2 - 1 if self.red else width // 2
        self.home_border = [(border_x, y) for y in range(height) if not self.walls[border_x][y]]

    ###########
    # Helpers #
    ###########

    def is_home(self, pos):
        """True if the cell is on our half of the board."""
        mid = self.walls.width // 2
        return pos[0] < mid if self.red else pos[0] >= mid

    def neighbors(self, pos):
        """(action, next_pos) pairs for every non-wall move from pos."""
        x, y = pos
        result = []
        for action in MOVES:
            dx, dy = Actions.direction_to_vector(action)
            nxt = (int(x + dx), int(y + dy))
            if not self.walls[nxt[0]][nxt[1]]:
                result.append((action, nxt))
        return result

    def dangerous_ghosts(self, game_state):
        """Positions of visible enemy ghosts that can still eat us."""
        my_pos = game_state.get_agent_position(self.index)
        ghosts = []
        for i in self.get_opponents(game_state):
            enemy = game_state.get_agent_state(i)
            pos = enemy.get_position()
            if pos is None or enemy.is_pacman:
                continue
            # A scared ghost is harmless only if it stays scared until we could reach it
            if enemy.scared_timer > self.get_maze_distance(my_pos, pos) + 1:
                continue
            ghosts.append(pos)
        return ghosts

    def ghost_cost(self, ghosts):
        """Step-cost function that makes cells near enemy ghosts expensive."""
        def cost(pos):
            if self.is_home(pos):  # ghosts cannot hurt us on our own side
                return 0
            penalty = 0
            for g in ghosts:
                d = self.get_maze_distance(pos, g)
                if d <= 1:
                    penalty += 1000  # practically a wall
                elif d == 2:
                    penalty += 50
                elif d == 3:
                    penalty += 10
            return penalty
        return cost

    ############
    # Planning #
    ############

    def a_star(self, start, goals, cost_fn=None):
        """
        A* from start to the closest (cheapest) cell in goals.
        g = steps taken + cost_fn(cell) for every cell entered.
        h = maze distance to the nearest goal, admissible since cost_fn >= 0.
        Returns the list of actions, or None if no goal is reachable.
        """
        goals = set(goals)
        if not goals:
            return None
        if start in goals:
            return []

        def heuristic(pos):
            return min(self.get_maze_distance(pos, g) for g in goals)

        frontier = util.PriorityQueue()
        frontier.push(start, heuristic(start))
        best_g = {start: 0}
        parent = {start: None}  # cell -> (previous cell, action taken)
        closed = set()

        while not frontier.is_empty():
            pos = frontier.pop()
            if pos in closed:
                continue
            if pos in goals:
                return self.rebuild_path(parent, pos)
            closed.add(pos)

            for action, nxt in self.neighbors(pos):
                step = 1 + (cost_fn(nxt) if cost_fn else 0)
                g = best_g[pos] + step
                if g < best_g.get(nxt, float('inf')):
                    best_g[nxt] = g
                    parent[nxt] = (pos, action)
                    frontier.push(nxt, g + heuristic(nxt))
        return None

    @staticmethod
    def rebuild_path(parent, pos):
        path = []
        while parent[pos] is not None:
            pos, action = parent[pos]
            path.append(action)
        path.reverse()
        return path

    def first_step(self, game_state, path):
        """First action of a path, STOP if the path is empty or missing."""
        if DEBUG and path:
            cells, pos = [], game_state.get_agent_position(self.index)
            for action in path:
                pos = Actions.get_successor(pos, action)
                cells.append((int(pos[0]), int(pos[1])))
            self.debug_draw(cells, [0, 0.6, 1] if self.index % 2 == 0 else [1, 0.5, 0], clear=True)
        if path:
            return path[0]
        return Directions.STOP

    def safest_action(self, game_state, ghosts):
        """Fallback when nothing is reachable: move as far as possible from the ghosts."""
        actions = [a for a in game_state.get_legal_actions(self.index) if a != Directions.STOP]
        if not actions:
            return Directions.STOP
        if not ghosts:
            return random.choice(actions)
        my_pos = game_state.get_agent_position(self.index)

        def margin(action):
            nxt = Actions.get_successor(my_pos, action)
            nxt = (int(nxt[0]), int(nxt[1]))
            return min(self.get_maze_distance(nxt, g) for g in ghosts)
        return max(actions, key=margin)


class AStarOffensiveAgent(AStarAgent):
    """
    Collects food on the enemy side and banks it before it gets risky.
    Goal priority:
      1. go home if time is running out, the bag is full, or a ghost is close
         (via a capsule if that is closer than home)
      2. otherwise the nearest food, avoiding ghosts
    """

    CARRY_LIMIT = 6      # food carried before heading home
    GHOST_ALERT = 5      # maze distance at which a ghost counts as a threat
    TIME_MARGIN = 10     # spare moves kept when racing the clock home

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        carrying = my_state.num_carrying
        food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)

        ghosts = self.dangerous_ghosts(game_state)
        cost = self.ghost_cost(ghosts)
        ghost_dist = min([self.get_maze_distance(my_pos, g) for g in ghosts], default=None)
        threatened = ghost_dist is not None and ghost_dist <= self.GHOST_ALERT

        home_dist = min(self.get_maze_distance(my_pos, b) for b in self.home_border)
        my_moves_left = game_state.data.timeleft // 4

        go_home = carrying > 0 and (
            carrying >= self.CARRY_LIMIT
            or home_dist + self.TIME_MARGIN >= my_moves_left
            or len(food) <= 2
            or threatened)

        if threatened and capsules:
            capsule_dist = min(self.get_maze_distance(my_pos, c) for c in capsules)
            if capsule_dist < home_dist or carrying == 0:
                path = self.a_star(my_pos, capsules, cost)
                if path is not None:
                    return self.first_step(game_state, path)

        if go_home:
            path = self.a_star(my_pos, self.home_border, cost)
        else:
            path = self.a_star(my_pos, food, cost)
            if path is None:  # food unreachable, at least stay safe
                path = self.a_star(my_pos, self.home_border, cost)

        if path is None:
            return self.safest_action(game_state, ghosts)
        return self.first_step(game_state, path)


class AStarDefensiveAgent(AStarAgent):
    """
    Guards our food without leaving our side.
    Goal priority:
      1. chase a visible invader (keep a distance of 2 while scared)
      2. go to where our food was last eaten
      3. patrol the border cell closest to the food we defend
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.last_eaten = None

    def stay_home(self, pos):
        """Step cost that keeps the defender on our half."""
        return 0 if self.is_home(pos) else 1000

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        self.update_last_eaten(game_state)
        if self.last_eaten == my_pos:
            self.last_eaten = None

        invaders = [game_state.get_agent_state(i).get_position() for i in self.get_opponents(game_state)
                    if game_state.get_agent_state(i).is_pacman
                    and game_state.get_agent_state(i).get_position() is not None]

        if invaders:
            if my_state.scared_timer > 0:
                # Shadow the invader from 2 cells away so we can pounce when the timer ends
                goals = [cell for inv in invaders for cell in self.cells_at_distance(inv, 2)]
            else:
                goals = invaders
        elif self.last_eaten is not None:
            goals = [self.last_eaten]
        else:
            goals = [self.patrol_point(game_state)]

        path = self.a_star(my_pos, goals, self.stay_home)
        if path is None:
            path = self.a_star(my_pos, [self.patrol_point(game_state)], self.stay_home)
        if path is None:
            return self.safest_action(game_state, [])
        return self.first_step(game_state, path)

    def update_last_eaten(self, game_state):
        """Remember the cell where our food most recently disappeared."""
        previous = self.get_previous_observation()
        if previous is None:
            return
        before = set(self.get_food_you_are_defending(previous).as_list())
        now = set(self.get_food_you_are_defending(game_state).as_list())
        eaten = before - now
        if eaten:
            my_pos = game_state.get_agent_position(self.index)
            self.last_eaten = min(eaten, key=lambda p: self.get_maze_distance(my_pos, p))

    def patrol_point(self, game_state):
        """Border cell closest (on average) to the food we defend."""
        food = self.get_food_you_are_defending(game_state).as_list()
        if not food:
            return self.home_border[len(self.home_border) // 2]
        return min(self.home_border,
                   key=lambda b: sum(self.get_maze_distance(b, f) for f in food))

    def cells_at_distance(self, pos, dist):
        x, y = int(pos[0]), int(pos[1])
        cells = []
        for cx in range(max(0, x - dist), min(self.walls.width, x + dist + 1)):
            for cy in range(max(0, y - dist), min(self.walls.height, y + dist + 1)):
                cell = (cx, cy)
                if not self.walls[cx][cy] and self.is_home(cell) \
                        and self.get_maze_distance(cell, (x, y)) == dist:
                    cells.append(cell)
        return cells or [(x, y)]
