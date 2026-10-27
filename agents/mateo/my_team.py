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
from contest.util import nearest_point


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

SIGHT_RANGE = 5      # Manhattan distance within which the engine shows enemy positions
MEMORY_TURNS = 6     # own turns a last-seen enemy position is trusted for


class AStarAgent(CaptureAgent):
    """
    Base class: maze bookkeeping plus the A* planner shared by both agents.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.walls = None
        self.home_border = []
        self.home_cells = []
        self.last_seen = {}  # enemy index -> (position, turn it was seen)

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)  # precomputes maze distances
        self.start = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()

        # Last column on our side; reaching any of these cells banks carried food
        width, height = self.walls.width, self.walls.height
        border_x = width // 2 - 1 if self.red else width // 2
        self.home_border = [(border_x, y) for y in range(height) if not self.walls[border_x][y]]
        self.home_cells = [(x, y) for x in range(width) for y in range(height)
                           if not self.walls[x][y] and self.is_home((x, y))]

    def get_action(self, game_state):
        # Runs before choose_action every turn, so both agents always have fresh memory
        self.observation_history.append(game_state)
        self.update_memory(game_state)
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos != nearest_point(my_pos):  # halfway between cells, as in CaptureAgent
            return game_state.get_legal_actions(self.index)[0]
        return self.choose_action(game_state)

    ##########
    # Memory #
    ##########

    def update_memory(self, game_state):
        """
        Remember where each enemy was last seen. A memory is dropped when it is
        older than MEMORY_TURNS, or when its cell is in view but the enemy is not
        there any more (it moved away, or was eaten and respawned).
        """
        turn = len(self.observation_history)
        team_positions = [game_state.get_agent_position(i) for i in self.get_team(game_state)]
        for i in self.get_opponents(game_state):
            pos = game_state.get_agent_position(i)
            if pos is not None:
                self.last_seen[i] = (pos, turn)
            elif i in self.last_seen:
                seen_pos, seen_turn = self.last_seen[i]
                too_old = turn - seen_turn > MEMORY_TURNS
                in_view = any(util.manhattan_distance(seen_pos, p) <= SIGHT_RANGE for p in team_positions)
                if too_old or in_view:
                    del self.last_seen[i]

    def tracked_enemies(self, game_state):
        """(index, position, age) for every enemy we see or remember; age 0 means visible now."""
        turn = len(self.observation_history)
        return [(i, pos, turn - seen_turn) for i, (pos, seen_turn) in self.last_seen.items()]

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
        """
        (position, weight) of enemy ghosts that can still eat us. Visible ghosts
        weigh 1; remembered ones weigh less the older the memory is, since the
        ghost has probably moved.
        """
        my_pos = game_state.get_agent_position(self.index)
        ghosts = []
        for i, pos, age in self.tracked_enemies(game_state):
            enemy = game_state.get_agent_state(i)  # is_pacman and scared_timer stay readable when hidden
            if enemy.is_pacman:
                continue
            # A scared ghost is harmless only if it stays scared until we could reach it
            if enemy.scared_timer > self.get_maze_distance(my_pos, pos) + 1:
                continue
            weight = 1.0 if age == 0 else 0.5 * (1 - age / (MEMORY_TURNS + 1))
            ghosts.append((pos, weight))
        return ghosts

    def ghost_cost(self, ghosts):
        """Step-cost function that makes cells near enemy ghosts expensive."""
        def cost(pos):
            if self.is_home(pos):  # ghosts cannot hurt us on our own side
                return 0
            penalty = 0
            for g, weight in ghosts:
                d = self.get_maze_distance(pos, g)
                if d <= 1:
                    penalty += 1000 * weight  # practically a wall for a visible ghost
                elif d == 2:
                    penalty += 50 * weight
                elif d == 3:
                    penalty += 10 * weight
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
        # Only ghosts we see right now trigger retreat; remembered ones just shape the path
        visible = [g for g, weight in ghosts if weight == 1.0]
        ghost_dist = min([self.get_maze_distance(my_pos, g) for g in visible], default=None)
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
            return self.safest_action(game_state, [g for g, _ in ghosts])
        return self.first_step(game_state, path)


class AStarDefensiveAgent(AStarAgent):
    """
    Guards our food without leaving our side.
    Goal priority:
      1. chase a visible invader (keep a distance of 2 while scared)
      2. go to where our food was last eaten
      3. go to where an invader was last seen
      4. an unseen invader is on our side: wait at the cell central to our food
      5. patrol the border cell closest to the food we defend
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

        # is_pacman stays readable for hidden enemies, so we know about invaders we cannot see
        tracked = {i: (pos, age) for i, pos, age in self.tracked_enemies(game_state)}
        invaders_seen, invaders_remembered, invader_hidden = [], [], False
        for i in self.get_opponents(game_state):
            if not game_state.get_agent_state(i).is_pacman:
                continue
            if i not in tracked:
                invader_hidden = True
            elif tracked[i][1] == 0:
                invaders_seen.append(tracked[i][0])
            else:
                invaders_remembered.append(tracked[i][0])

        if invaders_seen:
            if my_state.scared_timer > 0:
                # Shadow the invader from 2 cells away so we can pounce when the timer ends
                goals = [cell for inv in invaders_seen for cell in self.cells_at_distance(inv, 2)]
            else:
                goals = invaders_seen
        elif self.last_eaten is not None:
            goals = [self.last_eaten]
        elif invaders_remembered:
            goals = invaders_remembered
        elif invader_hidden:
            goals = [self.guard_point(game_state)]
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

    def guard_point(self, game_state):
        """Home cell with the smallest total distance to our food: quickest to reach any of it."""
        food = self.get_food_you_are_defending(game_state).as_list()
        if not food:
            return self.patrol_point(game_state)
        return min(self.home_cells, key=lambda c: sum(self.get_maze_distance(c, f) for f in food))

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
