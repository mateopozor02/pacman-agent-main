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
# routes around them instead of through them. Enemies we cannot see are
# tracked with a Bayesian filter over the noisy distance readings.

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
    agents = [eval(first)(first_index), eval(second)(second_index)]
    beliefs = EnemyBeliefs()  # one tracker per team: both agents feed and read it
    for agent in agents:
        agent.beliefs = beliefs
    return agents


##########
# Agents #
##########

MOVES = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

# Draw the planned path in the GUI (only works when graphics are on)
DEBUG = False

SIGHT_RANGE = 5      # Manhattan distance within which the engine shows enemy positions
HIDDEN_GUESSES = 5   # most likely cells of a hidden enemy used for planning
HIDDEN_WEIGHT = 0.5  # how much a hidden-ghost guess counts compared to a visible ghost
MOTION_BIAS = 2.0    # a hidden enemy is this many times likelier to step toward its goal than elsewhere
HOMEWARD_CARRY = 4   # an enemy Pacman carrying this many dots is assumed to be heading home


class EnemyBeliefs:
    """
    Probability distribution over cells for every enemy, shared by both agents.

    Updated at each of our turns, by whichever agent is moving:
      - visible enemy: all probability on its cell
      - just vanished from close range: we ate it, so it is back at its start
      - the enemy that moved since the last update: put it on the food or
        capsule that just disappeared from our side, or on the enemy's border
        column if its num_returned just went up (it crossed home to bank food);
        otherwise spread its probability to neighbouring cells (it moved at most
        one step), favouring the step toward its likely goal: our food if it is a
        Pacman, its home if it is a Pacman carrying a lot, the border if it is a ghost
      - every hidden enemy: keep only cells consistent with the noisy distance
        reading (noise is uniform, +-6), and remove cells that are impossible
        (in our sight range, or on the wrong half given whether it is a Pacman)

    Hidden enemies keep their position secret, but the engine still exposes
    is_pacman, scared_timer, num_carrying and num_returned; this class uses them.
    """

    def __init__(self):
        self.dist = {}              # enemy index -> {cell: probability}
        self.visible_at = {}        # enemy index -> cell, if it was visible at the previous update
        self.returned = {}          # enemy index -> num_returned at the previous update
        self.food = set()           # food we defended at the previous update
        self.capsules = set()       # capsules we defended at the previous update
        self.their_border = []      # first column on the enemy's side
        self.to_their_border = {}   # cell -> maze distance to the enemy's border column

    def initialize(self, agent, game_state):
        if self.dist:  # the teammate already did it
            return
        for i in agent.get_opponents(game_state):
            self.dist[i] = {game_state.get_initial_agent_position(i): 1.0}
            self.returned[i] = 0
        self.food = set(agent.get_food_you_are_defending(game_state).as_list())
        self.capsules = set(agent.get_capsules_you_are_defending(game_state))

        their_x = agent.home_border[0][0] + (1 if agent.red else -1)
        self.their_border = [(their_x, y) for y in range(agent.walls.height) if not agent.walls[their_x][y]]
        self.to_their_border = self.distances_from(agent, self.their_border)

    def update(self, agent, game_state):
        my_pos = game_state.get_agent_position(agent.index)
        team_positions = [game_state.get_agent_position(i) for i in agent.get_team(game_state)]
        noisy = game_state.get_agent_distances()

        # Turns go 0, 1, 2, 3, so the only enemy that moved since the last update is the previous index
        mover = (agent.index - 1) % game_state.get_num_agents()

        food_now = set(agent.get_food_you_are_defending(game_state).as_list())
        capsules_now = set(agent.get_capsules_you_are_defending(game_state))
        eaten = (self.food - food_now) | (self.capsules - capsules_now)
        self.food, self.capsules = food_now, capsules_now
        to_our_food = None  # computed only when a hidden invader needs it

        for i in agent.get_opponents(game_state):
            enemy = game_state.get_agent_state(i)
            banked = enemy.num_returned > self.returned[i]
            self.returned[i] = enemy.num_returned

            pos = game_state.get_agent_position(i)
            if pos is not None:
                self.dist[i] = {pos: 1.0}
                self.visible_at[i] = pos
                continue

            last = self.visible_at.pop(i, None)
            if last is not None and any(util.manhattan_distance(last, p) < SIGHT_RANGE for p in team_positions):
                # One step from there would still be in sight, so it did not walk away: it was eaten
                self.dist[i] = {game_state.get_initial_agent_position(i): 1.0}
                continue

            if i == mover and eaten:
                self.dist[i] = {eaten.pop(): 1.0}  # it is standing where our food or capsule just vanished
                continue

            is_pacman = enemy.is_pacman
            if banked:
                # Food is banked on the step that crosses home, so it is on its border column now
                belief = {cell: 1.0 for cell in self.their_border}
            elif i == mover:
                if is_pacman and enemy.num_carrying >= HOMEWARD_CARRY:
                    toward = self.to_their_border
                elif is_pacman:
                    if to_our_food is None:
                        to_our_food = self.distances_from(agent, food_now)
                    toward = to_our_food
                else:
                    toward = self.to_their_border
                belief = self.elapse(agent, self.dist[i], toward)
            else:
                belief = self.dist[i]
            belief = self.observe(agent, game_state, belief, my_pos, noisy[i], team_positions, is_pacman)
            if not belief:
                # Our belief ruled out every cell (rare): restart from "anywhere consistent"
                everywhere = {cell: 1.0 for cell in agent.walls.as_list(False)}
                belief = self.observe(agent, game_state, everywhere, my_pos, noisy[i], team_positions, is_pacman)
            self.dist[i] = belief or self.dist[i]

    @staticmethod
    def elapse(agent, belief, toward):
        """
        The enemy took one step: each cell shares its probability with itself and
        its neighbours. Steps that bring it closer to its goal (lower value in the
        distance map `toward`) get MOTION_BIAS times the share of the others.
        """
        far = float('inf')
        new = {}
        for cell, p in belief.items():
            options = [cell] + [nxt for _, nxt in agent.neighbors(cell)]
            here = toward.get(cell, far)
            weights = [MOTION_BIAS if toward.get(nxt, far) < here else 1.0 for nxt in options]
            total = sum(weights)
            for nxt, w in zip(options, weights):
                new[nxt] = new.get(nxt, 0) + p * w / total
        return new

    @staticmethod
    def distances_from(agent, sources):
        """Maze distance from every cell to the nearest source cell (multi-source BFS)."""
        dist = {cell: 0 for cell in sources}
        frontier = list(sources)
        while frontier:
            next_frontier = []
            for cell in frontier:
                for _, nxt in agent.neighbors(cell):
                    if nxt not in dist:
                        dist[nxt] = dist[cell] + 1
                        next_frontier.append(nxt)
            frontier = next_frontier
        return dist

    @staticmethod
    def observe(agent, game_state, belief, my_pos, noisy_distance, team_positions, is_pacman):
        """Bayes update with the noisy reading, then drop impossible cells and normalize."""
        new = {}
        for cell, p in belief.items():
            if any(util.manhattan_distance(cell, t) <= SIGHT_RANGE for t in team_positions):
                continue  # it would be visible there
            if agent.is_home(cell) != is_pacman:
                continue  # an enemy Pacman is on our half, an enemy ghost on theirs
            likelihood = game_state.get_distance_prob(
                util.manhattan_distance(cell, my_pos), noisy_distance)
            if likelihood > 0:
                new[cell] = p * likelihood
        total = sum(new.values())
        return {cell: p / total for cell, p in new.items()} if total > 0 else {}

    def guesses(self, i):
        """The HIDDEN_GUESSES most likely cells of enemy i, as (cell, probability)."""
        return sorted(self.dist[i].items(), key=lambda item: -item[1])[:HIDDEN_GUESSES]

    def most_likely(self, i):
        return max(self.dist[i], key=self.dist[i].get)


class AStarAgent(CaptureAgent):
    """
    Base class: maze bookkeeping plus the A* planner shared by both agents.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.walls = None
        self.home_border = []
        self.beliefs = EnemyBeliefs()  # replaced by the team-shared one in create_team

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)  # precomputes maze distances
        self.start = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()

        # Last column on our side; reaching any of these cells banks carried food
        width, height = self.walls.width, self.walls.height
        border_x = width // 2 - 1 if self.red else width // 2
        self.home_border = [(border_x, y) for y in range(height) if not self.walls[border_x][y]]

        self.beliefs.initialize(self, game_state)

    def get_action(self, game_state):
        # Runs before choose_action every turn, so beliefs are always fresh
        self.observation_history.append(game_state)
        self.beliefs.update(self, game_state)
        if DEBUG:
            self.display_distributions_over_positions(
                [util.Counter(self.beliefs.dist[i]) if i in self.beliefs.dist else None
                 for i in range(game_state.get_num_agents())])
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos != nearest_point(my_pos):  # halfway between cells, as in CaptureAgent
            return game_state.get_legal_actions(self.index)[0]
        return self.choose_action(game_state)

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

    def enemy_locations(self, game_state, i):
        """
        Where enemy i might be, as (cell, weight): its exact cell with weight 1
        if visible, otherwise its most likely cells weighted by probability.
        """
        pos = game_state.get_agent_position(i)
        if pos is not None:
            return [(pos, 1.0)]
        return [(cell, HIDDEN_WEIGHT * p) for cell, p in self.beliefs.guesses(i)]

    def dangerous_ghosts(self, game_state):
        """(position, weight) of enemy ghosts, seen or guessed, that can still eat us."""
        my_pos = game_state.get_agent_position(self.index)
        ghosts = []
        for i in self.get_opponents(game_state):
            enemy = game_state.get_agent_state(i)  # is_pacman and scared_timer stay readable when hidden
            if enemy.is_pacman:
                continue
            for pos, weight in self.enemy_locations(game_state, i):
                # A scared ghost is harmless only if it stays scared until we could reach it
                if enemy.scared_timer > self.get_maze_distance(my_pos, pos) + 1:
                    continue
                ghosts.append((pos, weight))
        return ghosts

    def ghost_cost(self, ghosts):
        """Step-cost function that makes cells near enemy ghosts expensive."""
        cache = {}

        def cost(pos):
            if self.is_home(pos):  # ghosts cannot hurt us on our own side
                return 0
            if pos not in cache:
                penalty = 0
                for g, weight in ghosts:
                    d = self.get_maze_distance(pos, g)
                    if d <= 1:
                        penalty += 1000 * weight  # practically a wall for a visible ghost
                    elif d == 2:
                        penalty += 50 * weight
                    elif d == 3:
                        penalty += 10 * weight
                cache[pos] = penalty
            return cache[pos]
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
        # Only ghosts we see right now trigger retreat; guessed ones just shape the path
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
            return self.safest_action(game_state, visible)
        return self.first_step(game_state, path)


class AStarDefensiveAgent(AStarAgent):
    """
    Guards our food without leaving our side.
    Goal priority:
      1. chase invaders: their exact cell if visible, else their most likely
         cell from the beliefs (keep a distance of 2 while scared). With two
         invaders, go for the one carrying more food
      2. patrol the border cell closest to the food we defend
    """

    def stay_home(self, pos):
        """Step cost that keeps the defender on our half."""
        return 0 if self.is_home(pos) else 1000

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # is_pacman and num_carrying stay readable for hidden enemies, so we know about invaders we cannot see
        invaders = []  # (cell, food carried)
        for i in self.get_opponents(game_state):
            enemy = game_state.get_agent_state(i)
            if enemy.is_pacman:
                pos = game_state.get_agent_position(i)
                invaders.append((pos if pos is not None else self.beliefs.most_likely(i), enemy.num_carrying))

        if invaders:
            most = max(carrying for _, carrying in invaders)
            targets = [cell for cell, carrying in invaders if carrying == most]
            if my_state.scared_timer > 0:
                # Shadow the invader from 2 cells away so we can pounce when the timer ends
                goals = [cell for inv in targets for cell in self.cells_at_distance(inv, 2)]
            else:
                goals = targets
        else:
            goals = [self.patrol_point(game_state)]

        path = self.a_star(my_pos, goals, self.stay_home)
        if path is None:
            path = self.a_star(my_pos, [self.patrol_point(game_state)], self.stay_home)
        if path is None:
            return self.safest_action(game_state, [])
        return self.first_step(game_state, path)

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
