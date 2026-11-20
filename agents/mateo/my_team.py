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
from collections import deque

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
    roles = {'AStarOffensiveAgent': AStarOffensiveAgent, 'AStarDefensiveAgent': AStarDefensiveAgent}
    agents = [roles[first](first_index), roles[second](second_index)]
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
STUCK_WINDOW = 10    # an attacker that used at most STUCK_CELLS cells in this many turns is stuck
STUCK_CELLS = 3
DETOUR_TURNS = 30    # turns a stuck attacker gets to reach its detour before giving up on it


def home_dist_due(state, distance, origin, border, margin):
    left = (state.data.timeleft + state.get_num_agents() - 1) // state.get_num_agents()
    return min(distance(origin, cell) for cell in border) + margin >= left


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
        self.attackers = set()      # enemies that have crossed to our side at least once
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
            if enemy.is_pacman:
                self.attackers.add(i)

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
    Base class: maze bookkeeping, the A* planner, and the two jobs (attack and
    defend). The subclasses decide which job to do each turn.
    """

    CARRY_LIMIT = 6      # food carried before heading home
    GHOST_ALERT = 5      # maze distance at which a ghost counts as a threat
    TIME_MARGIN = 10     # spare moves kept when racing the clock home
    SMART_RAIDS = True
    COVER_RAIDER = True
    SHORT_FEAST = True
    PRESSURE = False
    INTERCEPT = True
    SCARED_MARGIN = 4    # spare moves kept when racing the end of the enemy's scared time home

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.walls = None
        self.home_border = []
        self.home_cells = []
        self.beliefs = EnemyBeliefs()  # replaced by the team-shared one in create_team
        self.trail = deque(maxlen=STUCK_WINDOW)  # our last cells while attacking
        self.detour = None             # cell we committed to after getting stuck
        self.detour_turns = 0

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)  # precomputes maze distances
        self.start = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()

        # Last column on our side; reaching any of these cells banks carried food
        width, height = self.walls.width, self.walls.height
        border_x = width // 2 - 1 if self.red else width // 2
        self.home_border = [(border_x, y) for y in range(height) if not self.walls[border_x][y]]
        self.home_cells = [cell for cell in self.walls.as_list(False) if self.is_home(cell)]

        self.beliefs.initialize(self, game_state)
        self.crossings = [cell for cell in self.home_border
                          if any(not self.is_home(nxt) for _, nxt in self.neighbors(cell))]
        self.dead_ends = self.find_dead_ends()

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
        move = self.choose_action(game_state)
        legal = game_state.get_legal_actions(self.index)
        return move if move in legal else legal[0]

    ###########
    # Helpers #
    ###########

    def is_home(self, pos):
        """True if the cell is on our half of the board."""
        mid = self.walls.width // 2
        return pos[0] < mid if self.red else pos[0] >= mid

    def stay_home(self, pos):
        """Step cost that keeps a path on our half."""
        return 0 if self.is_home(pos) else 1000

    def enemy_cell(self, game_state, i):
        """Best single guess of where enemy i is: its cell if visible, else the most likely one."""
        pos = game_state.get_agent_position(i)
        return pos if pos is not None else self.beliefs.most_likely(i)

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

    def find_dead_ends(self):
        """Peel enemy-side leaves, then trace each pocket back to its exit."""
        remaining = {cell: {nxt for _, nxt in self.neighbors(cell)}
                     for cell in self.walls.as_list(False) if not self.is_home(cell)}
        leaves = deque(cell for cell, links in remaining.items() if len(links) == 1)
        parents = {}
        while leaves:
            cell = leaves.popleft()
            if len(remaining[cell]) != 1:
                continue
            exit_step = next(iter(remaining.pop(cell)))
            parents[cell] = exit_step
            if exit_step in remaining:
                remaining[exit_step].remove(cell)
                if len(remaining[exit_step]) == 1:
                    leaves.append(exit_step)
        pockets = {}
        for cell, step in reversed(list(parents.items())):
            exit_cell, depth = pockets.get(step, (step, 0))
            pockets[cell] = (exit_cell, depth + 1)
        return pockets

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


    #############
    # Attacking #
    #############

    def attack(self, game_state, carry_limit=None, targets=None):
        """
        Collect food on the enemy side and bank it before it gets risky.
        carry_limit and targets override CARRY_LIMIT and the food to go for (raids).
        Goal priority:
          1. finish a detour, if we got stuck (see pick_detour)
          2. go home if time is running out, the bag is full, or a ghost is close
             (via a capsule if that is closer than home)
          3. otherwise the nearest food, avoiding ghosts
        """
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

        if self.SHORT_FEAST and carrying and (threatened or home_dist_due(
                game_state, self.get_maze_distance, my_pos, self.home_border, self.TIME_MARGIN)):
            self.detour = None
            self.trail.clear()
        self.trail.append(my_pos)
        if self.detour is None and len(self.trail) == STUCK_WINDOW and len(set(self.trail)) <= STUCK_CELLS:
            # A ghost keeps mirroring us and we keep stepping back and forth: change the plan
            self.detour, self.detour_turns = self.pick_detour(game_state, my_pos, capsules), DETOUR_TURNS
        if self.detour is not None:
            self.detour_turns -= 1
            arrived = my_pos == self.detour
            eaten = self.detour not in self.home_border and self.detour not in capsules
            path = None if arrived or eaten or self.detour_turns < 0 else self.a_star(my_pos, [self.detour], cost)
            if path:
                return self.first_step(game_state, path)
            self.detour = None
            self.trail.clear()

        home_dist = min(self.get_maze_distance(my_pos, b) for b in self.home_border)
        my_moves_left = game_state.data.timeleft // 4

        # After a capsule, keep eating past CARRY_LIMIT while every enemy ghost stays
        # scared long enough for us to still get home
        scared = [game_state.get_agent_state(i).scared_timer for i in self.get_opponents(game_state)
                  if not game_state.get_agent_state(i).is_pacman]
        feasting = bool(scared) and min(scared) > home_dist + self.SCARED_MARGIN

        if self.SHORT_FEAST:
            cap = carry_limit or self.CARRY_LIMIT
            if (carry_limit is None and self.get_score(game_state) > 0
                    and my_moves_left <= 40):
                cap = 5
            carry_limit = cap
            if feasting and carrying >= cap:
                nearby = min(food if targets is None else targets,
                             key=lambda cell: self.get_maze_distance(my_pos, cell), default=None)
                if nearby is None:
                    feasting = False
                else:
                    distance = self.get_maze_distance
                    escape = min(distance(nearby, border) for border in self.home_border)
                    trip = distance(my_pos, nearby) + escape
                    feasting = (carrying < cap + 3 and distance(my_pos, nearby) <= 2
                                and trip + self.SCARED_MARGIN < min(min(scared), my_moves_left)
                                and nearby not in self.dead_ends)

        go_home = carrying > 0 and (
            (carrying >= (carry_limit or self.CARRY_LIMIT) and not feasting)
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
            path = self.a_star(my_pos, food if targets is None else targets, cost)
            if path is None:  # food unreachable, at least stay safe
                path = self.a_star(my_pos, self.home_border, cost)

        if path is None:
            return self.safest_action(game_state, visible)
        return self.first_step(game_state, path)

    def pick_detour(self, game_state, my_pos, capsules):
        """
        Plan B for a stuck attacker: an enemy capsule we can reach before any
        ghost, otherwise the border cell farthest from the ghosts, to cross there.
        """
        dist = self.get_maze_distance
        ghosts = [self.enemy_cell(game_state, i) for i in self.get_opponents(game_state)
                  if not game_state.get_agent_state(i).is_pacman]
        if not ghosts:
            return None
        free = [c for c in capsules if dist(my_pos, c) < min(dist(g, c) for g in ghosts)]
        if free:
            return min(free, key=lambda c: dist(my_pos, c))
        return max(self.home_border, key=lambda c: min(dist(g, c) for g in ghosts))

    #############
    # Defending #
    #############

    def invaders(self, game_state):
        """(cell, food carried, visible) of every enemy on our side; hidden ones use the most likely cell."""
        # is_pacman and num_carrying stay readable for hidden enemies, so we know about invaders we cannot see
        return [(self.enemy_cell(game_state, i), game_state.get_agent_state(i).num_carrying,
                 game_state.get_agent_position(i) is not None)
                for i in self.get_opponents(game_state) if game_state.get_agent_state(i).is_pacman]

    def defend(self, game_state):
        """
        Guard our food without leaving our side.
        Goal priority:
          1. chase invaders: their exact cell if visible, else their most likely
             cell from the beliefs (keep a distance of 2 while scared). With two
             invaders, go for the one carrying more food
          2. no invader: wait at the border crossing the enemies are expected to use
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        invaders = self.invaders(game_state)

        crossing = self.intercept(game_state, my_pos) if self.INTERCEPT and not my_state.scared_timer else None
        if crossing is not None:
            goals = [crossing]
        elif invaders:
            most = max(carrying for _, carrying, _ in invaders)
            targets = [cell for cell, carrying, _ in invaders if carrying == most]
            if my_state.scared_timer > 0:
                # Shadow the invader from 2 cells away so we can pounce when the timer ends
                goals = [cell for inv in targets for cell in self.cells_at_distance(inv, 2)]
                if self.SMART_RAIDS and (not goals or any(goal in targets for goal in goals)):
                    goals = [max(self.home_cells, key=lambda cell: (
                        min(self.get_maze_distance(cell, inv) for inv, _, _ in invaders),
                        -self.get_maze_distance(my_pos, cell)))]
            else:
                goals = targets
        else:
            goals = [self.guard_point(game_state)]

        path = self.a_star(my_pos, goals, self.stay_home)
        if path is None:
            path = self.a_star(my_pos, [self.patrol_point(game_state)], self.stay_home)
        if path is None:
            return self.safest_action(game_state, [])
        return self.first_step(game_state, path)

    def intercept(self, state, origin):
        if not self.is_home(origin):
            return None
        distance = self.get_maze_distance
        for invader, carried, seen in sorted(self.invaders(state), key=lambda item: -item[1]):
            if not seen or carried < 3 or not self.crossings:
                continue
            crossing = min(self.crossings, key=lambda cell: distance(invader, cell))
            escape = distance(invader, crossing)
            ours = distance(origin, crossing)
            if 0 < escape <= 5 and ours < escape and ours <= distance(origin, invader):
                return crossing
        return None

    def guard_point(self, game_state):
        """
        Where to wait while no enemy is on our side. Enemies come for our food, so
        each one is expected to cross at the border cell on its shortest way to the
        nearest dot. We want to be at that cell no later than the enemy: pick the
        home cell with the smallest lateness (our distance to the crossing minus
        the enemy's), taking the worse of the two when they use different crossings.
        Once some enemy has crossed, enemies that never have are ignored: they
        are probably defenders.
        """
        ghosts = [i for i in self.get_opponents(game_state) if not game_state.get_agent_state(i).is_pacman]
        watched = [i for i in ghosts if i in self.beliefs.attackers] or ghosts
        food = self.get_food_you_are_defending(game_state).as_list()
        if not watched or not food:
            return self.patrol_point(game_state)
        dist = self.get_maze_distance
        to_food = {c: min(dist(c, f) for f in food) for c in self.home_border}

        arrival = {}  # expected crossing -> steps until an enemy gets there
        for i in watched:
            enemy = self.enemy_cell(game_state, i)
            crossing = min(self.home_border, key=lambda c: dist(enemy, c) + to_food[c])
            arrival[crossing] = min(arrival.get(crossing, float('inf')), dist(enemy, crossing))
        return min(self.home_cells, key=lambda p: max(dist(p, c) - steps for c, steps in arrival.items()))

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


class AStarOffensiveAgent(AStarAgent):
    """
    Attacks. While it is on our side it also eats an invader that is close by
    and cannot step back to its side first.
    """

    HELP_DIST = 4        # chase a visible invader within this maze distance

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        if not my_state.is_pacman and my_state.scared_timer == 0:
            # We are a ghost here, so an invader passing close by is a cheap catch. Only go for
            # one that cannot step back to its side before we reach it, or it just keeps us busy
            if self.COVER_RAIDER:
                teammate = next(i for i in self.get_team(game_state) if i != self.index)
                if game_state.get_agent_state(teammate).is_pacman:
                    invaders = self.invaders(game_state)
                    distance = self.get_maze_distance
                    if any((visible or load >= 2) and distance(my_pos, cell) <= 7
                           for cell, load, visible in invaders):
                        self.trail.clear()
                        return self.defend(game_state)
            catchable = [cell for cell, _, visible in self.invaders(game_state)
                         if visible and self.get_maze_distance(my_pos, cell)
                         <= min(self.HELP_DIST, self.beliefs.to_their_border[cell])]
            path = self.a_star(my_pos, catchable, self.stay_home)
            if path:
                self.trail.clear()  # chasing an invader is not being stuck
                return self.first_step(game_state, path)
        return self.attack(game_state)


class AStarDefensiveAgent(AStarAgent):
    """
    Defends, but raids food near the border while no enemy is on our side or
    close to it. Enemies stay flagged as Pacmen even when hidden, so it knows the
    moment one crosses and runs back home.
    """

    RAID_RANGE = 8       # only food within this many steps of our border
    RAID_CARRY = 3       # food carried before heading home on a raid
    RAID_ALERT = 6       # no raid while a watched enemy is this close to our border

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.to_home_border = EnemyBeliefs.distances_from(self, self.crossings or self.home_border)
        self.raid_goal = None

    def choose_action(self, game_state):
        if self.SMART_RAIDS:
            return self.switch_roles(game_state)
        if not self.invaders(game_state):
            targets = self.raid_targets(game_state)
            if targets:
                return self.attack(game_state, carry_limit=self.RAID_CARRY, targets=targets)
        if game_state.get_agent_state(self.index).is_pacman:
            return self.retreat(game_state)
        return self.defend(game_state)

    def switch_roles(self, state):
        mine = state.get_agent_state(self.index)
        origin = mine.get_position()
        distance = self.get_maze_distance
        if self.invaders(state):
            self.raid_goal = None
            return self.retreat(state) if mine.is_pacman else self.defend(state)

        teammate = next(i for i in self.get_team(state) if i != self.index)
        partner = state.get_agent_state(teammate)
        moves_left = (state.data.timeleft + state.get_num_agents() - 1) // state.get_num_agents()
        score = self.get_score(state)
        raid_limit = 2 if score > 0 and moves_left < 65 else self.RAID_CARRY
        if mine.is_pacman and (mine.num_carrying >= raid_limit or moves_left <= 8):
            self.raid_goal = None
            return self.retreat(state)

        food = self.get_food(state).as_list()
        far = float('inf')
        blockers = [i for i in self.get_opponents(state)
                    if not state.get_agent_state(i).is_pacman]
        watched = [i for i in blockers if i in self.beliefs.attackers] or blockers
        entry = far
        for i in watched:
            current = state.get_agent_position(i)
            if current is not None:
                entry = min(entry, self.to_home_border.get(current, far))
            else:
                near = sorted(self.beliefs.dist[i].items(),
                              key=lambda item: self.to_home_border.get(item[0], far))
                probability = 0
                for cell, mass in near:
                    probability += mass
                    if probability >= 0.35:
                        entry = min(entry, self.to_home_border.get(cell, far))
                        break

        # If the other agent is home and active, a slightly longer raid can be covered.
        cover = not partner.is_pacman and not partner.scared_timer
        slack = 5 if cover and self.PRESSURE else 2 if cover else 0
        if score > 0 and moves_left < 55:
            slack -= 2
        if entry <= (3 if self.PRESSURE else 4):
            self.raid_goal = None
            return self.retreat(state) if mine.is_pacman else self.defend(state)

        ghosts = self.dangerous_ghosts(state)
        cost = self.ghost_cost(ghosts)
        partner_pos = partner.get_position()
        near_food = sorted(food, key=lambda target: (distance(origin, target), target))[:6]
        if self.raid_goal in food and self.raid_goal not in near_food:
            near_food.append(self.raid_goal)
        plans = []
        for target in near_food:
            back = self.to_home_border.get(target, far)
            if back > self.RAID_RANGE or target in self.dead_ends:
                continue
            eta = distance(origin, target)
            if back + eta + 1 > min(moves_left, entry + slack):
                continue
            if any(weight >= 0.15 and distance(ghost, target) <= min(3, eta + 1)
                   for ghost, weight in ghosts):
                continue
            nearby = sum(distance(target, other) <= 2 for other in near_food)
            overlap = (partner.is_pacman and distance(partner_pos, target) <= min(2, eta))
            value = 1 + 0.22 * min(3, nearby) - 0.13 * (eta + back) - 0.75 * overlap
            value += 0.30 * (target == self.raid_goal)
            plans.append((value, target))
        for _, target in sorted(plans, reverse=True)[:2]:
            path = self.a_star(origin, [target], cost)
            back = self.a_star(target, self.crossings or self.home_border, cost)
            if path and back and len(path) + len(back) + 1 <= min(moves_left, entry + slack):
                self.raid_goal = target
                return self.first_step(state, path)

        self.raid_goal = None
        return self.retreat(state) if mine.is_pacman else self.defend(state)

    def raid_targets(self, game_state):
        """Food worth raiding, or [] if raiding is unsafe right now."""
        ghosts = [i for i in self.get_opponents(game_state) if not game_state.get_agent_state(i).is_pacman]
        watched = [i for i in ghosts if i in self.beliefs.attackers] or ghosts
        for i in watched:
            if self.to_home_border.get(self.enemy_cell(game_state, i), 0) <= self.RAID_ALERT:
                return []
        return [f for f in self.get_food(game_state).as_list()
                if self.to_home_border.get(f, float('inf')) <= self.RAID_RANGE]

    def retreat(self, game_state):
        """Back to our side, avoiding ghosts."""
        my_pos = game_state.get_agent_position(self.index)
        ghosts = self.dangerous_ghosts(game_state)
        path = self.a_star(my_pos, self.home_border, self.ghost_cost(ghosts))
        if path is None:
            return self.safest_action(game_state, [g for g, weight in ghosts if weight == 1.0])
        return self.first_step(game_state, path)
