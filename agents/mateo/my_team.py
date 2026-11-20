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


def moves_left(game_state):
    """Moves each agent still gets: the clock counts every agent's move, rounded up."""
    agents = game_state.get_num_agents()
    return (game_state.data.timeleft + agents - 1) // agents


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
    SCARED_MARGIN = 4    # spare moves kept when racing the end of the enemy's scared time home

    # Feasting: past the carry limit while enemy ghosts are scared (see attack)
    FEAST_REACH = 2      # only food at most this many steps away is worth the extra risk
    FEAST_EXTRA = 3      # at most this many dots beyond the carry limit

    # Protecting a lead: when ahead with few moves left, bank smaller loads
    LEAD_MOVES = 40      # moves left at which the attacker starts banking early
    LEAD_CARRY = 5       # carry limit for the attacker then

    # Intercepting: block the crossing a loaded invader is heading for (see intercept)
    INTERCEPT_CARRY = 3  # only invaders carrying at least this many dots
    INTERCEPT_RANGE = 5  # only when the invader is at most this many steps from its crossing

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
        # Border cells with an open neighbour on the enemy side: where agents actually cross
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
        return move if move in legal else legal[0]  # never forfeit a turn on an illegal move

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
        """
        Dead ends on the enemy side: repeatedly remove cells with a single open
        neighbour; the removed cells are corridors with one way out. Returns
        {cell: (exit cell, steps to the exit)}; callers only test membership.
        """
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

    def attack(self, game_state):
        """
        Collect food on the enemy side and bank it before it gets risky.
        Goal priority:
          1. finish a detour, if we got stuck (see pick_detour); a detour is
             dropped when we carry food and a ghost is close or time runs short
          2. go home if time is running out, the bag is full, or a ghost is close
             (via a capsule if that is closer than home). The bag limit drops to
             LEAD_CARRY when we are ahead late in the game, and after a capsule we
             may feast past it: up to FEAST_EXTRA more dots, each at most
             FEAST_REACH steps away and outside dead ends, while the enemy ghosts
             stay scared long enough for the trip home
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

        home_dist = min(self.get_maze_distance(my_pos, b) for b in self.home_border)
        my_moves_left = moves_left(game_state)

        if carrying and (threatened or home_dist + self.TIME_MARGIN >= my_moves_left):
            # Banking comes first: a detour would only keep the food at risk
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

        limit = self.CARRY_LIMIT
        if self.get_score(game_state) > 0 and my_moves_left <= self.LEAD_MOVES:
            limit = self.LEAD_CARRY

        # After a capsule every enemy ghost may be scared; the bag limit can then wait
        # for one more nearby dot, as long as the trip home beats the scared timer
        scared = [game_state.get_agent_state(i).scared_timer for i in self.get_opponents(game_state)
                  if not game_state.get_agent_state(i).is_pacman]
        feasting = bool(scared) and min(scared) > home_dist + self.SCARED_MARGIN
        if feasting and carrying >= limit:
            dist = self.get_maze_distance
            nearby = min(food, key=lambda cell: dist(my_pos, cell), default=None)
            if nearby is None:
                feasting = False
            else:
                trip = dist(my_pos, nearby) + min(dist(nearby, b) for b in self.home_border)
                feasting = (carrying < limit + self.FEAST_EXTRA
                            and dist(my_pos, nearby) <= self.FEAST_REACH
                            and trip + self.SCARED_MARGIN < min(min(scared), my_moves_left)
                            and nearby not in self.dead_ends)

        go_home = carrying > 0 and (
            (carrying >= limit and not feasting)
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
          1. block the crossing a loaded invader is about to escape through, if
             we get there first (see intercept; not while scared)
          2. chase invaders: their exact cell if visible, else their most likely
             cell from the beliefs. With two invaders, go for the one carrying
             more food. While scared, shadow it from 2 cells away; if no such
             cell exists, keep as far from the invaders as possible
          3. no invader: wait at the border crossing the enemies are expected to use
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        invaders = self.invaders(game_state)

        crossing = None if my_state.scared_timer else self.intercept(game_state, my_pos)
        if crossing is not None:
            goals = [crossing]
        elif invaders:
            most = max(carrying for _, carrying, _ in invaders)
            targets = [cell for cell, carrying, _ in invaders if carrying == most]
            if my_state.scared_timer > 0:
                # Shadow the invader from 2 cells away so we can pounce when the timer ends
                goals = [cell for inv in targets for cell in self.cells_at_distance(inv, 2)]
                if not goals or any(goal in targets for goal in goals):
                    # cells_at_distance found nothing and fell back to the invader's own cell: flee
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

    def intercept(self, game_state, my_pos):
        """
        The crossing to block, or None. A visible invader carrying at least
        INTERCEPT_CARRY dots and at most INTERCEPT_RANGE steps from its nearest
        crossing is about to bank them; if we reach that crossing strictly
        before it (and no later than we could reach the invader itself),
        waiting there catches it on the way out. Loaded invaders first.
        """
        if not self.is_home(my_pos) or not self.crossings:
            return None
        dist = self.get_maze_distance
        for invader, carried, seen in sorted(self.invaders(game_state), key=lambda item: -item[1]):
            if not seen or carried < self.INTERCEPT_CARRY:
                continue
            crossing = min(self.crossings, key=lambda cell: dist(invader, cell))
            escape = dist(invader, crossing)
            ours = dist(my_pos, crossing)
            if 0 < escape <= self.INTERCEPT_RANGE and ours < escape and ours <= dist(my_pos, invader):
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
        """Home cells exactly dist maze steps from pos; [pos] itself if there are none."""
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
    Attacks. While it is still on our side as a ghost, it first helps at home:
    it defends when the defender is out raiding and an invader is close, and it
    eats an invader that is close by and cannot step back to its side first.
    """

    HELP_DIST = 4        # chase a visible invader within this maze distance
    COVER_DIST = 7       # defend instead of attacking while the raider is out and an invader is this close
    COVER_LOAD = 2       # a hidden invader counts for covering only if it carries at least this many dots

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        if not my_state.is_pacman and my_state.scared_timer == 0:
            teammate = next(i for i in self.get_team(game_state) if i != self.index)
            if game_state.get_agent_state(teammate).is_pacman:
                # The defender is raiding, so nobody guards our food: cover for it
                if any((visible or load >= self.COVER_LOAD) and self.get_maze_distance(my_pos, cell) <= self.COVER_DIST
                       for cell, load, visible in self.invaders(game_state)):
                    self.trail.clear()
                    return self.defend(game_state)
            # We are a ghost here, so an invader passing close by is a cheap catch. Only go for
            # one that cannot step back to its side before we reach it, or it just keeps us busy
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
    Defends, but raids food near the border while no enemy is on our side and
    the round trip fits before an enemy attacker could reach our border.
    Enemies stay flagged as Pacmen even when hidden, so it knows the moment one
    crosses and runs back home.
    """

    RAID_RANGE = 8       # only food within this many steps of our border
    RAID_CARRY = 3       # food carried before heading home on a raid
    RAID_ALERT = 4       # no raid while a watched enemy can reach our border within this many steps
    LAST_MOVES = 8       # head home for good with this many moves left

    # Protecting a lead: when ahead late in the game, raid shorter
    LEAD_RAID_MOVES = 65  # moves left from which the raid carry limit drops to LEAD_RAID_CARRY
    LEAD_RAID_CARRY = 2
    LEAD_SLACK_MOVES = 55  # moves left from which COVER_SLACK shrinks by LEAD_SLACK_CUT
    LEAD_SLACK_CUT = 2

    # Timing the raid against the enemy attacker (see enemy_entry and choose_action)
    ENTRY_CONFIDENCE = 0.35  # a hidden enemy is assumed at the closest cell holding this much probability
    COVER_SLACK = 2      # extra steps a raid may take while our attacker is home to cover

    # Choosing a raid target (see plan_raid)
    RAID_CANDIDATES = 6  # nearest dots considered each turn
    RAID_PLANS = 2       # best-scoring dots checked with full A* paths there and back
    GHOST_RADIUS = 3     # skip a dot this close to a likely ghost...
    GHOST_MIN_WEIGHT = 0.15  # ...where likely means a visible ghost or a guess with this much weight
    CLUSTER_RADIUS = 2   # dots this close to the target count as a cluster
    CLUSTER_MAX = 3      # at most this many cluster dots add to the value
    CLUSTER_BONUS = 0.22  # value per cluster dot
    TRIP_PENALTY = 0.13  # value lost per step of the round trip
    OVERLAP_RADIUS = 2   # a dot this close to our raiding attacker is its business...
    OVERLAP_PENALTY = 0.75  # ...and loses this much value
    STICKY_BONUS = 0.30  # value added to last turn's target so the raid does not flip-flop

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.to_home_border = EnemyBeliefs.distances_from(self, self.crossings or self.home_border)
        self.raid_goal = None  # food the current raid is heading for

    def choose_action(self, game_state):
        """
        1. any invader: go home if raiding, then defend
        2. raiding with a full bag (RAID_CARRY, or LEAD_RAID_CARRY when ahead
           late) or almost out of time: go home
        3. no raid while a watched enemy is within RAID_ALERT of our border
        4. otherwise raid the best dot whose round trip fits before the enemy
           could arrive (plus COVER_SLACK while our attacker is home); if none
           fits, go home or defend
        """
        my_state = game_state.get_agent_state(self.index)
        if self.invaders(game_state):
            return self.stand_down(game_state)

        left = moves_left(game_state)
        ahead = self.get_score(game_state) > 0
        raid_limit = self.LEAD_RAID_CARRY if ahead and left < self.LEAD_RAID_MOVES else self.RAID_CARRY
        if my_state.is_pacman and (my_state.num_carrying >= raid_limit or left <= self.LAST_MOVES):
            self.raid_goal = None
            return self.retreat(game_state)

        entry = self.enemy_entry(game_state)
        if entry <= self.RAID_ALERT:
            return self.stand_down(game_state)

        teammate = next(i for i in self.get_team(game_state) if i != self.index)
        partner = game_state.get_agent_state(teammate)
        covered = not partner.is_pacman and not partner.scared_timer
        slack = self.COVER_SLACK if covered else 0
        if ahead and left < self.LEAD_SLACK_MOVES:
            slack -= self.LEAD_SLACK_CUT

        action = self.plan_raid(game_state, partner, min(left, entry + slack))
        return action if action is not None else self.stand_down(game_state)

    def stand_down(self, game_state):
        """End any raid: back home if on the enemy side, otherwise defend."""
        self.raid_goal = None
        if game_state.get_agent_state(self.index).is_pacman:
            return self.retreat(game_state)
        return self.defend(game_state)

    def enemy_entry(self, game_state):
        """
        Fewest steps a watched enemy needs to reach our border: its distance if
        visible, otherwise the distance of the closest cell by which its belief
        has gathered ENTRY_CONFIDENCE of its probability. Watched enemies are the
        ghosts that have attacked before (all ghosts until one has).
        """
        far = float('inf')
        ghosts = [i for i in self.get_opponents(game_state) if not game_state.get_agent_state(i).is_pacman]
        watched = [i for i in ghosts if i in self.beliefs.attackers] or ghosts
        entry = far
        for i in watched:
            pos = game_state.get_agent_position(i)
            if pos is not None:
                entry = min(entry, self.to_home_border.get(pos, far))
                continue
            mass = 0
            for cell, p in sorted(self.beliefs.dist[i].items(), key=lambda item: self.to_home_border.get(item[0], far)):
                mass += p
                if mass >= self.ENTRY_CONFIDENCE:
                    entry = min(entry, self.to_home_border.get(cell, far))
                    break
        return entry

    def plan_raid(self, game_state, partner, budget):
        """
        First step toward the best raid target whose round trip fits in budget
        steps, or None. Candidates are the RAID_CANDIDATES nearest dots (plus
        last turn's target) within RAID_RANGE of our border, outside dead ends
        and away from likely ghosts. Value rewards clusters and penalizes trip
        length and overlap with our attacker; the best RAID_PLANS are checked
        with real A* paths, since ghosts can make the way longer.
        """
        far = float('inf')
        dist = self.get_maze_distance
        my_pos = game_state.get_agent_position(self.index)
        food = self.get_food(game_state).as_list()
        ghosts = self.dangerous_ghosts(game_state)
        cost = self.ghost_cost(ghosts)
        partner_pos = partner.get_position()

        candidates = sorted(food, key=lambda target: (dist(my_pos, target), target))[:self.RAID_CANDIDATES]
        if self.raid_goal in food and self.raid_goal not in candidates:
            candidates.append(self.raid_goal)
        plans = []
        for target in candidates:
            back = self.to_home_border.get(target, far)
            if back > self.RAID_RANGE or target in self.dead_ends:
                continue
            eta = dist(my_pos, target)
            if back + eta + 1 > budget:
                continue
            if any(weight >= self.GHOST_MIN_WEIGHT and dist(ghost, target) <= min(self.GHOST_RADIUS, eta + 1)
                   for ghost, weight in ghosts):
                continue
            cluster = sum(dist(target, other) <= self.CLUSTER_RADIUS for other in candidates)
            overlap = partner.is_pacman and dist(partner_pos, target) <= min(self.OVERLAP_RADIUS, eta)
            value = (1 + self.CLUSTER_BONUS * min(self.CLUSTER_MAX, cluster)
                     - self.TRIP_PENALTY * (eta + back) - self.OVERLAP_PENALTY * overlap
                     + self.STICKY_BONUS * (target == self.raid_goal))
            plans.append((value, target))

        for _, target in sorted(plans, reverse=True)[:self.RAID_PLANS]:
            path = self.a_star(my_pos, [target], cost)
            back = self.a_star(target, self.crossings or self.home_border, cost)
            if path and back and len(path) + len(back) + 1 <= budget:
                self.raid_goal = target
                return self.first_step(game_state, path)
        return None

    def retreat(self, game_state):
        """Back to our side, avoiding ghosts."""
        my_pos = game_state.get_agent_position(self.index)
        ghosts = self.dangerous_ghosts(game_state)
        path = self.a_star(my_pos, self.home_border, self.ghost_cost(ghosts))
        if path is None:
            return self.safest_action(game_state, [g for g, weight in ghosts if weight == 1.0])
        return self.first_step(game_state, path)
