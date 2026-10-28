# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import math
import os
import random
import time

import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Actions, Directions
from contest.util import nearest_point


##############
# Parameters #
##############

# Seconds MCTS may think per move. The game warns at 1 s (3rd warning = loss).
# Measured: a whole move takes up to ~0.2 s longer than the budget (state copy,
# freeing the tree), so 0.5 keeps moves under ~0.7 s. The limit is per agent,
# so attacker and defender can both use the full budget.
# For quick local tests: MCTS_BUDGET=0.1 python capture.py ...
MCTS_BUDGET = float(os.environ.get('MCTS_BUDGET', 0.2))
ROLLOUT_DEPTH = 12        # simulated moves per rollout before evaluating
UCB_C = 1.0               # exploration constant (rewards are roughly in [-3, 3])
GREEDY_PROB = 0.5         # rollout policy: chance to step towards the target instead of randomly
RETURN_AFTER = 4          # rollout policy / invader model: head home once carrying this many dots
FLEE_DIST = 3             # invader model: run when the (not scared) defender is this close


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='MCTSOffensiveAgent', second='MCTSDefensiveAgent', num_training=0):
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


#########
# World #
#########

class World:
    """What never changes during a game. Built once in register_initial_state,
    shared by every simulated state (the reflex agent doesn't need this)."""

    def __init__(self, neighbours, is_red, mid_x, distances, home_dist, enemy_home_dist):
        self.neighbours = neighbours            # dict: (x, y) -> list of free cells you can step to
        self.is_red = is_red
        self.mid_x = mid_x
        self.distances = distances              # dict: ((x1, y1), (x2, y2)) -> maze distance
        self.home_dist = home_dist              # dict: (x, y) -> maze distance to OUR nearest border cell
        self.enemy_home_dist = enemy_home_dist  # dict: (x, y) -> maze distance to THEIR nearest border cell

        # defender moves: neighbours that stay on our side, plus waiting in place
        self.guard_moves = {cell: [c for c in cells if self.on_my_side(c)] + [cell]
                            for cell, cells in neighbours.items()}

    def distance(self, a, b):
        return self.distances[(a, b)]

    def on_my_side(self, pos):
        if self.is_red:
            return pos[0] < self.mid_x
        return pos[0] >= self.mid_x


def build_world(agent, game_state):
    """Shared by both MCTS agents; called from register_initial_state."""
    walls = game_state.get_walls()
    mid_x = walls.width // 2

    # neighbours of every free cell (N/S/E/W, no Stop)
    neighbours = {}
    for x in range(walls.width):
        for y in range(walls.height):
            if walls[x][y]:
                continue
            cells = []
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                    cells.append((nx, ny))
            neighbours[(x, y)] = cells

    # the column on each side of the middle line
    border_x = mid_x - 1 if agent.red else mid_x
    enemy_border_x = mid_x if agent.red else mid_x - 1
    border = [(border_x, y) for y in range(walls.height) if not walls[border_x][y]]
    enemy_border = [(enemy_border_x, y) for y in range(walls.height) if not walls[enemy_border_x][y]]

    # distance from every cell to the nearest border cell (used every evaluate)
    home_dist = {cell: min(agent.get_maze_distance(cell, b) for b in border)
                 for cell in neighbours}
    enemy_home_dist = {cell: min(agent.get_maze_distance(cell, b) for b in enemy_border)
                       for cell in neighbours}

    return World(neighbours, agent.red, mid_x, agent.distancer._distances, home_dist, enemy_home_dist)


def direction_to(src, dst):
    """Action that moves from cell src to the neighbouring cell dst (STOP if equal)."""
    return Actions.vector_to_direction((dst[0] - src[0], dst[1] - src[1]))


########
# MCTS #
########
#
# mcts_search works with any state that has legal_moves(), step(move),
# is_terminal(), rollout_move() and evaluate(). AttackState and DefenseState
# both provide them.

class AttackState:
    """Simulated game for the attacker: I move, then the visible ghosts chase me."""

    def __init__(self, world, my_pos, ghosts, food, carrying, returned=0, dead=False):
        self.world = world         # shared, not copied
        self.my_pos = my_pos       # (x, y)
        self.ghosts = ghosts       # list of (x, y) of dangerous enemy ghosts
        self.food = food           # set of (x, y) you can still eat
        self.carrying = carrying   # food currently carried
        self.returned = returned   # food brought home during this simulation
        self.dead = dead

    def copy(self):
        return AttackState(
            world=self.world,          # shared
            my_pos=self.my_pos,
            ghosts=list(self.ghosts),  # own copy
            food=set(self.food),       # own copy
            carrying=self.carrying,
            returned=self.returned,
            dead=self.dead,
        )

    def legal_moves(self):
        return self.world.neighbours[self.my_pos]

    def on_my_side(self, pos):
        return self.world.on_my_side(pos)

    def is_terminal(self):
        return self.dead or len(self.food) == 0

    def step(self, move):
        """One simulated turn: I move to `move` (a cell), then every ghost moves."""
        new = self.copy()
        new.my_pos = move

        # eat?
        if new.my_pos in new.food:
            new.food.remove(new.my_pos)
            new.carrying += 1

        # home?
        if new.on_my_side(new.my_pos):
            new.returned += new.carrying
            new.carrying = 0

        # caught before ghosts move? (I walked into a ghost)
        if new.my_pos in new.ghosts:
            new.dead = True
            new.carrying = 0
            return new

        # ghosts move (greedy chase, staying on their own side)
        def dist_to_pacman(cell):
            return new.world.distance(cell, new.my_pos)

        moved = []
        for ghost in new.ghosts:
            options = []
            for cell in new.world.neighbours[ghost]:
                if not new.on_my_side(cell):
                    options.append(cell)
            if options:
                best = min(options, key=dist_to_pacman)
                moved.append(best)
            else:
                moved.append(ghost)
        new.ghosts = moved

        # caught after ghosts move? (a ghost walked into me)
        if new.my_pos in new.ghosts:
            new.dead = True
            new.carrying = 0

        return new

    def evaluate(self):
        """How good is this simulated state for my attacker? Roughly in [-1, 3]."""
        if self.dead:
            return -1.0

        world = self.world
        value = 1.0 * self.returned + 0.6 * self.carrying   # carried food is still at risk

        # pull towards the nearest food
        if self.food:
            nearest_food = min(world.distance(self.my_pos, f) for f in self.food)
            value -= 0.02 * nearest_food

        # the more I carry, the more being far from home costs
        if self.carrying > 0:
            value -= 0.01 * self.carrying * world.home_dist[self.my_pos]

        return value

    def rollout_move(self):
        """Cheap rollout policy: usually step towards the target, sometimes randomly."""
        moves = self.legal_moves()
        if random.random() > GREEDY_PROB:
            return random.choice(moves)

        world = self.world
        if self.carrying >= RETURN_AFTER or not self.food:
            return min(moves, key=lambda c: world.home_dist[c])
        target = min(self.food, key=lambda f: world.distance(self.my_pos, f))
        return min(moves, key=lambda c: world.distance(c, target))


class DefenseState:
    """Simulated game for the defender: I move (staying on our side), then every
    visible invader (enemy Pacman on our side) moves with a simple model."""

    def __init__(self, world, my_pos, invaders, food, scared,
                 caught=0, eaten=0, escaped=0, dead=False):
        self.world = world         # shared, not copied
        self.my_pos = my_pos       # (x, y)
        self.invaders = invaders   # list of (pos, carrying)
        self.food = food           # set of our dots still on the board
        self.scared = scared       # moves left that I'm scared: touching an invader then kills ME
        self.caught = caught       # invaders caught during this simulation
        self.eaten = eaten         # our dots eaten (minus dots dropped back by caught invaders)
        self.escaped = escaped     # dots invaders carried home (= points for them)
        self.dead = dead

    def copy(self):
        return DefenseState(self.world, self.my_pos, list(self.invaders), set(self.food), self.scared,
                            self.caught, self.eaten, self.escaped, self.dead)

    def legal_moves(self):
        return self.world.guard_moves[self.my_pos]

    def is_terminal(self):
        return self.dead or not self.invaders

    def step(self, move):
        """One simulated turn: I move to `move` (a cell), then every invader moves."""
        new = self.copy()
        new.my_pos = move

        # I walked into an invader? (capture.py checks this before counting down my scared timer)
        new.collide()
        if new.dead:
            return new
        new.scared = max(0, new.scared - 1)

        # invaders move
        moved = []
        for pos, carrying in new.invaders:
            pos = new.invader_move(pos, carrying)
            if not new.world.on_my_side(pos):   # made it home: what it carried is scored
                new.escaped += carrying
                continue
            if pos in new.food:
                new.food.remove(pos)
                carrying += 1
                new.eaten += 1
            moved.append((pos, carrying))
        new.invaders = moved

        # an invader walked into me?
        new.collide()
        return new

    def collide(self):
        """Resolve invaders standing on my cell."""
        remaining = []
        for pos, carrying in self.invaders:
            if pos != self.my_pos:
                remaining.append((pos, carrying))
            elif self.scared:
                self.dead = True                # scared ghost touched a Pacman: I respawn
                remaining.append((pos, carrying))
            else:
                self.caught += 1
                self.eaten -= carrying          # the dots it carried are dropped back on our side
        self.invaders = remaining

    def invader_move(self, pos, carrying):
        """Opponent model: eat me if I'm scared and next to it, run if I'm close,
        go home when carrying enough, otherwise walk to the nearest dot."""
        world = self.world
        options = world.neighbours[pos]
        me = self.my_pos

        if self.scared and me in options:
            return me
        if not self.scared and world.distance(pos, me) <= FLEE_DIST:
            return max(options, key=lambda c: world.distance(c, me) - 0.5 * world.enemy_home_dist[c])
        if carrying >= RETURN_AFTER or not self.food:
            return min(options, key=lambda c: world.enemy_home_dist[c])
        target = min(self.food, key=lambda f: world.distance(pos, f))
        return min(options, key=lambda c: world.distance(c, target))

    def evaluate(self):
        """How good is this simulated state for my defender? Roughly in [-3, 3]."""
        value = 1.0 * self.caught - 0.3 * self.eaten - 1.0 * self.escaped
        if self.dead:
            # Dying ends the simulation, so the dots eaten while I walk back from my
            # start are never counted. Make it clearly worse than any outcome where I
            # stay alive, otherwise MCTS learns that dying is the cheap option.
            return value - 10.0

        if self.invaders:
            nearest = min(self.world.distance(self.my_pos, p) for p, _ in self.invaders)
            if self.scared:
                nearest = abs(nearest - 2)      # shadow it from 2 cells while scared
            value -= 0.05 * nearest
        return value

    def rollout_move(self):
        """Cheap rollout policy: usually step towards the nearest invader, sometimes randomly."""
        moves = self.legal_moves()
        if random.random() > GREEDY_PROB or not self.invaders:
            return random.choice(moves)

        world = self.world
        target = min((p for p, _ in self.invaders), key=lambda p: world.distance(self.my_pos, p))
        if self.scared:
            return min(moves, key=lambda c: abs(world.distance(c, target) - 2))
        return min(moves, key=lambda c: world.distance(c, target))


def rollout(state, max_depth):
    """Play up to max_depth moves with the state's rollout policy, then evaluate."""
    for _ in range(max_depth):
        if state.is_terminal():
            break
        state = state.step(state.rollout_move())
    return state.evaluate()


class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move                          # the move that led from parent to here
        self.children = []
        self.untried = list(state.legal_moves())  # moves not expanded yet
        random.shuffle(self.untried)
        self.N = 0
        self.W = 0.0

    def ucb1(self, c):
        if self.N == 0:
            return float('inf')
        return self.W / self.N + c * math.sqrt(math.log(self.parent.N) / self.N)


def mcts_search(root_state, deadline, c=UCB_C, max_depth=ROLLOUT_DEPTH):
    """Returns (best move as a cell, number of iterations)."""
    root = Node(root_state)
    iterations = 0

    while time.perf_counter() < deadline:
        node = root

        # 1. select
        while not node.untried and node.children and not node.state.is_terminal():
            node = max(node.children, key=lambda child: child.ucb1(c))

        # 2. expand
        if node.untried and not node.state.is_terminal():
            move = node.untried.pop()
            child = Node(node.state.step(move), parent=node, move=move)
            node.children.append(child)
            node = child

        # 3. rollout
        reward = rollout(node.state, max_depth)

        # 4. backpropagate
        while node is not None:
            node.N += 1
            node.W += reward
            node = node.parent

        iterations += 1

    if not root.children:
        return None, iterations
    best = max(root.children, key=lambda child: child.N)
    return best.move, iterations


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class MCTSOffensiveAgent(OffensiveReflexAgent):
    """Attacker: MCTS while on the enemy side, reflex agent on our own side
    and as a fallback if anything goes wrong."""

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)   # sets self.start, self.red, maze distances
        self.world = build_world(self, game_state)
        self.last_iterations = 0

    def build_sim_state(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        my_pos = (int(my_pos[0]), int(my_pos[1]))
        food = set(self.get_food(game_state).as_list())

        ghosts = []
        for i in self.get_opponents(game_state):
            enemy = game_state.get_agent_state(i)
            pos = enemy.get_position()
            # dangerous = visible, a ghost (not a Pacman), not scared (or about to stop being scared)
            if pos is not None and not enemy.is_pacman and enemy.scared_timer <= 2:
                ghosts.append((int(pos[0]), int(pos[1])))

        return AttackState(self.world, my_pos, ghosts, food, my_state.num_carrying)

    def choose_action(self, game_state):
        start = time.perf_counter()
        legal = game_state.get_legal_actions(self.index)

        # on our own side there is nothing to plan: walk to food with the reflex agent
        if not game_state.get_agent_state(self.index).is_pacman:
            return super().choose_action(game_state)

        try:
            root_state = self.build_sim_state(game_state)
            best_cell, self.last_iterations = mcts_search(root_state, deadline=start + MCTS_BUDGET)
            if best_cell is not None:
                action = direction_to(root_state.my_pos, best_cell)
                if action in legal:
                    return action
        except Exception as e:
            print(f'MCTS failed, falling back to reflex: {e!r}')

        return super().choose_action(game_state)


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


class MCTSDefensiveAgent(DefensiveReflexAgent):
    """Defender: MCTS while an invader is visible and I'm on our side, reflex
    agent otherwise and as a fallback if anything goes wrong."""

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)   # sets self.start, self.red, maze distances
        self.world = build_world(self, game_state)
        self.last_iterations = 0

    def build_sim_state(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        my_pos = (int(my_pos[0]), int(my_pos[1]))
        food = set(self.get_food_you_are_defending(game_state).as_list())

        invaders = []
        for i in self.get_opponents(game_state):
            enemy = game_state.get_agent_state(i)
            pos = enemy.get_position()
            # invader = visible and a Pacman (on our side)
            if pos is not None and enemy.is_pacman:
                invaders.append(((int(pos[0]), int(pos[1])), enemy.num_carrying))

        return DefenseState(self.world, my_pos, invaders, food, my_state.scared_timer)

    def choose_action(self, game_state):
        start = time.perf_counter()
        legal = game_state.get_legal_actions(self.index)

        try:
            # on our side with a visible invader: plan. Otherwise the reflex agent patrols
            # (or walks back if I'm on the enemy side).
            if not game_state.get_agent_state(self.index).is_pacman:
                root_state = self.build_sim_state(game_state)
                if root_state.invaders:
                    best_cell, self.last_iterations = mcts_search(root_state, deadline=start + MCTS_BUDGET)
                    if best_cell is not None:
                        action = direction_to(root_state.my_pos, best_cell)
                        if action in legal:
                            return action
        except Exception as e:
            print(f'MCTS failed, falling back to reflex: {e!r}')

        return super().choose_action(game_state)
