# myTeam.py
# ---------
# Pacman Capture the Flag - Smart Agent Implementation
# Uses particle filtering for enemy tracking, A* search for pathfinding,
# dead-end detection, and evaluation-based action selection.

import random
import math
from collections import deque

import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point, manhattan_distance


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='SmartOffensiveAgent', second='SmartDefensiveAgent', num_training=0):
    """
    Returns a list of two agents forming the team.
    First agent is offensive, second is defensive.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


####################
# Shared State     #
####################

class SharedState:
    """
    Shared information between the two agents on the same team.
    Includes particle filters for enemy tracking and strategic state.
    """

    def __init__(self):
        self.initialized = False
        # Particle filters: dict mapping enemy index -> list of (x, y) positions
        self.particles = {}
        self.num_particles = 300
        # Previous food state for detecting eaten food
        self.prev_food = None
        # Track which positions have been identified as dead-ends
        self.dead_ends = set()
        # Boundary positions (the column of positions at the team border)
        self.boundary_positions = []
        # Legal positions cache
        self.legal_positions = []
        # ==== Teammate coordination ====
        self.agent_targets = {}      # agent_index -> target food position
        self.chase_targets = {}      # agent_index -> invader index being chased
        # ==== Bottleneck detection ====
        self.bottleneck_pos = None       # Best bottleneck position (or None)
        self.bottleneck_food_count = 0   # How many food items it blocks
        self.bottleneck_covers_capsule = False  # Whether it blocks capsules too
        # ==== Food depth analysis ====
        self.food_depths = {}  # food_pos -> round-trip depth from boundary

    def initialize(self, game_state, agent):
        """Initialize shared state on the first call."""
        if self.initialized:
            return

        walls = game_state.get_walls()
        width = walls.width
        height = walls.height

        # Cache all legal positions
        self.legal_positions = []
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    self.legal_positions.append((x, y))

        # Compute boundary positions (positions right at the border on our side)
        halfway = width // 2
        if agent.red:
            border_x = halfway - 1
        else:
            border_x = halfway

        self.boundary_positions = [(border_x, y) for y in range(height)
                                   if not walls[border_x][y]]

        # Determine enemy half positions for particle initialization
        if agent.red:
            enemy_positions = [pos for pos in self.legal_positions if pos[0] >= halfway]
        else:
            enemy_positions = [pos for pos in self.legal_positions if pos[0] < halfway]

        # Initialize particle filters for each opponent
        opponents = agent.get_opponents(game_state)
        for opp in opponents:
            self.particles[opp] = [random.choice(enemy_positions) for _ in range(self.num_particles)]

        # Detect dead-end positions
        self._compute_dead_ends(walls, width, height)

        # Detect bottleneck positions on our side
        our_food = agent.get_food_you_are_defending(game_state).as_list()
        our_capsules = agent.get_capsules_you_are_defending(game_state)
        self._compute_bottlenecks(walls, width, height, agent.red, our_food, our_capsules)

        # Compute food depth for enemy food (used by offensive agent)
        enemy_food = agent.get_food(game_state).as_list()
        self._compute_food_depths(walls, width, height, agent.red, enemy_food, agent)

        # Store previous defending food
        self.prev_food = our_food

        self.initialized = True

    def _compute_dead_ends(self, walls, width, height):
        """
        Identify dead-end positions on the map.
        A dead-end is a position with only one exit (corridor leading to a single way out).
        """
        self.dead_ends = set()
        # Find positions with only one non-wall neighbor (true dead ends)
        for x in range(width):
            for y in range(height):
                if walls[x][y]:
                    continue
                neighbors = 0
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and not walls[nx][ny]:
                        neighbors += 1
                if neighbors == 1:
                    self.dead_ends.add((x, y))

        # Extend dead-ends to corridors leading to dead-ends (depth 2)
        extended = set()
        for pos in self.dead_ends:
            x, y = pos
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not walls[nx][ny]:
                    # Check if this neighbor is a corridor (exactly 2 non-wall neighbors)
                    neighbor_count = 0
                    for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nnx, nny = nx + ddx, ny + ddy
                        if 0 <= nnx < width and 0 <= nny < height and not walls[nnx][nny]:
                            neighbor_count += 1
                    if neighbor_count <= 2:
                        extended.add((nx, ny))
        self.dead_ends.update(extended)

    def _compute_bottlenecks(self, walls, width, height, is_red, food_list, capsule_list):
        """
        Find bottleneck positions on our side of the map.
        A bottleneck is a position that, when blocked, cuts off the most food
        from the boundary entry points.
        Uses BFS reachability: for each candidate, remove it and count
        how many food/capsules become unreachable from boundary.
        """
        halfway = width // 2

        # Get all our-side legal positions
        if is_red:
            our_positions = set(pos for pos in self.legal_positions if pos[0] < halfway)
        else:
            our_positions = set(pos for pos in self.legal_positions if pos[0] >= halfway)

        food_set = set(food_list)
        capsule_set = set(capsule_list) if capsule_list else set()
        total_food = len(food_set)

        if total_food == 0:
            return

        boundary_set = set(self.boundary_positions)

        def bfs_reachable(blocked_pos):
            """BFS from boundary positions, skipping blocked_pos. Returns reachable positions."""
            visited = set()
            if blocked_pos:
                visited.add(blocked_pos)
            queue = []
            for bp in self.boundary_positions:
                if bp != blocked_pos:
                    queue.append(bp)
                    visited.add(bp)

            while queue:
                current = queue.pop(0)
                cx, cy = int(current[0]), int(current[1])
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    npos = (nx, ny)
                    if npos in our_positions and npos not in visited:
                        visited.add(npos)
                        queue.append(npos)
            return visited

        # Pre-compute: all food reachable without any blocking
        all_reachable = bfs_reachable(None)
        baseline_reachable_food = food_set & all_reachable

        # Only check candidate positions that are corridors (≤ 3 neighbors)
        # and on our side, and not on the boundary itself
        candidates = []
        for pos in our_positions:
            if pos in boundary_set:
                continue
            x, y = pos
            neighbor_count = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in our_positions:
                    neighbor_count += 1
            if neighbor_count <= 3:  # Corridors and junctions
                candidates.append(pos)

        best_pos = None
        best_blocked_food = 0
        best_blocks_capsule = False

        for candidate in candidates:
            reachable = bfs_reachable(candidate)
            reachable_food = food_set & reachable
            blocked_food = len(baseline_reachable_food) - len(reachable_food)

            if blocked_food <= 0:
                continue

            # Check if it blocks any capsules
            blocked_capsules = capsule_set - reachable
            blocks_capsule = len(blocked_capsules) > 0

            # Score: food blocked count, with capsule bonus
            score = blocked_food + (5 if blocks_capsule else 0)

            if score > best_blocked_food + (5 if best_blocks_capsule else 0):
                best_pos = candidate
                best_blocked_food = blocked_food
                best_blocks_capsule = blocks_capsule

        # Only use bottleneck if it blocks a significant portion of food (≥ 40%)
        if best_pos and best_blocked_food >= total_food * 0.4:
            self.bottleneck_pos = best_pos
            self.bottleneck_food_count = best_blocked_food
            self.bottleneck_covers_capsule = best_blocks_capsule

    def _compute_food_depths(self, walls, width, height, is_red, food_list, agent):
        """
        Compute round-trip depth for each enemy food position.
        Depth = distance from nearest boundary to food × 2 (enter + exit).
        Higher depth = riskier to pursue when ghosts are near.
        """
        for food in food_list:
            closest_bp = min(self.boundary_positions,
                             key=lambda bp: agent.get_maze_distance(food, bp))
            dist = agent.get_maze_distance(food, closest_bp)
            self.food_depths[food] = dist * 2  # Round-trip depth

    def get_food_depth(self, food_pos):
        """Get the round-trip depth for a food position. Returns 0 if unknown."""
        return self.food_depths.get(food_pos, 0)

    def update_particles(self, game_state, agent):
        """
        Update particle filters based on new observations.
        Called by each agent on their turn.
        """
        opponents = agent.get_opponents(game_state)
        my_pos = game_state.get_agent_position(agent.index)
        noisy_distances = game_state.get_agent_distances()

        if noisy_distances is None:
            return

        walls = game_state.get_walls()
        width = walls.width
        height = walls.height

        for opp in opponents:
            opp_state = game_state.get_agent_state(opp)
            opp_pos = game_state.get_agent_position(opp)

            # If we can directly observe the opponent, set all particles to that position
            if opp_pos is not None:
                self.particles[opp] = [opp_pos] * self.num_particles
                continue

            # If enemy was just eaten (e.g., we know they respawned), reset particles
            # We can detect this if the enemy was a pacman and is no longer
            # For now, we rely on noisy distance updates

            noisy_dist = noisy_distances[opp]

            # Transition model: move each particle to a legal neighboring position
            new_particles = []
            for p in self.particles[opp]:
                px, py = p
                # Possible next positions (including staying)
                successors = [(px, py)]
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = int(px + dx), int(py + dy)
                    if 0 <= nx < width and 0 <= ny < height and not walls[nx][ny]:
                        successors.append((nx, ny))
                new_particles.append(random.choice(successors))

            # Observation model: weight particles by noisy distance probability
            weights = []
            for p in new_particles:
                true_dist = manhattan_distance(my_pos, p)
                prob = game_state.get_distance_prob(true_dist, noisy_dist)
                weights.append(prob)

            total_weight = sum(weights)
            if total_weight == 0:
                # All particles have zero probability. Reinitialize uniformly.
                halfway = width // 2
                if agent.red:
                    enemy_positions = [pos for pos in self.legal_positions if pos[0] >= halfway]
                else:
                    enemy_positions = [pos for pos in self.legal_positions if pos[0] < halfway]
                self.particles[opp] = [random.choice(enemy_positions) for _ in range(self.num_particles)]
            else:
                # Resample based on weights
                normalized = [w / total_weight for w in weights]
                self.particles[opp] = random.choices(new_particles, weights=normalized, k=self.num_particles)

    def get_estimated_position(self, opp_index):
        """
        Returns the most likely position of an opponent based on particles.
        Uses the mode (most common position) of the particle distribution.
        """
        if opp_index not in self.particles or not self.particles[opp_index]:
            return None

        pos_count = {}
        for p in self.particles[opp_index]:
            pos_count[p] = pos_count.get(p, 0) + 1

        return max(pos_count, key=pos_count.get)

    def detect_food_eaten(self, game_state, agent):
        """
        Detect if any of our food has been eaten recently.
        Returns the position(s) where food was eaten.
        """
        current_food = agent.get_food_you_are_defending(game_state).as_list()
        eaten = []
        if self.prev_food is not None:
            prev_set = set(self.prev_food)
            curr_set = set(current_food)
            eaten = list(prev_set - curr_set)
        self.prev_food = current_food
        return eaten

    # ==== Teammate coordination methods ====

    def set_food_target(self, agent_index, food_pos):
        """Register which food position this agent is targeting."""
        self.agent_targets[agent_index] = food_pos

    def get_teammate_target(self, agent_index):
        """Get the food target of the teammate (the other agent)."""
        for idx, target in self.agent_targets.items():
            if idx != agent_index:
                return target
        return None

    def set_chase_target(self, agent_index, invader_index):
        """Register which invader this agent is chasing."""
        self.chase_targets[agent_index] = invader_index

    def get_teammate_chase_target(self, agent_index):
        """Get the invader that the teammate is already chasing."""
        for idx, target in self.chase_targets.items():
            if idx != agent_index:
                return target
        return None

    def clear_chase_target(self, agent_index):
        """Clear chase target when no longer chasing."""
        if agent_index in self.chase_targets:
            del self.chase_targets[agent_index]


# Singleton shared state — one per team
_shared_states = {}


def get_shared_state(is_red):
    """Get or create the shared state for the given team."""
    if is_red not in _shared_states:
        _shared_states[is_red] = SharedState()
    return _shared_states[is_red]


##############
# Base Agent #
##############

class BaseAgent(CaptureAgent):
    """
    Base class with common utilities for both offensive and defensive agents.
    """

    def __init__(self, index, time_for_computing=0.9):
        super().__init__(index, time_for_computing)
        self.start = None
        self.shared = None
        self.position_history = []  # Track recent positions to avoid oscillation

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.shared = get_shared_state(self.red)
        self.shared.initialize(game_state, self)
        self.position_history = []

    def choose_action(self, game_state):
        """Override in subclasses."""
        raise NotImplementedError

    # =================== Utility methods ===================

    def get_successor(self, game_state, action):
        """Get the successor state after taking an action."""
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def get_safe_actions(self, game_state, actions):
        """
        Filter out actions that would walk us directly into a non-scared ghost,
        including predicting ghost next-step positions (blind spot detection).
        Returns safe actions (or all actions if none are safe, to avoid getting stuck).
        """
        my_state = game_state.get_agent_state(self.index)
        if not my_state.is_pacman:
            return actions  # We're a ghost ourselves, no danger

        # Get ALL non-scared enemy ghosts (both precise and estimated)
        opponents = self.get_opponents(game_state)
        danger_positions = []       # Current ghost positions
        predicted_positions = []    # Where ghosts could be next turn
        walls = game_state.get_walls()

        for opp in opponents:
            state = game_state.get_agent_state(opp)
            pos = game_state.get_agent_position(opp)
            if pos is not None and not state.is_pacman and state.scared_timer <= 1:
                danger_positions.append(pos)
                # Predict ghost's next possible positions (blind spot detection)
                gx, gy = int(pos[0]), int(pos[1])
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                        predicted_positions.append((nx, ny))

        if not danger_positions:
            return actions

        safe = []
        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            is_safe = True

            # Check against current ghost positions (distance ≤ 1)
            for gp in danger_positions:
                if self.get_maze_distance(new_pos, gp) <= 1:
                    is_safe = False
                    break

            # Check against predicted next-step ghost positions (exact match)
            if is_safe:
                for pp in predicted_positions:
                    if new_pos == pp:
                        # Only block if the ghost is close (distance 2) — true blind spot
                        for gp in danger_positions:
                            if self.get_maze_distance(new_pos, gp) <= 2:
                                is_safe = False
                                break
                    if not is_safe:
                        break

            if is_safe:
                safe.append(action)

        return safe if safe else actions  # Fallback to all if truly cornered

    def get_enemies_info(self, game_state):
        """
        Returns a list of (index, state, position_or_estimate) for each opponent.
        position_or_estimate: exact position if observable, else particle filter estimate.
        """
        opponents = self.get_opponents(game_state)
        result = []
        for opp in opponents:
            state = game_state.get_agent_state(opp)
            pos = game_state.get_agent_position(opp)
            if pos is None:
                pos = self.shared.get_estimated_position(opp)
            result.append((opp, state, pos))
        return result

    def get_ghost_enemies(self, game_state):
        """
        Returns list of (index, state, position) for opponent ghosts that are NOT scared
        and could threaten us.
        """
        enemies = self.get_enemies_info(game_state)
        ghosts = []
        for idx, state, pos in enemies:
            if pos is not None and not state.is_pacman and state.scared_timer <= 2:
                ghosts.append((idx, state, pos))
        return ghosts

    def get_precise_ghost_enemies(self, game_state):
        """
        Returns list of (index, state, position) for opponent ghosts that we can
        DIRECTLY OBSERVE (within 5 Manhattan distance). These are reliable.
        """
        opponents = self.get_opponents(game_state)
        ghosts = []
        for opp in opponents:
            state = game_state.get_agent_state(opp)
            pos = game_state.get_agent_position(opp)  # None if not observable
            if pos is not None and not state.is_pacman and state.scared_timer <= 2:
                ghosts.append((opp, state, pos))
        return ghosts

    def get_estimated_ghost_enemies(self, game_state):
        """
        Returns list of (index, state, estimated_pos) for opponent ghosts that
        we cannot directly observe but estimate via particle filter.
        """
        opponents = self.get_opponents(game_state)
        ghosts = []
        for opp in opponents:
            state = game_state.get_agent_state(opp)
            exact_pos = game_state.get_agent_position(opp)
            if exact_pos is None and not state.is_pacman and state.scared_timer <= 2:
                est_pos = self.shared.get_estimated_position(opp)
                if est_pos is not None:
                    ghosts.append((opp, state, est_pos))
        return ghosts

    def get_invaders(self, game_state):
        """
        Returns list of (index, state, position) for opponents that are currently Pacman
        (invading our side).
        """
        enemies = self.get_enemies_info(game_state)
        invaders = []
        for idx, state, pos in enemies:
            if state.is_pacman and pos is not None:
                invaders.append((idx, state, pos))
        return invaders

    def get_closest_boundary_pos(self, pos):
        """Returns the closest boundary position to the given position."""
        if not self.shared.boundary_positions:
            return self.start
        return min(self.shared.boundary_positions,
                   key=lambda bp: self.get_maze_distance(pos, bp))

    def get_safe_boundary_pos(self, game_state, pos):
        """
        Returns a boundary position that balances closeness and safety from ghosts.
        Prefers crossing points that are far from enemy defenders.
        """
        if not self.shared.boundary_positions:
            return self.start

        # Get all known ghost positions (precise + estimated)
        ghost_positions = []
        for opp in self.get_opponents(game_state):
            state = game_state.get_agent_state(opp)
            opp_pos = game_state.get_agent_position(opp)
            if opp_pos is None:
                opp_pos = self.shared.get_estimated_position(opp)
            if opp_pos is not None and not state.is_pacman and state.scared_timer <= 2:
                ghost_positions.append(opp_pos)

        if not ghost_positions:
            # No ghost threat, just pick closest
            return self.get_closest_boundary_pos(pos)

        best_bp = None
        best_score = -float('inf')

        for bp in self.shared.boundary_positions:
            my_dist = self.get_maze_distance(pos, bp)
            min_ghost_dist = min(self.get_maze_distance(bp, gp) for gp in ghost_positions)

            # Score: prefer close to us AND far from ghosts
            score = -my_dist * 2 + min_ghost_dist * 3

            if score > best_score:
                best_score = score
                best_bp = bp

        return best_bp if best_bp else self.get_closest_boundary_pos(pos)

    def get_distance_to_home(self, game_state):
        """Returns the maze distance from current position to the closest border position."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        return self.get_maze_distance(my_pos, self.get_closest_boundary_pos(my_pos))

    def is_in_dead_end(self, pos):
        """Check if a position is in a dead-end."""
        return pos in self.shared.dead_ends

    def a_star_path(self, game_state, start, goal, avoid_positions=None):
        """
        A* search from start to goal, optionally avoiding certain positions.
        Returns list of positions forming the path, or empty list if no path.
        """
        if start == goal:
            return [start]

        walls = game_state.get_walls()
        avoid = set(avoid_positions) if avoid_positions else set()

        frontier = util.PriorityQueue()
        frontier.push((start, [start]), 0)
        explored = set()

        while not frontier.isEmpty():
            current, path = frontier.pop()

            if current == goal:
                return path

            if current in explored:
                continue
            explored.add(current)

            x, y = int(current[0]), int(current[1])
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                if (0 <= nx < walls.width and 0 <= ny < walls.height
                        and not walls[nx][ny]
                        and next_pos not in explored
                        and next_pos not in avoid):
                    new_path = path + [next_pos]
                    g = len(new_path)
                    h = manhattan_distance(next_pos, goal)
                    frontier.push((next_pos, new_path), g + h)

        return []  # No path found

    def get_time_remaining(self, game_state):
        """Get the number of moves remaining in the game."""
        return game_state.data.timeleft // game_state.get_num_agents()

    def are_enemies_scared(self, game_state):
        """
        Check if any enemy ghost is currently in scared state.
        Returns (is_scared, min_scared_timer) tuple.
        """
        opponents = self.get_opponents(game_state)
        max_timer = 0
        for opp in opponents:
            state = game_state.get_agent_state(opp)
            if not state.is_pacman and state.scared_timer > 0:
                max_timer = max(max_timer, state.scared_timer)
        return max_timer > 3, max_timer

    def get_scared_ghosts(self, game_state):
        """
        Returns list of (index, state, position) for enemy ghosts that ARE scared.
        Only includes those we can see or estimate.
        """
        enemies = self.get_enemies_info(game_state)
        scared = []
        for idx, state, pos in enemies:
            if pos is not None and not state.is_pacman and state.scared_timer > 2:
                scared.append((idx, state, pos))
        return scared

    def get_border_food(self, game_state, max_depth=5):
        """
        Returns food positions that are close to the border (within max_depth maze distance).
        These are 'quick raid' targets for the defensive agent.
        """
        food_list = self.get_food(game_state).as_list()
        border_food = []
        for food in food_list:
            closest_bp = self.get_closest_boundary_pos(food)
            dist = self.get_maze_distance(food, closest_bp)
            if dist <= max_depth:
                border_food.append((food, dist))
        border_food.sort(key=lambda x: x[1])
        return [f for f, d in border_food]

    def get_deep_food(self, game_state, min_depth=4):
        """
        Returns food positions that are far from the border (deeper in enemy territory).
        These are targets for the main offensive agent.
        """
        food_list = self.get_food(game_state).as_list()
        deep_food = []
        for food in food_list:
            closest_bp = self.get_closest_boundary_pos(food)
            dist = self.get_maze_distance(food, closest_bp)
            if dist >= min_depth:
                deep_food.append(food)
        # If no deep food, return all food
        return deep_food if deep_food else food_list


############################
# Smart Offensive Agent    #
############################

class SmartOffensiveAgent(BaseAgent):
    """
    An offensive agent that enters enemy territory to eat food and return home safely.

    Behavior priorities:
    1. Emergency escape from nearby ghosts
    2. Eat power capsule if threatened and capsule nearby
    3. Return home with carried food
    4. Eat food greedily (preferring safe, close food)
    5. Explore towards food-rich areas
    """

    def choose_action(self, game_state):
        # Update particle filters
        self.shared.update_particles(game_state, self)

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Track position history for anti-oscillation
        self.position_history.append(my_pos)
        if len(self.position_history) > 6:
            self.position_history = self.position_history[-6:]

        actions = game_state.get_legal_actions(self.index)

        # Remove STOP action to keep the agent moving
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        if not actions:
            return Directions.STOP

        # Get info
        carrying = my_state.num_carrying
        food_list = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        score = self.get_score(game_state)
        time_left = self.get_time_remaining(game_state)
        is_pacman = my_state.is_pacman

        # Check if enemies are scared (capsule effect)
        enemies_scared, scared_timer = self.are_enemies_scared(game_state)

        # ==== CAPSULE MODE: enemies are scared, be aggressive! ====
        if enemies_scared and scared_timer > 5:
            return self._capsule_aggro_action(game_state, actions, my_pos,
                                               food_list, carrying, time_left, scared_timer)

        # Separate precisely observed ghosts from estimated ones
        precise_ghosts = self.get_precise_ghost_enemies(game_state)
        estimated_ghosts = self.get_estimated_ghost_enemies(game_state)

        # Calculate distances for precisely observed ghosts
        precise_ghost_dists = []
        for idx, state, pos in precise_ghosts:
            dist = self.get_maze_distance(my_pos, pos)
            precise_ghost_dists.append((dist, idx, state, pos))
        precise_ghost_dists.sort()

        # Calculate distances for estimated ghosts (lower reliability)
        estimated_ghost_dists = []
        for idx, state, pos in estimated_ghosts:
            dist = self.get_maze_distance(my_pos, pos)
            estimated_ghost_dists.append((dist, idx, state, pos))
        estimated_ghost_dists.sort()

        # Combined for functions that need all ghosts
        all_ghost_dists = sorted(precise_ghost_dists + estimated_ghost_dists)

        nearest_precise_dist = precise_ghost_dists[0][0] if precise_ghost_dists else 999
        nearest_any_dist = all_ghost_dists[0][0] if all_ghost_dists else 999

        # Distance to home (use safe crossing point away from ghosts)
        home_pos = self.get_safe_boundary_pos(game_state, my_pos)
        home_dist = self.get_maze_distance(my_pos, home_pos)

        # ==== Priority 1: Emergency escape — ONLY for precisely observed ghosts ====
        if is_pacman and precise_ghost_dists and nearest_precise_dist <= 3:
            return self._escape_action(game_state, actions, my_pos,
                                        precise_ghost_dists, capsules, food_list)

        # ==== Priority 2: Return home with food ====
        # Distance-based return: farther from home = return with fewer food
        should_return = False
        if carrying > 0 and len(food_list) <= 2:
            should_return = True
        elif carrying > 0 and time_left < home_dist + 10:
            should_return = True
        elif carrying >= 4 and home_dist >= 15:
            should_return = True
        elif carrying >= 5 and home_dist >= 10:
            should_return = True
        elif carrying >= 7 and home_dist >= 5:
            should_return = True
        elif carrying >= 9:
            should_return = True
        elif carrying >= 4 and nearest_precise_dist <= 5:
            should_return = True

        if should_return:
            return self._return_home_action(game_state, actions, my_pos, all_ghost_dists)

        # ==== Priority 3: Eat food (prefer deeper food, leave border food for defender) ====
        if food_list:
            # Prefer deep food but fall back to all food
            target_food = self.get_deep_food(game_state, min_depth=4)
            return self._eat_food_action(game_state, actions, my_pos, target_food,
                                          precise_ghost_dists, estimated_ghost_dists,
                                          capsules, carrying)

        # ==== Priority 4: Return home if no food available ====
        return self._return_home_action(game_state, actions, my_pos, all_ghost_dists)

    def _capsule_aggro_action(self, game_state, actions, my_pos,
                              food_list, carrying, time_left, scared_timer):
        """
        Aggressive mode when enemies are scared from our capsule.
        Eat food boldly, and chase scared ghosts. Still avoids non-scared ghost collision.
        """
        scared_ghosts = self.get_scared_ghosts(game_state)
        home_pos = self.get_closest_boundary_pos(my_pos)
        home_dist = self.get_maze_distance(my_pos, home_pos)

        # If scared timer is running low and carrying food, return home
        if carrying > 0 and (scared_timer <= 8 or time_left < home_dist + 5):
            safe_actions = self.get_safe_actions(game_state, actions)
            best_action = None
            best_dist = 999
            for action in safe_actions:
                successor = self.get_successor(game_state, action)
                new_pos = successor.get_agent_state(self.index).get_position()
                d = self.get_maze_distance(new_pos, home_pos)
                if d < best_dist:
                    best_dist = d
                    best_action = action
            return best_action if best_action else random.choice(actions)

        # Use safe actions to avoid walking into non-scared ghosts
        safe_actions = self.get_safe_actions(game_state, actions)

        best_action = None
        best_score = -float('inf')

        for action in safe_actions:
            successor = self.get_successor(game_state, action)
            new_state = successor.get_agent_state(self.index)
            new_pos = new_state.get_position()

            score = 0

            # Eat food aggressively
            if food_list:
                nearest_food_dist = min(self.get_maze_distance(new_pos, f) for f in food_list)
                score -= nearest_food_dist * 2

            # Bonus for eating food
            if new_state.num_carrying > carrying:
                score += 100

            # Chase scared ghosts to eat them (bonus!)
            for _, _, gpos in scared_ghosts:
                ghost_dist = self.get_maze_distance(new_pos, gpos)
                if ghost_dist <= 5:
                    score += max(0, 50 - ghost_dist * 10)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action else random.choice(actions)

    def _escape_action(self, game_state, actions, my_pos, ghost_distances, capsules, food_list=None):
        """Find the best escape action when threatened by precisely observed ghosts."""
        nearest_ghost_dist = ghost_distances[0][0] if ghost_distances else 999
        nearest_ghost_pos = ghost_distances[0][3] if ghost_distances else None

        # Counter-attack: if we're closer to a capsule than the ghost is, rush capsule!
        if capsules:
            nearest_capsule = min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
            my_capsule_dist = self.get_maze_distance(my_pos, nearest_capsule)
            # Check if ghost is farther from capsule than we are (safe to reach first)
            ghost_capsule_dist = 999
            if nearest_ghost_pos is not None:
                ghost_capsule_dist = self.get_maze_distance(nearest_ghost_pos, nearest_capsule)

            # Go for capsule if: we can reach it before ghost, OR it's very close
            if my_capsule_dist < ghost_capsule_dist or my_capsule_dist <= 2:
                best_action = None
                best_dist = 999
                for action in actions:
                    successor = self.get_successor(game_state, action)
                    new_pos = successor.get_agent_state(self.index).get_position()
                    d = self.get_maze_distance(new_pos, nearest_capsule)
                    if d < best_dist:
                        best_dist = d
                        best_action = action
                if best_action:
                    return best_action

        # Build food set for grab-on-escape check
        food_set = set(food_list) if food_list else set()

        ghost_positions = [pos for _, _, _, pos in ghost_distances if pos is not None]

        best_action = None
        best_score = -float('inf')
        carrying = game_state.get_agent_state(self.index).num_carrying

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_state = successor.get_agent_state(self.index)
            new_pos = new_state.get_position()

            # Check if we land on a ghost (that would kill us)
            ghost_collision = False
            for gp in ghost_positions:
                if self.get_maze_distance(new_pos, gp) <= 1:
                    ghost_collision = True
                    break
            if ghost_collision:
                continue

            # Evaluate: maximize distance from ghosts + minimize distance to home
            min_ghost_dist = min(self.get_maze_distance(new_pos, gp) for gp in ghost_positions) if ghost_positions else 999
            home_pos = self.get_closest_boundary_pos(new_pos)
            home_dist = self.get_maze_distance(new_pos, home_pos)

            dead_end_penalty = -20 if self.is_in_dead_end(new_pos) else 0

            # Bonus: grab food on the way if it doesn't sacrifice distance (safe to eat)
            food_grab_bonus = 0
            if new_pos in food_set and min_ghost_dist >= 2:
                food_grab_bonus = 30

            # Huge priority on maximizing distance from ghost during emergency escape
            score = (min_ghost_dist * 100) - (home_dist * 2) + dead_end_penalty + food_grab_bonus

            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action else random.choice(actions)

    def _return_home_action(self, game_state, actions, my_pos, ghost_distances):
        """Find the best action to return home safely."""
        home_pos = self.get_safe_boundary_pos(game_state, my_pos)

        ghost_positions = [pos for _, _, _, pos in ghost_distances[:2] if pos is not None]

        best_action = None
        best_score = -float('inf')

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()

            home_dist = self.get_maze_distance(new_pos, home_pos)

            # Ghost avoidance
            min_ghost_dist = 999
            for gp in ghost_positions:
                d = self.get_maze_distance(new_pos, gp)
                min_ghost_dist = min(min_ghost_dist, d)

            ghost_bonus = 0
            if min_ghost_dist <= 1:
                ghost_bonus = -1000
            elif min_ghost_dist <= 3:
                ghost_bonus = -50

            dead_end_penalty = -30 if self.is_in_dead_end(new_pos) else 0

            score = -home_dist * 10 + ghost_bonus + dead_end_penalty

            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action else random.choice(actions)

    def _eat_food_action(self, game_state, actions, my_pos, food_list,
                          precise_ghost_dists, estimated_ghost_dists,
                          capsules, carrying):
        """Choose the best food-eating action, distinguishing precise vs estimated ghosts."""
        precise_positions = [pos for _, _, _, pos in precise_ghost_dists[:2] if pos is not None]
        estimated_positions = [pos for _, _, _, pos in estimated_ghost_dists[:2] if pos is not None]

        # Teammate coordination: get teammate's food target
        teammate_target = self.shared.get_teammate_target(self.index)

        best_action = None
        best_score = -float('inf')
        best_food_target = None  # Track which food we're heading towards

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_state = successor.get_agent_state(self.index)
            new_pos = new_state.get_position()

            # Feature 1: Distance to nearest food
            if food_list:
                nearest_food_dist = min(self.get_maze_distance(new_pos, f) for f in food_list)
                # Track which food is nearest for coordination
                nearest_food = min(food_list, key=lambda f: self.get_maze_distance(new_pos, f))
            else:
                nearest_food_dist = 0
                nearest_food = None

            # Feature 2: Food eaten (new_carrying > carrying means we just ate)
            food_eaten = 1 if new_state.num_carrying > carrying else 0

            # Feature 3a: Precise ghost threat (heavy penalty — these are real)
            min_precise_dist = 999
            for gp in precise_positions:
                d = self.get_maze_distance(new_pos, gp)
                min_precise_dist = min(min_precise_dist, d)

            precise_ghost_score = 0
            if min_precise_dist <= 1:
                precise_ghost_score = -500
            elif min_precise_dist <= 3:
                precise_ghost_score = -30
            elif min_precise_dist <= 5:
                precise_ghost_score = -5

            # Feature 3b: Estimated ghost threat (light penalty — may be inaccurate)
            min_estimated_dist = 999
            for gp in estimated_positions:
                d = self.get_maze_distance(new_pos, gp)
                min_estimated_dist = min(min_estimated_dist, d)

            estimated_ghost_score = 0
            if min_estimated_dist <= 2:
                estimated_ghost_score = -10
            elif min_estimated_dist <= 4:
                estimated_ghost_score = -3

            # Use closest ghost of either type for dead-end / capsule logic
            min_any_ghost_dist = min(min_precise_dist, min_estimated_dist)

            # Feature 4: Dead-end avoidance — only heavy penalty for precisely seen ghosts
            dead_end_penalty = 0
            if self.is_in_dead_end(new_pos):
                if min_precise_dist <= 5:
                    dead_end_penalty = -80
                elif min_estimated_dist <= 4:
                    dead_end_penalty = -15
                else:
                    dead_end_penalty = -3

            # Feature 5: Capsule proximity
            capsule_bonus = 0
            if capsules and min_any_ghost_dist <= 6:
                nearest_capsule_dist = min(self.get_maze_distance(new_pos, c) for c in capsules)
                if nearest_capsule_dist <= 3:
                    capsule_bonus = 30

            # Feature 6: Distance to home when carrying a lot
            home_pos = self.get_closest_boundary_pos(new_pos)
            home_factor = 0
            if carrying >= 3:
                home_factor = -self.get_maze_distance(new_pos, home_pos) * 0.5

            # Feature 7: Teammate coordination — avoid going for same food
            teammate_penalty = 0
            if teammate_target is not None and nearest_food is not None:
                dist_to_teammate_food = self.get_maze_distance(nearest_food, teammate_target)
                if dist_to_teammate_food <= 3:
                    teammate_penalty = -40

            # Feature 8: Food depth penalty — avoid very deep food when ghost is dangerously close
            depth_penalty = 0
            if nearest_food is not None and min_precise_dist <= 3:
                food_depth = self.shared.get_food_depth(nearest_food)
                if food_depth > 16:
                    depth_penalty = -(food_depth - 16) * 1.5

            score = (food_eaten * 100
                     - nearest_food_dist * 2
                     + precise_ghost_score
                     + estimated_ghost_score
                     + dead_end_penalty
                     + capsule_bonus
                     + home_factor
                     + teammate_penalty
                     + depth_penalty)

            # Anti-oscillation: penalize revisiting recent positions
            if new_pos in self.position_history[-4:]:
                score -= 50

            if score > best_score:
                best_score = score
                best_action = action
                best_food_target = nearest_food

        # Register our food target for teammate coordination
        if best_food_target is not None:
            self.shared.set_food_target(self.index, best_food_target)

        return best_action if best_action else random.choice(actions)


############################
# Smart Defensive Agent    #
############################

class SmartDefensiveAgent(BaseAgent):
    """
    A defensive agent that patrols the home side and chases invaders.

    Behavior priorities:
    1. Chase visible/detected invaders
    2. Respond to food being eaten (go to eaten food location)
    3. Patrol boundary near high-value food areas
    4. Handle scared state carefully
    """

    def __init__(self, index, time_for_computing=0.9):
        super().__init__(index, time_for_computing)
        self.patrol_index = 0
        self.last_seen_invader_pos = None

    def choose_action(self, game_state):
        # Update particle filters
        self.shared.update_particles(game_state, self)

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        actions = game_state.get_legal_actions(self.index)

        # Remove STOP action
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        if not actions:
            return Directions.STOP

        scared_timer = my_state.scared_timer
        invaders = self.get_invaders(game_state)
        score = self.get_score(game_state)
        carrying = my_state.num_carrying
        is_pacman = my_state.is_pacman

        # Check if enemies are scared
        enemies_scared, enemy_scared_timer = self.are_enemies_scared(game_state)

        # ==== If carrying food and on enemy side, return home first ====
        if is_pacman and carrying > 0:
            home_pos = self.get_closest_boundary_pos(my_pos)
            home_dist = self.get_maze_distance(my_pos, home_pos)
            time_left = self.get_time_remaining(game_state)

            # Check for nearby precise ghosts
            precise_ghosts = self.get_precise_ghost_enemies(game_state)
            nearest_ghost = 999
            for _, _, gpos in precise_ghosts:
                nearest_ghost = min(nearest_ghost, self.get_maze_distance(my_pos, gpos))

            if carrying >= 3 or nearest_ghost <= 4 or time_left < home_dist + 8:
                return self._quick_return_home(game_state, actions, my_pos)

        # ==== Dynamic role: switch to offense if losing badly ====
        if score <= -8 and not invaders:
            return self._offensive_fallback(game_state, actions, my_pos)

        # ==== Priority 1: Chase invaders ====
        if invaders:
            if scared_timer > 2:
                return self._scared_defense(game_state, actions, my_pos, invaders)
            else:
                return self._chase_invader(game_state, actions, my_pos, invaders)
        else:
            # No invaders — clear chase target
            self.shared.clear_chase_target(self.index)

        # ==== Priority 2: Respond to food being eaten ====
        eaten_food = self.shared.detect_food_eaten(game_state, self)
        if eaten_food:
            target = min(eaten_food, key=lambda f: self.get_maze_distance(my_pos, f))
            self.last_seen_invader_pos = target
            return self._move_towards(game_state, actions, my_pos, target)

        # ==== Priority 3: Move towards estimated enemy on or near our side ====
        for opp in self.get_opponents(game_state):
            est_pos = self.shared.get_estimated_position(opp)
            if est_pos is not None:
                walls = game_state.get_walls()
                halfway = walls.width // 2
                on_our_side = (est_pos[0] < halfway) if self.red else (est_pos[0] >= halfway)
                # Also intercept if enemy is close to border (within 3 squares)
                near_border = abs(est_pos[0] - halfway) <= 3
                if on_our_side or near_border:
                    # If near border, move to closest boundary position to intercept
                    if not on_our_side and near_border:
                        target = self.get_closest_boundary_pos(est_pos)
                    else:
                        target = est_pos
                    return self._move_towards(game_state, actions, my_pos, target)

        # ==== Priority 4: Camp at bottleneck if one exists ====
        if self.shared.bottleneck_pos is not None and not is_pacman:
            bottleneck = self.shared.bottleneck_pos
            if my_pos == bottleneck:
                # We're at the bottleneck — stay here!
                return Directions.STOP
            else:
                # Move towards the bottleneck
                return self._move_towards(game_state, actions, my_pos, bottleneck)

        # ==== Priority 5: Quick raid — grab nearby border food when safe ====
        # Only raid if: no invaders, not scared, and there's border food
        if scared_timer <= 0:
            border_food = self.get_border_food(game_state, max_depth=5)
            if border_food:
                # Check if it's safe (no precise ghosts within 5)
                precise_ghosts = self.get_precise_ghost_enemies(game_state)
                safe = True
                for _, _, gpos in precise_ghosts:
                    if self.get_maze_distance(my_pos, gpos) <= 5:
                        safe = False
                        break

                # Also check if enemies are scared (extra safe to raid)
                if safe or enemies_scared:
                    return self._quick_raid(game_state, actions, my_pos, border_food,
                                             enemies_scared)

        # ==== Priority 6: Patrol the boundary ====
        return self._patrol(game_state, actions, my_pos)

    def _chase_invader(self, game_state, actions, my_pos, invaders):
        """Chase invaders with teammate coordination."""
        # Determine which invader to chase, coordinating with teammate
        teammate_chase = self.shared.get_teammate_chase_target(self.index)

        if len(invaders) >= 2 and teammate_chase is not None:
            # Two invaders: chase the one the teammate ISN'T chasing
            my_invader = None
            for idx, state, pos in invaders:
                if idx != teammate_chase:
                    my_invader = (idx, state, pos)
                    break
            if my_invader is None:
                my_invader = min(invaders, key=lambda inv: self.get_maze_distance(my_pos, inv[2]))
        else:
            # One invader or no teammate info: chase the closest
            my_invader = min(invaders, key=lambda inv: self.get_maze_distance(my_pos, inv[2]))

        inv_idx, inv_state, target = my_invader
        self.shared.set_chase_target(self.index, inv_idx)
        self.last_seen_invader_pos = target

        # If teammate is also chasing the same invader (1 invader case),
        # try to intercept the invader's escape route instead
        if len(invaders) == 1 and teammate_chase == inv_idx:
            # We're the second chaser — try to cut off escape to border
            inv_home = self.get_closest_boundary_pos(target)
            if self.get_maze_distance(my_pos, inv_home) < self.get_maze_distance(my_pos, target):
                # We're closer to their escape route - go block it
                target = inv_home

        best_action = None
        best_dist = 999

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            dist = self.get_maze_distance(new_pos, target)

            if dist < best_dist:
                best_dist = dist
                best_action = action

        return best_action if best_action else random.choice(actions)

    def _scared_defense(self, game_state, actions, my_pos, invaders):
        """
        When scared, maintain distance from invaders but stay close enough to track.
        """
        closest_invader = min(invaders, key=lambda inv: self.get_maze_distance(my_pos, inv[2]))
        invader_pos = closest_invader[2]
        self.last_seen_invader_pos = invader_pos
        invader_dist = self.get_maze_distance(my_pos, invader_pos)

        best_action = None
        best_score = -float('inf')

        target_distance = 4  # Stay about 4 squares away

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            new_dist = self.get_maze_distance(new_pos, invader_pos)

            # Prefer distance close to target_distance
            score = -abs(new_dist - target_distance) * 10

            # Avoid going to enemy side (stay as ghost)
            new_state = successor.get_agent_state(self.index)
            if new_state.is_pacman:
                score -= 100

            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action else random.choice(actions)

    def _move_towards(self, game_state, actions, my_pos, target):
        """Move towards a target position."""
        best_action = None
        best_dist = 999

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()

            # Avoid becoming Pacman (leaving our side)
            new_state = successor.get_agent_state(self.index)
            if new_state.is_pacman:
                dist = self.get_maze_distance(new_pos, target) + 100
            else:
                dist = self.get_maze_distance(new_pos, target)

            if dist < best_dist:
                best_dist = dist
                best_action = action

        return best_action if best_action else random.choice(actions)

    def _patrol(self, game_state, actions, my_pos):
        """
        Patrol the boundary positions, prioritizing areas near food clusters.
        """
        defending_food = self.get_food_you_are_defending(game_state).as_list()

        if defending_food and self.shared.boundary_positions:
            scored_positions = []
            for bp in self.shared.boundary_positions:
                food_score = sum(1 for f in defending_food
                                 if manhattan_distance(bp, f) <= 5)
                scored_positions.append((bp, food_score))

            scored_positions.sort(key=lambda x: -x[1])
            patrol_targets = [sp[0] for sp in scored_positions[:3]]

            if patrol_targets:
                target = patrol_targets[self.patrol_index % len(patrol_targets)]
                if self.get_maze_distance(my_pos, target) <= 2:
                    self.patrol_index += 1
                    target = patrol_targets[self.patrol_index % len(patrol_targets)]

                return self._move_towards(game_state, actions, my_pos, target)

        if self.shared.boundary_positions:
            mid = self.shared.boundary_positions[len(self.shared.boundary_positions) // 2]
            return self._move_towards(game_state, actions, my_pos, mid)

        return random.choice(actions)

    def _quick_raid(self, game_state, actions, my_pos, border_food, enemies_scared):
        """
        Quick raid: cross border, eat 1-2 nearby food, and return immediately.
        Only targets food close to the border for minimal risk.
        """
        # Pick the closest border food as target
        target_food = min(border_food, key=lambda f: self.get_maze_distance(my_pos, f))

        # Check ghost threats (only precise)
        precise_ghosts = self.get_precise_ghost_enemies(game_state)

        best_action = None
        best_score = -float('inf')
        carrying = game_state.get_agent_state(self.index).num_carrying

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_state = successor.get_agent_state(self.index)
            new_pos = new_state.get_position()

            # Distance to target food
            food_dist = self.get_maze_distance(new_pos, target_food)

            # Bonus for eating food
            food_eaten = 50 if new_state.num_carrying > carrying else 0

            # Ghost avoidance (only for precise, and skip if enemies scared)
            ghost_penalty = 0
            if not enemies_scared:
                for _, _, gpos in precise_ghosts:
                    gdist = self.get_maze_distance(new_pos, gpos)
                    if gdist <= 1:
                        ghost_penalty = -500
                        break
                    elif gdist <= 3:
                        ghost_penalty = min(ghost_penalty, -30)

            score = -food_dist * 3 + food_eaten + ghost_penalty

            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action else random.choice(actions)

    def _quick_return_home(self, game_state, actions, my_pos):
        """
        Quickly return to home side to deposit carried food.
        """
        home_pos = self.get_closest_boundary_pos(my_pos)
        precise_ghosts = self.get_precise_ghost_enemies(game_state)

        best_action = None
        best_score = -float('inf')

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            home_dist = self.get_maze_distance(new_pos, home_pos)

            ghost_penalty = 0
            for _, _, gpos in precise_ghosts:
                gdist = self.get_maze_distance(new_pos, gpos)
                if gdist <= 1:
                    ghost_penalty = -500
                    break
                elif gdist <= 2:
                    ghost_penalty = min(ghost_penalty, -50)

            score = -home_dist * 10 + ghost_penalty

            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action else random.choice(actions)

    def _offensive_fallback(self, game_state, actions, my_pos):
        """
        When losing badly and no invaders, switch to offensive behavior.
        Uses a simplified version of the offensive agent strategy.
        """
        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return random.choice(actions)

        best_action = None
        best_score = -float('inf')
        carrying = game_state.get_agent_state(self.index).num_carrying
        ghosts = self.get_ghost_enemies(game_state)
        ghost_positions = [pos for _, _, pos in ghosts if pos is not None]

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()

            # Distance to nearest food
            nearest_food_dist = min(self.get_maze_distance(new_pos, f) for f in food_list)

            # Ghost avoidance
            min_ghost_dist = 999
            for gp in ghost_positions:
                d = self.get_maze_distance(new_pos, gp)
                min_ghost_dist = min(min_ghost_dist, d)

            ghost_penalty = 0
            if min_ghost_dist <= 1:
                ghost_penalty = -500
            elif min_ghost_dist <= 3:
                ghost_penalty = -30

            # Return home if carrying food
            home_factor = 0
            if carrying >= 3:
                home_pos = self.get_closest_boundary_pos(new_pos)
                home_factor = -self.get_maze_distance(new_pos, home_pos) * 5

            score = -nearest_food_dist * 2 + ghost_penalty + home_factor

            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action else random.choice(actions)