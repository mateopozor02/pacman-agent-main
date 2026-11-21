# my_team.py
# MIIS Autonomous Systems 2026 — Pacman Capture-the-Flag Contest
#
# Strategy overview:
#   - OffensiveAgent: A* food collection with ghost-avoidance cost, capsule pursuit,
#     dynamic retreat threshold, and scared-ghost hunting.
#   - DefensiveAgent: Visible-invader chasing, noisy-distance-guided interception,
#     and border-patrol when the home side is clear.

import heapq
import random

import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point



# Team factory


def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent',
                num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]



# Shared base


class BaseAgent(CaptureAgent):
    """Shared utilities for both offensive and defensive agents."""

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        self.walls = game_state.get_walls()
        self.width = self.walls.width
        self.height = self.walls.height
        mid_x = self.width // 2

        # Border column on our own side (last column before enemy territory)
        self.border_x = mid_x - 1 if self.red else mid_x
        self.border_positions = [
            (self.border_x, y)
            for y in range(self.height)
            if not self.walls[self.border_x][y]
        ]

    # Successor helper (handles half-step positions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    # A* with optional ghost-penalty cost

    def a_star(self, game_state, start, goal, ghost_positions=None):
        """
        Return the first action of the optimal path from *start* to *goal*.
        If ghost_positions is provided, cells near ghosts receive extra cost
        so the agent naturally routes around them.
        """
        if start == goal:
            return Directions.STOP

        def ghost_penalty(pos):
            if not ghost_positions:
                return 0
            md = min(self.get_maze_distance(pos, gp) for gp in ghost_positions)
            if md == 0:
                return 500
            if md == 1:
                return 100
            if md == 2:
                return 20
            if md == 3:
                return 5
            return 0

        # heap entries: (f, g, pos, first_action)
        open_set = []
        h0 = self.get_maze_distance(start, goal)
        heapq.heappush(open_set, (h0, 0, start, None))
        best_g = {}  # pos -> best g seen

        while open_set:
            f, g, pos, first_action = heapq.heappop(open_set)

            if pos in best_g and best_g[pos] <= g:
                continue
            best_g[pos] = g

            if pos == goal:
                return first_action if first_action is not None else Directions.STOP

            x, y = int(pos[0]), int(pos[1])
            for action, (dx, dy) in [
                (Directions.NORTH, (0, 1)),
                (Directions.SOUTH, (0, -1)),
                (Directions.EAST,  (1, 0)),
                (Directions.WEST,  (-1, 0)),
            ]:
                nx, ny = x + dx, y + dy
                if self.walls[nx][ny]:
                    continue
                new_pos = (nx, ny)
                step_cost = 1 + ghost_penalty(new_pos)
                new_g = g + step_cost
                if new_pos in best_g and best_g[new_pos] <= new_g:
                    continue
                h = self.get_maze_distance(new_pos, goal)
                fa = first_action if first_action is not None else action
                heapq.heappush(open_set, (new_g + h, new_g, new_pos, fa))

        # Goal unreachable — pick a random non-stop legal action as fallback
        legal = [a for a in game_state.get_legal_actions(self.index)
                 if a != Directions.STOP]
        return random.choice(legal) if legal else Directions.STOP

    # Convenience accessors

    def my_pos(self, game_state):
        return game_state.get_agent_position(self.index)

    def enemy_ghosts(self, game_state):
        """Observable, non-scared enemy ghosts: list of (AgentState, pos)."""
        return [
            (game_state.get_agent_state(i), game_state.get_agent_position(i))
            for i in self.get_opponents(game_state)
            if not game_state.get_agent_state(i).is_pacman
            and game_state.get_agent_position(i) is not None
            and game_state.get_agent_state(i).scared_timer == 0
        ]

    def scared_ghosts(self, game_state):
        """Observable, scared enemy ghosts: list of (AgentState, pos)."""
        return [
            (game_state.get_agent_state(i), game_state.get_agent_position(i))
            for i in self.get_opponents(game_state)
            if not game_state.get_agent_state(i).is_pacman
            and game_state.get_agent_position(i) is not None
            and game_state.get_agent_state(i).scared_timer > 0
        ]

    def invaders(self, game_state):
        """Observable enemy pacmen: list of (AgentState, pos)."""
        return [
            (game_state.get_agent_state(i), game_state.get_agent_position(i))
            for i in self.get_opponents(game_state)
            if game_state.get_agent_state(i).is_pacman
            and game_state.get_agent_position(i) is not None
        ]

    def closest_border(self, pos):
        return min(self.border_positions,
                   key=lambda b: self.get_maze_distance(pos, b))


# Offensive agent

class OffensiveAgent(BaseAgent):
    """
    Collects food while avoiding ghosts.

    Decision hierarchy (highest priority first):
      1. Flee if a dangerous ghost is within DANGER_DIST.
         → prefer eating a capsule if it is closer than the ghost.
      2. Return home if carrying >= carry_limit pellets.
      3. Return home if ≤ 2 food dots remain.
      4. Chase a scared ghost (free kill + disruption).
      5. Eat closest capsule if a ghost is within CAPSULE_DIST.
      6. Collect food via ghost-aware A*.
    """

    DANGER_DIST = 5      # ghost closer than this triggers escape
    CAPSULE_DIST = 8     # ghost closer than this prompts capsule-hunt

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.carry_limit = 5  # adjusted dynamically

    def choose_action(self, game_state):
        pos = self.my_pos(game_state)
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying

        food_list = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        ghosts = self.enemy_ghosts(game_state)
        scared = self.scared_ghosts(game_state)
        ghost_positions = [gp for _, gp in ghosts]

        # Distances to dangerous ghosts
        ghost_dists = sorted(
            [(self.get_maze_distance(pos, gp), gp) for _, gp in ghosts]
        )
        min_ghost_dist = ghost_dists[0][0] if ghost_dists else 9999

        # Dynamic carry limit: carry more when winning, less when at risk
        score = self.get_score(game_state)
        self.carry_limit = max(3, 7 - max(0, score))

        # ── 1. Flee ───────────────────────────────────────────────────────
        if my_state.is_pacman and min_ghost_dist <= self.DANGER_DIST:
            # Capsule rescue: go for capsule only if it is reachable before ghost
            if capsules:
                best_cap = min(capsules, key=lambda c: self.get_maze_distance(pos, c))
                cap_dist = self.get_maze_distance(pos, best_cap)
                if cap_dist < min_ghost_dist - 1:
                    return self.a_star(game_state, pos, best_cap)

            # Rush the border, routing away from ghosts
            target = self.closest_border(pos)
            return self._flee_action(game_state, pos, ghost_dists, target)

        # ── 2. Return home when loaded ────────────────────────────────────
        if carrying >= self.carry_limit:
            target = self.closest_border(pos)
            return self.a_star(game_state, pos, target)

        # ── 3. Return home when nearly done ──────────────────────────────
        if len(food_list) <= 2:
            target = self.closest_border(pos)
            return self.a_star(game_state, pos, target)

        # ── 4. Chase scared ghosts ────────────────────────────────────────
        if scared:
            target_state, target_pos = min(
                scared, key=lambda sg: self.get_maze_distance(pos, sg[1])
            )
            # Only chase if the scared timer is worth the detour
            dist_to_scared = self.get_maze_distance(pos, target_pos)
            if target_state.scared_timer > dist_to_scared + 2:
                return self.a_star(game_state, pos, target_pos)

        # ── 5. Hunt capsule when ghost is nearby ──────────────────────────
        if min_ghost_dist <= self.CAPSULE_DIST and capsules:
            best_cap = min(capsules, key=lambda c: self.get_maze_distance(pos, c))
            return self.a_star(game_state, pos, best_cap,
                               ghost_positions=ghost_positions)

        # ── 6. Collect food (ghost-aware routing) ─────────────────────────
        best_food = min(food_list,
                        key=lambda f: self._food_priority(pos, f, ghost_positions))
        return self.a_star(game_state, pos, best_food,
                           ghost_positions=ghost_positions)

    def _food_priority(self, my_pos, food_pos, ghost_positions):
        """Lower is better. Penalise food near ghosts."""
        dist = self.get_maze_distance(my_pos, food_pos)
        penalty = 0
        for gp in ghost_positions:
            gd = self.get_maze_distance(food_pos, gp)
            if gd < 5:
                penalty += (5 - gd) * 8
        return dist + penalty

    def _flee_action(self, game_state, pos, ghost_dists, target):
        """Pick the legal action that maximises distance from the nearest ghost
        while still heading toward the border."""
        legal = [a for a in game_state.get_legal_actions(self.index)
                 if a != Directions.STOP]
        if not legal:
            return Directions.STOP

        best_action, best_score = Directions.STOP, -9999
        for action in legal:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            min_gd = min(
                (self.get_maze_distance(new_pos, gp) for _, gp in ghost_dists),
                default=9999
            )
            home_dist = self.get_maze_distance(new_pos, target)
            # Heavily weight ghost distance; mildly reward getting closer to home
            score = min_gd * 3 - home_dist
            if score > best_score:
                best_score = score
                best_action = action

        return best_action



# Defensive agent

class DefensiveAgent(BaseAgent):
    """
    Defends the home side.

    Decision hierarchy:
      1. If scared, stay away from invaders (don't get eaten).
      2. Chase visible invaders.
      3. Move toward the most likely invader position inferred from
         noisy distance readings.
      4. Patrol border positions when home side is quiet.
    """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self._patrol_points = self._build_patrol_points(game_state)
        self._patrol_idx = 0

    def _build_patrol_points(self, game_state):
        """Evenly-spaced patrol positions just inside our border."""
        px = self.border_x - 1 if self.red else self.border_x + 1
        # Clamp to valid range
        px = max(1, min(self.width - 2, px))
        step = max(1, self.height // 6)
        pts = [
            (px, y)
            for y in range(step, self.height - step, step)
            if not self.walls[px][y]
        ]
        # Fallback: use the border itself
        if not pts:
            pts = list(self.border_positions)
        return pts

    def choose_action(self, game_state):
        pos = self.my_pos(game_state)
        my_state = game_state.get_agent_state(self.index)
        legal = game_state.get_legal_actions(self.index)

        inv = self.invaders(game_state)

        # ── 0. Return to home side if we drifted over ─────────────────────
        if my_state.is_pacman:
            return self.a_star(game_state, pos, self.closest_border(pos))

        # ── 1. Scared: evade invaders ─────────────────────────────────────
        if my_state.scared_timer > 0 and inv:
            _, threat_pos = min(inv, key=lambda iv: self.get_maze_distance(pos, iv[1]))
            return self._move_away(pos, legal, game_state, threat_pos)

        # ── 2. Chase visible invader ──────────────────────────────────────
        if inv:
            _, target_pos = min(inv, key=lambda iv: self.get_maze_distance(pos, iv[1]))
            return self.a_star(game_state, pos, target_pos)

        # ── 3. Use noisy distances to guess invader location ──────────────
        noisy = game_state.get_agent_distances()
        opponents = self.get_opponents(game_state)
        # Find the opponent on our side with the smallest noisy distance
        best_target = None
        best_noisy = 9999
        for opp_idx in opponents:
            opp_state = game_state.get_agent_state(opp_idx)
            # If the noisy distance suggests they're close and their last known
            # position was on our side, move toward our food clusters
            nd = noisy[opp_idx]
            if nd < best_noisy:
                best_noisy = nd
                best_target = opp_idx

        if best_target is not None and best_noisy < 10:
            # Head toward the middle of our food to intercept
            food_defending = self.get_food_you_are_defending(game_state).as_list()
            if food_defending:
                # Go to the food cluster centroid closest to us
                centroid_x = int(sum(f[0] for f in food_defending) / len(food_defending))
                centroid_y = int(sum(f[1] for f in food_defending) / len(food_defending))
                # Find nearest non-wall cell to centroid
                candidates = [
                    f for f in food_defending
                    if abs(f[0] - centroid_x) < 4 and abs(f[1] - centroid_y) < 4
                ]
                if candidates:
                    intercept = min(candidates,
                                    key=lambda f: self.get_maze_distance(pos, f))
                    return self.a_star(game_state, pos, intercept)

        # ── 4. Patrol ─────────────────────────────────────────────────────
        patrol_target = self._patrol_points[
            self._patrol_idx % len(self._patrol_points)
        ]
        if pos == patrol_target or self.get_maze_distance(pos, patrol_target) < 2:
            self._patrol_idx += 1
            patrol_target = self._patrol_points[
                self._patrol_idx % len(self._patrol_points)
            ]
        return self.a_star(game_state, pos, patrol_target)

    def _move_away(self, pos, legal, game_state, threat_pos):
        """Move to maximise distance from threat_pos."""
        best_action, best_dist = Directions.STOP, -1
        for action in legal:
            if action == Directions.STOP:
                continue
            succ = self.get_successor(game_state, action)
            new_pos = succ.get_agent_state(self.index).get_position()
            d = self.get_maze_distance(new_pos, threat_pos)
            if d > best_dist:
                best_dist = d
                best_action = action
        return best_action
