# my_team.py
# Architecture:
#   BaseAgent      – belief tracking, A*, danger map, siege, smart-flee, shared utils
#   OffensiveAgent – cluster-based attack, dynamic return, opportunistic kill
#   DefensiveAgent – entry-guard patrol, dual-role chase, invader pursuit

import random
import time
from collections import deque
import contest.util as util
from contest.capture_agents import CaptureAgent


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', **kwargs):
    return [eval(first)(first_index), eval(second)(second_index)]


# ──────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────

# General (base values – runtime may use dynamic versions)
FLEE_THRESHOLD       = 5    # my scared_timer above which I flee
DANGER_WEIGHT        = 1000000  # A* danger cost multiplier
WIN_THRESHOLD        = 8    # hard score lead → attacker plays safe
LOSE_THRESHOLD       = -5   # hard score deficit → defender attacks
NOISE_RANGE          = 6    # noisy distance tolerance |true-noisy|

# Offensive
CLUSTER_RADIUS       = 4    # food within this maze-dist counts as same cluster
RETURN_SAFE_MARGIN   = 3    # base ghost safety buffer (dynamic version overrides)
OPPORTUNISTIC_KILL_DIST = 3 # base attacker kill dist (dynamic version overrides)
ENDGAME_THRESHOLD    = 60   # base endgame threshold (dynamic version overrides)
DETOUR_TOLERANCE     = 4    # extra steps allowed for easy food on the way home

# Siege
SIEGE_THRESHOLD      = 5    # base siege threshold (dynamic version overrides)

# Smart-flee
SCARED_SAFE_DIST     = 4    # minimum distance to keep from threats while scared
SCARED_RESUME_BUFFER = 2    # start repositioning this many steps before scare ends

# Capsule state constants
CAP_NORMAL      = 'normal'
CAP_WE_SCARED   = 'we_scared'
CAP_THEY_SCARED = 'they_scared'
CAP_BOTH_SCARED = 'both_scared'

# Map structure bonus weights (used in _action_guard_entries)
CHOKE_SELF_BONUS      = 3.0   # entry is itself a choke point
CHOKE_ADJ_BONUS       = 1.5   # entry is adjacent to a choke point
BOTTLENECK_BONUS      = 2.0   # entry guards a cluster bottleneck
ENEMY_CHOKE_PENALTY   = 60    # A* penalty for walking through enemy choke point
# Dead-end penalties (applied only to OffensiveAgent as pacman, via _apply_dead_end_penalty)
DEAD_END_BASE_WEIGHT  = 15    # base cost per depth unit in dead-end cells
DEAD_END_GHOST_SCALE  = 10    # ghost dist below this makes penalty grow linearly

# Capsule strategy
CAPSULE_THREAT_DIST   = 8    # invader within this dist of our capsule → active guard
CAPSULE_GUARD_BONUS   = 4.0  # entry-point score bonus for proximity to our capsule
CAPSULE_SAFE_MARGIN   = 2    # min (ghost_dist - my_dist) to pursue enemy capsule
CAPSULE_HARVEST_MIN   = 10   # min remaining scared steps after eating to bother


# ──────────────────────────────────────────────
# Base Agent
# ──────────────────────────────────────────────

class BaseAgent(CaptureAgent):

    # ── initialisation ────────────────────────────────────────────────────

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.start = game_state.get_agent_position(self.index)

        walls        = game_state.get_walls()
        self.width   = walls.width
        self.height  = walls.height
        self.mid_x   = self.width // 2

        if self.red:
            self.home_x_range  = range(0, self.mid_x)
            self.enemy_x_range = range(self.mid_x, self.width)
        else:
            self.home_x_range  = range(self.mid_x, self.width)
            self.enemy_x_range = range(0, self.mid_x)

        self.legal_positions = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if not walls[x][y]
        ]

        # Pre-compute all-pairs maze distances (O(1) look-up at runtime)
        self._dist_cache = {}
        self._precompute_distances(walls)

        # Our boundary entry points (home side, adjacent to mid-line)
        self.entry_points = self._find_entry_points(walls)

        # Invader escape boundary (enemy side of mid-line) – used by siege/guardian
        self._enemy_entries = self._compute_enemy_entries()

        # ── Zone belief system (Tier 3 position estimates) ───────────────
        # Map is divided into 4×4 zones (8 wide × 4 tall = 32 zones total).
        # Zone IDs are (bx, by) where bx = (x-1)//4, by = (y-1)//4.
        # The outer ring of the 18×34 grid is all walls, so legal x ∈ [1,32]
        # and legal y ∈ [1,16], giving exactly 8×4 zones with no cross-boundary.
        self._zone_size   = 4
        self.pos_to_zone  = {
            (x, y): ((x-1) // self._zone_size, (y-1) // self._zone_size)
            for x, y in self.legal_positions
        }
        # Collect legal cells per zone
        zone_cells_tmp = {}
        for pos, z in self.pos_to_zone.items():
            zone_cells_tmp.setdefault(z, []).append(pos)
        self.zone_cells = {z: set(cells) for z, cells in zone_cells_tmp.items()}

        # Zone centre: legal cell in zone closest to geometric mean
        self.zone_center = {}
        for z, cells in self.zone_cells.items():
            cx = sum(p[0] for p in cells) / len(cells)
            cy = sum(p[1] for p in cells) / len(cells)
            self.zone_center[z] = min(cells,
                                      key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)

        # Zone neighbours (4-connected in zone grid)
        all_zones = set(self.zone_cells.keys())
        self.zone_neighbors = {}
        for bx, by in all_zones:
            self.zone_neighbors[(bx, by)] = [
                (bx+dx, by+dy)
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]
                if (bx+dx, by+dy) in all_zones
            ]

        # home/enemy zone sets (for boundary filtering during zone diffusion)
        self.home_zones  = {z for z, cells in self.zone_cells.items()
                            if any(p[0] in self.home_x_range  for p in cells)}
        self.enemy_zones = {z for z, cells in self.zone_cells.items()
                            if any(p[0] in self.enemy_x_range for p in cells)}

        # Zone belief distributions {opp: {zone: prob}}
        # Initialised to the zone containing each opponent's start position.
        opponents = self.get_opponents(game_state)
        self.zone_beliefs = {}
        for opp in opponents:
            opp_start = game_state.get_initial_agent_position(opp)
            start_zone = self.pos_to_zone.get(opp_start,
                                              next(iter(self.zone_cells)))
            self.zone_beliefs[opp] = {start_zone: 1.0}

        # Keep a thin cell-level belief dict solely for the attribution heuristic
        # in update_beliefs (comparing which opponent is closer to eaten food).
        # It is NOT used for path-planning or danger maps.
        self.beliefs = {}
        for opp in opponents:
            opp_start = game_state.get_initial_agent_position(opp)
            self.beliefs[opp] = {opp_start: 1.0}

        # Food snapshot for invader-attribution event detection
        self.prev_food_defending = self.get_food_you_are_defending(game_state).as_list()

        # Per-step state (updated in choose_action)
        self.my_carrying = 0

        # Invader-active flag: True for INVADER_ACTIVE_STEPS after our food is eaten
        self.invader_active       = False
        self.invader_active_timer = 0

        # Food-disappearance position cache (tier-2 position source)
        # last_eaten_pos[opp]  = position where food was eaten (attributed to opp)
        # last_eaten_age[opp]  = steps since that event (evicted after 15 steps)
        self.last_eaten_pos = {}
        self.last_eaten_age = {opp: 999 for opp in self.get_opponents(game_state)}

        # ── Choke-point precomputation (done once at init) ────────────────
        # entry_control_value[e]  = # home-side foods reachable from e (BFS)
        # cluster_bottleneck[id]  = best entry for each food cluster
        # choke_points            = set of articulation points in home territory
        self.entry_control_value  = self._compute_entry_control_values(game_state)
        self.cluster_bottleneck   = self._compute_cluster_bottlenecks(game_state)
        self.choke_points         = self._find_choke_points()

        # ── Map-symmetry mirror: enemy choke points ──────────────────────
        self.enemy_choke_points = self._mirror_positions(self.choke_points)

        # ── Enemy safe loop (biconnected component with ≥2 home exits) ───
        (self.enemy_safe_loop,
         self.enemy_loop_entries) = self._find_enemy_safe_loop(game_state)

        # ── Dead-end precomputation (tip-peeling, full map) ───────────────
        # dead_end_depth[pos]    = peeling wave (higher = deeper in dead end)
        # dead_end_entrance[pos] = exit cell that seals off the dead end pos is in
        self.dead_end_depth, self.dead_end_entrance = self._compute_dead_end_depth()

        # ── Situation evaluator state ─────────────────────────────────────
        self.situation_smooth  = 0.0   # smoothed [-1.5, 1.5] situation value
        self.prev_opp_banked   = 0     # for detecting sudden enemy carrying surge
        self.total_food_init   = (len(self.get_food(game_state).as_list()) +
                                  len(self.get_food_you_are_defending(game_state).as_list()))
        self.capsule_state     = CAP_NORMAL

        # Dynamic params (recomputed each step by evaluate_situation)
        self.d_return_margin   = RETURN_SAFE_MARGIN
        self.d_carry_threshold = 5
        self.d_siege_threshold = SIEGE_THRESHOLD
        self.d_endgame_thresh  = ENDGAME_THRESHOLD
        self.d_kill_dist       = OPPORTUNISTIC_KILL_DIST
        self.d_attack_window   = 999   # steps left to safely attack under THEY_SCARED

    def _precompute_distances(self, walls):
        for src in self.legal_positions:
            dist  = {src: 0}
            queue = deque([src])
            while queue:
                pos = queue.popleft()
                x, y = pos
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nb = (x+dx, y+dy)
                    if nb not in dist \
                            and 0 <= nb[0] < self.width \
                            and 0 <= nb[1] < self.height \
                            and not walls[nb[0]][nb[1]]:
                        dist[nb] = dist[pos] + 1
                        queue.append(nb)
            self._dist_cache[src] = dist

    def get_dist(self, a, b):
        """O(1) maze distance; Manhattan fallback for off-grid positions."""
        try:
            return self._dist_cache[a][b]
        except KeyError:
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _find_entry_points(self, walls):
        """Legal positions on our home side at the boundary column."""
        bx = (self.mid_x - 1) if self.red else self.mid_x
        entries = [(bx, y) for y in range(self.height) if not walls[bx][y]]
        return entries or self.legal_positions[:1]

    def _compute_enemy_entries(self):
        """Legal positions on the enemy side of the boundary (invader escape gates)."""
        legal_set = set(self.legal_positions)
        ex = self.mid_x if self.red else (self.mid_x - 1)
        entries = [(ex, y) for y in range(self.height) if (ex, y) in legal_set]
        return entries or list(self.entry_points)

    def _compute_entry_control_values(self, game_state):
        """
        For each entry point, count how many home-side food cells are reachable
        from it via BFS within our territory.  Higher = more valuable to guard.
        Computed once at init using the initial food layout.
        """
        food_set = set(self.get_food_you_are_defending(game_state).as_list())
        walls    = game_state.get_walls()
        control  = {}

        for e in self.entry_points:
            reachable = 0
            visited   = {e}
            queue     = deque([e])
            while queue:
                pos = queue.popleft()
                if pos in food_set:
                    reachable += 1
                x, y = pos
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nb = (x+dx, y+dy)
                    if nb in visited or walls[nb[0]][nb[1]]:
                        continue
                    if nb[0] not in self.home_x_range:
                        continue
                    visited.add(nb)
                    queue.append(nb)
            control[e] = reachable

        return control

    def _compute_cluster_bottlenecks(self, game_state):
        """
        For each food cluster (grouped by proximity), find the entry point
        that sits on the most paths into that cluster — i.e. the entry whose
        BFS to the cluster has the smallest distance sum across all cluster foods.
        Returns {cluster_id: best_entry_point}.
        Cluster id = index into the sorted cluster list.
        """
        food_list = self.get_food_you_are_defending(game_state).as_list()
        if not food_list or not self.entry_points:
            return {}

        # Build clusters (same single-linkage logic as OffensiveAgent)
        remaining = list(food_list)
        clusters  = []
        while remaining:
            cluster = [remaining.pop(0)]
            changed = True
            while changed:
                changed  = False
                leftover = []
                for f in remaining:
                    if any(self.get_dist(f, c) <= CLUSTER_RADIUS for c in cluster):
                        cluster.append(f)
                        changed = True
                    else:
                        leftover.append(f)
                remaining = leftover
            clusters.append(cluster)

        bottleneck = {}
        for cid, cluster in enumerate(clusters):
            best_entry = min(
                self.entry_points,
                key=lambda e: sum(self.get_dist(e, f) for f in cluster)
            )
            bottleneck[cid] = best_entry

        # Also store the clusters for runtime use
        self._home_clusters = clusters
        return bottleneck

    def _find_choke_points(self):
        """
        Identify articulation points in our home territory using an iterative
        version of Tarjan's algorithm (DFS-based).  An articulation point is a
        node whose removal disconnects the home subgraph.
        Returns a set of (x, y) positions.
        """
        home_nodes = [p for p in self.legal_positions
                      if p[0] in self.home_x_range]
        if not home_nodes:
            return set()

        # Build adjacency for home subgraph
        home_set = set(home_nodes)
        adj      = {p: [] for p in home_nodes}
        for p in home_nodes:
            x, y = p
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nb = (x+dx, y+dy)
                if nb in home_set:
                    adj[p].append(nb)

        # Iterative DFS Tarjan
        visited   = {}   # node → discovery time
        low       = {}
        parent    = {}
        chokes    = set()
        timer     = [0]

        for start in home_nodes:
            if start in visited:
                continue
            # Stack: (node, iterator over neighbours, is_first_visit)
            stack = [(start, iter(adj[start]), True)]
            parent[start] = None
            while stack:
                node, nbr_iter, first = stack[-1]
                if first:
                    visited[node] = low[node] = timer[0]
                    timer[0] += 1
                    stack[-1] = (node, nbr_iter, False)

                try:
                    nb = next(nbr_iter)
                    if nb not in visited:
                        parent[nb] = node
                        stack.append((nb, iter(adj[nb]), True))
                    elif nb != parent[node]:
                        low[node] = min(low[node], visited[nb])
                except StopIteration:
                    stack.pop()
                    if stack:
                        par = stack[-1][0]
                        low[par] = min(low[par], low[node])
                        # Root check
                        if parent[par] is None:
                            root_children = sum(
                                1 for nb in adj[par]
                                if parent.get(nb) == par)
                            if root_children > 1:
                                chokes.add(par)
                        # Non-root check
                        elif low[node] >= visited[par]:
                            chokes.add(par)

        return chokes

    def _mirror_positions(self, positions):
        """Map a set of home-side positions to their mirror on the enemy side."""
        return {(self.width - 1 - x, self.height - 1 - y) for x, y in positions}

    def _find_enemy_safe_loop(self, game_state):
        """
        Find the best 'safe loop' in enemy territory: a biconnected component
        (BCC) with ≥2 exits adjacent to home entry points, so our pacman can
        circle inside it while being chased by a single ghost.

        Returns (loop_cells, exit_nodes) or (set(), []) if none found.
        """
        enemy_nodes = [p for p in self.legal_positions
                       if p[0] in self.enemy_x_range]
        if not enemy_nodes:
            return set(), []

        enemy_set = set(enemy_nodes)
        entry_set = set(self.entry_points)

        adj = {p: [] for p in enemy_nodes}
        for p in enemy_nodes:
            x, y = p
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nb = (x+dx, y+dy)
                if nb in enemy_set:
                    adj[p].append(nb)

        # Iterative edge-based BCC decomposition
        bccs     = []
        disc     = {}
        low      = {}
        parent   = {}
        timer    = [0]
        edge_stk = []

        for start in enemy_nodes:
            if start in disc:
                continue
            dfs_stk = [(start, iter(adj[start]))]
            parent[start] = None
            disc[start] = low[start] = timer[0]
            timer[0] += 1

            while dfs_stk:
                u, nbr_iter = dfs_stk[-1]
                try:
                    v = next(nbr_iter)
                    if v not in disc:
                        parent[v] = u
                        disc[v] = low[v] = timer[0]
                        timer[0] += 1
                        edge_stk.append((u, v))
                        dfs_stk.append((v, iter(adj[v])))
                    elif v != parent[u] and disc[v] < disc[u]:
                        low[u] = min(low[u], disc[v])
                        edge_stk.append((u, v))
                except StopIteration:
                    dfs_stk.pop()
                    if dfs_stk:
                        p_node = dfs_stk[-1][0]
                        low[p_node] = min(low[p_node], low[u])
                        if low[u] >= disc[p_node]:
                            bcc_nodes = set()
                            while edge_stk:
                                eu, ev = edge_stk.pop()
                                bcc_nodes.add(eu)
                                bcc_nodes.add(ev)
                                if (eu == p_node and ev == u) or \
                                   (ev == p_node and eu == u):
                                    break
                            if len(bcc_nodes) > 2:
                                bccs.append(bcc_nodes)

        if not bccs:
            return set(), []

        init_food = set(self.get_food(game_state).as_list())
        best_loop, best_exits, best_score = None, None, -1

        for bcc in bccs:
            exits = [node for node in bcc
                     if any((node[0]+dx, node[1]+dy) in entry_set
                            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)])]
            if len(exits) < 2:
                continue
            food_count     = sum(1 for f in bcc if f in init_food)
            min_entry_dist = min(self.get_dist(n, e)
                                 for n in bcc for e in self.entry_points) + 1
            score = food_count / min_entry_dist
            if score > best_score:
                best_score, best_loop, best_exits = score, bcc, exits

        if best_loop is None:
            return set(), []
        return best_loop, best_exits

    def _compute_dead_end_depth(self):
        """
        Tip-peeling algorithm on the full legal graph.

        Returns (dead_end_depth, dead_end_entrance):

        dead_end_depth:   {pos: wave}  where wave >= 1.
                          Higher wave = deeper inside the dead end (closer to blind tip).
                          Cells not in any dead end are absent (depth == 0).

        dead_end_entrance: {pos: entrance_cell}
                          For every dead-end cell, the first cell *outside* the
                          dead-end corridor (the "seal point" a defender should occupy).
        """
        legal_set = set(self.legal_positions)

        # Build neighbour counts over the full legal graph
        nbr_count = {}
        nbr_list  = {}
        for p in self.legal_positions:
            x, y = p
            nbrs = [(x+dx, y+dy) for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]
                    if (x+dx, y+dy) in legal_set]
            nbr_count[p] = len(nbrs)
            nbr_list[p]  = nbrs

        # Iterative tip-peeling
        wave       = {}   # pos → wave number it was peeled
        current_w  = 1
        active     = {p for p, c in nbr_count.items() if c == 1}
        remaining  = dict(nbr_count)   # mutable copy

        while active:
            next_active = set()
            for p in active:
                if p in wave:
                    continue
                wave[p] = current_w
                for nb in nbr_list[p]:
                    if nb in wave:
                        continue
                    remaining[nb] -= 1
                    if remaining[nb] == 1:
                        next_active.add(nb)
            active    = next_active
            current_w += 1

        dead_end_depth = wave  # wave value IS the depth (higher = deeper)

        # Build dead_end_entrance: for each dead-end cell, walk toward lower
        # wave (shallower) until we step outside dead_end_depth entirely.
        dead_end_entrance = {}
        for p in dead_end_depth:
            current = p
            while current in dead_end_depth:
                # Move toward the neighbour with the lowest wave (exit direction)
                candidates = [nb for nb in nbr_list[current] if nb not in wave
                              or wave.get(nb, 999) < wave.get(current, 999)]
                if not candidates:
                    # All neighbours have higher or equal wave: p is a tip; its
                    # entrance is the single neighbour outside the dead end, if any
                    outside = [nb for nb in nbr_list[current]
                               if nb not in dead_end_depth]
                    dead_end_entrance[p] = outside[0] if outside else current
                    break
                nxt = min(candidates, key=lambda nb: wave.get(nb, 0))
                if nxt not in dead_end_depth:
                    dead_end_entrance[p] = nxt
                    break
                current = nxt
            else:
                dead_end_entrance[p] = current

        return dead_end_depth, dead_end_entrance

    # ── situation evaluator ───────────────────────────────────────────────

    def evaluate_situation(self, game_state):
        """
        Called once per step (after update_beliefs).
        Computes a smoothed situation score and derives all dynamic parameters.

        situation_smooth ∈ [-1.5, 1.5]:
          > 0  → we are ahead (considering carried food)
          < 0  → opponent is ahead
          ±1   ≈ one side has eaten all opponent food

        Capsule state:
          CAP_NORMAL      → no scared effects active
          CAP_WE_SCARED   → our ghosts are scared (opp ate our capsule)
          CAP_THEY_SCARED → opp ghosts are scared (we ate their capsule)
          CAP_BOTH_SCARED → both scared simultaneously
        """
        opponents = self.get_opponents(game_state)
        team      = self.get_team(game_state)

        real_diff  = self.get_score(game_state)
        my_banked  = sum(game_state.get_agent_state(i).num_carrying for i in team)
        opp_banked = sum(game_state.get_agent_state(o).num_carrying for o in opponents)

        denom      = max(self.total_food_init * 0.5, 1)
        sit_raw    = max(-1.5, min(1.5,
                         (real_diff + my_banked - opp_banked) / denom))

        # Emergency: opp carrying surged → skip smoothing for fast reaction
        if opp_banked - self.prev_opp_banked > 2:
            self.situation_smooth = sit_raw
        else:
            self.situation_smooth = 0.7 * self.situation_smooth + 0.3 * sit_raw
        self.prev_opp_banked = opp_banked

        s = self.situation_smooth

        # ── capsule state ────────────────────────────────────────────────
        my_scared  = any(game_state.get_agent_state(i).scared_timer > 0
                         for i in team)
        opp_scared = any(game_state.get_agent_state(o).scared_timer > 0
                         for o in opponents)
        if my_scared and opp_scared:
            self.capsule_state = CAP_BOTH_SCARED
        elif my_scared:
            self.capsule_state = CAP_WE_SCARED
        elif opp_scared:
            self.capsule_state = CAP_THEY_SCARED
        else:
            self.capsule_state = CAP_NORMAL

        # ── dynamic parameter derivation ─────────────────────────────────

        # Return safety margin
        if s > 0.4:    self.d_return_margin = 6
        elif s > 0.15: self.d_return_margin = 4
        elif s > -0.15: self.d_return_margin = 3
        elif s > -0.4: self.d_return_margin = 2
        else:          self.d_return_margin = 1

        # THEY_SCARED: ghost can't eat us, no margin needed
        if self.capsule_state == CAP_THEY_SCARED:
            self.d_return_margin = 0
        # WE_SCARED: our defender is offline, attacker should be more cautious
        elif self.capsule_state == CAP_WE_SCARED:
            self.d_return_margin = min(self.d_return_margin + 2, 6)

        # Carry threshold (how much food before returning)
        remaining_food = len(self.get_food(game_state).as_list())
        base_thresh = (1 if remaining_food <= 2 else
                       2 if remaining_food <= 5 else
                       3 if remaining_food <= 10 else
                       5 if remaining_food <= 20 else 5)
        adjustment = round(s * 2)   # leading → smaller thresh; behind → larger
        self.d_carry_threshold = max(1, base_thresh - adjustment)

        # Siege threshold
        if s > 0.3:    self.d_siege_threshold = 8
        elif s > 0:    self.d_siege_threshold = 6
        elif s > -0.3: self.d_siege_threshold = 4
        else:          self.d_siege_threshold = 2
        # Hard floor: always siege if opp_banked itself exceeds threshold
        if opp_banked >= self.d_siege_threshold:
            self.d_siege_threshold = opp_banked  # ensure trigger fires this step

        # Endgame threshold
        if s > 0.3:    self.d_endgame_thresh = 90
        elif s > 0:    self.d_endgame_thresh = 70
        elif s > -0.3: self.d_endgame_thresh = 50
        else:          self.d_endgame_thresh = 30

        # Opportunistic kill distance
        if s > 0.2:    self.d_kill_dist = 5
        elif s > -0.2: self.d_kill_dist = 3
        else:          self.d_kill_dist = 2

        # THEY_SCARED attack window
        if self.capsule_state == CAP_THEY_SCARED:
            opp_scared_timer = min(
                game_state.get_agent_state(o).scared_timer
                for o in opponents)
            my_pos    = game_state.get_agent_position(self.index)
            home_dist = min(self.get_dist(my_pos, e) for e in self.entry_points)
            self.d_attack_window = opp_scared_timer - home_dist - 3
        else:
            self.d_attack_window = 999

    def update_beliefs(self, game_state):
        opponents     = self.get_opponents(game_state)
        cur_defending = self.get_food_you_are_defending(game_state).as_list()
        eaten_by_opp  = set(self.prev_food_defending) - set(cur_defending)

        # ── Attribution: which opponent ate our food? ─────────────────────
        # Use zone-peak centre distance to attribute food disappearance.
        eating_opp = None
        eaten_pos  = None
        if eaten_by_opp:
            eaten_pos  = min(eaten_by_opp,
                             key=lambda p: min(
                                 self.get_dist(self.zone_center[
                                     self._get_zone_peak(opp)], p)
                                 for opp in opponents))
            eating_opp = min(
                opponents,
                key=lambda opp: self.get_dist(
                    self.zone_center[self._get_zone_peak(opp)], eaten_pos))

        my_pos = game_state.get_agent_position(self.index)
        noisy_dists = game_state.get_agent_distances()

        for opp in opponents:
            # ── Tier 1: exact observation ─────────────────────────────────
            pos = game_state.get_agent_position(opp)
            if pos is not None:
                z = self.pos_to_zone.get(pos, self._get_zone_peak(opp))
                self.zone_beliefs[opp] = {z: 1.0}
                # Keep cell-level belief in sync for attribution next step
                self.beliefs[opp] = {pos: 1.0}
                continue

            # ── Tier 2: food disappearance attribution ────────────────────
            if eaten_by_opp and opp == eating_opp:
                z = self.pos_to_zone.get(eaten_pos, self._get_zone_peak(opp))
                self.zone_beliefs[opp] = {z: 1.0}
                self.beliefs[opp]      = {eaten_pos: 1.0}
                self.last_eaten_pos[opp] = eaten_pos
                self.last_eaten_age[opp] = 0
                continue

            # ── Tier 3: zone-level diffusion ──────────────────────────────
            opp_is_pacman = game_state.get_agent_state(opp).is_pacman
            valid_zones   = self.home_zones if opp_is_pacman else self.enemy_zones

            # Diffuse: 50% stay, 50% spread equally to neighbours
            old_dist = self.zone_beliefs[opp]
            new_dist = {}
            for z, prob in old_dist.items():
                new_dist[z] = new_dist.get(z, 0.0) + prob * 0.5
                nbrs = self.zone_neighbors.get(z, [])
                if nbrs:
                    share = (prob * 0.5) / len(nbrs)
                    for nz in nbrs:
                        new_dist[nz] = new_dist.get(nz, 0.0) + share

            # Boundary filter
            new_dist = {z: p for z, p in new_dist.items() if z in valid_zones}
            if not new_dist:
                new_dist = {z: p for z, p in old_dist.items() if z in valid_zones}
            if not new_dist:
                new_dist = old_dist  # full fallback

            # Noisy-distance filter (zone-centre Manhattan, ±NOISE_RANGE+4)
            noisy = noisy_dists[opp]
            filtered = {
                z: p for z, p in new_dist.items()
                if abs(abs(my_pos[0] - self.zone_center[z][0]) +
                       abs(my_pos[1] - self.zone_center[z][1]) - noisy)
                   <= NOISE_RANGE + self._zone_size
            }
            if not filtered:
                filtered = new_dist

            # Normalise
            total = sum(filtered.values())
            self.zone_beliefs[opp] = ({z: p/total for z, p in filtered.items()}
                                      if total > 0 else filtered)

            # Sync thin cell-level belief to zone peak centre (attribution only)
            peak_centre = self.zone_center[self._get_zone_peak(opp)]
            self.beliefs[opp] = {peak_centre: 1.0}

        # Persist snapshot
        self.prev_food_defending = cur_defending

        # Age Tier-2 cache
        for opp in opponents:
            if opp != eating_opp:
                self.last_eaten_age[opp] = self.last_eaten_age.get(opp, 999) + 1

        # Invader-active flag
        if eaten_by_opp:
            self.invader_active       = True
            self.invader_active_timer = 12
        elif self.invader_active_timer > 0:
            self.invader_active_timer -= 1
            self.invader_active = self.invader_active_timer > 0
        else:
            self.invader_active = False

    def _get_zone_peak(self, opp):
        """Zone with highest probability for this opponent."""
        return max(self.zone_beliefs[opp], key=lambda z: self.zone_beliefs[opp][z])

    def get_belief_peak(self, opp):
        """Cell-level peak: returns zone-centre of highest-probability zone.
        Kept for attribution heuristic compatibility; planning code uses
        get_best_opp_pos instead."""
        return self.zone_center[self._get_zone_peak(opp)]

    def get_best_opp_pos(self, opp, game_state):
        """
        Best position estimate for an opponent (3-tier priority):
          Tier 1 – exact observation (visible within 5 cells)    → exact cell
          Tier 2 – food-disappearance attribution (≤15 steps)    → exact cell
          Tier 3 – zone belief peak centre                       → zone centre cell
        """
        # Tier 1: exact observation
        pos = game_state.get_agent_position(opp)
        if pos is not None:
            return pos

        # Tier 2: recent food-disappearance position
        age = self.last_eaten_age.get(opp, 999)
        if age <= 15 and opp in self.last_eaten_pos:
            return self.last_eaten_pos[opp]

        # Tier 3: zone peak centre (coarse directional estimate)
        return self.zone_center[self._get_zone_peak(opp)]

    # ── danger map ────────────────────────────────────────────────────────

    def get_danger_map(self, game_state):
        """
        Ghost-threat heat map (non-scared ghosts only).

        Tier 1 (visible):      exact position, weight 3.0, radius 4
        Tier 2 (food eaten):   eaten cell, weight decays with age, radius 4
        Tier 3 (zone belief):  zone-peak centre, weight 0.2 × peak_prob, radius 5
                               — low weight so coarse estimates don't dominate A*

        Also adds a static penalty for enemy choke points.
        """
        danger    = {}
        opponents = self.get_opponents(game_state)

        for opp in opponents:
            st = game_state.get_agent_state(opp)
            if st.scared_timer > 0 or st.is_pacman:
                continue

            exact_pos = game_state.get_agent_position(opp)

            if exact_pos is not None:
                for cand in self.legal_positions:
                    d = self.get_dist(exact_pos, cand)
                    if d <= 4:
                        danger[cand] = danger.get(cand, 0) + 3.0 / (d + 1)
            else:
                age = self.last_eaten_age.get(opp, 999)
                if age <= 15 and opp in self.last_eaten_pos:
                    eaten_pos  = self.last_eaten_pos[opp]
                    age_factor = max(0.2, 1.0 - age / 15.0)
                    for cand in self.legal_positions:
                        d = self.get_dist(eaten_pos, cand)
                        if d <= 4:
                            danger[cand] = (danger.get(cand, 0)
                                            + age_factor * 1.5 / (d + 1))
                else:
                    # Tier 3: zone-peak centre, low weight directional signal
                    peak_zone   = self._get_zone_peak(opp)
                    peak_prob   = self.zone_beliefs[opp][peak_zone]
                    zone_ctr    = self.zone_center[peak_zone]
                    weight      = 0.2 * peak_prob
                    for cand in self.legal_positions:
                        d = self.get_dist(zone_ctr, cand)
                        if d <= 5:
                            danger[cand] = danger.get(cand, 0) + weight / (d + 1)

        scale = 1.0 / max(DANGER_WEIGHT, 1)
        for p in self.enemy_choke_points:
            danger[p] = danger.get(p, 0) + ENEMY_CHOKE_PENALTY * scale

        return danger

    def _build_scared_danger_map(self, threat_positions):
        """Danger map treating pacman threats as sources (used during flee)."""
        danger = {}
        for pos in threat_positions:
            for cand in self.legal_positions:
                d = self.get_dist(pos, cand)
                if d <= SCARED_SAFE_DIST:
                    danger[cand] = danger.get(cand, 0) + 1.0 / (d + 1)
        return danger

    def _build_ghost_forbidden_set(self, game_state):
        """
        Hard-forbidden cells (treated as walls by a_star_safe) when we are a
        pacman or scared.  BFS radius-2 around each visible non-scared ghost.
        Returns empty set when CAP_THEY_SCARED (ghosts can't eat us).
        """
        if self.capsule_state == CAP_THEY_SCARED:
            return set()

        forbidden = set()
        walls     = game_state.get_walls()

        for opp in self.get_opponents(game_state):
            st = game_state.get_agent_state(opp)
            if st.is_pacman or st.scared_timer > 0:
                continue
            pos = game_state.get_agent_position(opp)
            if pos is None:
                continue   # not visible – skip (no Tier-2/3 for hard forbid)

            # BFS outward to radius 2 from ghost position
            visited = {pos}
            queue   = deque([(pos, 0)])
            while queue:
                cur, dist = queue.popleft()
                forbidden.add(cur)
                if dist >= 2:
                    continue
                cx, cy = cur
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nb = (cx+dx, cy+dy)
                    if (nb not in visited
                            and 0 <= nb[0] < self.width
                            and 0 <= nb[1] < self.height
                            and not walls[nb[0]][nb[1]]):
                        visited.add(nb)
                        queue.append((nb, dist+1))

        return forbidden

    def _no_active_ghost_nearby(self, game_state):
        """
        Returns True when no non-scared enemy ghost is currently visible
        (i.e. within manhattan distance 5, the game's vision radius).

        Visible = game_state.get_agent_position() returns a value.
        When this is True, no Tier-1 threat exists and greedy play is safe.
        """
        for opp in self.get_opponents(game_state):
            st  = game_state.get_agent_state(opp)
            pos = game_state.get_agent_position(opp)
            if pos is None:          continue  # outside vision range
            if st.is_pacman:         continue  # pacman, not a ghost threat
            if st.scared_timer > 0:  continue  # scared, harmless
            return False   # at least one visible active ghost
        return True

    # ── pathfinding ───────────────────────────────────────────────────────

    def a_star_safe(self, game_state, start, goal, danger_map,
                    scared_timer=0, forbidden_set=None):
        """A* with danger penalties.  forbidden_set cells are treated as walls."""
        if start == goal:
            return None
        d_weight  = 0 if scared_timer > 0 else DANGER_WEIGHT
        forbidden = forbidden_set or set()
        heap      = util.PriorityQueue()
        heap.push((start, None), 0)
        g_cost   = {start: 0}
        visited  = set()
        walls    = game_state.get_walls()

        while not heap.is_empty():
            pos, first_action = heap.pop()
            if pos in visited:
                continue
            visited.add(pos)
            if pos == goal:
                return first_action
            x, y = pos
            for action, (dx, dy) in [
                    ('North',(0,1)),('South',(0,-1)),
                    ('East',(1,0)),('West',(-1,0))]:
                nb = (x+dx, y+dy)
                if not (0 <= nb[0] < self.width and 0 <= nb[1] < self.height):
                    continue
                if walls[nb[0]][nb[1]] or nb in visited:
                    continue
                if nb in forbidden:
                    continue
                new_g = g_cost[pos] + 1 + danger_map.get(nb, 0) * d_weight
                if new_g < g_cost.get(nb, float('inf')):
                    g_cost[nb] = new_g
                    fa = first_action if first_action is not None else action
                    heap.push((nb, fa), new_g + self.get_dist(nb, goal))
        return None

    def bfs_action(self, game_state, start, goal):
        """Unweighted BFS; returns first action or None."""
        if start == goal:
            return None
        walls   = game_state.get_walls()
        queue   = deque([(start, None)])
        visited = {start}
        while queue:
            pos, first = queue.popleft()
            x, y = pos
            for action, (dx, dy) in [
                    ('North',(0,1)),('South',(0,-1)),
                    ('East',(1,0)),('West',(-1,0))]:
                nb = (x+dx, y+dy)
                if nb in visited or walls[nb[0]][nb[1]]:
                    continue
                fa = first if first is not None else action
                if nb == goal:
                    return fa
                visited.add(nb)
                queue.append((nb, fa))
        return None

    def _bfs_path(self, game_state, start, goals):
        """BFS returning full position list from start to nearest goal."""
        if not goals:
            return []
        goal_set = set(goals)
        walls    = game_state.get_walls()
        queue    = deque([(start, [start])])
        visited  = {start}
        while queue:
            pos, path = queue.popleft()
            if pos in goal_set:
                return path
            x, y = pos
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nb = (x+dx, y+dy)
                if nb in visited or walls[nb[0]][nb[1]]:
                    continue
                visited.add(nb)
                queue.append((nb, path + [nb]))
        return []

    # ── shared state helpers ──────────────────────────────────────────────

    def get_teammate_state(self, game_state):
        team   = self.get_team(game_state)
        tm_idx = [i for i in team if i != self.index][0]
        return tm_idx, game_state.get_agent_position(tm_idx), \
               game_state.get_agent_state(tm_idx)

    def safe_random_action(self, game_state, forbidden=None):
        """Random non-Stop action, avoiding forbidden cells when possible."""
        my_pos = game_state.get_agent_position(self.index)
        fb = forbidden or set()

        def action_result(a):
            dx, dy = {'North':(0,1),'South':(0,-1),
                      'East':(1,0),'West':(-1,0)}.get(a, (0,0))
            return (my_pos[0]+dx, my_pos[1]+dy)

        actions = [a for a in game_state.get_legal_actions(self.index)
                   if a != 'Stop']
        safe = [a for a in actions if action_result(a) not in fb]
        pool = safe if safe else actions   # degrade gracefully if all forbidden
        return random.choice(pool) if pool else 'Stop'

    def get_projected_score(self, game_state):
        """Current score + food carried by both our agents."""
        score = self.get_score(game_state)
        for idx in self.get_team(game_state):
            score += game_state.get_agent_state(idx).num_carrying
        return score

    # ── shared fallbacks ──────────────────────────────────────────────────

    def _offensive_fallback(self, game_state):
        """Large deficit: act as a second attacker."""
        my_pos    = game_state.get_agent_position(self.index)
        danger    = self.get_danger_map(game_state)
        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return self.safe_random_action(game_state)
        target = min(food_list, key=lambda f: self.get_dist(my_pos, f))
        return self.a_star_safe(game_state, my_pos, target, danger) \
               or self.safe_random_action(game_state)

    def _defensive_fallback(self, game_state, danger_map, forbidden=None):
        """Large lead: retreat to boundary and patrol."""
        my_pos   = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        fb       = forbidden or set()
        if my_state.is_pacman:
            best = min(self.entry_points,
                       key=lambda e: self.get_dist(my_pos, e))
            return self.a_star_safe(game_state, my_pos, best, danger_map,
                                    forbidden_set=fb) \
                   or self.safe_random_action(game_state)
        target = min(self.entry_points,
                     key=lambda e: self.get_dist(my_pos, e))
        return self.bfs_action(game_state, my_pos, target) \
               or self.safe_random_action(game_state)

    # ── smart flee ────────────────────────────────────────────────────────

    def _action_flee_smart(self, game_state, my_pos, mode='defend'):
        """
        Flee while minimising recovery cost after scare ends.

        mode='defend' : resume toward nearest entry point (defender/attacker)
        mode='siege'  : resume toward nearest invader (ready to re-engage)

        When scared_timer <= dist-to-resume + SCARED_RESUME_BUFFER,
        start moving back toward the resume target (still avoiding threats).
        """
        opponents    = self.get_opponents(game_state)
        scared_timer = game_state.get_agent_state(self.index).scared_timer

        threats = [game_state.get_agent_position(opp)
                   or self.get_best_opp_pos(opp, game_state)
                   for opp in opponents
                   if game_state.get_agent_state(opp).is_pacman]

        if mode == 'siege':
            invaders = [opp for opp in opponents
                        if game_state.get_agent_state(opp).is_pacman]
            if invaders:
                resume_target = min(
                    (game_state.get_agent_position(opp) or
                     self.get_best_opp_pos(opp, game_state)
                     for opp in invaders),
                    key=lambda p: self.get_dist(my_pos, p))
            else:
                resume_target = min(self.entry_points,
                                    key=lambda e: self.get_dist(my_pos, e))
        else:
            resume_target = min(self.entry_points,
                                key=lambda e: self.get_dist(my_pos, e))

        resume_dist = self.get_dist(my_pos, resume_target)

        # Pre-emptive reposition: scare almost over → move toward resume target
        # but still use a danger-penalised path to avoid threats
        if scared_timer <= resume_dist + SCARED_RESUME_BUFFER:
            scared_danger = self._build_scared_danger_map(threats)
            action = self.a_star_safe(game_state, my_pos, resume_target,
                                      scared_danger, scared_timer=0)
            return action or self.safe_random_action(game_state)

        # Full flee: stay in home territory, satisfy safety distance, then
        # pick the safe position closest to the resume target
        home_candidates = [p for p in self.legal_positions
                           if p[0] in self.home_x_range]

        safe_candidates = [
            c for c in home_candidates
            if not threats or
               all(self.get_dist(c, t) >= SCARED_SAFE_DIST for t in threats)
        ]
        if not safe_candidates:
            safe_candidates = home_candidates  # degrade gracefully

        target = min(safe_candidates,
                     key=lambda c: self.get_dist(c, resume_target))
        if target == my_pos:
            return self.safe_random_action(game_state)
        return self.bfs_action(game_state, my_pos, target) \
               or self.safe_random_action(game_state)

    # ── siege mode ────────────────────────────────────────────────────────

    def is_siege_mode(self, game_state):
        """
        True when opponents are carrying enough food to be a serious threat.
        Uses d_siege_threshold (dynamic, set by evaluate_situation).
        """
        opponents = self.get_opponents(game_state)
        opp_carrying = sum(
            game_state.get_agent_state(opp).num_carrying
            for opp in opponents
            if game_state.get_agent_state(opp).is_pacman
        )
        return opp_carrying >= self.d_siege_threshold

    def _siege_chase(self, game_state, my_pos):
        """
        Siege behaviour shared by both agents.
        Returns an action, or None if no invaders remain (caller should
        fall through to normal logic).
        """
        opponents        = self.get_opponents(game_state)
        all_invaders     = [opp for opp in opponents
                            if game_state.get_agent_state(opp).is_pacman]
        visible_invaders = [opp for opp in all_invaders
                            if game_state.get_agent_position(opp) is not None]

        if not all_invaders:
            return None   # siege target gone

        scared_timer = game_state.get_agent_state(self.index).scared_timer
        if scared_timer > 0:
            return self._action_flee_smart(game_state, my_pos, mode='siege')

        # Kill opportunity first
        kill_pos = self._find_kill_opportunity(
            game_state, my_pos, visible_invaders, scared_timer)
        if kill_pos:
            return self.bfs_action(game_state, my_pos, kill_pos)

        # Role assignment
        role = self._assign_defender_role(
            game_state, my_pos, all_invaders, visible_invaders)

        if role == 'chaser':
            if visible_invaders:
                return self._action_chase_visible(
                    game_state, my_pos, visible_invaders)
            return self._action_pursue_active_invader(
                game_state, my_pos, all_invaders)
        else:  # guardian → block the exit
            inv_pos = (game_state.get_agent_position(all_invaders[0])
                       or self.get_best_opp_pos(all_invaders[0], game_state))
            return self._siege_block_exit(game_state, my_pos, inv_pos)

    def _siege_block_exit(self, game_state, my_pos, inv_pos):
        """
        Siege guardian: go to the escape gate the invader is heading for,
        cutting off the route entirely.
        """
        if not self._enemy_entries:
            return self.bfs_action(game_state, my_pos, inv_pos)
        target = min(self._enemy_entries,
                     key=lambda e: self.get_dist(inv_pos, e))
        return self.bfs_action(game_state, my_pos, target) \
               or self.safe_random_action(game_state)

    # ── shared chase helpers (used by DefensiveAgent & _siege_chase) ─────

    def _find_kill_opportunity(self, game_state, my_pos,
                               visible_invaders, scared_timer):
        """Return invader position if we can intercept before it escapes, else None."""
        if scared_timer > 0 or not visible_invaders:
            return None
        best_pos, best_gain = None, 0
        for opp in visible_invaders:
            inv_pos = game_state.get_agent_position(opp)
            if inv_pos is None or not self._enemy_entries:
                continue
            inv_escape = min(self.get_dist(inv_pos, e) for e in self._enemy_entries)
            gain       = inv_escape - self.get_dist(my_pos, inv_pos)
            if gain > best_gain:
                best_gain, best_pos = gain, inv_pos
        return best_pos

    def _assign_defender_role(self, game_state, my_pos,
                               all_invaders, visible_invaders):
        """
        When both our agents are ghosts and there is exactly one invader,
        assign 'chaser' (closer agent, pursues directly) or
        'guardian' (farther agent, blocks escape exit).

        Previously both agents returned 'guardian' when equidistant, leaving
        nobody to chase.  Now the closer one always becomes chaser.
        """
        _, tm_pos, tm_state = self.get_teammate_state(game_state)

        if tm_state.is_pacman or len(all_invaders) != 1:
            return 'chaser'

        opp     = all_invaders[0]
        inv_pos = (game_state.get_agent_position(opp)
                   if opp in visible_invaders
                   else self.get_best_opp_pos(opp, game_state))

        if tm_pos is None:
            return 'chaser'

        my_dist = self.get_dist(my_pos, inv_pos)
        tm_dist = self.get_dist(tm_pos, inv_pos)

        # Closer agent chases, farther agent guards – no tie → both guard bug
        if my_dist <= tm_dist:
            return 'chaser'
        return 'guardian'

    def _action_chase_visible(self, game_state, my_pos, visible_invaders):
        """Chase the nearest visible invader directly."""
        best_action, best_dist = None, float('inf')
        for opp in visible_invaders:
            inv_pos = game_state.get_agent_position(opp)
            if inv_pos is None:
                continue
            d = self.get_dist(my_pos, inv_pos)
            if d < best_dist:
                best_dist   = d
                best_action = self.bfs_action(game_state, my_pos, inv_pos)
        return best_action

    def _action_approach_believed(self, game_state, my_pos, all_invaders):
        """
        No Tier-1/2 info available.  Use zone belief to pick the entry point
        closest to where each invader is estimated to be, then go wait there.
        This is more honest than chasing a vague zone-centre cell directly.
        """
        best_action, best_dist = None, float('inf')
        for opp in all_invaders:
            est  = self.get_best_opp_pos(opp, game_state)
            # Head for the entry point nearest to the estimated position
            target = min(self.entry_points,
                         key=lambda e: self.get_dist(e, est))
            d = self.get_dist(my_pos, target)
            if d < best_dist:
                best_dist   = d
                best_action = self.bfs_action(game_state, my_pos, target)
        return best_action

    def _action_pursue_active_invader(self, game_state, my_pos, all_invaders):
        """
        invader_active=True: food just disappeared – Tier-2 eaten position is
        fresh and reliable.  Go there directly rather than using zone centre.
        Falls back to zone-based entry approach if no Tier-2 data.
        """
        best_action, best_score = None, -float('inf')
        for opp in all_invaders:
            # Prefer Tier-2 (exact eaten cell) over Tier-3 (zone centre)
            age = self.last_eaten_age.get(opp, 999)
            if age <= 15 and opp in self.last_eaten_pos:
                target = self.last_eaten_pos[opp]
            else:
                target = self.get_best_opp_pos(opp, game_state)
            depth = (self.mid_x - target[0]) if self.red else (target[0] - self.mid_x)
            score = depth / (self.get_dist(my_pos, target) + 1)
            if score > best_score:
                best_score  = score
                best_action = self.bfs_action(game_state, my_pos, target)
        return best_action

    def _action_guardian(self, game_state, my_pos, inv_pos):
        """
        Guardian role: block the invader's escape.

        Priority:
          1. If the invader is inside a known food cluster, move to that
             cluster's precomputed bottleneck entry (O(1) look-up).
          2. Otherwise, walk the BFS escape path and find the earliest cell
             we can reach before the invader (dynamic interception).
          3. Fallback: return None (caller uses _action_guard_entries).
        """
        # Priority 1: cluster bottleneck look-up
        if hasattr(self, '_home_clusters') and self.cluster_bottleneck:
            for cid, cluster in enumerate(self._home_clusters):
                if inv_pos in cluster:
                    best_entry = self.cluster_bottleneck.get(cid)
                    if best_entry and best_entry != my_pos:
                        return self.bfs_action(game_state, my_pos, best_entry)
                    break  # invader found in cluster; skip dynamic check

        # Priority 2: dynamic escape-path interception
        if not self._enemy_entries:
            return None
        escape_path = self._bfs_path(game_state, inv_pos, self._enemy_entries)
        if not escape_path or len(escape_path) < 2:
            return None

        for i, path_pos in enumerate(escape_path[1:], start=1):
            if self.get_dist(my_pos, path_pos) <= i + 1:
                if path_pos != my_pos:
                    return self.bfs_action(game_state, my_pos, path_pos)
                break

        return None   # caller falls back to _action_guard_entries

# ──────────────────────────────────────────────
# Offensive Agent
# ──────────────────────────────────────────────

class OffensiveAgent(BaseAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.target_cluster   = None   # current food cluster we are eating
        self._prev_was_pacman = False  # for detecting home-return transition
        self.loop_mode_active = False  # True when patrolling the enemy safe loop

    # ── main decision ─────────────────────────────────────────────────────

    def choose_action(self, game_state):
        start_time = time.time()

        self.update_beliefs(game_state)
        self.evaluate_situation(game_state)          # sets all self.d_xxx params

        danger_map       = self.get_danger_map(game_state)
        my_pos           = game_state.get_agent_position(self.index)
        my_state         = game_state.get_agent_state(self.index)
        self.my_carrying = my_state.num_carrying
        scared_timer     = self._nearest_ghost_scared_timer(game_state)

        # ── Bug 1-B: reset target_cluster when we cross back to home side ──
        # is_pacman flips False the step we step onto home territory and food
        # is banked.  Clear the stale cluster reference so _cluster_finished
        # doesn't immediately return True on the next trip out.
        if not my_state.is_pacman and self._prev_was_pacman:
            self.target_cluster   = None
            self.loop_mode_active = False
        self._prev_was_pacman = my_state.is_pacman

        # ── Ghost forbidden set ───────────────────────────────────────────
        # Built once and reused throughout this step.  Non-empty only when we
        # are a pacman or scared; cleared automatically when opponents are scared.
        my_scared_now = my_state.scared_timer > 0
        if my_state.is_pacman or my_scared_now:
            forbidden = self._build_ghost_forbidden_set(game_state)
        else:
            forbidden = set()

        # ── Siege mode (uses dynamic d_siege_threshold) ───────────────────
        if self.is_siege_mode(game_state):
            if my_state.is_pacman:
                action = self._siege_pacman_response(
                    game_state, my_pos, danger_map, forbidden)
            else:
                action = self._siege_chase(game_state, my_pos)
            if action:
                return action
            # Siege target gone → fall through to normal logic

        # ── Both opponents are pacmen: switch to defense ──────────────────
        # If both enemies are on our side eating food, abandon attack and defend.
        # Condition lifted as soon as any opponent returns to ghost state.
        opponents = self.get_opponents(game_state)
        both_opp_pacman = all(game_state.get_agent_state(opp).is_pacman
                              for opp in opponents)
        if both_opp_pacman:
            if my_state.is_pacman:
                # Still on enemy side: get home first, then chase
                return self._action_return_home(game_state, my_pos, danger_map,
                                                forbidden) \
                       or self.safe_random_action(game_state, forbidden)
            else:
                # Already a ghost: chase nearest invader directly
                all_inv = [opp for opp in opponents
                           if game_state.get_agent_state(opp).is_pacman]
                vis_inv = [opp for opp in all_inv
                           if game_state.get_agent_position(opp) is not None]
                if vis_inv:
                    action = self._action_chase_visible(game_state, my_pos, vis_inv)
                else:
                    action = self._action_approach_believed(game_state, my_pos, all_inv)
                return action or self.safe_random_action(game_state)

        # ── Large lead: play defensively ─────────────────────────────────
        if self.get_score(game_state) >= WIN_THRESHOLD:
            return self._defensive_fallback(game_state, danger_map, forbidden)

        # ── Ghost branch (on home side) ───────────────────────────────────
        if not my_state.is_pacman:
            kill_pos = (self._check_opportunistic_kill(game_state, my_pos, allow_detour=False)
                        or self._check_opportunistic_kill(game_state, my_pos, allow_detour=True))
            if kill_pos:
                action = self.bfs_action(game_state, my_pos, kill_pos)
                if action:
                    return action

            # Capsule priority: if an enemy capsule is reachable safely,
            # go eat it before crossing into general attack mode.
            action = self._capsule_priority_action(
                game_state, my_pos, danger_map, forbidden,
                home_dist=self._home_dist(my_pos),
                safety=999,        # ghost on home side – no threat
                scared_timer=scared_timer)
            if action:
                return action

            action = self._action_attack(game_state, my_pos, danger_map,
                                         scared_timer, forbidden)
            return action or self.safe_random_action(game_state)

        # ── Pacman branch (on enemy side) ─────────────────────────────────
        home_dist  = self._home_dist(my_pos)
        ghost_dist = self._nearest_ghost_dist(game_state, my_pos)
        safety     = ghost_dist - home_dist   # >0 → can reach home safely

        # Apply dead-end penalty to danger_map (modifies a copy so we don't
        # pollute the base map used elsewhere this step)
        if self.capsule_state != CAP_THEY_SCARED:
            import copy
            danger_map = copy.copy(danger_map)
            self._apply_dead_end_penalty(danger_map, ghost_dist)

        food_list       = self.get_food(game_state).as_list()
        remaining_moves = game_state.data.timeleft // 4
        projected       = self.get_projected_score(game_state)

        # THEY_SCARED: ghosts are harmless – go greedy.
        # Only hard exits: attack window expired (must flee before scared ends)
        # or carrying so much that losing it all would be catastrophic.
        if self.capsule_state == CAP_THEY_SCARED:
            if self.d_attack_window <= 0:
                return self._action_return_home(game_state, my_pos, danger_map,
                                                forbidden) \
                       or self.safe_random_action(game_state, forbidden)
            # Loose carry cap: allow up to 2× normal threshold before banking.
            # Avoids hoarding indefinitely while still maximising scared window.
            if self.my_carrying >= self.d_carry_threshold * 2:
                return self._action_return_home(game_state, my_pos, danger_map,
                                                forbidden) \
                       or self.safe_random_action(game_state, forbidden)
            return self._greedy_attack(game_state, my_pos) \
                   or self.safe_random_action(game_state, forbidden)

        # WE_SCARED: our defender offline, be more conservative
        if self.capsule_state == CAP_WE_SCARED and self.my_carrying > 0:
            return self._action_return_home(game_state, my_pos, danger_map,
                                            forbidden) \
                   or self.safe_random_action(game_state, forbidden)

        # Endgame: use dynamic threshold (leading → enter sooner)
        if remaining_moves < self.d_endgame_thresh and projected > 0:
            if self.my_carrying > 0:
                return self._action_return_home(game_state, my_pos, danger_map,
                                                forbidden) \
                       or self.safe_random_action(game_state, forbidden)
            return self._defensive_fallback(game_state, danger_map, forbidden)

        # Hard return: about to die / out of time / no food left
        if (scared_timer == 0 and safety < 1) \
                or remaining_moves < home_dist + 8 \
                or not food_list:
            return self._action_return_home(game_state, my_pos, danger_map,
                                            forbidden) \
                   or self.safe_random_action(game_state, forbidden)

        # ── Greedy mode: no visible active ghost ─────────────────────────
        # When _no_active_ghost_nearby() is True there is no Tier-1 threat –
        # skip all safety machinery and go straight for the nearest food.
        # A loose carry cap (2× normal threshold) is the only return trigger,
        # preventing runaway hoarding without wasting the safe window.
        if self._no_active_ghost_nearby(game_state):
            if self.my_carrying >= self.d_carry_threshold * 2:
                return self._action_return_home(game_state, my_pos, danger_map,
                                                forbidden) \
                       or self.safe_random_action(game_state, forbidden)
            return self._greedy_attack(game_state, my_pos) \
                   or self.safe_random_action(game_state, forbidden)

        # Capsule escape: ghost nearby, capsule closer than home (passive safety)
        capsules = self.get_capsules(game_state)
        if capsules and scared_timer == 0 and safety < self.d_return_margin + 2:
            cap = min(capsules, key=lambda c: self.get_dist(my_pos, c))
            if self.get_dist(my_pos, cap) < home_dist:
                action = self.a_star_safe(game_state, my_pos, cap, danger_map,
                                          forbidden_set=forbidden)
                if action:
                    return action

        # Proactive capsule priority: eat enemy capsule before soft-return fires.
        # Higher priority than _should_return_now so a safe capsule path isn't
        # interrupted by a cluster-finish deposit trigger.
        action = self._capsule_priority_action(
            game_state, my_pos, danger_map, forbidden,
            home_dist, safety, scared_timer)
        if action:
            return action

        # Logic A: soft return (uses dynamic thresholds)
        if self._should_return_now(game_state, my_pos, safety, food_list, projected):
            return self._action_return_home(game_state, my_pos, danger_map,
                                            forbidden) \
                   or self.safe_random_action(game_state, forbidden)

        # ── Safe-loop patrol mode ─────────────────────────────────────────
        # Enter/stay in loop mode when normal routes are too risky but the
        # precomputed enemy safe loop is available and accessible.
        # Exit conditions (checked first): carrying enough, loop empty,
        # two ghosts near loop exits (would block both escape routes).
        if self.enemy_safe_loop:
            # Count active (non-scared) ghosts near loop exits
            opponents    = self.get_opponents(game_state)
            ghosts_near  = sum(
                1 for opp in opponents
                if not game_state.get_agent_state(opp).is_pacman
                and game_state.get_agent_state(opp).scared_timer == 0
                and any(self.get_dist(
                            self.get_best_opp_pos(opp, game_state), ex) <= 4
                        for ex in self.enemy_loop_entries))
            loop_food    = [f for f in food_list
                            if f in self.enemy_safe_loop]
            loop_blocked = ghosts_near >= 2   # both exits covered → trap

            # Deactivate loop mode if it's no longer useful
            if self.loop_mode_active:
                if (self.my_carrying >= self.d_carry_threshold
                        or not loop_food
                        or loop_blocked):
                    self.loop_mode_active = False

            # Activate loop mode when normal path is risky and loop helps
            if not self.loop_mode_active and not loop_blocked and loop_food:
                active_ghosts = [
                    opp for opp in opponents
                    if not game_state.get_agent_state(opp).is_pacman
                    and game_state.get_agent_state(opp).scared_timer == 0]
                only_one_ghost = len(active_ghosts) <= 1
                # Enter if: only one ghost active AND safety is tight
                # AND our position is reachable from the loop
                in_or_near_loop = (my_pos in self.enemy_safe_loop or
                                   min(self.get_dist(my_pos, p)
                                       for p in self.enemy_safe_loop) <= 4)
                if only_one_ghost and safety < self.d_return_margin + 3 \
                        and in_or_near_loop:
                    self.loop_mode_active = True

            if self.loop_mode_active:
                action = self._action_loop_patrol(
                    game_state, my_pos, danger_map, forbidden)
                if action:
                    return action
                self.loop_mode_active = False  # patrol found no path – fall through

        # Default: keep attacking
        action = self._action_attack(game_state, my_pos, danger_map,
                                     scared_timer, forbidden)
        if action is None or time.time() - start_time > 0.9:
            action = self.safe_random_action(game_state, forbidden)
        return action

    # ── capsule priority attack ───────────────────────────────────────────

    def _capsule_priority_action(self, game_state, my_pos, danger_map,
                                  forbidden, home_dist, safety, scared_timer):
        """
        Evaluate whether to prioritise eating an enemy capsule over normal food.

        Conditions to trigger:
          - CAP_NORMAL (opponent not already scared – capsule still has value)
          - scared_timer == 0 (opponent ghosts are active)
          - Enemy capsules exist

        For each capsule, scores:
          safety_margin = ghost_dist_to_cap - my_dist_to_cap
          scared_after  = 40 - my_dist_to_cap  (scared steps left after eating)
          score         = safety_margin * 2 + (scared_after / 40) * 3

        Only pursues capsules where:
          safety_margin >= CAPSULE_SAFE_MARGIN
          scared_after  >= CAPSULE_HARVEST_MIN
          capsule cell not in forbidden and danger below threshold

        Teammate coordination: skip capsules the teammate is clearly closer to.

        Returns an a_star_safe action or None.
        """
        if self.capsule_state != CAP_NORMAL or scared_timer != 0:
            return None

        capsules = self.get_capsules(game_state)
        if not capsules:
            return None

        opponents  = self.get_opponents(game_state)
        _, tm_pos, _ = self.get_teammate_state(game_state)

        best_action, best_score = None, -float('inf')

        for cap in capsules:
            # Hard exclusions
            if cap in (forbidden or set()):
                continue
            if danger_map.get(cap, 0) > 0.5:
                continue

            my_dist = self.get_dist(my_pos, cap)

            # Teammate coordination: skip if teammate is closer by >2
            if tm_pos and self.get_dist(tm_pos, cap) < my_dist - 2:
                continue

            # Ghost distance to this capsule (use best pos estimate)
            ghost_cap_dist = min(
                (self.get_dist(self.get_best_opp_pos(opp, game_state), cap)
                 for opp in opponents
                 if not game_state.get_agent_state(opp).is_pacman),
                default=999)

            safety_margin = ghost_cap_dist - my_dist
            scared_after  = 40 - my_dist

            if safety_margin < CAPSULE_SAFE_MARGIN:
                continue
            if scared_after < CAPSULE_HARVEST_MIN:
                continue

            score = safety_margin * 2 + (scared_after / 40.0) * 3
            if score > best_score:
                action = self.a_star_safe(game_state, my_pos, cap,
                                          danger_map, forbidden_set=forbidden)
                if action:
                    best_score  = score
                    best_action = action

        return best_action

    # ── siege response for pacman ─────────────────────────────────────────

    def _siege_pacman_response(self, game_state, my_pos, danger_map,
                               forbidden=None):
        """
        Siege triggered while we are a pacman (on enemy side).
        Priority: get home quickly (with easy food if possible), then join siege.
        """
        home_dist  = self._home_dist(my_pos)
        ghost_dist = self._nearest_ghost_dist(game_state, my_pos)
        safety     = ghost_dist - home_dist
        fb         = forbidden or set()

        # Safety first: if near-dead, ignore food and run
        if safety < 1:
            return self._action_return_home(game_state, my_pos, danger_map, fb)

        # If already carrying food, go home and deposit
        if self.my_carrying > 0:
            return self._action_return_home(game_state, my_pos, danger_map, fb)

        # Carrying nothing: try to grab easy food on the way home
        food_list  = self.get_food(game_state).as_list()
        easy_food  = self._find_easy_food(game_state, my_pos, danger_map,
                                          home_dist, safety, food_list)
        if easy_food:
            return self.a_star_safe(game_state, my_pos, easy_food, danger_map,
                                    forbidden_set=fb)

        return self._action_return_home(game_state, my_pos, danger_map, fb)

    def _find_easy_food(self, game_state, my_pos, danger_map,
                        home_dist, safety, food_list):
        """
        Find a food pellet that is safe and roughly on the way home.
        Conditions:
          1. The food cell itself is not in a high-danger zone.
          2. food_dist + dist(food → nearest_entry) <= home_dist + DETOUR_TOLERANCE
             (not too much of a detour)
          3. Total trip (food + home leg) still fits within safety margin.
        Returns the best candidate or None.
        """
        candidates = []
        for f in food_list:
            if danger_map.get(f, 0) > 0.3:
                continue
            food_dist  = self.get_dist(my_pos, f)
            home_after = min(self.get_dist(f, e) for e in self.entry_points)
            detour     = food_dist + home_after

            if detour > home_dist + DETOUR_TOLERANCE:
                continue
            if detour > safety - 1:
                continue
            candidates.append((f, detour))

        if not candidates:
            return None
        # Pick the one with the smallest total trip cost
        return min(candidates, key=lambda x: x[1])[0]

    # ── return-home decision ──────────────────────────────────────────────

    def _should_return_now(self, game_state, my_pos, safety, food_list, projected):
        """
        Logic A: multi-condition soft return trigger (uses dynamic thresholds).
        Only fires when carrying > 0.
        """
        carrying = self.my_carrying
        if carrying == 0:
            return False

        # 1. Big lead: lock it in (dynamic threshold)
        big_lead = self.total_food_init * 0.3
        if projected >= big_lead:
            return True

        # 2. Absolute carry threshold (dynamic, set by evaluate_situation)
        if carrying >= self.d_carry_threshold:
            return True

        # 3. Risk/reward: more food carried → lower dynamic risk tolerance
        if carrying >= 2 and safety < carrying + 1:
            return True

        # 4. Current cluster cleared → bank food before starting the next one
        if self._cluster_finished(game_state, food_list) and carrying >= 1:
            return True

        # 5. Ghost too close (dynamic margin)
        if safety < self.d_return_margin:
            return True

        return False

    # ── attack actions ────────────────────────────────────────────────────

    def _action_attack(self, game_state, my_pos, danger_map, scared_timer,
                       forbidden=None):
        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return self._action_return_home(game_state, my_pos, danger_map, forbidden)

        _, tm_pos, _ = self.get_teammate_state(game_state)

        if self._cluster_finished(game_state, food_list):
            # Cluster cleared: bank whatever we're carrying before starting the
            # next one.  If carrying == 0 we just passed through an empty cluster
            # (or just spawned), so select the next target immediately.
            if self.my_carrying > 0:
                return self._action_return_home(game_state, my_pos, danger_map,
                                                forbidden)
            self.target_cluster = self._select_target_cluster(
                game_state, my_pos, food_list, danger_map, tm_pos)

        if self.target_cluster:
            cluster_food = [f for f in self.target_cluster
                            if f in set(food_list)]
            target = (min(cluster_food, key=lambda f: self.get_dist(my_pos, f))
                      if cluster_food else None)
            if target is None:
                self.target_cluster = None
                target = min(food_list, key=lambda f: self.get_dist(my_pos, f))
        else:
            target = min(food_list, key=lambda f: self.get_dist(my_pos, f))

        return self.a_star_safe(game_state, my_pos, target, danger_map,
                                scared_timer=scared_timer,
                                forbidden_set=forbidden or set())

    def _action_return_home(self, game_state, my_pos, danger_map, forbidden=None):
        best_entry = min(self.entry_points,
                         key=lambda e: self.get_dist(my_pos, e))
        return self.a_star_safe(game_state, my_pos, best_entry, danger_map,
                                forbidden_set=forbidden or set())

    def _greedy_attack(self, game_state, my_pos):
        """
        Nearest-food BFS with zero safety overhead.

        Called only when _no_active_ghost_nearby() is True or when
        CAP_THEY_SCARED makes all ghosts harmless.  In both cases
        forbidden_set and danger_map are irrelevant (no Tier-1 threat),
        so plain BFS is both correct and faster.
        """
        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return None
        target = min(food_list, key=lambda f: self.get_dist(my_pos, f))
        return self.bfs_action(game_state, my_pos, target)

    # ── cluster helpers ───────────────────────────────────────────────────

    def _build_clusters(self, food_list):
        if not food_list:
            return []
        remaining = list(food_list)
        clusters  = []
        while remaining:
            cluster = [remaining.pop(0)]
            changed = True
            while changed:
                changed       = False
                next_rem      = []
                for f in remaining:
                    if any(self.get_dist(f, c) <= CLUSTER_RADIUS for c in cluster):
                        cluster.append(f)
                        changed = True
                    else:
                        next_rem.append(f)
                remaining = next_rem
            clusters.append(cluster)
        return clusters

    def _cluster_finished(self, game_state, food_list):
        if self.target_cluster is None:
            return True
        food_set = set(food_list)
        return not any(f in food_set for f in self.target_cluster)

    def _select_target_cluster(self, game_state, my_pos,
                                food_list, danger_map, tm_pos):
        """
        Three-tier cluster selection priority:

        Tier 1 – "We arrive first": clusters where our distance to the nearest
                 food is strictly less than every active enemy ghost's distance
                 to that same cluster.  Scored by arrival advantage and size.

        Tier 2 – "Safe approach": clusters with no Tier-1 advantage but whose
                 centroid lies in enemy territory AND that have at least two
                 distinct entry points reachable without passing through the
                 same choke point (so we can't be cornered).  Scored by size
                 and risk.

        Tier 3 – Original fallback: size / (dist × risk) for any remaining
                 cluster.

        All tiers also penalise clusters the teammate is already heading for.
        """
        clusters = self._build_clusters(food_list)
        if not clusters:
            return None

        opponents   = self.get_opponents(game_state)
        ghost_poses = [self.get_best_opp_pos(opp, game_state)
                       for opp in opponents
                       if not game_state.get_agent_state(opp).is_pacman]

        def teammate_penalty(cluster):
            if tm_pos is None:
                return 0
            tm_d  = min(self.get_dist(tm_pos, f) for f in cluster)
            my_d  = min(self.get_dist(my_pos, f) for f in cluster) + 1
            return len(cluster) if tm_d < my_d * 0.8 else 0

        def avg_risk(cluster):
            return sum(danger_map.get(f, 0) for f in cluster) / len(cluster)

        def my_dist_to(cluster):
            return min(self.get_dist(my_pos, f) for f in cluster) + 1

        def ghost_dist_to(cluster):
            if not ghost_poses:
                return 999
            return min(self.get_dist(gp, f)
                       for gp in ghost_poses for f in cluster)

        def has_two_escape_routes(cluster):
            """
            True if the cluster has at least two entry points reachable from
            its centroid that do NOT both pass through the same choke point.
            Uses precomputed enemy_choke_points as a proxy for single-route risk.
            """
            cx = sum(f[0] for f in cluster) / len(cluster)
            cy = sum(f[1] for f in cluster) / len(cluster)
            # Nearest legal cell to centroid
            centroid = min(self.legal_positions,
                           key=lambda p: abs(p[0]-cx) + abs(p[1]-cy))

            reachable_entries = [e for e in self.entry_points
                                 if self.get_dist(centroid, e) < 999]
            if len(reachable_entries) < 2:
                return False

            # Check whether all paths to entry share a single choke point.
            # Simple heuristic: if the centroid is NOT itself an enemy choke
            # point and at least 2 entries are within reasonable range, treat
            # as multi-route.
            if centroid in self.enemy_choke_points:
                return False
            return True

        # ── Tier 1: we arrive first ───────────────────────────────────────
        tier1 = []
        for cluster in clusters:
            my_d    = my_dist_to(cluster)
            ghost_d = ghost_dist_to(cluster)
            if ghost_d > my_d:   # we get there first
                advantage = ghost_d - my_d
                score = (advantage * len(cluster)
                         / (my_d * (1 + avg_risk(cluster) * 3)
                            + teammate_penalty(cluster)))
                tier1.append((score, cluster))

        if tier1:
            return max(tier1, key=lambda x: x[0])[1]

        # ── Tier 2: safe multi-route cluster ─────────────────────────────
        tier2 = []
        for cluster in clusters:
            # Centroid must be in enemy territory
            cx = sum(f[0] for f in cluster) / len(cluster)
            if cx not in self.enemy_x_range:
                continue
            if not has_two_escape_routes(cluster):
                continue
            my_d  = my_dist_to(cluster)
            score = len(cluster) / (my_d * (1 + avg_risk(cluster) * 3)
                                    + teammate_penalty(cluster))
            tier2.append((score, cluster))

        if tier2:
            return max(tier2, key=lambda x: x[0])[1]

        # ── Tier 3: original scoring (fallback) ───────────────────────────
        best, best_score = None, -float('inf')
        for cluster in clusters:
            my_d  = my_dist_to(cluster)
            score = len(cluster) / (my_d * (1 + avg_risk(cluster) * 3)
                                    + teammate_penalty(cluster))
            if score > best_score:
                best_score, best = score, cluster
        return best

    # ── opportunistic kill (ghost on home side) ───────────────────────────

    def _check_opportunistic_kill(self, game_state, my_pos, allow_detour=False):
        """
        allow_detour=False: invader within d_kill_dist (dynamic).
        allow_detour=True : we can intercept before invader escapes.
        Only fires when we are a ghost and not scared.
        """
        my_state = game_state.get_agent_state(self.index)
        if my_state.is_pacman or my_state.scared_timer > 0:
            return None

        kill_dist = self.d_kill_dist   # dynamic threshold

        best_pos, best_val = None, -1
        for opp in self.get_opponents(game_state):
            if not game_state.get_agent_state(opp).is_pacman:
                continue
            inv_pos = game_state.get_agent_position(opp)
            if inv_pos is None:
                continue
            chase = self.get_dist(my_pos, inv_pos)

            if not allow_detour:
                if chase <= kill_dist:
                    val = kill_dist - chase
                    if val > best_val:
                        best_val, best_pos = val, inv_pos
            else:
                if not self.entry_points:
                    continue
                inv_escape = min(self.get_dist(inv_pos, e)
                                 for e in self.entry_points)
                if chase < inv_escape and chase > kill_dist:
                    val = inv_escape - chase
                    if val > best_val:
                        best_val, best_pos = val, inv_pos
        return best_pos

    # ── safe-loop patrol ──────────────────────────────────────────────────

    def _action_loop_patrol(self, game_state, my_pos, danger_map, forbidden):
        """
        Patrol the precomputed enemy safe loop.

        Inside the loop we suppress ghost danger penalties in A* (the whole
        point is to stay on the loop and outrun a single ghost) but keep
        forbidden_set active so we never step directly onto a visible ghost
        or its immediate neighbours.

        Target selection:
          1. Any food still remaining inside the loop → go eat it.
          2. No food left in loop → head for the nearest loop exit to go home.
        """
        food_list = self.get_food(game_state).as_list()
        loop_food = [f for f in food_list if f in self.enemy_safe_loop]

        if loop_food:
            target = min(loop_food, key=lambda f: self.get_dist(my_pos, f))
        elif self.enemy_loop_entries:
            # No food left in loop; head to the nearest exit to go home
            target = min(self.enemy_loop_entries,
                         key=lambda e: self.get_dist(my_pos, e))
        else:
            return None

        # Use a zeroed danger map inside the loop so A* doesn't shy away from
        # loop cells due to ghost heat – forbidden_set still provides hard safety.
        empty_danger = {}
        return self.a_star_safe(game_state, my_pos, target, empty_danger,
                                forbidden_set=forbidden)

    # ── distance helpers ──────────────────────────────────────────────────

    def _home_dist(self, my_pos):
        return min(self.get_dist(my_pos, e) for e in self.entry_points)

    def _apply_dead_end_penalty(self, danger_map, ghost_dist):
        """
        Overlay dead-end depth penalties onto danger_map (in-place).
        Only applied to enemy-side cells (our pacman's territory of operation).
        Penalty grows as ghost gets closer:
          weight = DEAD_END_BASE_WEIGHT * (1 + max(0, DEAD_END_GHOST_SCALE - ghost_dist)
                                               / DEAD_END_GHOST_SCALE)
        Divided by DANGER_WEIGHT so it integrates naturally with ghost costs in A*.
        """
        scale      = 1.0 / max(DANGER_WEIGHT, 1)
        proximity  = max(0.0, (DEAD_END_GHOST_SCALE - ghost_dist) / DEAD_END_GHOST_SCALE)
        weight     = DEAD_END_BASE_WEIGHT * (1.0 + proximity) * scale

        for p, depth in self.dead_end_depth.items():
            if p[0] in self.enemy_x_range:   # only penalise enemy-side cells
                danger_map[p] = danger_map.get(p, 0) + depth * weight

    def _nearest_ghost_dist(self, game_state, my_pos):
        opponents = self.get_opponents(game_state)
        dists = []
        for opp in opponents:
            if game_state.get_agent_state(opp).is_pacman:
                continue
            best = self.get_best_opp_pos(opp, game_state)
            d    = self.get_dist(my_pos, best)
            # Tier 3 only: zone centre may be up to ~half-zone away from true pos.
            # Apply a conservative inward correction so we don't underestimate risk.
            if (game_state.get_agent_position(opp) is None and
                    self.last_eaten_age.get(opp, 999) > 15):
                d = max(d - self._zone_size // 2, 1)
            dists.append(d)
        return min(dists) if dists else 999

    def _nearest_ghost_scared_timer(self, game_state):
        opponents = self.get_opponents(game_state)
        timers = [game_state.get_agent_state(opp).scared_timer
                  for opp in opponents
                  if not game_state.get_agent_state(opp).is_pacman]
        return min(timers) if timers else 0

# ──────────────────────────────────────────────
# Defensive Agent
# ──────────────────────────────────────────────

class DefensiveAgent(BaseAgent):

    # register_initial_state inherited from BaseAgent (no extra fields needed)

    # ── main decision ─────────────────────────────────────────────────────

    def choose_action(self, game_state):
        start_time = time.time()

        self.update_beliefs(game_state)
        self.evaluate_situation(game_state)          # sets all self.d_xxx params

        my_pos       = game_state.get_agent_position(self.index)
        my_state     = game_state.get_agent_state(self.index)
        scared_timer = my_state.scared_timer

        # Large deficit → act as second attacker
        if self.get_score(game_state) <= LOSE_THRESHOLD:
            return self._offensive_fallback(game_state)

        # ── Siege mode ───────────────────────────────────────────────────
        if self.is_siege_mode(game_state):
            action = self._siege_chase(game_state, my_pos)
            if action:
                return action
            # Invaders gone → fall through to normal patrol

        # ── Scared: smart flee ───────────────────────────────────────────
        if scared_timer > FLEE_THRESHOLD:
            # Use 'siege' mode flee if siege would be active after scared ends
            flee_mode = 'siege' if self.invader_active else 'defend'
            return self._action_flee_smart(game_state, my_pos, mode=flee_mode) \
                   or self.safe_random_action(game_state)

        # Classify invaders
        opponents        = self.get_opponents(game_state)
        all_invaders     = [opp for opp in opponents
                            if game_state.get_agent_state(opp).is_pacman]
        visible_invaders = [opp for opp in all_invaders
                            if game_state.get_agent_position(opp) is not None]

        # ── Trap override: invader is inside a dead end → seal the exit ───
        # Highest priority (above kill_opportunity): if we are a ghost, not
        # scared, and a visible invader is in a dead-end cell, move to (or
        # wait at) the seal point.  The invader has no way out.
        if not my_state.is_pacman and scared_timer == 0 and visible_invaders:
            for opp in visible_invaders:
                inv_pos = game_state.get_agent_position(opp)
                if inv_pos and inv_pos in self.dead_end_entrance:
                    seal = self.dead_end_entrance[inv_pos]
                    if my_pos == seal:
                        return 'Stop'   # already at seal point, just wait
                    action = self.bfs_action(game_state, my_pos, seal)
                    if action:
                        return action

        # ── Kill opportunity (preempts role assignment) ───────────────────
        kill_target = self._find_kill_opportunity(
            game_state, my_pos, visible_invaders, scared_timer)
        if kill_target:
            action = self.bfs_action(game_state, my_pos, kill_target)
            if action:
                return action

        # ── Capsule active defence ────────────────────────────────────────
        # If an invader is closing in on our capsule and we can intercept,
        # drop everything else and go guard it.  Runs after kill_opportunity
        # so we don't skip a trivial kill just to guard a distant capsule.
        if not my_state.is_pacman and scared_timer == 0:
            action = self._action_guard_capsule(
                game_state, my_pos, all_invaders, visible_invaders)
            if action:
                return action

        # ── Dual-defender role assignment ─────────────────────────────────
        if all_invaders:
            role = self._assign_defender_role(
                game_state, my_pos, all_invaders, visible_invaders)

            if role == 'guardian':
                inv_pos = (game_state.get_agent_position(all_invaders[0])
                           or self.get_best_opp_pos(all_invaders[0], game_state))
                action  = self._action_guardian(game_state, my_pos, inv_pos)
                if action is None:
                    action = self._action_guard_entries(game_state, my_pos)
                if action is None or time.time() - start_time > 0.9:
                    action = self.safe_random_action(game_state)
                return action

        # ── Chase visible invader ─────────────────────────────────────────
        if visible_invaders:
            action = self._action_chase_visible(
                game_state, my_pos, visible_invaders)
            if action:
                return action

        # ── Pursue believed invader ───────────────────────────────────────
        if all_invaders:
            if self.invader_active:
                action = self._action_pursue_active_invader(
                    game_state, my_pos, all_invaders)
            else:
                action = self._action_approach_believed(
                    game_state, my_pos, all_invaders)
            if action:
                return action

        # ── No invaders: guard entry points ──────────────────────────────
        action = self._action_guard_entries(game_state, my_pos)
        if action is None or time.time() - start_time > 0.9:
            action = self.safe_random_action(game_state)
        return action

    # ── capsule active defence ────────────────────────────────────────────

    def _action_guard_capsule(self, game_state, my_pos,
                               all_invaders, visible_invaders):
        """
        If an invader is heading toward one of our capsules and we can
        intercept it (arrive no later than the invader), go block the capsule.

        Only fires when:
          - We are a ghost and not scared
          - Our side has capsules
          - An invader's best-known position is within CAPSULE_THREAT_DIST of
            a capsule
          - We can reach that capsule at least as fast as the invader
          - Our teammate is not already closer (avoid double-covering)

        Returns an action or None.
        """
        my_capsules = self.get_capsules_you_are_defending(game_state)
        if not my_capsules or not all_invaders:
            return None

        _, tm_pos, _ = self.get_teammate_state(game_state)

        best_action, best_urgency = None, -1

        for cap in my_capsules:
            my_dist = self.get_dist(my_pos, cap)

            # Teammate check: if teammate is closer, let them handle it
            if tm_pos and self.get_dist(tm_pos, cap) < my_dist:
                continue

            for opp in all_invaders:
                inv_pos = (game_state.get_agent_position(opp)
                           or self.get_best_opp_pos(opp, game_state))
                inv_dist = self.get_dist(inv_pos, cap)

                if inv_dist > CAPSULE_THREAT_DIST:
                    continue   # invader not heading this way yet

                # Can we intercept? (arrive no later than invader)
                if my_dist > inv_dist + 1:
                    continue

                # Urgency: lower inv_dist = more urgent
                urgency = CAPSULE_THREAT_DIST - inv_dist
                if urgency > best_urgency:
                    best_urgency = urgency
                    best_action  = self.bfs_action(game_state, my_pos, cap)

        return best_action

    # ── guard entry points ────────────────────────────────────────────────

    def _action_guard_entries(self, game_state, my_pos):
        """
        Patrol entry points weighted by:
          - opponent belief threat
          - precomputed entry_control_value (BFS-reachable foods from this entry)
          - bonus if this entry is a cluster bottleneck
          - bonus if this entry is adjacent to a choke point
        Logic B: amplify weights during endgame when we are leading.
        """
        food_list       = self.get_food_you_are_defending(game_state).as_list()
        opponents       = self.get_opponents(game_state)
        remaining_moves = game_state.data.timeleft // 4
        projected       = self.get_projected_score(game_state)

        time_mult = (3.0 if remaining_moves < self.d_endgame_thresh
                            and projected > 0 else 1.0)

        # Bottleneck entries: set of entries that guard at least one cluster
        bottleneck_entries = set(self.cluster_bottleneck.values())

        def entry_score(e):
            # Zone-based threat: use each opponent's best position estimate.
            threat = 0.0
            for opp in opponents:
                opp_pos  = self.get_best_opp_pos(opp, game_state)
                d        = self.get_dist(opp_pos, e)
                tier3 = (game_state.get_agent_position(opp) is None and
                         self.last_eaten_age.get(opp, 999) > 15)
                w     = 0.3 if tier3 else 1.0
                if d <= 8:
                    threat += w / (d + 1)

            struct_val = self.entry_control_value.get(
                e,
                sum(1 for f in food_list if self.get_dist(e, f) <= 5))

            bottleneck_bonus = BOTTLENECK_BONUS if e in bottleneck_entries else 0.0

            choke_bonus = 0.0
            if e in self.choke_points:
                choke_bonus = CHOKE_SELF_BONUS
            else:
                x, y = e
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    if (x+dx, y+dy) in self.choke_points:
                        choke_bonus = CHOKE_ADJ_BONUS
                        break

            # Capsule proximity bonus: entries closer to our capsules are
            # structurally more valuable to patrol regardless of current threat.
            my_capsules = self.get_capsules_you_are_defending(game_state)
            if my_capsules:
                cap_dist     = min(self.get_dist(e, cap) for cap in my_capsules)
                capsule_bonus = CAPSULE_GUARD_BONUS / (cap_dist + 1)
            else:
                capsule_bonus = 0.0

            base = threat * (struct_val + 1) + bottleneck_bonus + choke_bonus + capsule_bonus
            return base * time_mult

        if not self.entry_points:
            return self.safe_random_action(game_state)

        target = max(self.entry_points, key=entry_score)
        if target == my_pos:
            sorted_e = sorted(self.entry_points, key=entry_score, reverse=True)
            target   = sorted_e[1] if len(sorted_e) > 1 else None
            if target is None:
                return 'Stop'

        return self.bfs_action(game_state, my_pos, target) \
               or self.safe_random_action(game_state)