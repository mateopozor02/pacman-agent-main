# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from game import Actions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


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
        width = game_state.data.layout.width
        height = game_state.data.layout.height

        if self.red:
            border_x = width // 2 - 1
        else:
            border_x = width // 2

        self.border = []

        for y in range(height):
            if not game_state.get_walls()[border_x][y]:
                self.border.append((border_x, y))

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
        
    def a_star_search(self, game_state, goal_positions, avoid_positions=None):
        """
        A* pathfinding from current position to one of the goal positions.
        """
        from util import PriorityQueue

        if not goal_positions:
            return [] 

        start = game_state.get_agent_state(self.index).get_position()
        walls = game_state.get_walls()

        frontier = PriorityQueue()
        frontier.push((start, []), 0)

        visited = set()

        while not frontier.is_empty():
            position, path = frontier.pop()

            if position in visited:
                continue
            visited.add(position)

            if position in goal_positions:
                return path

            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                x, y = position
                dx, dy = Actions.direction_to_vector(action)
                nextx, nexty = int(x + dx), int(y + dy)

                if walls[nextx][nexty]:
                    continue

                next_pos = (nextx, nexty)

                new_path = path + [action]

                cost = len(new_path)

                # Ghost avoidance
                if avoid_positions:
                    for ghost in avoid_positions:
                        if self.get_maze_distance(next_pos, ghost) <= 2:
                            cost += 10

                heuristic = min(self.get_maze_distance(next_pos, g) for g in goal_positions)

                frontier.push((next_pos, new_path), cost + heuristic)

        return []

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
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features
    
    def choose_action(self, game_state):

        my_pos = game_state.get_agent_state(self.index).get_position()

        food_list = self.get_food(game_state).as_list()

        carrying = game_state.get_agent_state(self.index).num_carrying

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() != None]

        ghost_positions = [g.get_position() for g in ghosts]
        
        # If scared ghost close → chase
        scared_ghosts = [g for g in ghosts if g.scared_timer > 1]
        if scared_ghosts:
            scared_positions = [g.get_position() for g in scared_ghosts]
            path = self.a_star_search(game_state, scared_positions)
            if path:
                return path[0]
        # If ghost close → escape
        for g in ghosts:
            if self.get_maze_distance(my_pos, g.get_position()) < 4:
                path = self.a_star_search(game_state, self.border, ghost_positions)
                if path:
                    return path[0]

        # Calculate danger level based on nearby ghosts, and decide how much food is collected.
        danger_level = len([g for g in ghosts if self.get_maze_distance(my_pos, g.get_position()) < 6])
        return_threshold = 4 - danger_level  # Return sooner if ghosts are nearby

        if carrying >= max(2, return_threshold):  # At least 2, less if danger
            path = self.a_star_search(game_state, self.border, ghost_positions)
            if path:
                return path[0]
            


        capsules = self.get_capsules(game_state)

        if capsules:
            for g in ghosts:
                if self.get_maze_distance(my_pos, g.get_position()) < 5:
                    path = self.a_star_search(game_state, capsules)
                    if path:
                        return path[0]

        # Otherwise go to food
        path = self.a_star_search(game_state, food_list, ghost_positions)

        if path:
            return path[0]

        actions = game_state.get_legal_actions(self.index)
        actions = [a for a in actions if a != Directions.STOP]
        return random.choice(actions)


    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        # Track when enemies ate power capsules
        self.enemy_super_timers = {}
        
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

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
    
    def get_vulnerable_food(self, game_state):
        """
        Find the most vulnerable food on our side - the one closest to the border
        """
        my_food = self.get_food_you_are_defending(game_state).as_list()
        
        if not my_food:
            return None
        
        # For each food, calculate how close it is to the border
        food_vulnerability = []
        for food in my_food:
            # Find the closest border point to this food
            min_dist_to_border = min([self.get_maze_distance(food, border) for border in self.border])
            
            # Also consider distance to our current position (we want to be efficient)
            my_pos = game_state.get_agent_state(self.index).get_position()
            dist_to_me = self.get_maze_distance(my_pos, food)
            
            # Vulnerability score: lower distance to border = MORE vulnerable
            # We also factor in how far we are (don't cross entire map if not needed)
            vulnerability_score = (10 - min_dist_to_border) * 3  # Higher score = more vulnerable
            
            food_vulnerability.append((food, vulnerability_score, min_dist_to_border))
        
        # Sort by vulnerability (highest first) and return the most vulnerable food
        food_vulnerability.sort(key=lambda x: x[1], reverse=True)
        
        # For debugging if needed
        # print(f"Most vulnerable food: {food_vulnerability[0][0]} with score {food_vulnerability[0][1]}")
        
        return food_vulnerability[0][0]
    
    def choose_action(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        distances = game_state.get_agent_distances()
        opponents = self.get_opponents(game_state)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None 
                   and a.scared_timer == 0]
        super_enemies = []
        for enemy in enemies:
            if enemy.get_position() is not None:
                # If enemy is a Pacman (on our side) and has scared_timer > 0, they're SUPER!
                if enemy.is_pacman and enemy.scared_timer > 0:
                    super_enemies.append(enemy)
                    self.enemy_super_timers[enemy] = enemy.scared_timer
        
        # PRIORITY 1: RUN AWAY FROM SUPER ENEMIES! 
        if super_enemies:
            # Find the closest super enemy
            closest_super = min(super_enemies, 
                              key=lambda e: self.get_maze_distance(my_pos, e.get_position()))
            dist_to_super = self.get_maze_distance(my_pos, closest_super.get_position())
            
            # If they're close enough to be a threat
            if dist_to_super < 5:  # They're a threat!
                # Get all legal actions
                actions = game_state.get_legal_actions(self.index)
                
                best_action = None
                best_distance = -float('inf')  # We want to MAXIMIZE distance from super enemy
                
                for action in actions:
                    if action == Directions.STOP:
                        continue  # Never stop when running!
                    
                    # See where this action would take us
                    successor = self.get_successor(game_state, action)
                    new_pos = successor.get_agent_state(self.index).get_position()
                    
                    # Calculate how far this move would put us from the super enemy
                    new_distance = self.get_maze_distance(new_pos, closest_super.get_position())
                    total_score = new_distance
                    
                    if total_score > best_distance:
                        best_distance = total_score
                        best_action = action
                
                if best_action:
                    return best_action
                
                # If no good action found, just run in any direction that's not toward the super enemy
                for action in actions:
                    if action == Directions.STOP:
                        continue
                    successor = self.get_successor(game_state, action)
                    new_pos = successor.get_agent_state(self.index).get_position()
                    if self.get_maze_distance(new_pos, closest_super.get_position()) > dist_to_super:
                        return action
            
            # If all else fails, just keep moving - don't stop!
            non_stop_actions = [a for a in game_state.get_legal_actions(self.index) if a != Directions.STOP]
            if non_stop_actions:
                return random.choice(non_stop_actions)
        
        # Clean up old timers
        self.enemy_super_timers = {e: timer for e, timer in self.enemy_super_timers.items() 
                                   if timer > 0}
        

        # PRIORITY 2: Chase visible invader (if they're not super)
        if len(invaders) > 0:
            invader_positions = [a.get_position() for a in invaders]
            path = self.a_star_search(game_state, invader_positions)

            if path:
                return path[0]
            
        nearby_invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None 
                            and self.get_maze_distance(my_pos, a.get_position()) < 5]
        if nearby_invaders:
            nearby_positions = [a.get_position() for a in nearby_invaders]
            path = self.a_star_search(game_state, nearby_positions)
            if path:
                return path[0]
        # PRIORITY 3: No visible invaders, but check for NOISY enemies (6-15 steps)
        noisy_enemies = []
        for opp in opponents:
            opp_state = game_state.get_agent_state(opp)
            if opp_state.get_position() is None:  # Not visible
                noise_dist = distances[opp]
                if 6 <= noise_dist <= 15:  # In the noisy but close range
                    noisy_enemies.append((opp, noise_dist))

        if noisy_enemies:
            # Sort by closest noise first
            noisy_enemies.sort(key=lambda x: x[1])
            closest_noise_dist = noisy_enemies[0][1]
            
            # Find border points that are about this distance away (intercept points)
            intercept_points = []
            for border_point in self.border:
                dist_to_border = self.get_maze_distance(my_pos, border_point)
                # If this border point is roughly the same distance as the noise,
                # the enemy could be heading here!
                if abs(dist_to_border - closest_noise_dist) <= 2:
                    intercept_points.append(border_point)
            
            if intercept_points:
                # Move to cut off the most likely crossing point
                path = self.a_star_search(game_state, intercept_points)
                if path:
                    return path[0]
            
            # If no clear intercept point, move toward the general area
            # by heading to the border point closest to the noise distance
            closest_border = min(self.border, 
                                key=lambda b: abs(self.get_maze_distance(my_pos, b) - closest_noise_dist))
            path = self.a_star_search(game_state, [closest_border])
            if path:
                return path[0]
            
        # PRIORITY 4: No invaders - patrol toward the MOST VULNERABLE FOOD!
        vulnerable_food = self.get_vulnerable_food(game_state)
        
        if vulnerable_food:
            # Go toward the most vulnerable food (closest to border)
            path = self.a_star_search(game_state, [vulnerable_food])
            if path:
                return path[0]
        

        # Last resort: keep moving!
        non_stop_actions = [a for a in game_state.get_legal_actions(self.index) if a != Directions.STOP]
        if non_stop_actions:
            return random.choice(non_stop_actions)
        
        return random.choice(game_state.get_legal_actions(self.index))

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -40, 'stop': -100, 'reverse': -2}