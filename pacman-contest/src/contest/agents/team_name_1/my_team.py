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
from typing import List, Tuple
from contest import game
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Queue
from contest.util import nearest_point

import time

#################
# Team creation #
#################

#def create_team(first_index, second_index, is_red,
#                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
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
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1000, 'distance_to_food': -25}


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


class HybridAgent(CaptureAgent):
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        
        # --- INIT 1: MAP SPLITTING ---
        team_indices = self.get_team(game_state)
        team_indices.sort()
        self.is_top_agent = (self.index != team_indices[0])
        
        # --- INIT 2: PRE-CALCULATIONS ---
        if self.red:
            boundary_x = game_state.data.layout.width // 2 - 1
        else:
            boundary_x = game_state.data.layout.width // 2
        
        self.boundary_goals = []
        for y in range(game_state.data.layout.height):
            if not game_state.has_wall(boundary_x, y):
                self.boundary_goals.append((boundary_x, y))

        # Dead-End Detection
        self.dead_end_tips = set()
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        for x in range(width):
            for y in range(height):
                if not game_state.has_wall(x, y):
                    neighbors = 0
                    if game_state.has_wall(x+1, y): neighbors += 1
                    if game_state.has_wall(x-1, y): neighbors += 1
                    if game_state.has_wall(x, y+1): neighbors += 1
                    if game_state.has_wall(x, y-1): neighbors += 1
                    if neighbors >= 3:
                        self.dead_end_tips.add((x, y))

        self.patrol_point = self.calculate_patrol_point(game_state)
        
        # --- NEW: HISTORY FOR LOOP BREAKING ---
        self.recent_positions = [] 

    def calculate_patrol_point(self, game_state):
        layout = game_state.data.layout
        target_y = int(layout.height * 0.75) if self.is_top_agent else int(layout.height * 0.25)
        best_pt = self.boundary_goals[0]
        min_diff = 9999
        for pt in self.boundary_goals:
            diff = abs(pt[1] - target_y)
            if diff < min_diff:
                min_diff = diff
                best_pt = pt
        return best_pt

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        
        # Track history to detect loops
        self.recent_positions.append(my_pos)
        if len(self.recent_positions) > 10:
            self.recent_positions.pop(0)

        # 1. PERCEPTION
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        dangerous_ghosts = []
        for a in enemies:
            if not a.is_pacman and a.get_position() is not None:
                if a.scared_timer <= 3:
                    dangerous_ghosts.append(a.get_position())

        dist_to_ghost = 9999
        if len(dangerous_ghosts) > 0:
            dist_to_ghost = min([self.get_maze_distance(my_pos, g) for g in dangerous_ghosts])

        # 2. STRATEGIC DECISION TREE
        target_goals = []

        # PRIORITY A: SURVIVAL
        if my_state.is_pacman and dist_to_ghost <= 5:
            capsules = self.get_capsules(game_state)
            if capsules:
                closest_cap = min(capsules, key=lambda x: self.get_maze_distance(my_pos, x))
                if self.get_maze_distance(my_pos, closest_cap) < dist_to_ghost:
                    target_goals = [closest_cap]
            
            if not target_goals:
                target_goals = self.boundary_goals

        # PRIORITY B: DEFENSE
        elif len(invaders) > 0 and my_state.scared_timer == 0:
            invader_pos_list = [i.get_position() for i in invaders]
            target_goals = invader_pos_list

        # PRIORITY C: RETURN
        elif (my_state.num_carrying > 3) or (game_state.data.timeleft < 200 and my_state.num_carrying > 0):
            target_goals = self.boundary_goals

        # PRIORITY D: OFFENSE
        else:
            all_food = self.get_food(game_state).as_list()
            layout_height = game_state.data.layout.height
            mid_height = layout_height / 2

            my_food = []
            for f in all_food:
                if self.is_top_agent and f[1] >= mid_height:
                    my_food.append(f)
                elif not self.is_top_agent and f[1] < mid_height:
                    my_food.append(f)
            
            if not my_food: my_food = all_food
            
            safe_food = []
            if len(dangerous_ghosts) > 0:
                for f in my_food:
                    if f not in self.dead_end_tips:
                        safe_food.append(f)
                target_goals = safe_food if safe_food else my_food
            else:
                target_goals = my_food

            capsules = self.get_capsules(game_state)
            if capsules: target_goals += capsules
            if not target_goals: target_goals = [self.patrol_point]

        # 3. EXECUTE A* (With Anti-Dance Logic)
        next_action = self.a_star_search(game_state, my_pos, target_goals, dangerous_ghosts)
        
        if next_action:
            return next_action
        
        # 4. PANIC FALLBACK (If A* returns None, don't just STOP)
        # Pick a random move that doesn't kill us immediately
        legal_moves = game_state.get_legal_actions(self.index)
        safe_moves = []
        for action in legal_moves:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_position(self.index)
            if not self.is_unsafe(new_pos, dangerous_ghosts):
                safe_moves.append(action)
        
        if safe_moves:
            return random.choice(safe_moves)
            
        return Directions.STOP

    def a_star_search(self, game_state, start_pos, goal_list, dangerous_ghosts):
        if not goal_list: return None
        closest_goal = min(goal_list, key=lambda x: self.get_maze_distance(start_pos, x))
        goal_set = set(goal_list)

        from util import PriorityQueue
        pq = PriorityQueue()
        pq.push((start_pos, []), 0)
        visited = set()
        
        nodes_expanded = 0
        limit = 1000 

        while not pq.is_empty():
            if nodes_expanded > limit: break
            current_pos, path = pq.pop()
            nodes_expanded += 1

            if current_pos in goal_set:
                return path[0] if len(path) > 0 else Directions.STOP

            if current_pos in visited: continue
            visited.add(current_pos)

            x, y = int(current_pos[0]), int(current_pos[1])
            neighbors = []
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = 0, 0
                if action == Directions.NORTH: dy = 1
                elif action == Directions.SOUTH: dy = -1
                elif action == Directions.EAST: dx = 1
                elif action == Directions.WEST: dx = -1
                next_x, next_y = x + dx, y + dy
                if not game_state.has_wall(next_x, next_y):
                    neighbors.append(((next_x, next_y), action))

            for next_pos, action in neighbors:
                # 1. HARD CONSTRAINT: Immediate Death
                if self.is_unsafe(next_pos, dangerous_ghosts): continue
                
                if next_pos not in visited:
                    new_path = path + [action]
                    
                    # 2. SOFT CONSTRAINT (The "Fear Factor")
                    # Increase cost if we are "kind of close" (2-3 steps) to a ghost
                    proximity_penalty = 0
                    for ghost in dangerous_ghosts:
                        dist = util.manhattan_distance(next_pos, ghost)
                        if dist <= 2: proximity_penalty += 20 # High cost mud
                        elif dist <= 3: proximity_penalty += 5 # Low cost mud
                    # Boredom Penalty (Anti-Loop)
                    # If we have been here recently, make it expensive to go back
                    if next_pos in self.recent_positions:
                        proximity_penalty += 50
                    g_cost = len(new_path) + proximity_penalty
                    h_cost = self.get_maze_distance(next_pos, closest_goal)
                    
                    pq.push((next_pos, new_path), g_cost + h_cost)
                    
        return None

    def is_unsafe(self, pos, dangerous_ghosts):
        """Returns True ONLY if immediate death (distance <= 1)"""
        for ghost_pos in dangerous_ghosts:
            if util.manhattan_distance(pos, ghost_pos) <= 1:
                return True
        return False
    

class HybridAgent_V2(CaptureAgent):
    """
    Our updated Hybrid agent which now swarms in offense and has different states of being. 
    Also now it only avoids deadends if a ghost is closer than 6 distance away.
    """

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        
        # state init
        self.mode = 'ATTACK'  # Initial mode
        
        # swarm logic
        # we used team indices to deterministically assign roles
        team_indices = self.get_team(game_state)
        team_indices.sort()
        # index 1 is the top agent, 0 bottom
        self.is_top_agent = (self.index != team_indices[0])
        
        # identifying "home" region
        if self.red:
            boundary_x = game_state.data.layout.width // 2 - 1
        else:
            boundary_x = game_state.data.layout.width // 2
        
        self.boundary_goals = []
        for y in range(game_state.data.layout.height):
            if not game_state.has_wall(boundary_x, y):
                self.boundary_goals.append((boundary_x, y))

        # deadend logic
        self.dead_end_tips = set()
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        
        for x in range(width):
            for y in range(height):
                if not game_state.has_wall(x, y):
                    neighbors = 0
                    if game_state.has_wall(x+1, y): neighbors += 1
                    if game_state.has_wall(x-1, y): neighbors += 1
                    if game_state.has_wall(x, y+1): neighbors += 1
                    if game_state.has_wall(x, y-1): neighbors += 1
                    # A cell with 3 walls is the tip of a dead end
                    if neighbors >= 3:
                        self.dead_end_tips.add((x, y))

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        dangerous_ghosts = []
        
        for a in enemies:
            if not a.is_pacman and a.get_position() is not None:
                if a.scared_timer <= 5:
                    dangerous_ghosts.append(a.get_position())

        # nearest enemy
        dist_to_ghost = 9999
        if len(dangerous_ghosts) > 0:
            dist_to_ghost = min([self.get_maze_distance(my_pos, g) for g in dangerous_ghosts])

        # logic for changing states / modes
        
        carry_limit = 6  # greedy limit (collect more before returning) - too high? - testing!
        time_left = game_state.data.timeleft

        # prio1: flee using reflex
        if my_state.is_pacman and dist_to_ghost <= 4:
            self.mode = 'FLEE'
            
        elif self.mode == 'FLEE':
            # if escaped (ghost far), resume mission
            if dist_to_ghost > 6:
                if my_state.num_carrying > 0:
                    self.mode = 'RETURNING'
                else:
                    self.mode = 'ATTACK'
            # if respawn (no longer pacman), reset to attack
            elif not my_state.is_pacman:
                self.mode = 'ATTACK'
                
        elif self.mode == 'ATTACK':
            # do return if full OR time low
            if my_state.num_carrying >= carry_limit or (time_left < 200 and my_state.num_carrying > 0):
                self.mode = 'RETURNING'
                
        elif self.mode == 'RETURNING':
            # after scoring go back to attack
            if my_state.num_carrying == 0:
                self.mode = 'ATTACK'

        target_goals = []

        if self.mode == 'FLEE':
            # focus on capsule to get home 
            capsules = self.get_capsules(game_state)
            if capsules:
                target_goals = capsules
            else:
                target_goals = self.boundary_goals
                
        elif self.mode == 'RETURNING':
            target_goals = self.boundary_goals
            
        elif self.mode == 'ATTACK':
            target_goals = self.get_safe_food(game_state, dangerous_ghosts, dist_to_ghost)
            
            if not target_goals:
                capsules = self.get_capsules(game_state)
                if capsules:
                    target_goals = capsules
                else:
                    # patrol boundary closest to center
                    target_goals = self.boundary_goals

        # A* search! 
        # lower node limit when fleeing to ensure instant reaction
        limit = 400 if self.mode == 'FLEE' else 1500
        next_action = self.a_star_search(game_state, my_pos, target_goals, dangerous_ghosts, limit)
        
        if next_action:
            return next_action
            
        # failsafe - testing had this happen somehow ? -Yannic
        return Directions.STOP

    def get_safe_food(self, game_state, dangerous_ghosts, dist_to_ghost):
        """
        Returns a list of food targets.
        Logic: 
        1. Split map top/bottom based on agent index.
        2. Only filter out 'dead ends' if a ghost is actually nearby.
        """
        all_food = self.get_food(game_state).as_list()
        
        # split food based on y-coordinate
        layout_height = game_state.data.layout.height
        mid_height = layout_height / 2
        
        my_slice = []
        for f in all_food:
            if self.is_top_agent and f[1] >= mid_height:
                my_slice.append(f)
            elif not self.is_top_agent and f[1] < mid_height:
                my_slice.append(f)
        
        # if my side is empty, help teammate 
        if not my_slice:
            my_slice = all_food

        # lazy tunneling: if ghost far, ignore tunnels
        if dist_to_ghost > 6:
            return my_slice

        # if ghost near, avoid dead ends
        safe_food = []
        for f in my_slice:
            if f not in self.dead_end_tips:
                safe_food.append(f)
        
        return safe_food if safe_food else my_slice

    def a_star_search(self, game_state, start_pos, goal_list, dangerous_ghosts, node_limit):
        """
        Standard A* with a node expansion limit to prevent timeouts.
        Includes 'Soft' constraints for ghost proximity.
        """
        if not goal_list: 
            return None
        
        # Heuristic: closest goal by maze distance
        closest_goal = min(goal_list, key=lambda x: self.get_maze_distance(start_pos, x))
        goal_set = set(goal_list)

        pq = util.PriorityQueue()
        pq.push((start_pos, []), 0)
        visited = set()
        
        nodes_expanded = 0

        while not pq.is_empty():
            if nodes_expanded > node_limit:
                break 
                
            current_pos, path = pq.pop()
            nodes_expanded += 1

            # check for goall
            if current_pos in goal_set:
                return path[0] if len(path) > 0 else Directions.STOP

            if current_pos in visited: continue
            visited.add(current_pos)

            # generate successor states
            x, y = int(current_pos[0]), int(current_pos[1])
            neighbors = []
            
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = 0, 0
                if action == Directions.NORTH: dy = 1
                elif action == Directions.SOUTH: dy = -1
                elif action == Directions.EAST: dx = 1
                elif action == Directions.WEST: dx = -1
                
                next_x, next_y = x + dx, y + dy
                
                if not game_state.has_wall(next_x, next_y):
                    neighbors.append(((next_x, next_y), action))

            for next_pos, action in neighbors:
                is_death = False
                for g in dangerous_ghosts:
                    if util.manhattan_distance(next_pos, g) <= 1:
                        is_death = True
                
                if is_death: continue
                
                if next_pos not in visited:
                    new_path = path + [action]
                    
                    g_cost = len(new_path)
                    
                    # ""fear""
                    # penalty if path gets  kinda close to ghost (2-3 dist)
                    for g in dangerous_ghosts:
                        d = self.get_maze_distance(next_pos, g)
                        if d <= 3: 
                            g_cost += 10 
                    
                    # heuristiccost
                    h_cost = self.get_maze_distance(next_pos, closest_goal)
                    
                    pq.push((next_pos, new_path), g_cost + h_cost)
                    
        return None
