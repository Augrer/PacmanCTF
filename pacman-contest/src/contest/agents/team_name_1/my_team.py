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
import time
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

#def create_team(first_index, second_index, is_red,
#                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
def create_team(first_index, second_index, is_red,
            first='HybridAgent_V2', second='HybridAgent_V2', num_training=0):   
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
    
class PrelimAgent(CaptureAgent):
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
        self.dead_ends = set() # Renamed from dead_end_tips to imply full zones
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        
        # 1. Find the tips first (3 walls)
        candidates = []
        for x in range(width):
            for y in range(height):
                if not game_state.has_wall(x, y):
                    neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                    walls = sum(1 for nx, ny in neighbors if game_state.has_wall(nx, ny))
                    if walls >= 3:
                        self.dead_ends.add((x, y))
                        candidates.append((x, y))

        # 2. Backfill the corridors
        # If a tile has 3 neighbors that are (Walls OR Existing Dead Ends), it is also a Dead End
        while candidates:
            curr = candidates.pop(0)
            neighbors = [(curr[0]+1, curr[1]), (curr[0]-1, curr[1]), (curr[0], curr[1]+1), (curr[0], curr[1]-1)]
            
            for nx, ny in neighbors:
                if game_state.has_wall(nx, ny) or (nx, ny) in self.dead_ends:
                    continue
                
                # Check neighbors of this neighbor
                n_neighbors = [(nx+1, ny), (nx-1, ny), (nx, ny+1), (nx, ny-1)]
                blocked_count = 0
                for nnx, nny in n_neighbors:
                    if game_state.has_wall(nnx, nny) or (nnx, nny) in self.dead_ends:
                        blocked_count += 1
                
                if blocked_count >= 3:
                    self.dead_ends.add((nx, ny))
                    candidates.append((nx, ny))

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

        # --- NEW: SCORE AWARENESS ---
        score = game_state.get_score()
        if not self.red: score = -score # Normalize so positive is good for us

        # If we are winning by a safe margin (e.g., > 2 dots) and time is ticking down
        winning_securely = score > 2 and game_state.data.timeleft < 600
        # Or if we are winning at all in the very last moments
        winning_panic = score > 0 and game_state.data.timeleft < 100
        
        # Override goals to patrol if we just need to survive to win
        if (winning_securely or winning_panic) and not my_state.is_pacman:
             target_goals = [self.patrol_point]
             # Clear other priorities to force defense
             invaders = [] 
             dist_to_ghost = 9999

        
        # 2. STRATEGIC DECISION TREE
        target_goals = []

        # PRIORITY A: SURVIVAL
        if my_state.is_pacman and dist_to_ghost <= 5:
            capsules = self.get_capsules(game_state)
            # Filter capsules: Can I get there before the ghost catches me?
            # We add a buffer of 1 move to be safe
            reachable_capsules = []
            for cap in capsules:
                # Find the closest ghost to this capsule
                closest_ghost_dist = 9999
                for g_pos in dangerous_ghosts:
                    d = self.get_maze_distance(cap, g_pos)
                    if d < closest_ghost_dist: closest_ghost_dist = d
                
                my_dist_to_cap = self.get_maze_distance(my_pos, cap)
                
                # If I can get there faster than the ghost (with small buffer)
                if my_dist_to_cap < closest_ghost_dist:
                    reachable_capsules.append(cap)
            
            if reachable_capsules:
                # Go for the powerup!
                target_goals = reachable_capsules
            else:
                # Run home
                target_goals = self.boundary_goals

        # PRIORITY B: DEFENSE
        elif len(invaders) > 0 and my_state.scared_timer == 0:
            invader_pos_list = [i.get_position() for i in invaders]
            target_goals = invader_pos_list

        # PRIORITY C: RETURN
        dist_to_home = self.get_maze_distance(my_pos, self.patrol_point) # Approx distance to safety
        
        # If we have a lot, OR if we have a little but are right next to safety
        should_return = (my_state.num_carrying > 3) or \
                        (my_state.num_carrying > 0 and dist_to_home <= 2) or \
                        (game_state.data.timeleft < 200 and my_state.num_carrying > 0)
                        
        if should_return:
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
                # FIX: Only fear dead ends if ghost is VERY close (< 4 tiles)
                # If ghost is 5 tiles away, we can safely enter a tunnel
                fear_dead_ends = dist_to_ghost < 4
                
                for f in my_food:
                    if fear_dead_ends and f in self.dead_ends:
                        continue # Skip dangerous food only if ghost is close
                    safe_food.append(f)
                target_goals = safe_food if safe_food else my_food
            else:
                target_goals = my_food

            capsules = self.get_capsules(game_state)
            if capsules: target_goals += capsules
            
            # FIX: If no goals, don't stop. Pick a random border point to stay active.
            if not target_goals: 
                target_goals = [random.choice(self.boundary_goals)]

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
'''
class AStarAgent(CaptureAgent):
    """
    A* Agent with Proportional Home incentives, Dead-End avoidance, 
    aggressive Capsule hunting, and Defensive capabilities.
    """

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        
        # 1. Pre-calculate Home Boundaries
        if self.red:
            boundary_x = game_state.data.layout.width // 2 - 1
        else:
            boundary_x = game_state.data.layout.width // 2
        
        self.boundary_goals = [(boundary_x, y) for y in range(game_state.data.layout.height)
                               if not game_state.has_wall(boundary_x, y)]

        # 2. Pre-calculate Dead-Ends (Simple "Tip" detection)
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

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        
        # 1. ANALYZE GHOSTS
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        dangerous_ghosts = []
        scared_ghosts = []
        
        for enemy in enemies:
            if enemy.get_position():
                if enemy.is_pacman: continue
                if enemy.scared_timer > 5:
                    scared_ghosts.append(enemy.get_position())
                else:
                    dangerous_ghosts.append(enemy.get_position())

        # 2. FILTER TARGETS (The Dead-End Logic)
        all_food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        
        safe_food_list = []
        if len(dangerous_ghosts) > 0:
            for f in all_food:
                if f not in self.dead_end_tips:
                    safe_food_list.append(f)
            if not safe_food_list:
                safe_food_list = all_food
        else:
            safe_food_list = all_food

        # 3. CALCULATE INCENTIVES
        current_load = game_state.get_agent_state(self.index).num_carrying
        
        # A. Food Incentive
        best_food_dist = 9999
        if len(safe_food_list) > 0:
            best_food_dist = min([self.get_maze_distance(my_pos, f) for f in safe_food_list])
        
        food_incentive = 10.0 / max(1, best_food_dist)

        # B. Capsule Incentive
        capsule_incentive = 0
        if len(capsules) > 0:
            best_cap_dist = min([self.get_maze_distance(my_pos, c) for c in capsules])
            base_cap_score = 40.0 
            if len(dangerous_ghosts) > 0:
                base_cap_score = 150.0
            capsule_incentive = base_cap_score / max(1, best_cap_dist)

        # C. Home Incentive
        best_home_dist = min([self.get_maze_distance(my_pos, b) for b in self.boundary_goals])
        
        time_left = game_state.data.timeleft 
        urgency = 1.0
        if time_left < 200: urgency = 5.0 
        
        home_incentive = (current_load * 2.5 * urgency) / max(1, best_home_dist)

        # D. Power Pellet Logic
        if len(scared_ghosts) > 0 and len(dangerous_ghosts) == 0:
            home_incentive = 0 

        # E. Defensive Incentive (FIXED)
        defensive_incentive = 0
        invader_positions = []
        
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            min_invader_dist = min(dists)
            defensive_incentive = 10 / max(1, min_invader_dist)
            # IMPORTANT: We need coordinates for A*, not Agent objects
            invader_positions = [a.get_position() for a in invaders]

        # 4. SELECT GOAL
        goals = []
        
        if capsule_incentive > food_incentive and capsule_incentive > home_incentive:
            goals = capsules
        elif home_incentive > food_incentive:
            goals = self.boundary_goals
        elif defensive_incentive > food_incentive:
            # We go to the invaders
            goals = invader_positions
        else:
            goals = safe_food_list

        # 5. EXECUTE A*
        path = self.a_star_search(game_state, my_pos, goals, dangerous_ghosts)

        if len(path) > 0:
            return path[0]
        
        # 6. FALLBACK
        legal_moves = game_state.get_legal_actions(self.index)
        safe_moves = []
        for action in legal_moves:
            successor = game_state.generate_successor(self.index, action)
            next_pos = successor.get_agent_position(self.index)
            if not self.is_unsafe(next_pos, dangerous_ghosts):
                safe_moves.append(action)
        
        if len(safe_moves) > 0:
            return random.choice(safe_moves)
            
        return Directions.STOP

    def a_star_search(self, game_state, start_pos, goal_list, dangerous_ghosts):
        from util import PriorityQueue
        pq = PriorityQueue()
        pq.push((start_pos, []), 0)
        visited = set()
        goal_set = set(goal_list)

        while not pq.is_empty():
            current_pos, path = pq.pop()

            if current_pos in goal_set:
                return path

            if current_pos in visited:
                continue
            visited.add(current_pos)
            
            if len(path) > 50: continue

            x, y = int(current_pos[0]), int(current_pos[1])
            
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = 0, 0
                if action == Directions.NORTH: dy = 1
                elif action == Directions.SOUTH: dy = -1
                elif action == Directions.EAST: dx = 1
                elif action == Directions.WEST: dx = -1
                
                next_x, next_y = x + dx, y + dy
                next_pos = (next_x, next_y)

                if game_state.has_wall(next_x, next_y): continue
                if self.is_unsafe(next_pos, dangerous_ghosts): continue

                if next_pos not in visited:
                    new_path = path + [action]
                    g_cost = len(new_path)
                    h_cost = self.get_heuristic(next_pos, goal_list)
                    pq.push((next_pos, new_path), g_cost + h_cost)
        return []

    def is_unsafe(self, pos, dangerous_ghosts):
        for ghost_pos in dangerous_ghosts:
            if util.manhattan_distance(pos, ghost_pos) <= 1:
                return True
        return False

    def get_heuristic(self, pos, goal_list):
        if not goal_list: return 0
        return min([util.manhattan_distance(pos, goal) for goal in goal_list])

class StrategicHybridAgent(CaptureAgent):
    """
    A reflex agent that splits defense duties (Top/Bottom) 
    and switches between offense and defense.
    """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.start_pos = game_state.get_agent_state(self.index).get_position()
        self.boundary_coords = self.get_boundary_coords(game_state)
        self.history = []
        
        # --- NEW: DETERMINE ZONE (TOP or BOTTOM) ---
        team_indices = self.get_team(game_state)
        team_indices.sort() # Sort e.g., [1, 3]
        
        # If I am the lower index, I take Bottom. Higher index takes Top.
        if self.index == team_indices[0]:
            self.is_top_defender = False 
        else:
            self.is_top_defender = True
            
        # Pre-calculate my patrol point for when I am idling in defense
        self.patrol_point = self.get_defensive_patrol_point(game_state)

    def get_defensive_patrol_point(self, game_state):
        """
        Finds a point on our boundary to hover around.
        Top agent hovers at 75% height, Bottom agent at 25% height.
        """
        layout = game_state.data.layout
        height = layout.height
        
        # Target Y: 3/4 up for Top, 1/4 up for Bottom
        target_y = int(height * 0.75) if self.is_top_defender else int(height * 0.25)
        
        # Find the boundary coordinate closest to this target Y
        best_pt = self.boundary_coords[0]
        min_diff = 9999
        
        for coord in self.boundary_coords:
            diff = abs(coord[1] - target_y)
            if diff < min_diff:
                min_diff = diff
                best_pt = coord
                
        return best_pt

    def get_boundary_coords(self, game_state):
        layout = game_state.data.layout
        width = layout.width
        height = layout.height
        if self.red:
            x = int(width / 2) - 1
        else:
            x = int(width / 2)
        boundary_coords = []
        for y in range(height):
            if not layout.is_wall((x, y)):
                boundary_coords.append((x, y))
        return boundary_coords

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        start_pos = game_state.get_agent_state(self.index).get_position()
        self.history.append(start_pos)
        if len(self.history) > 5:
            self.history.pop(0)
        return random.choice(best_actions)

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Enemies
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        
        successor_pos = successor.get_agent_state(self.index).get_position()
        if successor_pos in self.history:
            features['reverse'] = 1
        # --- FIX 1: PERSONAL SPACE (AVOID CLUMPING) ---
        # Find where my teammate is
        team_indices = self.get_team(game_state)
        # Assuming 2 agents on a team, find the one that isn't me
        partner_index = [i for i in team_indices if i != self.index][0]
        partner_state = game_state.get_agent_state(partner_index)
        
        if partner_state.get_position():
            partner_pos = partner_state.get_position()
            dist_to_partner = self.get_maze_distance(my_pos, partner_pos)
            # If we are too close (within 2 steps), trigger a penalty feature
            if dist_to_partner <= 2:
                features['clumping'] = 1

        # --- FIX 2: SPLIT THE FOOD MAP ---
        all_food_list = self.get_food(successor).as_list()
        
        # Get board dimensions
        layout_height = game_state.data.layout.height
        mid_height = layout_height / 2
        
        # Filter: Only look at food in MY sector
        my_sector_food = []
        for food in all_food_list:
            food_y = food[1]
            if self.is_top_defender:
                # I am Top Agent: Only care about top food
                if food_y >= mid_height:
                    my_sector_food.append(food)
            else:
                # I am Bottom Agent: Only care about bottom food
                if food_y < mid_height:
                    my_sector_food.append(food)
        
        # FALLBACK: If I ate all food in my sector, help my partner!
        target_food_list = my_sector_food if len(my_sector_food) > 0 else all_food_list

        features['successor_score'] = -len(all_food_list) # Still want to reduce TOTAL food count

        if len(target_food_list) > 0:
            features['distance_to_food'] = min([self.get_maze_distance(my_pos, food) for food in target_food_list])

        # ... (Rest of your normal feature logic below) ...
        capsules = self.get_capsules(successor)
        if len(capsules) > 0:
            features['distance_to_capsule'] = min([self.get_maze_distance(my_pos, c) for c in capsules])

        # Defense Features
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            features['invader_distance'] = min([self.get_maze_distance(my_pos, i.get_position()) for i in invaders])
        
        # Patrol Logic
        features['dist_to_patrol'] = self.get_maze_distance(my_pos, self.patrol_point)

        # Survival
        if len(defenders) > 0:
            dist_to_defender = min([self.get_maze_distance(my_pos, d.get_position()) for d in defenders])
            if dist_to_defender <= 3:
                features['defender_distance'] = dist_to_defender
            else:
                 features['defender_distance'] = 0

        # Return Logic
        if my_state.is_pacman: features['is_pacman'] = 1
        
        if my_state.num_carrying > 2:
            features['distance_to_home'] = min([self.get_maze_distance(my_pos, b) for b in self.boundary_coords])

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        carrying = my_state.num_carrying
        current_state = game_state.get_agent_state(self.index)
        current_carrying = current_state.num_carrying

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer == 0]
        
        am_i_scared = my_state.scared_timer > 0

        # --- DECISION LOGIC ---

        # 1. DEFENSE MODE
        # We enter defense if there are invaders AND we are not scared.
        # OR if we are just patrolling (you can tweak this condition to always defend if winning)
        if len(invaders) > 0 and not am_i_scared:
            weights = util.Counter()
            weights['num_invaders'] = -90000
            weights['stop'] = -250
            weights['reverse'] = -100
            weights['clumping'] = -400
            
            if len(invaders) > 0:
                # ACTIVE DEFENSE: Chase the invader
                weights['invader_distance'] = -500 
            else:
                # PASSIVE DEFENSE: Go to your Zone (Top or Bottom)
                # This ensures they cover the map instead of clumping
                weights['dist_to_patrol'] = -50 
            
            return weights

        # 2. OFFENSE MODE
        weights = util.Counter()
        weights['successor_score'] = 1000
        weights['distance_to_food'] = -25
        weights['stop'] = -20000
        weights['reverse'] = -100
        weights['clumping'] = -200
        dist_to_ghost = 9999
        if len(defenders) > 0:
            dist_to_ghost = min([self.get_maze_distance(my_pos, d.get_position()) for d in defenders])
        
        # CASE 1: HUNGRY (Carrying little, Safe)
        # We are free to explore. We do NOT care about distance to home.
        if current_carrying <= 3 and dist_to_ghost >= 4:
            weights['distance_to_food'] = -3
            weights['distance_to_home'] = -10  
            weights['is_pacman'] = 100       

        # CASE 2: FULL OR HUNTED (Time to leave)
        # We activate the return weight AND disable the food weight.
        elif dist_to_ghost < 4 or current_carrying > 3:
            # Strong pull home
            weights['distance_to_home'] = -150 
            
            # TUNNEL VISION:
            # We set food weights to ZERO. The agent no longer sees the food 
            # behind it, so it won't be tempted to turn around at the border.
            weights['distance_to_food'] = 0 
            weights['successor_score'] = 0 
            weights['is_pacman'] = -10 # Stop paying it to stay

            # Emergency Survival Weights
            if dist_to_ghost <= 4:
                weights['defender_distance'] = 2000
                weights['distance_to_capsule'] = -1000
                if dist_to_ghost <= 3:
                    weights['reverse'] = -1000
        return weights


    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != util.nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor
'''
