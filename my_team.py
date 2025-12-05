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
from util import nearest_point


#################
# Team creation #
#################

agent = 'HybridAgent_V2'

#def create_team(first_index, second_index, is_red,
#                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
def create_team(first_index, second_index, is_red,
            first=agent, second=agent, num_training=0):   
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

class ForeskinAgent(CaptureAgent):
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
 
'''class HybridAgent_Bus(CaptureAgent):
    """
    Hybrid Agent V3: Swarm Attacker with active "Park the Bus" Patrol.
    Fixes 'staring' issues by forcing dynamic patrol movement.
    """

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        
        # State initialization
        self.mode = 'ATTACK'
        self.patrol_target = None  # New: Persist patrol target to stop jittering
        
        # Swarm/Split logic
        team_indices = self.get_team(game_state)
        team_indices.sort()
        self.is_top_agent = (self.index != team_indices[0])
        
        # Layout analysis
        self.width = game_state.data.layout.width
        self.height = game_state.data.layout.height
        
        # Identify home boundary
        if self.red:
            boundary_x = (self.width // 2) - 1
        else:
            boundary_x = (self.width // 2)
        
        self.boundary_goals = []
        for y in range(self.height):
            if not game_state.has_wall(boundary_x, y):
                self.boundary_goals.append((boundary_x, y))

        # Split patrol zones
        mid_height = self.height / 2
        self.my_patrol_zone = []
        for bg in self.boundary_goals:
            if self.is_top_agent and bg[1] >= mid_height:
                self.my_patrol_zone.append(bg)
            elif not self.is_top_agent and bg[1] < mid_height:
                self.my_patrol_zone.append(bg)
                
        if not self.my_patrol_zone:
            self.my_patrol_zone = self.boundary_goals

        # Dead-end detection
        self.dead_end_tips = set()
        for x in range(self.width):
            for y in range(self.height):
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
        my_state = game_state.get_agent_state(self.index)
        time_left = game_state.data.timeleft
        
        # --- 1. Strategic Assessment ---
        current_score = self.get_score(game_state)
        if not self.red: 
            current_score = current_score
            
        # Trigger: Lead by 8 late game, OR lead by 12+ anytime
        park_the_bus_active = (current_score >= 4 and time_left < 600) or (current_score >= 4)
        
        # Analyze enemies
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        dangerous_defenders = []
        active_invaders = []
        
        for a in enemies:
            if a.get_position():
                if not a.is_pacman and a.scared_timer <= 5:
                    dangerous_defenders.append(a.get_position())
                if a.is_pacman:
                    active_invaders.append(a.get_position())

        dist_to_defender = 9999
        if dangerous_defenders:
            dist_to_defender = min([self.get_maze_distance(my_pos, g) for g in dangerous_defenders])

        # --- 2. Mode Switching ---
        
        carry_limit = 6
        
        # Flee Override
        if my_state.is_pacman and dist_to_defender <= 4:
            self.mode = 'FLEE'
            
        elif self.mode == 'FLEE':
            if dist_to_defender > 6: # Safety margin reached
                if park_the_bus_active and not my_state.is_pacman:
                    self.mode = 'PARK_BUS'
                elif my_state.num_carrying > 0:
                    self.mode = 'RETURNING'
                else:
                    self.mode = 'ATTACK'
            elif not my_state.is_pacman: # Died or returned
                self.mode = 'PARK_BUS' if park_the_bus_active else 'ATTACK'
        
        # Strategy Switching
        elif park_the_bus_active:
            # If we are scared, we can't defend effectively, so play safe/avoid
            if my_state.scared_timer > 0:
                 self.mode = 'PARK_BUS' # Still park, but logic below handles scared behavior
            else:
                self.mode = 'PARK_BUS'
                
        elif self.mode == 'ATTACK':
            if my_state.num_carrying >= carry_limit or (time_left < 200 and my_state.num_carrying > 0):
                self.mode = 'RETURNING'
            # Check if we should switch to defense due to score change
            if park_the_bus_active:
                self.mode = 'PARK_BUS'
                
        elif self.mode == 'RETURNING':
            if my_state.num_carrying == 0:
                self.mode = 'PARK_BUS' if park_the_bus_active else 'ATTACK'

        # --- 3. Target Selection ---
        
        target_goals = []
        search_node_limit = 1500
        avoid_list = dangerous_defenders

        if self.mode == 'FLEE':
            search_node_limit = 400
            capsules = self.get_capsules(game_state)
            target_goals = capsules if capsules else self.boundary_goals
            
        elif self.mode == 'PARK_BUS':
            # Priority 1: Hunting Invaders
            if active_invaders:
                target_goals = active_invaders
                self.patrol_target = None # Reset patrol when hunting
                # If we are scared, we must avoid the invader (who can eat us)
                if my_state.scared_timer > 0:
                    avoid_list = active_invaders
                    target_goals = self.boundary_goals # Run to boundary
                else:
                    avoid_list = [] # We are the hunter, do not fear ghosts (unless they are on their side)
            
            # Priority 2: Patrol Boundary
            else:
                # If we have a valid patrol target that we haven't reached yet
                if self.patrol_target and my_pos != self.patrol_target:
                    target_goals = [self.patrol_target]
                else:
                    # Pick a new random target from our zone to keep moving
                    self.patrol_target = random.choice(self.my_patrol_zone)
                    target_goals = [self.patrol_target]

        elif self.mode == 'RETURNING':
            target_goals = self.boundary_goals
            
        elif self.mode == 'ATTACK':
            target_goals = self.get_safe_food(game_state, dangerous_defenders, dist_to_defender)
            if not target_goals:
                capsules = self.get_capsules(game_state)
                target_goals = capsules if capsules else self.boundary_goals

        # --- 4. Execution ---
        
        next_action = self.a_star_search(game_state, my_pos, target_goals, avoid_list, search_node_limit)
        
        # Anti-Stuck / Anti-Stare Fallback
        # If A* returns None or STOP (and we aren't at the exact goal), make a random move
        if next_action is None or next_action == Directions.STOP:
             # Don't stop unless we really want to (e.g. at patrol point). 
             # But here we force movement to avoid staring matches.
             legal_moves = game_state.get_legal_actions(self.index)
             legal_moves = [a for a in legal_moves if a != Directions.STOP]
             if legal_moves:
                 return random.choice(legal_moves)
        
        return next_action if next_action else Directions.STOP

    def get_safe_food(self, game_state, dangerous_ghosts, dist_to_ghost):
        all_food = self.get_food(game_state).as_list()
        mid_height = self.height / 2
        
        my_slice = []
        for f in all_food:
            if self.is_top_agent and f[1] >= mid_height:
                my_slice.append(f)
            elif not self.is_top_agent and f[1] < mid_height:
                my_slice.append(f)
        
        if not my_slice: my_slice = all_food
        if dist_to_ghost > 6: return my_slice

        safe_food = []
        for f in my_slice:
            if f not in self.dead_end_tips:
                safe_food.append(f)
        return safe_food if safe_food else my_slice

    def a_star_search(self, game_state, start_pos, goal_list, avoid_positions, node_limit):
        if not goal_list: return None
        
        closest_goal = min(goal_list, key=lambda x: self.get_maze_distance(start_pos, x))
        goal_set = set(goal_list)

        pq = util.PriorityQueue()
        pq.push((start_pos, []), 0)
        visited = set()
        nodes_expanded = 0

        while not pq.is_empty():
            if nodes_expanded > node_limit: break 
            current_pos, path = pq.pop()
            nodes_expanded += 1

            if current_pos in goal_set:
                return path[0] if len(path) > 0 else Directions.STOP

            if current_pos in visited: continue
            visited.add(current_pos)

            x, y = int(current_pos[0]), int(current_pos[1])
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = 0, 0
                if action == Directions.NORTH: dy = 1
                elif action == Directions.SOUTH: dy = -1
                elif action == Directions.EAST: dx = 1
                elif action == Directions.WEST: dx = -1
                
                next_x, next_y = x + dx, y + dy
                if not game_state.has_wall(next_x, next_y):
                    next_pos = (next_x, next_y)
                    
                    # HARD Constraint: Instant Death
                    is_death = False
                    for g in avoid_positions:
                        if util.manhattan_distance(next_pos, g) <= 1:
                            is_death = True
                    if is_death: continue
                    
                    if next_pos not in visited:
                        new_path = path + [action]
                        g_cost = len(new_path)
                        
                        # SOFT Constraint: Fear
                        for g in avoid_positions:
                            d = self.get_maze_distance(next_pos, g)
                            if d <= 3: g_cost += 10
                        
                        h_cost = self.get_maze_distance(next_pos, closest_goal)
                        pq.push((next_pos, new_path), g_cost + h_cost)
        return None
'''    
        
'''class PenBlocker(CaptureAgent):
    """
    A Specialized Companion Agent.
    Primary: Tethered Guard of Power Pellets.
    Secondary: If no pellets, standard defense or quick offense.
    """

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        
        # Identify Boundary
        layout = game_state.data.layout
        self.width = layout.width
        self.height = layout.height
        
        if self.red:
            self.boundary_x = (self.width // 2) - 1
            self.min_x, self.max_x = 0, self.boundary_x
        else:
            self.boundary_x = (self.width // 2)
            self.min_x, self.max_x = self.boundary_x, self.width - 1
        
        self.boundary_points = []
        for y in range(self.height):
            if not game_state.has_wall(self.boundary_x, y):
                self.boundary_points.append((self.boundary_x, y))

        # Safe spots for retreating
        self.retreat_points = [self.start]
        candidates = [
            (self.min_x + 1, 1), 
            (self.min_x + 1, self.height - 2),
            (self.max_x - 1, 1), 
            (self.max_x - 1, self.height - 2)
        ]
        for p in candidates:
            if not game_state.has_wall(p[0], p[1]):
                self.retreat_points.append(p)

        self.last_food_list = self.get_food_you_are_defending(game_state).as_list()
        self.inferred_invader_pos = None

    def get_safest_retreat_pos(self, game_state, enemies):
        invader_positions = [a.get_position() for a in enemies if a.get_position() is not None]
        if not invader_positions:
            return self.start 
            
        best_spot = self.start
        max_safety_distance = -1
        
        for spot in self.retreat_points:
            dist_to_nearest_enemy = min([util.manhattan_distance(spot, i_pos) for i_pos in invader_positions])
            if dist_to_nearest_enemy > max_safety_distance:
                max_safety_distance = dist_to_nearest_enemy
                best_spot = spot     
        return best_spot

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        current_score = self.get_score(game_state)
        my_state = game_state.get_agent_state(self.index)
        
        # 1. PERCEPTION
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        visible_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        
        curr_food = self.get_food_you_are_defending(game_state).as_list()
        if len(curr_food) < len(self.last_food_list):
            eaten = set(self.last_food_list) - set(curr_food)
            if eaten: self.inferred_invader_pos = list(eaten)[0]
        self.last_food_list = curr_food
        
        if self.inferred_invader_pos and util.manhattan_distance(my_pos, self.inferred_invader_pos) < 2:
            self.inferred_invader_pos = None

        # 2. DECIDE MODE
        target_pos = None
        mode = 'WAIT'

        capsules = self.get_capsules_you_are_defending(game_state)

        # A. SCARED -> RETREAT INTELLIGENTLY
        if my_state.scared_timer > 0:
            mode = 'RETREAT'
            target_pos = self.get_safest_retreat_pos(game_state, enemies)

        # B. HAS CAPSULES -> TETHERED DEFENSE (Priority 1)
        elif len(capsules) > 0:
            closest_cap = min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
            
            # Check for invaders in the "Kill Zone" (range 5)
            closest_invader = None
            min_inv_dist = 6 # Threshold (must be < 6)
            
            for inv in invaders:
                dist = self.get_maze_distance(my_pos, inv.get_position())
                if dist < min_inv_dist:
                    min_inv_dist = dist
                    closest_invader = inv
            
            if closest_invader:
                # Invader is close! Bite them.
                mode = 'KILL_ZONE'
                target_pos = closest_invader.get_position()
            else:
                # No one close? Sit on the pellet.
                mode = 'GUARD_PELLET'
                target_pos = closest_cap

        # C. NO CAPSULES -> STANDARD LOGIC (Unlimited Chase)
        elif len(invaders) > 0:
            mode = 'KILL'
            target_pos = min([a.get_position() for a in invaders], 
                             key=lambda p: self.get_maze_distance(my_pos, p))

        # D. INFERRED INVADER -> HUNT
        elif self.inferred_invader_pos is not None:
            mode = 'HUNT'
            target_pos = self.inferred_invader_pos

        # E. DEFAULT -> OFFENSE / PATROL
        else:
            if current_score > 0:
                mode = 'PATROL'
                mid_y = game_state.data.layout.height // 2
                target_pos = min(self.boundary_points, key=lambda p: abs(p[1] - mid_y))
            else:
                if my_state.num_carrying > 0:
                    mode = 'RETURNING'
                    target_pos = self.start
                else:
                    mode = 'ATTACK'
                    food_list = self.get_food(game_state).as_list()
                    if food_list:
                        target_pos = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
                    else:
                        target_pos = self.boundary_points[0]

        # 3. EXECUTE A*
        dangerous_ghosts = []
        if mode in ['ATTACK', 'RETURNING']:
             dangerous_ghosts = [g.get_position() for g in visible_ghosts if g.scared_timer <= 5]

        action = self.a_star_search(game_state, my_pos, target_pos, dangerous_ghosts, mode)
        return action

    def a_star_search(self, game_state, start_pos, target_pos, dangerous_ghosts, mode):
        if target_pos is None: return Directions.STOP
        
        pq = util.PriorityQueue()
        pq.push((start_pos, []), 0)
        visited = set()
        
        limit = 1000
        expanded = 0
        
        while not pq.is_empty():
            if expanded > limit: break
            expanded += 1
            curr, path = pq.pop()
            
            if curr == target_pos:
                return path[0] if path else Directions.STOP
                
            # Defense Optimization: If adjacent to target (invader) and can kill
            if mode in ['KILL', 'KILL_ZONE'] and util.manhattan_distance(curr, target_pos) <= 1:
                 return path[0] if path else Directions.STOP

            if curr in visited: continue
            visited.add(curr)
            
            x, y = int(curr[0]), int(curr[1])
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = 0, 0
                if action == Directions.NORTH: dy = 1
                elif action == Directions.SOUTH: dy = -1
                elif action == Directions.EAST: dx = 1
                elif action == Directions.WEST: dx = -1
                
                nx, ny = x + dx, y + dy
                npos = (nx, ny)
                
                if not game_state.has_wall(nx, ny):
                    is_safe = True
                    for g in dangerous_ghosts:
                        if util.manhattan_distance(npos, g) <= 1: is_safe = False
                    
                    if is_safe:
                        new_path = path + [action]
                        g_cost = len(new_path)
                        h_cost = self.get_maze_distance(npos, target_pos)
                        pq.push((npos, new_path), g_cost + h_cost)
                        
        return Directions.STOP'''