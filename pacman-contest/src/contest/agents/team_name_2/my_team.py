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

def create_team(first_index, second_index, is_red,
                first='HybridAgentV3', second='HybridAgentV3', num_training=0):
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
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v >= max_value - 0.1]

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
        return {'successor_score': 100, 'distance_to_food': -10}


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

class SmartHybridAgent(CaptureAgent):
    """
    A reflex agent that switches between offensive and defensive modes.
    """

    def register_initial_state(self, game_state):
        # Call the parent setup
        super().register_initial_state(game_state)
        
        # Cache the boundary coordinates for faster return-to-home calculations
        self.start_pos = game_state.get_agent_state(self.index).get_position()
        self.boundary_coords = self.get_boundary_coords(game_state)

    def get_boundary_coords(self, game_state):
        layout = game_state.data.layout
        width = layout.width
        height = layout.height
        
        # Identify the dividing x-coordinate
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
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can choose to ignore STOP to encourage movement
        # actions.remove(Directions.STOP)

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # 1. COMPUTATIONS
        
        # Enemies
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        # Invaders: Enemies on our side
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        # Defenders: Enemies on their side (Ghosts)
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        # Food
        food_list = self.get_food(successor).as_list()
        
        # Capsules
        capsules = self.get_capsules(successor)

        # 2. FEATURE EXTRACTION
        
        features['successor_score'] = -len(food_list) # Encourage eating food

        # --- OFFENSIVE FEATURES ---
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Distance to capsules (useful if chased)
        if len(capsules) > 0:
            features['distance_to_capsule'] = min([self.get_maze_distance(my_pos, c) for c in capsules])

        # Distance to closest ghost (Defender)
        if len(defenders) > 0:
            min_dist_defender = min([self.get_maze_distance(my_pos, d.get_position()) for d in defenders])
            features['defender_distance'] = min_dist_defender
            if min_dist_defender <= 1:
                features['caught_by_ghost'] = 1 # Huge penalty indicator

        # Logic for returning home
        current_carrying = my_state.num_carrying
        if current_carrying > 0:
            features['carrying_food'] = current_carrying
            min_dist_home = min([self.get_maze_distance(my_pos, b) for b in self.boundary_coords])
            features['distance_to_home'] = min_dist_home

        # --- DEFENSIVE FEATURES ---
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists_to_invader = [self.get_maze_distance(my_pos, i.get_position()) for i in invaders]
            features['invader_distance'] = min(dists_to_invader)

        # --- MOVEMENT PENALTIES ---
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Get basic state info to decide mode
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer == 0]
        
        # Am I scared? (If I ate a capsule recently, I am NOT scared, the enemy is)
        am_i_scared = my_state.scared_timer > 0

        # --- DETERMINE MODE ---
        
        # MODE 1: DEFENSE
        # We defend if there are invaders AND we are not scared.
        if len(invaders) > 0 and not am_i_scared:
            weights = util.Counter()
            weights['num_invaders'] = -1000  # Primary goal: Eliminate invader count
            weights['invader_distance'] = -50 # Run fast towards invader
            weights['stop'] = -100
            weights['reverse'] = -2
            # Small weight to food so we don't freeze if invader is far
            weights['distance_to_food'] = -1 
            return weights

        # MODE 2: OFFENSE (Default)
        weights = util.Counter()
        weights['successor_score'] = 100
        weights['distance_to_food'] = -2
        weights['stop'] = -100
        weights['reverse'] = -2
        weights['captured_by_ghost'] = -10000

        # Dynamic Ghost Avoidance
        dist_to_ghost = 9999
        if len(defenders) > 0:
            dist_to_ghost = min([self.get_maze_distance(my_pos, d.get_position()) for d in defenders])
        
        # If ghost is close, prioritize survival
        if dist_to_ghost <= 5:
            weights['defender_distance'] = 200 # Run away!
            weights['distance_to_capsule'] = -150 # Go for capsule!
            weights['distance_to_food'] = 0 # Forget food for a second
        
        # Logic: When to return home?
        # If we have a lot of food OR time is running out OR we are being chased
        carrying = my_state.num_carrying
        
        if carrying > 0:
            # Base motivation to return grows with amount carrying
            weights['distance_to_home'] = -2 * carrying 

            # If carrying a lot (e.g. 5+), really want to go home
            if carrying > 5:
                weights['distance_to_home'] = -50
                weights['successor_score'] = 0 # Stop eating, just leave
            
            # If ghost is close and we have food, RUSH home or capsule
            if dist_to_ghost < 4:
                weights['distance_to_home'] = -200 
                weights['distance_to_capsule'] = -150

        return weights

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != util.nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        return successor
    

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

# Ghost Avoidance
'''
        if dist_to_ghost <= 4:
            weights['defender_distance'] = 200
            weights['distance_to_capsule'] = -150
            weights['distance_to_food'] = 0 
        if my_state.num_carrying == 0:
            weights['is_pacman'] = 50
        
        # Return Home Logic
        carrying = my_state.num_carrying
        if carrying > 3:
            weights['distance_to_home'] = -5 * carrying 
            if carrying > 5 or (dist_to_ghost < 4 and carrying > 0):
                weights['distance_to_home'] = -500
                weights['successor_score'] = 0 '''


class HybridAgentV2(CaptureAgent):
    """
    The 'Chaos' Hybrid Agent.
    Includes explicit Loop Breaking (Jitter) and Desperation mechanics.
    """
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
        
        # --- NEW: ANTI-LOOP VARIABLES ---
        self.position_history = [] # Tracks last 12 positions
        self.jitter_timer = 0      # Force random moves when > 0
        self.stuck_counter = 0     # Long term stuck counter

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
        
        # --- 1. DETECT LOOPS (The "Dance" Breaker) ---
        self.position_history.append(my_pos)
        if len(self.position_history) > 12:
            self.position_history.pop(0)
            
        # If we have visited this exact spot 3 times in the last 12 moves, we are dancing.
        # Trigger the "Jitter" to break sync.
        if self.position_history.count(my_pos) > 3:
            self.jitter_timer = 5 # Force 5 random moves
            self.position_history = [] # Reset history so we don't trigger immediately again

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        dangerous_ghosts = []
        for a in enemies:
            if not a.is_pacman and a.get_position() is not None and a.scared_timer <= 3:
                dangerous_ghosts.append(a.get_position())

        # --- 2. EXECUTE JITTER (If active) ---
        if self.jitter_timer > 0:
            self.jitter_timer -= 1
            legal_moves = game_state.get_legal_actions(self.index)
            # Pick a random move that DOESN'T kill us immediately
            safe_moves = []
            for action in legal_moves:
                successor = game_state.generate_successor(self.index, action)
                next_pos = successor.get_agent_position(self.index)
                if not self.is_unsafe(next_pos, dangerous_ghosts):
                    safe_moves.append(action)
            
            if safe_moves:
                return random.choice(safe_moves)

        # --- 3. STANDARD LOGIC ---
        
        # Long-term stuck check (for Desperation)
        current_score = self.get_score(game_state)
        # (This is a simplified counter, resets if we eat food or return home)
        if my_state.num_carrying == 0 and my_pos in self.boundary_goals:
            self.stuck_counter = 0 # We just returned, reset
        else:
            self.stuck_counter += 1

        desperation = 0
        if self.stuck_counter > 80: # If 80 moves without returning home
            desperation = min(1.0, (self.stuck_counter - 80) / 100.0)

        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        dist_to_ghost = 9999
        if dangerous_ghosts:
            dist_to_ghost = min([self.get_maze_distance(my_pos, g) for g in dangerous_ghosts])

        # DECISION TREE
        target_goals = []

        # A: SURVIVAL
        if my_state.is_pacman and dist_to_ghost <= 5:
            capsules = self.get_capsules(game_state)
            if capsules:
                closest_cap = min(capsules, key=lambda x: self.get_maze_distance(my_pos, x))
                if self.get_maze_distance(my_pos, closest_cap) < dist_to_ghost:
                    target_goals = [closest_cap]
            if not target_goals:
                target_goals = self.boundary_goals

        # B: DEFENSE
        elif len(invaders) > 0 and my_state.scared_timer == 0:
            target_goals = [i.get_position() for i in invaders]

        # C: RETURN
        elif (my_state.num_carrying > (3 + int(desperation * 5))) or \
             (game_state.data.timeleft < 200 and my_state.num_carrying > 0):
            target_goals = self.boundary_goals

        # D: OFFENSE
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
            if dangerous_ghosts:
                for f in my_food:
                    if f not in self.dead_end_tips:
                        safe_food.append(f)
            else:
                safe_food = my_food

            if not safe_food:
                target_goals = my_food 
            else:
                if desperation > 0.6:
                    # FLANK: Go for furthest food
                    target_goals = sorted(safe_food, key=lambda x: self.get_maze_distance(my_pos, x), reverse=True)[:3]
                elif desperation > 0.2:
                    # DITHER: Random close food
                    sorted_food = sorted(safe_food, key=lambda x: self.get_maze_distance(my_pos, x))
                    target_goals = sorted_food[:4] 
                else:
                    target_goals = safe_food

            capsules = self.get_capsules(game_state)
            if capsules: target_goals += capsules
            if not target_goals: target_goals = [self.patrol_point]

        # EXECUTE A*
        next_action = self.a_star_search(game_state, my_pos, target_goals, dangerous_ghosts, desperation)
        
        if next_action:
            return next_action
        
        # FINAL FALLBACK
        legal_moves = game_state.get_legal_actions(self.index)
        safe_moves = []
        for action in legal_moves:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_position(self.index)
            if not self.is_unsafe(new_pos, dangerous_ghosts):
                safe_moves.append(action)
        if safe_moves: return random.choice(safe_moves)
        return Directions.STOP

    def a_star_search(self, game_state, start_pos, goal_list, dangerous_ghosts, desperation):
        if not goal_list: return None
        
        # Pick a goal
        if desperation > 0.2 and len(goal_list) > 1:
            closest_goal = random.choice(goal_list) 
        else:
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

            if current_pos in goal_set or current_pos == closest_goal:
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
                if self.is_unsafe(next_pos, dangerous_ghosts): continue
                
                if next_pos not in visited:
                    new_path = path + [action]
                    
                    proximity_penalty = 0
                    for ghost in dangerous_ghosts:
                        dist = util.manhattan_distance(next_pos, ghost)
                        if desperation < 0.8:
                            if dist <= 2: proximity_penalty += 20
                            elif dist <= 3: proximity_penalty += 5

                    g_cost = len(new_path) + proximity_penalty
                    h_cost = self.get_maze_distance(next_pos, closest_goal)
                    
                    # --- NEW: ADD NOISE TO BREAK TIES ---
                    # This prevents deterministic pathing when multiple paths are equal length.
                    noise = random.uniform(0.0, 0.5)
                    
                    pq.push((next_pos, new_path), g_cost + h_cost + noise)
                    
        return None

    def is_unsafe(self, pos, dangerous_ghosts):
        for ghost_pos in dangerous_ghosts:
            if util.manhattan_distance(pos, ghost_pos) <= 1:
                return True
        return False

class HybridAgentV3(CaptureAgent):
    """
    V3 Architecture:
    1. Inference Layer: Tracks enemy positions using noisy distance (Sonar).
    2. Strategic Layer: Decides Mode (Attack, Defend, Flee, Flank).
    3. Tactical Layer: A* with dynamic cost maps (Field of Fear).
    """

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.distancer.get_maze_distances()
        
        # --- INIT MAP DATA ---
        self.width = game_state.data.layout.width
        self.height = game_state.data.layout.height
        self.mid_x = int(self.width / 2)
        if self.red: self.mid_x -= 1
        
        # Identify boundary entry points
        self.boundary_goals = []
        for y in range(self.height):
            if not game_state.has_wall(self.mid_x, y):
                self.boundary_goals.append((self.mid_x, y))

        # --- STATE TRACKING ---
        self.position_history = [] 
        self.stuck_counter = 0     
        self.target_block_list = [] # List of (pos, duration) to avoid specific areas
        
        # --- INFERENCE INIT ---
        self.obs_particles = {} # Maps enemy_index -> list of possible positions
        self.opponents = self.get_opponents(game_state)
        for opp in self.opponents:
            self.obs_particles[opp] = []

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        
        # 1. UPDATE INFERENCE (Where are they?)
        self.update_belief_distributions(game_state)
        
        # 2. DETECT STALEMATES & LOOPS
        self.monitor_position_history(my_pos)
        self.update_block_list() # Decay blocked areas

        # 3. DETERMINE MODE
        mode = self.determine_mode(game_state)
        
        # 4. SELECT GOALS BASED ON MODE
        goals = self.select_goals(game_state, mode)
        
        # 5. EXECUTE A* WITH "FIELD OF FEAR"
        # We pass the inferred enemy positions to avoid them
        action = self.a_star_physics(game_state, goals, mode)
        
        return action

    def determine_mode(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        scared_timer = my_state.scared_timer
        
        # Check for visible dangerous ghosts
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        visible_threats = [e for e in enemies if not e.is_pacman and e.get_position() and e.scared_timer <= 5]
        closest_threat_dist = 999
        if visible_threats:
            closest_threat_dist = min([self.get_maze_distance(my_state.get_position(), e.get_position()) for e in visible_threats])

        # MODE: SURVIVAL (Run for life)
        if my_state.is_pacman and closest_threat_dist <= 3:
            return 'SURVIVAL'
        
        # MODE: FLEE (Carrying too much food)
        # If we have lots of food, or time is running out
        food_limit = 5 if self.stuck_counter < 20 else 2 # Get timid if we keep getting stuck
        if my_state.num_carrying >= food_limit or (game_state.data.timeleft < 200 and my_state.num_carrying > 0):
            return 'FLEE'

        # MODE: DEFEND
        # If we are winning significantly, or we see an invader deep in our territory
        score = self.get_score(game_state)
        my_pos = my_state.get_position()
        invaders = [a for a in enemies if a.is_pacman and a.get_position()]
        
        # If invader is visible and close, defend. 
        # But if we are an attacker deep in enemy territory, ignore defense unless urgent.
        if len(invaders) > 0:
            if not my_state.is_pacman: return 'DEFEND' # We are ghost, defend immediately
            if self.get_maze_distance(my_pos, invaders[0].get_position()) < 10: return 'DEFEND'

        # MODE: FLANK (If we detected a stalemate)
        if self.stuck_counter > 5:
            return 'FLANK'

        # MODE: ATTACK (Default)
        return 'ATTACK'

    def select_goals(self, game_state, mode):
        my_pos = game_state.get_agent_position(self.index)
        
        if mode == 'SURVIVAL':
            # Run to nearest capsule or home
            capsules = self.get_capsules(game_state)
            # If capsule is closer than home and reachable, go there
            if capsules: return capsules + self.boundary_goals
            return self.boundary_goals
            
        elif mode == 'FLEE':
            return self.boundary_goals
            
        elif mode == 'DEFEND':
            invaders = [game_state.get_agent_state(i) for i in self.get_opponents(game_state) if game_state.get_agent_state(i).is_pacman]
            visible_invaders = [a.get_position() for a in invaders if a.get_position()]
            if visible_invaders:
                return visible_invaders
            else:
                # Patrol our food
                food_defending = self.get_food_you_are_defending(game_state).as_list()
                if not food_defending: return self.boundary_goals
                # Go to the food cluster
                return [random.choice(food_defending)]

        elif mode == 'FLANK':
            # Pick the boundary goal FURTHEST from current position to force a wide rotation
            return sorted(self.boundary_goals, key=lambda x: self.get_maze_distance(my_pos, x), reverse=True)[:1]

        elif mode == 'ATTACK':
            food = self.get_food(game_state).as_list()
            if not food: return self.boundary_goals
            
            # Simple heuristic: Focus on one half of the board based on agent index to split work
            # But allow crossing over if one side is empty
            half_height = self.height / 2
            my_food = []
            is_top = (self.index % 4 < 2) # Arbitrary split 
            
            for f in food:
                if (is_top and f[1] >= half_height) or (not is_top and f[1] < half_height):
                    my_food.append(f)
            
            target_food = my_food if my_food else food
            
            # Filter out food in "Dead Ends" if a ghost is nearby?
            # For now, just return closest 3 pieces to allow A* to choose best path
            return sorted(target_food, key=lambda x: self.get_maze_distance(my_pos, x))[:3]

        return [self.start]

    def a_star_physics(self, game_state, goals, mode):
        """
        A* that treats the map as a physical field.
        Ghosts emit a 'repulsive force' (high cost).
        Food emits 'attractive force' (goals).
        """
        start_pos = game_state.get_agent_position(self.index)
        
        # Quick exit
        if start_pos in goals: return Directions.STOP
        
        pq = util.PriorityQueue()
        pq.push((start_pos, []), 0)
        visited = set()
        
        # Hard limit to prevent timeout
        nodes_expanded = 0
        
        while not pq.is_empty():
            if nodes_expanded > 500: break # Safety cutoff
            curr_pos, path = pq.pop()
            nodes_expanded += 1
            
            if curr_pos in visited: continue
            visited.add(curr_pos)
            
            if curr_pos in goals:
                return path[0]
            
            # Expand
            legal_actions = []
            if len(path) == 0:
                legal_actions = game_state.get_legal_actions(self.index)
            else:
                # We need to simulate the successor to get legal moves from that spot
                # Simplified: Just check NESW walls
                x, y = int(curr_pos[0]), int(curr_pos[1])
                for action, (dx, dy) in [(Directions.NORTH, (0,1)), (Directions.SOUTH, (0,-1)), (Directions.EAST, (1,0)), (Directions.WEST, (-1,0))]:
                    if not game_state.has_wall(x+dx, y+dy):
                         legal_actions.append(action)

            for action in legal_actions:
                # Calculate successor position
                dx, dy = 0, 0
                if action == Directions.NORTH: dy = 1
                elif action == Directions.SOUTH: dy = -1
                elif action == Directions.EAST: dx = 1
                elif action == Directions.WEST: dx = -1
                next_pos = (int(curr_pos[0] + dx), int(curr_pos[1] + dy))
                
                if next_pos in visited: continue
                
                # --- COST FUNCTION ---
                # Base movement cost
                cost = len(path) + 1
                
                # 1. Threat Cost (The most important part for Stalemate resolution)
                threat_level = self.evaluate_threat(next_pos, game_state)
                
                if mode == 'SURVIVAL': cost += threat_level * 100
                elif mode == 'FLANK':  cost += threat_level * 50
                else:                  cost += threat_level * 10 
                
                # 2. Blocked Area Cost (Anti-Loop)
                for block_pos, _ in self.target_block_list:
                    if next_pos == block_pos:
                        cost += 500 # Virtual Wall

                # 3. Heuristic (Distance to nearest goal)
                h = min([self.get_maze_distance(next_pos, g) for g in goals])
                
                pq.push((next_pos, path + [action]), cost + h)
                
        # Fallback if A* fails
        return self.random_legal_move(game_state)

    def evaluate_threat(self, pos, game_state):
        """
        Calculates how dangerous a position is based on:
        1. Visible Ghosts
        2. Inferred Ghost Positions (Beliefs)
        """
        risk = 0
        
        # A. Visible Ghosts
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        for e in enemies:
            if not e.is_pacman and e.get_position():
                dist = self.get_maze_distance(pos, e.get_position())
                if e.scared_timer > 0: continue # Scared ghosts are safe
                
                if dist <= 1: risk += 9999 # Death
                elif dist <= 2: risk += 50
                elif dist <= 4: risk += 10
        
        # B. Inferred Ghosts (The "Fog of War" Logic)
        # If we can't see them, check our particle filter
        for opp_index in self.obs_particles:
            # If we can see them precisely, we handled it in (A)
            if game_state.get_agent_position(opp_index) is not None: continue
            
            possible_locs = self.obs_particles[opp_index]
            if not possible_locs: continue
            
            # If a significant number of particles are near 'pos', it's risky
            nearby_particles = 0
            for p_loc in possible_locs:
                if util.manhattan_distance(pos, p_loc) <= 2:
                    nearby_particles += 1
            
            prob = nearby_particles / len(possible_locs)
            if prob > 0.1: risk += (prob * 20)
            
        return risk

    def update_belief_distributions(self, game_state):
        """
        Updates belief of where unobserved enemies are based on noisy distance.
        """
        my_pos = game_state.get_agent_position(self.index)
        noisy_distances = game_state.get_agent_distances()
        
        for opp in self.opponents:
            # 1. Exact Position Known?
            opp_pos = game_state.get_agent_position(opp)
            if opp_pos:
                self.obs_particles[opp] = [opp_pos]
                continue
                
            # 2. Update Beliefs
            current_particles = self.obs_particles[opp]
            
            # If we have no idea (start of game or lost track), init with all legal positions
            if not current_particles:
                # Optimization: In real game, iterate all, here we might simplify
                # to just boundary + food positions to save CPU
                current_particles = self.boundary_goals + self.get_food(game_state).as_list()
            
            new_particles = []
            noisy_d = noisy_distances[opp]
            
            # Prune impossible positions based on distance reading
            for p in current_particles:
                # Basic motion model: They could have moved 1 step NESW
                # Since we don't track history perfectly here, we just check likelihood
                true_dist = util.manhattan_distance(my_pos, p)
                # Noisy distance is within +/- 6 of true distance roughly (varies by contest settings)
                # Let's be generous:
                if abs(true_dist - noisy_d) <= 6:
                    new_particles.append(p)
            
            # Resample / Spread (Motion Update)
            # If the list gets too small, we might have over-pruned. Re-inject neighbors.
            final_particles = []
            for p in new_particles:
                x, y = int(p[0]), int(p[1])
                final_particles.append(p)
                # Add neighbors as possibilities (they move!)
                if random.random() < 0.5: # 50% chance they moved
                    neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                    for nx, ny in neighbors:
                        if not game_state.has_wall(nx, ny):
                            final_particles.append((nx, ny))
                            
            # Cap particle count for performance
            if len(final_particles) > 50:
                final_particles = random.sample(final_particles, 50)
                
            self.obs_particles[opp] = final_particles

    def monitor_position_history(self, current_pos):
        self.position_history.append(current_pos)
        if len(self.position_history) > 20:
            self.position_history.pop(0)
            
        # Check for oscillation (visiting same spots repeatedly)
        unique_positions = set(self.position_history)
        if len(self.position_history) == 20 and len(unique_positions) < 6:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
            
        # If deeply stuck, add current area to "Block List" for A*
        if self.stuck_counter > 5:
            # We are stuck. Mark this spot as "Lava" for a while.
            self.target_block_list.append((current_pos, 20)) # 20 moves duration
            self.position_history = [] # Reset history

    def update_block_list(self):
        # Decrement duration of blocked areas
        active_blocks = []
        for pos, duration in self.target_block_list:
            if duration > 0:
                active_blocks.append((pos, duration - 1))
        self.target_block_list = active_blocks

    def random_legal_move(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        return random.choice(actions)