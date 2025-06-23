import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
import os
from collections import namedtuple

from MAPPO_models import MAPPO_model
from MAPPO_obs import (
    ReplayBuffer,
    get_state,
    get_ship_state,
    update_relic_discovery_logic,
    update_reward_logic,
    get_discovery_horizon,
    discovery_ch_idx,
    reward_ch_idx,
    astroid_ch_idx,
    tile_energy_ch_idx,
    friendly_ch_idx,
    friendly_energy_ch_idx,
    enemy_ch_idx,
    enemy_energy_ch_idx,
    relic_ch_idx,
)

Experience = namedtuple('Experience', [
    'state',
    'additional_features',
    'actions', 
    'log_probs', 
    'team_points', # comes from step t+1
    'opp_team_points', # comes from step t+1
    'points_increase', # comes from step t+1
    'opp_points_increase', # comes from step t+1
    'value', 
    'done', 
    #'next_state', # comes from step t+1
    'valid_ships',
    'ship_states',
    'ship_rewards', 
])
channel_names = [
    #"Visible",
    "Discovery",
    "Reward status",
    #"Relic status",
    #"Nebula type",
    "Astroid type",
    "Tile energy",
    "Friendly ship density",
    "Friendly ship energy",
    "Enemy ship density",
    "Enemy ship energy"
]

class MAPPO_agent:
    # inspo - https://github.com/jason-huang03/mappo-warmup/blob/master/on-policy/onpolicy/algorithms/r_mappo/r_mappo.py
    def __init__(self, player: str, env_cfg, device="cpu", 
                 train_mode=True, buffer_size=505, action_dim=5, 
                 lr=2e-4, gamma=0.99, lam=0.95, eps_clip=0.1, 
                 batch_size=128, epochs=16, entropy_coef=0.01, 
                 value_coef=0.5, init_orthogonal=True,
                 state_shape=(9, 24, 24), ship_state_shape=(10, 21, 21), relicbound_warmup=0, 
                 temperature=None, use_clipped_critic_loss=False,
                 LR_scheduler=False, lr_min=1e-4, LR_num_steps=1000, rollout=1,
                 match_bonus=0, episode_bonus=0, save_dir=None, save_freq=9999999,
                 discovery_reward=0.1, reward_tile_reward=1.0,
                 use_multiple_actors=True, continue_training_path=None, move_onto_potential_reward_tile=0.1,
                 penalty_moving_off_reward_tile=-0.1, penalty_moving_off_potential_reward_tile=-0.05,
                 penalty_full_energy=-0.1, penalty_out_of_bounds=-0.1, penalty_asteroid=-0.1,
                 tile_energy_coefficient=0.1, behind_enemy_lines_coefficient=2.0, penalty_moving_cost=-0.05,
                 penalty_sap_cost=-0.1, rewards_divisor_factor=1.0, per_agent_normalize_advantages=True,
                 use_coord_channels=False, use_center_region_extraction_branch=False, use_nearby_features=False,
                 use_entropy_scheduler=False, min_entropy_coef=0.0, entropy_coef_step=0.0, num_updates=0, decay_rate=0.002): 
        self.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        self.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        self.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        self.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]
        # hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.batch_size = batch_size
        self.epochs = epochs
        self.buffer_size = buffer_size
        self.device = device
        self.train_mode = train_mode
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.init_orthogonal = init_orthogonal
        self.state_shape = state_shape
        self.ship_state_shape = ship_state_shape
        self.relicbound_warmup = relicbound_warmup
        self.temperature = temperature 
        self.use_clipped_critic_loss = use_clipped_critic_loss
        self.rollout = rollout
        self.match_bonus = match_bonus
        self.episode_bonus = episode_bonus
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.discovery_reward = discovery_reward
        self.reward_tile_reward = reward_tile_reward
        self.use_multiple_actors = use_multiple_actors
        self.move_onto_potential_reward_tile =  move_onto_potential_reward_tile
        self.penalty_moving_off_reward_tile = penalty_moving_off_reward_tile
        self.penalty_moving_off_potential_reward_tile = penalty_moving_off_potential_reward_tile
        self.penalty_full_energy = penalty_full_energy
        self.penalty_out_of_bounds = penalty_out_of_bounds
        self.penalty_asteroid = penalty_asteroid
        self.tile_energy_coefficient = tile_energy_coefficient
        self.behind_enemy_lines_coefficient = behind_enemy_lines_coefficient
        self.rewards_divisor_factor = rewards_divisor_factor
        self.penalty_moving_cost = penalty_moving_cost  * self.UNIT_MOVE_COST #penalty
        self.penalty_sap_cost = penalty_sap_cost * self.UNIT_SAP_COST  #penalty
        self.per_agent_normalize_advantages = per_agent_normalize_advantages
        self.use_coord_channels = use_coord_channels
        self.use_center_region_extraction_branch = use_center_region_extraction_branch
        self.use_nearby_features = use_nearby_features
        self.use_entropy_coef_scheduler = use_entropy_scheduler
        self.min_entropy_coef = min_entropy_coef
        self.entropy_coef_step = entropy_coef_step
        self.num_updates = num_updates
        self.initial_entropy_coef = entropy_coef
        self.decay_rate = decay_rate
        self.rollout_episode_count = 0
        self.save_frequency_counter = 0
        self.error_info = None
        # Player information
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id
        self.last_points = 0.0 
        self.last_opp_points = 0.0 
        # Buffer
        self.buffer = ReplayBuffer(self.buffer_size, self.state_shape, self.ship_state_shape)
        # Learning Rate Scheduler
        if LR_scheduler:
            LambdaLR = lambda step: 1 - (1 - (lr_min / lr)) * (min(step, LR_num_steps) / LR_num_steps)
        else:
            LambdaLR = lambda step: 1.0 # no scheduling
        # Actor & Critic networks
        self.critic = MAPPO_model(input_channels=self.state_shape[0], output_dim=1, init_orthogonal=self.init_orthogonal).to(self.device)
        if self.use_multiple_actors:
            self.actors = torch.nn.ModuleList([
                MAPPO_model(input_channels=self.ship_state_shape[0], output_dim=action_dim, init_orthogonal=self.init_orthogonal, use_coord_channels=use_coord_channels, use_center_region_extraction_branch=use_center_region_extraction_branch, use_nearby_features=use_nearby_features
                    ).to(self.device) for _ in range(16) ])
        else:
            self.actor = MAPPO_model(input_channels=self.ship_state_shape[0], output_dim=action_dim, init_orthogonal=self.init_orthogonal, use_coord_channels=use_coord_channels, use_center_region_extraction_branch=use_center_region_extraction_branch, use_nearby_features=use_nearby_features).to(self.device)
        # Optimizers 
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.scheduler_critic = optim.lr_scheduler.LambdaLR(self.optimizer_critic, lr_lambda=LambdaLR)
        if self.use_multiple_actors:
            self.optimizer_actors = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
            self.scheduler_actors = [optim.lr_scheduler.LambdaLR(opt, lr_lambda=LambdaLR) for opt in self.optimizer_actors]
        else:
            self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
            self.scheduler_actor = optim.lr_scheduler.LambdaLR(self.optimizer_actor, lr_lambda=LambdaLR)
        # Load model if continuing training
        if continue_training_path is not None:
            self._load_networks(continue_training_path)
        # @ End of turn, current -> last
        self.last_state = None # (y, x)
        self.last_additional_features = None
        self.last_actions = None
        self.last_log_probs = None
        self.last_value = None
        self.last_valid_ships = None
        self.last_ship_states = None
        # current info for SEQUENTIAL ship decision-making
        self.friendly_energies = None
        self.friendly_positions = None
        # for relic, reward, & discovery logic (discovery means uncovering tiles/grow visible area)
        self.ALL_RELICS_FOUND = False
        self.relics_found_this_match = False
        self.relics_found_this_step = set() 
        self.relics_set = set()       
        self.reward_set = set()    
        self.reward_tracking = [] # ! list of dictionaries, matching potential reward tiles to points rewarded
        self.last_reward_channel = None 
        # LOGGING Accessed in notebooks
        self.latest_metrics = None  
        # Exploration and obstacle flags/info - not implemented
        self.OBSTACLE_MOVEMENT_PERIOD_FOUND = False
        self.OBSTACLE_MOVEMENT_DIRECTION_FOUND = False
        self.OBSTACLE_MOVEMENT_PERIOD = 20
        self.OBSTACLE_MOVEMENT_DIRECTION = (0, 0)
        self.OBSTACLES_MOVEMENT_STATUS = []
        

    def act(self, step: int, obs, total_steps, remainingOverageTime: int = 60):
        '''
            RUN 2 functions
                process_observation()
                process_ships()
            RUN critic network
            ADD to buffer
            LEARN if end of episode (and rollout count)
            return 
                actions (16, 3)
        '''
        match_number = step // (101) # 0-4
        match_step = step % (101) # 0-99
        if match_step == 0:
            self._new_match(step)
        
        # Process obs into features for the network
        (   state, 
            additional_features, 
            team_points, 
            opp_team_points, 
            points_increase,
            opp_points_increase
        ) = self._process_observation(step, obs, match_step, match_number)

        # Actors & Critic
        with torch.no_grad():
            # Run the actor network to get actions (and log probabilities/ Collect rewards) 
            (   actions, # (16, 3)
                log_probs, # (16,)
                valid_ships, # (16, [valid, y, x])
                ship_rewards, # (16,)
                ship_states # (16, *ship_state_shape)
            ) = self._process_ships(state, additional_features, points_increase)
            
            # value from critic
            if self.player == "player_1":
                state = state[:, ::-1, ::-1].swapaxes(1, 2) 
            state_tensor = torch.from_numpy(state[0:self.state_shape[0], :, :]).float().reshape(1, *self.state_shape).to(self.device)
            # add_feat_tensor = torch.tensor(additional_features.reshape(1, 10), dtype=torch.float32, device=self.device)
            value = self.critic(state_tensor).squeeze(-1).detach().cpu().numpy() # add_feat_tensor).squeeze(-1)
            if self.player == "player_1":
                state = state[:, ::-1, ::-1].swapaxes(1, 2) 

        # Append to the buffer 
        if self.last_state is not None:
            self.buffer.append(Experience(
                done=np.array(1.0 if match_step == 100 else 0.0, dtype=np.float32),
                # next_state=state, # now that we're in this new state, what rewards do we witness from taking actions from last step
                team_points=team_points,
                opp_team_points=opp_team_points,
                points_increase=points_increase,
                opp_points_increase=opp_points_increase, 
                ship_rewards=ship_rewards,
                state=self.last_state, # LAST
                value=self.last_value, # LAST
                additional_features=self.last_additional_features, # each ship LAST
                actions=self.last_actions,         # each ship LAST
                log_probs=self.last_log_probs,     # each ship LAST
                valid_ships=self.last_valid_ships, # each ship LAST
                ship_states=self.last_ship_states, # each ship LAST
            ))
        # Save the current step’s data for the next update
        self.last_state         = state
        self.last_additional_features = additional_features
        self.last_actions       = actions
        self.last_log_probs     = log_probs
        self.last_value         = value
        self.last_valid_ships   = valid_ships
        self.last_ship_states   = ship_states

        # Error handling: let match finish, save state, then stop
        if self.error_info is not None:
            self._save_state(total_steps, obs, match_number, match_step, state, additional_features, 
                            team_points, opp_team_points, points_increase, value, actions, log_probs, 
                            valid_ships, ship_rewards, ship_states)
            print(self.error_info)
            raise Exception("error_info - Saved state")

        # end of episode
        if match_step == 100 and  match_number == 4:
            self.rollout_episode_count += 1
            if self.rollout_episode_count >= self.rollout:
                # save state every #save_freq# number of learning updates
                if self.save_frequency_counter >= self.save_freq:
                    self._save_state(total_steps, obs, match_number, match_step, state, additional_features, 
                                team_points, opp_team_points, points_increase, value, actions, log_probs, 
                                valid_ships, ship_rewards, ship_states)
                    self.save_networks(self.save_dir)
                    self.save_frequency_counter = 0
                else:
                    self.save_frequency_counter += 1 # to match notebook
                ##################################
                # LEARN
                ##################################
                if self.train_mode:
                    self._learn()  # will update self.latest_metrics
                self.buffer = ReplayBuffer(self.buffer_size, self.state_shape, self.ship_state_shape) # self.buffer.clear()  
                self.rollout_episode_count = 0
                ##################################
                ##################################

        # Flip actions for player_1 if necessary
        if self.player == "player_1":
            actions, _ = self._flip_features(actions)
        return actions.astype(np.int32)

    def _process_observation(self, step: int, obs, match_step, match_number):
        '''
            Run 3 functions
                get_state()
                update_relic_discovery_logic()
                update_reward_logic()
            return 
                state, additional_features, team_points, opp_team_points, points_increase
        '''
        # Points
        team_points = np.array(obs["team_points"][self.team_id], dtype=np.float32)
        opp_team_points = np.array(obs["team_points"][self.opp_team_id], dtype=np.float32)
        points_increase = max(0.0, team_points - self.last_points)
        self.last_points = team_points
        opp_points_increase = max(0.0, opp_team_points - self.last_opp_points)
        self.last_opp_points = opp_team_points
        # FRIENDLY SHIPS for SEQUENTIAL decision-making
        self.friendly_energies = obs["units"]["energy"][self.team_id]
        self.friendly_positions = obs["units"]["position"][self.team_id]
        # BUILD GAMEBOARD
        state = get_state(obs, self.team_id, match_step, match_number)  
        state = update_relic_discovery_logic(self, state, match_step, match_number)
        state = update_reward_logic(self, state, match_step, match_number, points_increase, obs)
        # Determine extra features - 'free' info - maybe concatenate to fully connected layers?
        if not self.OBSTACLE_MOVEMENT_DIRECTION_FOUND: # not implemented
            movement_direction = 0.0
        else:
            movement_direction = -1.0 if self.OBSTACLE_MOVEMENT_DIRECTION[0] < 0 else 1.0
        if not self.OBSTACLE_MOVEMENT_PERIOD_FOUND:
            obstacle_period = 0.0
        else:
            obstacle_period = self.OBSTACLE_MOVEMENT_PERIOD
        additional_features = [ # ALL normalized 0.0-1.0
            round(self.UNIT_MOVE_COST / 5.0, 2),
            round((self.UNIT_SAP_COST - 30) / 20, 2),
            round((self.UNIT_SAP_RANGE - 3) / 4, 2),
            movement_direction,
            round(obstacle_period / 40, 2),
            int(self.ALL_RELICS_FOUND), # DETERMINE EXPLORATION?
            int(0), # all rewards found
            round(match_number / 4.0, 2),
            round(match_step / 100.0, 2),
            round(team_points / 1000.0, 5)
        ]
        return (np.array(state, dtype=np.float32), 
                np.array(additional_features, dtype=np.float32), 
                np.array(team_points, dtype=np.float32), 
                np.array(opp_team_points, dtype=np.float32),
                np.array(points_increase, dtype=np.float32),
                np.array(opp_points_increase, dtype=np.float32)) 

    def _process_ships(self, state, additional_features, points_increase):
        '''
            For each ship
                RUN
                    get_ship_state()   -> ship_state
                    actor network      -> actions, log_probs
                    get_ship_rewards() -> ship_reward, ship_discovery_channel
            return
                actions, log_probs, valid_ships, ship_rewards, ship_states
        '''
        valid_ships = np.zeros((16, 3), dtype=np.float32) # [valid, y, x]
        actions     = np.zeros((16, 3), dtype=np.float32)
        log_probs   = np.zeros(16, dtype=np.float32)
        ship_rewards = np.zeros(16, dtype=np.float32)
        ship_states = np.zeros((16, *self.ship_state_shape), dtype=np.float32)
        
        # First ship will have same Discovery Channel as global state 
        # as each ship moves, sometimes new tiles get discovered/uncovered
        ship_discovery_channel = state[discovery_ch_idx].copy()

        # Loop over each ship 
        for ship_index, (ship_x, ship_y) in enumerate(self.friendly_positions):
            ship_energy = self.friendly_energies[ship_index]
            if ship_x < -0.5 or ship_energy < 0: # -1, -1
                continue  # Skip ships that are not on the board or not able to move
            valid_ships[ship_index] = [1.0, ship_x, ship_y]
            ################################################################
            # THIS SHIP’s STATE 
            ################################################################
            ship_state = get_ship_state(self, state, ship_index, ship_x, ship_y, ship_energy, actions, ship_discovery_channel)
            # ship_features = np.concatenate([ additional_features.reshape(1, -1), np.array((node_x, node_y)).reshape(1, -1)], axis=1)
            # ship_features_tensor = torch.tensor(ship_features.reshape(1, 5), dtype=torch.float32, device=self.device)
            if self.player == "player_1":
                ship_state = ship_state[:, ::-1, ::-1].swapaxes(1, 2) 
            ship_state_tensor = torch.tensor(ship_state.reshape(1, *self.ship_state_shape), dtype=torch.float32, device=self.device)
            ################################################################
            # THIS actor policy network to choose ACTION
            ################################################################
            if self.use_multiple_actors:
                policy_logits = self.actors[ship_index](ship_state_tensor)
            else:
                policy_logits = self.actor(ship_state_tensor) #, ship_features_tensor)
            if self.player == "player_1":
                indices = torch.tensor([0, 3, 4, 1, 2])
                policy_logits = policy_logits[indices]
            # optional: <1-less random, >1-more random, 1-does nothing
            if self.temperature:
                policy_logits = policy_logits / self.temperature
            # Sample an action and record its log probability
            policy_probs = F.softmax(policy_logits, dim=-1)
            action_sample = torch.multinomial(policy_probs, num_samples=1)
            action_value = action_sample.item()
            log_prob = torch.log(policy_probs.gather(1, action_sample)).item()
            # Collect
            ship_states[ship_index] = ship_state
            actions[ship_index, 0]  = action_value
            log_probs[ship_index]   = log_prob
            ################################################################
            # Compute REWARD for this ship 
            # and update Discovery Channel (if tiles were discovered)
            ################################################################
            ship_reward, ship_discovery_channel = self._get_ship_rewards(state, ship_state, ship_index, ship_x, ship_y, 
                                                                         ship_energy, actions, ship_discovery_channel)
            ship_rewards[ship_index] = ship_reward
        return actions, log_probs, valid_ships, ship_rewards, ship_states

    def _get_ship_rewards(self, state, ship_state, ship_index, ship_x, ship_y, ship_energy, actions, ship_discovery_channel):
        '''
            REWARDS/PENALTIES 
            needs tuning
                Out of bounds                        -> *self.penalty_out_of_bounds
                Asteroid                             -> *self.penalty_asteroid
                Tile energy                          -> *self.tile_energy_coefficient
                REWARD                               -> *self.reward_tile_reward
                Potential REWARD                     -> *self.move_onto_potential_reward_tile
                Moving off reward                    -> *self.penalty_moving_off_reward_tile
                Moving off p. reward                 -> *self.penalty_moving_off_potential_reward_tile
                Has full energy (400) & did not move -> *self.penalty_full_energy
                Penalty for moving                   -> *self.penalty_moving_cost
                SAP                                  -> *self.penalty_sap_cost
                behind enemy lines                   -> *self.behind_enemy_lines_coefficient
                Uncovering/Discovering               -> *self.discovery_reward
            return return ship_reward, ship_discovery_channel
        '''
        # initial running total
        ship_reward = 0.0

        # behind enemy lines 
        if (ship_x + ship_y) > 23 : 
            is_behind_enemy_lines = True
        else:
            is_behind_enemy_lines = False

        # center of ship_state (old position/original position)
        radius = self.ship_state_shape[1] // 2  # For board_size=11, radius=5. center = (radius, radius)
        new_ship_x = radius 
        new_ship_y = radius 

        ########################################################################
        # adjust for action
        #  + out of bounds penalty
        ########################################################################
        curr_ship_action = actions[ship_index, 0]
        ship_moved = False
        if curr_ship_action > 0 and curr_ship_action < 5: 
            # Up
            if curr_ship_action == 1:     
                if ship_y < 0.5: # out of bounds, penalty, no movement
                    ship_reward += self.penalty_out_of_bounds
                else:
                    new_ship_y = new_ship_y - 1
                    ship_moved = True
            # Right
            elif curr_ship_action == 2:  
                if ship_x > 22.5: # out of bounds, penalty, no movement
                    ship_reward += self.penalty_out_of_bounds
                else:
                    new_ship_x = new_ship_x + 1
                    ship_moved = True
            # Down
            elif curr_ship_action == 3:  
                if ship_y > 22.5: # out of bounds, penalty, no movement
                    ship_reward += self.penalty_out_of_bounds
                else:
                    new_ship_y = new_ship_y + 1
                    ship_moved = True
            # Left
            elif curr_ship_action == 4:  
                if ship_x < 0.5: # out of bounds, penalty, no movement
                    ship_reward += self.penalty_out_of_bounds
                else:
                    new_ship_x = new_ship_x - 1
                    ship_moved = True

        ########################################################################
        # asteroid penalty - no movement
        ########################################################################
        if ship_moved: 
            if ship_state[astroid_ch_idx, new_ship_y, new_ship_x] > 0.5 : # asteroid
                ship_reward += self.penalty_asteroid
                ship_moved = False

        ########################################################################
        # tile energy reward/penalty
        ########################################################################
        if ship_moved:
            tile_energy = ship_state[tile_energy_ch_idx, new_ship_y, new_ship_x]
        else:
            tile_energy = ship_state[tile_energy_ch_idx, radius, radius] 

        # tile has unknown energy
        if tile_energy != 0.5125: 
            # (0.0-1.0) -> (-0.5, 0.5) -> (-1.0, 1.0)
            ship_reward += (tile_energy - 0.5) * 2 * self.tile_energy_coefficient
            # ship_reward += tile_energy * tile_energy_coefficient

        ########################################################################
        # reward tiles REWARDS
        ########################################################################
        if ship_moved: 
            # new tile was empty
            if ship_state[friendly_ch_idx, new_ship_y, new_ship_x] < 0.05: 
                if ship_state[reward_ch_idx, new_ship_y, new_ship_x] > 0.9: # REWARD tile
                    ######## ON REWARD 
                    if is_behind_enemy_lines:
                        ship_reward += self.reward_tile_reward * self.behind_enemy_lines_coefficient
                    else:
                        ship_reward += self.reward_tile_reward
                    ########
                elif ship_state[reward_ch_idx, new_ship_y, new_ship_x] > 0.1: # POTENTIAL tile
                    ######## POTENTIAL REWARD
                    if is_behind_enemy_lines:
                        ship_reward += self.move_onto_potential_reward_tile * self.behind_enemy_lines_coefficient
                    else:
                        ship_reward += self.move_onto_potential_reward_tile
                    ########
        else: 
            # ship doesn't move 
            #   & NOBODY ELSE on tile
            if ship_state[friendly_ch_idx, radius, radius] < 0.15: 
                if ship_state[reward_ch_idx, radius, radius] > 0.9: # REWARD tile
                    ######## STAY ON REWARD 
                    if is_behind_enemy_lines:
                        ship_reward += self.reward_tile_reward * self.behind_enemy_lines_coefficient * 1.5
                    else:
                        ship_reward += self.reward_tile_reward * 1.5
                    ########
                elif ship_state[reward_ch_idx, radius, radius] > 0.1: # POTENTIAL REWARD for tile
                    ######## STAY ON POTENTIAL REWARD                               
                    if is_behind_enemy_lines:
                        ship_reward += self.move_onto_potential_reward_tile * self.behind_enemy_lines_coefficient
                    else:
                        ship_reward += self.move_onto_potential_reward_tile
                    ########

        ########################################################################
        # moving off reward tiles PENALTIES
        ########################################################################
        if ship_moved: 
            # old tile had nobody ELSE on it
            if ship_state[friendly_ch_idx, radius, radius] < 0.15:
                # Old tile = reward tile
                if ship_state[reward_ch_idx, radius, radius] > 0.6:
                    ship_reward += self.penalty_moving_off_reward_tile # moved off reward tile PENALTY
                elif ship_state[reward_ch_idx, radius, radius] > 0.1:
                    ship_reward += self.penalty_moving_off_potential_reward_tile # moved off potential tile PENALTY

        ########################################################################
        # move cost, penalty
        ########################################################################
        if curr_ship_action > 0 and curr_ship_action < 5:
            ship_reward += self.penalty_moving_cost

        ########################################################################
        # sap cost, penalty
        ########################################################################
        if curr_ship_action == 5:
            ship_reward += self.penalty_sap_cost
        
        ########################################################################
        # full energy penalty, NOT on REWARD, did not move
        ########################################################################
        if curr_ship_action == 0 and ship_energy > 395:
            if ship_state[reward_ch_idx, radius, radius] < 0.6:
                ship_reward += self.penalty_full_energy

        ########################################################################
        # Discovery
        ########################################################################
        # if moving, there will be a line of tiles that potentially uncover parts of the board
        # this will be determined by UNIT_SENSOR_RANGE (3-9)
        # Discovery 'Horizon' -> ship_state[1, y_min:y_max, x_min:x_max]
        # (1: Discovery Channel)
        num_tiles_made_visible = 0
        if ship_moved:
            y_min, y_max, x_min, x_max = get_discovery_horizon(curr_ship_action, radius, self.UNIT_SENSOR_RANGE)
            # number of tiles discovered/made visible
            num_tiles_made_visible = ship_state[discovery_ch_idx, y_min:y_max, x_min:x_max].sum()
            # Update Discovery Channel with this new discovery
            # future ships cannot receive discovery reward for these tiles
            ship_discovery_channel[y_min:y_max, x_min:x_max] = 0.0
        ship_reward += num_tiles_made_visible * self.discovery_reward


        # TODO: discover relic reward  (must come FROM next state)
        return ship_reward, ship_discovery_channel
    
    def _learn(self):
        # === Retrieve Trajectories ===
        trajectories = self.buffer.get_trajectories()
        # === Compute Advantages, Returns, Rewards ===
        advantages, returns, rewards, advantages_original = self._advantage_function(
            trajectories['ship_rewards'],
            trajectories['values'],
            trajectories['dones'],
            trajectories['team_points'],
            trajectories['opp_team_points'],
            trajectories['points_increase'],
            trajectories['opp_points_increase'],
            # trajectories['next_states'] #?
        )
        # === stack data === 
        states      = np.stack(trajectories['states'])[:, :self.state_shape[0], :, :]     # shape: (buffer_size, 8, 24, 24)
        values      = np.array(trajectories['values'])                                    # shape: (buffer_size, )
        ship_states = np.array(trajectories['ship_states'])                               # shape: (buffer_size, 16, 9, 24, 24)
        valid_ships = np.array(trajectories['valid_ships'])[:, :, 0]                      # shape: (buffer_size, 16 )
        actions     = np.array(trajectories['actions'])[:, :, 0]                          # shape: (buffer_size, 16 )   
        log_probs   = np.array(trajectories['log_probs'])                                 # shape: (buffer_size, 16,)

        # === Initialize Logging ===
        actor_loss_logs        = []
        critic_loss_logs       = []
        entropy_logs           = []
        policy_ratio_logs      = []
        clipping_fraction_logs = []
        advantages_mean_logs   = []
        value_error_logs       = []
        grad_norm_logs         = []

        # === Entropy Coefficient Scheduler ===
        if self.use_entropy_coef_scheduler:
            # self.entropy_coef = max(self.entropy_coef - self.entropy_coef_step, self.min_entropy_coef) # linear anneal entropy 
            self.entropy_coef = self.min_entropy_coef + ( self.initial_entropy_coef -  self.min_entropy_coef) * math.exp(-self.decay_rate * self.num_updates)
            self.num_updates += 1

        # === Step Schedulers ===
        self.scheduler_critic.step()
        for ship_idx in range(16):
            self.scheduler_actors[ship_idx].step()
        
        # === Training Epoch Loop ===
        num_samples = len(trajectories['states'])  # buffer_size
        indices = np.arange(num_samples)
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            # === Mini-Batch Loop ===
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_idxs = indices[start:end]

                # --- Batch Tensors with vectorized slicing ---
                states_batch        = torch.as_tensor(states[batch_idxs].reshape(-1, *self.state_shape), dtype=torch.float32, device=self.device)
                old_values_batch    = torch.as_tensor(values[batch_idxs], dtype=torch.float32, device=self.device)          # shape: (batch_size,)
                ship_states_batch   = torch.as_tensor(ship_states[batch_idxs].reshape(-1, *self.ship_state_shape), dtype=torch.float32, device=self.device)
                valid_ships_mask    = torch.as_tensor(valid_ships[batch_idxs] == 1.0, dtype=torch.bool, device=self.device) # shape: (batch_size, 16)
                actions_batch       = torch.as_tensor(actions[batch_idxs],   dtype=torch.int64, device=self.device)         # shape: (batch_size, 16)
                old_log_probs_batch = torch.as_tensor(log_probs[batch_idxs], dtype=torch.float32, device=self.device)       # shape: (batch_size, 16)
                returns_batch       = torch.as_tensor(returns[batch_idxs],   dtype=torch.float32, device=self.device)       # shape: (batch_size,)
                advantages_batch    = torch.as_tensor(advantages[batch_idxs], dtype=torch.float32, device=self.device)      # shape: (batch_size, 16)   

                ############################################################################################
                # === Critic === 
                ############################################################################################
                # --- Recompute Critic Values ---
                if self.player == "player_1":
                    states_batch = states_batch[:, ::-1, ::-1].swapaxes(1, 2) 
                new_values = self.critic(states_batch).squeeze(-1)

                # --- Critic Loss ---
                # use Huber? F.smooth_l1_loss
                critic_loss = self.value_coef * F.mse_loss(new_values, returns_batch)
                if self.use_clipped_critic_loss:
                    new_values_clipped = old_values_batch + torch.clamp(new_values - old_values_batch, min=-self.eps_clip, max=self.eps_clip)
                    critic_loss_clipped = self.value_coef * F.mse_loss(new_values_clipped, returns_batch)
                    critic_loss = torch.max(critic_loss, critic_loss_clipped)

                # --- Critic Update ---
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
                # self.scheduler_critic.step()

                # === Logging Metrics === Critic
                critic_loss_logs.append(critic_loss.item())
                value_error_logs.append((new_values - returns_batch).abs().mean().item()) #mean

                ############################################################################################
                # === Actor(s) === 
                ############################################################################################
                if self.use_multiple_actors:
                    # --- Reshape Ship States ---
                    actual_batch_size = ship_states_batch.numel() // (16 * np.prod(self.ship_state_shape))
                    ship_states_batch = ship_states_batch.contiguous().view(actual_batch_size, 16, *self.ship_state_shape)
                    # --- Update for Each Actor ---
                    for ship_idx in range(16):
                        curr_ship_mask     = valid_ships_mask[:, ship_idx] # number of times this shipindex was valid
                        # --- Remove ships not on the board --- 
                        curr_actions       = actions_batch[:, ship_idx][curr_ship_mask]                 # shape: (num_valid )
                        curr_old_log_probs = old_log_probs_batch[:, ship_idx][curr_ship_mask]           # shape: (num_valid )
                        curr_advantages    = advantages_batch[:, ship_idx][curr_ship_mask]              # shape: (num_valid )
                        # handle ship_states
                        curr_ship_states   = ship_states_batch[:, ship_idx, :, :, :]    # shape: (batch_size, *ship_state_shape) 
                        curr_ship_states   = curr_ship_states[curr_ship_mask]           # shape: (num_valid, *ship_state_shape) 

                        new_log_probs, new_entropies = self._recompute_ship_log_probs(self.actors[ship_idx], curr_ship_states, curr_actions)
                        
                        # --- Check for Shape Mismatch (Debugging) --- # shape: (num_valid_ships )
                        if curr_old_log_probs.shape[0] != new_log_probs.shape[0] or  curr_advantages.shape[0] != new_log_probs.shape[0]:
                            print(f"Mismatch in log probabilities: old({curr_old_log_probs.shape[0]}) vs new({new_log_probs.shape[0]})")
                            raise Exception("Mismatch in log probabilities")
                        
                        # --- PPO Ratios & Surrogate Loss ---
                        ratios = torch.exp(new_log_probs - curr_old_log_probs.detach())
                        surr1 = ratios * curr_advantages
                        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * curr_advantages
                        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * new_entropies.mean()

                        # === Actor Update === 
                        self.optimizer_actors[ship_idx].zero_grad()
                        actor_loss.backward() # float-value
                        # Use the return value of clip_grad_norm_ as grad_norm (it's the total norm)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.actors[ship_idx].parameters(), 0.5)
                        self.optimizer_actors[ship_idx].step()
                        # self.scheduler_actors[ship_idx].step()

                        # === Logging Metrics === Each Actor
                        grad_norm_logs.append(grad_norm.cpu())
                        actor_loss_logs.append(actor_loss.item())
                        entropy_logs.append(new_entropies.mean().item()) #mean
                        policy_ratio_logs.append(ratios.mean().item()) #mean
                        clipped = ((ratios < (1 - self.eps_clip)) | (ratios > (1 + self.eps_clip))).float()
                        clipping_fraction_logs.append(clipped.mean().item()) #mean
                        advantages_mean_logs.append(advantages_batch.mean().item()) #mean

                else:
                    # --- Remove ships not on the board --- 
                    # valid_ships_mask - shape: (batch_size, 16)
                    actions_batch       = actions_batch[valid_ships_mask].flatten()       # shape: (num_valid_ships )
                    old_log_probs_batch = old_log_probs_batch[valid_ships_mask].flatten() # shape: (num_valid_ships )
                    advantages_batch    = advantages_batch[valid_ships_mask].flatten()    # shape: (num_valid_ships )
                    ship_states_batch   = ship_states_batch[valid_ships_mask.flatten()]   # shape: (num_valid_ships, *ship_state_shape) 

                    # --- Recompute Actor Log-Probabilities for Each Ship's action ---
                    new_log_probs, new_entropies = self._recompute_ship_log_probs(self.actor, ship_states_batch, actions_batch) # shape: (num_valid_ships )

                    # --- Check for Shape Mismatch (Debugging) --- # shape: (num_valid_ships )
                    if old_log_probs_batch.shape[0] != new_log_probs.shape[0] or  advantages_batch.shape[0] != new_log_probs.shape[0]:
                        print(f"Mismatch in log probabilities: old({old_log_probs_batch.shape[0]}) vs new({new_log_probs.shape[0]})")
                        raise Exception("Mismatch in log probabilities")

                    # --- PPO Ratios & Surrogate Loss --- # shape: (num_valid_ships )
                    ratios = torch.exp(new_log_probs - old_log_probs_batch.detach())
                    surr1 = ratios * advantages_batch
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
                    actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * new_entropies.mean()
                    # TODO: early stopping with KL divergence

                    # === Actor Update === 
                    self.optimizer_actor.zero_grad()
                    actor_loss.backward() # float-value
                    # Use the return value of clip_grad_norm_ as grad_norm (it's the total norm)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.optimizer_actor.step()
                    self.scheduler_actor.step()

                    # === Logging Metrics ===
                    grad_norm_logs.append(grad_norm.cpu())
                    actor_loss_logs.append(actor_loss.item())
                    entropy_logs.append(new_entropies.mean().item()) #mean
                    policy_ratio_logs.append(ratios.mean().item()) #mean
                    clipped = ((ratios < (1 - self.eps_clip)) | (ratios > (1 + self.eps_clip))).float()
                    clipping_fraction_logs.append(clipped.mean().item()) #mean
                    advantages_mean_logs.append(advantages_batch.mean().item()) #mean

        # === Aggregate and Save Latest Metrics ===
        self.latest_metrics = {  # convert from numpy to native Python
            "actor_loss":        [float(np.mean(actor_loss_logs)), float(np.std(actor_loss_logs))],
            "critic_loss":       [float(np.mean(critic_loss_logs)), float(np.std(critic_loss_logs))],
            "entropy":           [float(np.mean(entropy_logs)), float(np.std(entropy_logs))],
            "policy_ratio":      [float(np.mean(policy_ratio_logs)), float(np.std(policy_ratio_logs))],
            "clipping_fraction": [float(np.mean(clipping_fraction_logs)), float(np.std(clipping_fraction_logs))],
            "value_abs_error":   [float(np.mean(value_error_logs)), float(np.std(value_error_logs))],
            "grad_norm":         [float(np.mean(grad_norm_logs)), float(np.std(grad_norm_logs))],
            "returns":           [float(np.mean(returns)), float(np.std(returns))],
            "rewards":           [float(np.mean(rewards)), float(np.std(rewards))],
            "advantages":        [float(np.mean(advantages_original)), float(np.std(advantages_original))],
            "ship_reward_means": [np.mean(trajectories['ship_rewards'], axis=0).squeeze().tolist()],
            "ship_reward_stds":  [np.std(trajectories['ship_rewards'], axis=0).squeeze().tolist()],
            "all_values":        values.tolist(),
            "all_rewards":       rewards.tolist(),
            "all_returns":       returns.tolist(),
            "entropy_coef":      float(self.entropy_coef),
        }

    def _advantage_function(self, ship_rewards, values, dones, team_points, opp_team_points, points_increase, opp_points_increase):
        """
        Compute advantages using Generalized Advantage Estimation (GAE) with adaptive reward scaling.

        Returns:
            advantages (np.array): Computed advantage values 
                -- used in actor loss
                    ratios = torch.exp(new_log_probs - masked_batch_log_probs_old.detach())
                    surr1 = ratios * masked_advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * masked_advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * new_entropies.mean()
            
            returns (np.array): Computed return values 
                -- used in critic loss
                    critic_loss = self.value_coef * F.mse_loss(new_values, batch_returns)
            rewards (np.array): final Computed rewards
            advantages_original (np.array): unnormalized Computed advantage values

        TODO:
            Normalize rewards dynamically Or Adaptive Reward Scaling
            Normalize rewards to stabilize training, especially if individual match scores vary widely.
            Curriculum Learning
                Train the agent first on single matches before transitioning to the best-of-five format.
                This can help the agent learn short-term strategies before optimizing for longer-term planning.

        """
        ############################################################
        # REWARDS
        #rewards = points_increase.copy()
        # relative to opponent (high variability of points between matches)
        rewards = points_increase #- opp_points_increase*0.2 # Coefficient 

        ############################################################
        # BONUSES
        ############################################################
        buffer_length = ship_rewards.shape[0]
        # num_episodes = buffer_length // 500
        # for episode_num in range(num_episodes): # 0-number of episodes in buffer
        #     i = episode_num * 500 # index offset
        #     match_one   = team_points[i+99]  > opp_team_points[i+99]
        #     match_two   = team_points[i+199] > opp_team_points[i+199]
        #     match_three = team_points[i+299] > opp_team_points[i+299]
        #     match_four  = team_points[i+399] > opp_team_points[i+399]
        #     match_five  = team_points[i+499] > opp_team_points[i+499]
        #     # match bonus
        #     rewards[i+99]  += self.match_bonus if match_one  else -0.2*self.match_bonus
        #     rewards[i+199] += self.match_bonus if match_two  else -0.2*self.match_bonus
        #     rewards[i+299] += self.match_bonus if match_three else -0.2*self.match_bonus
        #     rewards[i+399] += self.match_bonus if match_four else -0.2*self.match_bonus
        #     rewards[i+499] += self.match_bonus if match_five else -0.2*self.match_bonus
        #     # episode bonus
        #     if (int(match_one) + int(match_two) + int(match_three) + int(match_four) + int(match_five)) > 2:
        #         rewards[i+499] += self.episode_bonus
        #     else :
        #         rewards[i+499] -= 0.2*self.episode_bonus

        #### DEBUGGING
        # rewards_with_bonus = rewards.copy()
        # original_values = values.copy()
        # original_ship_rewards = ship_rewards.copy()
        ####
        values = np.pad(values, (0, 1), mode='edge') # shape (t+1)
        if self.use_multiple_actors:
            ############################################################
            # --- MULTIPLE ACTORS ---
            # 
            # A_t = r + gam*D*V(t+1) - V(t) + gam*lam*d*A(t+1)
            # 
            # advantages -> shape (buffer_length, 16)
            # returns    -> shape (buffer_length, )
            ############################################################
            advantages = np.zeros_like(ship_rewards, dtype=np.float32) # shape (buffer_length, 16)
            returns = np.zeros_like(ship_rewards, dtype=np.float32)    # shape (buffer_length, 16) -> shape (buffer_length, )

            ######
            # rewards sometimes overpower the ship_rewards ?
            # rewards = rewards/self.rewards_divisor_factor
            ######
            for ship_index in range(16):
                last_advantage = 0
                for t in reversed(range(buffer_length)):
                    if t % 500 == 0: 
                        last_advantage = 0
                    # delta = r + gam*D*V(t+1) - V(t)
                    delta = ship_rewards[t][ship_index] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t] # + rewards[t] 
                    # A_t = delta + gam*lam*D*A(t+1)
                    last_advantage = delta + self.gamma * self.lam * (1 - dones[t]) * last_advantage
                    advantages[t, ship_index] = last_advantage
                    returns[t, ship_index] = last_advantage + values[t]
            returns = returns.sum(axis=1)  # shape (buffer_length, 16) -> shape (buffer_length, )
            returns = returns/16
            # NORMALIZE ADVANTAGES (for policy ratio)
            if self.per_agent_normalize_advantages:
                normalized_advantages = np.zeros_like(advantages, dtype=np.float32)
                for ship_index in range(16):
                    ship_advantages = advantages[:, ship_index]
                    normalized_advantages[:, ship_index] = (ship_advantages - ship_advantages.mean()) / (ship_advantages.std() + 1e-6 )
            else:
                normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            return np.array(normalized_advantages), np.array(returns), np.array(rewards), np.array(advantages)
        else:
            ############################################################
            # --- SINGLE ACTOR ---
            # SUM ship_rewards 
            # ADD to rewards
            # 
            # A_t = r + gam*D*V(t+1) - V(t) + gam*lam*d*A(t+1)
            # 
            # advantages -> shape (t, 16) (repeated 16 times)
            # returns    -> shape (t, )
            ############################################################
            advantages = np.zeros_like(rewards, dtype=np.float32) # shape (buffer_length,)
            returns = np.zeros_like(rewards, dtype=np.float32)  # shape (buffer_length,)
            # sum 
            ship_rewards = ship_rewards.sum(axis=1)    # Shape: (buffer_length,)
            rewards = rewards + ship_rewards           # shape (buffer_length,)
            rewards = rewards/16
            last_advantage = 0
            for t in reversed(range(buffer_length)):
                # delta = r + gam*D*V(t+1) - V(t)
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                # A_t = delta + gam*lam*D*A(t+1)
                last_advantage = delta + self.gamma * self.lam * (1 - dones[t]) * last_advantage
                # combined equation 
                # A_t = r + gam*D*V(t+1) - V(t) + gam*lam*d*A(t+1)
                advantages[t] = last_advantage
                returns[t] = last_advantage + values[t]
            # for t in reversed(range(buffer_length)):
            #     returns[t] = rewards[t] + returns[t+1] * self.gamma * (1 - dones[t]) 
            #     returns[t] = ship_rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) 

            # NORMALIZE ADVANTAGES  (for policy ratio)
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-6  # Prevent division by zero
            normalized_advantages = (advantages - adv_mean) / adv_std
            normalized_advantages = normalized_advantages[:, np.newaxis].repeat(16, axis=1)

            return np.array(normalized_advantages), np.array(returns), rewards

        # rewards = rewards
            # ship_rewards = ship_rewards
            # ship_rewards = ship_rewards + rewards[:, np.newaxis]
            # for t in reversed(range(buffer_length)):
            #     returns[t] = rewards[t] + returns[t+1] * self.gamma * (1 - dones[t]) 
            #     returns[t] = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) 

        # # temporary save for tuning
        # temporary_rewards = pd.DataFrame({
        #     'points_increase': points_increase,       # shape (505,)
        #     'rewards_with_bonus': rewards_with_bonus,  # shape (505,)
        #     'rewards': rewards,                        # shape (505,)
        #     'advantages': advantages,                  # shape (505,)
        #     'normalized_advantages': normalized_advantages, # shape (505,)
        #     'done': dones,                            # shape (505,)
        #     'team_points': team_points,                # shape (505,)
        #     'opp_team_points': opp_team_points,        # shape (505,)
        #     'values': original_values,                          # shape (505,)
        #     'returns': returns,                        # shape (505,)
        #     'ship_rewards': ship_rewards,              # shape (505,)
        #     #'original_ship_rewards':                  # shape (505, 16)
        # })
        # # expand original_ship_rewards
        # for i in range(16):
        #     temporary_rewards[f'original_ship_rewards_{i}'] = original_ship_rewards[:, i]
        # temporary_rewards.to_csv(f'rewards_{random.randint(0, 10000)}.csv', index=False)
    
        # Normalize rewards?
        # reward_mean = np.mean(rewards) + 1e-6  # Prevent division by zero
        # reward_std = np.std(rewards) + 1e-6
        # normalized_rewards = (rewards - reward_mean) / reward_std  # Normalize

        # # ship_rewards normalization
        # ship_reward_mean = np.mean(total_ship_rewards) + 1e-6  # Prevent division by zero
        # ship_reward_std = np.std(total_ship_rewards) + 1e-6
        # ship_rewards = (ship_rewards - reward_mean) / reward_std  # Normalize

        
        #     # Compute returns: the return at each time step is the sum of discounted future rewards
        #     next_value = 0  # Initialize the value of the next state (next value is 0 for terminal states)
        #     for t in reversed(range(len(rewards))):
        #         # If the episode ends (done = True), set the next value to 0
        #         if dones[t]:
        #             next_value = 0
        #         returns[t] = rewards[t] + gamma * next_value
        #         next_value = values[t]
            
        #     # Compute advantages using GAE
        #     next_value = 0  # Reset next value for advantage calculation
        #     for t in reversed(range(len(rewards))):
        #         if dones[t]:
        #             next_value = 0
        #         delta = rewards[t] + gamma * next_value - values[t]
        #         advantages[t] = delta + gamma * lam * advantages[t + 1] if t + 1 < len(rewards) else delta
        #         next_value = values[t]
       
        #     # Initialize lists to store advantages for each agent
        #     all_agent_advantages = []
            
        #     # Compute the global advantages and returns (using global reward)
        #     global_advantages, global_returns = compute_gae(global_rewards, values, next_values, dones, gamma, lam)
            
        #     # For each agent, handle their individual rewards and compute the advantages
        #     for agent_idx in range(len(agent_rewards)):
        #         # Normalize or adjust agent rewards (you can apply more advanced scaling here if needed)
        #         weighted_agent_rewards = alpha * agent_rewards[agent_idx] + (1 - alpha) * global_rewards
                
        #         # Compute the advantages and returns for this agent
        #         agent_advantages, agent_returns = compute_gae(weighted_agent_rewards, values, next_values, dones, gamma, lam)
        #         all_agent_advantages.append(agent_advantages)
        
        # return advantages, returns, rewards

    def _recompute_ship_log_probs(self, current_actor, ship_states_batch, actions_batch):
        """
        Computes the new log probabilities and entropies for ship in the mini-batch,
        TODO: processing all valid ships in parallel?
        Returns:
            new_log_probs_tensor: Tensor of log probabilities for valid ships (1D)
            new_entropies_tensor: Tensor of entropies for valid ships (1D)
        """
        ##########################################################################
        # Single Actor (current_actor)
        ##########################################################################
        if self.player == "player_1":
            ship_states_batch = ship_states_batch[:, ::-1, ::-1]#.swapaxes(1, 2) ?.transpose(0, 2, 1)
        # should be shape (num_valid_ships, num_actions)
        new_logits = current_actor(ship_states_batch)
        if self.player == "player_1":
            indices = torch.tensor([0, 3, 4, 1, 2], device=self.device)
            new_logits = new_logits[:, indices] 
        # shape (num_valid_ships, num_actions)
        if self.temperature is not None:
            new_logits /= self.temperature # temperature scaling's
        new_probs = torch.nn.functional.softmax(new_logits, dim=-1)
        # reshape actions (num_valid_ships) to (num_valid_ships, 1) for gather
        actions_batch = actions_batch.unsqueeze(1)
        # shape (num_valid_ships, )
        new_log_probs = torch.log(new_probs.gather(1, actions_batch)).squeeze()
        new_entropies = -(new_probs * torch.log(new_probs + 1e-10)).sum(dim=-1).squeeze()
        # shape (num_valid_ships) (num_valid_ships)
        return new_log_probs, new_entropies
    
# other
    def new_episode(self, env_cfg):
        # Clear last-step experience information
        self.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        self.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        self.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        self.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]
        # Clear exploration and obstacle flags/info
        self.ALL_RELICS_FOUND = False
        self.relics_set = set()
        self.reward_set = set()
        self.reward_tracking = []
        self.last_reward_channel = None
        self.relics_found_this_match = False
        self.relics_found_this_step = set()   
        # self.OBSTACLE_MOVEMENT_PERIOD_FOUND = False
        # self.OBSTACLE_MOVEMENT_DIRECTION_FOUND = False
        # self.OBSTACLE_MOVEMENT_PERIOD = 20  
        # self.OBSTACLE_MOVEMENT_DIRECTION = (0, 0)
        # self.OBSTACLES_MOVEMENT_STATUS = []
   
    def save_networks(self, path):
        if self.use_multiple_actors:
            for i in range(16):
                torch.save(self.actors[i].state_dict(), os.path.join(path, f"actor_{i}.pt"))
                torch.save(self.optimizer_actors[i].state_dict(), os.path.join(path, f"optimizer_actor_{i}.pt"))
        else:
            torch.save(self.actor.state_dict(), os.path.join(path, "actor.pt"))
            torch.save(self.optimizer_actor.state_dict(), os.path.join(path, "optimizer_actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))
        torch.save(self.optimizer_critic.state_dict(), os.path.join(path, "optimizer_critic.pt"))

    def _new_match(self, step):
        self.last_state = None
        self.last_additional_features = None
        self.last_actions = None
        self.last_log_probs = None
        self.last_value = None
        self.last_valid_ships = None
        self.last_ship_states = None
        self.friendly_energies = None
        self.friendly_positions = None

    def _save_state(self, total_steps, obs, match_number, match_step, state, additional_features, 
               team_points, opp_team_points, points_increase, value, actions, log_probs, 
               valid_ships, ship_rewards, ship_states):

        file_path = os.path.join(self.save_dir, f"state_info{total_steps}.txt")
        
        def format_3d_array(arr):
            """
            Formats a 3D numpy array (channels, height, width) into a human-readable string.
            Each channel is labeled with a header and each row is prefixed with its row number.
            Values are rounded to 2 decimals; if a value is 0.00 it is replaced with "----".
            """
            # If the array doesn't have 3 dimensions, return its string representation.
            if not (hasattr(arr, "ndim") and arr.ndim == 3):
                return str(arr)
            
            lines = []
            num_channels, height, width = arr.shape
            for channel, channel_name in enumerate(channel_names):
                lines.append(f"Channel {channel}  {channel_name}:")
                header = "       " + " ".join(f"Co{col:02d}" for col in range(width))
                lines.append(header)
                for row_idx, row in enumerate(arr[channel]):
                    formatted_values = []
                    for val in row:
                        val_float = float(val)
                        if round(val_float, 2) == 0.0:
                            formatted_values.append("----")
                        elif round(val_float, 4) == 0.5125: # tile energy unknown
                            formatted_values.append("----")
                        else:
                            formatted_values.append(f"{val_float:.2f}")
                    row_str = " ".join(formatted_values)
                    lines.append(f"Row {row_idx:02d}: {row_str}")
                lines.append("")  # Blank line between channels
            if num_channels == 10:
                lines.append(f"Channel 11 discovery_horizons:")
                header = "       " + " ".join(f"Co{col:02d}" for col in range(width))
                lines.append(header)
                for row_idx, row in enumerate(arr[9]):
                    formatted_values = []
                    for val in row:
                        val_float = float(val)
                        if round(val_float, 2) == 0.0:
                            formatted_values.append("----")
                        elif round(val_float, 4) == 0.5125: # tile energy unknown
                            formatted_values.append("----")
                        else:
                            formatted_values.append(f"{val_float:.2f}")
                    row_str = " ".join(formatted_values)
                    lines.append(f"Row {row_idx:02d}: {row_str}")
                lines.append("")  # Blank line between channels
            return "\n".join(lines)
        
        with open(file_path, "w") as f:
            f.write(f"Match Number: {match_number}\n")
            f.write(f"Match Step: {match_step}\n\n")
            
            f.write("=== Additional Features ===\n")
            f.write(f"{additional_features}\n\n")
            
            f.write("=== Team Points ===\n")
            f.write(f"Team Points: {team_points}\n")
            f.write(f"Opponent Team Points: {opp_team_points}\n")
            f.write(f"Points Increase: {points_increase}\n\n")

            f.write("=== Relic Logic ===\n")
            f.write(f"self.ALL_RELICS_FOUND: {self.ALL_RELICS_FOUND}\n")
            f.write(f"self.relics_set: {self.relics_set}\n\n")
            f.write(f"self.reward_set: {self.reward_set}\n\n")
            f.write(f"self.reward_tracking: {self.reward_tracking}\n\n")

            f.write("=== Value ===\n")
            f.write(f"{value}\n\n")
            
            f.write("=== Actions ===\n")
            f.write(f"{actions}\n\n")
            
            f.write("=== Log Probs ===\n")
            f.write(f"{log_probs}\n\n")
            
            f.write("=== Valid Ships ===\n")
            f.write(f"{valid_ships}\n\n")
            
            f.write("=== Ship Rewards ===\n")
            f.write(f"{ship_rewards}\n\n")


            f.write("=== State Information ===\n")
            f.write(f"Observation: {obs}\n")

            f.write("=== State Tensor ===\n")
            f.write(format_3d_array(state))
            f.write("\n\n")
            
            for i, ship_state in enumerate(ship_states):
                f.write(f"=== Ship State {i} ===\n")
                f.write(format_3d_array(ship_state))
                f.write("\n\n")

            # if match_step == 100:
            #     trajectories = self.buffer.get_trajectories()
            #     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            #     print('buffer l    ',len(self.buffer))
            #     print('team_points     ', trajectories['team_points'])
            #     print('ship_rewards     ', trajectories['ship_rewards'])
            #     print('ship_rewards sums    ', trajectories['ship_rewards'].sum(axis=1))    
            #     print('points_increase     ', trajectories['points_increase'])
            #     print('ship_rewards     ', ship_rewards)
            #     print('dones     ', trajectories['dones'])

            #     self.show_explored_energy_field()
            #     self.show_visible_map()
            #     self.show_explored_map()
            #     self.show_exploration_map() 
            #     self.show_visible_energy_field()

    def _load_networks(self, path):
        if self.use_multiple_actors:
            for i in range(16):
                self.actors[i].load_state_dict(torch.load(os.path.join(path, f"actor_{i}.pt"), map_location=self.device))
                self.optimizer_actors[i].load_state_dict(torch.load(os.path.join(path, f"optimizer_actor_{i}.pt"), map_location=self.device))
        else:
            self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pt"), map_location=self.device))
            self.optimizer_actor.load_state_dict(torch.load(os.path.join(path, "optimizer_actor.pt"), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pt"), map_location=self.device))
        self.optimizer_critic.load_state_dict(torch.load(os.path.join(path, "optimizer_critic.pt"), map_location=self.device))
  
    def _flip_features(self, actions, additional_features=None):
        '''
            inflammation Mark
            flipped_grid = grid[::-1, ::-1].T
            state = state[:, ::-1, ::-1].swapaxes(1, 2)
        '''
        if additional_features is not None:
            additional_features[3] = 1.0 - additional_features[3]
        action_map = {1: 3, 3: 1, 2: 4, 4: 2}
        return additional_features





        # def _recompute_ship_log_probs_multiple_actors(self, valid_ships_mask, ship_states_batch, actions_batch):
        #     """
        #     Computes the new log probabilities and entropies for each ship in the mini-batch,
        #     processing all valid ships in parallel.

        #     Returns:
        #         new_log_probs_tensor: Tensor of log probabilities for valid ships (1D)
        #         new_entropies_tensor: Tensor of entropies for valid ships (1D)
        #     """
        #     ##########################################################################
        #     # Multiple Actors
        #     ##########################################################################
        #     # We'll compute logits per ship index, then stack along a new ship dimension.
        #     logits_list = []
        #     for ship_idx in range(16):
        #         # Shape: (batch_size, *ship_state_shape)
        #         ship_states_i = ship_states_batch[:, ship_idx]
        #         ship_states_i = ship_states_i[valid_ships_mask[ship_idx]]
        #         if self.player == "player_1":
        #             ship_states_i = ship_states_i[:, ::-1, ::-1]#.swapaxes(1, 2) ?
        #         logits_i = self.actors[ship_idx](ship_states_i)
        #         if self.player == "player_1":
        #             indices = torch.tensor([0, 3, 4, 1, 2], device=self.device)
        #             logits_i = logits_i[:, indices]
        #         logits_list.append(logits_i)
        #     # logits_all: shape (batch_size, num_ships, output_dim)
        #     logits_all = torch.stack(logits_list, dim=1)
        #     # Flatten to shape (batch_size*num_ships, output_dim)
        #     new_logits = logits_all.reshape(-1, logits_all.shape[-1])
        #     # shape (num_valid_ships, num_actions)
        #     new_probs = torch.nn.functional.softmax(new_logits, dim=-1)
        #     # shape (num_valid_ships, )
        #     new_log_probs = torch.log(new_probs.gather(1, actions)).squeeze()
        #     new_entropies = -(new_probs * torch.log(new_probs + 1e-10)).sum(dim=-1).squeeze()

        #     # shape (num_valid_ships, ) (num_valid_ships, )
        #     return new_log_probs, new_entropies

        # def _compute_ship_policy_logs(self, batch_idxs, valid_ships, ship_states, actions,  advantages):
        #     """
        #     Helper method to compute the new log probabilities and entropies for each ship 
        #     within each sample in the mini-batch.

        #     Returns:
        #         new_log_probs_tensor: Tensor of log probabilities (1D)
        #         new_entropies_tensor: Tensor of entropies (1D)
        #         valid_mask_tensor: Boolean tensor mask indicating valid ships
        #         ship_adv_tensor: Tensor of advantage values per ship (1D)
        #     """
        #     new_log_probs_list = []
        #     new_entropies_list = []
        #     valid_mask = []
        #     ship_adv_list = []

        #     # each sample index in the mini-batch.
        #     for i in batch_idxs:
        #         # Retrieve per-sample data.
        #         valid_ships = valid_ships[i]   # Expected shape: (num_ships, 3)
        #         ship_states = ship_states[i]   # Expected shape: (num_ships, *ship_state_shape)
        #         # add_feat = trajectories['additional_features'][i].reshape(1, -1)  # shape (1,10)
        #         actions = actions[i]           # shape (16,3)  

        #         # valid_ships = trajectories['valid_ships'][i]  # Expected shape: (num_ships, 3)
        #         # ship_states = trajectories['ship_states'][i]      # Expected shape: (num_ships, *ship_state_shape)
        #         # # add_feat = trajectories['additional_features'][i].reshape(1, -1)  # shape (1,10)
        #         # actions = trajectories['actions'][i]                        # shape (16,3)  

        #         # Iterate over each ship in this sample 
        #         for ship_idx in range(16):
        #             is_valid = (valid_ships[ship_idx][0] == 1.0)
        #             valid_mask.append(is_valid)
        #             # Use the same advantage for every ship in this sample.
        #             ship_adv_list.append(advantages[i]) #[ship_idx]) for now, each ship gets the same advantage
        #             if is_valid:
        #                 # ship_feat = np.concatenate([
        #                 #     add_feat,
        #                 #     np.array((node_x, node_y)).reshape(1, 2)
        #                 # ], axis=1)
        #                 # Optionally, you might use node_x, node_y if needed; here we use the ship state.
        #                 ship_state_np = ship_states[ship_idx].reshape(1, *self.ship_state_shape)
        #                 if self.player == "player_1":
        #                     ship_state_np = ship_state_np[:, ::-1, ::-1].swapaxes(1, 2) 
        #                 ship_state = torch.as_tensor(ship_state_np, dtype=torch.float32, device=self.device)
        #                 # ship_feat_tensor = torch.tensor(
        #                 #     ship_feat.reshape(1, 15), dtype=torch.float32, device=self.device)
        #                 if self.use_multiple_actors:
        #                     logits = self.actors[ship_idx](ship_state)
        #                 else:
        #                     logits = self.actor(ship_state)
        #                 if self.player == "player_1":
        #                     indices = torch.tensor([0, 3, 4, 1, 2])
        #                     logits = logits[:, indices]
        #                 probs = F.softmax(logits, dim=-1)
        #                 # Retrieve the stored action for this ship.
        #                 stored_action = int(actions[ship_idx][0])
        #                 stored_action_tensor = torch.tensor([[stored_action]], dtype=torch.int64, device=self.device)
        #                 log_prob = torch.log(probs.gather(1, stored_action_tensor)).squeeze()
        #                 new_log_probs_list.append(log_prob)
        #                 # Calculate the policy entropy.
        #                 entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).squeeze() 
        #                 # possible values 0.0-log(num_actions) 1.61 for 5 actions
        #                 # Minimum Entropy (0): This occurs when one outcome has a probability of 1 (Max- all 0.2)
        #                 new_entropies_list.append(entropy)
        #     new_log_probs_tensor = torch.stack(new_log_probs_list) 
        #     new_entropies_tensor = torch.stack(new_entropies_list) 
        #     valid_mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device)
        #     ship_adv_tensor = torch.as_tensor(ship_adv_list, dtype=torch.float32, device=self.device)

        #     return new_log_probs_tensor, new_entropies_tensor, valid_mask_tensor, ship_adv_tensor

