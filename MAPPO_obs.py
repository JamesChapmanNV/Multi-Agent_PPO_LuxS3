import numpy as np
import random
import os
from collections import namedtuple

discovery_ch_idx       = 0
reward_ch_idx          = 1
astroid_ch_idx         = 2
tile_energy_ch_idx     = 3
friendly_ch_idx        = 4
friendly_energy_ch_idx = 5
enemy_ch_idx           = 6
enemy_energy_ch_idx    = 7
relic_ch_idx           = 8

Experience = namedtuple('Experience', ['state',
                                       'additional_features',
                                       'actions', 
                                       'log_probs', 
                                       'team_points',
                                       'opp_team_points', 
                                       'points_increase',
                                       'opp_points_increase',
                                       'value', 
                                       'done', 
                                    #    'next_state',
                                       'valid_ships',
                                       'ship_states',
                                       'ship_rewards',])

class ReplayBuffer:
    def __init__(self, capacity, state_shape, ship_state_shape, dtype=np.float32):
        self.capacity = capacity
        self.state_shape = state_shape
        self.ship_state_shape = ship_state_shape
        self.action_shape = (16,3)
        self.additional_features_shape = 10
        self.position = 0
        self.full = False
        
        self.states = np.zeros((capacity, 9,24,24), dtype=dtype)
        #self.states = np.zeros((capacity, *self.state_shape), dtype=dtype)
        self.additional_features = np.zeros((capacity, self.additional_features_shape), dtype=np.int32)
        self.actions = np.zeros((capacity, *self.action_shape), dtype=np.int32)
        self.log_probs = np.zeros((capacity, 16), dtype=np.float32)
        self.team_points = np.zeros(capacity, dtype=np.float32)
        self.opp_team_points = np.zeros(capacity, dtype=np.float32)
        self.points_increase = np.zeros(capacity, dtype=np.float32)
        self.opp_points_increase = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.uint8)
        #self.next_states = np.zeros((capacity, *self.state_shape), dtype=dtype)
        self.valid_ships = np.zeros((capacity, 16, 3), dtype=dtype)
        self.ship_states = np.zeros((capacity, 16, *self.ship_state_shape), dtype=dtype)
        self.ship_rewards = np.zeros((capacity, 16), dtype=dtype)

    def __len__(self):
        return self.capacity if self.full else self.position

    def append(self, experience):
        if self.position >= self.capacity:
            print("Warning Buffer")
            return  

        idx = self.position
        self.states[idx] = experience.state
        self.additional_features[idx] = experience.additional_features
        self.actions[idx] = experience.actions
        self.log_probs[idx] = experience.log_probs
        self.team_points[idx] = experience.team_points
        self.points_increase[idx] = experience.points_increase
        self.opp_points_increase[idx] = experience.opp_points_increase
        self.opp_team_points[idx] = experience.opp_team_points
        self.values[idx] = experience.value
        self.dones[idx] = experience.done
        #self.next_states[idx] = experience.next_state
        self.valid_ships[idx] = experience.valid_ships
        self.ship_states[idx] = experience.ship_states
        self.ship_rewards[idx] = experience.ship_rewards

        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True

    def random_sample(self, batch_size):
        max_index = self.capacity if self.full else self.position
        indices = np.random.choice(max_index, batch_size, replace=False)
        return (
            self.states[indices], 
            self.additional_features[indices], 
            self.actions[indices], 
            self.log_probs[indices], 
            self.team_points[indices], 
            self.points_increase[indices],
            self.opp_points_increase[indices],
            self.opp_team_points[indices], 
            self.values[indices], 
            self.dones[indices], 
            #self.next_states[indices],
            self.valid_ships[indices],
            self.ship_states[indices],
            self.ship_rewards[indices],
        )
    
    def get_trajectories(self):
        last_position = self.capacity if self.full else self.position
        data = {
            'states': self.states[:last_position],
            'additional_features': self.additional_features[:last_position],
            'actions': self.actions[:last_position],
            'log_probs': self.log_probs[:last_position],
            'team_points': self.team_points[:last_position],
            'opp_team_points': self.opp_team_points[:last_position],
            'points_increase': self.points_increase[:last_position],
            'opp_points_increase': self.opp_points_increase[:last_position],
            'values': self.values[:last_position],
            'dones': self.dones[:last_position],
            #'next_states': self.next_states[:last_position],
            'valid_ships': self.valid_ships[:last_position],
            'ship_states': self.ship_states[:last_position],
            'ship_rewards': self.ship_rewards[:last_position],
        }
        # Reset buffer for next trajectory collection
        #self.position = 0
        return data

# global states
def get_state(obs, team_id, match_step, match_number, unknown_energy=0.5125):
    """
        Build an observation tensor of shape (C, 24, 24) from the observation dictionary.

        board[ C, y, x ] = (X, Y)
        board[ C ] = obs[key].T

        MIRROR #  SUBTRACT from 23 and SWITCH x, y !!!
        mirrored_board[ C, 23 - x, 23 - y ] = (X, Y)
        mirrored_board = board[::-1, ::-1].T

        out of date Channel Order:
        : Discovery: 1.0 not yet discovered (visible), else 0.0. (mirrored, binary).
        : Reward status: 1.0 reward, 0.5 potential reward, 0.0 not (mirrored).
        : Relic status: 1.0 relic (mirrored, binary).
        : Astroid type: 1.0 if tile_type == 2 (mirrored, binary).
        : Tile energy: normalized as (energy+20)/ 40 (mirrored).
        : Friendly ship density: normalized count: min(count/10, 1.0).
        : Friendly ship energy: normalized sum of energies on tile (min(total/1200, 1.0)).
        : Enemy ship density: normalized count: min(count/10, 1.0).
        : Enemy ship energy: normalized sum of energies on tile (min(total/1200, 1.0)).
    """
    state =  [] 
    opp_team_id = 1 - team_id

    # ---------------------------
    # Channel: Visible BINARY
    # ---------------------------
    # state.append(obs["sensor_mask"].astype(np.float32).T)

    # ---------------------------
    # Channel: Discovery for now, not visible==1 BINARY MIRRORED
    # --------   later give reward for discovery==1
    # ---------------------------
    discovery_ch_idx = 0
    visible = obs["sensor_mask"].astype(np.float32).T
    visible_flip = visible[::-1, ::-1].T
    visible_sym = np.maximum(visible, visible_flip) 
    state.append(1 - visible_sym)

    # ---------------------------
    # Channel: Reward status (all zeros for now) MIRRORED
    # ---------------------------
    reward_ch_idx = 1
    state.append(np.zeros((24, 24), dtype=np.float32))

    # ---------------------------
    # Channel: Astroid type BINARY MIRRORED
    # ---------------------------
    astroid_ch_idx = 2
    tile_type = obs["map_features"]["tile_type"]
    asteroid = (tile_type == 2).astype(np.float32).T
    asteroid_flip = asteroid[::-1, ::-1].T
    asteroid_sym = np.maximum(asteroid, asteroid_flip) 
    state.append(asteroid_sym)

    # ---------------------------
    # Channel: Tile energy MIRRORED
    # ---------------------------
    tile_energy_ch_idx = 3
    map_energy = obs["map_features"]["energy"].astype(np.float32).T
    sensor_mask = obs["sensor_mask"].astype(np.float32).T
    # map_energy == -1 can mean unknown energy or actually -1
    mask = (map_energy == -1) & (sensor_mask == 0)
    map_energy[mask] = -1000 # Set unknown energy to -1000 temporary
    tile_energy_channel = np.zeros((24, 24), dtype=np.float32)
    for x in range(24):
        for y in range(24):
            if x + y <= 23:
                val1 = map_energy[x, y] 
                val2 = map_energy[23 - y, 23 - x] # observation mirrored
                # if known energies are different, raise an error
                if val1 > -40 and val2 > -40 and not np.isclose(val1, val2):
                    print('energy mismatch at', x, y, val1, val2)
                    raise ValueError("Energy mismatch")
                max_val = max(val1, val2) # max of -1000
                norm_val = (max_val + 20) / 40.0 if max_val > -40 else unknown_energy
                # norm_val = (max_val) / 20.0 if max_val > -40 else 0.0
                tile_energy_channel[x, y] = round(norm_val, 4)
                tile_energy_channel[23 - y, 23 - x] = round(norm_val, 4) # mirror 
    state.append(tile_energy_channel)

    # ---------------------------
    # Channel: Friendly ship density
    # ---------------------------
    friendly_ch_idx = 4
    friendly_density_channel = np.zeros((24, 24), dtype=np.float32)
    friendly_positions = obs["units"]["position"][team_id]
    friendly_energies = obs["units"]["energy"][team_id]
    for i, energy in enumerate(friendly_energies):
        if energy < 0:
            friendly_positions[i] = [-1, -1]
    # ensure first step is zeros
    if match_step > 0: 
        for (x, y) in friendly_positions:
            if x < 0 or y < 0 or x >= 24 or y >= 24:
                continue
            friendly_density_channel[int(y), int(x)] += 1
        friendly_density_channel = np.minimum(friendly_density_channel / 10.0, 1.0)
    state.append(friendly_density_channel)

    # ---------------------------
    # Channel : Friendly ship energy / 1200
    # ---------------------------
    friendly_energy_ch_idx = 5
    friendly_energy_channel = np.zeros((24, 24), dtype=np.float32)
    if match_step > 0: # ensure first step is zeros
        for i, (x, y) in enumerate(friendly_positions): 
            if x < 0 or y < 0 or x >= 24 or y >= 24:
                continue
            friendly_energy_channel[int(y), int(x)] += friendly_energies[i]  
        friendly_energy_channel = np.minimum(friendly_energy_channel / 1200.0, 1.0)
    state.append(friendly_energy_channel)

    # ---------------------------
    # Channel : Enemy ship density
    # ---------------------------
    enemy_ch_idx = 6
    enemy_density_channel = np.zeros((24, 24), dtype=np.float32)
    enemy_positions = obs["units"]["position"][opp_team_id]
    if match_step > 0: # ensure first step is zeros
        for (x, y) in enemy_positions:
            if x < 0 or y < 0 or x >= 24 or y >= 24:
                continue
            enemy_density_channel[int(y), int(x)] += 1
        enemy_density_channel = np.minimum(enemy_density_channel / 10.0, 1.0)
    state.append(enemy_density_channel)

    # ---------------------------
    # Channel : Enemy ship energy / 1200
    # ---------------------------
    enemy_energy_ch_idx = 7
    enemy_energy_channel = np.zeros((24, 24), dtype=np.float32)
    enemy_energies = obs["units"]["energy"][opp_team_id]
    if match_step > 0: # ensure first step is zeros
        for i, (x, y) in enumerate(enemy_positions):
            if x < 0 or y < 0 or x >= 24 or y >= 24:
                continue
            enemy_energy_channel[int(y), int(x)] += enemy_energies[i]
        enemy_energy_channel = np.minimum(enemy_energy_channel / 1200.0, 1.0)
    state.append(enemy_energy_channel)

        # ---------------------------
    # Channel: Relic status BINARY MIRRORED
    # ---------------------------
    relic_ch_idx = 8
    relic_channel = np.zeros((24, 24), dtype=np.float32)
    for (x, y) in obs["relic_nodes"]:
        if 0 <= x < 24 and 0 <= y < 24: # skip -1, -1
            relic_channel[int(y), int(x)] = 1.0
            # SUBTRACT from 23 and SWITCH x, y !!! in
            relic_channel[23 - int(x), 23 - int(y)] = 1.0
    state.append(relic_channel)

    # # ---------------------------
    # # Channel: Nebula type BINARY MIRRORED
    # # ---------------------------
    # tile_type = obs["map_features"]["tile_type"]
    # nebula = (tile_type == 1).astype(np.float32).T # switch X and Y!
    # nebula_flip = nebula[::-1, ::-1].T
    # nebula_sym = np.maximum(nebula, nebula_flip) 
    # state.append(nebula_sym)

    return np.stack(state, axis=0)

def update_relic_discovery_logic(agent, state, match_step, match_number):
    # 1. Check for new relic this step (2: Relic) '''
    if state[relic_ch_idx].sum() > len(agent.relics_set): # new 
        for y, x in np.argwhere(np.isclose(state[relic_ch_idx], 1.0)): # (y, x)
            relic_coord = (int(y),int(x))
            if relic_coord not in agent.relics_set:
                agent.relics_set.add(relic_coord) # (y, x)
                agent.relics_found_this_step.add(relic_coord)
    # 2. Find number of pairs/times a relic was found (sometimes on centerline) '''                      
    num_relic_pairs = len(agent.relics_set)//2 # up to 3 pairs 
    for (y, x) in agent.relics_set:
        state[relic_ch_idx, y, x] = 1.0 # (ensure that the relic is visible)
        if x + y == 23: # on centerline
            num_relic_pairs += 1
    # 3. Check for relics found so far this match ''' # resets each match 
    agent.relics_found_this_match = (num_relic_pairs == (match_number + 1))
    # 4. Check for ALL relics found '''
    agent.ALL_RELICS_FOUND = (num_relic_pairs == 3)
    if num_relic_pairs < match_number: # last match no relic found
        # If entire map has been scene (all zeros) and at least one ship is present,
        if not state[discovery_ch_idx].any() and np.any(state[friendly_ch_idx] > 0.00001):
            agent.ALL_RELICS_FOUND = True 
    #5. if relic(s) found: discovery --> 0.0's this match '''
    if agent.relics_found_this_match or agent.ALL_RELICS_FOUND:
        state[discovery_ch_idx] = np.zeros((24, 24), dtype=np.float32) # disable discovery rewards
    elif match_step >= 50: 
    ##############################################################################
    # # now keep track of where we have looked           #########################
        if agent.last_state is not None:
            state[discovery_ch_idx] = np.minimum(state[discovery_ch_idx], agent.last_state[discovery_ch_idx])
    ##############################################################################
    # no relic found this match yet, 
    # AND it is still possible for a relic to show up (match_step < 50)
    # discovery is all not visible
    ##############################################################################
    return state

def update_reward_logic(agent, state, match_step, match_number, points_increase, obs):
    '''
        agent.reward_tracking 
            a list of dicts with keys
            "points_from_potential_tiles": an integer indicating how many rewards points
            "possible_coords": a set of candidate coordinates (potential reward tiles)
        RUN 4 functions 
            -> process_possible_coords(possible_coords, points_from_potential_tiles):
            -> process_new_known_rewards(new_known_rewards)
            -> process_not_reward_tiles(new_NOT_rewards)
            -> process_elimination_rules()
    '''
    # error checking for environment seed = 3856378
    points3856378 = {
        # Original 17 points, flipped:
        (0, 4),   # was (4, 0)
        (0, 5),   # was (5, 0)
        (2, 3),   # was (3, 2)
        (2, 5),   # was (5, 2)
        (8, 7),   # was (7, 8)
        (9, 11),  # was (11, 9)
        (10, 9),  # was (9, 10)
        (11, 8),  # was (8, 11)
        (11, 9),  # was (9, 11)
        (11, 10), # was (10, 11)
        (11, 11), # was (11, 11)
        (12, 7),  # was (7, 12)
        (12, 8),  # was (8, 12)
        (12, 9),  # was (9, 12)
        (12, 10), # was (10, 12)
        (13, 8),  # was (8, 13)
        (13, 9),  # was (9, 13)

        # Transformed 17 points, flipped:
        (19, 23), # was (23, 19) from (4, 0)
        (18, 23), # was (23, 18) from (5, 0)
        (20, 21), # was (21, 20) from (3, 2)
        (18, 21), # was (21, 18) from (5, 2)
        (16, 15), # was (15, 16) from (7, 8)
        (12, 14), # was (14, 12) from (11, 9)
        (14, 13), # was (13, 14) from (9, 10)
        (15, 12), # was (12, 15) from (8, 11)
        (14, 12), # was (12, 14) from (9, 11)
        (13, 12), # was (12, 13) from (10, 11)
        (12, 12), # was (12, 12) from (11, 11)
        (16, 11), # was (11, 16) from (7, 12)
        (15, 11), # was (11, 15) from (8, 12)
        (14, 11), # was (11, 14) from (9, 12)
        (13, 11), # was (11, 13) from (10, 12)
        (15, 10), # was (10, 15) from (8, 13)
        (14, 10)  # was (10, 14) from (9, 13)
    }
    

    def process_possible_coords(possible_coords, points_from_potential_tiles):
        """
            If the number of possible coordinates exactly equals the points increase,
                then each candidate is a known reward tile and is processed immediately.
            if there are tiles but no points, then all are known to NOT be rewards.
            Otherwise, the step is stored in reward_tracking.
        """
        if len(possible_coords) > 0 and points_from_potential_tiles == len(possible_coords): 
            # all are now known reward tiles
            process_new_known_rewards(possible_coords)

        elif len(possible_coords) > 0 and points_from_potential_tiles == 0:
            # all are now unknown to NOT be rewards
            process_not_reward_tiles(possible_coords)

        elif len(possible_coords) > points_from_potential_tiles and points_from_potential_tiles > 0:
            # some are reward tiles, but we don't know which yet
            new_entry = {
                "points_from_potential_tiles": points_from_potential_tiles,
                "possible_coords": possible_coords
            }
            if new_entry not in agent.reward_tracking:
                agent.reward_tracking.append(new_entry)
        elif len(possible_coords) > 0 or points_from_potential_tiles < 0:
            filepath = os.path.join("runs", "add_step_logs", f"{match_step}_{match_number}_{random.randint(1000, 9999)}.txt")
            with open(filepath, "a") as f:
                f.write("-----------------------------------\n")
                f.write(f"match_number, {match_number}\n")
                f.write(f"match_step, {match_step}\n")
                f.write(f"points_increase, {points_increase}\n")
                f.write(f"points_from_potential_tiles, {points_from_potential_tiles}\n")
                f.write(f"agent.reward_set, {agent.reward_set}\n")
                f.write(f"known_rewards_mask, {known_rewards_mask}\n")
                f.write(f"possible_coords, {possible_coords}\n")
                f.write(f"potential_reward_mask, {potential_reward_mask}\n")
                f.write(f"agent.reward_tracking, {agent.reward_tracking}\n")
                f.write(f"state[reward_ch_idx], {state[reward_ch_idx]}\n")
                f.write(f"state[friendly_ch_idx], {state[friendly_ch_idx]}\n")
                f.write(f"obs, {obs}\n")
                f.write("-----------------------------------\n")
                f.write("-----------------------------------\n")

    def process_new_known_rewards(new_known_rewards):
        """
            - If new_tile is in the candidate set, remove it and subtract one from points.
            - If points become zero, then every remaining candidate in that entry is marked as not a reward.
            After processing, also update any entry by removing all known not-reward tiles.
        """
        for (y, x) in new_known_rewards:
            # if (y, x) not in points3856378:
            #     print('-----------------------------------')
            #     print(y, x) 
            #     filepath = os.path.join("runs", "add_step_logs", f"{match_step}_{match_number}_{random.randint(1000, 9999)}.txt")
            #     with open(filepath, "a") as f:
            #         f.write("--------------process new known---------------------\n")
            #         f.write(f"match_number, {match_number}\n")
            #         f.write(f"match_step, {match_step}\n")
            #         f.write(f"points_increase, {points_increase}\n")
            #         f.write(f"points_from_potential_tiles, {points_from_potential_tiles}\n")
            #         f.write(f"agent.reward_set, {agent.reward_set}\n")
            #         f.write(f"known_rewards_mask, {known_rewards_mask}\n")
            #         f.write(f"possible_coords, {possible_coords}\n")
            #         f.write(f"potential_reward_mask, {potential_reward_mask}\n")
            #         f.write(f"agent.reward_tracking, {agent.reward_tracking}\n")
            #         f.write(f"state[reward_ch_idx], {state[reward_ch_idx]}\n")
            #         f.write(f"state[friendly_ch_idx], {state[friendly_ch_idx]}\n")
            #         f.write(f"obs, {obs}\n")
            #         f.write("-----------------------------------\n")
            #         f.write("-----------------------------------\n")
            #     agent.error_info = "this shouldn't happen new_known_rewards not in points"
            state[reward_ch_idx, y, x] = 1.0 # REWARD TILE found
            state[reward_ch_idx, 23-x, 23-y] = 1.0 
            agent.reward_set.add((y, x))
            agent.reward_set.add((23-x, 23-y))

        # Remove coord & deduct point - we know where point came from
        for (y, x) in new_known_rewards:
            for entry in agent.reward_tracking:
                if (y, x) in entry["possible_coords"]:
                    entry["possible_coords"].remove((y, x))
                    entry["points_from_potential_tiles"] -= 1
                    if (y, x) in entry["possible_coords"]:
                        agent.error_info = "DUPLICATES- this shouldn't happen, multiple entry of same coord"

        # we will potentially find some 'not' reward tiles
        NOT_rewards = set() 
        # collect entries that have been resolved 
        reward_tracking_entries_to_remove = []
        # go back through
        for entry in agent.reward_tracking:
            if entry["points_from_potential_tiles"] < 0:
                filepath = os.path.join("runs", "add_step_logs", f"{match_step}_{match_number}_-process new known-{random.randint(1000, 9999)}.txt")
                with open(filepath, "a") as f:
                    f.write("--------------process new known---------------------\n")
                    f.write(f"match_number, {match_number}\n")
                    f.write(f"match_step, {match_step}\n")
                    f.write(f"points_increase, {points_increase}\n")
                    f.write(f"points_from_potential_tiles, {points_from_potential_tiles}\n")
                    f.write(f"agent.reward_set, {agent.reward_set}\n")
                    f.write(f"known_rewards_mask, {known_rewards_mask}\n")
                    f.write(f"possible_coords, {possible_coords}\n")
                    f.write(f"potential_reward_mask, {potential_reward_mask}\n")
                    f.write(f"agent.reward_tracking, {agent.reward_tracking}\n")
                    f.write(f"state[reward_ch_idx], {state[reward_ch_idx]}\n")
                    f.write(f"state[friendly_ch_idx], {state[friendly_ch_idx]}\n")
                    f.write(f"obs, {obs}\n")
                    f.write("-----------------------------------\n")
                    f.write("-----------------------------------\n")
                reward_tracking_entries_to_remove.append(entry)
            if entry["points_from_potential_tiles"] == 0:
                # All remaining candidates in this entry are known to be not reward tiles.
                for (y, x) in entry["possible_coords"]:
                    NOT_rewards.add((y, x))
                reward_tracking_entries_to_remove.append(entry)
        # Remove entries that are fully resolved.
        for entry in reward_tracking_entries_to_remove:
            agent.reward_tracking.remove(entry)
        # Now process not_reward_tiles: remove these from every entry.
        if NOT_rewards:
            process_not_reward_tiles(NOT_rewards)

    def process_not_reward_tiles(NOT_rewards):
        """
            For each reward_tracking entry, remove any coordinate that is known not to be a reward tile.
            After removal, if the number of remaining candidate coordinates equals the points count,
            then all of these candidates become known reward tiles.
            (Note: No subtraction from points is performed here.)
        """
        for (y, x) in NOT_rewards:
            state[reward_ch_idx, y, x] = 0.0 # NOT REWARD TILE found
            state[reward_ch_idx, 23-x, 23-y] = 0.0

        # Remove coord, leave points
        for (y, x) in NOT_rewards:
            for entry in agent.reward_tracking:
                if (y, x) in entry["possible_coords"]:
                    entry["possible_coords"].remove((y, x))
                    if (y, x) in entry["possible_coords"]:
                        agent.error_info = "this shouldn't happen multiple entry of same coord"
        
        # we will potentially find some reward tiles
        new_known_rewards = set() 
        reward_tracking_entries_to_remove = []
        # go back through
        for entry in agent.reward_tracking:
            if entry["points_from_potential_tiles"] < 0:
                filepath = os.path.join("runs", "add_step_logs", f"{match_step}_{match_number}_-process not-{random.randint(1000, 9999)}.txt")
                with open(filepath, "a") as f:
                    f.write("----------------process not-------------------\n")
                    f.write(f"match_number, {match_number}\n")
                    f.write(f"match_step, {match_step}\n")
                    f.write(f"points_increase, {points_increase}\n")
                    f.write(f"points_from_potential_tiles, {points_from_potential_tiles}\n")
                    f.write(f"agent.reward_set, {agent.reward_set}\n")
                    f.write(f"known_rewards_mask, {known_rewards_mask}\n")
                    f.write(f"possible_coords, {possible_coords}\n")
                    f.write(f"potential_reward_mask, {potential_reward_mask}\n")
                    f.write(f"agent.reward_tracking, {agent.reward_tracking}\n")
                    f.write(f"state[reward_ch_idx], {state[reward_ch_idx]}\n")
                    f.write(f"state[friendly_ch_idx], {state[friendly_ch_idx]}\n")
                    f.write(f"obs, {obs}\n")
                    f.write("-----------------------------------\n")
                    f.write("-----------------------------------\n")
                reward_tracking_entries_to_remove.append(entry)
            if len(entry["possible_coords"]) == entry["points_from_potential_tiles"]:
            # If the number of candidates exactly matches the points needed, mark them as known.
                for (y, x) in entry["possible_coords"]:
                    new_known_rewards.add((y, x))
                reward_tracking_entries_to_remove.append(entry)
        # go back through
        for entry in reward_tracking_entries_to_remove:
            agent.reward_tracking.remove(entry)
        if new_known_rewards:
            process_new_known_rewards(new_known_rewards)


    def process_elimination_rules():
        """
            Go back through all possible pairs of entries 
                check for overlapping coordinates (intersection)
                    process the 'differences'
                        unique coordinates (diff )  
                        corresponding points  (diff _points)
                        Use Complementary Clues: If the union of two clue sets has a known total, infer missing rewards.
                        Apply Algebraic Reasoning: When two clues overlap, determine exact reward locations.
            Basically, the 'differences' are new entry of their own (maybe should do that TODO:)
        """
        filepath = os.path.join("runs", "process_elimination_rules_logs", f"{match_step}_{match_number}_{random.randint(1000, 9999)}.txt")
        new_known_rewards = [] 
        NOT_rewards = [] 
        for i in range(len(agent.reward_tracking)):
            for j in range(i+1, len(agent.reward_tracking)):
                coords1 = set(agent.reward_tracking[i]['possible_coords'])
                coords2 = set(agent.reward_tracking[j]['possible_coords'])
                pts1 = agent.reward_tracking[i]['points_from_potential_tiles'] 
                pts2 = agent.reward_tracking[j]['points_from_potential_tiles'] 

                intersection = coords1 & coords2
                if intersection:
                    diff1 = coords1 - intersection # all coords that are in 1 but not in 2
                    diff2 = coords2 - intersection # all coords that are in 2 but not in 1
                    diff1_points = pts1 - pts2
                    diff2_points = pts2 - pts1
                    if diff1:
                        if diff1_points > 0: 
                            if len(diff1) == diff1_points:
                                new_known_rewards.append(diff1) # all are rewards
                        if not diff2: # this means all coords that are in coords2 are in coords1
                            if diff1_points == 0: # some coords are in diff1, no points
                                NOT_rewards.append(diff1) # all are not

                    if diff2:
                        if diff2_points > 0:
                            if len(diff2) == diff2_points:
                                new_known_rewards.append(diff2) # all are rewards
                        if not diff1: # this means all coords that are in coords1 are in coords2
                            if diff2_points == 0:
                                NOT_rewards.append(diff2) 

        # process the new information
        NOT_rewards = set().union(*NOT_rewards)
        if NOT_rewards:
            process_not_reward_tiles(NOT_rewards) # all are not 
        # process the new information
        new_known_rewards = set().union(*new_known_rewards) 
        if new_known_rewards:
            process_new_known_rewards(new_known_rewards) # issue here with recursion?


    # begin
    ''' 1. Update with last step '''
    if agent.last_reward_channel is not None: # persists across matches NOT episodes
        state[reward_ch_idx] = agent.last_reward_channel.copy() 

    ''' 2. ADD POTENTIAL REWARDS (0.5) 5x5 square '''
    if agent.relics_found_this_step:
        for relic_coord in agent.relics_found_this_step:
            # 5x5 square centered around the relic
            for y in range(relic_coord[0] - 2, relic_coord[0] + 3): 
                for x in range(relic_coord[1] - 2, relic_coord[1] + 3):  
                    if 0 <= x < 24 and 0 <= y < 24:
                        if state[reward_ch_idx, y, x] < 0.9: # not if 1.0 (already known reward)
                            if state[reward_ch_idx, 23-x, 23-y] == 1.0: # not if 1.0 (already known reward)
                                agent.error_info = "this shouldn't happen potential reward must match"
                            state[reward_ch_idx, y, x] = 0.5
                            state[reward_ch_idx, 23-x, 23-y] = 0.5 #redundant?
        # TODO: track overlapping relic 5X5 squares and only reset that area
        # TODO: reward the ships that discovered the relic (last ship_rewards) 
        agent.reward_tracking = []
        agent.relics_found_this_step = set() 

    ''' 3. Check for new reward information (possible_coords & points_from_potential_tiles)'''
    # we might not be able to see a relic if it is giving rewards 
    # we cannot be certain where the reward came from
    # if relics range (2) is larger than unit sensor range (possible values: 1-4) -- agent.UNIT_SENSOR_RANGE > 1
    # then again, nebulas can block the view of a relic
    if agent.relics_found_this_match or agent.ALL_RELICS_FOUND:
        # subtract points scored from known reward tiles
        known_rewards_mask = (np.isclose(state[reward_ch_idx], 1.0)) & (state[friendly_ch_idx] > 0.00001)
        points_from_potential_tiles = points_increase - np.sum(known_rewards_mask) # points_from_known_rewards
        # potential reward tiles that have at least one friendly ship
        potential_reward_mask = (np.isclose(state[reward_ch_idx], 0.5)) & (state[friendly_ch_idx] > 0.00001)
        possible_coords = {tuple(map(int, coord)) for coord in np.argwhere(potential_reward_mask)}
        #######################################################
        process_possible_coords(possible_coords, points_from_potential_tiles)
        process_elimination_rules()
        #######################################################
        # will continue until no new information
    ''' 3. Update for next step '''
    agent.last_reward_channel = state[reward_ch_idx].copy()
    return state


# ship states
def get_ship_state(agent, state, ship_index, ship_x, ship_y, ship_energy, actions, ship_discovery_channel):
    """
        0. Update with Discovery Channel 
            (FROM previous get_ship_rewards())
        1. REDO friendly units positions density and energy according to actions SO FAR THIS ROUND
        2. REDO friendly units energy for moving UNIT_MOVE_COST
        3. ADD channel for this ship's CURRENT visibility and energy (at position x, y)
            RUN get_possible_discovery_horizons() 
        4. resize and center 
    """
    ship_state = state.copy()
    # Update Discovery Channel with previous ships actions so far this turn 
    ship_state[discovery_ch_idx] = ship_discovery_channel
    # REDO 2 channels
    friendly_density = np.zeros((24, 24), dtype=np.float32)
    friendly_energy = np.zeros((24, 24), dtype=np.float32)
    for prev_i, (prev_x, prev_y) in enumerate(agent.friendly_positions): # x, y
        prev_energy = agent.friendly_energies[prev_i]
        if prev_x < 0 or prev_y < 0 or prev_x >= 24 or prev_y >= 24: # or prev_energy < -2.0: TODO: is this valid?
            continue # Skip ships that are not on the board
        # adjust (prev_x, prev_y, prev_energy) for previous action this round
        act = int(actions[prev_i][0])
        if act > 0 and act < 5: 
            if prev_i >= ship_index: 
                print('this should never happen -in ship states')
            if act == 1:    # Up 
                prev_y = max((prev_y - 1), 0) # stay on board MAX
            elif act == 2:  # Right
                prev_x = min((prev_x + 1), 23) # stay on board Min
            elif act == 3:  # Down
                prev_y = min((prev_y + 1), 23) 
            elif act == 4:  # Left
                prev_x =  max((prev_x - 1), 0)
            # update energy for this ship
            prev_energy = prev_energy - agent.UNIT_MOVE_COST
        if act == 5: # stay
            prev_energy = prev_energy - agent.UNIT_SAP_COST
        # TODO: account for tile energy change
        friendly_density[int(prev_y), int(prev_x)] += 1
        friendly_energy[int(prev_y), int(prev_x)] += prev_energy
    ship_state[friendly_ch_idx] = np.minimum(friendly_density / 10.0, 1.0) 
    ship_state[friendly_energy_ch_idx] = np.minimum(friendly_energy / 1200.0, 1.0)
    # ADD Channel
    current_ship_channel = get_possible_discovery_horizons(ship_x, ship_y, agent.UNIT_SENSOR_RANGE)
    current_ship_channel[0, ship_y, ship_x] = round(ship_energy/400.0, 4)
    ship_state[7, :, :] = current_ship_channel[0, :, :] # REPLACE last channel
    # ship_state = np.concatenate((ship_state, current_ship_channel), axis=0)
    # Resize and center the map
    ship_state = resize_and_center(ship_state, agent.ship_state_shape, (ship_x, ship_y))
    return ship_state

def resize_and_center(ship_state, new_shape, center):
    """
    Extract an board_sizexboard_size window centered on the given (x, y) coordinate from a map.
    
    If the window goes out-of-bounds, those areas are filled with zeros.
    
    Parameters:
    -----------
    input_map : np.ndarray
        The input map with shape (channels, height, width), e.g. (10, 24, 24).
    center : tuple
        The (x, y) coordinate (with x corresponding to width and y to height) to center on.
    board_size : int, optional
        The size of the window to extract (default 11).
    
    Returns:
    --------
    new_map : np.ndarray
        A new map of shape (channels, board_size, board_size) with the resize_actored view.
    """
    channels, H, W = ship_state.shape
    radius = new_shape[1] // 2  # For board_size=11, radius=5.
    
    # Initialize the new map with zeros.
    new_map = np.zeros((channels, new_shape[1], new_shape[2]), dtype=np.float32)
    # out of bounds
    new_map[astroid_ch_idx, :, :] = 1.0 # set to 1.0 for the astroid channel
    
    # Determine the coordinates of the window in the original map.
    src_x_min = center[0] - radius
    src_x_max = center[0] + radius + 1  # +1 because slicing is exclusive 
    src_y_min = center[1] - radius
    src_y_max = center[1] + radius + 1
    
    # Find the overlapping region between the window and the input map.
    src_x_min_valid = max(src_x_min, 0)
    src_x_max_valid = min(src_x_max, W)
    src_y_min_valid = max(src_y_min, 0)
    src_y_max_valid = min(src_y_max, H)
    
    # Determine where this valid region goes in the new map.
    dest_x_min = src_x_min_valid - src_x_min
    dest_x_max = dest_x_min + (src_x_max_valid - src_x_min_valid)
    dest_y_min = src_y_min_valid - src_y_min
    dest_y_max = dest_y_min + (src_y_max_valid - src_y_min_valid)
    
    # Copy the valid part from input_map into new_map.
    new_map[:, dest_y_min:dest_y_max, dest_x_min:dest_x_max] = \
        ship_state[:, src_y_min_valid:src_y_max_valid, src_x_min_valid:src_x_max_valid]
    
    return new_map

def get_possible_discovery_horizons(ship_x, ship_y, UNIT_SENSOR_RANGE):
    current_ship_channel = np.zeros((1, 24, 24), dtype=np.float32)
    # # Determine the bounds of visibility, ensuring they remain within [0, 23)
    # visibility_y_min = max(0, y - UNIT_SENSOR_RANGE)
    # visibility_y_max = min(23, y + UNIT_SENSOR_RANGE + 1)
    # visibility_x_min = max(0, x - UNIT_SENSOR_RANGE)
    # visibility_x_max = min(23, x + UNIT_SENSOR_RANGE + 1)
    # # Set the corresponding region to ones
    # current_ship_channel[0, visibility_y_min:visibility_y_max, visibility_x_min:visibility_x_max] = 1.0    
    
    # if Up action
    if (ship_y - UNIT_SENSOR_RANGE - 1) >= 0: 
        x_min = max(ship_x - UNIT_SENSOR_RANGE, 0)
        x_max = min(ship_x + UNIT_SENSOR_RANGE + 1, 23)
        y_min = ship_y - UNIT_SENSOR_RANGE - 1 # change y direction
        y_max = y_min + 1
        current_ship_channel[0, y_min:y_max, x_min:x_max] = 1.0
    # if Down action
    if (ship_y + UNIT_SENSOR_RANGE + 1) <= 23:
        x_min = max(ship_x - UNIT_SENSOR_RANGE, 0)
        x_max = min(ship_x + UNIT_SENSOR_RANGE + 1, 23)
        y_min = ship_y + UNIT_SENSOR_RANGE + 1 # change y direction
        y_max = y_min + 1
        current_ship_channel[0, y_min:y_max, x_min:x_max] = 1.0
    # if Right action
    if (ship_x + UNIT_SENSOR_RANGE + 1) <= 23:
        x_min = ship_x + UNIT_SENSOR_RANGE + 1 # change x direction
        x_max = x_min + 1
        y_min = max(ship_y - UNIT_SENSOR_RANGE, 0)
        y_max = min(ship_y + UNIT_SENSOR_RANGE + 1, 23)
        current_ship_channel[0, y_min:y_max, x_min:x_max] = 1.0
    # if Left action
    if (ship_x - UNIT_SENSOR_RANGE - 1) >= 0:
        x_min = ship_x - UNIT_SENSOR_RANGE - 1
        x_max = x_min + 1
        y_min = max(ship_y - UNIT_SENSOR_RANGE, 0)
        y_max = min(ship_y + UNIT_SENSOR_RANGE + 1, 23)
        current_ship_channel[0, y_min:y_max, x_min:x_max] = 1.0

    return current_ship_channel


# used in ship rewards
def get_discovery_horizon(action, radius, UNIT_SENSOR_RANGE):
    if action == 1 or action == 3: # Up Down
        x_min = radius - UNIT_SENSOR_RANGE
        x_max = radius + UNIT_SENSOR_RANGE + 1
        if action == 1:
            y_min = radius - UNIT_SENSOR_RANGE - 1
            y_max = y_min + 1 # single column 
        elif action == 3:
            y_min = radius + UNIT_SENSOR_RANGE + 1
            y_max = y_min + 1 
    elif action == 2 or action == 4:  # Right Left
        if action == 2:
            x_min = radius + UNIT_SENSOR_RANGE + 1
            x_max = x_min + 1 # single row
        elif action == 4:
            x_min = radius - UNIT_SENSOR_RANGE - 1
            x_max = x_min + 1 
        y_min = radius - UNIT_SENSOR_RANGE
        y_max = radius + UNIT_SENSOR_RANGE + 1
    
    return y_min, y_max, x_min, x_max


