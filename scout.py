from simulator import *
import numpy as np


'''
	Notes : Use the env variable and its helper functions to 
	1. get the points for a set of player ids
	2. get the cost for a  set of player ids
'''

profiles = [{'cols': ['stats.minutes'],
              'order': [False],
              'prob_dist': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]},
            {'cols': ['stats.own_goals', 'stats.yellow_cards', 'stats.red_cards'],
              'order': [True, True, True],
              'prob_dist': [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0]},
            {'cols': ['stats.ict_index'],
              'order': [False],
              'prob_dist': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
            {'cols': ['selected_by_percent'],
              'order': [False],
              'prob_dist': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]},
            {'cols': ['saves_goal_conceded_ratio',
              'stats.saves',
              'stats.clean_sheets',
              'stats.penalties_saved',
              'stats.penalties_missed'],
              'order': [False, False, False, False, True],
              'prob_dist': [1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]}]


class Scout():

	def __init__(self, env:FPLSimulator, week_idx:int, min_balance:float, current_balance:float, k:int=15, multi_transfer=False):
		# multi transfer allows multiple transfers to happen in a game week
		self.env = env
		self.week_idx = week_idx
		self.min_balance = min_balance
		self.multi_transfer = multi_transfer
		self.current_balance = current_balance
		self.k = k

	def find_transfer_out_candidates(self):
		'''
			This functions finds the candidates in our FPL manager team to transfer out in that particular game week
			Parameters:
			-----------
			k : the top K players who should be transferred out
			-----------
			returns ndarray of players_ids . shape :(k,)

		'''
		k = self.k
		assert(k <= self.env.running_player_ids.shape[0])
		ROI = self.env.running_player_points/self.env.running_player_cost
		indices = np.argsort(ROI, 0)
		sorted_ids_based_on_ROI = self.env.running_player_ids[:,self.week_idx][indices[:,self.week_idx]][:k]
		return sorted_ids_based_on_ROI
  
	def find_transfer_in_candidates(self, player_profile_idx:int, transfer_out_candidates:np.ndarray):
		'''
			This functions finds the candidates in our FPL manager team to transfer out in that particular game week
			Parameters:
			-----------
			player_profile_idx : int : index of profile from profiles variable
			-----------
			returns ndarray of players_ids . shape :(k,)

		'''
		# print(self.env.all_week_data[self.week_idx].columns)
		k = self.k
		assert(k <= self.env.running_player_ids.shape[0])
		profile = profiles[player_profile_idx]
		# in_ids = np.setdiff1d(self.env.all_week_data[self.week_idx].sort_values(by=profile['cols'], ascending=profile['order']).index,transfer_out_candidates)[:k]
		rankings_table = self.env.all_week_data[self.week_idx].sort_values(by=profile['cols'], ascending=profile['order']).index
		in_ids = rankings_table[~np.in1d(rankings_table,transfer_out_candidates)][:k]
		return in_ids


	def get_transfer_in_out_players(self, transfer_in_candidates:np.ndarray, transfer_out_candidates:np.ndarray):
		'''
		 This function takes two sets of player candidates and uses their (cost, type, roi) for that game week to find the perfect pair of players to be transferred in and out. 
		 The pair of players (p_in, p_out). roi = (points / cost)
		 Parameters:
		 -----------
			balance : int . The current remaining balance for the manager
			transfer_in_candidates:np.ndarray . shape : (k,) : the ids of players returned from the find_transfer_in_candidates function
			transfer_out_candidates:np.ndarray . shape : (k,): the ids of players returned from the find_transfer_out_candidates function
		 -----------

		returns : ndarray : shape (N_t,15,10) . transfer in out matrix where N_t is the number of transfers
				  balance : the readjusted balance of the FPL team
		'''
		#print(transfer_in_candidates.shape , transfer_out_candidates.shape)
		balance = self.current_balance

		assert(transfer_in_candidates.shape == transfer_out_candidates.shape)
		transfer_in_candidates = np.broadcast_to(transfer_in_candidates[:,np.newaxis] , (transfer_in_candidates.shape[0], self.env.current_week)) # (K,10)
		transfer_out_candidates = np.broadcast_to(transfer_out_candidates[:,np.newaxis] , (transfer_out_candidates.shape[0], self.env.current_week)) # (K,10)

		all_player_types = self.env.all_player_other_data[self.env.all_player_other_data_cols.index("element_type")] #(620,10)

		transfer_in_candidates_cost = self.env.get_player_info_matrix(self.env.all_player_cost, transfer_in_candidates)[:,self.week_idx] # (K,)
		transfer_in_candidates_types = self.env.get_player_info_matrix(all_player_types, transfer_in_candidates)[:,self.week_idx] # (K,)
		transfer_out_candidates_cost = self.env.get_player_info_matrix(self.env.all_player_cost, transfer_out_candidates)[:,self.week_idx] # (K,)
		transfer_out_candidates_types = self.env.get_player_info_matrix(all_player_types, transfer_out_candidates)[:,self.week_idx] # (K,)

		in_out_type_match_mask = transfer_in_candidates_types[:,np.newaxis] == transfer_out_candidates_types[np.newaxis, :] #(K,K)

		transfer_out_candidates_balance_after_out = transfer_out_candidates_cost + balance - self.min_balance #(K,)
		in_out_cost_diff = transfer_in_candidates_cost[np.newaxis,:] - transfer_out_candidates_balance_after_out[:,np.newaxis] # (K,K)
		in_out_cost_match_mask =  in_out_cost_diff < 0 # (K,K)

		#print(in_out_type_match_mask.shape, in_out_cost_match_mask.shape)
		p_in_idxs, p_out_idxs = np.where(in_out_type_match_mask & in_out_cost_match_mask > 0)
		p_in_ids = transfer_in_candidates[:,self.week_idx][p_in_idxs] # (k_,)
		p_out_ids = transfer_out_candidates[:,self.week_idx][p_out_idxs] # (k_,)

		assert(p_in_ids.shape == p_out_ids.shape)
		#print(p_in_ids, p_out_ids)
		#print(in_out_cost_diff)
		
		if not self.multi_transfer:
			transfer_in_out_mat = np.zeros((1,)+self.env.running_player_ids.shape)
			transfer_in_out_mat[0,self.env.running_player_ids[:,self.week_idx] == p_out_ids[0], self.week_idx] = p_in_ids[0]

			remaining_balance = np.abs(in_out_cost_diff[p_in_idxs[0], p_out_idxs[0]]) + self.min_balance
			#assert(remaining_balance >= self.min_balance)

		return transfer_in_out_mat, remaining_balance