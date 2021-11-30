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

	def __init__(self, env:FPLSimulator, week_idx:int):
		self.env = env
		self.week_idx = week_idx

	#priyanka
	def find_transfer_out_candidates(self, k:int):
		'''
			This functions finds the candidates in our FPL manager team to transfer out in that particular game week
			Parameters:
			-----------
			k : the top K players who should be transferred out
			-----------
			returns ndarray of players_ids . shape :(k,)

		'''
		pass

	def find_transfer_in_candidates(self, k:int, player_profile_idx:int):
		'''
			This functions finds the candidates in our FPL manager team to transfer out in that particular game week
			Parameters:
			-----------
			k : the top K players who should be transferred in
			-----------
			returns ndarray of players_ids . shape :(k,)

		'''
		# print(self.env.all_week_data[self.week_idx].columns)
		profile = profiles[player_profile_idx]
		return list(self.env.all_week_data[self.week_idx].sort_values(by=profile['cols'], ascending=profile['order']).index)[:k]

	#priyanka
	def get_transfer_in_out_players(self, balance:float, transfer_in_candidates:np.ndarray, transfer_out_candidates:np.ndarray):
		'''
		 This function takes two sets of player candidates and uses their (cost, type, roi) for that game week to find the perfect pair of players to be transferred in and out. 
		 The pair of players (p_in, p_out). roi = (points / cost)
		 Parameters:
		 -----------
			balance : int . The current remaining balance for the manager
			transfer_in_candidates:np.ndarray . shape : (k,) : the ids of players returned from the find_transfer_in_candidates function
			transfer_out_candidates:np.ndarray . shape : (k,): the ids of players returned from the find_transfer_out_candidates function
		 -----------
		returns : tuple : (p_in,p_out) : p_in is the player id of transfer in,  p_out is the player id of transfer out
				  balance : the readjusted balance of the FPL team
		'''

		pass

