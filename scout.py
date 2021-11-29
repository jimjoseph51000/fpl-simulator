from simulator import *
import numpy as np


'''
	Notes : Use the env variable and its helper functions to 
	1. get the points for a set of player ids
	2. get the cost for a  set of player ids
'''


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

	#anshuman
	def find_transfer_in_candidates(self, k:int, player_profile:np.ndarray):
		'''
		use clustering or manually defined set of mutually exclusive list of players. Each set of players are of one profile. 
		This function returns top K players of one set which is mapped to the player_profile
		Parameters:
		-----------
		k : the top K players who should be transferred out
		player_profile: shape : (N,) . This is a softmax prob distributed 1-D array for player profile. This is one of the input to the recruiter model and
									transfer matches
		-----------
			returns ndarray of players_ids . shape :(k,)
		
		'''
		pass

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

