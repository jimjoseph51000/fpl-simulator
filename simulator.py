import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

import requests, json
from pprint import pprint
from IPython.core.debugger import set_trace
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
random.seed(11)
np.random.seed(11)
# For each FPL Manager

'''
  This is the transfer simulator code 
'''


class BaseSimulator():
  # this part of the simulator is for every interaction between agent's action and environment
  def __init__(self, state_dim , action_dim ):
    self.state = None
    # self.action = None
    self.state_dim = state_dim
    self.action_dim = action_dim
  
  def reset(self):
    '''
      Returns a random state vector 
    '''
    # self.state =  np.random.randn(self.state_dim) # (N,)
    # self.state = self.state / np.linalg(self.state)

    # self.state = self.actual_player_ids
    self.state = self.create_one_hot_embedding(self.actual_players_ids)
    self.state = self.state[:,0] # first week
    return self.state
  

  
  def create_one_hot_embedding(self, player_ids):
    # self.all_player_ids # (620,)
    # player_ids # (15,10)
    
    # for axis = 0 every value is repeated so comp1[0] = 0, comp1[1] = 1 . . . 
    comp1 = np.broadcast_to(self.all_player_ids[:,np.newaxis,np.newaxis], (self.all_player_ids.shape[0],)  + player_ids.shape) # (620,15,10)
    comp2 = np.broadcast_to(player_ids[np.newaxis,:,:], (self.all_player_ids.shape[0],)  + player_ids.shape) # (620,15,10)
    comp_mask = comp1 == comp2 # (620,15,10)
    comp_mask = comp_mask.sum(axis = 1) # (620,10)
    assert(np.all(comp_mask.astype(np.int) <= 1))
    assert(comp_mask.shape == (self.all_player_ids.shape[0], player_ids.shape[1])) # (620,10) one hot encoded
    return comp_mask

  def step(self, action:np.ndarray, week_idx:int = 0):
    '''
        action : ndarray shape:(N,). this is passed to get_transfer_in / out function to get the in/out players 
        week_idx : int 
        
    '''
    #TODO: writing dummy code for now just to simulate

    # 1. the recruiter has already given us a profile which is the action
    # 2. find a transfer_in players which considers this profile
    # 3. find transfer_out_players
    # 4. do the transfer
    
    # new_team_player_ids = np.array(self.running_player_ids)
    # print(self.all_player_points.shape, self.running_player_ids.shape)
    running_player_points = self.get_player_info_matrix(self.all_player_points, self.running_player_ids)
    
    # the action here has to instigate a transfer. lets do random for now independant of action #TODO: change this code
    sample_transfer_ins = self.get_swapped_in_players_test(self.running_player_ids, game_week = week_idx)
    
    self.transfers_in_episode.append(sample_transfer_ins) # IMP : this step is needed but change this to the actual transfer
    
    self.running_player_ids, new_running_player_points, _ = self.do_transfer(sample_transfer_ins, self.running_player_ids)
    
    self.state = self.create_one_hot_embedding(self.running_player_ids) # IMP : this step is needed
    self.state = self.state[:,week_idx]
    
    rewards = self.compute_rewards_matrix(running_player_points, new_running_player_points)
    r = rewards[week_idx]
    done = True if week_idx == (self.current_week-1) else False
    
    return self.state, r, done
  
  # 7. we need to have rewards in the simulator (check the diagram in the progress report)
  def compute_rewards_matrix(self, before_team_points , after_team_points):
    '''
      before_team_points : Team before transfer , ndarray shape: (15,10)
      after_team_points : Team after the transfer : (15,10)

      returns : reward : ndarray . shape (10,)
    '''
    rewards = np.repeat(-1, self.current_week) # (10,)
    rewards = after_team_points.sum(axis=0) - before_team_points.sum(axis=0)
    rewards[rewards <= 0] = -1 # play with these values too. TODO. check the fpl docs too
    rewards[rewards > 0] = 0
    return rewards
  


class FPLSimulator(BaseSimulator):

  def __init__(self, current_week, fpl_manager_id, req_cols = [], state_dim = 10, action_dim = 5):
    super(FPLSimulator, self).__init__(state_dim, action_dim)
    self.current_week = current_week
    self.fpl_manager_id = fpl_manager_id
    self.budget = 100
    self.all_player_ids = None
    self.all_player_cost = None
    self.all_player_points = None
    self.all_player_other_data_cols = req_cols
    self.all_player_other_data = None
    
    self.actual_players_ids = None
    self.actual_players_points = None
    self.actual_player_cost = None
    self.actual_player_other_data = None
    
    self.transfers_in_episode = []
    self.running_player_ids = None
    
    self.init_fpl_team()

  def reset(self):
      self.transfers_in_episode = []
      return super(FPLSimulator, self).reset()
    
  def init_fpl_team(self):
    #1. load from the CSV
    all_week_data = self.load_all_player_weekwise_data(self.current_week)
    
    #2. get the team
    self.actual_players_ids = self.get_players_of_manager(self.fpl_manager_id, self.current_week) # (15, W)
    #3. creating the ids, points and cost for all  players
    #creating a dummy cost matrix for now
    self.all_player_ids = np.unique(np.concatenate([np.unique(all_week_data[i].index) for i in range(len(all_week_data))])) # (620,)
    # self.all_player_cost = np.random.normal(5, 1, size=(self.all_player_ids.shape[0], len(all_week_data))).round(2) # (620,10)
    self.all_player_cost = self.load_all_player_cost_from_csv() # (620,10)
    self.all_player_points = np.zeros((self.all_player_ids.shape[0], len(all_week_data)))
    self.all_player_points, self.all_player_other_data = self.get_data_for_all_players(all_week_data, self.all_player_ids)
    print(self.all_player_ids.shape, self.all_player_points.shape, self.all_player_cost.shape, self.all_player_other_data.shape) # this is our universe
    
    # actual_players_points = get_points_for_players(all_week_data, actual_players_ids) # (15,W) before code
    #4. creating the ids, points and cost for actual players
    self.actual_players_points = self.get_player_info_matrix(self.all_player_points, self.actual_players_ids)
    per_week_total_points = self.actual_players_points.sum(axis=0) #(W,)
    print('cumsum of per_week_total_points: ',np.cumsum(per_week_total_points))
    self.actual_player_cost = self.get_player_info_matrix(self.all_player_cost, self.actual_players_ids)
    
    self.actual_player_other_data = []
    for i in range(len(self.all_player_other_data_cols)):
      self.actual_player_other_data.append(self.get_player_info_matrix(self.all_player_other_data[i], self.actual_players_ids))
    self.actual_player_other_data = np.array(self.actual_player_other_data)
    
    print(self.actual_players_ids.shape, self.actual_players_points.shape, self.actual_player_cost.shape, self.actual_player_other_data.shape) # this is our tuple
    
    self.running_player_ids = np.array(self.actual_players_ids)
    
    return 
  
  
  def load_from_csv_player_types(self):
    df_type = pd.read_csv("player_types.csv", index_col=0)
    df_type = df_type.set_index("id")
    return df_type
  
  def load_all_player_weekwise_data(self, current_week:int):
    '''
    returns list of dataframes (W,)
    '''
    all_week_data = []
    player_types_df = self.load_from_csv_player_types()
    for week in range(1,current_week+1):
      df = pd.read_csv("Players_Weekwise/week_"+str(week)+".csv")

      df = df.set_index('id')
      df = df.join(player_types_df, on='id', how='left')
      df = df[['stats.total_points'] + self.all_player_other_data_cols]
      all_week_data.append(df)

    return all_week_data

  def load_all_player_cost_from_csv(self):
    all_player_cost = pd.read_csv("Player_Cost_Weekwise/all_player_costs.csv")
    all_player_cost = all_player_cost.T.iloc[1:,:self.current_week]
    all_player_cost.index = all_player_cost.index.map(int)
    res = all_player_cost.loc[self.all_player_ids,:] # (620,10)
    # print(np.array(res.head()))
    return np.array(res)
    

  # def get_points_for_players(all_week_data: list, player_ids : np.ndarray):
  #   '''
  #   player_ids : (15,W) player ids for W game weeks

  #   returns : (15,W) ndarray of player points
  #   '''
  #   assert(len(all_week_data) == player_ids.shape[1])
  #   players_points = []
  #   for i in range(len(all_week_data)):
  #     players_points.append(all_week_data[i].loc[player_ids[:,i], :]['stats.total_points'])
  #   return np.array(players_points).T

#   def get_points_for_all_players(self, all_week_data: list, player_ids : np.ndarray):
#     all_player_points = np.zeros((self.all_player_ids.shape[0], len(all_week_data)))
#     for i in range(len(all_week_data)):
#       # cur_player_ids = np.unique(np.array(all_week_data[i].index))
#       cur_player_ids = np.unique(np.array(all_week_data[i].index)) # (N,)
#       act_P_reshaped = np.broadcast_to(cur_player_ids[:,np.newaxis], (cur_player_ids.shape[0],self.all_player_ids.shape[0])) # (N, 620)
#       all_P_reshaped = np.broadcast_to(self.all_player_ids[np.newaxis, :], (cur_player_ids.shape[0],self.all_player_ids.shape[0]) )# (N,620)
#       match_idx = np.argwhere(act_P_reshaped == all_P_reshaped) # this should have all the matches, lets do an assertion check
#       # match_idx[:,-1].reshape
#       assert(match_idx.shape == (cur_player_ids.shape[0],2)) # (N,2)
#       act_match_idx = match_idx[:,-1]
#       all_player_points[act_match_idx,i] = all_week_data[i].loc[cur_player_ids, :]['stats.total_points']
#     return all_player_points

  def get_data_for_all_players(self, all_week_data: list, player_ids : np.ndarray):
    all_player_points = np.zeros((self.all_player_ids.shape[0], len(all_week_data)))
    all_player_other_data = np.zeros((len(self.all_player_other_data_cols), self.all_player_ids.shape[0], len(all_week_data)) , dtype=np.object)
    for i in range(len(all_week_data)):
      # cur_player_ids = np.unique(np.array(all_week_data[i].index))
      cur_player_ids = np.unique(np.array(all_week_data[i].index)) # (N,)
      act_P_reshaped = np.broadcast_to(cur_player_ids[:,np.newaxis], (cur_player_ids.shape[0],self.all_player_ids.shape[0])) # (N, 620)
      all_P_reshaped = np.broadcast_to(self.all_player_ids[np.newaxis, :], (cur_player_ids.shape[0],self.all_player_ids.shape[0]) )# (N,620)
      match_idx = np.argwhere(act_P_reshaped == all_P_reshaped) # this should have all the matches, lets do an assertion check
      # match_idx[:,-1].reshape
      assert(match_idx.shape == (cur_player_ids.shape[0],2)) # (N,2)
      act_match_idx = match_idx[:,-1]
      all_player_points[act_match_idx,i] = all_week_data[i].loc[cur_player_ids, :]['stats.total_points']
      for j,col in enumerate(self.all_player_other_data_cols):
        all_player_other_data[j,act_match_idx,i] = all_week_data[i].loc[cur_player_ids, :][col]
    return all_player_points, all_player_other_data

  def get_players_of_manager(self, manager_id:int, current_week:int):
    '''
    return (15,W) ndarray of players for manager_id
    '''
    player_ids = []
    for week in range(1,current_week+1):
      r = requests.get('https://fantasy.premierleague.com/api/entry/'+manager_id+'/event/'+str(week)+'/picks/').json()
      player_ids.append([x['element'] for x in r['picks']])
    return np.array(player_ids).T


  def get_swapped_in_players(self, actual_player_ids, num_tranfers = 8):
    '''
      TODO: 
      this is the output from the scout model based on suggestion from the recruiter NN model
      returns a (N_t, 15, 10) ndarray of transfers
      1.this array will have only one point set. Only one cell
      2.the value will be the player in for that game week
      3.the value of team matrix at that index will be the player_out
      4. transfer_ins must be in order of transfer . ie; week(transfer_ins[0]) < week(transfer_ins[1]) < .... < week(transfer_ins[N_t])
    '''
    N_t = np.random.randint(1, num_tranfers) # dummy value for now
    random_game_weeks = np.random.choice(np.arange(self.current_week), N_t) # use replace = False if we dont want multiple transfers in same game week
    random_game_weeks = sorted(random_game_weeks) # (N_t,)
    random_in_players = np.random.choice(self.all_player_ids, N_t, replace=False)
    random_out_players = np.random.choice(np.arange(actual_player_ids.shape[0]), N_t, replace=False)
    transfer_ins = np.zeros((N_t, ) + actual_player_ids.shape)
    # transfer_ins[0,10,5] = 23
    # transfer_ins[1,12,6] = 24
    # transfer_ins[2,4,7] = 265
    
    transfer_ins[np.arange(N_t), random_out_players, random_game_weeks] = random_in_players
    return transfer_ins

  def get_swapped_in_players_test(self, actual_player_ids, num_transfers = 1, game_week = 0):
    '''
      this will just produce a random transfer for that game week
      TODO: 
      this is the output from the scout model based on suggestion from the recruiter NN model
      returns a (N_t, 15, 10) ndarray of transfers
      1.this array will have only one point set. Only one cell
      2.the value will be the player in for that game week
      3.the value of team matrix at that index will be the player_out
      4. transfer_ins must be in order of transfer . ie; week(transfer_ins[0]) < week(transfer_ins[1]) < .... < week(transfer_ins[N_t])
    '''
    N_t = num_transfers 
    # random_game_weeks = np.random.choice(np.arange(self.current_week), N_t) # use replace = False if we dont want multiple transfers in same game week
    # random_game_weeks = sorted(random_game_weeks) # (N_t,)
    random_game_weeks = np.array([game_week])
    assert(num_transfers == 1) # otherwise the below statement wont work
    other_players_ids = np.setdiff1d(self.all_player_ids,actual_player_ids[:,game_week])
    random_in_players = np.random.choice(other_players_ids, N_t, replace=False)
    random_out_players = np.random.choice(np.arange(actual_player_ids.shape[0]), N_t, replace=False)
    transfer_ins = np.zeros((N_t, ) + actual_player_ids.shape)
    # transfer_ins[0,10,5] = 23
    # transfer_ins[1,12,6] = 24
    # transfer_ins[2,4,7] = 265
    
    transfer_ins[np.arange(N_t), random_out_players, random_game_weeks] = random_in_players
    return transfer_ins


  def get_swapped_out_players(self, actual_player_ids):
    '''
      TODO: 
      this is the output from some logic 
      returns a (N_t, 15, 10) ndarray of transfers
      1.this array will have only one point set. Only one cell
      2.the value will be the player in for that game week
      3.the value of team matrix at that index will be the player_out
      4. transfer_ins must be in order of transfer . ie; week(transfer_ins[0]) < week(transfer_ins[1]) < .... < week(transfer_ins[N_t])
    '''
    N_t = np.random.randint(8) # dummy value for now
    transfer_outs = np.zeros((N_t, ) + actual_player_ids.shape)
    '''
      WRITE CODE HERE
    '''
    
    return transfer_outs


  def get_player_info_matrix(self, all_player_info, actual_players_ids):
    '''
      This is a generic function to retrieve the cost or points of the player_ids
      get the player index position in the all player id array. we need this to get the cost of the
      actual players from the all player cost array. this is a bit complicated but the fastest way to compare
      all_player_info : ndarray shape = (620,10)

    '''
    assert(all_player_info.shape == (self.all_player_ids.shape[0], self.current_week))
    act_P_reshaped = np.broadcast_to(actual_players_ids[:,:,np.newaxis], actual_players_ids.shape + (self.all_player_ids.shape[0], ) ) # (15, 10, 620)
    all_P_reshaped = np.broadcast_to(self.all_player_ids[np.newaxis, np.newaxis,:], actual_players_ids.shape + (self.all_player_ids.shape[0], ) )# (15, 10, 620)
    match_idx = np.argwhere(act_P_reshaped == all_P_reshaped) # this should have all the matches, lets do an assertion check
    assert(match_idx.shape == (actual_players_ids.reshape(-1).shape[0],3))
    # just see how hte 
    act_to_all_match_idx =  match_idx[:,-1].reshape((15,10))
    act_to_all_match_idx # (15,10)

    actual_player_info = all_player_info[act_to_all_match_idx, np.broadcast_to(np.arange(self.current_week)[np.newaxis, :]\
                                                                            ,act_to_all_match_idx.shape)]
    return actual_player_info


  def do_transfer(self, transfer_ins, actual_players_ids):
    '''
      transfer_ins : ndarray . shape = (N_t,15,10) 
      actual_players_ids : ndarray . shape = (15,10), 
      actual_players_points : ndarray . shape = (15,10), 
      actual_player_cost : ndarray . shape = (15,10)

      returns actual_players_ids, actual_players_points, actual_player_cost after applying the transfer in and out
    '''

    # 1. get the transfer weeks
    transfer_week_idxs = np.argmax(transfer_ins.sum(axis=1), axis=1) # (N_t,)

    # 2 . create the replicator mask for the transfers
    # well the transfer at a point means, we assume that the team formed after the transfer continue until end week
    # this is because all the actual transfers occuring after that will be bogus. we can only look at one week and see if a transfer is possible
    # to just replace the trajectory of tranferred out player in the team with the player_in is not simple
    replicator_masks = np.zeros(transfer_ins.shape) # (N_t,15,10) 
    for i, w in enumerate(transfer_week_idxs):
      replicator_masks[i,:,np.arange(w,actual_players_ids.shape[-1])] = 1
    
    # the replicator masks will just extend out the two matrices : actual_players_ids and transfer_ins
    replicated_actual_players_ids = replicator_masks * actual_players_ids[:,transfer_week_idxs[0]][np.newaxis,:,np.newaxis] # (N_t,15,10) 
    replicated_player_transfers = replicator_masks * transfer_ins[np.arange(transfer_ins.shape[0]),:,transfer_week_idxs][:,:,np.newaxis] # (N_t,15,10)
    
    # 3. create the transfer index matrix which has the transfer idx to be copied to that cell
    trf_broad = np.broadcast_to(np.arange(transfer_ins.shape[0])[:,np.newaxis,np.newaxis], transfer_ins.shape).copy()
    trf_broad[replicated_player_transfers == 0] = 0 # (N_t,15,10)
    transfer_order_idxs = np.argmax(trf_broad,axis=0) # (15,10)
    ww,pp = np.meshgrid(np.arange(actual_players_ids.shape[1]),\
                np.arange(actual_players_ids.shape[0]))
    
    # 4. now that we have the transfer_order_idxs we have to step by step sum up the matrices : player_ids and transfers
    # this indexing below will give you a flattened matrix with value corresponding to what transfer occured. This is done by the transfer_order_idxs matrix
    replicated_actual_players_ids = replicated_actual_players_ids[transfer_order_idxs, pp, ww] # (15,10)
    replicated_player_transfers = replicated_player_transfers[transfer_order_idxs, pp, ww] # (15,10)
    replicated_actual_players_ids[replicated_player_transfers > 0] = replicated_player_transfers[replicated_player_transfers > 0] # step here
    
    # 5. these are our new set of variables
    new_team_player_ids = np.array(actual_players_ids)
    new_team_player_points = np.zeros_like(new_team_player_ids)
    new_team_player_cost = np.zeros_like(new_team_player_ids)

    new_team_player_ids[replicated_actual_players_ids > 0] = replicated_actual_players_ids[replicated_actual_players_ids > 0] # step here
    
    # 6. we need to have the points and the cost matrix for the new team too
    new_team_player_points = self.get_player_info_matrix(self.all_player_points, new_team_player_ids)
    new_team_player_cost = self.get_player_info_matrix(self.all_player_cost, new_team_player_ids)
    
    assert(new_team_player_ids.shape == new_team_player_points.shape == new_team_player_points.shape ) # (15,10)
    # return new_team_player_ids, new_team_player_points, new_team_player_cost, replicated_player_transfers

    return new_team_player_ids, new_team_player_points, new_team_player_cost
  
  def todo_function(self):
    # 8. TODO : budget and cost constraints need to be implemented

    '''
    --->CODE HERE
    def get_swapped_out_mat(actual_players_ids):
      # get swapped out player : ndarray.shape = (15,10) where 0 for not swapped and player_id for swapped out
    def get_swapped_in_mat(actual_players_ids):
      # get swapped in player : ndarray.shape = (15,10) where 0 for not swapped and player_id for swapped in

    assert np.all((swapped_out_players > 0) & (swapped_in_players > 0) == False)


    swapped out player -> adjust budget , cost matrix and points matrix
    swapped in player -> adjust budget , cost matrix and points matrix
    # 

    '''
  def sample_visualization(self, num_proj = 10000, sample_projs = []):
    # 5 . lets see the simulate in action by having a few sets of transfers. These few sets of transfers are random strategies. Each strategy has some random transfers.
    # lets visualize some plots for these. 
    # we can call a trajectory as a set of transfers. We decide on a trajectory based on a strategy or decision. Look at the progress report to understand the terms and 
    # the whole pipeline
    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    plt.plot(self.actual_players_points.sum(axis=0), 'r-', label='actual')
    plt.title('Points Per week')
    plt.subplot(1,2,2)
    plt.title('Cumsum Points')
    actual_cumsum = np.cumsum(self.actual_players_points.sum(axis=0))
    plt.plot(actual_cumsum, 'r-', label= 'actual')

    actually_better_plots = []
    # here 10000 random projections 
    plots_final_points = []
    for i in tqdm(range(num_proj)):
      if len(sample_projs) > 0:
        sample_transfer_ins = sample_projs[i]
      else:
        sample_transfer_ins = self.get_swapped_in_players(self.actual_players_ids)
      new_team_player_ids, new_team_player_points, new_team_player_cost  = self.do_transfer(sample_transfer_ins, self.actual_players_ids)
      
      plt_label = ''
      plt_line = 'b--'
      new_cumsum = np.cumsum(new_team_player_points.sum(axis=0))
      if new_cumsum[-1] > actual_cumsum[-1]: # TODO : change this back
      # if True:
        plots_final_points.append(new_cumsum[-1])
        plt_label = 'random transfer in-out {}'.format(i+1)
        plt_line = 'g--'

        plt.subplot(1,2,1)
        plt.plot(new_team_player_points.sum(axis=0), plt_line, label = plt_label)
        plt.subplot(1,2,2)
        plt.plot(new_cumsum, plt_line, label=plt_label)
        actually_better_plots.append(i)
      # print(transfer_ins_flatten)
      # plt.pause(0.05)

    plt.xlabel('weeks')
    plt.ylabel('points')
    # plt.legend()
    plt.show()

    print('actually_better_plots : {}'.format(len(actually_better_plots)))
    return plots_final_points

  def sample_visualization2_transfer_after_transfer(self, num_proj = 10000):
    # 5 . lets see the simulate in action by having a few sets of transfers. These few sets of transfers are random strategies. Each strategy has some random transfers.
    # lets visualize some plots for these. 
    # we can call a trajectory as a set of transfers. We decide on a trajectory based on a strategy or decision. Look at the progress report to understand the terms and 
    # the whole pipeline
    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    plt.plot(self.actual_players_points.sum(axis=0), 'r-', label='actual')
    plt.title('Points Per week')
    plt.subplot(1,2,2)
    plt.title('Cumsum Points')
    actual_cumsum = np.cumsum(self.actual_players_points.sum(axis=0))
    plt.plot(actual_cumsum, 'r-', label= 'actual')

    actually_better_plots = []
    # here 10000 random projections 
    
    new_team_player_points = np.array(self.actual_players_points)
    new_team_player_cost = np.array(self.actual_player_cost)
    plots_final_points = []
    all_proj = []
    for i in tqdm(range(num_proj)):
      proj = []
      new_team_player_ids = np.array(self.actual_players_ids)
      for w in range(0,self.current_week):
        sample_transfer_ins = self.get_swapped_in_players_test(self.actual_players_ids, game_week = w)
        proj.append(sample_transfer_ins)
        assert(sample_transfer_ins.shape == (1,15,10))
        new_team_player_ids, new_team_player_points, new_team_player_cost  = self.do_transfer(sample_transfer_ins, new_team_player_ids)
      proj = np.array(proj).sum(axis=1)
      all_proj.append(proj)
  
      plt_label = ''
      plt_line = 'b--'
      new_cumsum = np.cumsum(new_team_player_points.sum(axis=0))
      if new_cumsum[-1] > actual_cumsum[-1]: #TODO : change this back
      # if True:
        plots_final_points.append(new_cumsum[-1])
        plt_label = 'random transfer in-out {}'.format(i+1)
        plt_line = 'g--'

        plt.subplot(1,2,1)
        plt.plot(new_team_player_points.sum(axis=0), plt_line, label = plt_label)
        plt.subplot(1,2,2)
        plt.plot(new_cumsum, plt_line, label=plt_label)
        actually_better_plots.append(i)
      # print(transfer_ins_flatten)
      # plt.pause(0.05)

    plt.xlabel('weeks')
    plt.ylabel('points')
    # plt.legend()
    plt.show()

    print('actually_better_plots : {}'.format(len(actually_better_plots)))
    return all_proj , plots_final_points