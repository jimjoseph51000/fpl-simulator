# fpl-simulator

Parameters  to work on :

 1. Reward function is based on current vs new team difference. So what if no team transfer takes place is it reward 0 and we pass a zero vector action profile to NN (so that say no transfer takes place). Or do we set reward to -1 and make it transfer with some profile
	
	So r = 0 when prev_team.points >= new_team.points else r = -1 with epsilon a little high to prevent the NN from learning no transfers are the way to go
	    Or  r = 0 when  prev_team.points > new_team.points else r = -1 with little bit of epsilon

2 . Should we have a state vector which includes player points vector , and current week as one hot ? 

3 . Each transfer itself is also an episode right? So we need to do some special processing for this? If thats the case we need to some combinatorial selection of transfers and make them an episode?




TODOs:
complete the step function
The state vector should be replicated. Also, the network doesnt output the state. So how will you get the next state vector? This has to be some form of embedding. 

