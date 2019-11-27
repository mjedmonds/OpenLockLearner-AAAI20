First, make sure to export the PYTHONPATH correctly:

export PYTHONPATH=/home/usr/work/OpenLock 
export PYTHONPATH=/home/usr/work/OpenLockAgents:$PYTHONPATH
Then, run the code with the follow commend:

python a3c-lstm-open-lock.py 1 reward_negative_change_state_partial_seq_solution_multiplier_door_seq

The last option is the reward mode, check reward.py to see full options.
