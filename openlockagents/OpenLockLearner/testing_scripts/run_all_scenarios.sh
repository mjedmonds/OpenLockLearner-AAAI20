#!/bin/bash

SAVEDIR="$1"
ABLATIONS="${@:2}"
# SCENARIOS=("CE3-CE4" "CE3-CC4" "CC3-CE4" "CC3-CC4" "CC4" "CE4")
# SCENARIOS=("CE3-CE4" "CE3-CC4" "CC3-CE4" "CC3-CC4")
# SCENARIOS=("CC4" "CE4")

source ~/Developer/virtualenv/OpenLockAgents/bin/activate

# move up to the root directory of OpenLockAgents
cd ../../../

# set to the directory of OpenLockAgents and OpenLock (in this case, relative to the root directory of OpenLockAgents)
OPENLOCKAGENTSPATH="./"
OPENLOCKPATH="../OpenLock/"

# add root of OpenLockAgents to pythonpath
export PYTHONPATH=${PYTHONPATH}:${OPENLOCKAGENTSPATH}:${OPENLOCKPATH}

for scenario in "${SCENARIOS[@]}"
do
  subject_dir="${SAVEDIR}/${scenario}_subjects"
  base_cmd="python openlockagents/OpenLockLearner/main/openlock_learner.py --savedir=${subject_dir} --scenario=${scenario} --bypass_confirmation"
  # if we don't have ablations, don't execute with them
  if [ -z ${ABLATIONS} ];
  then
    cmd=${base_cmd}
  # exceute with ablations
  else
    cmd="${base_cmd} --ablations=${ABLATIONS}"
  fi
  echo ${cmd}
  ${cmd} &
done

