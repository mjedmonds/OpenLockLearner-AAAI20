"""
This file loads subject data from a file and replays the actions the agent took.

Used to create videos of executions
"""
import glob
import re
import jsonpickle
import json
import numpy as np


from openlockagents.OpenLockLearner.io.data_loader import load_subject_data_from_json
from openlockagents.agent import Agent
from openlockagents.OpenLockLearner.util.common import decode_bad_jsonpickle_str
from openlock.common import Action


def main():
    # subject_dir = "/home/mark/Desktop/Mass/OpenLockLearningResultsTesting/subjects/ce3_ce4_example"
    # subject_dir = "/home/mark/Desktop/Mass/OpenLockLearningResultsTesting/subjects/ce4_ce4_example"
    # subject_dir = "/home/mark/Desktop/Mass/OpenLockLearningResultsTemp/subjects/108712014730923731"
    subject_dir = "/home/mark/Desktop/full_model_v2_action_in_BU_2_FINAL/CC3-CC4_subjects/541877814633114558"
    # subject_dir = "/home/mark/Desktop/Mass/OpenLockLearningResultsTesting/subjects/human_example"

    replay_subject_data(subject_dir)

    print("all done!")


def replay_subject_data(subject_dir):
    subject_data = load_subject_data_from_json(subject_dir, use_json_pickle_for_trial=False)

    print("Replaying subject {}".format(subject_data.subject_id))

    max_attempts = 999999999
    # subject has agent, more recent model data
    if hasattr(subject_data, "agent"):
        train_scenario_name = subject_data.agent["params"]["train_scenario_name"]
        train_action_limit = subject_data.agent["params"]["train_action_limit"]
        train_attempt_limit = max_attempts
        test_action_limit = subject_data.agent["params"]["test_action_limit"]
        test_attempt_limit = max_attempts
        lever_index_mode = "position"
    else:
        train_scenario_name = subject_data.trial_seq[0]["scenario_name"]
        train_action_limit = 3
        train_attempt_limit = max_attempts
        test_action_limit = 3
        test_attempt_limit = max_attempts
        # human subjects use role
        lever_index_mode = "role"

    # minimal construction of params
    params = dict()
    params["use_physics"] = True
    params["train_scenario_name"] = train_scenario_name
    params["src_dir"] = None

    env = Agent.pre_instantiation_setup(params, bypass_confirmation=True)
    env.lever_index_mode = lever_index_mode
    # setup dummy window so we can start recording
    env.setup_trial(
            "CC3",
            action_limit=3,
            attempt_limit=30,
            multiproc=False,
        )
    env.reset()

    input("Press enter to start")

    # for each trial, setup the env and execute all of the action sequences
    for trial in subject_data.trial_seq:
        trial_scenario_name = trial["scenario_name"]
        # 3 lever trial
        if trial_scenario_name == "CE3" or trial_scenario_name == "CC3":
            action_limit = train_action_limit
            attempt_limit = train_attempt_limit
        # 4 lever trial
        elif trial_scenario_name == "CE4" or trial_scenario_name == "CC4":
            action_limit = test_action_limit
            attempt_limit = test_attempt_limit
            # corrects a bug where some 4 lever testing trials did not properly save their trial name as a string
            if not isinstance(trial["name"], str):
                # we can directly reencode the bytes back into a python string
                raw_dtype, raw_bytes = jsonpickle.decode(json.dumps(trial["name"]["py/reduce"][1]))
                trial["name"] = decode_bad_jsonpickle_str(raw_bytes)
        else:
            raise ValueError("Unknown scenario name")

        # setup the env for the trial
        env.setup_trial(
            trial_scenario_name,
            action_limit=action_limit,
            attempt_limit=attempt_limit,
            specified_trial=trial["name"],
            multiproc=False,
        )

        # go through every attempt in the trial
        for attempt_seq in trial["attempt_seq"]:
            env.reset()
            # go through every action sequence in this attempt
            action_seq = attempt_seq["action_seq"]
            done = False
            action_num = 0
            # render in a loop
            while not done:
                # execute the next action if an action is not currently executing
                if env.action_executing is False:
                    action_str = action_seq[action_num]["name"]
                    action_env = env.action_map[action_str]
                    env.step(action_env)
                    action_num += 1

                env.render(env)
                done = env.determine_attempt_finished()



    print('hi')




if __name__ == "__main__":
    main()