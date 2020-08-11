import sys
import atexit
import time

from openlockagents.Human.human_agent import HumanAgent
from openlockagents.common.agent import Agent

from openlock.settings_scenario import select_scenario
from openlock.settings_trial import PARAMS, IDX_TO_PARAMS
from openlock.common import generate_effect_probabilities
import openlockagents.Human.common as common

# def exit_handler(signum, frame):
#    print 'saving results.csv'
#    np.savetxt('results.csv', env.results, delimiter=',', fmt='%s')
#    exit()


def run_specific_trial_and_scenario(
    agent, scenario_name, trial_name, action_limit, attempt_limit
):
    agent.run_trial_human(
        scenario_name, action_limit, attempt_limit, specified_trial=trial_name
    )
    agent.finish_subject()
    agent.write_agent()
    sys.exit(0)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        # general params
        # training params
        # PICK ONE and comment others
        params = PARAMS['CE3-CE4']
        # params = PARAMS['CE3-CC4']
        # params = PARAMS['CC3-CE4']
        # params = PARAMS['CC3-CC4']
        # params = PARAMS['CE4']
        # params = PARAMS["CC4"]
    else:
        setting = sys.argv[1]
        # pass a string or an index
        try:
            params = PARAMS[IDX_TO_PARAMS[int(setting) - 1]]
        except Exception:
            params = PARAMS[setting]

    human_config_data = common.load_human_config_json()

    # params["data_dir"] = os.path.dirname(ROOT_DIR) + "/OpenLockResults/subjects"
    params["data_dir"] = human_config_data["HUMAN_SAVE_DIR"]
    params["src_dir"] = "/tmp/openlocklearner/" + str(hash(time.time())) + "/src/"
    params["use_physics"] = True
    params["effect_probabilities"] = generate_effect_probabilities()

    # this section randomly selects a testing and training scenario
    # train_scenario_name, test_scenario_name = select_random_scenarios()
    # params['train_scenario_name'] = train_scenario_name
    # params['test_scenario_name'] = test_scenario_name

    scenario = select_scenario(params["train_scenario_name"])

    # todo: this should not be part of OpenLockLearnerAgent
    env = Agent.pre_instantiation_setup(params)
    env.lever_index_mode = "role"

    # create session/trial/experiment manager
    agent = HumanAgent(params, env)

    atexit.register(agent.cleanup)

    # used for debugging, runs a specific scenario & trial
    # run_specific_trial_and_scenario(manager, 'CC3', 'trial5', params['train_action_limit'], params['train_attempt_limit'])

    for trial_num in range(0, params["train_num_trials"]):
        agent.run_trial_human(
            params["train_scenario_name"],
            params["train_action_limit"],
            params["train_attempt_limit"],
            verify=True,
        )
        print("One trial complete for subject {}".format(agent.subject_id))

    # testing trial
    # print "INFO: STARTING TESTING TRIAL"
    if params["test_scenario_name"] is not None:
        scenario = select_scenario(params["test_scenario_name"])
        # run testing trial with specified trial7
        agent.run_trial_human(
            params["test_scenario_name"],
            params["test_action_limit"],
            params["test_attempt_limit"],
            specified_trial="trial7",
            test_trial=True,
        )
        print("One trial complete for subject {}".format(agent.subject_id))

    agent.env.render(agent.env, close=True)  # close the window
    print("The experiment is over. Thank you for participating.")
    print("Please answer the following questions:")
    agent.finish_subject()
    print("You are finished. Please alert the RA. Thank you!")
