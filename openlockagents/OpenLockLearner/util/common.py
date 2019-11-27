import os
import json
import multiprocessing
import heapq
import numpy as np
import argparse
import pickle as pkl

from shutil import copytree, ignore_patterns

from openlockagents.OpenLockLearner.causal_classes.CausalRelation import CausalRelationType

# typedef for values to use during chain generation
GRAPH_INT_TYPE = np.uint8

PARALLEL_MAX_NBYTES = '50000M'

SANITY_CHECK_ELEMENT_LIMIT = 1000000

ALL_CAUSAL_CHAINS = 1


class AblationParams:

    def __init__(self):
        self.TOP_DOWN = False
        self.BOTTOM_UP = False
        self.PRUNING = False
        self.TOP_DOWN_FIRST_TRIAL = False
        self.INDEXED_DISTRIBUTIONS = False
        self.ACTION_DISTRIBUTION = False

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)


def parse_arguments():
    parser = argparse.ArgumentParser("OpenLockLearner")
    parser.add_argument("--savedir", metavar="DIR", type=str, help="directory to save the output of the OpenLockLearner")
    parser.add_argument("--scenario", metavar="XX4", type=str, help="training-testing scenarios. E.g. CE3-CC4. For baselines, use CE4 or CC4")
    parser.add_argument("--ablations", type=str, nargs="+", help="ablations, options are: 'top_down', 'bottom_up', 'pruning'")
    parser.add_argument("--bypass_confirmation", action="store_true")
    args = parser.parse_args()
    return args


def decode_bad_jsonpickle_str(bad_bytes):
    # encode the bad byes as a unicode string
    bad_str = str(bad_bytes, "utf-8")
    # every 4th character is a legitimate character
    chars = []
    for i in range(0, len(bad_str), 4):
        chars.append(bad_str[i])
    return "".join(chars)


def write_source_code(project_src_path, destination_path):
    copytree(
        project_src_path,
        destination_path,
        ignore=ignore_patterns(
            "*.mp4", "*.pyc", ".git", ".gitignore", ".gitmodules"
        ),
    )


def setup_states_actions_attribute_labels(env, scenario_name):
    # setup a dummy trial so we can initialize states, actions, etc based on the state of the simulator
    trial_selected = env.setup_trial(scenario_name, action_limit=3, attempt_limit=3)

    states = env.obj_map.keys()


def setup_actions(states):
    actions = ["push_{}".format(x) for x in states if x is not "door_lock"] + [
        "pull_{}".format(x) for x in states if x is not "door_lock" and x is not "door"
    ]
    return actions


def renormalize(input_arr):
    return input_arr / sum(input_arr)


def verify_valid_probability_distribution(dist):
    return abs(sum(dist) - 1.0) < 0.000001 and min(dist) >= 0


def get_highest_N_values_and_idxs(N, arr, min_value=None):
    return get_highest_N_values(N, arr, min_value), get_highest_N_idxs(N, arr, min_value)


def get_highest_N_values(N, arr, min_value=None):
    return arr[get_highest_N_idxs(N,arr, min_value)]


def get_highest_N_idxs(N, arr, min_value=None):
    result_idxs = []
    if isinstance(arr, np.ndarray):
        result_idxs = heapq.nlargest(N, range(len(arr)), arr.take)
    elif isinstance(arr, list):
        result_idxs = heapq.nlargest(N, range(len(arr)), arr.__getitem__)
    else:
        ValueError("Unexpected array type")
    if min_value is not None:
        result_idxs = [x for x in result_idxs if arr[x] > min_value]
    return result_idxs


def get_lowest_N_values_and_idxs(N, arr, max_value=None):
    return get_lowest_N_values(N, arr, max_value), get_lowest_N_idxs(N, arr, max_value)


def get_lowest_N_values(N, arr, max_value=None):
    return arr[get_lowest_N_idxs(N, arr, max_value)]


def get_lowest_N_idxs(N, arr, max_value=None):
    result_idxs = []
    if isinstance(arr, np.ndarray):
        result_idxs = heapq.nsmallest(N, range(len(arr)), arr.take)
    elif isinstance(arr, list):
        result_idxs = heapq.nsmallest(N, range(len(arr)), arr.__getitem__)
    else:
        ValueError("Unexpected array type")
    if max_value is not None:
        result_idxs = [x for x in result_idxs if arr[x] < max_value]
    return result_idxs


with open("openlockagents/OpenLockLearner/config.json") as json_data_file:
    config_data = json.load(json_data_file)

FIXED_STRUCTURE_GRAPH_PATH = os.path.expanduser(
    config_data["FIXED_STRUCTURE_GRAPH_PATH"]
)
FIXED_STRUCTURE_ATTRIBUTES_GRAPH_PATH = os.path.expanduser(
    config_data["FIXED_STRUCTURE_ATTRIBUTES_GRAPH_PATH"]
)
FIXED_STRUCTURE_ATTRIBUTES_GRAPH_TWO_STEP_TESTING_PATH = os.path.expanduser(
    config_data["FIXED_STRUCTURE_ATTRIBUTES_GRAPH_TWO_STEP_TESTING_PATH"]
)
FIXED_STRUCTURE_ATTRIBUTES_GRAPH_SIMPLIFIED_TESTING_PATH = os.path.expanduser(
    config_data["FIXED_STRUCTURE_ATTRIBUTES_GRAPH_SIMPLIFIED_TESTING_PATH"]
)
ARBITRARY_STRUCTURE_GRAPH_PATH = os.path.expanduser(
    config_data["ARBITRARY_STRUCTURE_GRAPH_PATH"]
)
# path to human
HUMAN_MAT_DATA_PATH = os.path.expanduser(config_data["HUMAN_MAT_DATA_PATH"])
HUMAN_JSON_DATA_PATH = os.path.expanduser(config_data["HUMAN_JSON_DATA_PATH"])
HUMAN_PICKLE_DATA_PATH = os.path.expanduser(config_data["HUMAN_PICKLE_DATA_PATH"])
PERCEPTUALLY_CAUSAL_RELATION_DATA_PATH = os.path.expanduser(
    config_data["PERCEPTUALLY_CAUSAL_RELATION_DATA_PATH"]
)

# causal chain manager to use
# CAUSAL_GRAPH_MANAGER_BACKEND = "CausalChainManagerPythonLists"
CAUSAL_GRAPH_MANAGER_BACKEND = "CausalChainManagerNumpy"

ACTION_REGEX_STR = "action([0-9]+)"
STATE_REGEX_STR = "state([0-9]+)"

GRAPH_BATCH_SIZE = 1000000

# Attributes
# POSITION_ATTRIBUTE_LABELS = list(POSITION_TO_IDX.keys())
DUMMY_ATTRIBUTES = ["attr1", "attr2"]

# ATTRIBUTE_LABELS = {
#     "color": COLOR_ATTRIBUTE_LABELS,
#     "position": POSITION_ATTRIBUTE_LABELS,
# 'dummy': DUMMY_ATTRIBUTES,
# }

# define causal chain over both state_space x actions

FLUENTS = [CausalRelationType.one_to_zero, CausalRelationType.zero_to_one]
FLUENT_STATES = [0,1]
ACTIONS = ["push", "pull"]

STATES_ROLE = [
    "l0",
    "l1",
    "l2",
    "inactive0",
    "inactive1",
    "inactive2",
    "inactive3",
    "door_lock",
    "door",
]

ACTIONS_ROLE = setup_actions(STATES_ROLE)

DOOR_STATES = ["door_lock"]

# STATES_POSITION = POSITION_ATTRIBUTE_LABELS + DOOR_STATES
# STATES_POSITION = POSITION_ATTRIBUTE_LABELS

# ACTIONS_POSITION = setup_actions(STATES_POSITION)

CAUSAL_CHAIN_EDGES = (
    ("action0", "state0"),
    ("state0", "state1"),
    ("action1", "state1"),
    ("state1", "state2"),
    ("action2", "state2"),
)

THREE_LEVER_TRIALS = ["trial1", "trial2", "trial3", "trial4", "trial5", "trial6"]
FOUR_LEVER_TRIALS = ["trial7", "trial8", "trial9", "trial10", "trial11"]

TRUE_GRAPH_CPT_CHOICES = (1, 1, 0)
TRUE_GRAPH_CPT_CHOICES = tuple([GRAPH_INT_TYPE(x) for x in TRUE_GRAPH_CPT_CHOICES])
PLAUSIBLE_CPT_CHOICES = [
    TRUE_GRAPH_CPT_CHOICES,
    tuple([GRAPH_INT_TYPE(x) for x in (1, 1, 1)]),
    tuple([GRAPH_INT_TYPE(x) for x in (1, 0, 1)]),
]


# --------------------------------------------------------
# BELOW IS DEPRECATED - done by role, rather than position. Need position or role based solution chains based on version of
# human data process. Solutions are store in HUMAN_PICKLE_DATA_PATH/solution_by_trial.pickle, and chains can be generated
# using solution sequences and CausalChainCompact.construct_chain_from_actions_and_cpt_choices(), and TRUE_GRAPH_CPT_CHOICES
# --------------------------------------------------------

# True causal chains in english:
# CC3:
#   Solution 1: push_l0=1: changes l0 from 1->0
#               push_l1=1, l0=0: changes l1 from 1->0
#               push_door=1, l1=0: changes door from 0->1
#   Solution 2: push_l0=1: changes l0 from 1->0
#               push_l2=1, l0=0: changes l2 from 1->0
#               push_door=1, l1=0: changes door from 0->1
# TRUE_CAUSAL_CHAINS_CC3 =[
#     CausalChainCompact(states=('l0', 'l1', 'door'), actions=('push_l0', 'push_l1', 'push_door'), conditional_probability_table_choices=TRUE_GRAPH_CPT_CHOICES), # solution 1
#     CausalChainCompact(states=('l0', 'l2', 'door'), actions=('push_l0', 'push_l2', 'push_door'), conditional_probability_table_choices=TRUE_GRAPH_CPT_CHOICES) # solution 2
# ]

# CE3:
#   Solution 1: push_l1=1: changes l1 from 1->0
#               push_l0=1, l1=0: changes l0 from 1->0
#               push_door=1, l0=0: changes door from 0->1
#   Solution 2: push_l2=1: changes l2 from 1->0
#               push_l0=1, l2=0: changes l0 from 1->0
#               push_door=1, l0=0: changes door from 0->1
# TRUE_CAUSAL_CHAINS_CE3 =[
#     CausalChainCompact(states=('l1', 'l0', 'door'), actions=('push_l1', 'push_l0', 'push_door'), conditional_probability_table_choices=TRUE_GRAPH_CPT_CHOICES), # solution 1
#     CausalChainCompact(states=('l2', 'l0', 'door'), actions=('push_l2', 'push_l0', 'push_door'), conditional_probability_table_choices=TRUE_GRAPH_CPT_CHOICES) # solution 2
# ]
# CAUSALLY_PLAUSIBLE_CHAINS_CE3 = TRUE_CAUSAL_CHAINS_CE3
# CAUSALLY_PLAUSIBLE_CHAINS_CE3.extend([
#     CausalChainCompact(states=('l1', 'l0', 'l1'), actions=('push_l1', 'push_l0', 'pull_l1'), conditional_probability_table_choices=PLAUSIBLE_CPT_CHOICES[0]),
#     CausalChainCompact(states=('l2', 'l0', 'l2'), actions=('push_l2', 'push_l0', 'pull_l2'), conditional_probability_table_choices=PLAUSIBLE_CPT_CHOICES[0]),
#     CausalChainCompact(states=('l1', 'l0', 'l0'), actions=('push_l1', 'push_l0', 'pull_l0'), conditional_probability_table_choices=PLAUSIBLE_CPT_CHOICES[0]),
#     CausalChainCompact(states=('l2', 'l0', 'l0'), actions=('push_l2', 'push_l0', 'pull_l0'), conditional_probability_table_choices=PLAUSIBLE_CPT_CHOICES[0]),
#     CausalChainCompact(states=('l1', 'l2', 'l0'), actions=('push_l1', 'push_l2', 'push_l0'), conditional_probability_table_choices=PLAUSIBLE_CPT_CHOICES[1]),
#     CausalChainCompact(states=('l2', 'l1', 'l0'), actions=('push_l2', 'push_l1', 'push_l0'), conditional_probability_table_choices=PLAUSIBLE_CPT_CHOICES[1]),
#     CausalChainCompact(states=('l1', 'l0', 'l2'), actions=('push_l1', 'push_l0', 'push_l2'), conditional_probability_table_choices=PLAUSIBLE_CPT_CHOICES[1]),
#     CausalChainCompact(states=('l2', 'l0', 'l1'), actions=('push_l2', 'push_l0', 'push_l1'), conditional_probability_table_choices=PLAUSIBLE_CPT_CHOICES[1]),
# ])

# CC4:
#   Solution 1: push_l0=1: changes l0 from 1->0
#               push_l1=1, l0=0: changes l1 from 1->0
#               push_door=1, l1=0: changes door from 0->1
#   Solution 2: push_l0=1: changes l0 from 1->0
#               push_l2=1, l0=0: changes l2 from 1->0
#               push_door=1, l1=0: changes door from 0->1
#   Solution 3: push_l0=1: changes l0 from 1->0
#               push_l3=1, l0=0: changes l3 from 1->0
#               push_door=1, l3=0: changes door from 0->1
# TRUE_CAUSAL_CHAINS_CC4 =[
#     CausalChainCompact(states=('l0', 'l1', 'door'), actions=('push_l0', 'push_l1', 'push_door'), conditional_probability_table_choices=TRUE_GRAPH_CPT_CHOICES), # solution 1
#     CausalChainCompact(states=('l0', 'l2', 'door'), actions=('push_l0', 'push_l2', 'push_door'), conditional_probability_table_choices=TRUE_GRAPH_CPT_CHOICES), # solution 2
#     CausalChainCompact(states=('l0', 'l3', 'door'), actions=('push_l0', 'push_l3', 'push_door'), conditional_probability_table_choices=TRUE_GRAPH_CPT_CHOICES) # solution 3
# ]

# CE4:
#   Solution 1: push_l1=1: changes l1 from 1->0
#               push_l0=1, l1=0: changes l0 from 1->0
#               push_door=1, l0=0: changes door from 0->1
#   Solution 2: push_l2=1: changes l2 from 1->0
#               push_l0=1, l2=0: changes l0 from 1->0
#               push_door=1, l0=0: changes door from 0->1
#   Solution 3: push_l3=1: changes l3 from 1->0
#               push_l0=1, l3=0: changes l0 from 1->0
#               push_door=1, l0=0: changes door from 0->1
# TRUE_CAUSAL_CHAINS_CE4 =[
#     CausalChainCompact(states=('l1', 'l0', 'door'), actions=('push_l1', 'push_l0', 'push_door'), conditional_probability_table_choices=TRUE_GRAPH_CPT_CHOICES), # solution 1
#     CausalChainCompact(states=('l2', 'l0', 'door'), actions=('push_l2', 'push_l0', 'push_door'), conditional_probability_table_choices=TRUE_GRAPH_CPT_CHOICES), # solution 2
#     CausalChainCompact(states=('l3', 'l0', 'door'), actions=('push_l3', 'push_l0', 'push_door'), conditional_probability_table_choices=TRUE_GRAPH_CPT_CHOICES) # solution 3
# ]

# def true_chain_selector(scenario_name):
#     if scenario_name == 'CC3':
#         return TRUE_CAUSAL_CHAINS_CC3
#     if scenario_name == 'CE3':
#         return TRUE_CAUSAL_CHAINS_CE3
#     if scenario_name == 'CC4':
#         return TRUE_CAUSAL_CHAINS_CC4
#     if scenario_name == 'CE4':
#         return TRUE_CAUSAL_CHAINS_CE4


def print_message(trial_count, attempt_count, message, print_message=True):
    if print_message:
        print("T{}.A{}: ".format(trial_count, attempt_count) + message)


def pretty_write(json_str, filename):
    """
    Write json_str to filename with sort_keys=True, indents=4.

    :param filename: Name of file to be output.
    :param json_str: JSON str to write (e.g. from jsonpickle.encode()).
    :return: Nothing.
    """
    with open(filename, "w") as outfile:
        # reencode to pretty print
        json_obj = json.loads(json_str)
        json_str = json.dumps(json_obj, indent=4, sort_keys=True)
        outfile.write(json_str)

        # results_dir = trial_dir + '/results'
        # os.makedirs(results_dir)
        # for j in range(len(trial.attempt_seq)):
        #     attempt = trial.attempt_seq[j]
        #     results = attempt.results
        #     np.savetxt(results_dir + '/results_attempt' + str(j) + '.csv', results, delimiter=',', fmt='%s')


def write_pickle(obj, filename):
    with open(filename, "wb") as f:
        pkl.dump(obj, f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        obj = pkl.load(f)
        return obj


def merge_perceptually_causal_relations_from_dict_of_trials(
    perceptually_causal_relations
):
    merged_perceptually_causal_relations = []
    for key in perceptually_causal_relations.keys():
        merged_perceptually_causal_relations.extend(
            list(perceptually_causal_relations[key].values())
        )

    return merged_perceptually_causal_relations


def merge_solutions_from_dict_of_trials(true_chains):
    merged_true_chains = []
    for key in true_chains.keys():
        merged_true_chains.extend(true_chains[key])

    merged_true_chains = list(set(merged_true_chains))
    return merged_true_chains


def homogenous_list(seq):
    iseq = iter(seq)
    first_type = type(next(iseq))
    return first_type if all((type(x) is first_type) for x in iseq) else False


def create_birdirectional_dict(values, birdir_type):
    bidir_dict = {values[i]: birdir_type(i) for i in range(len(values))}
    bidir_dict.update(dict([reversed(i) for i in bidir_dict.items()]))
    return bidir_dict


def generate_slicing_indices(l, batch_size=None):
    # if no fixed batch size is specified, compute batch size based on cpu_count and list length
    if batch_size is None:
        # if the list is less than l, we can use len(l) cpus
        if len(l) < multiprocessing.cpu_count():
            batch_size = 1
        else:
            batch_size = max(2, len(l) // (multiprocessing.cpu_count()))
    # must start at 0 index
    slicing_indices = [0] + list(range(batch_size, len(l), batch_size))

    # assign last portion to end
    slicing_indices.append(len(l))
    return slicing_indices


def check_for_duplicates(l):
    seen = set()
    for i in range(len(l)):
        if l[i] in seen:
            return True, i
        seen.add(l[i])
    return False, None


if __name__ == "__main__":
    print("inside of causal_classes.py")
