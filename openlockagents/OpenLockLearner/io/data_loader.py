import glob
import os
import json
import re

import h5py
import jsonpickle
import pickle

from openlock.settings_trial import (
    NUM_LEVERS_IN_HUMAN_DATA,
)

from openlockagents.OpenLockLearner.util.common import (
    HUMAN_JSON_DATA_PATH,
    HUMAN_PICKLE_DATA_PATH,
)


# this is a terrible way to do this! Just load the JSON, but this example is how you would load the matlab file, if
# we ever need to do it
def load_human_data_from_mat(data_dir):
    for mat_filename in glob.glob(data_dir + "/*.mat"):
        with h5py.File(mat_filename, "r") as mat_file:
            print(mat_file.keys())
            print(mat_file["subject_summary"]["age"])


def load_human_data_from_json(data_dir, convert_to_position=False):
    print_update_rate = 10
    subjects = []
    trial_data_by_trail_name = dict()
    subject_dirs = os.listdir(data_dir)
    subject_dirs = [x for x in subject_dirs if os.path.isdir(os.path.join(data_dir, x))]
    solution_chains = dict()
    for i in range(len(subject_dirs)):
        subject_dir = os.path.join(data_dir, subject_dirs[i])

        subject_summary, solution_chains = load_subject_data_from_json(subject_dir, convert_to_position)

        for trial in subject_summary.trial_seq:
            if trial.name not in solution_chains.keys():
                solution_chains[trial.name] = trial.solutions

        subjects.append(subject_summary)

        if i % print_update_rate == 0:
            print("{}/{} subjects added".format(i, len(subject_dirs)))

        # with
        # json.load()
        # subject_summary = jsonpickle.loads
    return subjects, solution_chains


def load_subject_data_from_json(subject_dir, convert_to_position=False, use_json_pickle_for_trial=True):
    subject_trial_dirs = os.listdir(subject_dir)
    subject_trial_dirs = [
        x for x in subject_trial_dirs if os.path.isdir(os.path.join(subject_dir, x))
    ]
    subject_files = os.listdir(subject_dir)
    subject_files = [
        x for x in subject_files if os.path.isfile(os.path.join(subject_dir, x))
    ]

    summary_regex = re.compile("[0-9]+_summary.json")
    agent_regex = re.compile("[0-9]+_agent.json")

    subject_summary_filename = [x for x in subject_files if summary_regex.match(x)]
    assert (
        len(subject_summary_filename) == 1
    ), "Expected single subject summary, subject {}".format(subject_dir)
    subject_summary_filename = subject_summary_filename[0]
    agent_filename = [x for x in subject_files if agent_regex.match(x)]
    assert len(agent_filename) < 2, "Expected at most one agent"
    agent_filename = agent_filename[0] if len(agent_filename) > 0 else None

    # read the subject summary and agent summary
    with open(
        os.path.join(subject_dir, subject_summary_filename), "r"
    ) as subject_summary_file:
        json_str = subject_summary_file.read()
        subject_summary = jsonpickle.decode(json_str)
        subject_summary.trial_seq = []
    if agent_filename:
        with open(os.path.join(subject_dir, agent_filename), "r") as agent_file:
            agent = json.load(agent_file)
            subject_summary.agent = agent

    # only accept dirs with trial in the name
    trial_regex = re.compile("trial")
    subject_trial_dirs = list(filter(trial_regex.match, subject_trial_dirs))

    # order trials
    regex = re.compile('[^0-9]')
    subject_trial_dirs.sort(key=lambda x : int(regex.sub("", x)))

    # load trials
    for subject_trial_dir in subject_trial_dirs:
        with open(
            os.path.join(
                subject_dir,
                subject_trial_dir + "/" + subject_trial_dir + "_summary.json",
            ),
            "r",
        ) as trial_file:
            # print("trial dir: {}".format(subject_trial_dir))
            json_str = trial_file.read()
            if use_json_pickle_for_trial:
                trial = jsonpickle.decode(json_str)
                # convert lever roles to positions
                if convert_to_position:
                    trial = convert_trial_lever_roles_to_position(trial)
            else:
                trial = json.loads(json_str)
            subject_summary.trial_seq.append(trial)

    return subject_summary


def load_human_data_json():
    return load_human_data_from_json(HUMAN_JSON_DATA_PATH)


def load_human_data_pickle():
    with open(HUMAN_PICKLE_DATA_PATH, "rb") as infile:
        human_subjects = pickle.load(infile)
        return human_subjects


def convert_trial_lever_roles_to_position(trial):
    prev_role_to_position_mapping = None
    role_to_position_mapping = None
    for i in range(len(trial.attempt_seq)):
        attempt = trial.attempt_seq[i]
        role_to_position_mapping = construct_role_to_position_mapping(attempt)

        attempt.results[0] = rename_col_labels(
            attempt.results[0], role_to_position_mapping
        )

        attempt.action_seq = rename_action_sequence(
            attempt.action_seq, role_to_position_mapping
        )

        # sanity check: the role mapping should be consistent within a trial
        if prev_role_to_position_mapping is not None:
            assert role_to_position_mapping == prev_role_to_position_mapping
        prev_role_to_position_mapping = role_to_position_mapping

        trial.attempt_seq[i] = attempt

    assert role_to_position_mapping is not None, "No attempts in this trial"
    trial.solutions = rename_sequence_of_action_sequences(
        trial.solutions, role_to_position_mapping
    )
    trial.complete_solutions = rename_sequence_of_action_sequences(
        trial.completed_solutions, role_to_position_mapping
    )
    return trial


def construct_role_to_position_mapping(attempt):
    col_labels = attempt.results[0]
    agent_idx = col_labels.index("agent")  # used to split states/actions
    role_to_pos = dict()
    pos_idx = 0
    for i in range(agent_idx + 1, agent_idx + NUM_LEVERS_IN_HUMAN_DATA + 1):
        action = col_labels[i]
        lever_role = action.split("_", 1)[1]
        role_to_pos[lever_role] = IDX_TO_POSITION[pos_idx]
        pos_idx += 1
    return role_to_pos


def rename_col_labels(col_labels, role_to_position_mapping):
    agent_idx = col_labels.index("agent")
    # replace states roles to positions
    for i in range(agent_idx):
        if col_labels[i] in role_to_position_mapping.keys():
            col_labels[i] = role_to_position_mapping[col_labels[i]]
    # replace action roles to positions
    for i in range(agent_idx + 1, agent_idx + (NUM_LEVERS_IN_HUMAN_DATA * 2) + 1):
        col_labels[i] = rename_action(col_labels[i], role_to_position_mapping)
    return col_labels


def rename_sequence_of_action_sequences(sequence_action_seq, role_to_position_mapping):
    for i in range(len(sequence_action_seq)):
        sequence_action_seq[i] = rename_action_sequence(
            sequence_action_seq[i], role_to_position_mapping
        )
    return sequence_action_seq


def rename_action_sequence(action_seq, role_to_position_mapping):
    for i in range(len(action_seq)):
        action_seq[i].name = rename_action(action_seq[i].name, role_to_position_mapping)
    return action_seq


def rename_action(action, role_to_position_mapping):
    action_split = action.split("_", 1)
    if action_split[1] in role_to_position_mapping.keys():
        action_split[1] = role_to_position_mapping[action_split[1]]
    action = "_".join(action_split)
    return action


# groups human subjects according to the trial name and scenario,
# used to scan for relations within a specific trial (since lever positions are different for each trial)
def group_human_subjects_by_trial(human_subjects):
    trial_dict = dict()
    for human_subject in human_subjects:
        for trial in human_subject.trial_seq:
            trial_key = trial.name
            if trial_key not in trial_dict.keys():
                trial_dict[trial_key] = [trial]
            else:
                trial_dict[trial_key].append(trial)
    return trial_dict


def save_human_data_pickle(human_subjects, solutions, data_file):
    data_dir = os.path.dirname(data_file)
    solution_file = data_dir + "/solutions_by_trial.pickle"
    os.makedirs(data_dir, exist_ok=True)
    with open(data_file, "wb") as outfile:
        pickle.dump(human_subjects, outfile)
    with open(solution_file, "wb") as outfile:
        pickle.dump(solutions, outfile)


def load_solutions_by_trial(data_file):
    solutions_file = os.path.dirname(data_file) + "/solutions_by_trial.pickle"
    with open(solutions_file, "rb") as infile:
        solutions = pickle.load(infile)
        return solutions


def main():
    convert_to_position = True
    # data = load_human_data_from_mat(HUMAN_MAT_DATA_PATH)
    data, solutions = load_human_data_from_json(
        HUMAN_JSON_DATA_PATH, convert_to_position=convert_to_position
    )

    save_human_data_pickle(data, solutions, HUMAN_PICKLE_DATA_PATH)

    solutions2 = load_solutions_by_trial(HUMAN_PICKLE_DATA_PATH)

    return data


if __name__ == "__main__":
    main()
