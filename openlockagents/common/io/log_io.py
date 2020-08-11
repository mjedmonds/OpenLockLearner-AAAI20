import glob
import os
import json
import pickle as pkl
import re

import h5py
import jsonpickle
import pickle


from openlock.settings_trial import (
    NUM_LEVERS_IN_HUMAN_DATA,
)

from openlockagents.Human.common import (
    load_human_config_json
)


# this is a terrible way to do this! Just load the JSON, but this example is how you would load the matlab file, if
# we ever need to do it
def load_human_data_from_mat(data_dir):
    for mat_filename in glob.glob(data_dir + "/*.mat"):
        with h5py.File(mat_filename, "r") as mat_file:
            print(mat_file.keys())
            print(mat_file["subject_summary"]["age"])


def load_subjects_from_dir(data_dir, ext="json", participant_id_corrections=None,  convert_to_position=False):
    """

    :param data_dir: directory to load from
    :param ext: extension of data, either json or pkl
    :param convert_to_position: converts lever roles to positions
    :return:
    """
    print_update_rate = 10
    subjects = []
    trial_data_by_trail_name = dict()
    subject_dirs = os.listdir(data_dir)
    subject_dirs = [x for x in subject_dirs if os.path.isdir(os.path.join(data_dir, x))]
    for i in range(len(subject_dirs)):
        subject_dir = os.path.join(data_dir, subject_dirs[i])

        subject_summary = load_subject_data(subject_dir, ext, convert_to_position)

        if participant_id_corrections is not None and subject_summary.subject_id in participant_id_corrections.keys():
            subject_summary.participant_id = participant_id_corrections[subject_summary.subject_id]

        subjects.append(subject_summary)

        if i % print_update_rate == 0:
            print("{}/{} subjects added".format(i, len(subject_dirs)))

    return subjects


def load_subject_data(subject_dir, ext, convert_to_position=False, use_json_pickle_for_trial=True):
    """

    :param subject_dir:
    :param ext:
    :param convert_to_position:
    :param use_json_pickle_for_trial:
    :return:
    """
    subject_trial_dirs = os.listdir(subject_dir)
    subject_trial_dirs = [
        x for x in subject_trial_dirs if os.path.isdir(os.path.join(subject_dir, x))
    ]
    subject_files = os.listdir(subject_dir)
    subject_files = [
        x for x in subject_files if os.path.isfile(os.path.join(subject_dir, x))
    ]

    summary_regex = re.compile("[0-9]+_summary." + ext)
    agent_regex = re.compile("[0-9]+_agent." + ext)

    subject_summary_filename = [x for x in subject_files if summary_regex.match(x)]
    assert (
        len(subject_summary_filename) == 1
    ), "Expected exactly a single subject summary, subject {}".format(subject_dir)
    subject_summary_filename = subject_summary_filename[0]
    agent_filename = [x for x in subject_files if agent_regex.match(x)]
    assert len(agent_filename) < 2, "Expected at most one agent"
    agent_filename = agent_filename[0] if len(agent_filename) > 0 else None

    # read the subject summary
    subject_summary = load_subject_summary(subject_dir, subject_summary_filename, ext)

    # read the agent summary if present
    if agent_filename:
        agent = load_agent(subject_dir, agent_filename, ext)
        subject_summary.agent = agent

    # only accept dirs with trial in the name
    trial_regex = re.compile("trial")
    subject_trial_dirs = list(filter(trial_regex.match, subject_trial_dirs))

    # order trials
    regex = re.compile('[^0-9]')
    subject_trial_dirs.sort(key=lambda x: int(regex.sub("", x)))

    # load trials
    for subject_trial_dir in subject_trial_dirs:
        subject_trial_filename = subject_trial_dir + "/" + subject_trial_dir + "_summary." + ext
        trial = load_trial_from_file(subject_dir, subject_trial_filename, ext, convert_to_position, use_json_pickle_for_trial)
        subject_summary.trial_seq.append(trial)

    return subject_summary


def load_file_json_pickle(dir, filename):
    with open(os.path.join(dir, filename), "r") as file:
        json_str = file.read()
        obj = jsonpickle.decode(json_str)
        return obj


def load_file_json(dir, filename):
    with open(os.path.join(dir, filename), "r") as file:
        json_str = file.read()
        obj = json.loads(json_str)
        return obj


def load_file_pickle(dir, filename):
    with open(os.path.join(dir, filename), "rb") as file:
        obj = pkl.load(file)
        return obj


def load_subject_summary(subject_dir, subject_summary_filename, ext):
    if ext == "json":
        subject_summary = load_file_json_pickle(subject_dir, subject_summary_filename)
        subject_summary.trial_seq = []
        return subject_summary
    elif ext == "pkl":
        subject_summary = load_file_pickle(subject_dir, subject_summary_filename)
        return subject_summary
    else:
        raise ValueError("Unknown subject_summary file extension. Must be json or pkl.")


def load_agent(subject_dir, agent_filename, ext):
    if ext == "json":
        agent = load_file_json_pickle(subject_dir, agent_filename)
        return agent
    elif ext == "pkl":
        agent = load_file_pickle(subject_dir, agent_filename)
        return agent
    else:
        raise ValueError("Unknown subject_summary file extension. Must be json or pkl.")


def load_trial_from_file(subject_dir, subject_trial_filename, ext, convert_to_position=False, use_json_pickle_for_trial=True):
    """
    loads a single trial from a file
    :param subject_dir: root directory of subject data
    :param subject_trial_filename: filename of trial
    :param ext: extension to load, either json or pkl
    :param convert_to_position: convert the lever roles to positions
    :param use_json_pickle_for_trial: load trial using jsonpickle vs. raw json
    :return: trial object
    """
    if ext == "json":
        trial = load_trial_from_file_json(subject_dir, subject_trial_filename, use_json_pickle_for_trial)
        # convert lever roles to positions
        if convert_to_position and use_json_pickle_for_trial:
            trial = convert_trial_lever_roles_to_position(trial)
        return trial
    elif ext == "pkl":
        trial = load_file_pickle(subject_dir, subject_trial_filename)
        # convert lever roles to positions
        if convert_to_position:
            trial = convert_trial_lever_roles_to_position(trial)
        return trial
    else:
        raise ValueError("Unknown trial file extension. Must be json or pkl.")


def load_trial_from_file_json(subject_dir, subject_trial_filename, use_json_pickle_for_trial=True):
    if use_json_pickle_for_trial:
        trial = load_file_json_pickle(subject_dir, subject_trial_filename)
    else:
        trial = load_file_json(subject_dir, subject_trial_filename)
    return trial


def load_human_data_json():
    human_config_data = load_human_config_json()
    return load_subjects_from_dir(human_config_data["HUMAN_JSON_DATA_PATH"])


def load_human_data_pickle():
    human_config_data = load_human_config_json()
    with open(human_config_data["HUMAN_PICKLE_DATA_PATH"], "rb") as infile:
        human_subjects = pickle.load(infile)
        return human_subjects


def extract_solution_chains(subject_summary_list):
    solution_chains = dict()

    for subject_summary in subject_summary_list:
        for trial in subject_summary.trial_seq:
            if trial.name not in solution_chains.keys():
                solution_chains[trial.name] = trial.solutions

    return solution_chains


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


def main():
    convert_to_position = True
    config_data = load_human_config_json()
    # data = load_human_data_from_mat(config_data["HUMAN_MAT_DATA_PATH")
    data = load_subjects_from_dir(
        config_data["HUMAN_JSON_DATA_PATH"], convert_to_position=convert_to_position
    )

    solutions = extract_solution_chains(data)

    # save_human_data_pickle(data, solutions, HUMAN_PICKLE_DATA_PATH)

    # solutions2 = load_solutions_by_trial(HUMAN_PICKLE_DATA_PATH)

    return data


if __name__ == "__main__":
    main()

