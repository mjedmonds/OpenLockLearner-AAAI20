import pickle

import numpy as np

from openlock.settings_trial import generate_attributes_by_trial

from openlockagents.OpenLockLearner.causal_classes.CausalRelation import (
    CausalRelationType,
    CausalObservation,
)
from openlockagents.OpenLockLearner.util.common import (
    load_openlock_learner_config_json
)
from openlockagents.common.io.log_io import (
    load_human_data_pickle,
)


def scan_for_causal_relations_by_subject(human_subjects):
    causal_relations = dict()
    for subject_log in human_subjects:
        print("Processing subject {}".format(subject_log.subject_id))
        trial_count = 0
        for trial in subject_log.trial_seq:
            trial_name = trial.name
            print("Processing {} with trial number {}.".format(trial_name, trial_count))
            trial_count += 1
            attempt_count = 0
            for attempt in trial.attempt_seq:
                print("Processing attempt {}".format(attempt_count))
                attempt_count += 1
                causal_relations = scan_for_causal_relation(causal_relations, attempt)

    return causal_relations


def scan_for_causal_relations_by_trial(human_subjects, attributes_by_trial):
    # this will be a dict of dicts
    causal_relations = dict()
    for subject_log in human_subjects:
        # print('Processing subject {}'.format(subject_log.subject_id))
        trial_count = 0
        for trial in subject_log.trial_seq:
            trial_name = trial.name
            # print('Processing {} with trial number {}.'.format(trial_name, trial_count))

            if trial_name not in causal_relations.keys():
                causal_relations[trial_name] = dict()

            if attributes_by_trial is not None:
                trial_attributes = attributes_by_trial[trial_name]
            else:
                trial_attributes = None

            trial_count += 1
            attempt_count = 0
            for attempt in trial.attempt_seq:
                # print('Processing attempt {}'.format(attempt_count))
                attempt_count += 1
                causal_relations[trial_name] = scan_for_causal_relation(
                    causal_relations[trial_name], attempt, trial_attributes
                )

    return causal_relations


# causal_classes.py code for if we are scanning by trial or scanning per subject
def scan_for_causal_relation(causal_relations, attempt, trial_attributes=None):
    col_labels = np.array(attempt.results[0])
    results = np.array(attempt.results[1 : len(attempt.results)])
    # make frames monotonically increasing
    results[:, 0] = np.arange(1, results.shape[0] + 1)
    # add previous fluent value (add_inertial)
    perm_dat = append_previous_fluent_value(results)
    # find agent, which demarks the border between fluents and actions in the file.
    # setup idxs for fluents and actions
    agent_idx = np.where(col_labels == "agent")[0][0]
    action_idxs = np.arange(agent_idx + 1, col_labels.shape[0])
    fluent_idxs = np.arange(1, agent_idx + 1)
    for cur_fluent_idx in fluent_idxs:
        cur_fluent = col_labels[cur_fluent_idx]

        if cur_fluent != "agent":
            cur_attributes = trial_attributes[cur_fluent]
        else:
            cur_attributes = ("agent", "WHITE")

        # print('Scanning for causal relations for fluent {}'.format(cur_fluent))
        cols_to_remove = fluent_idxs[np.where(fluent_idxs != cur_fluent_idx)[0]]
        dat = np.delete(perm_dat, cols_to_remove, axis=1)
        dat_cols = np.delete(col_labels, cols_to_remove)
        # find all rows where there was a fluent change along the fluent column
        fluent_changes = np.diff(dat[:, 1])
        fluent_change_rows = np.where(fluent_changes != 0)[0]
        fluent_change_actions = []
        causal_action_labels = []
        causal_relation_types = []
        for row_idx in fluent_change_rows:
            # column indices correspond to frame, fluent value, followed by all actions, then inertial
            # so we start looking at the third column for actions, then add 2 to make it consistent indexing
            # with dat
            # todo: this only supports one action (we take the first one here)
            fluent_change_action = np.where(dat[row_idx, 2:])[0][0] + 2
            fluent_change_action_label = dat_cols[fluent_change_action]

            fluent_change_actions.append(fluent_change_action)
            causal_action_labels.append(fluent_change_action_label)

            causal_relation_type = fluent_changes[row_idx]
            causal_relation_types.append(causal_relation_type)

        # convert fluent changes to be of CausalRelationType
        # -1 = 1->0, CausalRelationType 2
        # 1 = 0->1, CausalRelationType 3
        causal_relation_types = np.array(causal_relation_types)
        causal_relation_types[np.where(causal_relation_types == -1)[0]] = 2
        causal_relation_types[np.where(causal_relation_types == 1)[0]] = 3

        for causal_idx in range(len(causal_action_labels)):
            causal_action_label = causal_action_labels[causal_idx]
            causal_relation_type = CausalRelationType(causal_relation_types[causal_idx])
            # skip relations with no fluent change
            if (
                causal_relation_type == CausalRelationType.one_to_one
                or causal_relation_type == CausalRelationType.zero_to_zero
            ):
                continue
            key = (
                cur_fluent + "_" + str(causal_relation_type) + "_" + causal_action_label
            )
            info_gain = -1  # dummy info gain to count occurrence
            if key in causal_relations.keys():
                causal_relations[key].add_info_gain(info_gain)
            else:
                new_node = CausalObservation(
                    cur_fluent,
                    causal_relation_type,
                    causal_action_label,
                    info_gain,
                    cur_attributes,
                )
                causal_relations[key] = new_node
            # print('Causal action found for {}: {}'.format(cur_fluent, causal_action_label))
    return causal_relations


# version of amy's add_inertial2 function. This one does not consider a frame/action lag
def append_previous_fluent_value(dat):
    new_dat = np.zeros((dat.shape[0], dat.shape[1] + 1), dtype=np.int64)
    new_dat[:, 0 : dat.shape[1]] = dat
    for row_idx in range(1, dat.shape[0]):
        # copy previous fluent value
        new_dat[row_idx, -1] = dat[row_idx - 1, 1]
    return new_dat


def remove_state_from_perceptually_causal_relations(causal_relations, state_name):
    for trial in causal_relations.keys():
        causal_relations[trial] = {
            k: v for k, v in causal_relations[trial].items() if v.state != state_name
        }
    return causal_relations


def write_perceptually_causal_relations(causal_relations):
    openlock_learner_config_data = load_openlock_learner_config_json()
    with open(openlock_learner_config_data["PERCEPTUALLY_CAUSAL_RELATION_DATA_PATH"], "wb") as outfile:
        pickle.dump(causal_relations, outfile)


def load_perceptually_causal_relations():
    openlock_learner_config_data = load_openlock_learner_config_json()
    with open(openlock_learner_config_data["PERCEPTUALLY_CAUSAL_RELATION_DATA_PATH"], "rb") as infile:
        return pickle.load(infile)


def generate_perceptually_causal_relations():
    # load from prepickled data
    human_subjects = load_human_data_pickle()
    # trial_data = group_human_subjects_by_trial(human_subjects)

    attributes_by_trial = generate_attributes_by_trial()

    # causal_relations = scan_for_causal_relations(human_subjects)
    # todo: use Amy's method
    causal_relations = scan_for_causal_relations_by_trial(
        human_subjects, attributes_by_trial
    )

    write_perceptually_causal_relations(causal_relations)

    return causal_relations


if __name__ == "__main__":
    generate_perceptually_causal_relations()
