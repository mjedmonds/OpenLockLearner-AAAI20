import itertools
import math
import time
import multiprocessing
import re
import glob
import os
from joblib import Parallel, delayed
from itertools import chain

from openlockagents.OpenLockLearner.causal_classes.CausalChainStructureSpace import (
    CausalChainStructureSpace,
)

from openlockagents.OpenLockLearner.causal_classes.CausalRelationSpace import (
    CausalRelationSpace
)
from openlockagents.OpenLockLearner.causal_classes.CausalChain import CausalChainCompact
from openlockagents.OpenLockLearner.util.common import (
    FIXED_STRUCTURE_GRAPH_PATH,
    ARBITRARY_STRUCTURE_GRAPH_PATH,
    ACTION_REGEX_STR,
    STATE_REGEX_STR,
    TRUE_GRAPH_CPT_CHOICES,
    GRAPH_BATCH_SIZE,
    CAUSAL_GRAPH_MANAGER_BACKEND,
    PARALLEL_MAX_NBYTES,
)



def generate_chain_structure_space(
    env,
    actions,
    attributes,
    fluents,
    fluent_states,
    perceptually_causal_relations,
    structure,
):
    t = time.time()
    causal_relation_space = CausalRelationSpace(
        actions, attributes, fluents, fluent_states, perceptually_causal_relations
    )

    causal_chain_structure_space = CausalChainStructureSpace(
        causal_relation_space,
        chain_length=3,
        attributes=attributes,
        structure=structure,
        attribute_order=env.attribute_order,
        lever_index_mode="position",
    )
    print("Generated {} chains in {}s".format(len(causal_chain_structure_space), time.time() - t))
    return causal_chain_structure_space


def delete_pickle_files_from_dir(dir):
    filelist = glob.glob(os.path.join(dir, "*.pickle"))
    for f in filelist:
        os.remove(f)


def generate_true_chains(solution_seqs_by_trial):
    true_chains = dict()
    for trial, solutions in solution_seqs_by_trial.items():
        true_chains_trial = []
        for actions in solutions:
            states = CausalChainStructureSpace.extract_states_from_actions(actions)
            actions = tuple([str(action) for action in actions])
            attributes = tuple([(state, "GREY") for state in states])
            true_chains_trial.append(
                CausalChainCompact(states, actions, TRUE_GRAPH_CPT_CHOICES, attributes)
            )
        true_chains[trial] = true_chains_trial
    return true_chains


def insert_into_dict(key, dict_):
    if key not in dict_.keys():
        dict_[key] = 1
    else:
        dict_[key] += 1
    return dict_


def partition(alist, indices):
    pairs = zip(chain([0], indices), chain(indices, [None]))
    return (alist[i:j] for i, j in pairs)


# extracts a batch from the generator. If the generator is finished (StopIteration thrown), return True
def extract_batch_from_generator(generator, chunk_size):
    iterator = iter(generator)
    batch = []
    while len(batch) < chunk_size:
        try:
            ele = iterator.__next__()
            batch.append(ele)
        except StopIteration:
            return batch, True
        # yield itertools.chain([first], itertools.islice(iterator, chunk_size-1))
    return batch, False


def generate_instantiations_of_base_schema(states, actions):
    pass


def main():
    causal_chain_space = None

    # possible_environments = generate_possible_environments(
    #     STATES_POSITION, ACTIONS_POSITION, ATTRIBUTE_LABELS
    # )

    # xxx: this main is out of date, see generate_prune_subsample.py
    # fixed structure: generation
    # generate based on role
    # causal_chain_space = generate(STATES_LABEL, ACTIONS_LABEL, lever_index_mode='role', mode='fixed', structure=CAUSAL_CHAIN_EDGES)
    # generate based on position
    # causal_chain_space = generate(STATES_POSITION, ACTIONS_POSITION, lever_index_mode='position', mode='fixed', structure=CAUSAL_CHAIN_EDGES)

    # generate based on position and attribute
    # todo: more cleanly define attributes x
    # causal_chain_space = generate_causal_chains_fixed_structure_attributes(ATTRIBUTE_PRODUCT, ACTIONS_POSITION, lever_index_mode='position', structure=CAUSAL_CHAIN_EDGES)

    # arbitrary structure: generating
    # causal_chain_space = generate(STATES, ACTIONS, mode='arbitrary', structure=None)

    return causal_chain_space


if __name__ == "__main__":
    main()
