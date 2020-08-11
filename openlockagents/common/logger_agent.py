import time
import jsonpickle
import os
import json
import copy
import sys
import texttable
import os
import subprocess
import signal

import numpy as np

from tensorboardX import SummaryWriter

from openlock.logger_env import ActionLog, AttemptLog, TrialLog
from openlockagents.common.io.log_io import pretty_write, write_pickle


class SubjectLogger(object):
    """
    Represents a subject for the purpose of logger.
    """

    subject_id = None
    age = None
    gender = None
    handedness = None
    eyewear = None
    major = None

    trial_seq = []
    trial_amount = 0
    cur_trial = None
    start_time = None
    end_time = None

    cur_scenario_name = None

    strategy = None
    random_seed = None

    def __init__(
        self,
        subject_id,
        participant_id,
        age,
        gender,
        handedness,
        eyewear,
        major,
        start_time,
        human=True,
        random_seed=None,
    ):
        """
        Create the subject.

        :param subject_id: Subject ID.
        :param participant_id: Participant ID.
        :param age: Age of subject.
        :param gender: Gender of subject.
        :param handedness: Handedness of subject.
        :param eyewear: Eyewear of subject (yes or no).
        :param major: Major of subject.
        :param start_time: Time that the subject starts.
        :param human: Whether subject is human, default True.
        :param random_seed: Default None.
        """
        self.subject_id = subject_id
        self.participant_id = participant_id
        self.start_time = start_time
        self.age = age
        self.gender = gender
        self.handedness = handedness
        self.eyewear = eyewear
        self.major = major
        self.human = human
        self.random_seed = random_seed

    def add_trial(self, trial_name, scenario_name, solutions):
        """
        Set the current trial to a new TrialLog object.

        :param trial_name: Name of the trial.
        :param scenario_name: Name of the scenario.
        :param solutions: Solutions of trial.
        :param random_seed:
        :return: Nothing.
        """
        self.cur_trial = TrialLog(trial_name, scenario_name, solutions, time.time())

    def finish_trial(self):
        """
        Finish the current trial.

        :return: True if trial was successful, False otherwise.
        """
        success = self.cur_trial.finish(time.time())
        self.trial_seq = []
        self.trial_seq.append(self.cur_trial)
        self.trial_amount += 1
        self.cur_trial = None
        return success

    def finish(self, end_time):
        """
        Set end time of the subject.

        :param end_time: End time of the subject.
        :return: Nothing.
        """
        self.end_time = end_time


# SubjectLogger used to be called SubjectLog, so we'll allow the pickler to
# properly instantiate the class
SubjectLog = SubjectLogger


class TerminalWriter:
    """
    Logs stdout output to agent's log
    """

    def __init__(self, logfile):
        self.stdout = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.stdout.write(message)
        self.log.write(message)

    def flush(self):
        pass


class SubjectWriter:
    """
    Writes the log files for a subject.
    """

    subject_path = None

    def __init__(self, data_path):
        """
        Create a log file for the subject inside the data_path directory.

        :param data_path: Path to keep the log files in.
        """
        self.subject_id = str(hash(time.time()))
        self.subject_path = data_path + "/" + self.subject_id
        while True:
            # make sure directory does not exist
            if not os.path.exists(self.subject_path):
                os.makedirs(self.subject_path)
                break
            else:
                self.subject_id = str(hash(time.time()))
                self.subject_path = data_path + "/" + self.subject_id
                continue
        # setup writing stdout to file and to stdout
        self.terminal_writer = TerminalWriter(
            self.subject_path + "/" + self.subject_id + "_stdout.log"
        )
        tensorboard_path = "/tmp/tensorboard"
        # child runs tensorboard
        # XXX: this has problem; be weary
        # self.tensorboard_pid = os.fork()
        # if self.tensorboard_pid == 0:
        #     subprocess.run(['tensorboard', '--logdir', tensorboard_path])
        #     sys.exit(0)
        self.tensorboard_writer = TensorboardWriter(tensorboard_path)

    def terminate(self):
        if hasattr(self, "tensorboard_writer") and self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        # kill tensorboard
        # if self.tensorboard_pid > 0:
        #     os.kill(self.tensorboard_pid, signal.SIGTERM)

    def write_trial(self, logger, test_trial=False):
        """
        Make trial directory, trial summary, and write trial summary.

        :param logger: SubjectLogger.
        :param test_trial: True if test trial, False otherwise. Default False.
        :return: Nothing.
        """
        # i = len(logger.trial_seq)-1
        # trial = logger.trial_seq[i]
        i = logger.trial_amount - 1
        trial = logger.trial_seq[-1]

        trial_dir = self.subject_path + "/trial" + str(i)
        # if test_trial:
        #     trial_dir = trial_dir + '_test'
        os.makedirs(trial_dir)
        trial_summary_filename_base = trial_dir + "/trial" + str(i) + "_summary"

        trial_str = jsonpickle.encode(trial)

        # write to pickle and json
        write_pickle(trial, trial_summary_filename_base + ".pkl")
        pretty_write(trial_str, trial_summary_filename_base + ".json")

        # remove the trial after writing to disk
        logger.trial_seq = []

    def write(self, logger, agent):
        """
        Write subject summary log.

        :param logger: SubjectLogger object.
        :param agent: Agent object.
        :return: Nothing.
        """
        # json_results = self.JSONify_subject(logger)

        print("Writing subject data to: {}".format(self.subject_path))
        subject_summary_filename_base = (
            self.subject_path + "/" + logger.subject_id + "_summary"
        )

        subject_summary_str = jsonpickle.encode(logger)

        write_pickle(logger, subject_summary_filename_base + ".pkl")
        pretty_write(subject_summary_str, subject_summary_filename_base + ".json")

        # write out the RL agent
        if agent is not None:
            agent_cpy = copy.copy(agent)
            things_to_delete = ["env", "target_model", "model", "memory"]
            for thing_to_delete in things_to_delete:
                if hasattr(agent_cpy, thing_to_delete):
                    delattr(agent_cpy, thing_to_delete)
            things_to_delete = ["tensorboard_writer", "terminal_writer"]
            for thing_to_delete in things_to_delete:
                if hasattr(agent_cpy.writer, thing_to_delete):
                    delattr(agent_cpy.writer, thing_to_delete)
            if hasattr(agent_cpy, "profiler"):
                agent_cpy.profiler.disable()
                agent_cpy.profiler.dump_stats(
                    self.subject_path + "/" + logger.subject_id + "_profile.pstat"
                )
                delattr(agent_cpy, "profiler")
            agent_file_name_base = (
                self.subject_path + "/" + logger.subject_id + "_agent"
            )

            agent_str = jsonpickle.encode(agent_cpy)

            write_pickle(agent_cpy, agent_file_name_base + ".pkl")
            pretty_write(agent_str, agent_file_name_base + ".json")

    #
    # def JSONify_subject(self, subject):
    #     trial_jsons = []
    #     for trial in subject.trial_seq:
    #         trial_jsons.append( self.JSONify_trial(subject.trial))
    #     subject_json = jsonpickle.encode(subject)
    #     print trial_jsons
    #
    # def JSONify_trial(self, trial_seq):
    #     attempt_jsons = []
    #     for attempt in trial.attempt_seq:
    #         attempt_jsons.append(self.JSONify_attempt(attempt))
    #     trial_json = jsonpickle.encode(trial)
    #     return trial_json
    #
    # def JSONify_attempt(self, attempt):
    #     results_seq_str = jsonpickle.encode(attempt.results_seq)
    #     attempt.results_seq_str = results_seq_str
    #     return jsonpickle.encode(attempt)
    #
    # def JSONify_action(self, action):
    #     return jsonpickle.encode(action)


class TensorboardWriter(object):
    def __init__(self, log_path):
        self.writer = SummaryWriter(log_path)

    def write_scalar(self, name, scalar, epoch):
        self.writer.add_scalar(name, scalar, epoch)

    def close(self):
        self.writer.close()

    def display(self):
        return

    @staticmethod
    def compute_mean(vect, range):
        return np.array(vect[max(len(vect) - range, range) :]).mean()


def obj_dict(obj):
    """
    Get object dict.

    :param obj: An object.
    :return: __dict__ of the object passed in.
    """
    return obj.__dict__
