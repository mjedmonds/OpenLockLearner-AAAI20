from openlockagents.common.agent import Agent


class HumanAgent(Agent):
    def __init__(self, params, env):
        super(HumanAgent, self).__init__("Human", params, env)

        self.human = True

        participant_id, age, gender, handedness, eyewear, major = self.prompt_subject()

        super(HumanAgent, self).setup_subject(
            human=True,
            participant_id=participant_id,
            age=age,
            gender=gender,
            handedness=handedness,
            eyewear=eyewear,
            major=major,
            project_src=params["src_dir"]
        )

    def finish_subject(self, strategy="human", transfer_strategy="human"):
        strategy = self.prompt_strategy()
        transfer_strategy = self.prompt_transfer_strategy()
        super(HumanAgent, self).finish_subject(strategy, transfer_strategy)

    # code to run a human subject
    def run_trial_human(
        self,
        scenario_name,
        action_limit,
        attempt_limit,
        specified_trial=None,
        verify=False,
        test_trial=False,
    ):
        """
        Run trial for a human subject.

        :param scenario_name: name of scenario (e.g. those defined in settings_trial.PARAMS)
        :param action_limit: number of actions permitted
        :param attempt_limit: number of attempts permitted
        :param specified_trial: optional specified trial
        :param verify: flag to indicate whether or not to call verify_fsm_matches_simulator()
        :param test_trial: default: False
        :return: Nothing
        """
        self.env.human_agent = True
        trial_selected = self.setup_trial(
            scenario_name, action_limit, attempt_limit, specified_trial
        )

        obs_space = None
        while not self.determine_trial_finished(attempt_limit):
            done = False
            self.env.reset()
            while not done:
                self.env.render(self.env)

                done = self.env.determine_attempt_finished()

                # acknowledge any acks that may have occurred (action executed, attempt ended, etc)
                # self.update_acknowledgments()
                # used to verify simulator and fsm states are always the same (they should be)
                if verify:
                    obs_space = self.verify_fsm_matches_simulator(obs_space)

            self.finish_attempt()

        self.finish_trial(trial_selected, test_trial)

    @staticmethod
    def prompt_subject():
        print("Welcome to OpenLock!")
        participant_id = HumanAgent.prompt_participant_id()
        age = HumanAgent.prompt_age()
        gender = HumanAgent.prompt_gender()
        handedness = HumanAgent.prompt_handedness()
        eyewear = HumanAgent.prompt_eyewear()
        major = HumanAgent.prompt_major()
        return participant_id, age, gender, handedness, eyewear, major

    @staticmethod
    def prompt_participant_id():
        while True:
            try:
                participant_id = int(
                    input("Please enter the participant ID (ask the RA for this): ")
                )
            except ValueError:
                print("Please enter an integer for the participant ID")
                continue
            else:
                return participant_id

    @staticmethod
    def prompt_age():
        while True:
            try:
                age = int(input("Please enter your age: "))
            except ValueError:
                print("Please enter your age as an integer")
                continue
            else:
                return age

    @staticmethod
    def prompt_gender():
        while True:
            gender = input(
                "Please enter your gender ('M' for male, 'F' for female, or 'O' for other): "
            )
            if gender == "M" or gender == "F" or gender == "O":
                return gender
            else:
                continue

    @staticmethod
    def prompt_handedness():
        while True:
            handedness = input(
                "Please enter your handedness ('right' for right-handed or 'left' for left-handed): "
            )
            if handedness == "right" or handedness == "left":
                return handedness
            else:
                continue

    @staticmethod
    def prompt_eyewear():
        while True:
            eyewear = input(
                "Please enter 'yes' if you wear glasses or contacts or 'no' if you do not wear glasses or contacts: "
            )
            if eyewear == "yes" or eyewear == "no":
                return eyewear
            else:
                continue

    @staticmethod
    def prompt_major():
        major = input("Please enter your major: ")
        return major

    @staticmethod
    def prompt_strategy():
        print(
            "Did you develop any particular technique or strategy to solve the problem? If so, what was your technique/strategy? "
        )
        strategy = input()
        return strategy

    @staticmethod
    def prompt_transfer_strategy():
        print(
            "If you used a particular technique/strategy, did you find that it also worked when the number of colored levers increased from 3 to 4? "
        )
        transfer_strategy = input()
        return transfer_strategy
