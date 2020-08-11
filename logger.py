import openlockagents.common.logger_agent
import openlock.logger_env

# file maintained for backward compatibility with existing human subject logs
# provides a mapping from previous logger structure to new logger structure
SubjectLog = openlockagents.common.logger_agent.SubjectLogger
TrialLog = openlock.logger_env.TrialLog
AttemptLog = openlock.logger_env.AttemptLog
ActionLog = openlock.logger_env.ActionLog
