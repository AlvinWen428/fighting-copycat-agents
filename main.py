from sys import argv

from sacred.arg_parser import get_config_updates

from imitation_learning.imitate import imitate
from imitation_learning.utils.experiment import combine_configs
from imitation_learning.utils.commands import normalize_command


command, *updates = argv[1:]
config_updates, config_paths = get_config_updates(updates)
configs = combine_configs(config_paths, config_updates)


configs['command'] = command

commands = {
    'main': imitate,
    'normalize': normalize_command
}

assert command in commands, f"Command {command} not found"

commands[command](configs)
