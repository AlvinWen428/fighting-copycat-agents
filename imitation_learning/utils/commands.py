from imitation_learning.factory import build_container
from imitation_learning.modes import normalize


def normalize_command(config):
    container = build_container(config)
    normalize(
        container.environment,
        container.dataset,
        container.config.mode,
        container.mode_unnormalized,
    )
