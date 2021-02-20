"""Helper functions for evaluating models in the environment."""
import random
from typing import List
from itertools import count
import numpy as np
from functools import partial

import torch
from tqdm import tqdm

from imitation_learning.utils.misc import call_single, no_grad
from imitation_learning.utils.data import TensorLoader
from imitation_learning.trajectory import Trajectory, TransitionDataset, Batch
from imitation_learning.modes import ModeType
from imitation_learning.models import ImitationPolicy


@no_grad()
def run_policy_gen(
    mode: ModeType,
    environment,
    agent,
    collect_info=False,
):
    env = environment.env
    expert = partial(call_single, environment.expert_fn)
    expert_mode = environment.expert_mode
    expert_steps = environment.expert_steps

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for it in count():
        state, done, step = env.reset(), False, 0
        trajectory = None
        while not done:
            pixels = None
            if (
                step < expert_steps
                or trajectory is None
                or len(trajectory) < mode.stack_need - 1
            ):
                x = expert_mode.step(state, pixels, trajectory)
                action = expert(x)
            else:
                x = mode.step(state, pixels, trajectory)
                if isinstance(agent, ImitationPolicy):
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x)
                    x = x.to(device)
                    x = x.unsqueeze(0)
                    output = agent(x)
                    action = output.data.detach().cpu().numpy()
                else:
                    action = agent(x)

            prev_action, prev_state, prev_pixels = action, state, pixels
            state, rew, done, info = env.step(action)

            if not collect_info:
                info = None

            trajectory = Trajectory.add_step(
                trajectory, prev_state, prev_action, rew, prev_pixels, info=info
            )
            step += 1

        trajectory.finished()
        yield trajectory


@no_grad()
def run_policy(
    mode: ModeType,
    environment,
    agent,
    num_trajectories=None,
    num_steps=None,
    collect_info=False,
    progress_bar=False,
) -> List[Trajectory]:
    """Evaluate model and collect data.

    Return:
        - Reward mean
        - Trajectory state data
        - Trajectory pixel data (if requested in kwargs)
        - The ratio of actions that repeated the previous action
    """
    assert (num_trajectories is None) != (
        num_steps is None
    ), "Provide num_trajectories XOR num_steps"
    gen = run_policy_gen(
        mode,
        environment,
        agent,
        collect_info=collect_info,
    )

    trajectories = []
    if num_trajectories is not None:
        iterator = tqdm(range(num_trajectories), disable=not progress_bar)
        progress_bar = iterator
    else:
        iterator = count()
        progress_bar = tqdm(total=num_steps, disable=not progress_bar)

    steps = 0
    for _ in iterator:
        trajectory = next(gen)
        trajectories.append(trajectory)
        steps += len(trajectory)

        if num_steps is not None:
            progress_bar.update(len(trajectory))
            if steps >= num_steps:
                break

    progress_bar.close()
    gen.close()
    return trajectories


def run_expert(environment, n=20, expert_noise=0.0):
    def act(x):
        if random.random() < expert_noise:
            return environment.env.action_space.sample()
        else:
            return call_single(environment.expert_fn, x)

    return run_policy(
        environment.expert_mode, environment, act, num_trajectories=n
    )


def evaluate_expert(environment, n=20, render=False):
    """Evaluate expert agent. Return achieved reward."""
    trajectories = run_expert(environment, n, render)
    return (
        Trajectory.reward_sum_mean(trajectories),
        Trajectory.action_repeat_mean(trajectories),
        trajectories,
    )


def run_eval(
    *,
    environment,
    mode,
    model,
    dataset_stack_size,
    eval_runs,
    progress_bar=False,
):
    """Evaluate policy and label trajectories with expert."""

    eval_trajectories = run_policy(
        mode,
        environment,
        model,
        num_trajectories=eval_runs,
        collect_info=True,
        progress_bar=progress_bar,
    )

    eval_dataset, eval_acc = create_labelled_transition_dataset(
        eval_trajectories, environment, dataset_stack_size
    )

    rewards = [t.reward_sum() for t in eval_trajectories]

    infos = {
        "eval_rew": Trajectory.reward_sum_mean(eval_trajectories),
        "eval_rew_std": np.std(rewards).item(),
        "eval_repeats": Trajectory.action_repeat_mean(eval_trajectories),
        "eval_time": np.mean([len(t) for t in eval_trajectories]).item(),
        "eval_acc": eval_acc,
        **{
            k: Trajectory.info_sum_mean(k, eval_trajectories)
            for k in getattr(environment, "log_info_sum", [])
        },
    }

    return eval_trajectories, eval_dataset, infos


@no_grad()
def label_trajectories(expert_fn, dataset, expert_mode):
    loader = TensorLoader(dataset, batch_size=64, shuffle=False)
    expert_actions = []
    expert_action_indices = []
    agreements = []
    for batch in loader:
        x, imitator_a = expert_mode.batch(batch), batch.actions[:, -1]
        expert_a = torch.from_numpy(expert_fn(x.cpu().numpy()))
        if expert_a.dim() == 1:
            expert_a = expert_a[:, None]
        expert_actions.append(expert_a)
        expert_action_indices.append(batch.indices[:, -1])
        agreements.append(expert_a.view(-1) == imitator_a.contiguous().view(-1))
    return (
        torch.cat(expert_actions),
        torch.cat(expert_action_indices),
        torch.cat(agreements),
    )


@no_grad()
def create_labelled_transition_dataset(trajectories, environment, dataset_stack_size):
    unlabelled_dataset = TransitionDataset.from_trajectories(
        trajectories, dataset_stack_size
    )

    expert_actions, expert_action_indices, agreements = label_trajectories(
        environment.expert_fn, unlabelled_dataset, environment.expert_mode
    )
    agreements = agreements.to(torch.float32).mean().item()

    full_expert_actions = expert_actions.new_full(
        (len(unlabelled_dataset.states), *expert_actions.shape[1:]),
        Batch.absent_sentinel,
    )
    full_expert_actions[expert_action_indices] = expert_actions

    dataset = TransitionDataset.from_trajectories(
        trajectories, dataset_stack_size, expert_actions=full_expert_actions
    )

    dataset.actions[expert_action_indices] = expert_actions.view(
        len(expert_actions), -1
    )
    return dataset, agreements
