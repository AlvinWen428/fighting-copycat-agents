import numpy as np
from imitation_learning.utils.misc import call_single
from imitation_learning.evaluation import run_policy
import argparse


def generate(environment, n, **kwargs):
    def act(x):
        return call_single(environment.expert_fn, x)

    return run_policy(
        environment.expert_mode,
        environment,
        agent=act,
        num_steps=n,
        progress_bar=True,
        **kwargs,
    )


def main(environment, **kwargs):
    print("Starting " + repr(environment))
    print(environment.data_path)
    with environment.setup():
        data = list(generate(environment, **kwargs))
        np.save(environment.data_path, data)
    print("Finished " + repr(environment))


def mainmain():
    from imitation_learning.environments import construct_environment

    parser = argparse.ArgumentParser("Data generation")
    parser.add_argument("env")
    parser.add_argument("num", type=int)
    args = parser.parse_args()

    env = construct_environment(args.env)

    main(env, n=args.num)


if __name__ == "__main__":
    mainmain()
