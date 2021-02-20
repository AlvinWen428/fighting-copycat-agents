"""Main behavioural cloning file."""
import logging

from imitation_learning.supervised import test
from imitation_learning.utils.misc import EarlyStopper
from imitation_learning.utils.experiment import setup_experiment, mean_logs, sort_dict
from imitation_learning.utils.data import cycle
from imitation_learning.evaluation import evaluate_expert, run_eval
from imitation_learning.factory import build_container
from imitation_learning.models import FightCopycatPolicy


def train_it(
        *,
        mode,
        policy_model,
        criterion,
        batch,
        device
):
    """Run single train batch."""

    policy_model.train()

    x, y = mode.batch(batch), batch.labels()

    x, y = x.to(device), y.to(device)

    info = policy_model.train_step(x, y, criterion)

    return info


def train_loop(
        environment,
        policy_model,
        lr_decay_rate,
        lr_decay_times,
        train_dataloader,
        test_dataloader,
        mode,
        logger,
        model_save_dir,
        report_freq,
        eval_runs,
        num_its,
        eval_at_end,
        early_stop,
        dataset_stack_size,
        device
):
    """Main train loop."""
    print("eval at end", eval_at_end)

    criterion = environment.criterion

    early_stopper = EarlyStopper(early_stop, num_its, report_freq, lr_decay_rate=lr_decay_rate,
                                 decay_times_threshold=lr_decay_times)

    train_infos, infos = [], {}
    test_agg_loader = test_dataloader

    train_iter = iter(cycle(train_dataloader))

    it = 0
    for it in range(1, num_its + 1):

        batch = next(train_iter)

        # training loop
        train_info = train_it(
            policy_model=policy_model,
            mode=mode,
            criterion=criterion,
            batch=batch,
            device=device
        )

        train_infos.append(train_info)

        if it % report_freq == 0:
            # Evaluate test loss and test reward
            policy_model.eval()
            agg_train_infos = mean_logs(train_infos[-report_freq:])

            if model_save_dir is not None:
                policy_model.save_model(model_save_dir)

            # start evaluation

            # if not eval_at_end:
            if not eval_at_end and it >= (num_its - 5 * report_freq):
                eval_trajectories, eval_data_batch, eval_infos = run_eval(
                    environment=environment,
                    mode=mode,
                    model=policy_model,
                    dataset_stack_size=dataset_stack_size,
                    eval_runs=eval_runs,
                )
            else:
                eval_trajectories, eval_data_batch, eval_infos = [], None, {}

            test_infos = test(
                mode=mode,
                model=policy_model,
                test_dataloader=test_agg_loader,
                criterion=criterion,
            )

            infos = {**agg_train_infos, **test_infos, **eval_infos}
            logger.log_scalars(infos, it)

            log_format = " ".join(
                [k + ": {:.4f}".format(v) for k, v in sort_dict(infos).items()]
            )
            logging.info(f"\n{it} {log_format}")

            # Evaluate factors separately
            if early_stopper(infos["test_loss"], policy=policy_model, it=it):
                logging.info("Converged")
                logging.info("learning rate is reduced for {} times".format(early_stopper.decay_times))

    if eval_at_end:
        policy_model.eval()
        _, _, eval_infos = run_eval(
            environment=environment,
            mode=mode,
            model=policy_model,
            dataset_stack_size=dataset_stack_size,
            eval_runs=eval_runs,
        )
        logger.log_scalars(eval_infos, it)
        log_format = " ".join(
            [k + ": {:.4f}".format(v) for k, v in sort_dict(eval_infos).items()]
        )
        logging.info(f"\n{it} {log_format}")

    return infos.get("eval_rew", 0)


def imitate(config):
    container = build_container(config)

    setup_experiment(container, config)

    environment = container.environment
    out_dir_fn = container.out_dir_fn
    state_dim = container.mode.shape()[0] // container.config.stack_size
    device = container.device

    learning_rate = container.config.optim.learning_rate

    imitator = FightCopycatPolicy(state_dim, container.config.stack_size, container.output_dims,
                                  container.config.policy.policy_mode, load_path=container.config.load_path,
                                  device=device, learning_rate=learning_rate,
                                  discriminator_lr=container.config.optim.discriminator_lr,
                                  embedding_noise_std=container.config.policy.embedding_noise_std,
                                  gan_loss_weight=container.config.policy.gan_loss_weight)

    # Initiate environment and expert
    with environment.setup():

        if container.config.eval_expert:
            expert_rew, expert_repeat, _ = evaluate_expert(
                environment,
                n=container.config.eval_runs,
                render=container.config.render,
            )

            container.logger.log_scalars(
                {"expert_rew": expert_rew, "expert_repeat": expert_repeat}, 0
            )
            logging.info(
                "Expert reward: {:.4f} repeat: {:.4f}".format(expert_rew, expert_repeat)
            )

        results = train_loop(
            environment=environment,
            policy_model=imitator,
            lr_decay_rate=container.config.optim.lr_decay_rate,
            lr_decay_times=container.config.optim.lr_decay_times,
            train_dataloader=container.train_dataloader,
            test_dataloader=container.test_dataloader,
            mode=container.mode,
            logger=container.logger,
            model_save_dir=out_dir_fn("imitators"),
            early_stop=container.config.early_stop,
            eval_at_end=container.config.eval_at_end,
            eval_runs=container.config.eval_runs,
            num_its=container.config.num_its,
            report_freq=container.config.report_freq,
            dataset_stack_size=container.dataset_stack_size,
            device=device
        )

        container.logger.close()

        return results
