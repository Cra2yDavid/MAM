import time
from collections import defaultdict
from typing import Callable, Dict, Optional, Union

import numpy as np
import tqdm

from collector import Collector
from tianshou.policy import BasePolicy
from utils import gather_info, test_episode
from tianshou.utils import BaseLogger, LazyLogger, MovAvg, tqdm_config


def onpolicy_trainer(
    policy: BasePolicy,
    train_collector: Collector,
    test_collector: Optional[Collector],
    max_epoch: int,
    step_per_epoch: int,
    repeat_per_collect: int,
    episode_per_test: int,
    batch_size: int,
    step_per_collect: Optional[int] = None,
    episode_per_collect: Optional[int] = None,
    train_fn: Optional[Callable[[int, int], None]] = None,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    stop_fn: Optional[Callable[[float], bool]] = None,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    resume_from_log: bool = False,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    logger: BaseLogger = LazyLogger(),
    verbose: bool = True,
    test_in_train: bool = True,
) -> Dict[str, Union[float, str]]:
    """A wrapper for on-policy trainer procedure.

    The "step" in trainer means an environment step (a.k.a. transition).

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing. If it's None, then
        no testing will be performed.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int repeat_per_collect: the number of repeat time for policy learning, for
        example, set it to 2 means the policy needs to learn each given batch data
        twice.
    :param int episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in the
        policy network.
    :param int step_per_collect: the number of transitions the collector would collect
        before the network update, i.e., trainer will collect "step_per_collect"
        transitions and do some policy network update repeatedly in each epoch.
    :param int episode_per_collect: the number of episodes the collector would collect
        before the network update, i.e., trainer will collect "episode_per_collect"
        episodes and do some policy network update repeatedly in each epoch.
    :param function train_fn: a hook called at the beginning of training in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function test_fn: a hook called at the beginning of testing in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function save_fn: a hook called when the undiscounted average mean reward in
        evaluation phase gets better, with the signature ``f(policy: BasePolicy) ->
        None``.
    :param function save_checkpoint_fn: a function to save training process, with the
        signature ``f(epoch: int, env_step: int, gradient_step: int) -> None``; you can
        save whatever you want.
    :param bool resume_from_log: resume env_step/gradient_step and other metadata from
        existing tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature ``f(rewards: np.ndarray
        with shape (num_episode, agent_num)) -> np.ndarray with shape (num_episode,)``,
        used in multi-agent RL. We need to return a single scalar for each episode's
        result to monitor training in the multi-agent RL setting. This function
        specifies what is the desired metric, e.g., the reward of agent 1 or the
        average reward over all agents.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool test_in_train: whether to test in the training phase. Default to True.

    :return: See :func:`~tianshou.trainer.gather_info`.

    .. note::

        Only either one of step_per_collect and episode_per_collect can be specified.
    """
    start_epoch, env_step, gradient_step = 0, 0, 0
    if resume_from_log:
        start_epoch, env_step, gradient_step = logger.restore_data()
    last_rew, last_len = 0.0, 0
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()
    train_collector.reset_stat()
    test_in_train = test_in_train and (
        train_collector.policy == policy and test_collector is not None
    )

    if test_collector is not None:
        test_c: Collector = test_collector  # for mypy
        test_collector.reset_stat()
        test_result = test_episode(
            policy, test_c, test_fn, start_epoch, episode_per_test, logger, env_step,
            reward_metric
        )
        best_epoch = start_epoch
        best_reward, best_reward_std = test_result["rew"], test_result["rew_std"]
    if save_fn:
        save_fn(policy)

    for epoch in range(1 + start_epoch, 1 + max_epoch):
        # train
        policy.train()
        with tqdm.tqdm(
            total=step_per_epoch, desc=f"Epoch #{epoch}", bar_format='{desc}:{percentage:3.0f}%| {n_fmt}/{total_fmt}{postfix}', **tqdm_config
        ) as t:
            while t.n < t.total:
                if train_fn:
                    train_fn(epoch, env_step)
                result = train_collector.collect(
                    n_step=step_per_collect, n_episode=episode_per_collect
                )
                if result["n/ep"] > 0 and reward_metric:
                    rew = reward_metric(result["rews"])
                    result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
                env_step += int(result["n/st"])
                t.update(result["n/st"])
                logger.log_train_data(result, env_step)
                last_rew = result['rew'] if result["n/ep"] > 0 else last_rew
                last_len = result['len'] if result["n/ep"] > 0 else last_len
                data = {
                    "env_step": str(env_step),
                    "rew": f"{last_rew:.2f}",
                    "len": str(int(last_len)),
                    "n/ep": str(int(result["n/ep"])),
                    "n/st": str(int(result["n/st"])),
                }
                if result["n/ep"] > 0:
                    if test_in_train and stop_fn and stop_fn(result["rew"]):
                        test_result = test_episode(
                            policy, test_c, test_fn, epoch, episode_per_test, logger,
                            env_step
                        )
                        if stop_fn(test_result["rew"]):
                            if save_fn:
                                save_fn(policy)
                            logger.save_data(
                                epoch, env_step, gradient_step, save_checkpoint_fn
                            )
                            t.set_postfix(**data)
                            return gather_info(
                                start_time, train_collector, test_collector,
                                test_result["rew"], test_result["rew_std"]
                            )
                        else:
                            policy.train()
                losses = policy.update(
                    0,
                    train_collector.buffer,
                    batch_size=batch_size,
                    repeat=repeat_per_collect
                )
                train_collector.reset_buffer(keep_statistics=True)
                step = max(
                    [1] + [len(v) for v in losses.values() if isinstance(v, list)]
                )
                gradient_step += step
                for k in losses.keys():
                    stat[k].add(losses[k])
                    losses[k] = stat[k].get()
                    data[k] = f"{losses[k]:.3f}"
                logger.log_update_data(losses, gradient_step)
                t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        logger.save_data(epoch, env_step, gradient_step, save_checkpoint_fn)
        # test
        if test_collector is not None:
            test_result = test_episode(
                policy, test_c, test_fn, epoch, episode_per_test, logger, env_step,
                reward_metric
            )
            rew, rew_std, success_rate = test_result["rew"], test_result["rew_std"], test_result['success_rate']
            if best_epoch < 0 or best_reward < rew:
                best_epoch, best_reward, best_reward_std = epoch, rew, rew_std
                if save_fn:
                    save_fn(policy)
            if verbose:
                print(
                    f"Epoch #{epoch}: test_reward: {rew:.6f} ± {rew_std:.6f}, success_rate:{success_rate:.2f}, best_rew"
                    f"ard: {best_reward:.6f} ± {best_reward_std:.6f} in #{best_epoch}"
                )
            if stop_fn and stop_fn(best_reward):
                break

    if test_collector is None and save_fn:
        save_fn(policy)

    if test_collector is None:
        return gather_info(start_time, train_collector, None, 0.0, 0.0)
    else:
        return gather_info(
            start_time, train_collector, test_collector, best_reward, best_reward_std
        )
