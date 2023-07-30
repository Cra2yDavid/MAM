import pickle
import time
from typing import Any, Callable, Dict, Optional, Union
import numpy as np
from collector import Collector
from tianshou.policy import BasePolicy
from tianshou.utils import BaseLogger
import os
from tianshou.utils.logger.tensorboard import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

code_path = 'data'
ROOT_PATH = os.path.join(code_path)


class NewLogger(TensorboardLogger):

    def __init__(
        self,
        writer: SummaryWriter,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        save_interval: int = 1,
    ) -> None:
        super().__init__(writer, train_interval, test_interval, update_interval, save_interval)

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        assert collect_result["n/ep"] > 0
        if step - self.last_log_test_step >= self.test_interval:
            log_data = {
                "test/env_step": step,
                "test/reward": collect_result["rew"],
                "test/length": collect_result["len"],
                "test/reward_std": collect_result["rew_std"],
                "test/length_std": collect_result["len_std"],
                # change
                "test/success_rate": collect_result["success_rate"]
            }
            self.write("test/env_step", step, log_data)
            self.last_log_test_step = step


def save_variable(v, filename):
    """
    sava python variable
    :param v: variable
    :param filename: filename to save
    """
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()


def load_variable(filename):
    """
    load python variable
    :param filename: filename to load
    :return v: loaded variable
    """
    f = open(filename, 'rb')
    v = pickle.load(f)
    f.close()
    return v


def test_episode(
    policy: BasePolicy,
    collector: Collector,
    test_fn: Optional[Callable[[int, Optional[int]], None]],
    epoch: int,
    n_episode: int,
    logger: Optional[BaseLogger] = None,
    global_step: Optional[int] = None,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """A simple wrapper of testing policy in collector."""
    collector.reset_env()
    collector.reset_buffer()
    policy.eval()
    if test_fn:
        test_fn(epoch, global_step)
    result = collector.collect(n_episode=n_episode, log_success=True)
    if reward_metric:
        rew = reward_metric(result["rews"])
        result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
    if logger and global_step is not None:
        logger.log_test_data(result, global_step)
    return result


def gather_info(
    start_time: float,
    train_c: Optional[Collector],
    test_c: Optional[Collector],
    best_reward: float,
    best_reward_std: float,
) -> Dict[str, Union[float, str]]:
    """A simple wrapper of gathering information from collectors.

    :return: A dictionary with the following keys:

        * ``train_step`` the total collected step of training collector;
        * ``train_episode`` the total collected episode of training collector;
        * ``train_time/collector`` the time for collecting transitions in the \
            training collector;
        * ``train_time/model`` the time for training models;
        * ``train_speed`` the speed of training (env_step per second);
        * ``test_step`` the total collected step of test collector;
        * ``test_episode`` the total collected episode of test collector;
        * ``test_time`` the time for testing;
        * ``test_speed`` the speed of testing (env_step per second);
        * ``best_reward`` the best reward over the test results;
        * ``duration`` the total elapsed time.
    """
    duration = time.time() - start_time
    model_time = duration
    result: Dict[str, Union[float, str]] = {
        "duration": f"{duration:.2f}s",
        "train_time/model": f"{model_time:.2f}s",
    }
    if test_c is not None:
        model_time = duration - test_c.collect_time
        test_speed = test_c.collect_step / test_c.collect_time
        result.update(
            {
                "test_step": test_c.collect_step,
                "test_episode": test_c.collect_episode,
                "test_time": f"{test_c.collect_time:.2f}s",
                "test_speed": f"{test_speed:.2f} step/s",
                "best_reward": best_reward,
                "best_result": f"{best_reward:.2f} ± {best_reward_std:.2f}",
                "duration": f"{duration:.2f}s",
                "train_time/model": f"{model_time:.2f}s",
            }
        )
    if train_c is not None:
        model_time -= train_c.collect_time
        if test_c is not None:
            train_speed = train_c.collect_step / (duration - test_c.collect_time)
        else:
            train_speed = train_c.collect_step / duration
        result.update(
            {
                "train_step": train_c.collect_step,
                "train_episode": train_c.collect_episode,
                "train_time/collector": f"{train_c.collect_time:.2f}s",
                "train_time/model": f"{model_time:.2f}s",
                "train_speed": f"{train_speed:.2f} step/s",
            }
        )
    return result
