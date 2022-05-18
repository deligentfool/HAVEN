from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch as th
import csv
import os


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.test_wins = []

        # Log the first run
        self.log_train_stats_t = -1000000
        env_name = args.env_args['env_name']

        self.csv_dir = f'./csv_files/{args.name[:-6]}/{env_name}/'
        self.csv_path = f'{self.csv_dir}seed_{args.seed}.csv'
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)

    def setup(self, scheme, macro_scheme, groups, preprocess, macro_preprocess, mac, macro_mac, value_mac, learner):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.new_macro_batch = partial(EpisodeBatch, macro_scheme, groups, self.batch_size, (self.episode_limit // self.args.k) + 1 + (self.episode_limit % self.args.k != 0),
                                       preprocess=macro_preprocess, device=self.args.device)
        self.mac = mac
        self.macro_mac = macro_mac
        self.value_mac = value_mac
        self.learner = learner

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.macro_batch = self.new_macro_batch()
        self.env.reset()
        self.t = 0

    def writereward(self, win_rate, step):
        if os.path.isfile(self.csv_path):
            with open(self.csv_path, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([step, win_rate])
        else:
            with open(self.csv_path, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['step', 'win_rate'])
                csv_write.writerow([step, win_rate])

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        win_tag = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        self.macro_mac.init_hidden(batch_size=self.batch_size)
        self.value_mac.init_hidden(batch_size=self.batch_size)
        env_info = {"alive_allies_list": [1 for _ in range(self.env.n_agents)]}

        macro_reward = 0
        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
            self.batch.update(pre_transition_data, ts=self.t)

            if self.t % self.args.k == 0:
                pre_macro_transition_data = {
                    "state": [self.env.get_state()],
                    "obs": [self.env.get_obs()],
                }
                self.macro_batch.update(pre_macro_transition_data, ts=self.t // self.args.k)


            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if self.t % self.args.k == 0 and self.t != 0:
                post_macro_transition_data = {
                    "macro_actions": macro_actions,
                    "macro_reward": [(macro_reward,)],
                    "terminated": [(False,)]
                }
                macro_reward = 0
                self.macro_batch.update(post_macro_transition_data, ts=self.t//self.args.k-1)
            if self.t % self.args.k == 0:
                macro_actions = self.macro_mac.select_actions(self.macro_batch, t_ep=self.t//self.args.k, t_env=self.t_env, test_mode=test_mode)
            pre_transition_data = {
                "subgoals": macro_actions,
            }
            self.batch.update(pre_transition_data, ts=self.t)
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info, win = self.env.step(actions[0])
            episode_return += reward
            macro_reward += reward
            win_tag += win

            post_transition_data = {
                "reward": [(reward,)],
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)]
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        post_macro_transition_data = {
            "macro_actions": macro_actions,
            "macro_reward": [(macro_reward,)],
            "terminated": [(terminated != env_info.get("episode_limit", False),)]
        }
        #macro_index = (self.t - 1) // self.args.k if ((self.t - 1) % self.args.k) == 0 else (self.t - 1) // self.args.k + 1
        macro_index = (self.t - 1) // self.args.k
        self.macro_batch.update(post_macro_transition_data, ts=macro_index)
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }

        last_macro_data = {
            "state": [self.env.get_state()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)
        self.macro_batch.update(last_macro_data, ts=macro_index+1)

        # Select actions in the last stored state
        macro_actions = self.macro_mac.select_actions(self.macro_batch, t_ep=macro_index+1, t_env=self.t_env, test_mode=test_mode)
        pre_transition_data = {
            "subgoals": macro_actions,
        }
        self.batch.update(pre_transition_data, ts=self.t)
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.macro_batch.update({"macro_actions": macro_actions}, ts=macro_index+1)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        if test_mode:
            self.test_wins.append(win_tag)
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            cur_wins_mean = np.array(
                [0 if win <= 0 else 1 for win in self.test_wins]).mean()
            self.writereward(cur_wins_mean, self.t_env)
            self.test_wins.clear()

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
        return self.batch, self.macro_batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()