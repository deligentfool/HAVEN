from .. import MultiAgentEnv
import gfootball.env as football_env
from gfootball.env import observation_preprocessing
import gym
import numpy as np


class Academy_Corner(MultiAgentEnv):

    def __init__(
        self,
        dense_reward=False,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
        dump_freq=1000,
        render=False,
        n_agents=4,
        time_limit=150,
        time_step=0,
        obs_dim=34,
        env_name='academy_corner',
        stacked=False,
        representation="simple115v2",
        rewards='scoring',
        logdir='football_dumps',
        write_video=False,
        number_of_right_players_agent_controls=0,
        reward_sparse=False,
        reward_max=10.,
        reward_positive=False,
        reward_reset_punish=False,
        seed=0
    ):
        self.dense_reward = dense_reward
        self.write_full_episode_dumps = write_full_episode_dumps
        self.write_goal_dumps = write_goal_dumps
        self.dump_freq = dump_freq
        self.render = render
        self.n_agents = n_agents
        self.episode_limit = time_limit
        self.time_step = time_step
        self.obs_dim = obs_dim
        self.env_name = env_name
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls
        self.seed = seed
        self.reward_sparse = reward_sparse
        self.reward_max = reward_max
        self.reward_positive = reward_positive
        self.reward_reset_punish = reward_reset_punish

        self.env = football_env.create_environment(
            write_full_episode_dumps=self.write_full_episode_dumps,
            write_goal_dumps=self.write_goal_dumps,
            env_name=self.env_name,
            stacked=self.stacked,
            representation=self.representation,
            rewards=self.rewards,
            logdir=self.logdir,
            render=self.render,
            write_video=self.write_video,
            dump_frequency=self.dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls=self.number_of_right_players_agent_controls,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT))
        self.env.seed(self.seed)

        obs_space_low = self.env.observation_space.low[0][:self.obs_dim]
        obs_space_high = self.env.observation_space.high[0][:self.obs_dim]

        self.action_space = [gym.spaces.Discrete(
            self.env.action_space.nvec[1]) for _ in range(self.n_agents)]
        self.observation_space = [
            gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype) for _ in range(self.n_agents)
        ]

        self.n_actions = self.action_space[0].n

        self.unit_dim = self.obs_dim  # QPLEX unit_dim for cds_gfootball
        # self.unit_dim = 8  # QPLEX unit_dim set like that in Starcraft II
        full_obs = self.env.unwrapped.observation()[0]
        ball_pos = full_obs['ball']
        distance_to_goal = np.sqrt((ball_pos[0] - 1.) ** 2 + (ball_pos[1] - 0.) ** 2)
        self.reward_scale = self.reward_max / 1.
        self.last_distance = distance_to_goal

    def get_simple_obs(self, index=-1):
        full_obs = self.env.unwrapped.observation()[0]
        simple_obs = []

        if index == -1:
            # global state, absolute position
            simple_obs.append(full_obs['left_team']
                              [-self.n_agents:].reshape(-1))
            simple_obs.append(
                full_obs['left_team_direction'][-self.n_agents:].reshape(-1))

            simple_obs.append(full_obs['right_team'][0])
            simple_obs.append(full_obs['right_team'][1])
            simple_obs.append(full_obs['right_team'][2])
            simple_obs.append(full_obs['right_team_direction'][0])
            simple_obs.append(full_obs['right_team_direction'][1])
            simple_obs.append(full_obs['right_team_direction'][2])

            simple_obs.append(full_obs['ball'])
            simple_obs.append(full_obs['ball_direction'])

        else:
            # local state, relative position
            ego_position = full_obs['left_team'][-self.n_agents +
                                                 index].reshape(-1)
            simple_obs.append(ego_position)
            simple_obs.append((np.delete(
                full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position).reshape(-1))

            simple_obs.append(
                full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))
            simple_obs.append(np.delete(
                full_obs['left_team_direction'][-self.n_agents:], index, axis=0).reshape(-1))

            simple_obs.append(full_obs['right_team'][0] - ego_position)
            simple_obs.append(full_obs['right_team'][1] - ego_position)
            simple_obs.append(full_obs['right_team'][2] - ego_position)
            simple_obs.append(full_obs['right_team_direction'][0])
            simple_obs.append(full_obs['right_team_direction'][1])
            simple_obs.append(full_obs['right_team_direction'][2])

            simple_obs.append(full_obs['ball'][:2] - ego_position)
            simple_obs.append(full_obs['ball'][-1].reshape(-1))
            simple_obs.append(full_obs['ball_direction'])

        simple_obs = np.concatenate(simple_obs)
        return simple_obs

    def get_global_state(self):
        return self.get_simple_obs(-1)

    def check_if_done(self):
        cur_obs = self.env.unwrapped.observation()[0]
        ball_loc = cur_obs['ball']
        ours_loc = cur_obs['left_team'][-self.n_agents:]

        if ball_loc[0] < 0 or any(ours_loc[:, 0] < 0):
            return True

        return False

    def step(self, actions):
        """Returns reward, terminated, info."""
        self.time_step += 1
        _, original_rewards, done, infos = self.env.step(
            actions.to('cpu').numpy().tolist())
        rewards = list(original_rewards)
        # obs = np.array([self.get_obs(i) for i in range(self.n_agents)])

        if self.time_step >= self.episode_limit:
            done = True

        if self.check_if_done():
            done = True

        if self.reward_sparse:
            if sum(rewards) <= 0:
                # return obs, self.get_global_state(), -int(done), done, infos
                return -int(done), done, infos, sum(rewards) > 0
            # return obs, self.get_global_state(), 100, done, infos
            return self.reward_max, done, infos, sum(rewards) > 0
        else:
            full_obs = self.env.unwrapped.observation()[0]
            ball_pos = full_obs['ball']
            distance = np.sqrt((ball_pos[0] - 1.) ** 2 + (ball_pos[1] - 0.) ** 2)
            reward = (self.last_distance - distance) * self.reward_scale
            if self.reward_positive:
                if reward > 0:
                    self.last_distance = distance
                else:
                    reward = 0
            else:
                self.last_distance = distance
                if reward < -0.5 * self.reward_max and not self.reward_reset_punish:
                    reward = 0
            # * when done, the ball position reset to the (0, 0)
            return reward if sum(rewards) <= 0 else self.reward_max, done, infos, sum(rewards) > 0

    def get_obs(self):
        """Returns all agent observations in a list."""
        # obs = np.array([self.get_simple_obs(i) for i in range(self.n_agents)])
        obs = [self.get_simple_obs(i) for i in range(self.n_agents)]
        return obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.get_simple_obs(agent_id)

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_dim

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        # TODO: in wrapper_grf_3vs1.py, author set state_shape=obs_shape
        return self.obs_dim

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n

    def reset(self):
        """Returns initial observations and states."""
        self.time_step = 0
        self.env.reset()
        obs = np.array([self.get_simple_obs(i) for i in range(self.n_agents)])
        # * reset the last distance
        full_obs = self.env.unwrapped.observation()[0]
        ball_pos = full_obs['ball']
        self.last_distance = np.sqrt((ball_pos[0] - 1.) ** 2 + (ball_pos[1] - 0.) ** 2)

        return obs, self.get_global_state()

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass
