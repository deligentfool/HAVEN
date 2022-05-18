import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class HAVENLearner:
    def __init__(self, mac, macro_mac, value_mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.macro_mac = macro_mac
        self.value_mac = value_mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.macro_params = list(macro_mac.parameters())
        self.value_params = list(value_mac.parameters())

        self.last_target_update_episode = 0
        self.last_target_macro_update_episode = 0
        self.last_target_value_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.value_mixer = None
        if args.value_mixer is not None:
            if args.value_mixer == "vdn":
                self.value_mixer = VDNMixer()
            elif args.value_mixer == "qmix":
                self.value_mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.value_mixer))
            self.value_params += list(self.value_mixer.parameters())
            self.target_value_mixer = copy.deepcopy(self.value_mixer)

        self.macro_mixer = None
        if args.macro_mixer is not None:
            if args.macro_mixer == "vdn":
                self.macro_mixer = VDNMixer()
            elif args.macro_mixer == "qmix":
                self.macro_mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.macro_mixer))
            self.macro_params += list(self.macro_mixer.parameters())
            self.target_macro_mixer = copy.deepcopy(self.macro_mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.macro_optimiser = RMSprop(params=self.macro_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.value_optimiser = RMSprop(params=self.value_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.target_value_mac = copy.deepcopy(value_mac)
        self.target_macro_mac = copy.deepcopy(macro_mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def value_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["macro_reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        value_out = []
        self.value_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            value = self.value_mac.forward(batch, t=t)
            value_out.append(value)
        value_out = th.stack(value_out, dim=1)

        mac_out = []
        self.macro_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.macro_mac.forward(batch, t=t)
            mac_out.append(agent_outs)

        mac_out = th.stack(mac_out[1:], dim=1)
        max_qvals = mac_out.max(dim=3)[0]
        max_qvals = self.macro_mixer(max_qvals, batch["state"][:, 1:])

        values = self.value_mixer(value_out[:, :-1], batch["state"][:, :-1])
        #target_values = self.target_value_mixer(target_value_out[:, 1:], batch["state"][:, 1:])

        #target_values = rewards + self.args.gamma * max_qvals * (1 - terminated)
        target_values = rewards + self.args.gamma * max_qvals * (1 - terminated)
        td_loss = (values - target_values.detach())

        mask = mask.expand_as(td_loss)
        masked_loss = td_loss * mask
        masked_loss = (masked_loss ** 2).sum() / mask.sum()

        self.value_optimiser.zero_grad()
        masked_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.value_params, self.args.grad_norm_clip)
        self.value_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_value_targets()
            self.last_target_value_update_episode = episode_num
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("value_loss", masked_loss.item(), t_env)


    def train(self, batch: EpisodeBatch, macro_batch: EpisodeBatch, t_env: int, episode_num: int):
        actions = batch["actions"][:, :-1]
        intrinsic_reward = self.calc_intrinsic_reward(batch, macro_batch)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = intrinsic_reward + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env


    def macro_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["macro_reward"][:, :-1]
        actions = batch["macro_actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Calculate estimated Q-Values
        mac_out = []
        self.macro_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.macro_mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_macro_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_macro_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.macro_mixer is not None:
            chosen_action_qvals = self.macro_mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_macro_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.macro_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.macro_params, self.args.grad_norm_clip)
        self.macro_optimiser.step()
        if (episode_num - self.last_target_macro_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_macro_targets()
            self.last_target_macro_update_episode = episode_num
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("macro_loss", loss.item(), t_env)


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _update_macro_targets(self):
        self.target_macro_mac.load_state(self.macro_mac)
        if self.macro_mixer is not None:
            self.target_macro_mixer.load_state_dict(self.macro_mixer.state_dict())
        self.logger.console_logger.info("Updated target macro network")

    def _update_value_targets(self):
        self.target_value_mac.load_state(self.value_mac)
        if self.value_mixer is not None:
            self.target_value_mixer.load_state_dict(self.value_mixer.state_dict())
        self.logger.console_logger.info("Updated target value network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.macro_mac.cuda()
        self.target_macro_mac.cuda()
        self.value_mac.cuda()
        self.target_value_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.macro_mixer is not None:
            self.macro_mixer.cuda()
            self.target_macro_mixer.cuda()
        if self.value_mixer is not None:
            self.value_mixer.cuda()
            self.target_value_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        self.value_mac.save_models(path)
        if self.value_mixer is not None:
            th.save(self.value_mixer.state_dict(), "{}/value_mixer.th".format(path))
        th.save(self.value_optimiser.state_dict(), "{}/value_opt.th".format(path))
        self.macro_mac.save_models(path)
        if self.macro_mixer is not None:
            th.save(self.macro_mixer.state_dict(), "{}/macro_mixer.th".format(path))
        th.save(self.macro_optimiser.state_dict(), "{}/macro_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.value_mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_value_mac.load_models(path)
        if self.value_mixer is not None:
            self.value_mixer.load_state_dict(th.load("{}/value_mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.value_optimiser.load_state_dict(th.load("{}/value_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.macro_mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_macro_mac.load_models(path)
        if self.macro_mixer is not None:
            self.macro_mixer.load_state_dict(th.load("{}/macro_mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.macro_optimiser.load_state_dict(th.load("{}/macro_opt.th".format(path), map_location=lambda storage, loc: storage))


    def calc_intrinsic_reward(self, batch, macro_batch):
        origin_reward = batch["reward"][:, :-1]
        if self.args.mean_weight:
            origin_reward = th.ones_like(origin_reward)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        value_out = []
        self.value_mac.init_hidden(macro_batch.batch_size)
        for t in range(macro_batch.max_seq_length):
            value = self.value_mac.forward(macro_batch, t=t)
            value_out.append(value)
        value_out = th.stack(value_out, dim=1)
        values = self.value_mixer(value_out, macro_batch["state"])
        #values = value_out.squeeze(-1)

        macro_mac_out = []
        self.macro_mac.init_hidden(macro_batch.batch_size)
        for t in range(macro_batch.max_seq_length):
            agent_outs = self.macro_mac.forward(macro_batch, t=t)
            macro_mac_out.append(agent_outs)
        macro_mac_out = th.stack(macro_mac_out, dim=1) # Concat over time
        macro_mac_out = th.gather(macro_mac_out, dim=3, index=macro_batch["macro_actions"]).squeeze(3)
        macro_mac_out = self.macro_mixer(macro_mac_out, macro_batch["state"])

        #intrinsic_reward = (macro_mac_out[:, :-1] - values[:, :-1])
        intrinsic_reward = (macro_batch["macro_reward"][:, :-1] + self.args.gamma * values[:, 1:] - values[:, :-1])
        intrinsic_reward = intrinsic_reward.unsqueeze(-2)
        gap = intrinsic_reward.size(1) * self.args.k - origin_reward.size(1)
        if gap != 0:
            origin_reward = th.cat([origin_reward, th.zeros([intrinsic_reward.size(0), intrinsic_reward.size(1) * self.args.k - origin_reward.size(1), 1]).cuda()], dim=1)
        origin_reward = origin_reward.view(origin_reward.size(0), -1, self.args.k, 1)
        if not self.args.mean_weight:
            origin_reward[origin_reward == 0] = -9999999
            origin_reward = intrinsic_reward.sign() * origin_reward
        if gap != 0:
            origin_reward[:, :, -gap:] = -9999999
        origin_reward_weight = th.softmax(origin_reward, dim=-2)
        intrinsic_reward = intrinsic_reward * origin_reward_weight
        intrinsic_reward = intrinsic_reward.view(intrinsic_reward.size(0), -1, 1 if self.args.mixer is not None else self.args.n_agents)
        intrinsic_reward = (intrinsic_reward[:, :batch.max_seq_length-1] * mask).detach()
        intrinsic_reward = self.args.intrinsic_switch * intrinsic_reward + self.args.reward_switch * batch["reward"][:, :batch.max_seq_length-1]
        return intrinsic_reward