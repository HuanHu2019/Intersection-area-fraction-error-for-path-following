import os
import torch
import numpy as np
import numpy.random as rd
from copy import deepcopy
from net import QNet, QNetDuel, QNetTwin, QNetTwinDuel
from net import Actor, ActorSAC, ActorPPO
from net import Critic, CriticAdv, CriticTwin
from net import SharedDPG, SharedSPG, SharedPPO

"""[ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)"""


class AgentBase:
    def __init__(self):
        self.learning_rate = 5e-7
        self.soft_update_tau = 2 ** -8  # 5e-3 ~= 2 ** -8
        self.state = None  # set for self.update_buffer(), initialize before training
        self.device = None

        self.act = self.act_target = None
        self.cri = self.cri_target = None
        self.act_optimizer = None
        self.cri_optimizer = None
        self.criterion = None
        self.get_obj_critic = None

    def init(self, net_dim, state_dim, action_dim, if_per=False):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        `int net_dim` the dimension of networks (the width of neural networks)
        `int state_dim` the dimension of state (the number of state vector)
        `int action_dim` the dimension of action (the number of discrete action)
        `bool if_per` Prioritized Experience Replay for sparse reward
        """

    def select_action(self, state) -> np.ndarray:
        """Select actions for exploration

        :array state: state.shape==(state_dim, )
        :return array action: action.shape==(action_dim, ), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action = self.act(states)[0]
        return action.cpu().numpy()

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        """actor explores in env, then stores the env transition to ReplayBuffer

        :env: RL training environment. env.reset() env.step()
        :buffer: Experience Replay Buffer.
        :int target_step: explored target_step number of step in env
        :float reward_scale: scale reward, 'reward * reward_scale'
        :float gamma: discount factor, 'mask = 0.0 if done else gamma'
        :return int target_step: collected target_step number of step in env
        """
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """update the neural network by sampling batch data from ReplayBuffer

        replace by different DRL algorithms.
        return the objective value as training information to help fine-tuning

        `buffer` Experience replay buffer.
        :int target_step: explore target_step number of step in env
        `int batch_size` sample batch_size of data for Stochastic Gradient Descent
        :float repeat_times: the times of sample batch = int(target_step * repeat_times) in off-policy
        :return float obj_a: the objective value of actor
        :return float obj_c: the objective value of critic
        """

    def save_load_model(self, cwd, if_save):
        """save or load model files

        :str cwd: current working directory, we save model file here
        :bool if_save: save model or load model
        """
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if self.act is not None:
                torch.save(self.act.state_dict(), act_save_path)
            if self.cri is not None:
                torch.save(self.cri.state_dict(), cri_save_path)
        elif (self.act is not None) and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            print("Loaded act:", cwd)
        elif (self.cri is not None) and os.path.exists(cri_save_path):
            load_torch_file(self.cri, cri_save_path)
            print("Loaded cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update a target network via current network

        :nn.Module target_net: target network update via a current network, it is more stable
        :nn.Module current_net: current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))


class AgentPPO(AgentBase):
    def __init__(self):
        super().__init__()
        self.ratio_clip = 0.2  # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip),
        self.lambda_entropy = 0.01  # could be 0.01 ~ 0.05
        self.lambda_gae_adv = 0.97  # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.if_use_gae = True  # if use Generalized Advantage Estimation
        self.if_on_policy = True  # AgentPPO is an on policy DRL algorithm
        self.if_use_dn = False
        
        self.if_use_cri_target = None
        
        self.noise = None
        self.optimizer = None
        self.compute_reward = None  # attribution

    def init(self, net_dim, state_dim, action_dim, if_per=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv

        self.cri = CriticAdv(state_dim, net_dim, self.if_use_dn).to(self.device)
        self.act = ActorPPO(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri

        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])
        self.criterion = torch.nn.SmoothL1Loss()
        assert if_per is False  # on-policy don't need PER

    def select_action(self, state,buffer) -> tuple:
        """select action for PPO

        :array state: state.shape==(state_dim, )

        :return array action: state.shape==(action_dim, )
        :return array noise: noise.shape==(action_dim, ), the noise
        """
        state_chuli = buffer.guiyi.transform(state.reshape(1,-1))
        
        states = torch.as_tensor((state_chuli[0],), dtype=torch.float32, device=self.device).detach()
        actions, noises = self.act.get_action_noise(states)
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()  # todo remove detach()

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        buffer.empty_buffer_before_explore()  # NOTICE! necessary for on-policy
        # assert target_step == buffer.max_len - max_step

        actual_step = 0
        while actual_step < target_step:
            state = env.reset()
            for _ in range(env.max_step):
                
                action, noise = self.select_action(state,buffer)

                next_state, reward, done, _ = env.step(np.tanh(action))
                actual_step += 1

                other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
                buffer.append_buffer(state, other)
                if done:
                    break
                state = next_state
        return actual_step

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_all()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            #buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = -(buf_noise.pow(2).__mul__(0.5) + self.act.a_std_log + self.act.sqrt_2pi_log).sum(1)

            buf_r_sum, buf_advantage = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)
            del buf_reward, buf_mask, buf_noise

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        for _ in range(int(repeat_times * buf_len / batch_size)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob = self.act.compute_logprob(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)

            obj_united = obj_actor + obj_critic / (r_sum.std() + 1e-5)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau) if self.cri_target is not self.cri else None

        return self.act.a_std_log.mean().item(), obj_critic.item()

    def compute_reward_adv(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_sum = 0  # reward sum of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    def compute_reward_gae(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0  # reward sum of previous step
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + buf_mask[i] * pre_advantage - buf_value[i]
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv

        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage


class AgentSharedPPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.01  # could be 0.02
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(self, net_dim, state_dim, action_dim, if_per=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.act = SharedPPO(state_dim, action_dim, net_dim).to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.act.enc_s.parameters(), 'lr': self.learning_rate * 0.9},
            {'params': self.act.dec_a.parameters(), },
            {'params': self.act.a_std_log, },
            {'params': self.act.dec_q1.parameters(), },
            {'params': self.act.dec_q2.parameters(), },
        ], lr=self.learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()
        assert if_per is False  # on-policy don't need PER

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):  # old version
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using Generalized Advantage Estimation (GAE)'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_all()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = -(buf_noise.pow(2).__mul__(0.5) + self.act.a_std_log + self.act.sqrt_2pi_log).sum(1)

            buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
            buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

            pre_r_sum = 0  # reward sum of previous step
            pre_advantage = 0  # advantage value of previous step
            for i in range(buf_len - 1, -1, -1):
                buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
                pre_r_sum = buf_r_sum[i]

                buf_advantage[i] = buf_reward[i] + buf_mask[i] * pre_advantage - buf_value[i]
                pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv

            buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
            del buf_reward, buf_mask, buf_noise

        '''PPO: Clipped Surrogate objective of Trust Region'''
        for _ in range(int(repeat_times * buf_len / batch_size)):
            indices = torch.randint(buf_len, size=(batch_size,), device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            advantage = buf_advantage[indices]
            old_value = buf_r_sum[indices]
            old_logprob = buf_logprob[indices]

            new_logprob = self.act.compute_logprob(state, action)  # it is obj_actor
            ratio = (new_logprob - old_logprob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.clip, 1 + self.clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            new_value = self.cri(state).squeeze(1)
            obj_critic = self.criterion(new_value, old_value)
            self.obj_c = 0.995 * self.obj_c + 0.005 * obj_critic.item()  # for reliable_lambda
            reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda

            obj_united = obj_actor * reliable_lambda + obj_critic / (old_value.std() + 1e-5)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

        return self.act.a_std_log.mean().item(), self.obj_c


'''Utils'''


class OrnsteinUhlenbeckNoise:
    def __init__(self, size, theta=0.15, sigma=0.3, ou_noise=0.0, dt=1e-2):
        """The noise of Ornstein-Uhlenbeck Process

        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        Don't abuse OU Process. OU process has too much hyper-parameters and over fine-tuning make no sense.

        :int size: the size of noise, noise.shape==(-1, action_dim)
        :float theta: related to the not independent of OU-noise
        :float sigma: related to action noise std
        :float ou_noise: initialize OU-noise
        :float dt: derivative
        """
        self.theta = theta
        self.sigma = sigma
        self.ou_noise = ou_noise
        self.dt = dt
        self.size = size

    def __call__(self) -> float:
        """output a OU-noise

        :return array ou_noise: a noise generated by Ornstein-Uhlenbeck Process
        """
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise
