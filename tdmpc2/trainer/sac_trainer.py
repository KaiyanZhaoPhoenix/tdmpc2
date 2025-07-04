import numpy as np
import torch
from time import time

from trainer.base import Trainer
from common.replay_buffer import ReplayBuffer


class SACTrainer(Trainer):
        """Trainer for the SAC agent."""

        def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                obs_dim = self.env.observation_space.shape[0]
                action_dim = self.env.action_space.shape[0]
                self.buffer = ReplayBuffer(obs_dim, action_dim, self.cfg.buffer_size)
                self._step = 0
                self._ep_idx = 0
                self._start_time = time()

        def common_metrics(self):
                elapsed_time = time() - self._start_time
                return dict(step=self._step, episode=self._ep_idx, elapsed_time=elapsed_time, steps_per_second=self._step / elapsed_time)

        def eval(self):
                ep_rewards, ep_lengths = [], []
                for i in range(self.cfg.eval_episodes):
                        obs, done, ep_reward, t = self.env.reset(), False, 0, 0
                        if self.cfg.save_video:
                                self.logger.video.init(self.env, enabled=(i == 0))
                        while not done:
                                action = self.agent.act(obs, eval_mode=True)
                                obs, reward, done, info = self.env.step(action)
                                ep_reward += reward
                                t += 1
                                if self.cfg.save_video:
                                        self.logger.video.record(self.env)
                        ep_rewards.append(ep_reward)
                        ep_lengths.append(t)
                        if self.cfg.save_video:
                                self.logger.video.save(self._step)
                return dict(episode_reward=np.nanmean(ep_rewards), episode_length=np.nanmean(ep_lengths))

        def train(self):
                train_metrics, done = {}, True
                obs = self.env.reset()
                while self._step <= self.cfg.steps:
                        if self._step % self.cfg.eval_freq == 0:
                                eval_metrics = self.eval()
                                eval_metrics.update(self.common_metrics())
                                self.logger.log(eval_metrics, 'eval')

                        if done:
                                if self._step > 0:
                                        train_metrics.update(self.common_metrics())
                                        self.logger.log(train_metrics, 'train')
                                obs = self.env.reset()
                                self._ep_idx += 1

                        if self._step < self.cfg.seed_steps:
                                action = self.env.rand_act()
                        else:
                                action = self.agent.act(obs)
                        next_obs, reward, done, info = self.env.step(action)
                        self.buffer.add(obs, action, torch.tensor([reward]), next_obs, torch.tensor([done], dtype=torch.float32))
                        obs = next_obs

                        if self._step >= self.cfg.seed_steps:
                                train_metrics = self.agent.update(self.buffer)

                        self._step += 1

                self.logger.finish(self.agent)
