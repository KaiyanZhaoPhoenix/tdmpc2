import torch
import torch.nn as nn
import torch.nn.functional as F
from common.layers import mlp


class Actor(nn.Module):
        def __init__(self, obs_dim, action_dim, hidden_dim=256):
                super().__init__()
                self.net = mlp(obs_dim, [hidden_dim, hidden_dim], action_dim * 2)
                self.log_std_min = -10
                self.log_std_max = 2

        def forward(self, obs):
                mu, log_std = self.net(obs).chunk(2, dim=-1)
                log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
                std = log_std.exp()
                return mu, std

        def sample(self, obs, eval_mode=False):
                mu, std = self(obs)
                dist = torch.distributions.Normal(mu, std)
                if eval_mode:
                        action = torch.tanh(mu)
                else:
                        action = torch.tanh(dist.rsample())
                log_prob = dist.log_prob(action).sum(-1, keepdim=True)
                return action, log_prob


class Critic(nn.Module):
        def __init__(self, obs_dim, action_dim, hidden_dim=256):
                super().__init__()
                self.q1 = mlp(obs_dim + action_dim, [hidden_dim, hidden_dim], 1)
                self.q2 = mlp(obs_dim + action_dim, [hidden_dim, hidden_dim], 1)

        def forward(self, obs, action):
                x = torch.cat([obs, action], dim=-1)
                q1 = self.q1(x)
                q2 = self.q2(x)
                return q1, q2


class SACAgent:
        def __init__(self, cfg):
                self.cfg = cfg
                obs_dim = cfg.obs_shape[cfg.get('obs', 'state')][0]
                action_dim = cfg.action_dim
                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                self.actor = Actor(obs_dim, action_dim).to(self.device)
                self.critic = Critic(obs_dim, action_dim).to(self.device)
                self.target_critic = Critic(obs_dim, action_dim).to(self.device)
                self.target_critic.load_state_dict(self.critic.state_dict())

                lr = cfg.lr
                self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
                self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

                self.gamma = cfg.get('sac_gamma', 0.99)
                self.tau = cfg.get('sac_tau', 0.005)
                self.alpha = cfg.get('sac_alpha', 0.2)
                self.batch_size = cfg.get('batch_size', 256)

        def act(self, obs, eval_mode=False, **_):
                obs = obs.to(self.device).unsqueeze(0)
                with torch.no_grad():
                        action, _ = self.actor.sample(obs, eval_mode)
                return action[0].cpu()

        def update(self, buffer):
                obs, action, reward, next_obs, done = buffer.sample(self.batch_size)
                obs = obs.to(self.device)
                action = action.to(self.device)
                reward = reward.to(self.device)
                next_obs = next_obs.to(self.device)
                done = done.to(self.device)

                with torch.no_grad():
                        next_action, next_log_prob = self.actor.sample(next_obs)
                        target_q1, target_q2 = self.target_critic(next_obs, next_action)
                        target_v = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
                        target_q = reward + (1 - done) * self.gamma * target_v

                q1, q2 = self.critic(obs, action)
                critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                self.critic_opt.step()

                # Actor update
                new_action, log_prob = self.actor.sample(obs)
                q1_pi, q2_pi = self.critic(obs, new_action)
                actor_loss = (self.alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_opt.step()

                # Update target networks
                with torch.no_grad():
                        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                                tp.data.mul_(1 - self.tau)
                                tp.data.add_(self.tau * p.data)

                return dict(actor_loss=actor_loss.item(), critic_loss=critic_loss.item())
