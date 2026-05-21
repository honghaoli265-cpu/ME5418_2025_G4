from typing import Tuple

import torch
import torch.nn as nn


def build_feature_mlp(
    input_dim: int,
    hidden_dims: Tuple[int, ...],
    activation: nn.Module = nn.SiLU,
    dropout_p: float = 0,
) -> Tuple[nn.Sequential, int]:
    # Stack linear blocks to smooth the temporal embedding / 线性块堆叠以平滑时间编码
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation())
        if dropout_p > 0.0:
            layers.append(nn.Dropout(dropout_p))
        last_dim = hidden_dim
    return nn.Sequential(*layers), last_dim


class PolicyNetwork(nn.Module):
    # LSTM-based policy head / 基于 LSTM 的策略头
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Tuple[int, ...],
        lstm_hidden_size: int,
    ):
        super().__init__()
        # Recurrent encoder / 循环编码器
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=lstm_hidden_size, batch_first=True)
        self.core_norm = nn.LayerNorm(lstm_hidden_size)
        self.feature_net, feature_dim = build_feature_mlp(lstm_hidden_size, hidden_dims)
        head_input_dim = feature_dim
        # Policy heads / 策略头
        self.mean_head = nn.Linear(head_input_dim, act_dim)
        self.log_std_head = nn.Linear(head_input_dim, act_dim)
        self.log_std_min = -5.0
        self.log_std_max = 2.0

    def forward(self, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode sequence / 编码观测序列
        _, (h_n, _) = self.lstm(obs_seq)
        features = h_n[-1]
        features = self.core_norm(features)
        projected = self.feature_net(features)
        # Mean action / 均值动作
        mean = self.mean_head(projected)
        # Clamp log-variance for stable exploration / 裁剪对数方差以保持探索稳定
        log_std = torch.clamp(self.log_std_head(projected), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs_seq: torch.Tensor):
        # Sample with Tanh-squash to keep actions bounded / 使用 Tanh 压缩以保证动作有界
        mean, log_std = self.forward(obs_seq)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        noise = normal.rsample()
        action = torch.tanh(noise)
        log_prob = (
            normal.log_prob(noise) - torch.log(1 - action.pow(2) + 1e-7)
        ).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return action, log_prob, entropy

    def evaluate(self, obs_seq: torch.Tensor, actions: torch.Tensor):
        # Evaluate tanh-squashed Gaussian log-prob / 评估 Tanh 压缩高斯的对数概率
        mean, log_std = self.forward(obs_seq)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        unsquashed = torch.atanh(torch.clamp(actions, -0.999999, 0.999999))
        log_prob = (
            normal.log_prob(unsquashed) - torch.log(1 - actions.pow(2) + 1e-7)
        ).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, log_std


class ValueNetwork(nn.Module):
    # LSTM-based value head / 基于 LSTM 的价值头
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Tuple[int, ...],
        lstm_hidden_size: int,
    ):
        super().__init__()
        # Shared encoder / 价值网络序列编码器
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=lstm_hidden_size, batch_first=True)
        self.core_norm = nn.LayerNorm(lstm_hidden_size)
        self.feature_net, feature_dim = build_feature_mlp(lstm_hidden_size, hidden_dims)
        head_input_dim = feature_dim
        # Scalar value head / 标量价值头
        self.value_head = nn.Linear(head_input_dim, 1)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        # Encode then regress value / 编码后回归状态价值
        _, (h_n, _) = self.lstm(obs_seq)
        features = h_n[-1]
        features = self.core_norm(features)
        projected = self.feature_net(features)
        return self.value_head(projected)
