from typing import Tuple

import torch
import torch.nn as nn


def build_feature_mlp(
    input_dim: int,
    hidden_dims: Tuple[int, ...],
    activation: nn.Module = nn.SiLU,
    dropout_p: float = 0.0,
) -> Tuple[nn.Sequential, int]:
    # Feedforward refinement after attention / 注意力后的前馈细化层
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
    # Transformer-based policy head / 基于 Transformer 的策略头
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Tuple[int, ...],
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Input projection / 输入嵌入层
        self.input_proj = nn.Linear(obs_dim, embed_dim)
        # Positional encoding / 位置编码
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        # Transformer encoder / Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.feature_net, feature_dim = build_feature_mlp(embed_dim, hidden_dims)
        head_input_dim = feature_dim
        # Policy heads / 策略输出头
        self.mean_head = nn.Linear(head_input_dim, act_dim)
        self.log_std_head = nn.Linear(head_input_dim, act_dim)
        self.log_std_min = -5.0
        self.log_std_max = 2.0

    def forward(self, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sequence check / 检查输入维度
        if obs_seq.dim() != 3:
            raise ValueError("obs_seq must be of shape (batch, seq_len, obs_dim)")
        # Encode sequence / 序列编码
        x = self.input_proj(obs_seq)
        pos = self.pos_embedding[:, : x.size(1)]
        x = x + pos
        features = self.encoder(x)
        pooled = features[:, -1, :]
        # Head / 输出均值
        pooled = self.output_norm(pooled)
        projected = self.feature_net(pooled)
        mean = self.mean_head(projected)
        # Clamp log-variance to keep exploration bounded / 裁剪对数方差以限定探索边界
        log_std = torch.clamp(self.log_std_head(projected), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs_seq: torch.Tensor):
        # Sample with Tanh to stay within joint limits / 使用 Tanh 保持在关节限制内
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
    # Transformer-based value head / 基于 Transformer 的价值头
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Tuple[int, ...],
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Shared embedding / 共享嵌入
        self.input_proj = nn.Linear(obs_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.feature_net, feature_dim = build_feature_mlp(embed_dim, hidden_dims)
        head_input_dim = feature_dim
        # Value head / 价值头
        self.value_head = nn.Linear(head_input_dim, 1)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        # Encode then regress value / 编码后回归价值
        if obs_seq.dim() != 3:
            raise ValueError("obs_seq must be of shape (batch, seq_len, obs_dim)")
        x = self.input_proj(obs_seq)
        pos = self.pos_embedding[:, : x.size(1)]
        x = x + pos
        features = self.encoder(x)
        pooled = features[:, -1, :]
        pooled = self.output_norm(pooled)
        projected = self.feature_net(pooled)
        return self.value_head(projected)
