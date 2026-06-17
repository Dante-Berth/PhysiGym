import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import numpy as np
###########################
# Classes Neural Networks #
###########################


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0).sub(0.5)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.Mish()

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x + residual


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)
        self.activation = nn.Mish()

    def forward(self, x):
        x = self.activation(self.conv(x))
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class HadamaxBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # First block
        self.conv1a = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv1b = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.activation = nn.Mish()

    def forward(self, x):
        # First block
        x1 = self.activation(F.layer_norm(self.conv1a(x), self.conv1a(x).shape[1:]))
        x2 = self.activation(F.layer_norm(self.conv1b(x), self.conv1b(x).shape[1:]))
        x = self.pool(x1 * x2)
        return x


class GraphFeatureExtractor(nn.Module):
    def __init__(self, in_channels=-1, out_channels=32, heads=4, **kwargs):
        super().__init__()
        self.gat1 = GATConv(in_channels=in_channels, out_channels=4, heads=heads)
        self.gat2 = GATConv(4 * heads, out_channels, heads=1)
        self.activation = nn.Mish()

    def forward(self, data):
        x = self.activation(self.gat1(data.x, data.edge_index, data.edge_attr))
        x = self.activation(self.gat2(x, data.edge_index, data.edge_attr))
        return global_mean_pool(x, data.batch)

class RelativeBias(nn.Module):
    def __init__(self, heads, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, heads)
        )

    def forward(self, xy):  # (B, N, 2)
        delta = xy[:, :, None, :] - xy[:, None, :, :]  # (B, N, N, 2)
        return self.mlp(delta)  # (B, N, N, H)


# -----------------------------
# Fast attention block
# -----------------------------
class FastAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None, bias=None):
        B, N, D = x.shape
        H = self.heads

        qkv = self.qkv(x).view(B, N, 3, H, D // H)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if bias is not None:
            attn = attn + bias.permute(0, 3, 1, 2)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask[:, None, None, :] == 0, -1e9)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)

        return self.proj(out)


# -----------------------------
# Encoder block
# -----------------------------
class FastBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2):
        super().__init__()
        self.attn = FastAttention(dim, heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.ReLU(), nn.Linear(dim * mlp_ratio, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None, bias=None):
        x = x + self.attn(self.norm1(x), attn_mask, bias)
        x = x + self.ff(self.norm2(x))
        return x


# -----------------------------
# Full encoder
# -----------------------------
class FastSetEncoder(nn.Module):
    def __init__(
        self, input_dim, dim=64, depth=2, heads=4, use_relative_bias=True
    ):
        super().__init__()

        self.embed = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList([FastBlock(dim, heads) for _ in range(depth)])

        self.rel_bias = RelativeBias(heads) if use_relative_bias else None
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: (B, N, input_dim)
           zero rows = padding
        """

        # padding mask: 1 = valid, 0 = padding
        mask = (x.abs().sum(dim=-1) > 0).float()

        xy = x[..., :2]  # raw x,y

        x = self.embed(x)

        bias = self.rel_bias(xy) if self.rel_bias else None

        for block in self.blocks:
            x = block(x, mask, bias)

        x = self.norm(x)

        # mean pooling over valid nodes
        x = (x * mask[..., None]).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6)

        return x


class FeatureCNN(nn.Module):
    """
    Lightweight CNN over feature dimension (NOT nodes).
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
        )
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: (B, N, F)
        B, N, F = x.shape
        x = x.view(B * N, 1, F)
        x = self.net(x)
        x = x.view(B, N, F)
        return self.proj(x)


class RLTransformerBlock(nn.Module):
    def __init__(self, dim, heads=2, mlp_ratio=2):
        super().__init__()
        self.attn = FastAttention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, mask=None, bias=None):
        x = x + self.attn(self.norm1(x), mask, bias)
        x = x + self.ff(self.norm2(x))
        return x

class RLSetEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        token_dim=32,
        depth=1,
        heads=2,
        use_relative_bias=True,
    ):
        super().__init__()

        self.pre = FeatureCNN(input_dim, token_dim)
        self.blocks = nn.ModuleList(
            [RLTransformerBlock(token_dim, heads) for _ in range(depth)]
        )

        self.rel_bias = RelativeBias(heads) if use_relative_bias else None
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x):
        """
        x: (B, N, F), zero rows = padding
        """
        mask = (x.abs().sum(dim=-1) > 0).float()

        xy = x[..., :2] if x.size(-1) >= 2 else None

        x = self.pre(x)
        bias = self.rel_bias(xy) if self.rel_bias else None

        for block in self.blocks:
            x = block(x, mask, bias)

        x = self.norm(x)

        # masked mean pooling
        x = (x * mask[..., None]).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6)

        return x


class FeatureExtractor(nn.Module):
    """Handles both image-based and vector-based state inputs dynamically."""

    def __init__(self, cfg, neural_architecture_image="impala", **kwargs):
        super().__init__()

        self.is_graph = cfg["is_graph"]
        self.is_image = False
        self.is_tranformer_node = True if "transformer" in cfg["observation_mode"] else False

        obs_shape = cfg["observation_space_shape"] if not self.is_graph else None

        if self.is_graph:
            # Assume node features have fixed dimension
            node_feature_dim = cfg["node_feature_dim"]
            self.feature_extractor = GraphFeatureExtractor(
                node_feature_dim=node_feature_dim
            )
            self.feature_size = 128
        elif self.is_tranformer_node:
            if neural_architecture_image == "transformer_encoder":
                self.feature_extractor = FastSetEncoder(input_dim=obs_shape[-1], dim=64)
            else:
                self.feature_extractor = RLSetEncoder(input_dim=obs_shape[-1])

            self.feature_size = self._get_feature_size(obs_shape)
        else:
            self.is_image = len(obs_shape) == 3  # (C, H, W)
            if self.is_image:
                if neural_architecture_image == "impala":
                    layers = [
                        PixelPreprocess(),
                        ImpalaBlock(obs_shape[0], 16),
                        ImpalaBlock(16, 32),
                        ImpalaBlock(32, 32),
                        nn.Flatten(),
                    ]
                elif neural_architecture_image == "hadamax":
                    layers = [
                        PixelPreprocess(),
                        HadamaxBlock(obs_shape[0], 16),
                        HadamaxBlock(16, 32),
                        HadamaxBlock(32, 32),
                        nn.Flatten(),
                    ]
                else:
                    raise ValueError(
                        f"Error: unknown neural architecture: {neural_architecture_image}"
                    )

                self.feature_extractor = nn.Sequential(*layers)
                self.feature_size = self._get_feature_size(obs_shape)
            else:
                self.feature_extractor = nn.Identity()
                self.feature_size = int(np.prod(obs_shape))

    def _get_feature_size(self, obs_shape):
        """Pass a dummy tensor through CNN to compute feature size dynamically."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            out = self.feature_extractor(dummy_input)
            return int(np.prod(out.shape[1:]))

    def forward(self, x):
        if self.is_image or self.is_tranformer_node:
            x = self.feature_extractor(x)  # Apply CNN
            x = x.view(x.size(0), -1)  # Flatten
        elif self.is_graph:
            x = self.feature_extractor(x)
        return x


class QNetwork(nn.Module):
    """Critic network (Q-function)"""

    def __init__(self, cfg, neural_architecture_image, **kwargs):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            cfg, neural_architecture_image, **kwargs
        )

        self.fc1 = nn.LazyLinear(256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.LazyLinear(256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.LazyLinear(1)
        self.mish = nn.Mish()

    def forward(self, x, a):
        x = self.feature_extractor(x)
        x = torch.cat([x, a], dim=1)

        x = self.mish(self.ln1(self.fc1(x)))
        x = self.mish(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    """Policy network (Actor)"""

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(self, cfg, neural_architecture_image, **kwargs):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            cfg, neural_architecture_image, **kwargs
        )
        action_dim = np.prod(cfg["action_space_shape"])

        # Fully connected layers
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.LazyLinear(256)
        self.fc_mean = nn.LazyLinear(action_dim)
        self.fc_logstd = nn.LazyLinear(action_dim)
        self.relu = nn.ReLU()
        # Action scaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (cfg["action_space_high"] - cfg["action_space_low"]) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (cfg["action_space_high"] + cfg["action_space_low"]) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # Extract features

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )  # Stable variance scaling

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class Encoder(nn.Module):
    """Handles both image-based and vector-based state inputs dynamically."""

    def __init__(self, cfg, out_channels=32):
        super().__init__()

        obs_shape = cfg["observation_space_shape"]
        self.out_channels = out_channels
        self.is_image = len(obs_shape) == 3  # (C, H, W)
        if self.is_image:
            layers = [
                PixelPreprocess(),
                ImpalaBlock(obs_shape[0], 16),
                ImpalaBlock(16, 32),
                ImpalaBlock(32, self.out_channels),
                nn.Flatten(),
            ]

            self.feature_extractor = nn.Sequential(*layers)
            self.feature_size = self._get_feature_size(obs_shape)
        else:
            # simple vector
            self.feature_extractor = self.fc = nn.Sequential(
                nn.Linear(obs_shape[0], out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
            )
            self.feature_size = out_channels

    def _get_feature_size(self, obs_shape):
        """Pass a dummy tensor through CNN to compute feature size dynamically."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            out = self.feature_extractor(dummy_input)
            return int(np.prod(out.shape[1:]))

    def forward(self, x):
        x = self.feature_extractor(x)
        return x