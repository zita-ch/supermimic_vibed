"""World model, policy (motion encoder + shared decoder), and critic (paper Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .mdp import INV_DIM, NUM_J, XI_DIM, invariant_features
from .motion import CONTROL_DT
from .rotations import so3_exp


def mlp(inp: int, hidden: list[int], out: int) -> nn.Sequential:
    """``inp -> (Linear -> LayerNorm -> GELU)* -> Linear(out)``, as the official code uses for every net."""
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


WM_OUT = 3 + 3 + NUM_J  # [dv_b, dw_b, dqd]: acceleration-level increments only

# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


# World-model loss: Bryson's rule, each channel group weighted by 1 / nominal_error^2 (official values).
# Root velocities are supervised too: the tracking reward depends on body velocities.
NOMINAL_ERROR = {"root_pos": 0.01, "root_ori": 0.02, "root_lin_vel": 0.1, "root_ang_vel": 0.3, "joint_pos": 0.1, "joint_vel": 3.14}


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi).

    The network predicts only acceleration-level increments in the current root frame
    (root linear/angular velocity and joint velocity); positions and the root rotation are
    integrated analytically and semi-implicitly, so every prediction is kinematically
    consistent. Pose updates weight the increment by ``(d + 1) / (2 d)`` for ``d`` physics
    substeps of constant acceleration, as in the official implementation.
    """

    def __init__(self, hidden=(2048, 2048, 2048), dt: float = CONTROL_DT, substeps: int = 4):
        super().__init__()
        self.in_dim = INV_DIM + NUM_J + XI_DIM
        self.net = mlp(self.in_dim, list(hidden), WM_OUT)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)
        self.dt = dt
        self.c = (substeps + 1) / (2 * substeps)

    def inputs(self, S, a, xi_n):
        return torch.cat((invariant_features(S), a, xi_n), -1)

    @staticmethod
    def targets(S, S1):
        """Acceleration-level increments from S to S1 in the root frame of S (the network output space)."""
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        Rt = R.transpose(-1, -2)
        return torch.cat(((Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1), (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1), qd1 - qd), -1)

    def step(self, S, a, xi_n):
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n)))
        d = y * self.out_norm.std() + self.out_norm.mean
        dv_b, dw_b, dqd = d.split((3, 3, NUM_J), -1)
        dv = (R @ dv_b.unsqueeze(-1)).squeeze(-1)
        dw = (R @ dw_b.unsqueeze(-1)).squeeze(-1)
        c, dt = self.c, self.dt
        p1 = p + (v + c * dv) * dt
        R1 = so3_exp((w + c * dw) * dt) @ R  # world-frame angular velocity
        return (p1, R1, v + dv, w + dw, q + (qd + c * dqd) * dt, qd + dqd)

    @staticmethod
    def weighted_sq_error(S_hat, S):
        """Per-channel squared errors weighted by Bryson's rule (1 / nominal^2): root position, root rotation
        (``sqrt(1/2) * R`` so the squared difference is ~theta^2), root velocities, joint positions and velocities."""
        ph, Rh = S_hat[0], S_hat[1]
        p, R = S[0], S[1]
        e = (
            ((ph - p) / NOMINAL_ERROR["root_pos"]) ** 2,
            (((Rh - R).flatten(-2) * 0.5**0.5) / NOMINAL_ERROR["root_ori"]) ** 2,
            ((S_hat[2] - S[2]) / NOMINAL_ERROR["root_lin_vel"]) ** 2,
            ((S_hat[3] - S[3]) / NOMINAL_ERROR["root_ang_vel"]) ** 2,
            ((S_hat[4] - S[4]) / NOMINAL_ERROR["joint_pos"]) ** 2,
            ((S_hat[5] - S[5]) / NOMINAL_ERROR["joint_vel"]) ** 2,
        )
        return torch.cat(e, -1)


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh."""

    def __init__(self, obs_dim: int, goal_dim: int, hist: int = 10, z_dim: int = 32):
        super().__init__()
        self.hist = hist
        self.enc = mlp(goal_dim, [2048, 1024, 512, 256], z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, [2048, 1024, 512, 256], NUM_J)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last."""
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        o = self.obs_norm(obs_hist).flatten(-2)
        return run_mlp(self.dec, torch.cat((z, o), -1))


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
