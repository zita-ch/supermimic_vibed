"""World-model integration: acceleration-level increments integrate to a consistent next state."""

import unittest

import torch

from supermimic.networks import WorldModel
from supermimic.rotations import so3_exp


def random_state(B, g):
    p = torch.randn(B, 3, generator=g)
    R = so3_exp(torch.randn(B, 3, generator=g))
    return (p, R, torch.randn(B, 3, generator=g), torch.randn(B, 3, generator=g),
            torch.randn(B, 29, generator=g), torch.randn(B, 29, generator=g))  # fmt: skip


class WorldModelTest(unittest.TestCase):
    def setUp(self):
        self.g = torch.Generator().manual_seed(0)
        self.wm = WorldModel(hidden=(16,))  # zero-initialized output layer -> prediction = out_norm.mean

    def predict(self, S, d):
        """Make the network output exactly the increments ``d`` (B, 35)."""
        self.wm.out_norm.mean.copy_(d[0])
        a = torch.zeros(S[0].shape[0], 29)
        return self.wm.step(S, a, torch.zeros(S[0].shape[0], 33))

    def test_semi_implicit_integration(self):
        S = random_state(1, self.g)
        p, R, v, w, q, qd = S
        d = torch.randn(1, 35, generator=self.g)
        p1, R1, v1, w1, q1, qd1 = self.predict(S, d)
        dv, dw, dqd = R @ d[:, 0:3, None], R @ d[:, 3:6, None], d[:, 6:]
        c, dt = self.wm.c, self.wm.dt
        self.assertAlmostEqual(c, 5 / 8)
        torch.testing.assert_close(v1, v + dv[..., 0])
        torch.testing.assert_close(w1, w + dw[..., 0])
        torch.testing.assert_close(qd1, qd + dqd)
        torch.testing.assert_close(q1, q + (qd + c * dqd) * dt)
        torch.testing.assert_close(p1, p + (v + c * dv[..., 0]) * dt)
        torch.testing.assert_close(R1, so3_exp((w + c * dw[..., 0]) * dt) @ R)
        torch.testing.assert_close(R1 @ R1.transpose(-1, -2), torch.eye(3).expand(1, 3, 3), atol=1e-5, rtol=0)

    def test_targets_invert_velocity_update(self):
        S = random_state(1, self.g)
        d = torch.randn(1, 35, generator=self.g)
        S1 = self.predict(S, d)
        torch.testing.assert_close(WorldModel.targets(S, S1), d, atol=1e-5, rtol=1e-5)

    def test_loss_is_zero_only_at_the_truth(self):
        S = random_state(4, self.g)
        self.assertEqual(WorldModel.weighted_sq_error(S, S).abs().max().item(), 0.0)
        S2 = (S[0] + 0.01, *S[1:])
        e = WorldModel.weighted_sq_error(S2, S)
        torch.testing.assert_close(e[:, :3], torch.ones(4, 3))  # 1 cm root error == one nominal error


if __name__ == "__main__":
    unittest.main()
