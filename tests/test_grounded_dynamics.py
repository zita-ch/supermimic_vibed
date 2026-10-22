"""Behavioral checks for grounded dynamics; requires only PyTorch, no simulator."""

import os
import unittest

import torch

from supermimic.buffer import ReplayBuffer
from supermimic.networks import WorldModel
from supermimic.rotations import so3_exp
from supermimic.state import INV_DIM, NUM_J, XI_DIM, flatten_state, unflatten_state

DEVICE = os.environ.get("SM_TEST_DEVICE", "cpu")


def states(batch=4, window=3):
    shape = (batch, window)
    p = torch.randn(*shape, 3, device=DEVICE) * 0.1
    p[..., 2] += 0.8
    R = so3_exp(torch.randn(*shape, 3, device=DEVICE) * 0.2)
    return (p, R, *(torch.randn(*shape, d, device=DEVICE) * 0.1 for d in (3, 3, NUM_J, NUM_J)))


class GroundedDynamicsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.S, self.S1 = states(), states()
        self.a = torch.randn(4, 3, NUM_J, device=DEVICE)
        self.xi = torch.randn(4, 3, XI_DIM, device=DEVICE)
        self.mask = torch.ones(4, 3, device=DEVICE)
        self.wm = WorldModel(hidden=(32, 32), latent_dim=16).to(DEVICE)
        self.wm.in_norm.update(self.wm.inputs(self.S, self.a, self.xi))

    def losses(self, S1=None, mask=None):
        return self.wm.grounding_losses(
            self.S, self.S1 if S1 is None else S1, self.a, self.xi,
            self.mask if mask is None else mask,
        )

    def test_original_checkpoint_layout_and_reload(self):
        original = WorldModel(hidden=(32, 32)).to(DEVICE)
        keys = set(original.state_dict())
        self.assertFalse(any(k.startswith(("encoder.", "state_head.", "idm_head.")) for k in keys))
        restored = WorldModel(hidden=(32, 32), latent_dim=0).to(DEVICE)
        restored.load_state_dict(original.state_dict(), strict=True)
        for actual, expected in zip(restored.step(self.S, self.a, self.xi), original.step(self.S, self.a, self.xi)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "latent_dim"):
            original.grounding_losses(self.S, self.S1, self.a, self.xi, self.mask)

    def test_grounded_checkpoint_reload(self):
        restored = WorldModel(hidden=(32, 32), latent_dim=16).to(DEVICE)
        restored.load_state_dict(self.wm.state_dict())
        actual = restored.grounding_losses(self.S, self.S1, self.a, self.xi, self.mask)
        for a, b in zip(actual, self.losses()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_each_auxiliary_loss_trains_shared_encoder(self):
        for which in (0, 1):
            self.wm.zero_grad(set_to_none=True)
            self.losses()[which].backward()
            grads = [p.grad for p in self.wm.encoder.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
            # The auxiliary losses must not see the action-conditioned predictor.
            self.assertTrue(all(p.grad is None for p in self.wm.net.parameters()))

    def test_encoder_never_reads_actions_or_context(self):
        inputs = []
        handle = self.wm.encoder.register_forward_pre_hook(lambda _module, args: inputs.append(args[0].detach().clone()))
        self.wm.step(self.S, self.a, self.xi)
        self.wm.step(self.S, self.a + 100, self.xi - 100)
        handle.remove()
        self.assertEqual(inputs[0].shape[-1], INV_DIM)
        torch.testing.assert_close(inputs[0], inputs[1], rtol=0, atol=0)

    def test_post_reset_targets_are_excluded(self):
        mask = self.mask.clone()
        mask[:, 2] = 0
        changed = tuple(x.clone() for x in self.S1)
        for i in (0, 2, 3, 4, 5):
            changed[i][:, 2] += 100
        changed[1][:, 2] = so3_exp(torch.ones(4, 3, device=DEVICE))
        for before, after in zip(self.losses(mask=mask), self.losses(S1=changed, mask=mask)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        for loss in self.losses(mask=torch.zeros_like(mask)):
            self.assertEqual(loss.item(), 0)
        self.wm.zero_grad(set_to_none=True)
        sum(self.losses(mask=torch.zeros_like(mask))).backward()
        self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in self.wm.parameters()))

    def test_freezing_model_keeps_multistep_action_gradient(self):
        for latent_dim in (0, 16):
            wm = WorldModel(hidden=(32, 32), latent_dim=latent_dim).to(DEVICE)
            torch.nn.init.normal_(wm.net[-1].weight, std=0.01)
            wm.requires_grad_(False)
            actions = self.a.detach().clone().requires_grad_()
            S = tuple(x[:, 0] for x in self.S)
            for t in range(3):
                S = wm.step(S, actions[:, t], self.xi[:, t])
            S[4].square().mean().backward()
            self.assertTrue(torch.isfinite(actions.grad).all())
            self.assertTrue((actions.grad.abs().sum((0, 2)) > 0).all())
            self.assertTrue(all(p.grad is None for p in wm.parameters()))

    def test_grounding_is_heading_and_xy_translation_invariant(self):
        yaw = so3_exp(torch.tensor([0., 0., 0.7], device=DEVICE))
        shift = torch.tensor([2., -3., 0.], device=DEVICE)
        def transform(S):
            p, R, v, w, q, qd = S
            rotate = lambda x: (yaw @ x.unsqueeze(-1)).squeeze(-1)
            return rotate(p) + shift, yaw @ R, rotate(v), rotate(w), q, qd
        actual = self.wm.grounding_losses(transform(self.S), transform(self.S1), self.a, self.xi, self.mask)
        for a, b in zip(actual, self.losses()):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)

    def test_state_roundtrip(self):
        for actual, expected in zip(unflatten_state(flatten_state(self.S)), self.S):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    @unittest.skipUnless(os.environ.get("SM_TEST_COMPILE") == "1", "opt-in compile check")
    def test_compiled_step_and_grounding_match_eager(self):
        step = torch.compile(self.wm.step, fullgraph=True)
        losses = torch.compile(self.wm.grounding_losses, fullgraph=True)
        for actual, expected in zip(step(self.S, self.a, self.xi), self.wm.step(self.S, self.a, self.xi)):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        actual = losses(self.S, self.S1, self.a, self.xi, self.mask)
        for a, b in zip(actual, self.losses()):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        sum(actual).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.wm.encoder.parameters()))


class ReplayTests(unittest.TestCase):
    def test_terminal_successor_and_mask_survive_ring_wrap(self):
        replay = ReplayBuffer(T=3, N=1, device=DEVICE)
        for t in range(5):
            tr = {k: torch.zeros_like(v[0]) for k, v in replay.data.items()}
            tr["s"].fill_(t)
            tr["s1"].fill_(100 + t)  # deliberately different from next row's state
            tr["done"].fill_(t == 3)
            replay.add(tr)
        w = replay.sample_windows(B=4, k=3)
        torch.testing.assert_close(w["s"][0, :, 0], torch.tensor([2., 3., 4.], device=DEVICE))
        torch.testing.assert_close(w["s1"][0, :, 0], torch.tensor([102., 103., 104.], device=DEVICE))
        torch.testing.assert_close(w["mask"], torch.tensor([[1., 1., 0.]], device=DEVICE).expand(4, -1))

    def test_empty_replay_fails_with_actionable_error(self):
        replay = ReplayBuffer(T=8, N=1, device=DEVICE)
        with self.assertRaisesRegex(ValueError, "collect transitions"):
            replay.sample_windows(2, 8)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            replay.sample_starts(2, 10)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
