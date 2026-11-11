"""Regression tests for the strided goal window and held-pose padding at motion clip boundaries."""

import unittest

import torch

from supermimic.motion import ACTOR_GOAL_DIM, GOAL_FRAMES, GOAL_STRIDE, WM_GOAL_DIM, MotionLib

PER_FRAME = 29 + 29 + 3 + 3 + 3 + 6  # q, 0.1*qd, ref displacement, ref v, ref w, rel ori
VEL = (slice(29, 58), slice(61, 67))  # 0.1*qd and ref root v, w within one frame


def yaw(a):
    c, s, z, o = a.cos(), a.sin(), torch.zeros_like(a), torch.ones_like(a)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(*a.shape, 3, 3)


class MotionGoalTest(unittest.TestCase):
    def setUp(self):
        self.lib = lib = MotionLib.__new__(MotionLib)
        n0, n1 = 30, 40
        lib.clip_start = torch.tensor([0, n0])
        lib.clip_end = torch.tensor([n0 - 1, n0 + n1 - 1])
        lib.clip_id = torch.tensor([0] * n0 + [1] * n1)
        t = torch.arange(n0 + n1, dtype=torch.float32)
        lib.root_R = yaw(t * 0.05)
        lib.root_p = torch.stack((t * 0.02, (t * 0.01).square(), 0.7 + 0.01 * t), -1)
        lib.root_v = torch.stack((t + 1, t + 2, t + 3), -1) * 0.01
        lib.root_w = lib.root_v * 0.2
        lib.root_v_local = (lib.root_R.transpose(-1, -2) @ lib.root_v[..., None])[..., 0]
        lib.root_w_local = (lib.root_R.transpose(-1, -2) @ lib.root_w[..., None])[..., 0]
        lib.q = t[:, None] * 0.01 + torch.arange(29, dtype=torch.float32)[None, :] * 0.01
        lib.qd = lib.q + 2
        # robot root pose, deliberately off the reference
        self.p = torch.tensor([[0.3, -0.2, 0.75], [1.0, 0.5, 0.6], [0.0, 0.0, 0.8]])
        self.R = yaw(torch.tensor([0.4, -1.0, 2.0]))

    def expected(self, frames, p, R):
        lib = self.lib
        actors, wms = [], []
        for b, frame in enumerate(frames.tolist()):
            end = lib.clip_end[lib.clip_id[frame]].item()
            Ht = lib.root_R[frame].T  # fixture rotations are yaw only
            feats, rel_pos = [], []
            for i in range(GOAL_FRAMES):
                req = frame + 1 + i * GOAL_STRIDE
                fut, rec = min(req, end), req <= end
                feats.append(torch.cat((
                    lib.q[fut],
                    lib.qd[fut] * 0.1 if rec else torch.zeros(29),
                    Ht @ (lib.root_p[fut] - lib.root_p[frame]),
                    lib.root_v_local[fut] if rec else torch.zeros(3),
                    lib.root_w_local[fut] if rec else torch.zeros(3),
                    (R[b].T @ lib.root_R[fut])[:, :2].reshape(6),
                )))  # fmt: skip
                rel_pos.append(R[b].T @ (lib.root_p[fut] - p[b]))
            actor = torch.cat(feats)
            actors.append(actor)
            wms.append(torch.cat((actor, torch.cat(rel_pos))))
        return torch.stack(actors), torch.stack(wms)

    def test_layout_and_values(self):
        frames = torch.tensor([0, 30, 40])  # clip starts and an interior frame
        actor, wm = self.lib.goal(frames, self.p, self.R)
        self.assertEqual(actor.shape, (3, ACTOR_GOAL_DIM))
        self.assertEqual(wm.shape, (3, WM_GOAL_DIM))
        self.assertEqual(ACTOR_GOAL_DIM, GOAL_FRAMES * PER_FRAME)
        torch.testing.assert_close(wm[:, :ACTOR_GOAL_DIM], actor, rtol=0, atol=0)
        exp_actor, exp_wm = self.expected(frames, self.p, self.R)
        torch.testing.assert_close(actor, exp_actor)
        torch.testing.assert_close(wm, exp_wm)

    def test_window_never_crosses_into_next_clip_and_padding_is_stationary(self):
        frames = torch.tensor([20, 27, 60])  # windows that run past the end of their clip
        actor, wm = self.lib.goal(frames, self.p, self.R)
        exp_actor, exp_wm = self.expected(frames, self.p, self.R)
        torch.testing.assert_close(actor, exp_actor)
        torch.testing.assert_close(wm, exp_wm)
        per = actor.reshape(3, GOAL_FRAMES, PER_FRAME)
        for b, frame in enumerate(frames.tolist()):
            end = self.lib.clip_end[self.lib.clip_id[frame]].item()
            for i in range(GOAL_FRAMES):
                if frame + 1 + i * GOAL_STRIDE > end:  # padded slot: final pose, zero velocity
                    for sl in VEL:
                        self.assertEqual(torch.count_nonzero(per[b, i, sl]).item(), 0)
                    torch.testing.assert_close(per[b, i, :29], self.lib.q[end])

    def test_queries_do_not_mutate_source_or_reference_state(self):
        lib = self.lib
        source = {name: value.clone() for name, value in vars(lib).items()}
        frames = lib.clip_end.clone()
        before = tuple(value.clone() for value in lib.ref_state(frames))
        p, R = self.p[:2], self.R[:2]
        for _ in range(2):
            actor, wm = lib.goal(frames, p, R)
            exp_actor, exp_wm = self.expected(frames, p, R)
            torch.testing.assert_close(actor, exp_actor)
            torch.testing.assert_close(wm, exp_wm)
        for name, value in source.items():
            torch.testing.assert_close(getattr(lib, name), value, rtol=0, atol=0)
        for actual, expected in zip(lib.ref_state(frames), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
