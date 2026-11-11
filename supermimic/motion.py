"""Reference motion library: LAFAN1 G1 retargets resampled to 50 Hz, with FK of tracked bodies."""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from .kinematics import G1Kinematics
from .rotations import mat_to_6d, quat_slerp, quat_to_mat, so3_log, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50  # 1 s bins for return-based reset sampling
# Goal window: GOAL_FRAMES reference frames GOAL_STRIDE control steps apart (~0.9 s lookahead).
GOAL_FRAMES = 10
GOAL_STRIDE = 5
ACTOR_GOAL_DIM = GOAL_FRAMES * (29 + 29 + 3 + 3 + 3 + 6)
WM_GOAL_DIM = ACTOR_GOAL_DIM + GOAL_FRAMES * 3


class MotionLib:
    def __init__(self, files: list[str], kin: G1Kinematics, device: str, src_fps: float = 30.0):
        self.device = device
        self.names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        ps, qs, js = [], [], []
        lens = []
        for f in files:
            raw = torch.tensor(np.loadtxt(f, delimiter=","), dtype=torch.float32)
            p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7:36]
            quat = torch.cat((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1)
            # resample to 50 Hz
            T_src = raw.shape[0]
            dur = (T_src - 1) / src_fps
            n = int(np.floor(dur / CONTROL_DT)) + 1
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            a = (t - i0.float()).clamp(0, 1)
            p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
            j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
            quat = quat_slerp(quat[i0], quat[i0 + 1], a)
            ps.append(p), qs.append(quat), js.append(j)
            lens.append(n)
        self.clip_len = torch.tensor(lens, device=device)
        self.clip_start = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(lens[:-1], dtype=torch.long))), 0).to(device)  # dtype: one clip gives an empty (float) tensor
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]).to(device)

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = quat_to_mat(quat)
            v = _gradient(p, CONTROL_DT)
            qd = _gradient(j, CONTROL_DT)
            # angular velocity (world) by central differences of rotations
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            root_p.append(p), root_R.append(R), root_v.append(v), root_w.append(w), jq.append(j), jqd.append(qd)
        cat = lambda xs: torch.cat(xs).to(device)  # noqa: E731
        self.root_p, self.root_R, self.root_v, self.root_w = cat(root_p), cat(root_R), cat(root_v), cat(root_w)
        self.q, self.qd = cat(jq), cat(jqd)
        with torch.no_grad():
            bp, bR, bv, bw = [], [], [], []
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                out = kin.forward(self.root_p[s:e], self.root_R[s:e], self.root_v[s:e], self.root_w[s:e], self.q[s:e], self.qd[s:e])
                bp.append(out[0]), bR.append(out[1]), bv.append(out[2]), bw.append(out[3])
            self.body_p, self.body_R, self.body_v, self.body_w = torch.cat(bp), torch.cat(bR), torch.cat(bv), torch.cat(bw)
        # per-frame static goal parts: q, qd, root velocities in the root frame
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]  # -(projected gravity z) of the reference root: g_b = R^T (0,0,-1)
        # 1 s bins
        bin_clip, bin_start = [], []
        for c in range(len(lens)):
            nb = int(np.ceil(lens[c] / BIN_FRAMES))
            for b in range(nb):
                bin_clip.append(c)
                bin_start.append(int(self.clip_start[c]) + b * BIN_FRAMES)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for b in range(self.num_bins):
            c = bin_clip[b]
            s = bin_start[b]
            e = min(s + BIN_FRAMES, int(self.clip_end[c]) + 1)
            self.frame_bin[s:e] = b

    @staticmethod
    def from_dir(path: str, kin: G1Kinematics, device: str) -> MotionLib:
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        return MotionLib(files, kin, device)

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal_frames(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Strided window f+1, f+1+S, ..., clamped to the clip's last frame, and which slots are recorded."""
        end = self.clip_end[self.clip_id[f]].unsqueeze(-1)
        requested = f.unsqueeze(-1) + 1 + torch.arange(GOAL_FRAMES, device=f.device) * GOAL_STRIDE
        return torch.minimum(requested, end), requested <= end

    def goal(self, f: torch.Tensor, p: torch.Tensor, R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(actor_goal, wm_goal)`` from a strided window of future reference frames.

        Per frame (actor, deployable): joint positions and velocities; the reference root trajectory in
        the current reference heading frame (displacement and root-frame velocities, reference only);
        and the reference root orientation relative to the robot root (6D, IMU-observable).
        ``wm_goal`` (critic only) adds the reference root position relative to the robot root.
        Slots beyond the clip hold the final pose with zero velocity.
        """
        fut, recorded = self.goal_frames(f)  # (B,N)
        rec = recorded.unsqueeze(-1)
        qd = torch.where(rec, self.qd[fut], 0.0)
        v = torch.where(rec, self.root_v_local[fut], 0.0)
        w = torch.where(rec, self.root_w_local[fut], 0.0)
        Ht = yaw_mat(self.root_R[f]).transpose(-1, -2).unsqueeze(-3)
        dp = (Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        Rt = R.transpose(-1, -2).unsqueeze(-3)
        rel_ori = mat_to_6d(Rt @ self.root_R[fut])
        rel_pos = (Rt @ (self.root_p[fut] - p.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        actor = torch.cat((self.q[fut], qd * 0.1, dp, v, w, rel_ori), -1).flatten(-2)
        return actor, torch.cat((actor, rel_pos.flatten(-2)), -1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g
