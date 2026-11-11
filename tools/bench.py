"""Offline benchmark/diagnostics of world-model and policy updates on a synthetic buffer (no simulator)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib
from supermimic.mdp import MDP, flatten_state, unflatten_state, normalize_xi, OBS_DIM
from supermimic.networks import WorldModel
from supermimic.robot import default_joint_pos_list
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
kin = G1Kinematics(dev)
motion = MotionLib.from_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lafan1_g1"), kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
wm = WorldModel().to(dev)
B = 2048
f = torch.randint(0, motion.num_frames - 20, (B * 8,), device=dev)
S = motion.ref_state(f); S1 = motion.ref_state(f + 1)
a = torch.randn(B * 8, 29, device=dev); xi = torch.zeros(B * 8, 33, device=dev); xi[:, 0] = 0.75
wm.in_norm.update(wm.inputs(S, a, normalize_xi(xi))); wm.out_norm.update(wm.targets(S, S1))
std = wm.out_norm.std()
names = ["dp"] * 3 + ["dth"] * 3 + ["dv"] * 3 + ["dw"] * 3 + ["dq"] * 29 + ["dqd"] * 29
print("out std per group:", {n: f"{std[[i for i, m in enumerate(names) if m == n]].min().item():.2e}..{std[[i for i, m in enumerate(names) if m == n]].max().item():.2e}" for n in dict.fromkeys(names)})
def timeit(fn, n=5):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n
Sb = tuple(x[:B] for x in S)
def wm_step_fb():
    out = wm.step(Sb, a[:B], normalize_xi(xi[:B]))
    sum(o.sum() for o in out).backward()
print("wm step fwd+bwd (B=2048): %.1f ms" % (1000 * timeit(wm_step_fb)))
Sb4 = tuple(x[:4096].clone().requires_grad_(True) for x in motion.ref_state(f[:4096]))
fb = f[:4096]
print("FK fwd (B=4096): %.2f ms" % (1000 * timeit(lambda: kin.forward(*Sb4))))
print("reward fwd+bwd: %.2f ms" % (1000 * timeit(lambda: mdp.reward(Sb4, fb, a[:4096], a[:4096]).sum().backward())))
print("obs fwd: %.2f ms" % (1000 * timeit(lambda: mdp.obs(Sb4, a[:4096], xi[:4096]))))
print("goal: %.2f ms" % (1000 * timeit(lambda: motion.goal(fb, Sb4[0], Sb4[1]))))
print("critic feats: %.2f ms" % (1000 * timeit(lambda: mdp.critic_features(Sb4, fb, normalize_xi(xi[:4096]), motion.goal(fb, Sb4[0], Sb4[1])[1]))))
print("---- torch.compile ----")
kin_fwd_c = torch.compile(kin.forward)
t0 = time.time(); kin_fwd_c(*Sb4); torch.cuda.synchronize(); print("compile FK %.1fs" % (time.time() - t0))
print("FK compiled fwd: %.2f ms" % (1000 * timeit(lambda: kin_fwd_c(*Sb4))))
rew_c = torch.compile(mdp.reward)
t0 = time.time(); rew_c(Sb4, fb, a[:4096], a[:4096]).sum().backward(); torch.cuda.synchronize(); print("compile reward %.1fs" % (time.time() - t0))
print("reward compiled fwd+bwd: %.2f ms" % (1000 * timeit(lambda: rew_c(Sb4, fb, a[:4096], a[:4096]).sum().backward())))
