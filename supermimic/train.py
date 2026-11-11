"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=320_000)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "muon"],
                    help="muon: Muon on hidden Linear weights, Adam on the rest")
add_launcher_args(parser)
args = parser.parse_args()

import torch
import torch.nn as nn

from supermimic.buffer import ReplayBuffer
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MAX_STEP_REWARD, MDP, OBS_DIM, normalize_xi, unflatten_state
from supermimic.motion import ACTOR_GOAL_DIM, WM_GOAL_DIM, MotionLib
import supermimic.networks as networks
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.robot import default_joint_pos_list

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"
os.makedirs(args.log_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

kin = G1Kinematics(dev)
motion = MotionLib.from_dir(args.motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
Hh, HZ = args.hist, args.horizon
CRITIC_IN = INV_DIM + CRITIC_EXTRA + WM_GOAL_DIM
V_MAX = MAX_STEP_REWARD / (1 - args.gamma)

wm = WorldModel().to(dev)
pi = Policy(OBS_DIM, ACTOR_GOAL_DIM, hist=Hh).to(dev)
critic = Critic(CRITIC_IN).to(dev)


def build_optimizers(module: nn.Module, betas=(0.9, 0.999)) -> list[torch.optim.Optimizer]:
    """Adam; with --optimizer muon, Muon on the hidden Linear weights of each MLP and Adam on the rest."""
    if args.optimizer == "adam":
        return [torch.optim.Adam(module.parameters(), lr=args.lr, betas=betas)]
    hidden = [lin.weight for seq in module.modules() if isinstance(seq, nn.Sequential)
              for lin in [m for m in seq if isinstance(m, nn.Linear)][1:-1]]  # fmt: skip
    ids = {id(p) for p in hidden}
    rest = [p for p in module.parameters() if id(p) not in ids]
    return [torch.optim.Adam(rest, lr=args.lr, betas=betas), torch.optim.Muon(hidden, lr=args.lr, adjust_lr_fn="match_rms_adamw")]


def opt_step(opts, loss, params, max_norm, clip_value=0.0):
    params = list(params)  # a generator would be exhausted by the first clip
    for o in opts:
        o.zero_grad(set_to_none=True)
    loss.backward()
    gn = nn.utils.clip_grad_norm_(params, max_norm)
    if clip_value > 0:
        nn.utils.clip_grad_value_(params, clip_value)
    if torch.isfinite(gn):
        for o in opts:
            o.step()
    return gn


opt_wm = build_optimizers(wm)
opt_pi = build_optimizers(pi, betas=(0.9, 0.95))
opt_v = build_optimizers(critic)
start_it, wall_offset, sim_steps = 0, 0.0, 0
if args.resume:
    ck = torch.load(args.resume, map_location=dev)
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
    for opts, key in ((opt_wm, "opt_wm"), (opt_pi, "opt_pi"), (opt_v, "opt_v")):
        for o, sd in zip(opts, ck[key]):
            o.load_state_dict(sd)
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]


def _wm_rollout(s, a, xi):
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], normalize_xi(xi[:, j]))
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)


def _wm_err(S_hat, s1, mask):
    e = wm.weighted_sq_error(S_hat, unflatten_state(s1)).mean(-1)  # (B, k), Bryson-weighted
    return (e * mask).sum() / mask.sum().clamp(min=1), e[:, -1]


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g):
    a = pi(o_hist, g)
    S1 = wm.step(S, a, xi_n)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    g1, gw1 = motion.goal(f1, S1[0], S1[1])  # the relative-orientation terms carry gradient through S1
    return S1, a, o_hist, f1, g1, gw1


def _reward_term(S, f, a, a_prev):
    return mdp.reward(S, f, a, a_prev), mdp.soft_termination(S, f)


def _critic_input(S, f, xi_n, gw):
    return mdp.critic_features(S, f, xi_n, gw)


wm_rollout = torch.compile(_wm_rollout)
wm_err = torch.compile(_wm_err)
imag_step = torch.compile(_imag_step)
reward_term = torch.compile(_reward_term)
critic_input_c = torch.compile(_critic_input)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    S_hat = wm_rollout(w["s"], w["a"], w["xi"])
    loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    opt_step(opt_wm, loss, wm.parameters(), 1.0)
    m = w["mask"][:, -1]
    return loss.detach(), (e_last.detach() * m).sum() / m.sum().clamp(min=1)


def sample_valid_starts(B: int):
    starts, hists, n = [], [], 0
    while n < B:
        s, h = buffer.sample_starts(2 * B, Hh)
        f = s["f"]
        ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    return cat(starts), cat(hists)


PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))


def policy_update():
    if PROFILE:
        torch.cuda.synchronize(); tq0 = time.time()
    s, h = sample_valid_starts(args.pi_batch)
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    xi_n = normalize_xi(xi)
    with torch.no_grad():
        o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"])
    o_hist = mdp.add_noise(o_hist)
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    g, gw0 = motion.goal(f, S[0], S[1])
    gw = gw0
    S0 = S
    Ss, As, Aps, Fs, Gws = [], [], [], [], []
    for t in range(HZ):
        Aps.append(a_prev)
        S, a_prev, o_hist, f, g, gw = imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
        Ss.append(S), As.append(a_prev), Fs.append(f), Gws.append(gw)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    f_all = flat(Fs)
    r, d = reward_term(S_all, f_all, flat(As), flat(Aps))
    r, d = r.view(HZ, B), d.view(HZ, B)
    alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    J = (disc * c_t * r).sum(0)
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    if not args.no_critic:
        J = J + (args.gamma**HZ) * c * critic(_critic_input(S, f, xi_n, gw)).clamp(max=V_MAX)
    loss = -J.mean()
    if PROFILE:
        torch.cuda.synchronize(); tq1 = time.time()
    gn = opt_step(opt_pi, loss, pi.parameters(), 1.0, clip_value=0.1)
    if PROFILE:
        torch.cuda.synchronize(); tq2 = time.time()
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    v_loss = torch.zeros(())
    if not args.no_critic:
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((gw0.unsqueeze(0), torch.stack(Gws)), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        with torch.no_grad():
            V = critic(X).clamp(max=V_MAX)
            R = V[-1]
            targets = []
            for t in range(HZ - 1, -1, -1):
                R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1]).clamp(max=V_MAX)  # (H, B)
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                opt_step(opt_v, v_loss, critic.parameters(), 1.0)
    if PROFILE:
        torch.cuda.synchronize()
        print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
    return loss.detach(), v_loss.detach(), (r_sum / HZ / 0.1).mean(), gn.detach(), c.mean().detach()


# ------------------------------------------------------------------------------------------------
sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma)
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev)
    obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)

    @torch.no_grad()
    def collect(n_steps: int, stats: dict):
        global obs_hist, sim_steps
        for _ in range(n_steps):
            S = env.state()
            o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi))
            fresh = env.ep_step == 0
            obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
            g, _ = motion.goal(env.f, S[0], S[1])
            pi.obs_norm.update(o)
            pi.goal_norm.update(g)
            a = pi(obs_hist, g)
            a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
            out = env.step(a)
            buffer.add(out)
            S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
            wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
            wm.out_norm.update(wm.targets(S0, S1))
            sim_steps += args.num_envs
            stats["r"].append(out["r"].mean() / 0.1)
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].mean())
                stats["ep_ret"].append(out["ep_ret_norm"].mean())
                stats["term"].append(out["term"].float().sum())
                stats["done"].append(out["done"].float().sum())

    stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
    if buffer.size < args.wm_window:
        collect(max(args.warmup_steps, args.wm_window), stats)
    log_f = open(os.path.join(args.log_dir, "train_log.csv"), "a")
    if start_it == 0:
        log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr\n")
    t_start = time.time() - wall_offset
    t_log = time.time()
    step_acc = 0.0
    agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
    for it in range(start_it, args.iterations):
        lr = lr_at(it)
        for o in (*opt_wm, *opt_pi, *opt_v):
            for gparam in o.param_groups:
                gparam["lr"] = lr
        step_acc += 1.0 / args.utd
        n = int(step_acc)
        step_acc -= n
        prof = PROFILE and it % 20 == 0 and it > args.wm_only_iters + 5
        if prof:
            torch.cuda.synchronize(); tp0 = time.time()
        collect(n, stats)
        if prof:
            torch.cuda.synchronize(); tp1 = time.time()
        for _ in range(args.wm_batches):
            l, e8 = wm_update()
        if prof:
            torch.cuda.synchronize(); tp2 = time.time()
        agg["wm"].append(l), agg["wm8"].append(e8)
        if it >= args.wm_only_iters:
            pl, vl, ir, gn, surv = policy_update()
            if prof:
                torch.cuda.synchronize(); tp3 = time.time()
                print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
            agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
        if (it + 1) % args.log_every == 0:
            mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
            ips = args.log_every / (time.time() - t_log)
            t_log = time.time()
            wall_h = (time.time() - t_start) / 3600
            term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
            row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
                   mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["ep_len"]), mean(stats["ep_ret"]), term_frac, ips, lr]  # fmt: skip
            log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
            log_f.flush()
            print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
                  f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s", flush=True)  # fmt: skip
            stats = {k: [] for k in stats}
            agg = {k: [] for k in agg}
        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
            ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": [o.state_dict() for o in opt_wm],
                  "opt_pi": [o.state_dict() for o in opt_pi], "opt_v": [o.state_dict() for o in opt_v], "iteration": it + 1,
                  "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
                  "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
            last = os.path.join(args.log_dir, "last.pt")
            torch.save(ck, last + ".tmp")
            os.replace(last + ".tmp", last)
            small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
            path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
            torch.save(small, path + ".tmp")
            os.replace(path + ".tmp", path)
