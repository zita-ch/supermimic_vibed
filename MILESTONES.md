# SuperMimic reproduction — milestone log
Times are local time (CEST, UTC+2) unless marked UTC. One GPU (RTX PRO 6000 Blackwell), 48 CPU threads.

| Time | Milestone |
|---|---|
| 2026-10-21 16:55 | Task start; paper read (no public code: project page says "Coming soon") |
| 2026-10-21 17:00 | Cloned IsaacLab main @ 46d6cc3 (VERSION 3.0.0); started kitless image build (Dockerfile.kitless) |
| 2026-10-21 17:01 | Kitless image built (isaac-lab-kitless:main-46d6cc3, +tmux/ffmpeg -> supermimic:latest); container 'supermimic' up |
| 2026-10-21 17:07 | Task 1 done: stock Isaac-Velocity-Flat-G1 PPO trains kitless on Newton/MuJoCo-Warp (physics=newton_mjwarp), 4096 envs, ~224k steps/s |
| 2026-10-21 17:16 | G1 29-DoF asset working on MJWarp (URDF->MJCF->USD, BeyondMimic actuators): ~640k control steps/s @4096 envs |
| 2026-10-21 17:26 | Full SuperMimic pipeline (sim collect -> WM fit -> BPTT policy+critic) runs end-to-end; debugging WM loss scale and speed (0.66 it/s) |
| 2026-10-21 17:49 | Speed fixes (torch.compile, batched reward/critic/WM-error, bf16 MLPs): 0.66 -> 3.1 it/s. MAIN RUN STARTED: logs/lafan1_main (tmux 'lafan1_main' in container) |
| 2026-10-21 17:50 | Restarted main run with njmax=500/nconmax=128 (rare MJWarp constraint overflow during falls) |
| 2026-10-21 18:11 | First checkpoint (it 2000, 0.15 h train time): eval normalized return 0.541, 12/40 clips failed. Eval watcher started (tmux 'eval_main') |
| 2026-10-21 18:19 | it 4000 (0.41 h train time): eval normalized return 0.685, 4/40 failed |
| 2026-10-21 18:32 | it 6000 (0.62 h train time): eval normalized return 0.781, 1/40 failed |
| 2026-10-21 18:47 | it 8000 (0.82 h train time): eval normalized return 0.779, 2/40 failed. Failure analysis of fallAndGetUp1_subject5 (it 6000): genuine tracking failure in a get-up (robot rolls over); the paper tilt termination fires correctly |
| 2026-10-21 18:39 | Success criterion set: eval return >= 0.82 AND 40/40 clips without termination |
| 2026-10-21 18:57 | it 10000 (1.04 h train time): eval normalized return 0.811, 1/40 failed (fallAndGetUp2_subject3) |
| 2026-10-21 19:10 | **RETURN TARGET REACHED**: it 12000 (1.24 h train time, ckpt saved 2026-10-21 17:05:52 UTC): eval normalized return 0.824 >= 0.82 (paper: 0.82 at 1.2 h on H100); 1/40 failed (jumps1_subject2) |
| 2026-10-21 19:23 | **REPRODUCED (both criteria)**: it 14000, 1.47 h train time (ckpt saved 2026-10-21 17:19:24 UTC = 19:19 CEST): eval normalized return 0.829 >= 0.82 AND 0/40 clips failed |
| 2026-10-21 19:31 | Training stopped by request at it ~15700 (1.67 h train time); last checkpoint ckpt_014000 (the 40/40 one) |
| 2026-10-21 19:44 | Reproduction video rendered (results/supermimic_lafan1_it14000_reproduced.mp4); container removed, GPU released |
| 2026-11-11 02:27 | v2 (acceleration-only world model, strided goal, LayerNorm+GELU): it 10000 eval return 0.842, 0/40 failed at 0.95 h of training (1.05 h incl. evaluations); training stopped at it ~10450; checkpoint in checkpoints/ |
