# Results — MiniGrid-Empty-6x6, MPC over an RSSM world model

Success = agent reaches the green goal within the step cap.
`wm` and `wm_vlm` use disjoint objectives (reward-only vs VLM-only) so the
comparison isolates each signal.

| Mode | Success rate | Mean return | Mean steps | N (episodes) |
|------|:-----------:|:-----------:|:---------:|:-----------:|
| Random | 0.37 | 0.176 | 54.1 | 30 (10 × seeds {0,1,2}) |
| WM planning (no VLM) | 0.93 | 0.422 | 40.6 | 30 (10 × seeds {0,1,2}) |
| WM planning + VLM (VLM-only) | 0.13 | 0.084 | 59.0 | 15 (5 × seeds {0,1,2}) |

Notes:
- `wm` (world-model planning on the predicted reward) clearly beats `random`:
  0.93 vs 0.37 success, in fewer steps.
- `wm_vlm` (planning on the CLIP goal score only) does *worse than random*.
  Root cause (report §4): CLIP detects the goal only on the full 192px native render
  (AUC 0.98); on the world model's decoded 128px frames the cue is anti-correlated
  (AUC 0.30 < 0.5), so optimising it steers the agent away from the goal.
- `wm_vlm` uses fewer episodes because CLIP scoring is the bottleneck
  (~120–150 s/episode on MPS); the outcome does not change with more episodes.
