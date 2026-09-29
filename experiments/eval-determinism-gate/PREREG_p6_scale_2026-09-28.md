# Pre-registration: P6 magnitude at n = 100 per arm (2026-09-28)

Written and committed before any episode of this run. Analysis fixed here; not edited after data.
Status in any paper: **EXPLORATORY** (Stage 1 of the original pre-registration returned INCONCLUSIVE and the Stage 2
harness was patched after that pre-registration). This file fixes n and the analysis so the scale run cannot be
tuned after seeing results.

## Design
- Policy `HuggingFaceVLA/smolvla_libero`, suite `libero_object`, task 0, MAX_STEPS 300, BURN_DRAWS 1000,
  10-step warm-up, harness = Stage 2 patch v3 (as in `p6_magnitude_resume_2026-09-28.ipynb`).
- **n = 100 per arm, seeds 10000–10099**, the identical list for both arms.
- Arm A (leaky): seed once with 10000, `env.reset()` each episode, 1000 global `np.random` draws between episodes.
  Arm B (seeded): `np.random.seed(s); env.seed(s)` before every reset.
- One arm per Kaggle Save Version (T4): `p6_scale_n100_armA_2026-09-28.ipynb`, `p6_scale_n100_armB_2026-09-28.ipynb`.
- Episode i of arm A is paired with episode i (seed 10000 + i) of arm B, the notebook's existing position pairing.
- Stage 1 init-state overlap between arms is expected to include episode 0: both arms reset from seed 10000 with a
  fresh RNG at i = 0, so episode 0 is shared **by construction**. Any other overlap is reported.

## Analysis (fixed now)
- **Primary:** paired difference in success rate, B − A, over the 100 pairs, with an **exact two-sided McNemar test**
  (binomial on the discordant pairs, p = 0.5). α = 0.05. No direction predicted.
- **Secondary:** arm-A run-to-run variation. The first 20 arm-A episodes of this run use the same initial states as
  the 09-27 and 09-28 runs (checked by init_hash against the 09-28 records). Report arm-A successes on i = 0–19 for
  all three runs (09-27: 19/20, 09-28: 16/20, scale: ?) and the same for arm B (15/20, 17/20, ?). Descriptive only; no test.
- **Validity checks, reported before any result:** each episode's init_hash equals Stage 1's hash for that arm and
  index; stack-drift control HARMLESS 6/6; `policy device: cuda`. A failed check is reported, not silently fixed.
- If a version dies before 100 episodes, the analysis runs on the completed pairs only and states n.
