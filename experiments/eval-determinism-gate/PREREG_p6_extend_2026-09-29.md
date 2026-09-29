# Pre-registration: P6 extension to n = 300 per arm (2026-09-29)

Written and committed before any new episode. The n = 100 result (A 92/100, B 85/100, exact McNemar p = 0.167,
`0084a13`) has already been seen, so this file fixes the final totals and the single analysis now. Not edited after data.
Status in any paper: **EXPLORATORY**, as for the n = 100 run (Stage 1 of the original prereg was INCONCLUSIVE and the
harness was patched after it).

## Design (Part A: same task, same policy)
- Policy `HuggingFaceVLA/smolvla_libero`, `libero_object` task 0, MAX_STEPS 300, BURN_DRAWS 1000, 10-step warm-up,
  Stage 2 patch v3, arm A / arm B logic exactly as `PREREG_p6_scale_2026-09-28.md` (`ec209c0`).
- **Final total: 300 pairs per arm** = the committed 100 (i = 0–99, seeds 10000–10099, `0084a13`) + **200 new**
  (i = 100–299, seeds 10100–10299). Arm A is one continuous leaky chain from seed 10000: its episodes i ≥ 100 continue
  the chain by replaying the committed resets + burn draws without rollouts and checking every replayed init_hash
  (mismatch = abort, reported). Arm B is reseeded per episode.
- Run in chunks of at most 100 new episodes per GPU, because 200 SmolVLA episodes (~213 s each) exceed Kaggle's 12 h
  session cap. Chunk 1 = i 100–199 (`p6_extend_c1_arm{A,B}_2026-09-29.ipynb`); chunk 2 = i 200–299, same logic with
  `N_EPISODES = 300` and the chunk-1 episodes added to the resume file. Chunking changes no episode's seed or init state.
- Stopping rule: the totals are fixed at 300. No look at interim counts changes n. If compute runs out before 300,
  the analysis runs on the completed pairs and states n; no further episodes are added after the analysis is run.

## Analysis (fixed now)
- **Primary:** paired B − A difference over **all 300 pairs**, **exact two-sided McNemar** (binomial on the discordant
  pairs, p = 0.5), α = 0.05, no direction predicted.
- **Replication (secondary):** the same test on the **200 new pairs alone** (i = 100–299), reported alongside.
- Descriptive: per-chunk counts (0–99, 100–199, 200–299); no test.
- **Power (director's estimate, stated before data):** at ~19 % discordance (19/100 at n = 100), 300 pairs give ~57
  discordant pairs, roughly 80 % power for a 7 pp paired gap at α = 0.05. This is an estimate, not a guarantee; a null
  at n = 300 bounds the effect, it does not prove zero.
- **Validity checks, reported before any result:** every episode's init_hash equals Stage 1's hash for that arm and
  index; arm A replay reproduces all committed hashes; Stage 1 hash lists for i < 100 equal the `0084a13` ones;
  stack-drift control HARMLESS 6/6; `policy device: cuda`. A failed check is reported, not silently fixed.

## Part B (generalization)
Its design is registered in a separate addendum **committed before any Part B episode**. Nothing in Part B changes
the Part A analysis above.
