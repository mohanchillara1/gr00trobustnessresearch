# Eval-determinism in LIBERO — experiment summary

**One sentence:** LIBERO's evaluation environment draws its per-episode initial state from the
**global** `numpy` RNG, so any code in the same process that consumes randomness between episodes
silently changes which episodes a policy is scored on — and calling `seed()` before every reset
fixes it completely.

Two runs, both CPU-only, no GPU and no checkpoint. All hashes below are the first 16 hex characters
of a SHA-256 over LIBERO's own `get_sim_state()` (the full MuJoCo state vector), so a match is a
bit-identical simulator state, not a similarity score.

---

## What was tested

No policy is involved anywhere. Actions come from a pinned `np.random.RandomState`, episode counts
are fixed, and nothing is filtered. **Any difference in simulator state is therefore the
environment**, which is what made the earlier 2026-08-25 attempt uninterpretable: it compared runs
of 6 vs 5 episodes driven by a *sampling* policy, so both the episode count and the action stream
differed between arms before the environment was even considered.

| Test | Question |
|---|---|
| **T1** | Is a seeded reset reproducible at all? `seed(s)` + `reset()` twice in fresh envs. |
| **T2** | Does the next episode depend on **how many steps** the previous one took? K=40 vs K=80, no reseed. |
| **T3** | Does calling `seed()` again before the reset restore alignment? (no burn) |
| **T4** | Do a 6-reset and a 5-reset run share the same first five episodes? |
| **T5** | Does consuming the **global** RNG between episodes change the next one? 1000 `np.random.random()` calls. |
| **T6** | Does the fix survive the disease? `seed()` **and** 1000 global draws. |

Every prediction was written into the notebook **before** the run. T2 was predicted FAIL in the
first draft; a preliminary gymnasium run refuted that, and the prediction was corrected to PASS
before the robosuite run rather than after seeing the result.

## Stack

`python 3.11.16` · `numpy 1.26.4` · `robosuite 1.4.0` · `mujoco 3.1.6` · `gym 0.25.2` ·
`torch 2.14.0+cpu` · `bddl` (no `__version__`; 1.0.1 requested) · LIBERO `main`, shallow clone
2026-09-12/13. `OffScreenRenderEnv` at 128×128, EGL. Seeds 11 / 12 / 13 throughout.

Run in a `uv`-built Python 3.11 venv on free Colab because LIBERO's pins have no wheels on Colab's
own Python 3.13.

---

## Run 1 — depth, one task (v5, 2026-09-12 22:32 UTC, 13.7 min)

`libero_object` / `pick_up_the_alphabet_soup_and_place_it_in_the_basket`. **Verdict: CONFOUND
CONFIRMED.**

| Test | Seed 11 | Seed 12 | Seed 13 | Result |
|---|---|---|---|---|
| T1 | `abef73e1a57d0aeb` | `2ef16bcbf9deb737` | `c7f55ad206b80b4e` | **PASS** |
| T2 | `cbb3470b4c447f68` | `c732927029d36125` | `c09097460e7814a5` | **PASS** (K40 = K80) |
| T3 | `2ef16bcbf9deb737` | `c7f55ad206b80b4e` | `f2cc23d230f9b9f3` | **PASS** |
| T4 | aligned, distinct | aligned, distinct | aligned, distinct | **PASS** |
| T5 | → `e78e0d1a180b3e6a` | → `42bec853db81b54b` | → `dfa9091d9724f85f` | 🔴 **FAIL 3/3** |

## Run 2 — breadth, three suites (v6, 2026-09-13 05:36 UTC, 38.1 min)

T2, T5, T6 on task 0 of three suites. **Verdict: GENERAL.**

| Suite (task 0) | T2 | T5 (burned) | T6 |
|---|---|---|---|
| `libero_object` · alphabet soup | PASS `cbb3470b…` `c7329270…` `c0909746…` | **FAIL** `e78e0d1a…` `42bec853…` `dfa9091d…` | PASS `2ef16bcb…` `c7f55ad2…` `f2cc23d2…` |
| `libero_spatial` · black bowl between plate and ramekin | PASS `c8c5d6f7…` `3adb49be…` `f0c44c4e…` | **FAIL** `8ce291ea…` `59302ec5…` `21855380…` | PASS `be32f15e…` `4b709dcc…` `6f102254…` |
| `libero_goal` · open the middle drawer | PASS `d70b8e1a…` `b2eac4e2…` `1caf678d…` | **FAIL** `48170b1c…` `04474892…` `7434500d…` | PASS `2dbffb7c…` `cf677202…` `2f38318d…` |

**T5 fails 9/9. T2 and T6 pass 9/9. All 27 hashes distinct. Every control reproduced.**

The three tasks are structurally different — a pick-and-place, a spatial-reference pick, and a
drawer articulation — so this is not a property of one scene.

### The replication control

v6 re-ran `libero_object` and reproduced **all six** of v5's T2 and T5-burned hashes exactly. The
v5 result is not an artifact, and the two runs are directly comparable.

---

## The finding

**T2 passes everywhere: step count is irrelevant.** The obvious hypothesis — that a longer rollout
advances the RNG further — is wrong, and was refuted before it reached this write-up.

**T5 fails everywhere: global-RNG consumption is what matters.** robosuite 1.4's placement sampling
draws from the process-global `np.random` after the initial `seed()`. So anything else in that
process that touches `np.random` between episodes — a policy sampling actions, an augmentation, a
logger, a different batch size — shifts every subsequent episode's initial state. Two arms that
consume different amounts of randomness are scored on **different episodes**, which is precisely
the confound a paired comparison exists to remove.

A second, independent contributor sits in LIBERO itself: `ControlEnv.reset()` retries on
`RandomizationError` in a `while` loop, so the randomness consumed *per reset* is variable.

## The fix, and why it is more than "T6 passed"

Call `seed()` before every reset, not only the first.

T6's hashes are **bit-identical to T3's** on all three seeds, and T3's are in turn identical to T1's
freshly-seeded hashes for seed s+1. Across three independent runs:

> `reseed + 1000 global draws` == `reseed + no draws` == `a fresh env seeded with s+1`

**The initial state becomes a pure function of the seed — independent of rollout history and
independent of global-RNG state.** The fix does not mask the leak; it removes the state's
dependence on everything except the seed.

---

## Two things this does **not** show

**1 · Magnitude, on a real policy.** Every number here is a simulator-state hash. How far a
*success rate* actually moves when two arms are scored on different episode sets is unmeasured, and
measuring it needs a policy in the loop — the only item on this list that costs money. Until it
exists, the honest claim is "a defect exists and here is the verified fix," not "here is what it
cost the field."

**2 · Whether this repo's own headline numbers are affected.** The novel-object eval
(`evaluationScriptForNovelObj.sh`) passes no `--seed` and never sets `GR00T_EVAL_SEED`, and
GR00T's `seed_everything` is documented as a **no-op** when both are absent — *"Seeding is opt-in.
If no seed is supplied and the `GR00T_EVAL_SEED` environment variable is unset, these helpers are
no-ops."* So `rollout_policy.py:309` runs a bare `env.reset()` and that evaluation was **never
seeded at all**, with a fresh process per condition.

**But the training-object evaluation script — the one behind the 0.59 vs 0.77 headline — is not in
this repo.** Whether it passed a seed cannot be determined from here and is not being inferred from
its sibling. **That is the single highest-value open question for this project.**

Worth noting against the exciting reading: the T5 leak probably *cannot* fire in GR00T's eval
anyway, because the policy runs in a **separate process** (a server on `127.0.0.1:5555`) and
`rollout_policy.py` contains no `np.random` use of its own. The defect in these numbers is the
simpler one — no seeding — not the leak.

---

## Run 3 — Mohan's own replication (v6, 2026-09-21 21:51 UTC, 23.0 min)

Run by **Mohan Chillara Jr** on a free Colab CPU runtime: the local `robosuite_gate_v6.ipynb` (commit `69c76e8`)
uploaded via File → Upload notebook, Runtime → Run all, no edits. Output pasted into chat by dad the same evening;
saved verbatim as `results/det_result_v6_mohan_2026-09-21.json`.

**Verdict: GENERAL. T2 pass 9/9, T5 fail 9/9, T6 pass 9/9. All 27 hashes identical to Run 2 (2026-09-13).**
The v5 replication control reproduced all six `libero_object` hashes exactly. Stack: Python 3.11.16, numpy 1.26.4,
robosuite 1.4.0, mujoco 3.1.6, gym 0.25.2, torch 2.14.0+cpu — same as Run 2.

This is the first run of the gate by a human on the team. It clears item 1 of the council's 2026-09-15 gate list
for the P6 submissions (re-run end to end yourself); the code-read, intro and related work remain his to do.

## Files

| | |
|---|---|
| `eval_determinism_gate.ipynb` | preliminary run on gymnasium MuJoCo; refuted the step-count hypothesis |
| `robosuite_gate_v5.ipynb` | depth run, one task, T1–T5 |
| `robosuite_gate_v6.ipynb` | breadth run, three suites, T2/T5/T6 |
| `results/det_result_robosuite_v5.json`, `results/robosuite_gate_v5_output.txt` | v5 record |
| `results/det_result_v6.json` | v6 record |
| `README.md` | how to run, and the four earlier notebook revisions that failed and why |

Primary records are the Colab autosaves on Drive (v5 `1RVhcaH-bbIAW5hdLsMHgkH1aDOKdAxkU`,
v6 `1PXQLI7vbAffB2fdLKeqDCZa4IEpiUypn`); the JSON files here are transcriptions, verified against
the autosave and against each other on 2026-09-13.
