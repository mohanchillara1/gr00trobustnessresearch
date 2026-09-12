# Eval-determinism gate — do two policy arms see the same LIBERO episodes?

**Run this first. It is the gate for the winning paper, and nothing downstream is interpretable
until it returns a dated verdict.**

## What it tests

> When two policy arms take a different number of steps, do they see the same evaluation episodes,
> or different ones?

If different, **every paired comparison on the stock LIBERO eval path is confounded** — including
the A_clean vs B_dim comparison behind this repo's own +18pp result.

## The mechanism, stated before the run

Gymnasium's API documentation, verbatim:

> *"However, if the environment already has a PRNG and **seed=None** is passed, **the PRNG will not
> be reset** and the env's np_random_seed will not be altered."*

In LIBERO: `LiberoEnv.reset()` calls `self._env.seed(int(seed))` then `self._env.reset()` and never
calls `set_init_state()`; and only the **first** reset of a rollout is seeded
(`rollout_policy.py:302-309`). Every later episode therefore draws from a PRNG whose position
depends on how many times it was advanced — i.e. on how many steps the policy took.

This has never been filed as a LIBERO bug because it is **documented, intentional Gymnasium
behaviour**, not a defect. That is also why it survived 15 months unnoticed.

## Why it supersedes the 2026-08-25 check

`mhh-gate/run_gate.py::verify_eval_determinism` ran one eval shard twice and compared per-episode
length and success vectors. It returned `IDENTICAL:False` on 2026-08-25 and was never usable,
because it carried two confounds:

1. **Mismatched episode counts** — run A produced 6 episodes, run B 5 (unseeded autoreset extras /
   invalid-episode filtering). `la == lb` on unequal-length lists is False for free.
2. **Unpinned policy sampling** — GR00T's action head samples, so identical observations give
   different actions and different trajectory lengths with no environment nondeterminism at all.

**This notebook removes both by construction, not by re-running and hoping:** episode count is
fixed explicitly and nothing is filtered, and **there is no policy** — actions come from a pinned
`np.random.RandomState`, so any difference in environment state is the environment.

## 🔴 Why it needs no checkpoint and no GPU

The question is a property of **LIBERO / robosuite / Gymnasium**, not of a trained model. Removing
the policy is what makes the test clean *and* what makes it runnable at all.

That matters, because as of 2026-09-12 **the six A_clean / B_dim checkpoints could not be found on
Drive** and the RunPod pod that held the harness was terminated 2026-09-06. This experiment does
not care.

## Exact command

Open `eval_determinism_gate.ipynb` in Colab and run top to bottom. No phase switching, no restart.

## Runtime and cost

| | |
|---|---|
| Hardware | **CPU only. No GPU. No checkpoint.** Free-tier Colab. |
| Wall clock | ~10–20 min, most of it the LIBERO install |
| Cost | ≈ $0 |

## Pre-registered outcomes

| Test | What it checks | Prediction |
|---|---|---|
| **T1** | `seed(s)` + `reset()` twice in fresh envs → identical first observation? | PASS |
| **T2** | reset *after* a rollout of K steps, no reseed → does the initial state depend on K? | **FAIL** |
| **T3** | same as T2 but reseeding explicitly every episode | PASS — the fix |

Plus a **control**: two runs of the *same* arm with the same K. If the control disagrees with
itself, T2 is uninterpretable and the notebook says `INCONCLUSIVE` rather than claiming a result.

**T2 failing is the finding, not a bug in this notebook.** Written down before the first run so it
cannot be rationalised afterwards.

## What "done" looks like

`/content/drive/MyDrive/mhh-determinism-gate/` contains:

- `determinism_gate.json` — every fingerprint, per seed, per test
- `determinism_gate.csv` — the flat verdict row
- `freeze.txt` — full `pip freeze`

and the notebook prints one of four verdicts with a UTC timestamp:

| Verdict | Meaning |
|---|---|
| **CONFOUND CONFIRMED** | T1 passes, T2 fails. Arms are scored on different episodes. **This is the paper.** |
| **HARNESS SOUND** | T1 and T2 pass. Paired comparisons are arm-matched, and the 2026-08-25 result was the two confounds. |
| **INCONCLUSIVE** | The control disagrees with itself. Do not use T2 either way. |
| **BROKEN** | Seeded resets are not reproducible at all. Suspect the install before LIBERO. |

**Copy the verdict line and its timestamp into
`mohanvault/01 Projects/Paper Choice 2026-09-12.md` §13.** A verdict that exists only in a Colab
output has not happened.

## If T2 fails

Three things follow, in order:

1. **The fix is T3** — seed every episode, not just the first. If T3 passes, the paper ships a
   finding *and* a one-line remedy, which is a materially stronger submission.
2. **The repo's own numbers need a caveat.** The A_clean vs B_dim comparison was run on this path.
3. Only then is the renderer work (`../renderer-luminance-gate/`) worth building out, because a
   2×2 measured on a confounded harness measures nothing.

## Status

⚠️ **Not executed.** The `chrome-devtools` MCP server was disconnected in the session that wrote
this, so Colab could not be driven. Verified instead, at primary:

- Every code cell parses (`ast.parse`, 14 cells, clean).
- `mujoco==3.3.3` exists and is installable (PyPI JSON API).
- The task BDDL file exists in LIBERO's `libero_object` suite.
- The Gymnasium sentence quoted above was fetched and confirmed at
  `https://gymnasium.farama.org/api/env/`.

**Not verified:** that the LIBERO install succeeds on current Colab, and that `OffScreenRenderEnv`
renders headless without an EGL/OSMesa flag. Expect that to break first; `MUJOCO_GL=egl` is the
usual fix.
