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
(`rollout_policy.py:302-309`).

🔴 **SUPERSEDED 2026-09-12 19:32 UTC — the inference that used to sit here was wrong.** It read:
*"Every later episode therefore draws from a PRNG whose position depends on how many times it was
advanced — i.e. on how many steps the policy took."* **That does not follow.** A PRNG that is *not
reset* simply continues deterministically, identically for every arm reaching the same point. The
gymnasium run confirmed it: T2 and T4 both passed. Original wording kept above the line so the
error stays visible. The live hypothesis is now **T5** (global-RNG consumption), not step count —
see "Two runs" below.

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
| **T2** | reset *after* a rollout of K steps, no reseed → does the initial state depend on K? | ~~**FAIL**~~ → **predicted FAIL, actually PASSED** |
| **T3** | same as T2 but reseeding explicitly every episode | PASS — the fix |

Plus a **control**: two runs of the *same* arm with the same K. If the control disagrees with
itself, T2 is uninterpretable and the notebook says `INCONCLUSIVE` rather than claiming a result.

**T2 failing is the finding, not a bug in this notebook.** Written down before the first run so it
cannot be rationalised afterwards. **It did not fail** — see "Two runs" below. The pre-registration
is the only reason that was undeniable rather than quietly reframed.

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

## If T2 fails — ⚠️ moot for the gymnasium run, still live for the robosuite one

Three things follow, in order:

1. **The fix is T3** — seed every episode, not just the first. If T3 passes, the paper ships a
   finding *and* a one-line remedy, which is a materially stronger submission.
2. **The repo's own numbers need a caveat.** The A_clean vs B_dim comparison was run on this path.
3. Only then is the renderer work (`../renderer-luminance-gate/`) worth building out, because a
   2×2 measured on a confounded harness measures nothing.

## Two runs: gymnasium (done) and robosuite/LIBERO (built, not run)

| Notebook | Stack | Status |
|---|---|---|
| `eval_determinism_gate.ipynb` | gymnasium MuJoCo | **RUN 2026-09-12 19:32 UTC** in the Composio workbench. Verdict **HARNESS SOUND**. |
| ~~`robosuite_gate.ipynb`~~ | robosuite + LIBERO | 🔴 **RUN 2026-09-12, FAILED AT INSTALL. No evidence produced.** Superseded. |
| **`robosuite_gate_v2.ipynb`** | **robosuite 1.4.0 + LIBERO in a uv Python 3.11 venv** | built, not run |

### Why v1 failed, recorded so it is not repeated

Two faults, both in cell 1, and the tests never executed:

1. **Colab is Python 3.13 now**, not 3.12. `numpy<2` has no cp313 wheel, so the resolver silently
   overrode the `numpy==1.26.4` pin with **numpy 2.1.3**, and `mujoco` resolved to **3.13.0** —
   both far outside what robosuite 1.4.0 expects. `gym 0.25.2` warned about NumPy 2.
2. **`pip install git+… --no-deps` produced no importable package.** Cell 2 raised
   `ModuleNotFoundError: No module named 'libero'`.

**Wrong assumption on my part: that pinning a version in a pip command means that version gets
installed.** A pin with no wheel for the live interpreter is a *suggestion* — pip resolves past it
and reports success. The lesson generalises past this notebook: print resolved versions from the
interpreter that will run the code, before running it.

**v2's fixes are structural, not cosmetic:** a `uv`-built **Python 3.11** venv so the pins actually
have wheels; LIBERO `git clone`d and installed **editable** so it is importable; `mujoco<3.2`; all
five tests in one `/content/gate.py` run as a subprocess under `/content/venv/bin/python`, because
a Colab cell cannot import from another interpreter's venv; resolved versions printed from inside
the venv before any test; and EGL with a one-shot OSMesa fallback.

**The gymnasium run refuted the original hypothesis.** On `Reacher-v5`, T2 passed: stepping does
not advance the env PRNG, so the next episode is independent of step count. T4 passed too: a
6-reset run and a 5-reset run share the same first five episodes. **The predicted failure did not
happen**, and confidence in the eval-nondeterminism paper dropped from MEDIUM-HIGH to LOW.

That result does not transfer, and reading LIBERO's source says why:

1. LIBERO never calls gymnasium's `reset(seed=)`. `ControlEnv.seed(s)` calls robosuite's
   `self.env.seed(s)`, and `reset()` takes no seed.
2. **robosuite 1.4's placement samplers draw from the GLOBAL `np.random`**, which any code in the
   process can advance. A per-env PRNG cannot be disturbed that way. Different failure mode.
3. 🔴 **`ControlEnv.reset()` retries on `RandomizationError` in a `while` loop**, so the number of
   RNG draws consumed per reset is *variable*. Two arms can consume different randomness per
   episode with no step-count difference at all.

`robosuite_gate.ipynb` adds **T5** for exactly this: burn 1000 `np.random.random()` calls between
resets and see whether the next episode changes. **Pre-registered prediction: T5 FAILS.** If it
does, that is the paper. If it passes, P6 is dead and gets recorded as dead.

**Version note:** LIBERO pins `numpy==1.22.4`, which has no wheel for Colab's Python; the notebook
uses `1.26.4`, the newest `numpy<2` that does, and records the deviation in its output JSON.
`robosuite==1.4.0` is required, not cosmetic — `ControlEnv` calls `suite.load_controller_config`,
removed in 1.5. mujoco is left to the resolver rather than pinned to 3.3.3, because robosuite 1.4
predates mujoco 3.x; the resolved version is recorded in the JSON.

## Status

⚠️ **Not executed.** The `chrome-devtools` MCP server was disconnected in the session that wrote
this, so Colab could not be driven. Verified instead, at primary:

- Every code cell parses — `eval_determinism_gate.ipynb` 14 cells, `robosuite_gate.ipynb` 17 cells.
- LIBERO's `requirements.txt` pins (`robosuite==1.4.0`, `numpy==1.22.4`, `gym==0.25.2`,
  `bddl==1.0.1`) and `ControlEnv.seed` / `.reset` / `.get_sim_state` were read at primary from
  `libero/libero/envs/env_wrapper.py`.
- `mujoco==3.3.3` exists and is installable (PyPI JSON API).
- The task BDDL file exists in LIBERO's `libero_object` suite.
- The Gymnasium sentence quoted above was fetched and confirmed at
  `https://gymnasium.farama.org/api/env/`.

**Not verified:** that the LIBERO install succeeds on current Colab, and that `OffScreenRenderEnv`
renders headless without an EGL/OSMesa flag. Expect that to break first; `MUJOCO_GL=egl` is the
usual fix.
