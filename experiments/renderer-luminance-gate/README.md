# Renderer luminance gate — MuJoCo 3.3.2 vs 3.3.3 on LIBERO-Object

**Run this before anything else in the renderer line of work. It is a gate, not a result.**

## What it tests

One question: **does the MuJoCo version change how bright LIBERO-Object renders the specific task
we trained on, by enough to matter?**

It does *not* test whether the renderer changes policy success. That is the next experiment, and
it is only worth building if this one passes.

## Why it exists

Two facts that only mean something together.

**1. LIBERO issue #88** (`Lifelong-Robot-Learning/LIBERO#88`) — open, **0 comments**, filed
2025-06-19 by `moojink`, who is Moo Jin Kim, first author of OpenVLA. Verbatim:

> *"MuJoCo's latest version release (3.3.3) may cause LIBERO-Object images to be rendered
> differently, which is a problem since it creates a distribution shift when testing policies
> trained on the original LIBERO data."*

He reports darker floors under 3.3.3 and reverts with `pip install mujoco==3.3.2`. Nobody has
quantified it in 15 months.

**2. Our own unexplained result.** Dimming the *training* video to ×0.30 beat the clean baseline
by **+18pp** (0.77 vs 0.59, `../../README.md`). If evaluation renders darker than the training
video, then dimming the training data moves training toward evaluation — and part of our flagship
result would be a renderer artifact rather than a fact about degradation.

**Neither of those is testable until we know the renderer delta is real on our task.** Hence this
gate.

## The pre-registered gate

Written down before the first run. Do not move it afterwards.

| Measure | Threshold to PROCEED |
|---|---|
| Mean absolute luminance delta, full agentview frame | **≥ 2.0 / 255** (0.8%) |
| Mean absolute luminance delta, lower-third (floor) region | **≥ 5.0 / 255** (2.0%) |

Either passing is enough. **Both failing means stop** — the renderer paper is dead and it cost a
day instead of six weeks. That is a good outcome, not a wasted one.

## Exact command

Open `renderer_luminance_gate.ipynb` in Colab, then:

```
PHASE = "A"        # run cells 1-3   → mujoco 3.3.2, renders 20 fixed states, saves to Drive
Runtime → Restart session
PHASE = "B"        # run cells 1-3   → mujoco 3.3.3, same 20 states
PHASE = "COMPARE"  # run cells 5-7   → the gate, the dependency diff, the plots
```

The restart is not optional. A live MuJoCo cannot be swapped under an imported module.

## Runtime and cost

| | |
|---|---|
| Hardware | **CPU only. No GPU.** Free-tier Colab is enough. |
| Wall clock | ~10–15 min total, most of it the LIBERO install (twice) |
| Cost | ≈ $0. Nothing is billed against the cloud credits. |

## What "done" looks like

`/content/drive/MyDrive/mhh-renderer-gate/` contains:

- `luminance_gate.csv` — the numbers, including `gate_passed` as an explicit boolean
- `frames_A.npy`, `frames_B.npy` — the raw rendered frames, so the analysis is re-runnable
- `freeze_A.txt`, `freeze_B.txt` — full `pip freeze` from both phases
- the three PNGs below

**Done also requires cell 6 printing `CLEAN: mujoco is the only difference.`** If other packages
moved between phases, the delta is not attributable to the renderer and the number is not usable
yet. This is the silent failure mode the whole design is built around: `pip install mujoco==X`
drags transitive dependencies, and a confounded luminance delta throws no error.

## The three plots the paper needs

1. **`plot1_side_by_side.png`** — the same initial state under 3.3.2 and 3.3.3, plus a signed
   difference map. This is the figure that makes the phenomenon visible in one glance, and it is
   the one Moo Jin Kim's issue is missing.
2. **`plot2_luma_hist.png`** — pixel-luminance distributions under both versions, overlaid.
   Shows whether the shift is a uniform offset or a change in the distribution's shape. A uniform
   offset and a gamma change imply different things about policy impact.
3. **`plot3_per_state.png`** — per-initial-state delta against the gate threshold. Shows whether
   the effect is consistent across scenes or driven by a few. Consistency is what licenses
   generalising from 20 states.

## Design choices worth knowing

**Fixed initial states, not seeded resets.** The notebook uses
`task_suite.get_task_init_states(task_id)` and `env.set_init_state()`, so the simulator state is
*set* rather than sampled. Any pixel difference is then the renderer and nothing else.

This matters beyond convenience. LIBERO's registered evaluation environment **never calls
`set_init_state()`** — `LiberoEnv.reset()` calls `env.seed()` then `env.reset()`, and the
canonical `get_task_init_states()` appears only in a `__main__` demo block. Only the *first* reset
of a rollout is seeded; later episodes come from gymnasium's unseeded autoreset, whose RNG
position depends on how many steps the policy took, so it **differs between arms**. A paired
comparison run on the stock eval path is therefore not trustworthy. That is a separate finding,
tracked in the vault; here it is simply avoided.

**Luminance is ITU-R BT.601** (`0.299R + 0.587G + 0.114B`) on raw 0–255 frames.

**"Floor" is approximated as the lower third of the frame**, because that is where issue #88
reports the change. It is a crop, not a segmentation, and is labelled that way in the notebook.

## If the gate passes

The next experiment is the 2×2 — {A_clean, B_dim} checkpoints × {3.3.2, 3.3.3} — testing whether
the *benefit of dimming* depends on the renderer version.

**Do not build it at three seeds.** The research council ran on 2026-09-12 fired its PLAUSIBLE
veto on exactly this: the interaction term is a difference of differences, the within-condition
spread is already 0.10 against a 0.18 main effect, and three seeds cannot resolve it. **≥5 seeds
per cell**, with the power calculation written down first. Full reasoning:
`mohanvault/01 Projects/Paper Choice 2026-09-12.md` §7.

## Status

⚠️ **The setup cells have NOT been executed on Colab.** The `chrome-devtools` MCP server was
disconnected in the session that wrote this, so Colab could not be driven. What *was* verified,
at primary:

- `mujoco==3.3.2` (uploaded 2025-04-28) and `mujoco==3.3.3` (2025-06-10) both exist on PyPI and
  are installable — checked against the PyPI JSON API.
- The task BDDL file `pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl` exists in
  LIBERO's `libero_object` suite — fetched from the repo.
- Every code cell parses (`ast.parse`, all cells clean).

**Not verified:** that the LIBERO install succeeds on current Colab, and that
`OffScreenRenderEnv` renders headless there without an EGL/OSMesa flag. Expect that to be the
first thing to break, and if it does, `MUJOCO_GL=egl` is the usual fix.
