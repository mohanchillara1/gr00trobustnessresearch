# GR00T N1.7 under degraded training video

**Training Data Visual Degradation and Object Transfer in GR00T N1.7: A Systematic Study**

Mohan Chillara, Wakeland High School, Frisco TX
Aahan Kumbham, Panther Creek High School, Frisco TX

## What this is

We fine-tuned NVIDIA GR00T N1.7 (3B) six separate times on the same 44 LIBERO-Object demonstrations of `pick_up_the_alphabet_soup_and_place_it_in_the_basket`, changing nothing except the visual quality of the training video: one clean baseline plus five degradations applied in post (brightness x0.30, a 30 degree rotation, Gaussian blur, all three at once, and a 50/50 mix of clean and blurred episodes). Action labels and proprioceptive state were never modified, so the only variable is what the policy sees during training. Each checkpoint then ran 150 evaluation episodes on the training object (3 trials of 50) and 50 episodes on each of four objects it had never been trained on. Two results came out of it. Dimming the training video to 30% brightness beat the clean baseline, 0.77 against 0.59 mean success rate. Rotating it 30 degrees produced 0.00 across all 150 episodes, though that condition has a padding confound described under Known limitations. Transfer to novel objects was near zero everywhere, which says single-object fine-tuning does not generalize regardless of how clean the video is.

Paper in preparation. This repository holds the pipeline, the raw evaluation logs, and the rollout videos.

## Results

Training object, `pick_up_the_alphabet_soup_and_place_it_in_the_basket`. T1 to T3 are three independent 50-episode evaluation runs of the same checkpoint. Raw terminal output is in `Raw Data/`.

| Condition | T1 | T2 | T3 | Mean SR | OTS |
| --- | --- | --- | --- | --- | --- |
| A_clean (baseline) | 0.66 | 0.56 | 0.56 | 0.59 | 0.017 |
| B_dim (brightness x0.30) | 0.72 | 0.82 | 0.78 | 0.77 | 0.013 |
| C_rotation (30 deg) | 0.00 | 0.00 | 0.00 | 0.00 | not evaluated |
| D_blur (Gaussian, k=11) | 0.38 | 0.32 | 0.22 | 0.31 | 0.000 |
| E_combined (dim + rotation + blur) | 0.00 | 0.00 | 0.00 | 0.00 | not evaluated |
| F_diverse (50/50 clean and blurred) | 0.68 | 0.80 | 0.72 | 0.73 | 0.021 |

C_rotation and E_combined were not carried into the novel-object stage because a training-object success rate of 0.00 leaves the Object Transfer Score undefined.

Novel objects, 50 episodes per cell, one run each:

| Novel object | A_clean | B_dim | D_blur | F_diverse |
| --- | --- | --- | --- | --- |
| tomato_sauce | 0.00 | 0.00 | 0.00 | 0.00 |
| bbq_sauce | 0.00 | 0.00 | 0.00 | 0.00 |
| ketchup | 0.04 | 0.04 | 0.00 | 0.00 |
| milk | 0.00 | 0.00 | 0.00 | 0.06 |

Object Transfer Score is defined as the mean novel-object success rate divided by the training-object success rate for the same checkpoint. It is meant to separate "this policy transfers" from "this policy is good at its one task." Every value here is under 0.03, so on this task the answer is that none of them transfer.

## Known limitations

Read this before drawing conclusions from the table above.

**The rotation result is confounded by padding.** `apply_rotation()` in `convert_and_degrade.py` calls `cv2.warpAffine` with `borderMode=cv2.BORDER_REFLECT`. Rotating a square frame by 30 degrees empties the corners, and BORDER_REFLECT fills them by mirroring the scene back in, so a condition C frame contains duplicated table and object pixels that no camera would ever produce. Condition C therefore varies two things at once, viewpoint and image content, and the collapse to 0.00 may be partly an artifact of the mirrored padding rather than the rotation. Rerunning condition C with `cv2.BORDER_CONSTANT` is a one-line change and is planned. Until that is done, treat the rotation result as unresolved.

**Three trials, no significance test.** The per-trial numbers above are everything we have. The 0.77 against 0.59 gap between B_dim and A_clean is larger than the spread within either condition, but three runs per condition is not enough for a statistical claim, and none was computed. The novel-object numbers are worse off: one 50-episode run per cell, no repeats, so a 0.04 and a 0.00 are not meaningfully different.

**The converter in this repo is the single-camera version.** `convert_and_degrade.py` writes only `observation.images.image` and declares only that key in `modality.json`. GR00T N1.7 needs the wrist stream as well, which is Bug 2 below. The fix is described but is not applied in the committed file, so this script as it stands reproduces the broken configuration, not the one that produced the results.

**One task, one training object, one seed per condition.** Every number comes from a single LIBERO task with a single fine-tuning run per condition. Nothing here separates a real effect from run-to-run variance in fine-tuning itself.

**The degradations are synthetic post-processing, not real camera conditions.** A uniform 0.30 brightness multiplier is not the same thing as filming in a dim room, where sensor noise and exposure behavior both change.

## Repository contents

```text
convert_and_degrade.py           TFRecord to LeRobot conversion + the 6 degradation conditions
evaluationScriptForNovelObj.sh   Novel-object rollouts across the 4 eligible checkpoints
novelobjdata.txt                 Novel-object success rates and the OTS values
Raw Data/                        Unedited terminal output behind every number in this README
Degradation Visualization/       One sample frame per condition, main and wrist camera
Training Videos/                 Rollout recordings, 7 failures and 1 success
Failed Replication Videos/       Hand-recorded videos from the abandoned first approach
LICENSE
README.md
```

`Failed Replication Videos/` is from an earlier version of the project that did not work. We first tried to have GR00T replicate motions from hand-recorded human video, but LAPA could not produce usable action tokens from those clips, so the project moved into LIBERO simulation. `READMEFAILEDVID.txt` in that folder has the details.

## Environment setup

Runs on RunPod or any Ubuntu machine with an NVIDIA A40 (48 GB VRAM). On RunPod, use the PyTorch 2.1 template. PyTorch 2.8 does not work.

```bash
# 1. Clone Isaac-GR00T
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T && bash scripts/deployment/dgpu/install_deps.sh
export GR00T_ROOT=$PWD

# 2. Activate environment
source $HOME/.local/bin/env && source .venv/bin/activate
uv pip install -e . --no-deps
uv pip install torch==2.7.1+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. System dependencies
apt install -y libnpp-12-8 cuda-nvcc-12-8 libegl1-mesa-dev cmake git-lfs
bash "$GR00T_ROOT/gr00t/eval/sim/LIBERO/setup_libero.sh"
```

Four config patches have to be applied before any training. Their file locations depend on which Isaac-GR00T revision you cloned, so resolve them by filename rather than hard-coding a path:

```bash
TRAIN_CFG=$(find "$GR00T_ROOT" -name training_config.py   | head -1)
EMB_CFG=$(find   "$GR00T_ROOT" -name embodiment_configs.py | head -1)
FT_CFG=$(find    "$GR00T_ROOT" -name finetune_config.py    | head -1)

sed -i 's/bf16: bool = True/bf16: bool = False/'                                "$TRAIN_CFG"
sed -i 's/tf32: bool = True/tf32: bool = False/'                                "$TRAIN_CFG"
sed -i 's/modality_keys=\["image"\]/modality_keys=["image","wrist_image"]/'     "$EMB_CFG"
sed -i 's/save_only_model: bool = False/save_only_model: bool = True/'          "$FT_CFG"
```

Re-export these in every new shell:

```bash
export PYTHONPATH=$GR00T_ROOT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
```

## Running the pipeline

### Step 1: convert and degrade the dataset

Download the source dataset first:

```bash
python3 -c "import tensorflow_datasets as tfds; tfds.load('libero_object_no_noops', data_dir='/workspace/datasets/libero')"
```

Then:

```bash
python3 convert_and_degrade.py
```

This reads the 44 alphabet soup episodes and writes six dataset folders under `/workspace/datasets/`, one per condition, plus one spot-check PNG per condition in `/workspace/spotcheck/` so you can confirm the degradations look the way you expect before spending GPU hours. Paths are set at the top of the script.

### Step 2: fine-tune each condition

```bash
for COND in A_clean B_dim C_rotation D_blur E_combined F_diverse; do
    python scripts/gr00t_finetune.py \
        --dataset-path /workspace/datasets/${COND} \
        --output-dir /workspace/checkpoints/${COND}_v2 \
        --embodiment-tag libero_sim \
        --num-steps 2000 \
        --batch-size 8
done
```

AdamW, lr 1e-4, cosine annealing, weight decay 1e-4, gradient clipping norm 1.0.

### Step 3: evaluate on the training object

```bash
for COND in A_clean B_dim C_rotation D_blur E_combined F_diverse; do
    CUDA_VISIBLE_DEVICES=0 python gr00t/eval/run_gr00t_server.py \
        --model-path /workspace/checkpoints/${COND}_v2/checkpoint-2000 \
        --embodiment-tag libero_sim \
        --use-sim-policy-wrapper &
    SERVER_PID=$!

    sleep 120   # the checkpoint shards need this long to load

    python gr00t/eval/rollout_policy.py \
        --n-episodes 50 \
        --env-name "libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket" \
        --n-action-steps 8 \
        --n-envs 1

    kill $SERVER_PID 2>/dev/null
    sleep 20
done
```

Run this three times to reproduce the T1 to T3 columns.

### Step 4: evaluate novel-object transfer

```bash
bash evaluationScriptForNovelObj.sh
```

Runs A_clean, B_dim, D_blur and F_diverse against tomato_sauce, bbq_sauce, ketchup and milk, and appends everything to `/workspace/novel_obj_results.txt`.

## Two bugs to check for first

Both of these produce 0% success on every condition, including the clean baseline, so they look like a modeling failure when they are a data plumbing failure. Worth ruling out before anything else if you are wiring up the same pipeline.

### Bug 1: inverted gripper action convention

LIBERO TFRecords store gripper values in [-1, +1]. The evaluation pipeline runs `normalize_gripper_action` then `invert_gripper_action` and expects [0, 1] going in. Without a fix, every grasp command is reversed: the gripper opens when it should close.

```python
# Apply at data conversion time, in convert_and_degrade.py
def fix_gripper(actions):
    actions = actions.copy()
    actions[..., -1] = (-actions[..., -1] + 1.0) / 2.0
    return actions

# Closed (-1.0): (-(-1)+1)/2 = 1.0 -> normalize -> +1 -> invert -> -1, closes
# Open   (+1.0): (-(+1)+1)/2 = 0.0 -> normalize -> -1 -> invert -> +1, opens
```

### Bug 2: missing wrist camera

The first conversion pass extracted only the main camera. GR00T N1.7 needs both streams. The fix is to extract `observation.images.wrist_image` during conversion and declare both video keys in `modality.json`. As noted under Known limitations, the converter committed here is still the single-camera version.

## Citation

```bibtex
@misc{chillara2026groot,
  author = {Chillara, Mohan and Kumbham, Aahan},
  title  = {Training Data Visual Degradation and Object Transfer in {GR00T} {N1.7}: A Systematic Study},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## Authors and contributions

**Mohan Chillara**, Wakeland High School, Frisco TX. Conditions A through C, the fine-tuning pipeline, the LIBERO training-object evaluation, and the introduction, hypotheses, discussion and conclusion of the paper.

**Aahan Kumbham**, Panther Creek High School, Frisco TX. Conditions D through F, the LIBERO-PRO novel-object transfer evaluation, the Object Transfer Score computation, and the figures.

## Acknowledgments

We thank Dr. Aravind Chiruvelli, Sr. Director of Data Science at Walmart Global Technology, who conceived the object-transfer question this study is built around, reviewed the experimental design, shaped the discussion, and gave final approval. He asked to be acknowledged rather than listed as an author.

We thank Dr. Alice E. Smith of Auburn University, NAE member and IEEE Life Fellow, for reviewing the methodology.

We also thank NVIDIA for open-sourcing the Isaac-GR00T framework and the GR00T N1.7 weights, and the LIBERO team at UT Austin for the benchmark and dataset. Compute on RunPod.

## License

MIT. See [LICENSE](LICENSE).
