#!/usr/bin/env python3
"""
GR00T Research — TFRecord to LeRobot Conversion + Visual Degradation Pipeline
==============================================================================
Reads LIBERO alphabet_soup episodes from TFRecord format.
Converts to LeRobot parquet + mp4 format that GR00T can train on.
Applies 5 visual degradations to the video stream, plus a clean baseline.
Action labels and state data are NEVER modified.

Output: 6 dataset folders at /workspace/datasets/<condition>/
  A_clean      no modification (baseline)
  B_dim        brightness x 0.30
  C_rotation   30 degree affine rotation
  D_blur       Gaussian blur kernel=11
  E_combined   dim + rotation + blur together
  F_diverse    50% clean + 50% blurred per episode

Usage:
  python3 convert_and_degrade.py

KNOWN ISSUES with this file, see the "Known limitations" section of README.md:
  1. apply_rotation() pads with cv2.BORDER_REFLECT, so a rotated frame carries
     mirrored copies of the scene in its corners. Condition C therefore changes
     two things at once, viewpoint and image content, and the 0.00 success rate
     it produces cannot be attributed to the rotation alone. A BORDER_CONSTANT
     rerun is planned.
  2. This is the single-camera version of the converter. It writes only
     observation.images.image. GR00T N1.7 also needs the wrist stream, and the
     fix for that (Bug 2 in the README) is not applied here.

Authors:
  Mohan Chillara  (Wakeland High School, Frisco TX)
  Aahan Kumbham   (Panther Creek High School, Frisco TX)
"""

import os
import json
import random
import shutil
import numpy as np
import pandas as pd
import cv2
import subprocess
import tempfile
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

LIBERO_DATA_DIR = "/workspace/datasets/libero"
OUTPUT_BASE     = "/workspace/datasets"
TASK_NAME       = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
FPS             = 20
IMAGE_SIZE      = 256  # 256x256 from features.json
CHUNKS_SIZE     = 1000

CONDITIONS = {
    "A_clean":    "clean",
    "B_dim":      "dim",
    "C_rotation": "rotation",
    "D_blur":     "blur",
    "E_combined": "combined",
    "F_diverse":  "diverse",
}

# ── Degradation Functions ─────────────────────────────────────────────────────

def apply_clean(frame):
    """Condition A — no modification."""
    return frame

def apply_dim(frame):
    """Condition B — brightness x 0.30."""
    return (frame.astype(np.float32) * 0.30).clip(0, 255).astype(np.uint8)

def apply_rotation(frame):
    """
    Condition C: 30 degree affine rotation.

    CONFOUND: BORDER_REFLECT mirrors the scene into the corners the rotation
    empties out, so the frame gains duplicated table and object pixels on top
    of the viewpoint change. Switching this to cv2.BORDER_CONSTANT is the
    one-line change needed to separate the two effects. See README.md.
    """
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

def apply_blur(frame):
    """Condition D — Gaussian blur kernel=11."""
    return cv2.GaussianBlur(frame, (11, 11), 0)

def apply_combined(frame):
    """Condition E — dim + rotation + blur."""
    frame = apply_dim(frame)
    frame = apply_rotation(frame)
    frame = apply_blur(frame)
    return frame

def get_degradation_fn(condition, episode_idx=None):
    """Return the degradation function for a given condition."""
    if condition == "clean":
        return apply_clean
    elif condition == "dim":
        return apply_dim
    elif condition == "rotation":
        return apply_rotation
    elif condition == "blur":
        return apply_blur
    elif condition == "combined":
        return apply_combined
    elif condition == "diverse":
        # 50% clean, 50% blurred — determined per episode
        if episode_idx is not None and episode_idx % 2 == 0:
            return apply_blur
        return apply_clean
    else:
        raise ValueError(f"Unknown condition: {condition}")

# ── Video Writing ─────────────────────────────────────────────────────────────

def frames_to_mp4(frames, output_path, fps=FPS):
    """
    Write a list of numpy frames (H, W, 3) RGB to an mp4 file.
    Uses H.264 encoding compatible with torchcodec.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = frames[0].shape[:2]

    # Write to temp file then move — avoids partial writes
    tmp_path = output_path + ".tmp.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    for frame in frames:
        # OpenCV expects BGR
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()

    # Re-encode with ffmpeg for torchcodec compatibility (H.264 in MP4 container)
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-loglevel", "error",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  WARNING: ffmpeg failed, using raw mp4v: {result.stderr.decode()}")
        shutil.move(tmp_path, output_path)
    else:
        os.remove(tmp_path)

# ── Dataset Structure Creation ────────────────────────────────────────────────

def create_dataset_structure(base_path):
    """Create the LeRobot folder structure."""
    for folder in [
        "meta",
        "videos/chunk-000/observation.images.image",
        "data/chunk-000",
    ]:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)

def write_meta_files(base_path, episodes_meta, task_description):
    """Write meta/episodes.jsonl, meta/tasks.jsonl, meta/info.json, meta/modality.json."""

    # tasks.jsonl
    with open(os.path.join(base_path, "meta/tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_description}) + "\n")

    # episodes.jsonl
    with open(os.path.join(base_path, "meta/episodes.jsonl"), "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    total_frames = sum(ep["length"] for ep in episodes_meta)

    # info.json
    info = {
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_0", "gripper_1"]
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            },
            "observation.images.image": {
                "dtype": "video",
                "shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
                "names": ["height", "width", "channel"]
            }
        }
    }
    with open(os.path.join(base_path, "meta/info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # modality.json — single camera (image only, no wrist_image)
    modality = {
        "state": {
            "x":       {"start": 0, "end": 1},
            "y":       {"start": 1, "end": 2},
            "z":       {"start": 2, "end": 3},
            "roll":    {"start": 3, "end": 4},
            "pitch":   {"start": 4, "end": 5},
            "yaw":     {"start": 5, "end": 6},
            "gripper": {"start": 6, "end": 8}
        },
        "action": {
            "x":       {"start": 0, "end": 1},
            "y":       {"start": 1, "end": 2},
            "z":       {"start": 2, "end": 3},
            "roll":    {"start": 3, "end": 4},
            "pitch":   {"start": 4, "end": 5},
            "yaw":     {"start": 5, "end": 6},
            "gripper": {"start": 6, "end": 7}
        },
        "video": {
            "image": {"original_key": "observation.images.image"}
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"}
        }
    }
    with open(os.path.join(base_path, "meta/modality.json"), "w") as f:
        json.dump(modality, f, indent=2)

# ── Main Conversion + Degradation Pipeline ────────────────────────────────────

def load_episodes_from_tfrecord():
    """
    Load all alphabet_soup episodes from TFRecord.
    Returns list of dicts with keys: frames_image, actions, states
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    print("Loading TFRecord dataset...")
    ds = tfds.load(
        "libero_object_no_noops",
        data_dir=LIBERO_DATA_DIR,
        split="train"
    )

    episodes = []
    for ep in ds:
        path = ep["episode_metadata"]["file_path"].numpy().decode()
        task = path.split("/")[-1].replace("_demo.hdf5", "")
        if task != TASK_NAME:
            continue

        steps = ep["steps"]

        # Extract all frames and actions
        frames_image  = []
        actions       = []
        states        = []

        for step in steps:
            obs     = step["observation"]
            img     = obs["image"].numpy()          # (256, 256, 3) uint8 RGB
            action  = step["action"].numpy()        # (7,) float32
            # Use eef_state (8D) as observation state
            state   = obs["state"].numpy()  # (8,) float32

            frames_image.append(img)
            actions.append(action)
            states.append(state)

        episodes.append({
            "frames_image": frames_image,
            "actions":      np.array(actions,  dtype=np.float32),
            "states":       np.array(states,   dtype=np.float32),
        })

        print(f"  Loaded episode {len(episodes):02d} — {len(frames_image)} frames")

    print(f"\nTotal episodes loaded: {len(episodes)}")
    return episodes

def build_condition_dataset(episodes, condition_name, condition_type, output_path):
    """
    Build one complete dataset folder for a given condition.
    episodes     — list of raw episode dicts from load_episodes_from_tfrecord()
    condition_name — e.g. "A_clean"
    condition_type — e.g. "clean"
    output_path  — e.g. /workspace/datasets/A_clean
    """
    print(f"\n{'='*60}")
    print(f"Building condition: {condition_name} ({condition_type})")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    create_dataset_structure(output_path)

    episodes_meta = []
    global_frame_idx = 0

    for ep_idx, ep in enumerate(episodes):
        frames_image = ep["frames_image"]
        actions      = ep["actions"]
        states       = ep["states"]
        n_frames     = len(frames_image)

        # Get degradation function for this condition + episode
        degrade_fn = get_degradation_fn(condition_type, episode_idx=ep_idx)

        # Apply degradation to all frames
        degraded_frames = [degrade_fn(f) for f in frames_image]

        # Write video
        video_path = os.path.join(
            output_path,
            f"videos/chunk-000/observation.images.image/episode_{ep_idx:06d}.mp4"
        )
        print(f"  Episode {ep_idx:02d}: writing {n_frames} frames → {os.path.basename(video_path)}")
        frames_to_mp4(degraded_frames, video_path)

        # Write parquet
        rows = []
        for frame_idx in range(n_frames):
            rows.append({
                "observation.state": states[frame_idx].tolist(),
                "action":            actions[frame_idx].tolist(),
                "timestamp":         frame_idx / FPS,
                "frame_index":       frame_idx,
                "episode_index":     ep_idx,
                "index":             global_frame_idx,
                "task_index":        0,
            })
            global_frame_idx += 1

        df = pd.DataFrame(rows)
        parquet_path = os.path.join(
            output_path,
            f"data/chunk-000/episode_{ep_idx:06d}.parquet"
        )
        df.to_parquet(parquet_path, index=False)

        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [TASK_NAME.replace("_", " ")],
            "length": n_frames,
        })

    # Write meta files
    write_meta_files(output_path, episodes_meta, TASK_NAME.replace("_", " "))

    print(f"  ✓ Done — {len(episodes)} episodes, {global_frame_idx} total frames")

def save_spotcheck_frames(episodes, output_dir="/workspace/spotcheck"):
    """
    Save one sample frame per condition as PNG for visual verification.
    Uses episode 0, frame 10 from each condition dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_frame = episodes[0]["frames_image"][10]  # episode 0, frame 10

    for cond_name, cond_type in CONDITIONS.items():
        degrade_fn = get_degradation_fn(cond_type, episode_idx=0)
        degraded = degrade_fn(sample_frame.copy())
        out_path = os.path.join(output_dir, f"{cond_name}.png")
        cv2.imwrite(out_path, cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR))
        print(f"  Spotcheck saved: {out_path}")

# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    print("GR00T Research — Convert + Degrade Pipeline")
    print(f"Task: {TASK_NAME}")
    print(f"Output base: {OUTPUT_BASE}")
    print()

    # Step 1: Load all episodes once from TFRecord
    episodes = load_episodes_from_tfrecord()

    if len(episodes) == 0:
        print("ERROR: No episodes found for task. Check TASK_NAME and data path.")
        return

    # Step 2: Save spotcheck frames
    print("\nSaving spotcheck frames...")
    save_spotcheck_frames(episodes)

    # Step 3: Build all 6 condition datasets
    for cond_name, cond_type in CONDITIONS.items():
        output_path = os.path.join(OUTPUT_BASE, cond_name)
        build_condition_dataset(episodes, cond_name, cond_type, output_path)

    print("\n" + "="*60)
    print("ALL CONDITIONS COMPLETE")
    print("="*60)
    print("\nDataset folders:")
    for cond_name in CONDITIONS:
        path = os.path.join(OUTPUT_BASE, cond_name)
        print(f"  {path}")
    print("\nSpotcheck frames: /workspace/spotcheck/")
    print("\nNext step: verify spotcheck PNGs look correct, then run pilot training.")

if __name__ == "__main__":
    main()
