# ---- STAGE 2 PATCH v3, 2026-09-28 (= v2 + CPU fallback when Colab refuses a GPU; device recorded per episode)
# ---- was: STAGE 2 PATCH v2, 2026-09-28 (gpu-runs worker, at dad's word via director) ----
# Identical to p6_stage2_patch_2026-09-27.py in everything that touches the experiment (loader, processors,
# obs conventions, warm-up, clipping, arm A / arm B reset logic, init-state hash, pre-registered constants).
# Adds only persistence, because two Colab runs on 09-27 lost every number when the VM went away:
#   1. per-episode JSON written to P6_PERSIST (Drive or a pod volume) after EVERY episode, plus one printed line;
#   2. resume: arm B resumes at the first missing seed (each episode is seeded on its own). Arm A resumes by
#      REPLAYING its resets + burn draws without rollouts and asserting each replayed init_hash equals the
#      recorded one; any mismatch discards arm A's saved episodes and restarts arm A from episode 0.
import numpy as np, torch, time, json, os
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

P6_PERSIST = os.environ.get("P6_PERSIST") or ("/content/drive/MyDrive/p6_runs" if os.path.isdir("/content/drive/MyDrive") else "/content/p6_runs")
os.makedirs(P6_PERSIST, exist_ok=True)
P6_PLATFORM = os.environ.get("P6_PLATFORM", "colab")
_ckpt = os.path.join(P6_PERSIST, "p6_stage2_episodes.json")

P6_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_raw = SmolVLAPolicy.from_pretrained(POLICY_REPO)
_raw.config.device = P6_DEVICE
_raw = _raw.to(P6_DEVICE).eval()
try:
    _pre, _post = make_pre_post_processors(_raw.config, pretrained_path=POLICY_REPO,
        preprocessor_overrides={"device_processor": {"device": P6_DEVICE}})
except Exception as _e:
    print("processor override failed, default processors:", repr(_e)[:200], flush=True)
    _pre, _post = make_pre_post_processors(_raw.config, pretrained_path=POLICY_REPO)
print("policy device:", P6_DEVICE, flush=True)

class _Wrapped:
    def reset(self):
        _raw.reset()
    def select_action(self, batch):
        with torch.no_grad():
            return _post(_raw.select_action(_pre(batch)))

policy, POLICY_OK, pol_err = _Wrapped(), True, None
print("policy loaded (patched loader v2):", POLICY_REPO, "| persist ->", P6_PERSIST, flush=True)

def _q2aa(q):
    x, y, z, w = [float(v) for v in q]
    w = max(-1.0, min(1.0, w))
    den = (1.0 - w * w) ** 0.5
    if den < 1e-10:
        return np.zeros(3)
    return np.array([x, y, z]) / den * 2.0 * np.arccos(w)

def obs_to_policy(obs, device="cuda"):
    img = torch.from_numpy(obs["agentview_image"][::-1, ::-1].copy()).permute(2, 0, 1)[None].float() / 255.
    img2 = torch.from_numpy(obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()).permute(2, 0, 1)[None].float() / 255.
    st = np.concatenate([obs["robot0_eef_pos"], _q2aa(obs["robot0_eef_quat"]),
                         obs["robot0_gripper_qpos"]]).astype(np.float32)
    return {"observation.images.image": img, "observation.images.image2": img2,
            "observation.state": torch.from_numpy(st)[None], "task": [lang]}

NUM_STEPS_WAIT = 10
DUMMY = [0, 0, 0, 0, 0, 0, -1]

_state = {"A": [], "B": []}
if os.path.exists(_ckpt):
    _prev = json.load(open(_ckpt))
    _state = {"A": _prev.get("armA", []), "B": _prev.get("armB", [])}
    print("resume: found %d arm-A and %d arm-B episodes in %s" % (len(_state["A"]), len(_state["B"]), _ckpt), flush=True)

def _save():
    tmp = _ckpt + ".tmp"
    json.dump({"prereg": PREREG, "versions": VERS, "stage1": stage1, "platform": P6_PLATFORM,
               "stage2_patched": "v3 2026-09-28 (= 09-27 patch + persistence/resume + cpu fallback)",
               "armA": _state["A"], "armB": _state["B"], "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(tmp, "w"), indent=2)
    os.replace(tmp, _ckpt)

def rollout(arm):
    env, task = make_env(SUITE, TASK_ID)
    results = _state[arm]
    try:
        if arm == "A":
            np.random.seed(SEEDS[0]); env.seed(SEEDS[0])
            ok = True
            for j, r in enumerate(list(results)):          # replay without rollouts, verify hashes
                env.reset()
                if state_hash(env) != r["init_hash"]:
                    ok = False
                    print("resume arm A: replay hash mismatch at episode %d -> restarting arm A" % j, flush=True)
                    break
                _ = np.random.rand(BURN_DRAWS)
            if not ok:
                env.close(); env, task = make_env(SUITE, TASK_ID)
                results.clear(); _state["A"] = results
                np.random.seed(SEEDS[0]); env.seed(SEEDS[0])
        for i, s in enumerate(SEEDS):
            if i < len(results):
                continue
            if arm == "B":
                np.random.seed(s); env.seed(s)
            obs = env.reset()
            h0 = state_hash(env)
            policy.reset()
            for _ in range(NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(DUMMY)
            done, steps, success = False, 0, False
            te = time.time()
            while not done and steps < MAX_STEPS:
                a = policy.select_action(obs_to_policy(obs)).cpu().numpy().flatten()[:7]
                obs, _, done, info = env.step(np.clip(a, -1.0, 1.0).tolist())
                steps += 1
                success = bool(env.check_success())
                if success:
                    break
            results.append({"i": i, "seed": s, "init_hash": h0, "success": success, "steps": steps,
                            "sec": round(time.time() - te, 1), "device": P6_DEVICE})
            _save()
            print("EP arm=%s i=%d seed=%s init_hash=%s success=%s steps=%d sec=%.0f dev=%s | arm %s so far %d/%d" % (
                arm, i, s, h0, success, steps, time.time() - te, P6_DEVICE, arm, sum(r["success"] for r in results), len(results)), flush=True)
            if arm == "A":
                _ = np.random.rand(BURN_DRAWS)
    finally:
        env.close()
    return results

t0 = time.time()
rA = rollout("A"); rB = rollout("B")
sA = sum(r["success"] for r in rA); sB = sum(r["success"] for r in rB)
paired_agree = sum(1 for a, b in zip(rA, rB) if a["success"] == b["success"])
stage2 = {
    "armA_successes": sA, "armB_successes": sB, "n": len(SEEDS),
    "armA_rate": sA / len(SEEDS), "armB_rate": sB / len(SEEDS),
    "delta_pp": (sB - sA) / len(SEEDS) * 100.0,
    "paired_agreement": paired_agree, "paired_agreement_frac": paired_agree / len(SEEDS),
    "init_hash_overlap_stage2": len(set(r["init_hash"] for r in rA) & set(r["init_hash"] for r in rB)),
    "hours_this_session": round((time.time() - t0) / 3600, 2), "platform": P6_PLATFORM,
    "harness_patch": "v2 2026-09-28: 09-27 patch (lerobot 0.6.1 loader + checkpoint processors + 8-dim state + HW flip + 10-step warmup) + persistence/resume",
    "armA": rA, "armB": rB,
}
json.dump({"prereg": PREREG, "versions": VERS, "stage1": stage1, "stage2": stage2},
          open(os.path.join(P6_PERSIST, "p6_magnitude_result.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in stage2.items() if k not in ("armA", "armB")}, indent=2))
