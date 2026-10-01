"""Build a chunk-2 notebook (P6 extension, prereg PREREG_p6_extend_2026-09-29.md, 1acc963) from the chunk-1 notebook.

usage: python3 make_p6_extend_c2.py ARM START STAGE1_END BUDGET_MIN DATA_SHA TAG
  ARM          A or B
  START        first new episode index (= episodes already committed for this arm)
  STAGE1_END   last index this version may roll out (Stage 1 hashes are computed up to here)
  BUDGET_MIN   wall-clock budget from kernel start; no new episode starts after it (so a quota-limited
               session ends cleanly instead of being killed)
  DATA_SHA     commit holding results/p6_extend_resume_arm{ARM}.json (episodes 0..START-1)
  TAG          short label in file names, e.g. c2a

Experiment logic is unchanged from chunk 1 (seeds, arm A leaky chain + burn draws, arm B per-episode reseed, loader,
processors, warm-up, clipping, MAX_STEPS, hash function). Changes are bookkeeping only:
  1. resume file = committed episodes 0..START-1 (arm A replays them and ABORTS on any hash mismatch);
  2. Stage 1 runs only this arm: arm A the chain 0..STAGE1_END, arm B only indices START..STAGE1_END (each is seeded
     on its own, so the values are the same as in a full loop);
  3. HASHCHECK covers the new episodes (old ones are covered by the arm-A replay / their original runs);
  4. a time budget stops starting new episodes, so the session ends cleanly before the quota does.
"""
import json, sys, ast

arm, start, s1end, budget, sha, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
assert arm in "AB" and 200 <= start <= s1end <= 299
src_nb = "p6_extend_c1_arm%s_2026-09-29.ipynb" % arm
nb = json.load(open(src_nb))
c = nb["cells"]
S = lambda k: "".join(c[k]["source"])
def put(k, s): c[k]["source"] = s.splitlines(True)
def rep(s, old, new):
    assert old in s, old[:80]
    return s.replace(old, new)

DATA = ("https://raw.githubusercontent.com/mohanchillara1/gr00trobustnessresearch/%s/experiments/eval-determinism-gate/"
        "results/p6_extend_resume_arm%s.json" % (sha, arm))

put(2, f"""## EXTEND chunk 2 ({tag}): arm {arm}, new episodes i = {start}–{s1end} at most, time budget {budget} min

Pre-registration: `PREREG_p6_extend_2026-09-29.md` (`1acc963`). Built by `make_p6_extend_c2.py` from
`p6_extend_c1_arm{arm}_2026-09-29.ipynb`. Bookkeeping changes only (see the generator's docstring): resume from the
committed episodes 0–{start - 1} (`{sha[:7]}`), Stage 1 for this arm only, HASHCHECK on new episodes, and a time budget
so a quota-limited session stops cleanly. Seeds, arms, policy, harness: unchanged.
""")

s = S(3)
s = rep(s, 'N_EPISODES = 200          # EXTEND chunk 1: Stage 1 over seeds 10000..10199; rollouts i=100..199 only',
        f'N_EPISODES = {s1end + 1}          # EXTEND chunk 2: Stage 1 / rollouts up to i={s1end}\n'
        f'C2_START = {start}; C2_BUDGET_S = {budget * 60}; C2_TAG = "{tag}"\n'
        'import time as _t; C2_T0 = _t.time()')
s = rep(s, 'print("EXTEND C1 RUN:', 'print("EXTEND C2 RUN [%s] start=%d budget_min=%d:" % (C2_TAG, C2_START, C2_BUDGET_S // 60), "')
i = s.index("# ---- R0-EXTEND")
s = s[:i] + f'''# ---- R0-EXTEND C2: seed the resume file with the committed episodes 0..C2_START-1 of this arm ----
import urllib.request, hashlib as _hl
_url = "{DATA}"
_raw = urllib.request.urlopen(_url, timeout=120).read()
print("committed episodes file sha256:", _hl.sha256(_raw).hexdigest(), "bytes", len(_raw), flush=True)
_prev = json.loads(_raw)
_prev_eps = _prev["arm%s" % P6_ARM]
assert [r["i"] for r in _prev_eps] == list(range(C2_START)), "committed file is not episodes 0..C2_START-1"
json.dump({{"armA": _prev_eps if P6_ARM == "A" else [], "armB": _prev_eps if P6_ARM == "B" else []}},
          open(os.path.join(os.environ["P6_PERSIST"], "p6_stage2_episodes.json"), "w"))
COMMITTED_N = len(_prev_eps)
print("resume file seeded with %d committed arm-%s episodes (successes %d)" % (COMMITTED_N, P6_ARM, sum(r["success"] for r in _prev_eps)), flush=True)
'''
put(3, s)

# Stage 1: this arm only.
s = S(12)
s = rep(s, '''        for i, s in enumerate(SEEDS):
            if arm == "B":
                np.random.seed(s); env.seed(s)''', '''        for i, s in enumerate(SEEDS):
            if arm == "B" and i < C2_START:
                hashes.append(None); continue          # C2: arm B indices are independently seeded
            if arm == "B":
                np.random.seed(s); env.seed(s)''')
s = rep(s, '''hA, lang = init_states("A")
hB, _    = init_states("B")
overlap  = len(set(hA) & set(hB))''', '''hX, lang = init_states(P6_ARM)                 # C2: this arm only
hA = hX if P6_ARM == "A" else []
hB = hX if P6_ARM == "B" else []
overlap  = None''')
s = rep(s, '"overlap": overlap, "armA_distinct": len(set(hA)), "armB_distinct": len(set(hB)),',
        '"overlap": overlap, "armA_distinct": len(set(hA)), "armB_distinct": len(set(hB)), "c2_arm_only": P6_ARM,')
s = rep(s, 'VERDICT_S1 = "LEAK CONFIRMED (arms disjoint)" if overlap == 0 else "INCONCLUSIVE — arms overlap; Stage 2 is not interpretable"',
        'VERDICT_S1 = "C2: single-arm Stage 1 (reference hashes for HASHCHECK only; overlap not computed)"')
put(12, s)

s = S(17)
s = rep(s, '''        for i, s in enumerate(SEEDS):
            if i < len(results):
                continue''', '''        for i, s in enumerate(SEEDS):
            if i < len(results):
                continue
            if time.time() - C2_T0 > C2_BUDGET_S:
                print("TIME BUDGET reached (%.0f min): stopping before i=%d" % ((time.time() - C2_T0) / 60, i), flush=True)
                break''')
s = rep(s, '_mis = [r["i"] for r in rX if r["init_hash"] != _ref[r["i"]]]',
        '_mis = [r["i"] for r in rX if r["i"] >= COMMITTED_N and r["init_hash"] != _ref[r["i"]]]')
s = rep(s, 'print("HASHCHECK arm %s: %d/%d episodes match Stage 1 init_hash; mismatches at %s" % (P6_ARM, len(rX) - len(_mis), len(rX), _mis), flush=True)',
        '_nnew = sum(1 for r in rX if r["i"] >= COMMITTED_N)\n'
        'print("HASHCHECK arm %s (new episodes): %d/%d match Stage 1 init_hash; mismatches at %s" % (P6_ARM, _nnew - len(_mis), _nnew, _mis), flush=True)')
s = rep(s, '"p6_extend_c1_arm%s_result.json" % P6_ARM', '"p6_extend_%s_arm%s_result.json" % (C2_TAG, P6_ARM)')
s = rep(s, 'print("EXTEND C1 NEW arm %s:', 'print("EXTEND C2 NEW [" + C2_TAG + "] arm %s:')
s = rep(s, '"harness_patch": "v3 2026-09-28 + scale + extend c1 (abort on replay mismatch)"',
        '"harness_patch": "v3 2026-09-28 + scale + extend c2 (single-arm Stage 1, time budget)", "c2_tag": C2_TAG')
put(17, s)

for cc in c:
    if cc["cell_type"] == "code":
        cc["outputs"] = []; cc["execution_count"] = None
        ast.parse("\n".join(l for l in "".join(cc["source"]).splitlines() if not l.lstrip().startswith(("!", "%"))))
out = "p6_extend_%s_arm%s_2026-10-01.ipynb" % (tag, arm)
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
