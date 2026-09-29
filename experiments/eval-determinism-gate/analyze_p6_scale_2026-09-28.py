"""Analysis for the P6 scale run (n = 100 per arm), exactly as fixed in PREREG_p6_scale_2026-09-28.md.

Inputs (results/): p6_scale_armA_result.json and p6_scale_armB_result.json from the two Kaggle versions, plus
p6_magnitude_2026-09-28_final.json for the run-to-run comparison. Writes results/p6_scale_2026-09-28_analysis.json.
"""
import json, os
from math import comb

R = os.environ.get("P6_RESULTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


A = json.load(open(os.path.join(R, "p6_scale_armA_result.json")))
B = json.load(open(os.path.join(R, "p6_scale_armB_result.json")))
eA = {r["i"]: r for r in A["stage2"]["episodes"]}
eB = {r["i"]: r for r in B["stage2"]["episodes"]}

# Validity checks, reported before any result.
checks = {
    "stage1_identical_across_versions": A["stage1"]["armA_initstate_hashes"] == B["stage1"]["armA_initstate_hashes"]
    and A["stage1"]["armB_initstate_hashes"] == B["stage1"]["armB_initstate_hashes"],
    "armA_hashcheck_mismatches": A["stage2"]["hashcheck_mismatches"],
    "armB_hashcheck_mismatches": B["stage2"]["hashcheck_mismatches"],
    "armA_devices": sorted({r.get("device") for r in eA.values()}),
    "armB_devices": sorted({r.get("device") for r in eB.values()}),
    "stage1_overlap": A["stage1"]["overlap"],
    "stage1_overlap_indices": [i for i, (a, b) in enumerate(zip(A["stage1"]["armA_initstate_hashes"],
                                                                 A["stage1"]["armB_initstate_hashes"])) if a == b],
    "stage2_overlap_A_B": len({r["init_hash"] for r in eA.values()} & {r["init_hash"] for r in eB.values()}),
}

pairs = sorted(set(eA) & set(eB))
a_only = sum(1 for i in pairs if eA[i]["success"] and not eB[i]["success"])
b_only = sum(1 for i in pairs if eB[i]["success"] and not eA[i]["success"])
sA = sum(eA[i]["success"] for i in pairs)
sB = sum(eB[i]["success"] for i in pairs)
primary = {
    "n_pairs": len(pairs), "armA_successes": sA, "armB_successes": sB,
    "delta_pp_B_minus_A": round((sB - sA) / len(pairs) * 100, 2),
    "paired_agreement": len(pairs) - a_only - b_only, "A_success_B_fail": a_only, "A_fail_B_success": b_only,
    "mcnemar_exact_two_sided_p": round(mcnemar_exact(a_only, b_only), 4),
    "armA_fail_i": [i for i in pairs if not eA[i]["success"]],
    "armB_fail_i": [i for i in pairs if not eB[i]["success"]],
}

# Secondary: i = 0..19 across the three runs (descriptive, no test).
P = json.load(open(os.path.join(R, "p6_magnitude_2026-09-28_final.json")))
p28A = {r["i"]: r for r in P["armA"]}
p28B = {r["i"]: r for r in P["armB"]}
first20 = range(20)
secondary = {
    "hash_match_vs_0928_armA": sum(eA[i]["init_hash"] == p28A[i]["init_hash"] for i in first20),
    "hash_match_vs_0928_armB": sum(eB[i]["init_hash"] == p28B[i]["init_hash"] for i in first20),
    "armA_first20": {"0927": P["comparison_0927"]["armA"], "0928": P["armA_successes"],
                     "scale": sum(eA[i]["success"] for i in first20)},
    "armB_first20": {"0927": P["comparison_0927"]["armB"], "0928": P["armB_successes"],
                     "scale": sum(eB[i]["success"] for i in first20)},
    "armA_same_state_outcome_flips_0928_vs_scale": [i for i in first20 if eA[i]["success"] != p28A[i]["success"]],
    "armB_same_state_outcome_flips_0928_vs_scale": [i for i in first20 if eB[i]["success"] != p28B[i]["success"]],
}

out = {"prereg": "PREREG_p6_scale_2026-09-28.md", "status": "EXPLORATORY", "checks": checks,
       "primary": primary, "secondary": secondary,
       "versions": {"A": A["versions"], "B": B["versions"]},
       "hours": {"A": A["stage2"]["hours_this_session"], "B": B["stage2"]["hours_this_session"]}}
json.dump(out, open(os.path.join(R, "p6_scale_2026-09-28_analysis.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "versions"}, indent=2))
