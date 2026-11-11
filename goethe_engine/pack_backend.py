"""Engine-side view of a pack bucket: capabilities (spec.CAPS) and numerics, from the manifest."""

RECURRENT = {"fold": "chain", "tree": "tree", "step": "step", "sequence": "step"}  # pack recurrent_commit -> caps


def caps_of_bucket(b, spec=None):
    """Engine capabilities (spec.CAPS) of a target bucket's spec_caps; spec: the pair's policy (device rounds)."""
    sc = b.get("spec_caps") or {}
    rc = [RECURRENT.get(x, x) for x in sc.get("recurrent_commit", [])]
    c = dict(tree="tree" in sc.get("verify", []) and bool(sc.get("tree_mask")), taps={t["node"]: t["n"] for t in sc.get("taps", [])},
             recurrent=("tree" if "tree" in rc else "chain" if "chain" in rc else rc[0] if rc else None),
             drafters=sc.get("drafter_roles", []))
    if spec:
        c.update(round=spec["kind"], verify=max(sc.get("verify_widths") or [b["ncol"]]), logits=False)
    return c


def numerics_of(m, b):
    """{contract, bodies, desc} of bucket b: its numerics contract, the per-format body labels the pack declares for it
    (ABI 1.1 buckets[].bodies, {} where it does not say) and the manifest's contract_desc text (None before 1.1).
    All strings pass through unchanged."""
    return dict(contract=b["contract"], bodies=b.get("bodies") or {},
                desc=(m.get("contract_desc") or {}).get(b["contract"]))
