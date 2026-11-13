# Speculative decoding: one abstraction, every algorithm a policy

## The abstraction

| stage | what | where |
|---|---|---|
| proposal | a token tree: node 0 the anchor (last committed token, not yet run), drafted node i with parent < i, optional draft distribution q per node | `spec.Tree` |
| verify | one target pass over [anchor] + drafts: node i at position base + depth(i), attending to the cached prefix and its ancestors; a chain is the linear case | `Work.parents`, `Backend.step` |
| accept | per path from the anchor: exact (the target's own seeded sample, greedy: argmax, must equal a child) or rejection (multi-step speculative sampling) | `engine._post`, `spec.mss` |
| commit | the accepted path's rows to positions base .. base + len - 1, recurrent state after its last node, KV blocks of the computed rows into the radix tree | `Backend.commit`, `kv.commit`; GPU: `gc_commit` |

Policies build proposals and nothing else; no algorithm name appears in the engine core or a kernel.

| algorithm | policy | drafter (an ordinary model) |
|---|---|---|
| n-gram (prompt lookup) | `NgramDrafter` chain | none |
| ar:K | `TreeDrafter(m, "chain")` | any LM (`ref.DraftLM`, `pack_backend._GcLM` on a GPU artifact) |
| SpecInfer | `TreeDrafter(m, "dynamic", temperature > 0)`: children drawn from q, q kept for rejection | any LM |
| EAGLE-1 | `TreeDrafter(EagleHead(taps=("result_norm",)), "static", choices)` | `ref.EagleHead`: fc [rms emb; rms f], own layers, target head |
| EAGLE-2 | `TreeDrafter(EagleHead(...), "dynamic", topk, depth)`: expand the topk best of each level by cumulative log-probability, rerank all, keep k | same |
| EAGLE-3 | same, `taps` = three l_out layers fused by a linear map, own hidden past the committed rows | same |
| Medusa | `TreeDrafter(MedusaHeads, "static", medusa_choices)` | `ref.MedusaHeads`: ResBlock heads on result_norm |
| MTP (DeepSeek-V3, Qwen3.5) | chain over `EagleHead(taps=("result_norm",))` (host) or a device round `mtp` in a pack runtime | the model's MTP module |
| DFlash | device round `walk` in a pack runtime | a block drafter |

## Pack interface

| what | rule |
|---|---|
| capabilities | the target bucket's `spec_caps` map to `spec.CAPS` (`pack_backend.caps_of_bucket`); `spec.check` refuses with the first failing requirement and `Engine.spec_algorithms()` lists every algorithm with None or its reason |
| device acceptance | a speculative pair loads only if its target bucket advertises it: `io.accept`, or for gc-c buckets a non-empty `spec_caps.accept` |
| acceptance sampling | exact: accept a draft only when it equals the target's seeded sample (a function of its logits and `splitmix64(seed, i)`); rejection: multi-step speculative sampling on the exact sampler distribution, residual relu(target - draft) |
| commit | accepted rows move forward in one pass (`path[j] >= j`), recurrent state of the last accepted node, KV blocks of the computed rows into the radix tree |
