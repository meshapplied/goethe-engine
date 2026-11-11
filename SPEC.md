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

## Design choices and their sources

- Tree with parent pointers, ancestor mask, depth positions: SpecInfer (arXiv 2305.09781) Algorithm 2 (Speculate /
  TreeParallelDecode / VerifyGreedy); EAGLE `eagle/model/cnets.py` `topK_genrate` (draft_tokens, mask_index parents,
  tree_mask by ancestor closure, `tree_position_ids = sum(tree_mask) - 1`); vLLM v0.11 `vllm/v1/attention/backends/tree_attn.py`
  `_prepare_tree_attn_bias` (root and ancestors visible); Medusa (arXiv 2401.10774) section 2.1.2.
- EAGLE-2 dynamic tree: arXiv 2406.16858 section 4 (expansion of the top-k nodes by value, reranking); code
  `cnets.py topK_genrate` (beam of top_k by cumulative log-prob, k x k candidates per level, global top total_tokens
  sorted by index so parents precede children). Our builder adds a parent-closure check at the cut instead of
  relying on strictly decreasing scores.
- EAGLE-3 taps: `eagle/model/modeling_llama_kv.py` records hidden_states before layers 2, L // 2, L - 3; vLLM
  `qwen3_5.py get_eagle3_aux_hidden_state_layers` (2, L // 2, L - 3); fc over 3 x hidden: `cnets.py` line 532.
- MTP: DeepSeek-V3 (arXiv 2412.19437) section 2.2 eq. 21 h' = M [RMSNorm(h); RMSNorm(Emb(t))] with shared embedding
  and head; the code orders the concat embedding first (vLLM `deepseek_mtp.py` `cat([enorm(emb), hnorm(h)])`,
  `qwen3_5_mtp.py` `cat([pre_fc_norm_embedding, pre_fc_norm_hidden])`, llama.cpp `qwen35.cpp graph_mtp`
  `ggml_concat(e_norm, h_norm)`); the weights decide, so we follow the code. The target state is h_nextn = output_norm
  of the last layer (llama.cpp `qwen35.cpp` line 210-213, vLLM `qwen3_next.py` returns the normed hidden), the pairing
  is (t_p, h_(p-1)) at position p (llama.cpp `common/speculative.cpp`: "pair each token with the tgt embedding shifted
  right by one position", first token with zeros).
- Exact acceptance at temperature > 0: the target's sample at token index i is a deterministic function of its
  logits and splitmix64(seed, i), so accepting a draft only when it equals that sample emits exactly the non-spec
  tokens; for any deterministic draft this has the acceptance probability of rejection sampling, p(x) = min(1, p/q)
  with q one-hot. A drafter-coupled form (drafter sampling on the same noise) is the drafter-invariant scheme of
  Daliri, Musco, Suresh (arXiv 2408.07978); not built.
- Rejection acceptance: Leviathan et al. (arXiv 2211.17192) and Chen et al. (arXiv 2302.01318, Algorithm 2,
  Theorem 1) for one child; SpecInfer Algorithm 2 VerifyStochastic (accept x_s if r <= P_LLM / P_SSM, else
  P_LLM := norm(max(0, P_LLM - P_SSM))) for several; deterministic children are one-hot q (EAGLE `utils.py
  evaluate_posterior`: qx = 1, rejected token zeroed, renormalized; SGLang `speculative_sampling.cuh`
  TreeSpeculativeSamplingTargetOnly: one coin against the running sibling mass, residual relu(target - draft)).
  Coins: uniform(mix(seed, SALT + rank), i), residual: uniform(mix(seed, SALT_RES), i), the drafter's own draws a
  third stream; vLLM main seeds by (seed, position) too (`v1/worker/gpu/spec_decode/rejection_sampler_utils.py`).
- Residual and accept tests run on the exact distribution the sampler draws from (`sampling.dist`: temperature,
  top-k, top-p, grammar mask), so a row without children is sampled by `sample()` itself and rejection mode with no
  drafts is the non-spec sampler.
- Seeded reproducibility under rejection sampling (drafts change which coin decides): a strict request (rejection,
  temperature > 0) never has its tree cut by budget or pool (the scheduler skips the step or preempts others), never
  drafts token 0 (the prompt's own sample), and a recompute chunk stops before the anchor so the token after a
  preemption comes from a verify as before. Then its tokens are a function of (seed, prompt, drafter) alone. vLLM v0.11's
  rejection sampler draws its uniforms from a per-request generator per step (`generate_uniform_probs`), so its
  stream depends on how many drafts each step carried (our reading of the code; not measured).
- Commit as row moves: SGLang `spec_utils.py move_accept_tokens_to_target_kvcache` (out_cache_loc of accepted nodes
  moved to the front); recurrent state of the last accepted step: `commit_mamba_states_after_verify`. Rows only move
  forward (path[j] >= j), so one pass in order is safe.
- Drafter KV and features keyed by token prefix (reference) / positional rows (GPU): vLLM v0.11 `v1/spec_decode/eagle.py`
  propose uses the target's block table for the draft KV (`block_table_tensor.gather`), so drafter rows share the
  target's paging and prefix hits.
- Capabilities from the pack: target bucket `spec_caps` (pack ABI 1.x) map to
  `spec.CAPS` (`pack_backend.caps_of_bucket`); `spec.check` refuses with the first failing requirement and
  `Engine.spec_algorithms()` lists every algorithm with None or its reason.

## Proof

CPU tests: every policy above, greedy, equals `RefModel.dense` (4 concurrent streams,
preemption, chunked prefill, prefix cache); tree commits that move rows are exercised; a recurrent target with tree
verify equals the oracle; multi-stream random traces (greedy, exact sampled, rejection sampled, n > 1, aborts, pool
pressure) equal their references and free every block; rejection sampling (chain with q, SpecInfer tree with q,
EAGLE top-k tree) matches the target's exact 3-token joint law by chi-square, and a residual-free sampler is caught;
seeded rejection outputs equal across batch sizes, budgets and pool pressure.
