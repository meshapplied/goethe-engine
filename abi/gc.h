/* gc.h: the C ABI of a kernel pack runtime (runtime kind gc-c).
 *
 * Plain C99, no CUDA headers, stdlib-ctypes friendly. The runtime is the pack's libgck.so; the engine loads it with
 * dlopen and binds these symbols only. Capabilities (verify widths, tree mask, taps, recurrent commit, spec policy,
 * contracts) are in the pack manifest JSON, never in symbols.
 *
 * Versioning: ABI is MAJOR.MINOR (gc_abi_version() = MAJOR << 16 | MINOR). A minor only appends: new struct fields at
 * the end, new flag bits, new functions. A major may change anything. A caller checks that the library's major equals
 * GC_ABI_MAJOR and that its minor is >= the minor it needs.
 *
 * Structs: every in/out struct starts with `size`, its own byte size. The caller sets it to sizeof of the struct it
 * was compiled with; the library reads and writes only fields inside min(size, its own size) and treats absent
 * fields as zero. Output structs are written the same way: a field past the caller's size is not written.
 * All integers are fixed width, little endian, naturally aligned; pointers are 64-bit.
 *
 * Handles: each gc_open returns an independent instance (own CUDA stream, KV, slots, scratch). Any number may live in
 * one process, on one or several devices. One handle is not thread safe; different handles may run on different
 * threads. Only gc_abi_version and gc_open may be called without a handle.
 *
 * Errors: functions return GC_OK (0) or a negative GC_E_* code. gc_last_error(h) returns the text of the last error
 * on that handle; gc_last_error(NULL) the last error of gc_open on the calling thread. The string stays valid until
 * the next call on the same handle (or thread, for NULL). */
#ifndef GC_H
#define GC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GC_ABI_MAJOR 1
#define GC_ABI_MINOR 1
#define GC_ABI_VERSION ((GC_ABI_MAJOR << 16) | GC_ABI_MINOR)

#if defined(__GNUC__)
#define GC_API __attribute__((visibility("default")))
#else
#define GC_API
#endif

typedef struct gc_handle gc_handle;

enum {
  GC_OK = 0,
  GC_E_ARG = -1,         /* bad argument or struct size */
  GC_E_MANIFEST = -2,    /* manifest missing, unreadable, hash mismatch, or no such bucket */
  GC_E_WEIGHTS = -3,     /* weight file missing, unreadable, or its sha256 differs from the manifest's */
  GC_E_DEVICE = -4,      /* no such device, wrong architecture, out of device memory, CUDA failure */
  GC_E_UNSUPPORTED = -5, /* a request the pack or this library version cannot run (see the manifest caps) */
  GC_E_STATE = -6,       /* call order: unknown sequence, position not contiguous, no free slot */
  GC_E_CAPACITY = -7     /* an output buffer is too small; the needed size is in the out struct */
};

/* gc_open_params.flags */
enum {
  GC_OPEN_SPEC = 1u << 0,      /* buckets = {target, drafter}: a speculative pair, one linear stream */
  GC_OPEN_NO_CHECK = 1u << 1   /* skip the weight file sha256 check (the manifest hash is still checked) */
};

typedef struct gc_open_params {
  uint32_t size;
  uint32_t flags;               /* GC_OPEN_* */
  const char *pack_dir;         /* extracted pack: a directory with manifest.json, manifest.sha256, blobs/ */
  const char *const *weights;   /* weight file paths in the manifest's model.weights order; a drafter's is last */
  uint32_t n_weights;
  int32_t device;               /* CUDA ordinal; -1 = the current device */
  const char *const *buckets;   /* manifest bucket ids: a batch ladder in ascending columns, or {target, drafter} */
  uint32_t n_buckets;
  uint32_t pool_blocks;         /* KV blocks to allocate (0 = the pack's maximum) */
  const char *spec_policy;      /* GC_OPEN_SPEC: "ar:K" or "kind:k=v,..." (NULL = the pack's) */
  const char *scratch_dir;      /* writable dir for caches (NULL = none) */
  uint32_t state_pages;         /* 1.1: snapshot pages to add to a page pack (see gc_layout.state_pages); 0 = none */
} gc_open_params;

/* Returns the library's ABI version: MAJOR << 16 | MINOR. */
GC_API uint32_t gc_abi_version(void);

/* Verifies the pack, loads and relays out the weights on the device, allocates KV and state. On failure *out is NULL
 * and the text is in gc_last_error(NULL). */
GC_API int gc_open(const gc_open_params *p, gc_handle **out);

/* gc_layout.kv_dtype */
enum { GC_KV_F16 = 0, GC_KV_FP8 = 1 };

/* gc_layout.flags */
enum {
  GC_L_PAGED = 1u << 0,        /* KV is paged: block_size rows per block, a sequence's table maps its rows. Clear: a block is a
                                  whole sequence page (block_size = max_ctx): its KV rows, window ring and recurrent state;
                                  a sequence's table is its one page */
  GC_L_DEVICE_ARGMAX = 1u << 1,/* a greedy step can return tokens without logits */
  GC_L_SPEC = 1u << 2          /* opened as a speculative pair: a step with GC_W_SPEC runs a round */
};

typedef struct gc_layout {
  uint32_t size;
  uint32_t flags;              /* GC_L_* */
  uint32_t vocab;
  uint32_t max_cols;           /* most tokens one gc_step accepts across all work items (spec: verify width) */
  uint32_t max_seqs;           /* live sequences (slots) */
  uint32_t max_ctx;            /* longest sequence, tokens */
  uint32_t block_size;         /* KV rows per block (paged); max_ctx for slot-contiguous KV */
  uint32_t num_blocks;         /* blocks in the pool */
  uint32_t layers;             /* attention layers with KV */
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t kv_dtype;           /* GC_KV_* */
  uint32_t spec_width;         /* GC_L_SPEC: columns of one verify */
  uint32_t feat_width;         /* floats per token row of the tap features (0: no taps) */
  uint64_t kv_bytes;           /* device bytes of the KV pool */
  uint64_t state_bytes;        /* device bytes of recurrent state per sequence (0: none) */
  uint64_t device_bytes;       /* total device bytes held by the handle */
  uint32_t state_pages;        /* 1.1: pages beyond num_blocks (ids num_blocks .. num_blocks + state_pages - 1) that only
                                  gc_copy_blocks addresses: snapshots of a page pack's pages; 0 for a paged pack */
  uint32_t pad1;
} gc_layout;

GC_API int gc_layout_get(gc_handle *h, gc_layout *out);

/* gc_work.flags */
enum {
  GC_W_GREEDY = 1u << 0,       /* every wanted row is an unmasked argmax: tokens may be returned instead of logits */
  GC_W_SPEC = 1u << 1,         /* a decode the library may advance by a whole speculative round (GC_L_SPEC) */
  GC_W_FEATS = 1u << 2         /* also return the tap features of every token row */
};

/* One sequence's tokens for a step: a decode (1 token), a prefill chunk (n tokens), or a spec verify (anchor and
 * drafts as a chain, or a tree when parents is set). Token i writes KV row pos + i and runs RoPE and attention at
 * position pos + depth(i) (depth: 0 for a parent of -1, else depth(parent) + 1; a chain has depth i); it attends to
 * rows < pos and to its ancestors' rows. gc_commit then moves row base + path[j] to base + j. Rows do not depend on
 * the rest of the batch (batch invariance). */
typedef struct gc_work {
  uint32_t size;
  uint32_t flags;              /* GC_W_* */
  uint64_t sid;                /* sequence id, stable while the sequence lives */
  uint32_t pos;                /* first KV row of tokens[0] */
  uint32_t n_tokens;
  const int32_t *tokens;       /* n_tokens */
  const int32_t *parents;      /* n_tokens, parent index of each token (-1: the cached prefix); NULL: a chain */
  const uint32_t *blocks;      /* block table covering rows 0 .. pos + n_tokens - 1 (paged KV); NULL otherwise */
  uint32_t n_blocks;
  uint32_t want;               /* logits or argmax rows wanted: those of the last `want` tokens */
} gc_work;

/* gc_step_in.flags */
enum {
  GC_S_HOST_LOGITS = 1u << 0   /* copy float32 logits to the caller's host buffer instead of exposing a device ptr */
};

typedef struct gc_step_in {
  uint32_t size;
  uint32_t flags;              /* GC_S_* */
  const gc_work *const *work;  /* n_work items */
  uint32_t n_work;
  uint32_t pad;
} gc_step_in;

typedef struct gc_step_out {
  uint32_t size;
  uint32_t n_rows;             /* rows produced: sum of want, or the tokens of every item for a spec round */
  /* Greedy: tokens, concatenated in work order; counts[i] = tokens of item i (want, or 1..spec_width for a round).
   * The caller provides tokens/counts with capacities; if too small GC_E_CAPACITY is returned and n_rows is set. */
  int32_t *tokens;
  uint32_t tokens_cap;
  uint32_t pad0;
  uint32_t *counts;            /* n_work entries */
  /* Not greedy: float32 [n_rows, vocab] in work order, either copied to `logits` (GC_S_HOST_LOGITS, capacity in
   * floats in logits_cap) or left on the device: logits_dev, valid until the next step on this handle. */
  float *logits;
  uint64_t logits_cap;
  uint64_t logits_dev;         /* CUDA device pointer (CUdeviceptr) */
  /* GC_W_FEATS: float32 [token rows of those items, feat_width] to `feats` (host), capacity in floats. */
  float *feats;
  uint64_t feats_cap;
} gc_step_out;

/* Runs the items as one step: decode, chunked prefill, spec verify or a spec round, by the items' shapes. */
GC_API int gc_step(gc_handle *h, const gc_step_in *in, gc_step_out *out);

/* Tree verify commit: keeps the rows of `path` (indices into the last verify's tokens) as the sequence's rows
 * base .. base + n - 1. A chain backend accepts only the identity path. */
GC_API int gc_commit(gc_handle *h, uint64_t sid, uint32_t base, const uint32_t *path, uint32_t n);

/* Copies block src into block dst for each pair (src[i] -> dst[i]): KV rows of a paged pack, the whole page (KV rows, window
 * ring, recurrent state) of a page pack. */
GC_API int gc_copy_blocks(gc_handle *h, const uint32_t *src, const uint32_t *dst, uint32_t n);

/* Releases a sequence's slot, recurrent state and spec state. Unknown sid is not an error. */
GC_API int gc_release(gc_handle *h, uint64_t sid);

GC_API const char *gc_last_error(gc_handle *h);

/* Frees everything the handle holds. NULL is accepted. */
GC_API void gc_close(gc_handle *h);

#ifdef __cplusplus
}
#endif
#endif
