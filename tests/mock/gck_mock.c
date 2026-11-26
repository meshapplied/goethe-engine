/* Mock libgck: the whole gc.h ABI 1.3 (or 1.2 with -DGC_MOCK_ABI_12) on the CPU, deterministic, no model. Logits are a
 * hash of the KV rows a sequence sees (same function as tests/conftest.py), so a wrong block table, stale row, wrong
 * tree path, or broken GC_W_PREV_TOKEN changes tokens. */
#include "../../abi/gc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 53
#define FEAT 4
#define BSZ 4
#define NBLK 64
#define MAXSEQ 64
#define SPECW 4
#define SPECCTX 200
#define MAX_Q 2

typedef unsigned __int128 u128;

typedef struct {
  uint64_t sid;
  uint32_t nb;
  uint32_t *blocks;
  int32_t last_tok;
  int has_last;
  int used;
} seqent;

typedef struct {
  int used;
  int dev_tok;
  uint32_t n_work;
  uint32_t n_rows;
  uint32_t n_frows;
  uint32_t counts[MAXSEQ];
  int32_t *tokens;
  uint32_t tokens_cap;
  float *logits;
  uint64_t logits_cap;
  float *feats;
  uint64_t feats_cap;
} async_slot;

struct gc_handle {
  int spec;
  uint32_t nblk;
  int32_t *tok;
  int64_t *rpos;
  seqent seqs[MAXSEQ];
  int32_t buf[SPECCTX];
  uint32_t nbuf, rounds;
  async_slot q[MAX_Q];
  int q_head, q_tail, q_len;
  char err[256];
};

static char open_err[256];

static uint64_t fold(uint64_t h, int64_t t, int64_t p) {
  return (uint64_t)(((u128)h * 1000003u + (u128)t * 31 + (u128)p) % ((u128)1 << 61));
}

static float rowv(uint64_t h, int v) {
  return (float)(((uint64_t)v * 2654435761ull + (h % 1000003u) * 40503u) % 1009u) / 90.0f;
}

static int argmax_masked(const float *lg, const uint32_t *mask, uint32_t mwords) {
  for (int v = 0; v < VOCAB; v++) if (isnan(lg[v])) return -1;
  int best = -1;
  float bv = -INFINITY;
  for (int v = 0; v < VOCAB; v++) {
    if (mask && ((uint32_t)(v >> 5) >= mwords || !((mask[v >> 5] >> (v & 31)) & 1u))) continue;
    float x = lg[v];
    if (best < 0 || x > bv) { bv = x; best = v; }
  }
  if (best < 0 || bv == -INFINITY) return -1;
  if (bv == INFINITY) {
    int any_fin = 0;
    for (int v = 0; v < VOCAB; v++) {
      if (mask && ((uint32_t)(v >> 5) >= mwords || !((mask[v >> 5] >> (v & 31)) & 1u))) continue;
      if (isfinite(lg[v])) { any_fin = 1; break; }
    }
    if (!any_fin) return -1;
  }
  return best;
}

static int sample_row(const float *lg, const uint32_t *mask, uint32_t mwords,
                      float temp, float top_p, uint32_t top_k, uint64_t seed, uint64_t step_idx) {
  for (int v = 0; v < VOCAB; v++) if (isnan(lg[v])) return -1;
  float x[VOCAB];
  float xmax = -INFINITY;
  int any_fin = 0;
  for (int v = 0; v < VOCAB; v++) {
    if (mask && ((uint32_t)(v >> 5) >= mwords || !((mask[v >> 5] >> (v & 31)) & 1u))) {
      x[v] = -INFINITY;
    } else {
      x[v] = lg[v];
      if (x[v] > xmax) xmax = x[v];
      if (isfinite(x[v])) any_fin = 1;
    }
  }
  if (xmax == -INFINITY || (xmax == INFINITY && !any_fin)) return -1;
  if (temp <= 0.0f) return argmax_masked(lg, mask, mwords);
  float z[VOCAB];
  int ord[VOCAB];
  for (int v = 0; v < VOCAB; v++) { z[v] = x[v] / temp; ord[v] = v; }
  /* Stable descending insertion sort by z[v], ties broken by ascending token id */
  for (int i = 1; i < VOCAB; i++) {
    int id = ord[i];
    float zv = z[id];
    int j = i - 1;
    while (j >= 0 && z[ord[j]] < zv) { ord[j + 1] = ord[j]; j--; }
    ord[j + 1] = id;
  }
  double e[VOCAB], s = 0.0, z0 = (double)z[ord[0]];
  for (int i = 0; i < VOCAB; i++) {
    e[i] = isinf(z[ord[i]]) && z[ord[i]] < 0 ? 0.0 : exp((double)z[ord[i]] - z0);
    s += e[i];
  }
  double c[VOCAB], acc = 0.0;
  int k = VOCAB;
  for (int i = 0; i < VOCAB; i++) {
    double q = e[i] / s;
    if (k == VOCAB && acc >= (double)top_p) k = i;
    acc += q;
    c[i] = acc;
  }
  if (top_k > 0 && (int)top_k < k) k = (int)top_k;
  if (k < 1) k = 1;
  double u = gc_uniform(seed, step_idx);
  double target = u * c[k - 1];
  int j = k - 1;
  for (int i = 0; i < k; i++) {
    if (c[i] >= target) { j = i; break; }
  }
  return ord[j];
}

static int fail(gc_handle *h, int code, const char *m) {
  snprintf(h ? h->err : open_err, 256, "%s", m);
  return code;
}

GC_API uint32_t gc_abi_version(void) {
#ifdef GC_MOCK_ABI_12
  return (1u << 16) | 2u;
#else
  return GC_ABI_VERSION;
#endif
}

GC_API int gc_open(const gc_open_params *p, gc_handle **out) {
  *out = NULL;
  if (!p || p->size < sizeof(uint32_t) * 2 || !p->pack_dir || !out) return fail(NULL, GC_E_ARG, "bad params");
  char path[4096];
  snprintf(path, sizeof path, "%s/manifest.json", p->pack_dir);
  FILE *f = fopen(path, "rb");
  if (!f) return fail(NULL, GC_E_MANIFEST, "no manifest.json");
  fclose(f);
  if (p->n_weights < 1 || !p->weights) return fail(NULL, GC_E_WEIGHTS, "no weights");
  for (uint32_t i = 0; i < p->n_weights; i++)
    if (!p->weights[i]) continue;
    else if (!(f = fopen(p->weights[i], "rb"))) return fail(NULL, GC_E_WEIGHTS, "weights unreadable");
    else fclose(f);
  int spec = (p->flags & GC_OPEN_SPEC) != 0;
  if (p->n_buckets < 1 || (spec && p->n_buckets != 2)) return fail(NULL, GC_E_ARG, "buckets");
  if (p->pool_blocks > NBLK) return fail(NULL, GC_E_ARG, "pool_blocks above the pack's maximum");
  gc_handle *h = calloc(1, sizeof *h);
  h->spec = spec;
  h->nblk = p->pool_blocks ? p->pool_blocks : NBLK;
  h->tok = calloc(NBLK * BSZ, sizeof(int32_t));
  h->rpos = malloc(NBLK * BSZ * sizeof(int64_t));
  for (int i = 0; i < NBLK * BSZ; i++) h->rpos[i] = -1;
  *out = h;
  return GC_OK;
}

GC_API int gc_layout_get(gc_handle *h, gc_layout *o) {
  if (!h || !o || o->size < 8) return GC_E_ARG;
  gc_layout l;
  memset(&l, 0, sizeof l);
  l.size = sizeof l;
  l.flags = GC_L_DEVICE_ARGMAX | (h->spec ? GC_L_SPEC : GC_L_PAGED) |
            (!h->spec && getenv("GCK_MOCK_BOUNDED") ? GC_L_BOUNDED : 0);
#ifndef GC_MOCK_ABI_12
  if (!h->spec) {
    l.flags |= GC_L_DEVICE_SAMPLE | GC_L_DEVICE_BITMASK | GC_L_ASYNC | GC_L_FLAT;
    l.n_graph_buckets = 4;
    l.graph_buckets[0] = 1;
    l.graph_buckets[1] = 2;
    l.graph_buckets[2] = 4;
    l.graph_buckets[3] = 8;
  }
#endif
  l.vocab = VOCAB;
  l.max_cols = 4096;
  l.layers = 1;
  l.kv_heads = 1;
  l.head_dim = 1;
  l.kv_dtype = getenv("GCK_MOCK_KV_FP8") ? GC_KV_FP8 : GC_KV_F16;
  l.feat_width = FEAT;
  if (h->spec) {
    l.max_seqs = 1;
    l.max_ctx = SPECCTX;
    l.block_size = SPECCTX;
    l.num_blocks = 1;
    l.spec_width = SPECW;
  } else {
    l.max_seqs = MAXSEQ;
    l.block_size = BSZ;
    l.num_blocks = h->nblk;
    l.max_ctx = h->nblk * BSZ;
  }
  l.kv_bytes = (uint64_t)l.num_blocks * l.block_size * 8;
  uint32_t n = o->size < sizeof l ? o->size : sizeof l;
  memcpy(o, &l, n);
  o->size = n;
  return GC_OK;
}

static seqent *seq_of(gc_handle *h, uint64_t sid, int make) {
  seqent *free_ = NULL;
  for (int i = 0; i < MAXSEQ; i++) {
    if (h->seqs[i].used && h->seqs[i].sid == sid) return &h->seqs[i];
    if (!h->seqs[i].used && !free_) free_ = &h->seqs[i];
  }
  if (make && free_) {
    memset(free_, 0, sizeof *free_);
    free_->used = 1;
    free_->sid = sid;
  }
  return make ? free_ : NULL;
}

#define SLOT(e, p) ((e)->blocks[(p) / BSZ] * BSZ + (p) % BSZ)

static int spec_step(gc_handle *h, const gc_step_in *in, gc_step_out *o) {
  const gc_work *w = in->work[0];
  if (in->n_work != 1 || !(w->flags & GC_W_GREEDY)) return fail(h, GC_E_UNSUPPORTED, "spec pack runs one greedy sequence");
  if (w->pos > h->nbuf || w->pos + w->n_tokens > SPECCTX) return fail(h, GC_E_STATE, "position not contiguous");
  memcpy(h->buf + w->pos, w->tokens, w->n_tokens * 4);
  h->nbuf = w->pos + w->n_tokens;
  o->n_rows = 0;
  o->counts[0] = 0;
  if (!w->want) return GC_OK;
  uint64_t hh = 7;
  uint32_t n = h->nbuf;
  for (uint32_t p = 0; p < n; p++) hh = fold(hh, h->buf[p], p);
  uint32_t k = (w->flags & GC_W_SPEC) ? 1 + (++h->rounds) % SPECW : 1;
  if (o->tokens_cap < k) { o->n_rows = k; return fail(h, GC_E_CAPACITY, "tokens_cap"); }
  for (uint32_t i = 0; i < k; i++) {
    float lg[VOCAB];
    for (int v = 0; v < VOCAB; v++) lg[v] = rowv(hh, v);
    int t = argmax_masked(lg, NULL, 0);
    o->tokens[i] = t;
    hh = fold(hh, t, n + i);
  }
  if (w->flags & GC_W_SPEC) {
    for (uint32_t i = 0; i + 1 < k; i++) h->buf[h->nbuf++] = o->tokens[i];
  }
  o->counts[0] = k;
  o->n_rows = k;
  return GC_OK;
}

typedef struct {
  uint32_t flags;
  uint64_t sid;
  uint32_t pos;
  uint32_t n_tokens;
  const int32_t *tokens;
  const int32_t *parents;
  const uint32_t *blocks;
  uint32_t n_blocks;
  uint32_t want;
  float temperature;
  float top_p;
  uint32_t top_k;
  uint32_t bitmask_words;
  uint64_t seed;
  uint64_t step_idx;
  const uint32_t *bitmask;
} item_view;

static int unpack_items(gc_handle *h, const gc_step_in *in, item_view *iv, uint32_t *out_n) {
#ifndef GC_MOCK_ABI_12
  if ((in->flags & GC_S_FLAT) && in->size >= sizeof(gc_step_in) && in->flat) {
    const gc_batch_flat *f = in->flat;
    if (!f->n_seqs || f->n_seqs > MAXSEQ) return fail(h, GC_E_ARG, "bad flat n_seqs");
    uint32_t r0 = 0;
    for (uint32_t i = 0; i < f->n_seqs; i++) {
      uint32_t ts = f->tok_offsets[i], te = f->tok_offsets[i + 1];
      uint32_t bs = f->blk_offsets ? f->blk_offsets[i] : 0;
      uint32_t be = f->blk_offsets ? f->blk_offsets[i + 1] : 0;
      iv[i].flags = f->flags[i];
      iv[i].sid = f->sids[i];
      iv[i].pos = f->pos[i];
      iv[i].n_tokens = te - ts;
      iv[i].tokens = f->tokens + ts;
      iv[i].parents = f->parents ? f->parents + ts : NULL;
      iv[i].blocks = f->blocks ? f->blocks + bs : NULL;
      iv[i].n_blocks = be - bs;
      iv[i].want = f->want[i];
      iv[i].temperature = f->temperature ? f->temperature[i] : 0.0f;
      iv[i].top_p = f->top_p ? f->top_p[i] : 1.0f;
      iv[i].top_k = f->top_k ? f->top_k[i] : 0;
      iv[i].seed = f->seed ? f->seed[i] : 0;
      iv[i].step_idx = f->step_idx ? f->step_idx[i] : 0;
      iv[i].bitmask_words = (iv[i].flags & GC_W_BITMASK) ? f->bitmask_words : 0;
      iv[i].bitmask = ((iv[i].flags & GC_W_BITMASK) && f->bitmasks) ? f->bitmasks + (uint64_t)r0 * f->bitmask_words : NULL;
      r0 += iv[i].want;
    }
    *out_n = f->n_seqs;
    return GC_OK;
  }
#endif
  if (!in->n_work || in->n_work > MAXSEQ || !in->work) return fail(h, GC_E_ARG, "bad n_work");
  for (uint32_t i = 0; i < in->n_work; i++) {
    const gc_work *w = in->work[i];
    iv[i].flags = w->flags;
    iv[i].sid = w->sid;
    iv[i].pos = w->pos;
    iv[i].n_tokens = w->n_tokens;
    iv[i].tokens = w->tokens;
    iv[i].parents = w->parents;
    iv[i].blocks = w->blocks;
    iv[i].n_blocks = w->n_blocks;
    iv[i].want = w->want;
    if (w->size >= sizeof(gc_work)) {
      iv[i].temperature = w->temperature;
      iv[i].top_p = w->top_p;
      iv[i].top_k = w->top_k;
      iv[i].bitmask_words = w->bitmask_words;
      iv[i].seed = w->seed;
      iv[i].step_idx = w->step_idx;
      iv[i].bitmask = w->bitmask;
    } else {
      iv[i].temperature = 0.0f;
      iv[i].top_p = 1.0f;
      iv[i].top_k = 0;
      iv[i].bitmask_words = 0;
      iv[i].seed = 0;
      iv[i].step_idx = 0;
      iv[i].bitmask = NULL;
    }
  }
  *out_n = in->n_work;
  return GC_OK;
}

static int exec_step(gc_handle *h, const gc_step_in *in, gc_step_out *o) {
  if (!h || !in || !o) return GC_E_ARG;
  if (h->spec) return spec_step(h, in, o);
#ifndef GC_MOCK_ABI_12
  if (in->size >= sizeof(gc_step_in) && in->pad_cols > 4096) return fail(h, GC_E_ARG, "pad_cols above max_cols");
#endif
  item_view iv[MAXSEQ];
  uint32_t n = 0;
  int rc = unpack_items(h, in, iv, &n);
  if (rc != GC_OK) return rc;
  const char *lm = getenv("GCK_MOCK_LAUNCH_MAX");   /* a launch of more columns cannot get its device memory */
  if (lm && n > (uint32_t)atoi(lm)) return fail(h, GC_E_DEVICE, "launch: out of memory");
  int dev_tok = 1;
  uint32_t rows = 0, frows = 0;
  for (uint32_t i = 0; i < n; i++) {
    const item_view *w = &iv[i];
#ifdef GC_MOCK_ABI_12
    dev_tok &= (w->flags & GC_W_GREEDY) != 0;
#else
    dev_tok &= ((w->flags & (GC_W_GREEDY | GC_W_SAMPLE)) != 0);
#endif
    rows += w->want;
    if (w->flags & GC_W_FEATS) frows += w->n_tokens;
    if (!w->n_tokens || w->want > w->n_tokens || !w->blocks || (uint64_t)w->n_blocks * BSZ < (uint64_t)w->pos + w->n_tokens)
      return fail(h, GC_E_ARG, "bad work item");
    for (uint32_t b = 0; b < w->n_blocks; b++) if (w->blocks[b] >= h->nblk) return fail(h, GC_E_ARG, "block out of pool");
  }
  o->n_rows = rows;
  if (dev_tok ? o->tokens_cap < rows : (!(in->flags & GC_S_HOST_LOGITS) || o->logits_cap < (uint64_t)rows * VOCAB))
    return fail(h, GC_E_CAPACITY, "output buffer too small");
  if (frows && o->feats_cap < (uint64_t)frows * FEAT) return fail(h, GC_E_CAPACITY, "feats buffer too small");
  int force_nan = getenv("GCK_MOCK_FORCE_NAN") != NULL;
  int break_prev = getenv("GCK_MOCK_BREAK_PREV_TOKEN") != NULL;
  uint32_t r = 0, fr = 0;
  for (uint32_t i = 0; i < n; i++) {
    const item_view *w = &iv[i];
    seqent *e = seq_of(h, w->sid, 1);
    if (!e) return fail(h, GC_E_STATE, "no free slot");
    free(e->blocks);
    e->nb = w->n_blocks;
    e->blocks = malloc(w->n_blocks * 4);
    memcpy(e->blocks, w->blocks, w->n_blocks * 4);
    int dep[4096];
    if (w->n_tokens > 4096) return fail(h, GC_E_ARG, "too many tokens");
    for (uint32_t j = 0; j < w->n_tokens; j++) {
      int par = w->parents ? w->parents[j] : (int)j - 1;
      dep[j] = w->parents ? (par < 0 ? 0 : dep[par] + 1) : (int)j;
      uint32_t s = SLOT(e, w->pos + j);
      int32_t tok_val = w->tokens[j];
      if ((w->flags & GC_W_PREV_TOKEN) && j == 0) {
        if (w->n_tokens != 1 || !e->has_last) return fail(h, GC_E_STATE, "prev_token requires prior device token");
        tok_val = break_prev ? 0 : e->last_tok;
      }
      h->tok[s] = tok_val;
      h->rpos[s] = w->pos + dep[j];
    }
    uint32_t local_r = 0;
    for (uint32_t j = 0; j < w->n_tokens; j++) {
      uint64_t hh = 7;
      for (uint32_t p = 0; p < w->pos; p++) {
        uint32_t s = SLOT(e, p);
        if (h->rpos[s] != (int64_t)p) return fail(h, GC_E_STATE, "stale or foreign KV row");
        hh = fold(hh, h->tok[s], p);
      }
      int anc[4096], na = 0;
      for (int a = (int)j; a >= 0; a = w->parents ? w->parents[a] : a - 1) anc[na++] = a;
      for (int k = na - 1; k >= 0; k--) {
        uint32_t s = SLOT(e, w->pos + (uint32_t)anc[k]);
        if (h->rpos[s] != (int64_t)w->pos + dep[anc[k]]) return fail(h, GC_E_STATE, "tree row at the wrong position");
        hh = fold(hh, h->tok[s], h->rpos[s]);
      }
      if (w->flags & GC_W_FEATS) {
        for (int v = 0; v < FEAT; v++) o->feats[(uint64_t)fr * FEAT + v] = rowv(hh, v);
        fr++;
      }
      if (j + w->want >= w->n_tokens) {
        float lg[VOCAB];
        for (int v = 0; v < VOCAB; v++) lg[v] = rowv(hh, v);
        if (force_nan) lg[0] = NAN;
        if (dev_tok) {
          const uint32_t *bm = ((w->flags & GC_W_BITMASK) && w->bitmask) ? w->bitmask + (uint64_t)local_r * w->bitmask_words : NULL;
          int picked;
          if (w->flags & GC_W_SAMPLE) {
            picked = sample_row(lg, bm, w->bitmask_words, w->temperature, w->top_p, w->top_k, w->seed, w->step_idx + local_r);
          } else {
            picked = argmax_masked(lg, bm, w->bitmask_words);
          }
          o->tokens[r] = picked;
          e->last_tok = picked;
          e->has_last = picked >= 0;
        } else {
          memcpy(o->logits + (uint64_t)r * VOCAB, lg, VOCAB * sizeof(float));
        }
        local_r++;
        r++;
      }
    }
    if (dev_tok) o->counts[i] = w->want;
  }
  return GC_OK;
}

GC_API int gc_step(gc_handle *h, const gc_step_in *in, gc_step_out *o) {
  return exec_step(h, in, o);
}

#ifndef GC_MOCK_ABI_12
GC_API int gc_step_begin(gc_handle *h, const gc_step_in *in, gc_step_out *o) {
  if (!h || !in || !o) return GC_E_ARG;
  if (h->q_len >= MAX_Q) return fail(h, GC_E_STATE, "async step queue full");
  async_slot *sl = &h->q[h->q_tail];
  memset(sl->counts, 0, sizeof sl->counts);
  uint32_t n = (in->flags & GC_S_FLAT) && in->flat ? in->flat->n_seqs : in->n_work;
  if (n > MAXSEQ) return fail(h, GC_E_ARG, "too many seqs");
  int dev_tok = !(in->flags & GC_S_HOST_LOGITS);
  if (dev_tok && o->tokens_cap > sl->tokens_cap) {
    free(sl->tokens);
    sl->tokens_cap = o->tokens_cap ? o->tokens_cap : 16;
    sl->tokens = malloc(sl->tokens_cap * sizeof(int32_t));
  }
  if (!dev_tok && o->logits_cap > sl->logits_cap) {
    free(sl->logits);
    sl->logits_cap = o->logits_cap ? o->logits_cap : VOCAB;
    sl->logits = malloc(sl->logits_cap * sizeof(float));
  }
  if (o->feats_cap > sl->feats_cap) {
    free(sl->feats);
    sl->feats_cap = o->feats_cap;
    sl->feats = malloc(sl->feats_cap * sizeof(float));
  }
  gc_step_out tmp;
  memset(&tmp, 0, sizeof tmp);
  tmp.size = sizeof tmp;
  tmp.tokens = sl->tokens;
  tmp.tokens_cap = o->tokens_cap;
  tmp.counts = sl->counts;
  tmp.logits = sl->logits;
  tmp.logits_cap = o->logits_cap;
  tmp.feats = sl->feats;
  tmp.feats_cap = o->feats_cap;
  int rc = exec_step(h, in, &tmp);
  if (rc != GC_OK) { o->n_rows = tmp.n_rows; return rc; }
  sl->used = 1;
  sl->dev_tok = dev_tok;
  sl->n_work = n;
  sl->n_rows = tmp.n_rows;
  sl->n_frows = (uint32_t)(o->feats_cap / FEAT);
  h->q_tail = (h->q_tail + 1) % MAX_Q;
  h->q_len++;
  return GC_OK;
}

GC_API int gc_step_wait(gc_handle *h, gc_step_out *o) {
  if (!h || !o) return GC_E_ARG;
  if (h->q_len <= 0) return fail(h, GC_E_STATE, "no in-flight step");
  async_slot *sl = &h->q[h->q_head];
  o->n_rows = sl->n_rows;
  if (sl->dev_tok) {
    if (o->tokens_cap < sl->n_rows) return fail(h, GC_E_CAPACITY, "tokens_cap");
    if (sl->n_rows) memcpy(o->tokens, sl->tokens, sl->n_rows * sizeof(int32_t));
    if (sl->n_work) memcpy(o->counts, sl->counts, sl->n_work * sizeof(uint32_t));
  } else {
    if (o->logits_cap < (uint64_t)sl->n_rows * VOCAB) return fail(h, GC_E_CAPACITY, "logits_cap");
    if (sl->n_rows) memcpy(o->logits, sl->logits, (uint64_t)sl->n_rows * VOCAB * sizeof(float));
  }
  if (o->feats && o->feats_cap && sl->feats) {
    memcpy(o->feats, sl->feats, o->feats_cap * sizeof(float));
  }
  sl->used = 0;
  h->q_head = (h->q_head + 1) % MAX_Q;
  h->q_len--;
  return GC_OK;
}
#endif

GC_API int gc_commit(gc_handle *h, uint64_t sid, uint32_t base, const uint32_t *path, uint32_t n) {
  seqent *e = h ? seq_of(h, sid, 0) : NULL;
  if (!e || !path) return GC_E_STATE;
  for (uint32_t j = 0; j < n; j++) if (path[j] != j) {
    uint32_t a = SLOT(e, base + path[j]), b = SLOT(e, base + j);
    h->tok[b] = h->tok[a];
    h->rpos[b] = base + j;
  }
  return GC_OK;
}

GC_API int gc_copy_blocks(gc_handle *h, const uint32_t *src, const uint32_t *dst, uint32_t n) {
  if (!h || h->spec) return GC_E_UNSUPPORTED;
  for (uint32_t i = 0; i < n; i++) {
    if (src[i] >= h->nblk || dst[i] >= h->nblk) return fail(h, GC_E_ARG, "block out of pool");
    memcpy(h->tok + dst[i] * BSZ, h->tok + src[i] * BSZ, BSZ * 4);
    memcpy(h->rpos + dst[i] * BSZ, h->rpos + src[i] * BSZ, BSZ * 8);
  }
  return GC_OK;
}

GC_API int gc_state_save(gc_handle *h, uint64_t sid, uint32_t snap) {
  (void)sid; (void)snap;
  return h && getenv("GCK_MOCK_BOUNDED") ? GC_OK : GC_E_ARG;
}

GC_API int gc_state_restore(gc_handle *h, uint32_t snap, uint64_t sid) {
  (void)sid; (void)snap;
  return h && getenv("GCK_MOCK_BOUNDED") ? GC_OK : GC_E_ARG;
}

GC_API int gc_release(gc_handle *h, uint64_t sid) {
  if (!h) return GC_E_ARG;
  seqent *e = seq_of(h, sid, 0);
  if (e) { free(e->blocks); memset(e, 0, sizeof *e); }
  if (h->spec) h->nbuf = 0;
  return GC_OK;
}

GC_API const char *gc_last_error(gc_handle *h) { return h ? h->err : open_err; }

GC_API void gc_close(gc_handle *h) {
  if (!h) return;
  for (int i = 0; i < MAXSEQ; i++) free(h->seqs[i].blocks);
  for (int i = 0; i < MAX_Q; i++) {
    free(h->q[i].tokens);
    free(h->q[i].logits);
    free(h->q[i].feats);
  }
  free(h->tok);
  free(h->rpos);
  free(h);
}

GC_API int gc_probe(int device, void *constants, uint64_t cap, uint64_t *size) {
  (void)device;
  const char *e = getenv("GCK_MOCK_PROBE");
  double v[2];
  if (!e || sscanf(e, "%lf,%lf", &v[0], &v[1]) != 2) return GC_E_UNSUPPORTED;
  *size = sizeof v;
  if (cap < sizeof v) return GC_E_CAPACITY;
  memcpy(constants, v, sizeof v);
  return GC_OK;
}

GC_API int gc_price(const void *constants, uint64_t size, const char *bucket_id, double *step_us) {
  if (size != 2 * sizeof(double)) return GC_E_ARG;
  const double *v = constants;
  *step_us = strstr(bucket_id, ".mock.") ? v[0] : v[1];
  return GC_OK;
}
