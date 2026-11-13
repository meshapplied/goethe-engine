/* Mock libgck: the whole gc.h ABI 1.0 on the CPU, deterministic, no model. Logits are a hash of the KV rows a sequence
 * sees (same function as tests/mockref.py), so a wrong block table, stale row or wrong tree path changes tokens. */
#include "../../abi/gc.h"
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
typedef unsigned __int128 u128;
typedef struct { uint64_t sid; uint32_t nb; uint32_t *blocks; int used; } seqent;
struct gc_handle {
  int spec; uint32_t nblk; int32_t *tok; int64_t *rpos; seqent seqs[MAXSEQ];
  int32_t buf[SPECCTX]; uint32_t nbuf, rounds; char err[256];
};
static char open_err[256];

static uint64_t fold(uint64_t h, int64_t t, int64_t p) { return (uint64_t)(((u128)h * 1000003u + (u128)t * 31 + (u128)p) % ((u128)1 << 61)); }
static float rowv(uint64_t h, int v) { return (float)(((uint64_t)v * 2654435761ull + (h % 1000003u) * 40503u) % 1009u) / 90.0f; }
static int argmax(uint64_t h) { int b = 0; for (int v = 1; v < VOCAB; v++) if (rowv(h, v) > rowv(h, b)) b = v; return b; }
static int fail(gc_handle *h, int code, const char *m) { snprintf(h ? h->err : open_err, 256, "%s", m); return code; }

GC_API uint32_t gc_abi_version(void) { return GC_ABI_VERSION; }

GC_API int gc_open(const gc_open_params *p, gc_handle **out) {
  *out = NULL;
  if (!p || p->size < sizeof(uint32_t) * 2 || !p->pack_dir || !out) return fail(NULL, GC_E_ARG, "bad params");
  char path[4096];
  snprintf(path, sizeof path, "%s/manifest.json", p->pack_dir);
  FILE *f = fopen(path, "rb");
  if (!f) return fail(NULL, GC_E_MANIFEST, "no manifest.json");
  fclose(f);
  if (p->n_weights < 1 || !p->weights) return fail(NULL, GC_E_WEIGHTS, "no weights");
  for (uint32_t i = 0; i < p->n_weights; i++) if (!(f = fopen(p->weights[i], "rb"))) return fail(NULL, GC_E_WEIGHTS, "weights unreadable"); else fclose(f);
  int spec = (p->flags & GC_OPEN_SPEC) != 0;
  if (p->n_buckets < 1 || (spec && p->n_buckets != 2)) return fail(NULL, GC_E_ARG, "buckets");
  if (p->pool_blocks > NBLK) return fail(NULL, GC_E_ARG, "pool_blocks above the pack's maximum");
  gc_handle *h = calloc(1, sizeof *h);
  h->spec = spec; h->nblk = p->pool_blocks ? p->pool_blocks : NBLK;
  h->tok = calloc(NBLK * BSZ, sizeof(int32_t));
  h->rpos = malloc(NBLK * BSZ * sizeof(int64_t));
  for (int i = 0; i < NBLK * BSZ; i++) h->rpos[i] = -1;
  *out = h;
  return GC_OK;
}

GC_API int gc_layout_get(gc_handle *h, gc_layout *o) {
  if (!h || !o || o->size < 8) return GC_E_ARG;
  gc_layout l; memset(&l, 0, sizeof l); l.size = sizeof l;
  l.flags = GC_L_DEVICE_ARGMAX | (h->spec ? GC_L_SPEC : GC_L_PAGED) | (!h->spec && getenv("GCK_MOCK_BOUNDED") ? GC_L_BOUNDED : 0);
  l.vocab = VOCAB; l.max_cols = 4096; l.layers = 1; l.kv_heads = 1; l.head_dim = 1; l.kv_dtype = GC_KV_F16; l.feat_width = FEAT;
  if (h->spec) { l.max_seqs = 1; l.max_ctx = SPECCTX; l.block_size = SPECCTX; l.num_blocks = 1; l.spec_width = SPECW; }
  else { l.max_seqs = MAXSEQ; l.block_size = BSZ; l.num_blocks = h->nblk; l.max_ctx = h->nblk * BSZ; }
  l.kv_bytes = (uint64_t)l.num_blocks * l.block_size * 8;
  uint32_t n = o->size < sizeof l ? o->size : sizeof l;
  memcpy(o, &l, n); o->size = n;
  return GC_OK;
}

static seqent *seq_of(gc_handle *h, uint64_t sid, int make) {
  seqent *free_ = NULL;
  for (int i = 0; i < MAXSEQ; i++) {
    if (h->seqs[i].used && h->seqs[i].sid == sid) return &h->seqs[i];
    if (!h->seqs[i].used && !free_) free_ = &h->seqs[i];
  }
  if (make && free_) { free_->used = 1; free_->sid = sid; free_->nb = 0; free_->blocks = NULL; }
  return make ? free_ : NULL;
}
#define SLOT(e, p) ((e)->blocks[(p) / BSZ] * BSZ + (p) % BSZ)

static int spec_step(gc_handle *h, const gc_step_in *in, gc_step_out *o) {
  const gc_work *w = in->work[0];
  if (in->n_work != 1 || !(w->flags & GC_W_GREEDY)) return fail(h, GC_E_UNSUPPORTED, "spec pack runs one greedy sequence");
  if (w->pos > h->nbuf || w->pos + w->n_tokens > SPECCTX) return fail(h, GC_E_STATE, "position not contiguous");
  memcpy(h->buf + w->pos, w->tokens, w->n_tokens * 4); h->nbuf = w->pos + w->n_tokens;
  o->n_rows = 0; o->counts[0] = 0;
  if (!w->want) return GC_OK;
  uint64_t hh = 7; uint32_t n = h->nbuf;
  for (uint32_t p = 0; p < n; p++) hh = fold(hh, h->buf[p], p);
  uint32_t k = (w->flags & GC_W_SPEC) ? 1 + (++h->rounds) % SPECW : 1;
  if (o->tokens_cap < k) { o->n_rows = k; return fail(h, GC_E_CAPACITY, "tokens_cap"); }
  for (uint32_t i = 0; i < k; i++) { int t = argmax(hh); o->tokens[i] = t; hh = fold(hh, t, n + i); }
  if (w->flags & GC_W_SPEC) { for (uint32_t i = 0; i + 1 < k; i++) h->buf[h->nbuf++] = o->tokens[i]; }
  o->counts[0] = k; o->n_rows = k;
  return GC_OK;
}

GC_API int gc_step(gc_handle *h, const gc_step_in *in, gc_step_out *o) {
  if (!h || !in || !o || !in->n_work) return GC_E_ARG;
  if (h->spec) return spec_step(h, in, o);
  int greedy = 1; uint32_t rows = 0, frows = 0;
  for (uint32_t i = 0; i < in->n_work; i++) {
    const gc_work *w = in->work[i];
    greedy &= (w->flags & GC_W_GREEDY) != 0;
    rows += w->want; if (w->flags & GC_W_FEATS) frows += w->n_tokens;
    if (!w->n_tokens || w->want > w->n_tokens || !w->blocks || (uint64_t)w->n_blocks * BSZ < (uint64_t)w->pos + w->n_tokens)
      return fail(h, GC_E_ARG, "bad work item");
    for (uint32_t b = 0; b < w->n_blocks; b++) if (w->blocks[b] >= h->nblk) return fail(h, GC_E_ARG, "block out of pool");
  }
  o->n_rows = rows;
  if (greedy ? o->tokens_cap < rows : (!(in->flags & GC_S_HOST_LOGITS) || o->logits_cap < (uint64_t)rows * VOCAB))
    return fail(h, GC_E_CAPACITY, "output buffer too small");
  if (frows && o->feats_cap < (uint64_t)frows * FEAT) return fail(h, GC_E_CAPACITY, "feats buffer too small");
  uint32_t r = 0, fr = 0;
  for (uint32_t i = 0; i < in->n_work; i++) {
    const gc_work *w = in->work[i];
    seqent *e = seq_of(h, w->sid, 1);
    if (!e) return fail(h, GC_E_STATE, "no free slot");
    free(e->blocks); e->nb = w->n_blocks; e->blocks = malloc(w->n_blocks * 4); memcpy(e->blocks, w->blocks, w->n_blocks * 4);
    int dep[4096]; if (w->n_tokens > 4096) return fail(h, GC_E_ARG, "too many tokens");
    for (uint32_t j = 0; j < w->n_tokens; j++) {
      int par = w->parents ? w->parents[j] : (int)j - 1;
      dep[j] = w->parents ? (par < 0 ? 0 : dep[par] + 1) : (int)j;
      uint32_t s = SLOT(e, w->pos + j);
      h->tok[s] = w->tokens[j]; h->rpos[s] = w->pos + dep[j];
    }
    for (uint32_t j = 0; j < w->n_tokens; j++) {
      uint64_t hh = 7;
      for (uint32_t p = 0; p < w->pos; p++) {
        uint32_t s = SLOT(e, p);
        if (h->rpos[s] != (int64_t)p) return fail(h, GC_E_STATE, "stale or foreign KV row");
        hh = fold(hh, h->tok[s], p);
      }
      int anc[4096], na = 0;
      for (int a = j; a >= 0; a = w->parents ? w->parents[a] : a - 1) anc[na++] = a;
      for (int k = na - 1; k >= 0; k--) {
        uint32_t s = SLOT(e, w->pos + anc[k]);
        if (h->rpos[s] != (int64_t)w->pos + dep[anc[k]]) return fail(h, GC_E_STATE, "tree row at the wrong position");
        hh = fold(hh, h->tok[s], h->rpos[s]);
      }
      if (w->flags & GC_W_FEATS) { for (int v = 0; v < FEAT; v++) o->feats[(uint64_t)fr * FEAT + v] = rowv(hh, v); fr++; }
      if (j + w->want >= w->n_tokens) {
        if (greedy) o->tokens[r] = argmax(hh); else for (int v = 0; v < VOCAB; v++) o->logits[(uint64_t)r * VOCAB + v] = rowv(hh, v);
        r++;
      }
    }
    if (greedy) o->counts[i] = w->want;
  }
  return GC_OK;
}

GC_API int gc_commit(gc_handle *h, uint64_t sid, uint32_t base, const uint32_t *path, uint32_t n) {
  seqent *e = h ? seq_of(h, sid, 0) : NULL;
  if (!e || !path) return GC_E_STATE;
  for (uint32_t j = 0; j < n; j++) if (path[j] != j) {
    uint32_t a = SLOT(e, base + path[j]), b = SLOT(e, base + j);
    h->tok[b] = h->tok[a]; h->rpos[b] = base + j;
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

/* the mock has no bounded state: its closed-form oracle reads only the tokens, so a save and a restore are accepted and do nothing */
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
  free(h->tok); free(h->rpos); free(h);
}
