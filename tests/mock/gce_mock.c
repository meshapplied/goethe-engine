/* Mock libgce: the gc.h 1.2 embedding subset on the CPU, no model. A sequence's pooled row is a function of its tokens only
 * (dim 4: sum, length, first, last), so a row that moved to another item or depends on its batch is caught. */
#include "../../abi/gc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 4
#define ROWS 64
struct gc_handle { char err[256]; };
static char open_err[256];
static int fail(gc_handle *h, int code, const char *m) { snprintf(h ? h->err : open_err, 256, "%s", m); return code; }

GC_API uint32_t gc_abi_version(void) { return GC_ABI_VERSION; }

GC_API int gc_open(const gc_open_params *p, gc_handle **out) {
  *out = NULL;
  if (!p || !p->pack_dir || p->n_buckets != 1 || p->n_weights < 1) return fail(NULL, GC_E_ARG, "bad params");
  *out = calloc(1, sizeof **out);
  return GC_OK;
}

GC_API int gc_layout_get(gc_handle *h, gc_layout *o) {
  if (!h || !o || o->size < 8) return GC_E_ARG;
  gc_layout l; memset(&l, 0, sizeof l); l.size = sizeof l;
  l.flags = GC_L_EMBED; l.max_cols = ROWS; l.max_seqs = 8; l.max_ctx = ROWS; l.embd_dim = DIM; l.embd_pool = GC_POOL_MEAN;
  uint32_t n = o->size < sizeof l ? o->size : sizeof l, sz = o->size;
  memcpy(o, &l, n); o->size = sz;
  return GC_OK;
}

GC_API int gc_embed(gc_handle *h, const gc_step_in *in, gc_embed_out *out) {
  if (!h || !in || !out) return GC_E_ARG;
  out->n_rows = in->n_work;
  if (out->rows_cap < (uint64_t)in->n_work * DIM) return fail(h, GC_E_CAPACITY, "rows capacity");
  uint32_t cols = 0;
  for (uint32_t i = 0; i < in->n_work; i++) cols += in->work[i]->n_tokens;
  if (cols > ROWS || in->n_work > 8) return fail(h, GC_E_ARG, "more rows or sequences than one launch takes");
  for (uint32_t i = 0; i < in->n_work; i++) {
    const gc_work *w = in->work[i];
    if (w->pos || !w->n_tokens || w->want != 1) return fail(h, GC_E_ARG, "an item is one whole sequence");
    float *r = out->rows + (size_t)i * DIM, s = 0;
    for (uint32_t k = 0; k < w->n_tokens; k++) s += (float)w->tokens[k];
    r[0] = s; r[1] = (float)w->n_tokens; r[2] = (float)w->tokens[0]; r[3] = (float)w->tokens[w->n_tokens - 1];
    if (in->flags & GC_S_NORMALIZE) {
      double q = 0;
      for (int k = 0; k < DIM; k++) q += r[k] * r[k];
      const float nm = q > 0 ? (float)(1.0 / sqrt(q)) : 0.0f;
      for (int k = 0; k < DIM; k++) r[k] *= nm;
    }
  }
  return GC_OK;
}

GC_API int gc_release(gc_handle *h, uint64_t sid) { (void)h; (void)sid; return GC_OK; }
GC_API const char *gc_last_error(gc_handle *h) { return h ? h->err : open_err; }
GC_API void gc_close(gc_handle *h) { free(h); }
